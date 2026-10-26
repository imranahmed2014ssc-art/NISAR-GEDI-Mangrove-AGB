from pathlib import Path
from datetime import datetime, timezone
import csv
import math
import re
import sys
import platform

import h5py
import numpy as np
import pandas as pd
from pyproj import Transformer
import matplotlib.pyplot as plt

# ============================================================
# PROJECT SETTINGS
# ============================================================
PROJECT_ROOT = Path(r"D:\NISAR_Mangrove_Paper")
GEDI_DIR = PROJECT_ROOT / "03_GEDI_L4A_V3" / "ORIGINAL"
PROCESSED_DIR = PROJECT_ROOT / "03_GEDI_L4A_V3" / "Processed"
QC_DIR = PROJECT_ROOT / "03_GEDI_L4A_V3" / "QC"
NISAR_DIR = PROJECT_ROOT / "02_NISAR_GCOV" / "Original"
LOG_PATH = PROJECT_ROOT / "00_Documentation" / "analysis_log.txt"

WEST, EAST = 89.58, 89.84
SOUTH, NORTH = 21.95, 22.15
PROJECTED_CRS = "EPSG:32645"

# Operational threshold for detecting repeated / near-duplicate footprint locations.
# This is configurable. We use 25 m because it is approximately the GEDI footprint diameter.
REPEAT_RADIUS_M = 25.0

# Primary proposal quality filter
QUALITY_FILTER_FIELDS = {
    "l2_algrunflag": 1,
    "l2a_quality_flag_rel3": 1,
    "l4a_quality_flag_rel3": 1,
    "degrade_include_flag": 1,
    "elev_highestreturn_outlier_flag": 0,
}

CORE_FIELDS = [
    "shot_number", "lat_lowestmode", "lon_lowestmode", "agbd", "agbd_se",
    "l2_algrunflag", "l2a_quality_flag_rel3", "l4a_quality_flag_rel3",
    "degrade_include_flag", "elev_highestreturn_outlier_flag",
    "predictor_limit_flag", "response_limit_flag", "sensitivity", "delta_time"
]

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
QC_DIR.mkdir(parents=True, exist_ok=True)

def parse_gedi_datetime(name):
    m = re.search(r"GEDI04_A_(\d{13})_", name)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y%j%H%M%S").replace(tzinfo=timezone.utc)

def parse_nisar_datetime(name):
    m = re.search(r"_(\d{8}T\d{6})_", name)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)

def package_version(name):
    try:
        import importlib.metadata as md
        return md.version(name)
    except Exception:
        return "unknown"

def append_log(lines):
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    block = ["", f"[{stamp}] PHASE 1 — GEDI L4A V3 PROCESSING"] + lines + ["-" * 72]
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write("\n".join(block) + "\n")

def first_nisar_datetime():
    files = sorted(NISAR_DIR.glob("*.h5"))
    for fp in files:
        dt = parse_nisar_datetime(fp.name)
        if dt is not None:
            return dt, fp.name
    return None, ""

def safe_array(group, name, n):
    if name not in group:
        return np.full(n, np.nan)
    arr = np.asarray(group[name][:])
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr

def extract_all_gedi():
    rows = []
    files = sorted(GEDI_DIR.glob("*.h5"))
    if len(files) != 22:
        raise RuntimeError(f"Expected 22 GEDI HDF5 files, found {len(files)} in {GEDI_DIR}")

    print(f"GEDI files found: {len(files)}")
    for i, fp in enumerate(files, 1):
        file_dt = parse_gedi_datetime(fp.name)
        with h5py.File(fp, "r") as h5:
            beams = [k for k in h5.keys() if str(k).startswith("BEAM")]
            for beam in beams:
                g = h5[beam]
                required_core = ["shot_number", "lat_lowestmode", "lon_lowestmode", "agbd", "agbd_se"]
                if not all(x in g for x in required_core):
                    continue

                shot = np.asarray(g["shot_number"][:]).reshape(-1)
                n = len(shot)
                lon = np.asarray(g["lon_lowestmode"][:], dtype=float).reshape(-1)
                lat = np.asarray(g["lat_lowestmode"][:], dtype=float).reshape(-1)

                inside = (
                    np.isfinite(lon) & np.isfinite(lat) &
                    (lon >= WEST) & (lon <= EAST) &
                    (lat >= SOUTH) & (lat <= NORTH)
                )
                idx = np.where(inside)[0]
                if len(idx) == 0:
                    continue

                arrays = {name: safe_array(g, name, n) for name in CORE_FIELDS if name != "shot_number"}

                for j in idx:
                    r = {
                        "shot_number": int(shot[j]),
                        "longitude": float(lon[j]),
                        "latitude": float(lat[j]),
                        "beam": beam,
                        "source_file": fp.name,
                        "acquisition_date": file_dt.isoformat() if file_dt else "",
                    }
                    for name, arr in arrays.items():
                        val = arr[j]
                        if isinstance(val, np.generic):
                            val = val.item()
                        r[name] = val
                    rows.append(r)

        print(f"[{i:02d}/22] extracted from {fp.name}")

    return pd.DataFrame(rows)

def apply_primary_filter(df):
    mask = np.ones(len(df), dtype=bool)
    for field, required_value in QUALITY_FILTER_FIELDS.items():
        mask &= pd.to_numeric(df[field], errors="coerce").eq(required_value).to_numpy()
    agbd = pd.to_numeric(df["agbd"], errors="coerce")
    mask &= np.isfinite(agbd.to_numpy()) & (agbd.to_numpy() > 0)
    return df.loc[mask].copy()

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

def assign_repeat_groups(df, radius_m=25.0):
    if len(df) == 0:
        df["utm_x"] = []
        df["utm_y"] = []
        df["repeat_group_id"] = []
        df["repeat_group_size"] = []
        return df

    transformer = Transformer.from_crs("EPSG:4326", PROJECTED_CRS, always_xy=True)
    x, y = transformer.transform(df["longitude"].to_numpy(), df["latitude"].to_numpy())
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    n = len(df)
    uf = UnionFind(n)
    cell = radius_m
    grid = {}

    for i in range(n):
        gx = math.floor(x[i] / cell)
        gy = math.floor(y[i] / cell)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in grid.get((gx + dx, gy + dy), []):
                    if (x[i] - x[j])**2 + (y[i] - y[j])**2 <= radius_m**2:
                        uf.union(i, j)
        grid.setdefault((gx, gy), []).append(i)

    roots = [uf.find(i) for i in range(n)]
    root_to_id = {}
    group_ids = []
    next_id = 1
    for r in roots:
        if r not in root_to_id:
            root_to_id[r] = next_id
            next_id += 1
        group_ids.append(root_to_id[r])

    out = df.copy()
    out["utm_x"] = x
    out["utm_y"] = y
    out["repeat_group_id"] = group_ids
    sizes = out.groupby("repeat_group_id")["shot_number"].transform("size")
    out["repeat_group_size"] = sizes.astype(int)
    return out

def add_temporal_fields(df, nisar_dt):
    out = df.copy()
    out["gedi_datetime"] = pd.to_datetime(out["acquisition_date"], utc=True, errors="coerce")
    if nisar_dt is not None:
        nisar_ts = pd.Timestamp(nisar_dt)
        out["nisar_datetime"] = nisar_ts.isoformat()
        out["delta_days_to_NISAR"] = (
            (nisar_ts - out["gedi_datetime"]).dt.total_seconds() / 86400.0
        )
    else:
        out["nisar_datetime"] = ""
        out["delta_days_to_NISAR"] = np.nan
    return out

def make_reference_table(qdf):
    # Exact duplicate shot_number control first.
    qdf = qdf.sort_values("gedi_datetime").drop_duplicates("shot_number", keep="last").copy()

    # Detect near-duplicate / repeat-track locations.
    grouped = assign_repeat_groups(qdf, REPEAT_RADIUS_M)

    # Primary modelling table: retain the most recent valid record per repeat group.
    ref = (
        grouped.sort_values("gedi_datetime")
        .groupby("repeat_group_id", as_index=False, group_keys=False)
        .tail(1)
        .copy()
    )
    ref["is_primary_record"] = True
    return grouped, ref

def coarse_grid_coverage(df, nx=5, ny=5):
    if len(df) == 0:
        return 0, np.zeros((ny, nx), dtype=int)
    xi = np.clip(((df["longitude"].to_numpy() - WEST) / (EAST - WEST) * nx).astype(int), 0, nx-1)
    yi = np.clip(((df["latitude"].to_numpy() - SOUTH) / (NORTH - SOUTH) * ny).astype(int), 0, ny-1)
    counts = np.zeros((ny, nx), dtype=int)
    for xg, yg in zip(xi, yi):
        counts[yg, xg] += 1
    return int(np.sum(counts > 0)), counts

def save_plots(all_df, qdf, ref_df):
    # AGBD histogram
    vals = pd.to_numeric(qdf["agbd"], errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(vals, bins=35)
    ax.set_xlabel("GEDI AGBD (Mg/ha)")
    ax.set_ylabel("Count")
    ax.set_title("Quality-filtered GEDI AGBD")
    fig.tight_layout()
    fig.savefig(QC_DIR / "GEDI_AGBD_histogram.png", dpi=300)
    plt.close(fig)

    # AGBD_SE histogram
    vals = pd.to_numeric(qdf["agbd_se"], errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(vals, bins=35)
    ax.set_xlabel("GEDI AGBD_SE (Mg/ha)")
    ax.set_ylabel("Count")
    ax.set_title("Quality-filtered GEDI AGBD uncertainty")
    fig.tight_layout()
    fig.savefig(QC_DIR / "GEDI_AGBD_SE_histogram.png", dpi=300)
    plt.close(fig)

    # Footprint map: all vs quality filtered vs primary reference
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(all_df["longitude"], all_df["latitude"], s=2, alpha=0.20, label="All AOI footprints")
    ax.scatter(qdf["longitude"], qdf["latitude"], s=6, alpha=0.55, label="Quality-filtered")
    ax.scatter(ref_df["longitude"], ref_df["latitude"], s=9, alpha=0.75, label="Primary reference")
    ax.set_xlim(WEST, EAST)
    ax.set_ylim(SOUTH, NORTH)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("GEDI footprint coverage in locked AOI")
    ax.legend(loc="best", markerscale=2)
    fig.tight_layout()
    fig.savefig(QC_DIR / "GEDI_footprint_coverage_map.png", dpi=300)
    plt.close(fig)

def save_summaries(all_df, qdf, grouped_qdf, ref_df, nisar_dt, nisar_name):
    # Filter summary
    occupied_q, counts_q = coarse_grid_coverage(qdf)
    occupied_ref, counts_ref = coarse_grid_coverage(ref_df)

    agbd = pd.to_numeric(qdf["agbd"], errors="coerce").dropna()
    agbd_ref = pd.to_numeric(ref_df["agbd"], errors="coerce").dropna()

    summary_rows = [
        ("raw_GEDI_files", len(list(GEDI_DIR.glob("*.h5")))),
        ("AOI_records_all", len(all_df)),
        ("quality_filtered_records", len(qdf)),
        ("unique_quality_shot_numbers", qdf["shot_number"].nunique()),
        ("repeat_location_groups", grouped_qdf["repeat_group_id"].nunique()),
        ("quality_records_in_repeat_groups_gt1", int((grouped_qdf["repeat_group_size"] > 1).sum())),
        ("primary_reference_records", len(ref_df)),
        ("quality_5x5_cells_occupied", occupied_q),
        ("reference_5x5_cells_occupied", occupied_ref),
        ("quality_AGBD_min", float(agbd.min()) if len(agbd) else np.nan),
        ("quality_AGBD_median", float(agbd.median()) if len(agbd) else np.nan),
        ("quality_AGBD_mean", float(agbd.mean()) if len(agbd) else np.nan),
        ("quality_AGBD_p95", float(agbd.quantile(0.95)) if len(agbd) else np.nan),
        ("quality_AGBD_max", float(agbd.max()) if len(agbd) else np.nan),
        ("reference_AGBD_min", float(agbd_ref.min()) if len(agbd_ref) else np.nan),
        ("reference_AGBD_max", float(agbd_ref.max()) if len(agbd_ref) else np.nan),
        ("NISAR_filename", nisar_name),
        ("NISAR_datetime", nisar_dt.isoformat() if nisar_dt else ""),
        ("repeat_detection_radius_m", REPEAT_RADIUS_M),
        ("projected_CRS_for_repeat_detection", PROJECTED_CRS),
    ]
    pd.DataFrame(summary_rows, columns=["metric", "value"]).to_csv(
        PROCESSED_DIR / "GEDI_quality_filter_summary.csv", index=False
    )

    pd.DataFrame(counts_q).to_csv(QC_DIR / "GEDI_quality_5x5_counts.csv", index=False, header=False)
    pd.DataFrame(counts_ref).to_csv(QC_DIR / "GEDI_reference_5x5_counts.csv", index=False, header=False)

    # Counts by acquisition date
    tmp = qdf.copy()
    tmp["date"] = pd.to_datetime(tmp["gedi_datetime"], utc=True).dt.date.astype(str)
    tmp.groupby("date").size().reset_index(name="quality_count").to_csv(
        QC_DIR / "GEDI_counts_by_date.csv", index=False
    )

    # Counts by beam
    qdf.groupby("beam").size().reset_index(name="quality_count").to_csv(
        QC_DIR / "GEDI_counts_by_beam.csv", index=False
    )

    # Repeat-group diagnostics
    repeat_summary = (
        grouped_qdf.groupby("repeat_group_id")
        .agg(
            repeat_group_size=("shot_number", "size"),
            first_date=("gedi_datetime", "min"),
            last_date=("gedi_datetime", "max"),
            mean_longitude=("longitude", "mean"),
            mean_latitude=("latitude", "mean"),
        )
        .reset_index()
        .sort_values(["repeat_group_size", "repeat_group_id"], ascending=[False, True])
    )
    repeat_summary.to_csv(QC_DIR / "GEDI_repeat_location_summary.csv", index=False)

def main():
    print("=" * 72)
    print("PHASE 1 — GEDI L4A V3 PROCESSING AND QUALITY FILTERING")
    print("=" * 72)
    print(f"Input:  {GEDI_DIR}")
    print(f"Output: {PROCESSED_DIR}")
    print(f"QC:     {QC_DIR}")

    nisar_dt, nisar_name = first_nisar_datetime()
    if nisar_dt:
        print(f"NISAR reference acquisition: {nisar_dt.isoformat()} ({nisar_name})")
    else:
        print("WARNING: Could not parse NISAR acquisition date; delta_days_to_NISAR will be blank.")

    all_df = extract_all_gedi()
    print(f"\nAOI records extracted: {len(all_df):,}")

    # Save all AOI footprints before filtering.
    all_df = add_temporal_fields(all_df, nisar_dt)
    all_df.to_csv(PROCESSED_DIR / "GEDI_AOI_all.csv", index=False)

    qdf = apply_primary_filter(all_df)
    print(f"Primary quality-filtered records: {len(qdf):,}")

    # Exact duplicate shot_number diagnostic
    duplicate_shots = int(qdf.duplicated("shot_number").sum())
    print(f"Duplicate shot_number records before exact de-duplication: {duplicate_shots:,}")

    grouped_qdf, ref_df = make_reference_table(qdf)

    # Save quality-filtered table WITH repeat diagnostics.
    grouped_qdf.to_csv(PROCESSED_DIR / "GEDI_AOI_quality_filtered.csv", index=False)
    ref_df.to_csv(PROCESSED_DIR / "GEDI_reference_for_NISAR_matching.csv", index=False)

    save_summaries(all_df, qdf, grouped_qdf, ref_df, nisar_dt, nisar_name)
    save_plots(all_df, grouped_qdf, ref_df)

    occupied_ref, _ = coarse_grid_coverage(ref_df)
    repeat_records = int((grouped_qdf["repeat_group_size"] > 1).sum())
    repeat_groups_gt1 = int(
        grouped_qdf.loc[grouped_qdf["repeat_group_size"] > 1, "repeat_group_id"].nunique()
    )

    print("\n" + "=" * 72)
    print("PHASE 1 PROCESSING COMPLETE — GATE 1 REVIEW REQUIRED")
    print("=" * 72)
    print(f"AOI records:                         {len(all_df):,}")
    print(f"Quality-filtered records:            {len(grouped_qdf):,}")
    print(f"Unique quality shot_number:          {grouped_qdf['shot_number'].nunique():,}")
    print(f"Records in repeated-location groups: {repeat_records:,}")
    print(f"Repeated-location groups (>1):       {repeat_groups_gt1:,}")
    print(f"Primary reference records:           {len(ref_df):,}")
    print(f"Reference 5x5 cells occupied:        {occupied_ref}/25")
    if len(ref_df):
        agbd = pd.to_numeric(ref_df["agbd"], errors="coerce").dropna()
        print(f"Reference AGBD range:                 {agbd.min():.2f} to {agbd.max():.2f} Mg/ha")

    print("\nCreated:")
    for p in [
        PROCESSED_DIR / "GEDI_AOI_all.csv",
        PROCESSED_DIR / "GEDI_AOI_quality_filtered.csv",
        PROCESSED_DIR / "GEDI_reference_for_NISAR_matching.csv",
        PROCESSED_DIR / "GEDI_quality_filter_summary.csv",
        QC_DIR / "GEDI_AGBD_histogram.png",
        QC_DIR / "GEDI_AGBD_SE_histogram.png",
        QC_DIR / "GEDI_footprint_coverage_map.png",
        QC_DIR / "GEDI_counts_by_date.csv",
        QC_DIR / "GEDI_counts_by_beam.csv",
        QC_DIR / "GEDI_repeat_location_summary.csv",
    ]:
        print(f" - {p}")

    append_log([
        f"Script: process_gedi_phase1.py",
        f"Input GEDI folder: {GEDI_DIR}",
        f"NISAR reference acquisition: {nisar_dt.isoformat() if nisar_dt else 'not parsed'}",
        f"AOI records extracted: {len(all_df):,}",
        f"Quality-filtered records: {len(grouped_qdf):,}",
        f"Exact duplicate shot_number records before deduplication: {duplicate_shots:,}",
        f"Repeated-location threshold: {REPEAT_RADIUS_M} m",
        f"Records in repeated-location groups: {repeat_records:,}",
        f"Repeated-location groups (>1): {repeat_groups_gt1:,}",
        f"Primary reference records after keeping most recent valid record per repeat group: {len(ref_df):,}",
        f"Reference coarse spatial coverage: {occupied_ref}/25 5x5 cells",
        f"Python: {sys.version.split()[0]}",
        f"h5py: {package_version('h5py')}; numpy: {package_version('numpy')}; pandas: {package_version('pandas')}; "
        f"pyproj: {package_version('pyproj')}; matplotlib: {package_version('matplotlib')}",
        "Decision: Phase 1 processing finished; Gate 1 requires review before NISAR predictor generation.",
    ])

    print("\nIMPORTANT: Do not start Phase 3 NISAR predictor generation yet.")
    print("Upload GEDI_quality_filter_summary.csv and the terminal output to ChatGPT for Gate 1 review.")
    input("\nPress Enter to close...")

if __name__ == "__main__":
    main()
