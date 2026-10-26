from pathlib import Path
from datetime import datetime
import hashlib
import shutil
import sys
import traceback
import math

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds, Window
from shapely.geometry import Point, box
from pyproj import Transformer
from scipy.stats import pearsonr, spearmanr

# ============================================================
# PATHS — JUNE SENSITIVITY OUTPUTS STAY ISOLATED
# ============================================================
JUNE_ROOT = Path(r"D:\new_nisar_data\Phase8B2_June25_Sensitivity_V2")
PRED_DIR = JUNE_ROOT / "01_Predictors"
OUT_ROOT = Path(r"D:\new_nisar_data\Phase8B2_June25_Sensitivity_V2\02_GEDI_Matching")
SNAP_DIR = OUT_ROOT / "Reference_Snapshot"
TABLE_DIR = OUT_ROOT / "Tables"
QC_DIR = OUT_ROOT / "QC"

for d in [SNAP_DIR, TABLE_DIR, QC_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# The existing project is READ ONLY in this phase.
MAIN_FROZEN_TABLE = Path(
    r"D:\NISAR_Mangrove_Paper\05_ML_DL\Spatial_CV"
    r"\GEDI_NISAR_modeling_table_with_spatial_folds.csv"
)
EXPECTED_FROZEN_SHA256 = (
    "c624d8c3c29e295778012b2998058cd454d82d7e9655d0e128bdd8e8779db24e"
)

RASTERS = {
    "June25_HH_dB": PRED_DIR / "June25_HH_dB.tif",
    "June25_HV_dB": PRED_DIR / "June25_HV_dB.tif",
    "June25_HH_minus_HV": PRED_DIR / "June25_HH_minus_HV.tif",
    "June25_HV_SD": PRED_DIR / "June25_HV_SD.tif",
}

JUNE_NISAR_DT = pd.Timestamp("2026-06-25T23:29:21Z")
FOOTPRINT_RADIUS_M = 12.5
MIN_VALID_FRACTION = 0.80

# ============================================================
# HELPERS
# ============================================================
def sha256_file(path, chunk_size=8 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def detect_col(df, candidates, required=True):
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    if required:
        raise RuntimeError(
            "Could not find any of these columns: " + ", ".join(candidates)
        )
    return None

def detect_xy(df):
    projected_pairs = [
        ("easting", "northing"),
        ("utm_easting", "utm_northing"),
        ("x_utm", "y_utm"),
        ("utm_x", "utm_y"),
        ("x_coord", "y_coord"),
    ]
    for xc, yc in projected_pairs:
        if xc in df.columns and yc in df.columns:
            return xc, yc, "projected"

    lon_candidates = ["longitude", "lon", "longitude_deg", "gedi_lon"]
    lat_candidates = ["latitude", "lat", "latitude_deg", "gedi_lat"]
    lon = detect_col(df, lon_candidates, required=False)
    lat = detect_col(df, lat_candidates, required=False)
    if lon and lat:
        return lon, lat, "geographic"

    # Last-resort x/y if values look projected.
    if "x" in df.columns and "y" in df.columns:
        xmed = pd.to_numeric(df["x"], errors="coerce").median()
        ymed = pd.to_numeric(df["y"], errors="coerce").median()
        if abs(xmed) > 1000 and abs(ymed) > 1000:
            return "x", "y", "projected"

    raise RuntimeError(
        "Could not detect GEDI coordinates in frozen modelling table."
    )

def read_window_as_nan(src, window):
    a = src.read(1, window=window, masked=True).astype(np.float32)
    if np.ma.isMaskedArray(a):
        a = a.filled(np.nan)
    else:
        a = np.asarray(a, dtype=np.float32)
        if src.nodata is not None:
            a[a == src.nodata] = np.nan
    return a

def pixel_polygon(transform, row, col):
    # Works for the north-up grids used here.
    x0 = transform.c + col * transform.a + row * transform.b
    y0 = transform.f + col * transform.d + row * transform.e
    x1 = transform.c + (col + 1) * transform.a + (row + 1) * transform.b
    y1 = transform.f + (col + 1) * transform.d + (row + 1) * transform.e
    return box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

def exact_footprint_extract(x, y, sources):
    # High-resolution polygonal approximation of 25 m GEDI footprint circle.
    footprint = Point(float(x), float(y)).buffer(FOOTPRINT_RADIUS_M, resolution=64)
    footprint_area = float(footprint.area)

    ref = next(iter(sources.values()))
    wf = from_bounds(*footprint.bounds, transform=ref.transform)

    col0 = max(0, int(math.floor(wf.col_off)))
    row0 = max(0, int(math.floor(wf.row_off)))
    col1 = min(ref.width, int(math.ceil(wf.col_off + wf.width)))
    row1 = min(ref.height, int(math.ceil(wf.row_off + wf.height)))

    if col1 <= col0 or row1 <= row0:
        out = {k: np.nan for k in sources}
        out.update({
            "June25_valid_fraction_all": 0.0,
            "June25_intersection_area_m2": 0.0,
            "June25_intersecting_pixels": 0,
        })
        return out

    win = Window(col0, row0, col1-col0, row1-row0)
    arrays = {k: read_window_as_nan(src, win) for k, src in sources.items()}

    sums = {k: 0.0 for k in sources}
    valid_area = {k: 0.0 for k in sources}
    all4_area = 0.0
    intersect_area = 0.0
    npix = 0

    h = row1 - row0
    w = col1 - col0

    for rr in range(h):
        global_row = row0 + rr
        for cc in range(w):
            global_col = col0 + cc
            pix = pixel_polygon(ref.transform, global_row, global_col)
            area = float(footprint.intersection(pix).area)
            if area <= 0:
                continue

            npix += 1
            intersect_area += area

            finite_all = True
            for k in sources:
                v = arrays[k][rr, cc]
                if np.isfinite(v):
                    sums[k] += float(v) * area
                    valid_area[k] += area
                else:
                    finite_all = False

            if finite_all:
                all4_area += area

    out = {}
    for k in sources:
        out[k] = (
            sums[k] / valid_area[k] if valid_area[k] > 0 else np.nan
        )
        out[k + "_valid_fraction"] = (
            valid_area[k] / footprint_area if footprint_area > 0 else 0.0
        )

    out["June25_valid_fraction_all"] = (
        all4_area / footprint_area if footprint_area > 0 else 0.0
    )
    out["June25_intersection_area_m2"] = intersect_area
    out["June25_intersecting_pixels"] = npix
    return out

def temporal_gap_days(gedi_series):
    dt = pd.to_datetime(gedi_series, utc=True, errors="coerce")
    return np.abs((JUNE_NISAR_DT - dt).dt.total_seconds() / 86400.0)

def safe_corr(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan, np.nan, int(m.sum())
    return (
        float(pearsonr(x[m], y[m]).statistic),
        float(spearmanr(x[m], y[m]).statistic),
        int(m.sum()),
    )

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 80)
    print("PHASE 8B.2B — JUNE 25 NISAR / FROZEN GEDI FOOTPRINT MATCHING")
    print("=" * 80)
    print("Existing project is READ ONLY.")
    print("All new files are written under D:\\new_nisar_data only.\n")

    if not MAIN_FROZEN_TABLE.exists():
        raise FileNotFoundError(f"Missing frozen table:\n{MAIN_FROZEN_TABLE}")

    actual_sha = sha256_file(MAIN_FROZEN_TABLE)
    print("Frozen table SHA256:", actual_sha)
    if actual_sha != EXPECTED_FROZEN_SHA256:
        raise RuntimeError(
            "Frozen modelling table hash does not match the locked Phase-6 table.\n"
            f"Expected: {EXPECTED_FROZEN_SHA256}\n"
            f"Found:    {actual_sha}\n"
            "Stop here so the June sensitivity uses exactly the original records."
        )

    # Copy the frozen input into the isolated workspace for audit.
    snapshot = SNAP_DIR / "GEDI_NISAR_modeling_table_with_spatial_folds_READONLY_COPY.csv"
    shutil.copy2(MAIN_FROZEN_TABLE, snapshot)
    if sha256_file(snapshot) != EXPECTED_FROZEN_SHA256:
        raise RuntimeError("Reference snapshot copy hash mismatch.")

    df = pd.read_csv(snapshot)
    print(f"Frozen records: {len(df):,}")
    if len(df) != 1641:
        raise RuntimeError(f"Expected 1,641 frozen records; found {len(df):,}.")

    # Detect key columns.
    xcol, ycol, coord_type = detect_xy(df)
    agbd_col = detect_col(df, ["agbd", "observed_agbd", "AGBD"])
    shot_col = detect_col(df, ["shot_number", "shot", "gedi_shot_number"], required=False)
    datetime_col = detect_col(
        df,
        ["gedi_datetime", "gedi_date", "acquisition_datetime", "acquisition_date"],
        required=False
    )

    print(f"Coordinates: {xcol}, {ycol} ({coord_type})")
    print(f"AGBD column: {agbd_col}")
    if shot_col:
        print(f"Shot ID column: {shot_col}")
    if datetime_col:
        print(f"GEDI time column: {datetime_col}")

    if coord_type == "geographic":
        transformer = Transformer.from_crs(4326, 32645, always_xy=True)
        lon = pd.to_numeric(df[xcol], errors="coerce").to_numpy(float)
        lat = pd.to_numeric(df[ycol], errors="coerce").to_numpy(float)
        xs, ys = transformer.transform(lon, lat)
    else:
        xs = pd.to_numeric(df[xcol], errors="coerce").to_numpy(float)
        ys = pd.to_numeric(df[ycol], errors="coerce").to_numpy(float)

    if not (np.isfinite(xs).all() and np.isfinite(ys).all()):
        raise RuntimeError("Non-finite GEDI coordinates found.")

    # Open and verify raster grids.
    for k, p in RASTERS.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing June predictor raster:\n{p}")

    srcs = {k: rasterio.open(p) for k, p in RASTERS.items()}
    try:
        ref = next(iter(srcs.values()))
        for k, src in srcs.items():
            same = (
                src.crs == ref.crs
                and src.transform == ref.transform
                and src.width == ref.width
                and src.height == ref.height
            )
            if not same:
                raise RuntimeError(f"Raster grid mismatch for {k}.")
        if ref.crs.to_epsg() != 32645:
            raise RuntimeError(f"Expected EPSG:32645; got {ref.crs}.")
        print(
            f"June predictor grid: {ref.height} x {ref.width}, "
            f"EPSG:{ref.crs.to_epsg()}, "
            f"{abs(ref.transform.a):.1f} m pixels"
        )

        # Extract exact footprint-area-weighted values.
        results = []
        n = len(df)
        for i, (x, y) in enumerate(zip(xs, ys), start=1):
            r = exact_footprint_extract(x, y, srcs)
            results.append(r)
            if i == 1 or i % 100 == 0 or i == n:
                print(f"\rMatched {i:,}/{n:,} GEDI footprints", end="", flush=True)
        print()

    finally:
        for src in srcs.values():
            src.close()

    res = pd.DataFrame(results)
    out = pd.concat([df.reset_index(drop=True), res], axis=1)

    # Temporal mismatch to June scene.
    if datetime_col:
        out["June25_delta_days_to_NISAR"] = temporal_gap_days(out[datetime_col])
    elif "delta_days_to_NISAR" in out.columns:
        # Since June 25 is 84 days earlier than Sep 17, reconstruct original
        # GEDI date from the preserved Sep temporal gap, then recompute exactly.
        sep_dt = pd.Timestamp("2026-09-17T23:29:17Z")
        original_gap = pd.to_numeric(out["delta_days_to_NISAR"], errors="coerce")
        reconstructed_gedi = sep_dt - pd.to_timedelta(original_gap, unit="D")
        out["June25_delta_days_to_NISAR"] = np.abs(
            (JUNE_NISAR_DT - reconstructed_gedi).dt.total_seconds() / 86400.0
        )
    else:
        out["June25_delta_days_to_NISAR"] = np.nan

    out["June25_valid_ge_80"] = (
        out["June25_valid_fraction_all"] >= MIN_VALID_FRACTION
    )

    all_path = TABLE_DIR / "GEDI_June25_matched_all_1641.csv"
    pass_path = TABLE_DIR / "GEDI_June25_valid80_subset.csv"
    out.to_csv(all_path, index=False)
    out[out["June25_valid_ge_80"]].to_csv(pass_path, index=False)

    # --------------------------------------------------------
    # SAME-FOOTPRINT June vs Sep predictor comparison
    # --------------------------------------------------------
    compare_rows = []
    pairs = [
        ("HH_dB", "June25_HH_dB"),
        ("HV_dB", "June25_HV_dB"),
        ("HH_minus_HV", "June25_HH_minus_HV"),
        ("HV_SD", "June25_HV_SD"),
    ]
    for sep_col, june_col in pairs:
        if sep_col not in out.columns:
            continue
        sep = pd.to_numeric(out[sep_col], errors="coerce").to_numpy(float)
        jun = pd.to_numeric(out[june_col], errors="coerce").to_numpy(float)
        m = np.isfinite(sep) & np.isfinite(jun)
        pr, sr, nn = safe_corr(sep, jun)
        diff = jun[m] - sep[m]
        compare_rows.append({
            "predictor": sep_col,
            "n_paired": nn,
            "Sep17_mean": float(np.mean(sep[m])),
            "June25_mean": float(np.mean(jun[m])),
            "Sep17_median": float(np.median(sep[m])),
            "June25_median": float(np.median(jun[m])),
            "June_minus_Sep_mean": float(np.mean(diff)),
            "June_minus_Sep_median": float(np.median(diff)),
            "paired_Pearson_r": pr,
            "paired_Spearman_rho": sr,
        })

    compare_df = pd.DataFrame(compare_rows)
    compare_path = TABLE_DIR / "June25_vs_Sep17_same_GEDI_predictor_comparison.csv"
    compare_df.to_csv(compare_path, index=False)

    # --------------------------------------------------------
    # June predictor vs AGBD descriptive correlations
    # --------------------------------------------------------
    corr_rows = []
    agbd = pd.to_numeric(out[agbd_col], errors="coerce").to_numpy(float)
    for june_col in [
        "June25_HH_dB",
        "June25_HV_dB",
        "June25_HH_minus_HV",
        "June25_HV_SD",
    ]:
        x = pd.to_numeric(out[june_col], errors="coerce").to_numpy(float)
        pr, sr, nn = safe_corr(x, agbd)
        corr_rows.append({
            "predictor": june_col,
            "n": nn,
            "Pearson_r_with_AGBD": pr,
            "Spearman_rho_with_AGBD": sr,
        })

    corr_df = pd.DataFrame(corr_rows)
    corr_path = TABLE_DIR / "June25_predictor_AGBD_correlations.csv"
    corr_df.to_csv(corr_path, index=False)

    valid_n = int(out["June25_valid_ge_80"].sum())
    valid_pct = 100.0 * valid_n / len(out)

    gap = pd.to_numeric(out["June25_delta_days_to_NISAR"], errors="coerce")
    gap_stats = {
        "n": int(gap.notna().sum()),
        "min": float(gap.min()) if gap.notna().any() else np.nan,
        "median": float(gap.median()) if gap.notna().any() else np.nan,
        "mean": float(gap.mean()) if gap.notna().any() else np.nan,
        "max": float(gap.max()) if gap.notna().any() else np.nan,
    }

    # Distribution summaries at the same 1,641 GEDI footprints.
    foot_rows = []
    for col in [
        "June25_HH_dB",
        "June25_HV_dB",
        "June25_HH_minus_HV",
        "June25_HV_SD",
    ]:
        v = pd.to_numeric(out[col], errors="coerce")
        v = v[np.isfinite(v)]
        foot_rows.append({
            "predictor": col,
            "n": int(len(v)),
            "min": float(v.min()),
            "median": float(v.median()),
            "mean": float(v.mean()),
            "p95": float(v.quantile(0.95)),
            "max": float(v.max()),
        })
    foot_df = pd.DataFrame(foot_rows)
    foot_path = TABLE_DIR / "June25_predictor_stats_at_frozen_GEDI_footprints.csv"
    foot_df.to_csv(foot_path, index=False)

    # Final QC rule: preserve same 1,641 if at least 99% have >=80% valid radar.
    # If fewer, do not automatically drop them; stop for scientific review.
    status = "PASS" if valid_pct >= 99.0 else "REVIEW"

    report = [
        "PHASE 8B.2B — JUNE 25 NISAR / FROZEN GEDI FOOTPRINT MATCHING QC",
        "=" * 80,
        f"Run time: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "ISOLATION",
        "Existing D:\\NISAR_Mangrove_Paper project was read only.",
        f"Frozen table copied to: {snapshot}",
        f"Frozen table SHA256: {actual_sha}",
        f"All new outputs: {OUT_ROOT}",
        "",
        "LOCKED DESIGN",
        f"Frozen records: {len(out):,}",
        f"Nominal GEDI footprint diameter: {2*FOOTPRINT_RADIUS_M:.1f} m",
        "Extraction: exact footprint/pixel polygon intersection-area weighted mean",
        "June raster CRS: EPSG:32645",
        "June raster resolution: 10 m",
        f"Minimum valid radar fraction checked: {MIN_VALID_FRACTION:.2f}",
        "No new mangrove filter was used: the original 1,641 frozen mangrove GEDI",
        "records remain the comparison universe.",
        "",
        "JUNE VALID-RADAR COVERAGE",
        f"Records with >=80% all-four June predictor coverage: {valid_n:,}/{len(out):,}",
        f"Coverage percentage: {valid_pct:.3f}%",
        "",
        "JUNE TEMPORAL GAP TO GEDI",
        f"n={gap_stats['n']:,}",
        f"min={gap_stats['min']:.3f} days",
        f"median={gap_stats['median']:.3f} days",
        f"mean={gap_stats['mean']:.3f} days",
        f"max={gap_stats['max']:.3f} days",
        "",
        "JUNE PREDICTOR STATISTICS AT THE SAME FROZEN GEDI FOOTPRINTS",
        foot_df.to_string(index=False),
        "",
        "JUNE PREDICTOR / GEDI AGBD DESCRIPTIVE CORRELATIONS",
        corr_df.to_string(index=False),
        "",
        "JUNE 25 VS SEP 17 PAIRED PREDICTOR COMPARISON AT SAME GEDI FOOTPRINTS",
        compare_df.to_string(index=False),
        "",
        f"PHASE 8B.2B STATUS: {status}",
        "",
        "NEXT RULE",
        "Do not rerun models yet.",
        "First review whether June valid coverage is essentially complete and whether",
        "the June predictors at the frozen GEDI footprints are plausible/comparable.",
        "If accepted, Phase 8B.2C will rerun only the pre-specified limited models",
        "using the same frozen spatial folds and 500 m buffer.",
    ]

    report_path = QC_DIR / "JUNE25_GEDI_MATCHING_QC.txt"
    report_path.write_text("\n".join(report), encoding="utf-8")

    print("\n" + "=" * 80)
    print(f"PHASE 8B.2B STATUS: {status}")
    print("=" * 80)
    print(
        f"June >=80% valid coverage: {valid_n:,}/{len(out):,} "
        f"({valid_pct:.3f}%)"
    )
    print(
        f"June temporal gap: {gap_stats['min']:.1f}–{gap_stats['max']:.1f} days; "
        f"median={gap_stats['median']:.1f}"
    )
    print("\nJune predictor stats at the frozen GEDI footprints:")
    print(foot_df.to_string(index=False))
    print("\nJune predictor–AGBD correlations:")
    print(corr_df.to_string(index=False))
    print("\nJune vs September same-footprint predictor comparison:")
    print(compare_df.to_string(index=False))

    print("\nCreated:")
    print(report_path)
    print(all_path)
    print(pass_path)
    print(foot_path)
    print(corr_path)
    print(compare_path)

    print("\nIMPORTANT:")
    print("Do not run model sensitivity yet.")
    print("Send JUNE25_GEDI_MATCHING_QC.txt and the terminal output to ChatGPT.")
    input("\nPress Enter to close...")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n" + "=" * 80)
        print("PHASE 8B.2B FAILED")
        print("=" * 80)
        traceback.print_exc()
        input("\nPress Enter to close...")
        sys.exit(1)
