from pathlib import Path
from datetime import datetime
import hashlib
import math
import sys
import traceback

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window, from_bounds
from shapely.geometry import Point, box
from pyproj import Transformer

# ============================================================
# PHASE 8C.2 — MATCH EXPANDED SEPTEMBER PREDICTORS TO FROZEN GEDI
# ============================================================
PROJECT = Path(r"D:\NISAR_Mangrove_Paper")

FROZEN_TABLE = (
    PROJECT
    / "05_ML_DL"
    / "Spatial_CV"
    / "GEDI_NISAR_modeling_table_with_spatial_folds.csv"
)
EXPECTED_FROZEN_SHA256 = (
    "c624d8c3c29e295778012b2998058cd454d82d7e9655d0e128bdd8e8779db24e"
)
EXPECTED_N = 1641

PRED_DIR = (
    PROJECT
    / "02_NISAR_GCOV"
    / "Predictor_Expansion"
    / "Phase8C1_Multiscale"
    / "Rasters"
)

OUT_ROOT = (
    PROJECT
    / "04_Matched_Data"
    / "Predictor_Expansion"
    / "Phase8C2"
)
TABLE_DIR = OUT_ROOT / "Tables"
QC_DIR = OUT_ROOT / "QC"
for d in [TABLE_DIR, QC_DIR]:
    d.mkdir(parents=True, exist_ok=True)

PREDICTORS = [
    "HH_SD_dB_3x3",
    "HH_MeanLin_dB_5x5",
    "HV_MeanLin_dB_5x5",
    "HH_SD_dB_5x5",
    "HV_SD_dB_5x5",
    "PolContrast_SD_dB_5x5",
    "HH_MeanLin_dB_9x9",
    "HV_MeanLin_dB_9x9",
    "HH_SD_dB_9x9",
    "HV_SD_dB_9x9",
    "PolContrast_SD_dB_9x9",
]

RASTERS = {name: PRED_DIR / f"{name}.tif" for name in PREDICTORS}

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
        ("utm_x", "utm_y"),
        ("easting", "northing"),
        ("utm_easting", "utm_northing"),
        ("x_utm", "y_utm"),
        ("utm_x_m", "utm_y_m"),
    ]
    for xc, yc in projected_pairs:
        if xc in df.columns and yc in df.columns:
            return xc, yc, "projected"

    lon = detect_col(df, ["longitude", "lon", "gedi_lon"], required=False)
    lat = detect_col(df, ["latitude", "lat", "gedi_lat"], required=False)
    if lon and lat:
        return lon, lat, "geographic"

    raise RuntimeError("Could not detect GEDI coordinates in frozen table.")

def read_window_nan(src, window):
    a = src.read(1, window=window, masked=True).astype(np.float32)
    if np.ma.isMaskedArray(a):
        return a.filled(np.nan)
    out = np.asarray(a, dtype=np.float32)
    if src.nodata is not None:
        out[out == src.nodata] = np.nan
    return out

def pixel_poly(transform, row, col):
    x0 = transform.c + col * transform.a + row * transform.b
    y0 = transform.f + col * transform.d + row * transform.e
    x1 = transform.c + (col + 1) * transform.a + (row + 1) * transform.b
    y1 = transform.f + (col + 1) * transform.d + (row + 1) * transform.e
    return box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

def extract_one_footprint(x, y, srcs):
    footprint = Point(float(x), float(y)).buffer(FOOTPRINT_RADIUS_M, resolution=64)
    footprint_area = float(footprint.area)

    ref = next(iter(srcs.values()))
    wf = from_bounds(*footprint.bounds, transform=ref.transform)

    col0 = max(0, int(math.floor(wf.col_off)))
    row0 = max(0, int(math.floor(wf.row_off)))
    col1 = min(ref.width, int(math.ceil(wf.col_off + wf.width)))
    row1 = min(ref.height, int(math.ceil(wf.row_off + wf.height)))

    if col1 <= col0 or row1 <= row0:
        d = {}
        for k in srcs:
            d[k] = np.nan
            d[k + "_valid_fraction"] = 0.0
        d["expanded_all_valid_fraction"] = 0.0
        d["expanded_intersecting_pixels"] = 0
        return d

    win = Window(col0, row0, col1-col0, row1-row0)
    arrays = {k: read_window_nan(src, win) for k, src in srcs.items()}

    sums = {k: 0.0 for k in srcs}
    valid_area = {k: 0.0 for k in srcs}
    all_valid_area = 0.0
    npix = 0

    for rr in range(row1-row0):
        gr = row0 + rr
        for cc in range(col1-col0):
            gc = col0 + cc
            pp = pixel_poly(ref.transform, gr, gc)
            area = float(footprint.intersection(pp).area)
            if area <= 0:
                continue
            npix += 1

            all_good = True
            for k in srcs:
                v = arrays[k][rr, cc]
                if np.isfinite(v):
                    sums[k] += float(v) * area
                    valid_area[k] += area
                else:
                    all_good = False
            if all_good:
                all_valid_area += area

    d = {}
    for k in srcs:
        d[k] = sums[k] / valid_area[k] if valid_area[k] > 0 else np.nan
        d[k + "_valid_fraction"] = (
            valid_area[k] / footprint_area if footprint_area > 0 else 0.0
        )

    d["expanded_all_valid_fraction"] = (
        all_valid_area / footprint_area if footprint_area > 0 else 0.0
    )
    d["expanded_intersecting_pixels"] = npix
    return d

def summary_table(df):
    rows = []
    for p in PREDICTORS:
        v = pd.to_numeric(df[p], errors="coerce")
        vf = pd.to_numeric(df[p + "_valid_fraction"], errors="coerce")
        finite = v[np.isfinite(v)]
        rows.append({
            "predictor": p,
            "n_finite": int(np.isfinite(v).sum()),
            "n_valid_fraction_ge_80": int((vf >= MIN_VALID_FRACTION).sum()),
            "min": float(finite.min()) if len(finite) else np.nan,
            "median": float(finite.median()) if len(finite) else np.nan,
            "mean": float(finite.mean()) if len(finite) else np.nan,
            "p95": float(finite.quantile(0.95)) if len(finite) else np.nan,
            "max": float(finite.max()) if len(finite) else np.nan,
        })
    return pd.DataFrame(rows)

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 84)
    print("PHASE 8C.2 — MATCH EXPANDED PREDICTORS TO FROZEN 1,641 GEDI FOOTPRINTS")
    print("=" * 84)
    print("No model fitting and no predictor selection are performed in this phase.\n")

    if not FROZEN_TABLE.exists():
        raise FileNotFoundError(FROZEN_TABLE)

    actual_hash = sha256_file(FROZEN_TABLE)
    print("Frozen table SHA256:", actual_hash)
    if actual_hash != EXPECTED_FROZEN_SHA256:
        raise RuntimeError(
            "Frozen modelling table hash mismatch.\n"
            f"Expected: {EXPECTED_FROZEN_SHA256}\n"
            f"Found:    {actual_hash}"
        )

    df = pd.read_csv(FROZEN_TABLE)
    print(f"Frozen records: {len(df):,}")
    if len(df) != EXPECTED_N:
        raise RuntimeError(f"Expected {EXPECTED_N:,} records; found {len(df):,}")

    shot_col = detect_col(df, ["shot_number", "shot", "gedi_shot_number"])
    xcol, ycol, coord_type = detect_xy(df)

    if coord_type == "geographic":
        tr = Transformer.from_crs(4326, 32645, always_xy=True)
        lon = pd.to_numeric(df[xcol], errors="coerce").to_numpy(float)
        lat = pd.to_numeric(df[ycol], errors="coerce").to_numpy(float)
        xs, ys = tr.transform(lon, lat)
    else:
        xs = pd.to_numeric(df[xcol], errors="coerce").to_numpy(float)
        ys = pd.to_numeric(df[ycol], errors="coerce").to_numpy(float)

    if not (np.isfinite(xs).all() and np.isfinite(ys).all()):
        raise RuntimeError("Non-finite GEDI coordinates found.")

    for name, path in RASTERS.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing Phase 8C.1 raster:\n{path}")

    srcs = {name: rasterio.open(path) for name, path in RASTERS.items()}
    try:
        ref = next(iter(srcs.values()))
        for name, src in srcs.items():
            aligned = (
                src.crs == ref.crs
                and src.transform == ref.transform
                and src.width == ref.width
                and src.height == ref.height
            )
            if not aligned:
                raise RuntimeError(f"Raster alignment mismatch: {name}")

        if ref.crs.to_epsg() != 32645:
            raise RuntimeError(f"Expected EPSG:32645; got {ref.crs}")

        print(
            f"Expanded raster grid: {ref.height} x {ref.width}, "
            f"EPSG:{ref.crs.to_epsg()}, {abs(ref.transform.a):.1f} m"
        )

        rows = []
        for i, (x, y) in enumerate(zip(xs, ys), start=1):
            rows.append(extract_one_footprint(x, y, srcs))
            if i == 1 or i % 100 == 0 or i == len(df):
                print(f"\rMatched {i:,}/{len(df):,} footprints", end="", flush=True)
        print()

    finally:
        for src in srcs.values():
            src.close()

    ext = pd.DataFrame(rows)
    out = pd.concat([df.reset_index(drop=True), ext], axis=1)

    # Frozen identity checks
    if out[shot_col].duplicated().any():
        raise RuntimeError("Duplicate shot ID found after matching.")

    all80 = out["expanded_all_valid_fraction"] >= MIN_VALID_FRACTION
    all80_n = int(all80.sum())

    summ = summary_table(out)

    # Save FULL candidate table; never overwrite frozen input.
    out_path = TABLE_DIR / "GEDI_NISAR_expanded_predictors_1641.csv"
    qc_only_path = TABLE_DIR / "expanded_predictor_footprint_QC.csv"
    summ_path = TABLE_DIR / "expanded_predictor_footprint_summary.csv"

    out.to_csv(out_path, index=False)

    qc_cols = [shot_col, "expanded_all_valid_fraction", "expanded_intersecting_pixels"]
    for p in PREDICTORS:
        qc_cols += [p, p + "_valid_fraction"]
    out[qc_cols].to_csv(qc_only_path, index=False)
    summ.to_csv(summ_path, index=False)

    out_hash = sha256_file(out_path)

    # We expect essentially complete coverage because all 1,641 frozen points
    # were already valid in the accepted September scene.
    status = "PASS" if all80_n == EXPECTED_N else "REVIEW"

    report = [
        "PHASE 8C.2 — EXPANDED PREDICTOR / FROZEN GEDI MATCHING QC",
        "=" * 84,
        f"Run time: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "LOCKED INPUT",
        f"Frozen table: {FROZEN_TABLE}",
        f"Frozen input SHA256: {actual_hash}",
        f"Frozen records: {len(df):,}",
        "",
        "MATCHING METHOD",
        f"GEDI nominal footprint diameter: {2*FOOTPRINT_RADIUS_M:.1f} m",
        "Exact footprint/pixel polygon intersection-area weighted extraction.",
        "All 11 expanded predictor rasters were sampled on the same frozen GEDI centers.",
        "No GEDI record was re-filtered and no fold assignment was changed.",
        "",
        "EXPANDED PREDICTOR COVERAGE",
        f"Records with >=80% joint coverage across all 11 predictors: {all80_n:,}/{len(out):,}",
        f"Joint coverage percentage: {100*all80_n/len(out):.3f}%",
        "",
        "FOOTPRINT-LEVEL PREDICTOR SUMMARY",
        summ.to_string(index=False),
        "",
        "OUTPUT",
        f"Expanded candidate table: {out_path}",
        f"Expanded candidate table SHA256: {out_hash}",
        f"QC-only table: {qc_only_path}",
        f"Summary table: {summ_path}",
        "",
        f"PHASE 8C.2 STATUS: {status}",
        "",
        "NEXT RULE",
        "Do not fit models yet.",
        "Phase 8C.3 will assess predictor redundancy and descriptive biomass sensitivity.",
        "Any response-based feature selection must be confined to training data inside the",
        "spatial-CV framework; the outer test folds remain untouched.",
    ]

    report_path = QC_DIR / "PHASE8C2_EXPANDED_GEDI_MATCHING_QC.txt"
    report_path.write_text("\n".join(report), encoding="utf-8")

    print("\n" + "=" * 84)
    print(f"PHASE 8C.2 STATUS: {status}")
    print("=" * 84)
    print(
        f"Joint >=80% coverage: {all80_n:,}/{len(out):,} "
        f"({100*all80_n/len(out):.3f}%)"
    )
    print("\nFootprint-level summaries:")
    print(summ.to_string(index=False))
    print("\nCreated:")
    print(report_path)
    print(out_path)
    print(qc_only_path)
    print(summ_path)
    print("\nIMPORTANT:")
    print("Do not run models yet.")
    print("Send the QC report and expanded predictor summary to ChatGPT.")
    input("\nPress Enter to close...")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n" + "=" * 84)
        print("PHASE 8C.2 FAILED")
        print("=" * 84)
        traceback.print_exc()
        input("\nPress Enter to close...")
        sys.exit(1)
