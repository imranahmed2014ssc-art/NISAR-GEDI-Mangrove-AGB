import os
from pathlib import Path
from datetime import datetime
import sys
import math

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box
from pyproj import Transformer
import rasterio
import matplotlib.pyplot as plt

# ============================================================
# PROJECT SETTINGS
# ============================================================
PROJECT_ROOT = Path(os.environ.get("NISAR_PROJECT_ROOT", r"D:\NISAR_Mangrove_Paper"))

GEDI_REF = (
    PROJECT_ROOT / "03_GEDI_L4A_V3" / "Processed" /
    "GEDI_reference_for_NISAR_matching.csv"
)

PRED_DIR = PROJECT_ROOT / "02_NISAR_GCOV" / "Predictors"
PREDICTORS = {
    "HH_dB": PRED_DIR / "HH_dB.tif",
    "HV_dB": PRED_DIR / "HV_dB.tif",
    "HH_minus_HV": PRED_DIR / "HH_minus_HV.tif",
    "HV_SD": PRED_DIR / "HV_SD.tif",
}

MANGROVE_MASK = (
    PROJECT_ROOT / "01_AOI" / "Mangrove_Mask" / "Processed" /
    "GMW_2025_NISAR_grid_mask.tif"
)

OUT_DIR = PROJECT_ROOT / "04_Matched_Data"
QC_DIR = OUT_DIR / "QC"
LOG_PATH = PROJECT_ROOT / "00_Documentation" / "analysis_log.txt"

# Approximate GEDI footprint diameter is ~25 m in the approved workflow.
GEDI_RADIUS_M = 12.5

# IMPORTANT:
# These are QC/candidate thresholds only. The final frozen modelling table
# is NOT created until Gate 2 review.
CANDIDATE_MIN_VALID_FRACTION = 0.80
CANDIDATE_MIN_MANGROVE_FRACTION = 0.80

OUT_DIR.mkdir(parents=True, exist_ok=True)
QC_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# HELPERS
# ============================================================
def package_version(name):
    try:
        import importlib.metadata as md
        return md.version(name)
    except Exception:
        return "unknown"

def append_log(lines):
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    block = ["", f"[{stamp}] PHASE 4 — GEDI–NISAR FOOTPRINT MATCHING"] + lines + ["-" * 72]
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write("\n".join(block) + "\n")

def raster_signature(src):
    return (
        str(src.crs),
        src.width,
        src.height,
        tuple(src.transform),
        src.nodata,
    )

def load_rasters():
    for name, p in PREDICTORS.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing predictor: {p}")
    if not MANGROVE_MASK.exists():
        raise FileNotFoundError(f"Missing aligned mangrove mask: {MANGROVE_MASK}")

    datasets = {name: rasterio.open(path) for name, path in PREDICTORS.items()}
    msk_ds = rasterio.open(MANGROVE_MASK)

    ref_name = next(iter(datasets))
    ref = datasets[ref_name]
    ref_grid = (str(ref.crs), ref.width, ref.height, tuple(ref.transform))

    for name, src in datasets.items():
        grid = (str(src.crs), src.width, src.height, tuple(src.transform))
        if grid != ref_grid:
            raise RuntimeError(f"Grid mismatch: {name} does not match {ref_name}.")
    mgrid = (str(msk_ds.crs), msk_ds.width, msk_ds.height, tuple(msk_ds.transform))
    if mgrid != ref_grid:
        raise RuntimeError("GMW raster mask is not exactly aligned with predictor rasters.")

    arrays = {}
    valid_masks = {}
    for name, src in datasets.items():
        arr = src.read(1).astype(np.float64)
        nodata = src.nodata
        valid = np.isfinite(arr)
        if nodata is not None and np.isfinite(nodata):
            valid &= (arr != nodata)
        arrays[name] = arr
        valid_masks[name] = valid

    mangrove = msk_ds.read(1)
    mangrove_valid = mangrove == 1

    info = {
        "crs": ref.crs,
        "transform": ref.transform,
        "width": ref.width,
        "height": ref.height,
        "bounds": ref.bounds,
        "pixel_width": abs(ref.transform.a),
        "pixel_height": abs(ref.transform.e),
    }

    for src in datasets.values():
        src.close()
    msk_ds.close()

    return arrays, valid_masks, mangrove_valid, info

def candidate_pixel_range(footprint, transform, width, height):
    inv = ~transform
    xmin, ymin, xmax, ymax = footprint.bounds

    # Transform bounding-box corners to fractional raster coordinates.
    c0f, r1f = inv * (xmin, ymin)
    c1f, r0f = inv * (xmax, ymax)

    c0 = max(0, int(math.floor(min(c0f, c1f))) - 1)
    c1 = min(width - 1, int(math.ceil(max(c0f, c1f))) + 1)
    r0 = max(0, int(math.floor(min(r0f, r1f))) - 1)
    r1 = min(height - 1, int(math.ceil(max(r0f, r1f))) + 1)
    return r0, r1, c0, c1

def pixel_polygon(transform, row, col):
    x0, y0 = transform * (col, row)
    x1, y1 = transform * (col + 1, row + 1)
    return box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

def extract_one(footprint, arrays, valid_masks, mangrove, info):
    transform = info["transform"]
    height = info["height"]
    width = info["width"]

    r0, r1, c0, c1 = candidate_pixel_range(
        footprint, transform, width, height
    )

    footprint_area = footprint.area
    weighted_sum = {k: 0.0 for k in arrays}
    valid_area = {k: 0.0 for k in arrays}
    all_predictors_valid_area = 0.0
    mangrove_area = 0.0
    intersected_area = 0.0
    intersected_pixel_count = 0
    all_valid_pixel_count = 0

    for row in range(r0, r1 + 1):
        for col in range(c0, c1 + 1):
            pix = pixel_polygon(transform, row, col)
            inter_area = footprint.intersection(pix).area
            if inter_area <= 0:
                continue

            intersected_pixel_count += 1
            intersected_area += inter_area

            if mangrove[row, col]:
                mangrove_area += inter_area

            all_ok = True
            for name in arrays:
                ok = bool(valid_masks[name][row, col])
                if ok:
                    weighted_sum[name] += float(arrays[name][row, col]) * inter_area
                    valid_area[name] += inter_area
                else:
                    all_ok = False

            if all_ok:
                all_predictors_valid_area += inter_area
                all_valid_pixel_count += 1

    out = {}
    for name in arrays:
        out[name] = (
            weighted_sum[name] / valid_area[name]
            if valid_area[name] > 0 else np.nan
        )
        out[f"{name}_valid_fraction"] = (
            valid_area[name] / footprint_area if footprint_area > 0 else np.nan
        )

    out["valid_NISAR_fraction"] = (
        all_predictors_valid_area / footprint_area if footprint_area > 0 else np.nan
    )
    out["mangrove_fraction"] = (
        mangrove_area / footprint_area if footprint_area > 0 else np.nan
    )
    out["footprint_grid_coverage_fraction"] = (
        intersected_area / footprint_area if footprint_area > 0 else np.nan
    )
    out["intersected_pixel_count"] = intersected_pixel_count
    out["all_predictors_valid_pixel_count"] = all_valid_pixel_count
    out["footprint_area_m2"] = footprint_area
    return out

def sample_center_mask(x, y, mangrove, info):
    inv = ~info["transform"]
    colf, rowf = inv * (x, y)
    row = int(math.floor(rowf))
    col = int(math.floor(colf))
    if 0 <= row < info["height"] and 0 <= col < info["width"]:
        return int(bool(mangrove[row, col]))
    return 0

def summarize(df):
    rows = []
    def add(metric, value):
        rows.append({"metric": metric, "value": value})

    add("input_GEDI_reference_records", len(df))
    add("records_with_all_4_predictors", int(df[list(PREDICTORS)].notna().all(axis=1).sum()))
    add("center_inside_mangrove", int((df["center_mangrove"] == 1).sum()))

    for t in [0.50, 0.80, 0.90, 0.95, 0.99, 1.00]:
        add(
            f"valid_NISAR_fraction_ge_{t:.2f}",
            int((df["valid_NISAR_fraction"] >= t - 1e-12).sum())
        )
        add(
            f"mangrove_fraction_ge_{t:.2f}",
            int((df["mangrove_fraction"] >= t - 1e-12).sum())
        )
        add(
            f"center_mangrove_and_valid_ge_{t:.2f}",
            int(((df["center_mangrove"] == 1) &
                 (df["valid_NISAR_fraction"] >= t - 1e-12)).sum())
        )

    candidate = (
        (df["center_mangrove"] == 1) &
        (df["mangrove_fraction"] >= CANDIDATE_MIN_MANGROVE_FRACTION) &
        (df["valid_NISAR_fraction"] >= CANDIDATE_MIN_VALID_FRACTION) &
        df[list(PREDICTORS)].notna().all(axis=1)
    )
    add("candidate_primary_records_80pct_rules", int(candidate.sum()))

    for col in ["agbd", "agbd_se", "delta_days_to_NISAR",
                "valid_NISAR_fraction", "mangrove_fraction"]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(s):
                add(f"{col}_min", float(s.min()))
                add(f"{col}_median", float(s.median()))
                add(f"{col}_mean", float(s.mean()))
                add(f"{col}_max", float(s.max()))

    return pd.DataFrame(rows)

def save_qc_figures(df):
    # Map with legend outside the plotted data.
    fig, ax = plt.subplots(figsize=(8.3, 7.0))
    status = np.where(
        (df["center_mangrove"] == 1) &
        (df["mangrove_fraction"] >= CANDIDATE_MIN_MANGROVE_FRACTION) &
        (df["valid_NISAR_fraction"] >= CANDIDATE_MIN_VALID_FRACTION),
        "Candidate after QC",
        "Needs review / excluded"
    )
    temp = df.copy()
    temp["status"] = status

    for label in ["Candidate after QC", "Needs review / excluded"]:
        sub = temp[temp["status"] == label]
        if len(sub):
            ax.scatter(
                sub["longitude"], sub["latitude"],
                s=12 if label == "Candidate after QC" else 18,
                alpha=0.65,
                label=f"{label} (n={len(sub)})"
            )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("GEDI–NISAR footprint matching QC", pad=12)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        frameon=False
    )
    fig.tight_layout()
    fig.savefig(QC_DIR / "GEDI_NISAR_matching_map.png", dpi=450, bbox_inches="tight")
    fig.savefig(QC_DIR / "GEDI_NISAR_matching_map.pdf", bbox_inches="tight")
    plt.close(fig)

    # Valid NISAR fraction
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    ax.hist(df["valid_NISAR_fraction"].dropna(), bins=30)
    ax.axvline(CANDIDATE_MIN_VALID_FRACTION, linestyle="--", linewidth=1.2)
    ax.set_xlabel("Valid NISAR footprint fraction")
    ax.set_ylabel("GEDI footprint count")
    ax.set_title("Valid NISAR coverage within GEDI footprints", pad=12)
    fig.tight_layout()
    fig.savefig(QC_DIR / "valid_NISAR_fraction_histogram.png", dpi=450, bbox_inches="tight")
    fig.savefig(QC_DIR / "valid_NISAR_fraction_histogram.pdf", bbox_inches="tight")
    plt.close(fig)

    # Mangrove fraction
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    ax.hist(df["mangrove_fraction"].dropna(), bins=30)
    ax.axvline(CANDIDATE_MIN_MANGROVE_FRACTION, linestyle="--", linewidth=1.2)
    ax.set_xlabel("GMW mangrove fraction within GEDI footprint")
    ax.set_ylabel("GEDI footprint count")
    ax.set_title("Mangrove coverage within GEDI footprints", pad=12)
    fig.tight_layout()
    fig.savefig(QC_DIR / "mangrove_fraction_histogram.png", dpi=450, bbox_inches="tight")
    fig.savefig(QC_DIR / "mangrove_fraction_histogram.pdf", bbox_inches="tight")
    plt.close(fig)

def main():
    print("=" * 72)
    print("PHASE 4 — GEDI FOOTPRINT TO NISAR PREDICTOR MATCHING")
    print("=" * 72)

    if not GEDI_REF.exists():
        raise FileNotFoundError(f"GEDI reference table not found:\n{GEDI_REF}")

    df = pd.read_csv(GEDI_REF)
    required = {"shot_number", "longitude", "latitude", "agbd", "agbd_se"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"GEDI reference table is missing columns: {sorted(missing)}")

    print(f"GEDI reference records: {len(df):,}")
    print(f"Nominal footprint diameter: {GEDI_RADIUS_M*2:.1f} m")

    arrays, valid_masks, mangrove, info = load_rasters()
    print(f"NISAR CRS: {info['crs']}")
    print(f"NISAR grid: {info['width']} cols x {info['height']} rows")
    print(f"Pixel size: {info['pixel_width']:.1f} x {info['pixel_height']:.1f} m")
    print("Predictor/mask alignment: PASS")

    transformer = Transformer.from_crs("EPSG:4326", info["crs"], always_xy=True)
    xs, ys = transformer.transform(
        df["longitude"].to_numpy(dtype=float),
        df["latitude"].to_numpy(dtype=float)
    )

    out_rows = []
    for i, (idx, row) in enumerate(df.iterrows(), 1):
        x = float(xs[i-1])
        y = float(ys[i-1])
        footprint = Point(x, y).buffer(GEDI_RADIUS_M, quad_segs=32)

        ext = extract_one(footprint, arrays, valid_masks, mangrove, info)

        rec = row.to_dict()
        rec["nisar_x"] = x
        rec["nisar_y"] = y
        rec["GEDI_footprint_radius_m"] = GEDI_RADIUS_M
        rec["center_mangrove"] = sample_center_mask(x, y, mangrove, info)
        rec.update(ext)
        out_rows.append(rec)

        if i % 100 == 0 or i == len(df):
            print(f"Matched {i:,}/{len(df):,} footprints")

    matched = pd.DataFrame(out_rows)

    # Candidate rule is transparent and provisional; it is NOT the frozen final table.
    matched["candidate_primary_80pct"] = (
        (matched["center_mangrove"] == 1) &
        (matched["mangrove_fraction"] >= CANDIDATE_MIN_MANGROVE_FRACTION) &
        (matched["valid_NISAR_fraction"] >= CANDIDATE_MIN_VALID_FRACTION) &
        matched[list(PREDICTORS)].notna().all(axis=1)
    )

    all_path = OUT_DIR / "GEDI_NISAR_matched_all.csv"
    candidate_path = OUT_DIR / "GEDI_NISAR_matched_candidate_80pct.csv"
    matched.to_csv(all_path, index=False)
    matched.loc[matched["candidate_primary_80pct"]].to_csv(candidate_path, index=False)

    summary = summarize(matched)
    summary_path = QC_DIR / "GEDI_NISAR_matching_summary.csv"
    summary.to_csv(summary_path, index=False)

    save_qc_figures(matched)

    # Predictor descriptive stats at footprints, for Gate 2 sanity checking.
    pstats = []
    for col in list(PREDICTORS) + ["valid_NISAR_fraction", "mangrove_fraction"]:
        s = pd.to_numeric(matched[col], errors="coerce").dropna()
        pstats.append({
            "field": col,
            "n": len(s),
            "min": s.min() if len(s) else np.nan,
            "p01": s.quantile(0.01) if len(s) else np.nan,
            "median": s.median() if len(s) else np.nan,
            "mean": s.mean() if len(s) else np.nan,
            "p99": s.quantile(0.99) if len(s) else np.nan,
            "max": s.max() if len(s) else np.nan,
        })
    pd.DataFrame(pstats).to_csv(QC_DIR / "GEDI_NISAR_matched_predictor_stats.csv", index=False)

    candidate_n = int(matched["candidate_primary_80pct"].sum())
    valid4_n = int(matched[list(PREDICTORS)].notna().all(axis=1).sum())
    center_m_n = int((matched["center_mangrove"] == 1).sum())

    qc_text = [
        "PHASE 4 — GEDI–NISAR FOOTPRINT MATCHING QC",
        "=" * 72,
        f"Input GEDI reference records: {len(matched):,}",
        f"Nominal GEDI footprint diameter: {GEDI_RADIUS_M*2:.1f} m",
        "Extraction method: exact footprint/pixel intersection-area weighted mean",
        f"NISAR CRS: {info['crs']}",
        f"NISAR pixel size: {info['pixel_width']:.1f} x {info['pixel_height']:.1f} m",
        "Predictor and GMW-mask grid alignment: PASS",
        f"Records with all four predictor means: {valid4_n:,}",
        f"Footprint centers inside GMW mangrove mask: {center_m_n:,}",
        f"Candidate records under provisional 80% valid + 80% mangrove rules: {candidate_n:,}",
        "",
        "IMPORTANT:",
        "The 80% coverage thresholds are provisional QC thresholds only.",
        "GEDI_NISAR_matched_all.csv preserves every Phase-1 reference footprint and all diagnostics.",
        "Do not freeze the final modelling table until Gate 2 review.",
    ]
    qc_txt_path = QC_DIR / "GEDI_NISAR_Gate2_QC.txt"
    qc_txt_path.write_text("\n".join(qc_text), encoding="utf-8")

    append_log([
        "Script: match_gedi_nisar_phase4.py",
        f"Input GEDI reference table: {GEDI_REF}",
        f"Input GEDI reference records: {len(matched):,}",
        f"Nominal footprint radius: {GEDI_RADIUS_M:.1f} m",
        "Extraction: exact area-weighted mean of intersecting 10 m NISAR pixels.",
        f"Records with all four predictor means: {valid4_n:,}",
        f"Footprint centers inside GMW mask: {center_m_n:,}",
        f"Provisional candidate rule: valid_NISAR_fraction >= {CANDIDATE_MIN_VALID_FRACTION:.2f}, "
        f"mangrove_fraction >= {CANDIDATE_MIN_MANGROVE_FRACTION:.2f}, center inside mangrove.",
        f"Provisional candidate records: {candidate_n:,}",
        "All matching diagnostics retained; final modelling table intentionally not frozen until Gate 2 review.",
        f"Python: {sys.version.split()[0]}; pandas={package_version('pandas')}; "
        f"numpy={package_version('numpy')}; rasterio={package_version('rasterio')}; "
        f"shapely={package_version('shapely')}; pyproj={package_version('pyproj')}; "
        f"matplotlib={package_version('matplotlib')}",
    ])

    print("\n" + "=" * 72)
    print("PHASE 4 EXTRACTION COMPLETE — GATE 2 REVIEW REQUIRED")
    print("=" * 72)
    print(f"Input GEDI reference records:         {len(matched):,}")
    print(f"Records with all four predictors:     {valid4_n:,}")
    print(f"Centers inside GMW mangrove mask:     {center_m_n:,}")
    print(f"Provisional 80% candidate records:    {candidate_n:,}")

    print("\nCreated:")
    print(f" - {all_path}")
    print(f" - {candidate_path}")
    print(f" - {summary_path}")
    print(f" - {QC_DIR / 'GEDI_NISAR_matched_predictor_stats.csv'}")
    print(f" - {qc_txt_path}")
    print(f" - {QC_DIR / 'GEDI_NISAR_matching_map.png'}")
    print(f" - {QC_DIR / 'valid_NISAR_fraction_histogram.png'}")
    print(f" - {QC_DIR / 'mangrove_fraction_histogram.png'}")

    print("\nIMPORTANT:")
    print("Do not begin EDA/modelling yet.")
    print("Upload Gate-2 QC outputs and terminal result to ChatGPT.")
    None  # interactive pause removed for portable execution

if __name__ == "__main__":
    main()
