import os
from pathlib import Path
from datetime import datetime
import gc
import sys
import traceback

import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import uniform_filter

# ============================================================
# PHASE 8C.1 — COMPACT PRE-SPECIFIED MULTISCALE PREDICTORS
# ============================================================
PROJECT = Path(os.environ.get("NISAR_PROJECT_ROOT", r"D:\NISAR_Mangrove_Paper"))

HH_PATH = PROJECT / "02_NISAR_GCOV" / "Predictors" / "HH_dB.tif"
HV_PATH = PROJECT / "02_NISAR_GCOV" / "Predictors" / "HV_dB.tif"
DIFF_PATH = PROJECT / "02_NISAR_GCOV" / "Predictors" / "HH_minus_HV.tif"
OLD_HVSD_PATH = PROJECT / "02_NISAR_GCOV" / "Predictors" / "HV_SD.tif"

GMW_MASK = (
    PROJECT
    / "01_AOI"
    / "Mangrove_Mask"
    / "Processed"
    / "GMW_2025_NISAR_grid_mask.tif"
)

OUT_ROOT = (
    PROJECT
    / "02_NISAR_GCOV"
    / "Predictor_Expansion"
    / "Phase8C1_Multiscale"
)
RASTER_DIR = OUT_ROOT / "Rasters"
QC_DIR = OUT_ROOT / "QC"
RASTER_DIR.mkdir(parents=True, exist_ok=True)
QC_DIR.mkdir(parents=True, exist_ok=True)

# Fixed, pre-specified windows.
# 10 m NISAR grid => 5x5 ≈ 50 m and 9x9 ≈ 90 m context.
WINDOWS = [5, 9]

NODATA = -9999.0

# ============================================================
# HELPERS
# ============================================================
def read_float(path):
    with rasterio.open(path) as src:
        a = src.read(1, masked=True).astype(np.float32)
        arr = a.filled(np.nan) if np.ma.isMaskedArray(a) else np.asarray(a, np.float32)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        shape = (src.height, src.width)
    return arr, profile, transform, crs, shape

def read_mask(path, ref_shape, ref_transform, ref_crs):
    with rasterio.open(path) as src:
        if (src.height, src.width) != ref_shape:
            raise RuntimeError(
                f"GMW mask shape mismatch: {(src.height, src.width)} vs {ref_shape}"
            )
        if src.transform != ref_transform:
            raise RuntimeError("GMW mask transform does not match predictor grid.")
        if src.crs != ref_crs:
            raise RuntimeError("GMW mask CRS does not match predictor grid.")
        m = src.read(1)
    return m > 0

def write_tif(path, arr, profile):
    p = profile.copy()
    p.update(
        dtype="float32",
        count=1,
        nodata=NODATA,
        compress="LZW",
        tiled=True,
        BIGTIFF="IF_SAFER",
    )
    out = np.where(np.isfinite(arr), arr, NODATA).astype(np.float32)
    with rasterio.open(path, "w", **p) as dst:
        dst.write(out, 1)

def nan_local_mean(arr, size):
    """
    NaN-aware local arithmetic mean in the input domain.
    """
    valid = np.isfinite(arr).astype(np.float32)
    x = np.where(np.isfinite(arr), arr, 0.0).astype(np.float32)

    # uniform_filter returns local mean, so sum/count can be represented as
    # mean(xfilled)/mean(valid).
    sx = uniform_filter(x, size=size, mode="constant", cval=0.0)
    cnt = uniform_filter(valid, size=size, mode="constant", cval=0.0)

    out = np.full(arr.shape, np.nan, dtype=np.float32)
    np.divide(sx, cnt, out=out, where=cnt > 0)
    return out

def nan_local_std(arr, size):
    """
    NaN-aware population SD in dB domain.
    """
    valid_bool = np.isfinite(arr)
    valid = valid_bool.astype(np.float32)
    x = np.where(valid_bool, arr, 0.0).astype(np.float32)

    m1_num = uniform_filter(x, size=size, mode="constant", cval=0.0)
    m2_num = uniform_filter(x*x, size=size, mode="constant", cval=0.0)
    cnt = uniform_filter(valid, size=size, mode="constant", cval=0.0)

    mean = np.zeros(arr.shape, dtype=np.float32)
    mean2 = np.zeros(arr.shape, dtype=np.float32)
    np.divide(m1_num, cnt, out=mean, where=cnt > 0)
    np.divide(m2_num, cnt, out=mean2, where=cnt > 0)

    var = mean2 - mean*mean
    np.maximum(var, 0.0, out=var)
    out = np.sqrt(var, dtype=np.float32)
    out[cnt <= 0] = np.nan
    return out

def linear_power_mean_to_db(db_arr, size):
    """
    Physically appropriate local averaging:
        10*log10(mean(10^(dB/10)))
    rather than arithmetic averaging in dB.
    """
    valid = np.isfinite(db_arr)
    lin = np.zeros(db_arr.shape, dtype=np.float32)
    lin[valid] = np.power(10.0, db_arr[valid] / 10.0, dtype=np.float32)

    mean_lin = nan_local_mean(np.where(valid, lin, np.nan).astype(np.float32), size)
    out = np.full(db_arr.shape, np.nan, dtype=np.float32)
    good = np.isfinite(mean_lin) & (mean_lin > 0)
    out[good] = 10.0 * np.log10(mean_lin[good])
    return out

def summary_row(name, arr, mangrove):
    allv = arr[np.isfinite(arr)]
    mv = arr[np.isfinite(arr) & mangrove]
    row = {
        "predictor": name,
        "valid_AOI_n": int(allv.size),
        "valid_mangrove_n": int(mv.size),
    }
    for prefix, v in [("AOI", allv), ("mangrove", mv)]:
        if v.size:
            row.update({
                f"{prefix}_min": float(np.min(v)),
                f"{prefix}_median": float(np.median(v)),
                f"{prefix}_mean": float(np.mean(v)),
                f"{prefix}_p95": float(np.percentile(v, 95)),
                f"{prefix}_max": float(np.max(v)),
            })
        else:
            for s in ["min","median","mean","p95","max"]:
                row[f"{prefix}_{s}"] = np.nan
    return row

def verify_output(path, shape, transform, crs):
    with rasterio.open(path) as src:
        return (
            (src.height, src.width) == shape
            and src.transform == transform
            and src.crs == crs
            and src.count == 1
        )

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 84)
    print("PHASE 8C.1 — SEPTEMBER COMPACT MULTISCALE NISAR PREDICTORS")
    print("=" * 84)
    print("Pre-specified feature engineering only.")
    print("No GEDI/AGBD information is used in predictor construction or selection.\n")

    for p in [HH_PATH, HV_PATH, DIFF_PATH, OLD_HVSD_PATH, GMW_MASK]:
        if not p.exists():
            raise FileNotFoundError(p)

    hh, profile, transform, crs, shape = read_float(HH_PATH)
    hv, profile_hv, transform_hv, crs_hv, shape_hv = read_float(HV_PATH)
    diff, _, transform_d, crs_d, shape_d = read_float(DIFF_PATH)
    old_hvsd, _, transform_s, crs_s, shape_s = read_float(OLD_HVSD_PATH)

    if not (
        shape == shape_hv == shape_d == shape_s
        and transform == transform_hv == transform_d == transform_s
        and crs == crs_hv == crs_d == crs_s
    ):
        raise RuntimeError("Existing September predictor rasters are not perfectly aligned.")

    mangrove = read_mask(GMW_MASK, shape, transform, crs)

    print(f"Grid: {shape[0]} rows x {shape[1]} cols")
    print(f"CRS: {crs}")
    print(f"Pixel size: {abs(transform.a):.1f} m")
    print(f"Mangrove pixels: {int(mangrove.sum()):,}")

    created = []
    stats = []

    # Legacy variables are included only in QC summary, not regenerated.
    stats.append(summary_row("legacy_HH_dB", hh, mangrove))
    stats.append(summary_row("legacy_HV_dB", hv, mangrove))
    stats.append(summary_row("legacy_HH_minus_HV", diff, mangrove))
    stats.append(summary_row("legacy_HV_SD_3x3", old_hvsd, mangrove))

    # 3x3 HH heterogeneity complements the legacy 3x3 HV_SD.
    print("\nCreating HH_SD_dB_3x3...")
    arr = nan_local_std(hh, 3)
    p = RASTER_DIR / "HH_SD_dB_3x3.tif"
    write_tif(p, arr, profile)
    stats.append(summary_row("HH_SD_dB_3x3", arr, mangrove))
    created.append(p)
    del arr
    gc.collect()

    for w in WINDOWS:
        print(f"\nWindow {w}x{w} (~{w*10} m context)")

        # Physically appropriate local mean of linear gamma0, expressed in dB.
        for base_name, base_arr in [("HH", hh), ("HV", hv)]:
            print(f"  {base_name}_MeanLin_dB_{w}x{w}")
            arr = linear_power_mean_to_db(base_arr, w)
            name = f"{base_name}_MeanLin_dB_{w}x{w}"
            p = RASTER_DIR / f"{name}.tif"
            write_tif(p, arr, profile)
            stats.append(summary_row(name, arr, mangrove))
            created.append(p)
            del arr
            gc.collect()

        # Local heterogeneity in dB domain.
        for base_name, base_arr in [("HH", hh), ("HV", hv)]:
            print(f"  {base_name}_SD_dB_{w}x{w}")
            arr = nan_local_std(base_arr, w)
            name = f"{base_name}_SD_dB_{w}x{w}"
            p = RASTER_DIR / f"{name}.tif"
            write_tif(p, arr, profile)
            stats.append(summary_row(name, arr, mangrove))
            created.append(p)
            del arr
            gc.collect()

        # Polarization-contrast heterogeneity. This is not simply another copy
        # of HH-HV; it measures local spatial variability of that contrast.
        print(f"  PolContrast_SD_dB_{w}x{w}")
        arr = nan_local_std(diff, w)
        name = f"PolContrast_SD_dB_{w}x{w}"
        p = RASTER_DIR / f"{name}.tif"
        write_tif(p, arr, profile)
        stats.append(summary_row(name, arr, mangrove))
        created.append(p)
        del arr
        gc.collect()

    # Alignment verification
    align_rows = []
    for p in created:
        ok = verify_output(p, shape, transform, crs)
        align_rows.append({"file": p.name, "alignment_pass": bool(ok)})

    align_df = pd.DataFrame(align_rows)
    stats_df = pd.DataFrame(stats)

    stats_path = QC_DIR / "Phase8C1_predictor_summary.csv"
    align_path = QC_DIR / "Phase8C1_alignment_check.csv"
    stats_df.to_csv(stats_path, index=False)
    align_df.to_csv(align_path, index=False)

    all_align = bool(align_df["alignment_pass"].all())
    finite_ok = bool(
        (stats_df.loc[~stats_df["predictor"].str.startswith("legacy_"),
                      "valid_mangrove_n"] > 0).all()
    )
    status = "PASS" if all_align and finite_ok else "REVIEW"

    report = [
        "PHASE 8C.1 — SEPTEMBER COMPACT MULTISCALE NISAR PREDICTORS",
        "=" * 84,
        f"Run time: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "DESIGN BASIS",
        "- Phase 8C.0 confirmed the Frequency-A GCOV contains only HHHH and HVHV",
        "  diagonal covariance power rasters; no off-diagonal polarimetric covariance",
        "  layer is available in this dual-pol GCOV.",
        "- Therefore predictor expansion uses spatial structure/context from the",
        "  accepted HH/HV gamma0 backscatter rather than inventing polarimetric",
        "  coherence variables that are not present.",
        "- No GEDI AGBD values were used to construct or choose these predictors.",
        "",
        "FIXED NEW PREDICTOR SET",
        "1. HH_SD_dB_3x3",
        "2. HH_MeanLin_dB_5x5",
        "3. HV_MeanLin_dB_5x5",
        "4. HH_SD_dB_5x5",
        "5. HV_SD_dB_5x5",
        "6. PolContrast_SD_dB_5x5",
        "7. HH_MeanLin_dB_9x9",
        "8. HV_MeanLin_dB_9x9",
        "9. HH_SD_dB_9x9",
        "10. HV_SD_dB_9x9",
        "11. PolContrast_SD_dB_9x9",
        "",
        "FORMULAS",
        "MeanLin_dB(w) = 10*log10(mean(10^(dB/10))) over w x w neighborhood.",
        "SD_dB(w) = population standard deviation of dB values over w x w neighborhood.",
        "PolContrast = HH_dB - HV_dB; its local SD measures spatial variability",
        "of polarization contrast, not a new simple algebraic ratio.",
        "",
        "EXCLUDED FROM PRIMARY BIOMASS PREDICTORS",
        "- numberOfLooks: acquisition/multilooking diagnostic, not vegetation structure.",
        "- mask and inputDataExceptionMask: quality-control layers only.",
        "- rtcGammaToSigmaFactor: normalization/geometry layer; retain as a possible",
        "  later sensitivity diagnostic, not a primary biomass predictor.",
        "- Additional algebraic ratio indices (RVI, normalized difference, HH/HV ratio)",
        "  are not added because with only HH and HV they are deterministic transforms",
        "  of the same polarization contrast and add little independent information.",
        "",
        "GRID",
        f"Shape: {shape}",
        f"CRS: {crs}",
        f"Pixel size: {abs(transform.a)} m",
        f"Mangrove pixels: {int(mangrove.sum()):,}",
        "",
        "QC SUMMARY",
        stats_df.to_string(index=False),
        "",
        "ALIGNMENT",
        align_df.to_string(index=False),
        "",
        f"PHASE 8C.1 STATUS: {status}",
        "",
        "NEXT RULE",
        "Do not use AGBD correlations to cherry-pick predictors from the full dataset.",
        "Phase 8C.2 will match this fixed candidate set to the same frozen 1,641 GEDI",
        "footprints. Phase 8C.3 will evaluate redundancy and predictive utility without",
        "changing the frozen outer spatial validation design.",
    ]

    report_path = QC_DIR / "PHASE8C1_MULTISCALE_PREDICTOR_QC_REPORT.txt"
    report_path.write_text("\n".join(report), encoding="utf-8")

    print("\n" + "=" * 84)
    print(f"PHASE 8C.1 STATUS: {status}")
    print("=" * 84)
    print(f"Created {len(created)} new predictor rasters.")
    for p in created:
        print(" -", p.name)
    print("\nQC:")
    print(report_path)
    print(stats_path)
    print(align_path)

    print("\nIMPORTANT:")
    print("Do not rerun models yet.")
    print("Send the QC report and predictor summary to ChatGPT for review.")
    None  # interactive pause removed for portable execution

if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n" + "=" * 84)
        print("PHASE 8C.1 FAILED")
        print("=" * 84)
        traceback.print_exc()
        None  # interactive pause removed for portable execution
        sys.exit(1)
