import os
from pathlib import Path
from datetime import datetime
import sys
import re
import json

import h5py
import numpy as np
import pandas as pd
from pyproj import Transformer
import geopandas as gpd
from shapely.geometry import Polygon
import rasterio
from rasterio.transform import from_origin
from rasterio.features import rasterize
from scipy.ndimage import uniform_filter
import matplotlib.pyplot as plt

# ============================================================
# PROJECT SETTINGS
# ============================================================
PROJECT_ROOT = Path(os.environ.get("NISAR_PROJECT_ROOT", r"D:\NISAR_Mangrove_Paper"))

NISAR_DIR = PROJECT_ROOT / "02_NISAR_GCOV" / "Original"
AOI_SUBSET_DIR = PROJECT_ROOT / "02_NISAR_GCOV" / "AOI_Subset"
PRED_DIR = PROJECT_ROOT / "02_NISAR_GCOV" / "Predictors"
QC_DIR = PROJECT_ROOT / "02_NISAR_GCOV" / "QC"

GMW_PROJECTED = (
    PROJECT_ROOT / "01_AOI" / "Mangrove_Mask" / "Processed" /
    "GMW_2025_Chandpai_Sarankhola_EPSG32645.gpkg"
)
GMW_RASTER_OUT = (
    PROJECT_ROOT / "01_AOI" / "Mangrove_Mask" / "Processed" /
    "GMW_2025_NISAR_grid_mask.tif"
)

LOG_PATH = PROJECT_ROOT / "00_Documentation" / "analysis_log.txt"

WEST, EAST = 89.58, 89.84
SOUTH, NORTH = 21.95, 22.15

FREQ = "/science/LSAR/GCOV/grids/frequencyA"
HHHH_PATH = FREQ + "/HHHH"
HVHV_PATH = FREQ + "/HVHV"
X_PATH = FREQ + "/xCoordinates"
Y_PATH = FREQ + "/yCoordinates"
MASK_PATH = FREQ + "/mask"

OUT_NODATA = -9999.0

for d in [AOI_SUBSET_DIR, PRED_DIR, QC_DIR]:
    d.mkdir(parents=True, exist_ok=True)

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
    block = ["", f"[{stamp}] PHASE 3 — NISAR GCOV PROCESSING"] + lines + ["-" * 72]
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write("\n".join(block) + "\n")

def find_nisar():
    files = sorted(NISAR_DIR.glob("*.h5"))
    if len(files) != 1:
        raise RuntimeError(f"Expected exactly 1 NISAR .h5 file, found {len(files)} in {NISAR_DIR}")
    return files[0]

def scalar(v):
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, np.generic):
        return v.item()
    return v

def attrs_dict(ds):
    out = {}
    for k, v in ds.attrs.items():
        try:
            a = np.asarray(v)
            if a.ndim == 0:
                out[str(k)] = scalar(a.item())
            else:
                out[str(k)] = [scalar(x) for x in a.reshape(-1).tolist()]
        except Exception:
            out[str(k)] = str(v)
    try:
        out["h5py_fillvalue"] = scalar(ds.fillvalue)
    except Exception:
        pass
    return out

def get_fill_values(ds):
    vals = []
    try:
        fv = ds.fillvalue
        if fv is not None:
            vals.append(scalar(fv))
    except Exception:
        pass
    for key in ["_FillValue", "fill_value", "fillValue"]:
        if key in ds.attrs:
            try:
                a = np.asarray(ds.attrs[key]).reshape(-1)
                vals.extend([scalar(x) for x in a])
            except Exception:
                pass
    unique = []
    for v in vals:
        if not any(
            (isinstance(v, float) and isinstance(u, float) and np.isnan(v) and np.isnan(u))
            or v == u for u in unique
        ):
            unique.append(v)
    return unique

def remove_fill(arr, fill_values):
    valid = np.isfinite(arr)
    for fv in fill_values:
        try:
            if isinstance(fv, float) and np.isnan(fv):
                valid &= np.isfinite(arr)
            else:
                valid &= (arr != fv)
        except Exception:
            pass
    return valid

def projected_aoi_polygon(dst_crs):
    tr = Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True)
    # Densify edges so rasterized projected AOI follows transformed geographic rectangle well.
    n = 100
    pts = []
    for lon in np.linspace(WEST, EAST, n):
        pts.append(tr.transform(lon, SOUTH))
    for lat in np.linspace(SOUTH, NORTH, n)[1:]:
        pts.append(tr.transform(EAST, lat))
    for lon in np.linspace(EAST, WEST, n)[1:]:
        pts.append(tr.transform(lon, NORTH))
    for lat in np.linspace(NORTH, SOUTH, n)[1:]:
        pts.append(tr.transform(WEST, lat))
    return Polygon(pts)

def get_epsg(h5):
    # Accepted file previously reported projection=32645.
    candidates = [
        "/science/LSAR/GCOV/grids/frequencyA/projection",
        "/science/LSAR/GCOV/grids/projection",
    ]
    for p in candidates:
        if p in h5:
            v = scalar(h5[p][()])
            try:
                if isinstance(v, str):
                    m = re.search(r"(\d{4,6})", v)
                    if m:
                        return int(m.group(1))
                return int(v)
            except Exception:
                pass

    # Last-resort search by dataset name.
    found = []
    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset) and name.split("/")[-1].lower() == "projection":
            found.append("/" + name)
    h5.visititems(visitor)
    for p in found:
        try:
            v = scalar(h5[p][()])
            if isinstance(v, str):
                m = re.search(r"(\d{4,6})", v)
                if m:
                    return int(m.group(1))
            return int(v)
        except Exception:
            pass
    raise RuntimeError("Could not determine NISAR GCOV EPSG/projection.")

def subset_indices(coords, vmin, vmax):
    coords = np.asarray(coords, dtype=float).reshape(-1)
    ok = np.where((coords >= min(vmin, vmax)) & (coords <= max(vmin, vmax)))[0]
    if len(ok) == 0:
        raise RuntimeError("AOI does not intersect coordinate array.")
    # Add one-pixel buffer around selected centers, bounded by grid.
    i0 = max(0, int(ok.min()) - 1)
    i1 = min(len(coords), int(ok.max()) + 2)
    return i0, i1

def normalize_north_up(arrays, x, y):
    x = np.asarray(x, dtype=float).copy()
    y = np.asarray(y, dtype=float).copy()
    out = [a.copy() for a in arrays]

    # GeoTIFF convention: columns west->east, rows north->south.
    if len(x) > 1 and x[1] < x[0]:
        x = x[::-1]
        out = [a[:, ::-1] for a in out]
    if len(y) > 1 and y[1] > y[0]:
        y = y[::-1]
        out = [a[::-1, :] for a in out]
    return out, x, y

def nan_local_sd_3x3(arr):
    arr = np.asarray(arr, dtype=np.float64)
    finite = np.isfinite(arr)
    z = np.where(finite, arr, 0.0)
    z2 = np.where(finite, arr * arr, 0.0)
    w = 9.0
    count = uniform_filter(finite.astype(np.float64), size=3, mode="constant", cval=0.0) * w
    sm = uniform_filter(z, size=3, mode="constant", cval=0.0) * w
    sm2 = uniform_filter(z2, size=3, mode="constant", cval=0.0) * w

    mean = np.divide(sm, count, out=np.full_like(sm, np.nan), where=count > 0)
    var = np.divide(sm2, count, out=np.full_like(sm2, np.nan), where=count > 0) - mean * mean
    var = np.maximum(var, 0.0)
    sd = np.sqrt(var)
    # Require center valid and at least 5 valid pixels in the 3x3 neighborhood.
    sd[(~finite) | (count < 5)] = np.nan
    return sd.astype(np.float32)

def write_float_tif(path, arr, transform, crs, description):
    data = np.where(np.isfinite(arr), arr, OUT_NODATA).astype(np.float32)
    profile = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": OUT_NODATA,
        "compress": "DEFLATE",
        "predictor": 3,
        "tiled": True,
        "BIGTIFF": "IF_SAFER",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)
        dst.set_band_description(1, description)

def write_uint8_tif(path, arr, transform, crs, description):
    data = np.asarray(arr, dtype=np.uint8)
    profile = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": "uint8",
        "crs": crs,
        "transform": transform,
        "nodata": 0,
        "compress": "DEFLATE",
        "tiled": True,
        "BIGTIFF": "IF_SAFER",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)
        dst.set_band_description(1, description)

def write_original_mask(path, arr, transform, crs, fill_value=None):
    # Preserve source dtype as much as possible.
    data = np.asarray(arr)
    profile = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": str(data.dtype),
        "crs": crs,
        "transform": transform,
        "compress": "DEFLATE",
        "tiled": True,
        "BIGTIFF": "IF_SAFER",
    }
    if fill_value is not None:
        try:
            if np.issubdtype(data.dtype, np.integer) and np.isfinite(fill_value):
                profile["nodata"] = int(fill_value)
            elif np.issubdtype(data.dtype, np.floating):
                profile["nodata"] = float(fill_value)
        except Exception:
            pass
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)
        dst.set_band_description(1, "Original NISAR GCOV frequency-A mask subset")

def stats_row(name, arr, domain=None):
    a = np.asarray(arr)
    m = np.isfinite(a)
    if domain is not None:
        m &= domain
    vals = a[m]
    if vals.size == 0:
        return {
            "raster": name, "valid_n": 0, "valid_pct_domain": 0,
            "min": np.nan, "p01": np.nan, "median": np.nan,
            "mean": np.nan, "p99": np.nan, "max": np.nan, "sd": np.nan
        }
    denom = int(np.sum(domain)) if domain is not None else a.size
    return {
        "raster": name,
        "valid_n": int(vals.size),
        "valid_pct_domain": float(100.0 * vals.size / denom) if denom else np.nan,
        "min": float(np.min(vals)),
        "p01": float(np.percentile(vals, 1)),
        "median": float(np.median(vals)),
        "mean": float(np.mean(vals)),
        "p99": float(np.percentile(vals, 99)),
        "max": float(np.max(vals)),
        "sd": float(np.std(vals)),
    }

def save_histogram(name, arr, domain):
    vals = np.asarray(arr)[np.isfinite(arr) & domain]
    if vals.size == 0:
        return
    lo, hi = np.percentile(vals, [0.5, 99.5])
    plot_vals = vals[(vals >= lo) & (vals <= hi)]
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.hist(plot_vals, bins=50)
    ax.set_xlabel(name.replace("_", " "))
    ax.set_ylabel("Pixel count")
    ax.set_title(f"{name} distribution within GMW mangrove mask", pad=12)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(QC_DIR / f"{name}_histogram.png", dpi=450, bbox_inches="tight")
    fig.savefig(QC_DIR / f"{name}_histogram.pdf", bbox_inches="tight")
    plt.close(fig)

def save_map(name, arr, transform, domain):
    a = np.asarray(arr, dtype=float).copy()
    a[~domain] = np.nan
    vals = a[np.isfinite(a)]
    if vals.size == 0:
        return
    vmin, vmax = np.percentile(vals, [2, 98])

    left = transform.c
    top = transform.f
    right = left + transform.a * a.shape[1]
    bottom = top + transform.e * a.shape[0]

    fig, ax = plt.subplots(figsize=(8.2, 7.0))
    im = ax.imshow(
        a, extent=[left, right, bottom, top], origin="upper",
        vmin=vmin, vmax=vmax, interpolation="nearest"
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
    cb.set_label(name.replace("_", " "))
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title(f"{name} — mangrove-domain QC", pad=12)
    # No legend over data; colorbar is kept outside the plotting area.
    fig.tight_layout()
    fig.savefig(QC_DIR / f"{name}_map.png", dpi=450, bbox_inches="tight")
    fig.savefig(QC_DIR / f"{name}_map.pdf", bbox_inches="tight")
    plt.close(fig)

def main():
    print("=" * 72)
    print("PHASE 3 — NISAR L2 GCOV AOI EXTRACTION + PREDICTOR GENERATION")
    print("=" * 72)

    nisar_fp = find_nisar()
    print(f"NISAR input: {nisar_fp.name}")
    print(f"Input size: {nisar_fp.stat().st_size / 1024**3:.2f} GB")

    if not GMW_PROJECTED.exists():
        raise FileNotFoundError(
            "Projected Phase-2 GMW mask not found:\n"
            f"{GMW_PROJECTED}\nRun/verify Phase 2 first."
        )

    with h5py.File(nisar_fp, "r") as h5:
        for p in [HHHH_PATH, HVHV_PATH, X_PATH, Y_PATH, MASK_PATH]:
            if p not in h5:
                raise RuntimeError(f"Required NISAR dataset missing: {p}")

        epsg = get_epsg(h5)
        crs = f"EPSG:{epsg}"
        print(f"Detected GCOV CRS: {crs}")

        x_full = np.asarray(h5[X_PATH][:], dtype=float).reshape(-1)
        y_full = np.asarray(h5[Y_PATH][:], dtype=float).reshape(-1)

        dx = float(np.median(np.abs(np.diff(x_full))))
        dy = float(np.median(np.abs(np.diff(y_full))))
        print(f"Detected spacing: dx={dx}, dy={dy}")

        # Exact AOI polygon projected to NISAR CRS; use bounds only to window-read.
        aoi_poly = projected_aoi_polygon(crs)
        xmin, ymin, xmax, ymax = aoi_poly.bounds
        c0, c1 = subset_indices(x_full, xmin, xmax)
        r0, r1 = subset_indices(y_full, ymin, ymax)

        print(f"AOI read window: rows {r0}:{r1}, cols {c0}:{c1}")
        print(f"AOI window shape: {(r1-r0, c1-c0)}")

        hh_ds = h5[HHHH_PATH]
        hv_ds = h5[HVHV_PATH]
        mask_ds = h5[MASK_PATH]

        hh = np.asarray(hh_ds[r0:r1, c0:c1], dtype=np.float32)
        hv = np.asarray(hv_ds[r0:r1, c0:c1], dtype=np.float32)
        nmask = np.asarray(mask_ds[r0:r1, c0:c1])
        x = x_full[c0:c1]
        y = y_full[r0:r1]

        hh_attrs = attrs_dict(hh_ds)
        hv_attrs = attrs_dict(hv_ds)
        mask_attrs = attrs_dict(mask_ds)

        hh_fill = get_fill_values(hh_ds)
        hv_fill = get_fill_values(hv_ds)
        mask_fill = get_fill_values(mask_ds)

    # Normalize arrays to north-up / west-east before creating geospatial outputs.
    (hh, hv, nmask), x, y = normalize_north_up([hh, hv, nmask], x, y)

    dx = float(np.median(np.abs(np.diff(x))))
    dy = float(np.median(np.abs(np.diff(y))))
    left = float(np.min(x) - dx / 2.0)
    top = float(np.max(y) + dy / 2.0)
    transform = from_origin(left, top, dx, dy)

    # Rasterize exact geographic AOI polygon to exact NISAR subset grid.
    aoi_poly = projected_aoi_polygon(crs)
    aoi_grid = rasterize(
        [(aoi_poly, 1)],
        out_shape=hh.shape,
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=False,
    ).astype(bool)

    # Rasterize Phase-2 GMW mask to exact same NISAR grid.
    gmw = gpd.read_file(GMW_PROJECTED)
    if gmw.crs is None:
        raise RuntimeError("Phase-2 GMW projected mask has no CRS.")
    gmw = gmw.to_crs(crs)
    mangrove_grid = rasterize(
        [(geom, 1) for geom in gmw.geometry if geom is not None and not geom.is_empty],
        out_shape=hh.shape,
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=False,
    ).astype(bool)
    mangrove_grid &= aoi_grid

    # Conservative validity logic:
    # - Use HHHH/HVHV fill values and finite/positive power.
    # - Use NISAR mask only to exclude its explicit fill value(s), NOT to guess mask class meanings.
    valid = aoi_grid.copy()
    valid &= remove_fill(hh, hh_fill)
    valid &= remove_fill(hv, hv_fill)
    valid &= np.isfinite(hh) & np.isfinite(hv)
    valid &= (hh > 0) & (hv > 0)

    if mask_fill:
        valid &= remove_fill(nmask, mask_fill)

    hh_linear = np.where(valid, hh, np.nan).astype(np.float32)
    hv_linear = np.where(valid, hv, np.nan).astype(np.float32)

    hh_db = np.full(hh.shape, np.nan, dtype=np.float32)
    hv_db = np.full(hv.shape, np.nan, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        hh_db[valid] = (10.0 * np.log10(hh_linear[valid])).astype(np.float32)
        hv_db[valid] = (10.0 * np.log10(hv_linear[valid])).astype(np.float32)

    hh_minus_hv = (hh_db - hv_db).astype(np.float32)
    hv_sd = nan_local_sd_3x3(hv_db)

    # Keep outputs to the exact AOI only.
    for arr in [hh_linear, hv_linear, hh_db, hv_db, hh_minus_hv, hv_sd]:
        arr[~aoi_grid] = np.nan

    # Save AOI source layers.
    write_float_tif(
        AOI_SUBSET_DIR / "NISAR_HHHH_linear_AOI.tif",
        hh_linear, transform, crs, "NISAR GCOV Frequency-A HHHH linear gamma0 power"
    )
    write_float_tif(
        AOI_SUBSET_DIR / "NISAR_HVHV_linear_AOI.tif",
        hv_linear, transform, crs, "NISAR GCOV Frequency-A HVHV linear gamma0 power"
    )
    write_original_mask(
        AOI_SUBSET_DIR / "NISAR_frequencyA_mask_AOI.tif",
        nmask, transform, crs, mask_fill[0] if mask_fill else None
    )
    write_uint8_tif(
        AOI_SUBSET_DIR / "NISAR_valid_power_mask_AOI.tif",
        valid.astype(np.uint8), transform, crs, "Valid positive finite HHHH/HVHV power mask"
    )
    write_uint8_tif(
        AOI_SUBSET_DIR / "AOI_exact_grid_mask.tif",
        aoi_grid.astype(np.uint8), transform, crs, "Exact locked AOI on NISAR grid"
    )

    # Exact GMW mask on NISAR grid.
    write_uint8_tif(
        GMW_RASTER_OUT,
        mangrove_grid.astype(np.uint8), transform, crs,
        "GMW 2025 mangrove extent aligned to exact NISAR GCOV AOI grid"
    )

    # Save predictors.
    predictor_paths = {
        "HH_dB": PRED_DIR / "HH_dB.tif",
        "HV_dB": PRED_DIR / "HV_dB.tif",
        "HH_minus_HV": PRED_DIR / "HH_minus_HV.tif",
        "HV_SD": PRED_DIR / "HV_SD.tif",
    }
    write_float_tif(predictor_paths["HH_dB"], hh_db, transform, crs, "HH_dB = 10 log10(HHHH)")
    write_float_tif(predictor_paths["HV_dB"], hv_db, transform, crs, "HV_dB = 10 log10(HVHV)")
    write_float_tif(
        predictor_paths["HH_minus_HV"], hh_minus_hv, transform, crs,
        "HH_minus_HV = HH_dB - HV_dB"
    )
    write_float_tif(
        predictor_paths["HV_SD"], hv_sd, transform, crs,
        "HV_SD = local 3x3 standard deviation of HV_dB"
    )

    # QC: confirm all four predictor files are exactly aligned.
    alignment_rows = []
    reference_sig = None
    alignment_pass = True
    for name, p in predictor_paths.items():
        with rasterio.open(p) as src:
            sig = (
                str(src.crs), src.width, src.height,
                tuple(src.transform), src.bounds.left, src.bounds.bottom,
                src.bounds.right, src.bounds.top
            )
            if reference_sig is None:
                reference_sig = sig
            same = (sig == reference_sig)
            alignment_pass &= same
            alignment_rows.append({
                "predictor": name,
                "path": str(p),
                "crs": str(src.crs),
                "width": src.width,
                "height": src.height,
                "pixel_width": src.transform.a,
                "pixel_height": abs(src.transform.e),
                "left": src.bounds.left,
                "bottom": src.bounds.bottom,
                "right": src.bounds.right,
                "top": src.bounds.top,
                "matches_reference_grid": same,
            })
    pd.DataFrame(alignment_rows).to_csv(QC_DIR / "NISAR_predictor_alignment.csv", index=False)

    # NISAR source-mask diagnostics, without guessing class semantics.
    unique_vals, counts = np.unique(nmask, return_counts=True)
    order = np.argsort(counts)[::-1]
    mask_rows = []
    for idx in order[:100]:
        mask_rows.append({
            "mask_value": scalar(unique_vals[idx]),
            "count": int(counts[idx]),
            "percent_window": float(100.0 * counts[idx] / nmask.size),
        })
    pd.DataFrame(mask_rows).to_csv(QC_DIR / "NISAR_mask_value_counts.csv", index=False)

    with open(QC_DIR / "NISAR_HDF5_layer_attributes.json", "w", encoding="utf-8") as f:
        json.dump({
            "HHHH": hh_attrs,
            "HVHV": hv_attrs,
            "mask": mask_attrs,
            "HHHH_fill_values": hh_fill,
            "HVHV_fill_values": hv_fill,
            "mask_fill_values": mask_fill,
            "note": (
                "The GCOV mask is preserved and inspected. This script only excludes explicit "
                "mask fill values; it does not assume the scientific mask classes are binary-validity codes."
            )
        }, f, indent=2, default=str)

    # Statistics within the mangrove domain.
    stats = []
    for name, arr in [
        ("HH_dB", hh_db),
        ("HV_dB", hv_db),
        ("HH_minus_HV", hh_minus_hv),
        ("HV_SD", hv_sd),
    ]:
        stats.append(stats_row(name, arr, mangrove_grid))
        save_histogram(name, arr, mangrove_grid)
        save_map(name, arr, transform, mangrove_grid)

    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(QC_DIR / "NISAR_predictor_stats_mangrove.csv", index=False)

    mangrove_pixels = int(np.sum(mangrove_grid))
    valid_mangrove_pixels = int(np.sum(valid & mangrove_grid))
    valid_mangrove_pct = (
        100.0 * valid_mangrove_pixels / mangrove_pixels if mangrove_pixels else np.nan
    )

    # Calculate rasterized mangrove area from exact NISAR grid for cross-check.
    pixel_area_m2 = abs(transform.a * transform.e)
    raster_mangrove_area_km2 = mangrove_pixels * pixel_area_m2 / 1e6

    qc_summary = [
        "PHASE 3 — NISAR GCOV PREDICTOR QC",
        "=" * 72,
        f"NISAR file: {nisar_fp.name}",
        f"Filename identity: L2 GCOV (from product filename)",
        f"GCOV Frequency-A group: {FREQ}",
        f"CRS: {crs}",
        f"Pixel spacing: {dx:.3f} m x {dy:.3f} m",
        f"AOI subset dimensions: {hh.shape[1]} columns x {hh.shape[0]} rows",
        f"AOI subset transform: {transform}",
        f"HHHH fill values: {hh_fill}",
        f"HVHV fill values: {hv_fill}",
        f"NISAR mask fill values: {mask_fill}",
        f"Mangrove-grid pixels: {mangrove_pixels:,}",
        f"Valid radar pixels inside mangrove grid: {valid_mangrove_pixels:,}",
        f"Valid radar percentage inside mangrove grid: {valid_mangrove_pct:.3f}%",
        f"Rasterized GMW mangrove area: {raster_mangrove_area_km2:.3f} km2",
        f"Predictor grid alignment: {'PASS' if alignment_pass else 'FAIL'}",
        "HV_SD rule: 3x3 local SD; center valid; >=5 valid neighbors required.",
        (
            "NISAR mask handling: original mask preserved; explicit fill excluded; "
            "mask class values are not guessed as binary-validity codes."
        ),
        "Visual QC required: inspect HH_dB, HV_dB, HH_minus_HV, and HV_SD maps for seams, striping, blank blocks, or extreme artifacts.",
    ]
    (QC_DIR / "NISAR_Phase3_QC_summary.txt").write_text("\n".join(qc_summary), encoding="utf-8")

    # Metadata table.
    metadata = [
        ("nisar_file", nisar_fp.name),
        ("product_identity_from_filename", "NISAR L2 PR GCOV"),
        ("frequency_group", FREQ),
        ("hhhh_path", HHHH_PATH),
        ("hvhv_path", HVHV_PATH),
        ("crs", crs),
        ("pixel_spacing_x_m", dx),
        ("pixel_spacing_y_m", dy),
        ("subset_rows", hh.shape[0]),
        ("subset_cols", hh.shape[1]),
        ("grid_left", left),
        ("grid_top", top),
        ("grid_pixel_area_m2", pixel_area_m2),
        ("gmw_raster_mask", str(GMW_RASTER_OUT)),
        ("mangrove_pixels", mangrove_pixels),
        ("valid_mangrove_pixels", valid_mangrove_pixels),
        ("valid_mangrove_pct", valid_mangrove_pct),
        ("rasterized_mangrove_area_km2", raster_mangrove_area_km2),
        ("predictor_alignment_pass", alignment_pass),
        ("nodata_value_float_outputs", OUT_NODATA),
    ]
    pd.DataFrame(metadata, columns=["field", "value"]).to_csv(
        QC_DIR / "NISAR_Phase3_metadata.csv", index=False
    )

    append_log([
        "Script: process_nisar_phase3.py",
        f"Input NISAR file: {nisar_fp}",
        f"Product identity from filename: NISAR L2 PR GCOV",
        f"Frequency-A HHHH: {HHHH_PATH}",
        f"Frequency-A HVHV: {HVHV_PATH}",
        f"GCOV CRS: {crs}",
        f"GCOV pixel spacing: {dx:.3f} x {dy:.3f} m",
        f"AOI subset dimensions: rows={hh.shape[0]}, cols={hh.shape[1]}",
        f"Valid radar percentage inside GMW mangrove grid: {valid_mangrove_pct:.3f}%",
        f"Rasterized GMW mangrove area: {raster_mangrove_area_km2:.3f} km2",
        "Predictors created: HH_dB, HV_dB, HH_minus_HV, HV_SD.",
        f"Predictor grid alignment: {'PASS' if alignment_pass else 'FAIL'}",
        "NISAR GCOV mask preserved and inspected; only explicit fill excluded from validity logic.",
        "Phase 3 QC maps/histograms created at 450 dpi plus PDF copies.",
        "Decision: Phase 3 generation complete; visual/statistical QC review required before GEDI-NISAR matching.",
        f"Python: {sys.version.split()[0]}; h5py={package_version('h5py')}; "
        f"numpy={package_version('numpy')}; rasterio={package_version('rasterio')}; "
        f"scipy={package_version('scipy')}; geopandas={package_version('geopandas')}; "
        f"matplotlib={package_version('matplotlib')}",
    ])

    print("\n" + "=" * 72)
    print("PHASE 3 PROCESSING COMPLETE — QC REVIEW REQUIRED")
    print("=" * 72)
    print(f"CRS:                           {crs}")
    print(f"Pixel spacing:                 {dx:.1f} x {dy:.1f} m")
    print(f"AOI grid:                      {hh.shape[1]} cols x {hh.shape[0]} rows")
    print(f"Mangrove pixels:               {mangrove_pixels:,}")
    print(f"Valid radar in mangrove:       {valid_mangrove_pct:.3f}%")
    print(f"Rasterized mangrove area:      {raster_mangrove_area_km2:.3f} km2")
    print(f"Predictor grid alignment:      {'PASS' if alignment_pass else 'FAIL'}")

    print("\nPredictors:")
    for name, p in predictor_paths.items():
        print(f" - {name}: {p}")

    print("\nExact aligned mangrove mask:")
    print(f" - {GMW_RASTER_OUT}")

    print("\nQC outputs:")
    print(f" - {QC_DIR / 'NISAR_Phase3_QC_summary.txt'}")
    print(f" - {QC_DIR / 'NISAR_predictor_stats_mangrove.csv'}")
    print(f" - {QC_DIR / 'NISAR_predictor_alignment.csv'}")
    print(f" - {QC_DIR / 'NISAR_mask_value_counts.csv'}")
    print(f" - {QC_DIR / 'NISAR_HDF5_layer_attributes.json'}")
    print(f" - maps/histograms in {QC_DIR}")

    print("\nIMPORTANT:")
    print("Do not start GEDI-NISAR matching yet.")
    print("Upload the QC summary, predictor stats/alignment CSVs, and the four predictor map PNGs to ChatGPT.")
    None  # interactive pause removed for portable execution

if __name__ == "__main__":
    main()
