import os
from pathlib import Path
from datetime import datetime
import sys
import math

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from shapely.ops import unary_union
from shapely import make_valid
from pyproj import Transformer

try:
    import pyogrio
except Exception as e:
    raise SystemExit(
        "pyogrio is required. Install with:\n"
        "  python -m pip install --upgrade geopandas pyogrio shapely pyproj pandas matplotlib\n"
        f"Original error: {e}"
    )

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(os.environ.get("NISAR_PROJECT_ROOT", r"D:\NISAR_Mangrove_Paper"))
MASK_ROOT = PROJECT_ROOT / "01_AOI" / "Mangrove_Mask"
OUT_DIR = MASK_ROOT / "Processed"
QC_DIR = MASK_ROOT / "QC"
LOG_PATH = PROJECT_ROOT / "00_Documentation" / "analysis_log.txt"

WEST, EAST = 89.58, 89.84
SOUTH, NORTH = 21.95, 22.15
AOI_CRS = "EPSG:4326"
NISAR_CRS = "EPSG:32645"

OUT_DIR.mkdir(parents=True, exist_ok=True)
QC_DIR.mkdir(parents=True, exist_ok=True)

def package_version(name):
    try:
        import importlib.metadata as md
        return md.version(name)
    except Exception:
        return "unknown"

def append_log(lines):
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    block = ["", f"[{stamp}] PHASE 2 — MANGROVE EXTENT MASK"] + lines + ["-" * 72]
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write("\n".join(block) + "\n")

def find_input_gpkg():
    candidates = [p for p in MASK_ROOT.glob("*.gpkg") if p.is_file()]
    if not candidates:
        raise FileNotFoundError(
            f"No .gpkg file was found directly inside:\n{MASK_ROOT}\n"
            "Place the real GMW GeoPackage file in this folder."
        )
    candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
    print("GeoPackage candidates:")
    for p in candidates:
        print(f" - {p.name} ({p.stat().st_size / 1024**2:.1f} MB)")
    chosen = candidates[0]
    print(f"\nSelected input: {chosen.name}")
    return chosen

def bounds_intersect(a, b):
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])

def transform_bounds(bounds, src_crs, dst_crs):
    if str(src_crs) == str(dst_crs):
        return bounds
    tr = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    xmin, ymin, xmax, ymax = bounds
    pts = [tr.transform(x, y) for x, y in [(xmin,ymin),(xmin,ymax),(xmax,ymin),(xmax,ymax)]]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))

def choose_layer(gpkg):
    layers = pyogrio.list_layers(gpkg)
    if len(layers) == 0:
        raise RuntimeError("No layers found inside the GeoPackage.")

    aoi_wgs = (WEST, SOUTH, EAST, NORTH)
    candidates = []
    print("\nLayers discovered:")
    for row in layers:
        name = str(row[0])
        hint = str(row[1])
        try:
            info = pyogrio.read_info(gpkg, layer=name)
        except Exception as e:
            print(f" - {name}: metadata read failed ({e})")
            continue

        geom_type = str(info.get("geometry_type", hint))
        crs = info.get("crs")
        total_bounds = info.get("total_bounds")
        feature_count = info.get("features", 0)
        print(f" - {name}: geometry={geom_type}, features={feature_count}, CRS={crs}")

        if crs is None or total_bounds is None or "Polygon" not in geom_type:
            continue

        try:
            aoi_src = transform_bounds(aoi_wgs, AOI_CRS, crs)
            if bounds_intersect(tuple(total_bounds), tuple(aoi_src)):
                candidates.append((name, crs, aoi_src))
        except Exception:
            pass

    if not candidates:
        raise RuntimeError("No polygon layer could be confirmed to intersect the locked AOI.")

    tested = []
    for name, crs, aoi_src in candidates:
        try:
            subset = pyogrio.read_dataframe(gpkg, layer=name, bbox=aoi_src)
            print(f"   AOI test read — {name}: {len(subset)} feature(s)")
            if len(subset) > 0:
                tested.append((len(subset), name, crs, aoi_src))
        except Exception as e:
            print(f"   AOI test read failed — {name}: {e}")

    if not tested:
        raise RuntimeError("No intersecting polygon features were found in candidate layers.")

    tested.sort(reverse=True, key=lambda x: x[0])
    count, name, crs, aoi_src = tested[0]
    print(f"\nSelected layer: {name} ({count} AOI candidate feature(s))")
    return name, crs, aoi_src

def clean_geometry(gdf):
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf["geometry"] = gdf.geometry.apply(make_valid)
    gdf = gdf[~gdf.geometry.is_empty].copy()
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    return gdf

def save_qc_map(aoi_gdf, mask_gdf):
    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    mask_gdf.plot(ax=ax, alpha=0.75, edgecolor="none")
    aoi_gdf.boundary.plot(ax=ax, linewidth=1.2)
    ax.set_title("GMW 2025 mangrove extent — Chandpai–Sarankhola AOI", pad=12)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(WEST, EAST)
    ax.set_ylim(SOUTH, NORTH)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(QC_DIR / "GMW_2025_mask_QC.png", dpi=450, bbox_inches="tight")
    fig.savefig(QC_DIR / "GMW_2025_mask_QC.pdf", bbox_inches="tight")
    plt.close(fig)

def main():
    print("=" * 72)
    print("PHASE 2 — GLOBAL MANGROVE WATCH 2025 MASK PREPARATION")
    print("=" * 72)
    print(f"Mask folder: {MASK_ROOT}")
    print(f"Processed:   {OUT_DIR}")
    print(f"QC:          {QC_DIR}")
    print(f"Locked AOI:  {WEST}, {SOUTH}, {EAST}, {NORTH}")
    print(f"Target CRS:  {NISAR_CRS}")

    gpkg = find_input_gpkg()
    layer, src_crs, aoi_src_bbox = choose_layer(gpkg)

    print("\nReading only features near the AOI...")
    local = pyogrio.read_dataframe(gpkg, layer=layer, bbox=aoi_src_bbox)
    if local.empty:
        raise RuntimeError("The selected GMW layer returned zero features near the AOI.")

    local = gpd.GeoDataFrame(local, geometry="geometry", crs=src_crs)
    local = clean_geometry(local)
    print(f"Candidate features read: {len(local):,}")
    print(f"Source CRS: {local.crs}")

    local_wgs = local.to_crs(AOI_CRS)
    aoi_geom = box(WEST, SOUTH, EAST, NORTH)
    aoi_gdf = gpd.GeoDataFrame({"AOI_ID": ["CS_AOI_01"]}, geometry=[aoi_geom], crs=AOI_CRS)

    clipped = gpd.clip(local_wgs, aoi_gdf)
    clipped = clean_geometry(clipped)
    if clipped.empty:
        raise RuntimeError("No GMW 2025 mangrove polygons remained after exact AOI clipping.")

    features_wgs_path = OUT_DIR / "GMW_2025_AOI_features_WGS84.gpkg"
    clipped.to_file(features_wgs_path, layer="gmw_2025_aoi_features", driver="GPKG")

    dissolved_geom = make_valid(unary_union(list(clipped.geometry)))
    mask_wgs = gpd.GeoDataFrame(
        {
            "mask_id": ["GMW2025_CS_AOI"],
            "source": ["Global Mangrove Watch v4.1.12 2025 mangrove extent"],
            "role": ["Spatial mask only; not a model predictor"],
        },
        geometry=[dissolved_geom],
        crs=AOI_CRS,
    )

    mask_wgs_path = OUT_DIR / "GMW_2025_Chandpai_Sarankhola_WGS84.gpkg"
    mask_wgs.to_file(mask_wgs_path, layer="mangrove_mask", driver="GPKG")

    mask_utm = mask_wgs.to_crs(NISAR_CRS)
    mask_utm_path = OUT_DIR / "GMW_2025_Chandpai_Sarankhola_EPSG32645.gpkg"
    mask_utm.to_file(mask_utm_path, layer="mangrove_mask", driver="GPKG")

    aoi_utm = aoi_gdf.to_crs(NISAR_CRS)
    mangrove_area_km2 = float(mask_utm.geometry.area.sum() / 1e6)
    aoi_area_km2 = float(aoi_utm.geometry.area.sum() / 1e6)
    mangrove_pct = 100.0 * mangrove_area_km2 / aoi_area_km2 if aoi_area_km2 > 0 else np.nan

    meta_rows = [
        ("source_file", gpkg.name),
        ("source_file_size_mb", round(gpkg.stat().st_size / 1024**2, 3)),
        ("source_layer", layer),
        ("source_crs", str(src_crs)),
        ("source_product", "Global Mangrove Watch v4.1.12 2025 mangrove extent"),
        ("locked_aoi_west", WEST),
        ("locked_aoi_east", EAST),
        ("locked_aoi_south", SOUTH),
        ("locked_aoi_north", NORTH),
        ("aoi_crs", AOI_CRS),
        ("nisar_processing_crs", NISAR_CRS),
        ("clipped_source_feature_count", len(clipped)),
        ("aoi_area_km2_projected", aoi_area_km2),
        ("mangrove_area_km2", mangrove_area_km2),
        ("mangrove_percent_of_aoi", mangrove_pct),
        ("mask_role", "Spatial restriction only; excluded from predictor matrix"),
        ("rasterization_note", "Exact NISAR-grid raster mask will be created in Phase 3 after the GCOV AOI grid is extracted."),
    ]
    pd.DataFrame(meta_rows, columns=["field", "value"]).to_csv(
        OUT_DIR / "mask_metadata.csv", index=False
    )

    qc_text = [
        "PHASE 2 — GMW 2025 MANGROVE MASK QC",
        "=" * 72,
        f"Input GeoPackage: {gpkg}",
        f"Selected layer: {layer}",
        f"Source CRS: {src_crs}",
        f"Clipped source features: {len(clipped):,}",
        f"Projected CRS: {NISAR_CRS}",
        f"Projected AOI area: {aoi_area_km2:.3f} km2",
        f"Mangrove mask area: {mangrove_area_km2:.3f} km2",
        f"Mangrove share of AOI: {mangrove_pct:.2f}%",
        "Geometry status: non-empty polygon/multipolygon mask created",
        "Role: spatial mask only; not a predictor",
        "Rasterization: deferred to Phase 3 for exact alignment to the NISAR GCOV grid",
    ]
    (QC_DIR / "GMW_2025_mask_QC_summary.txt").write_text("\n".join(qc_text), encoding="utf-8")

    save_qc_map(aoi_gdf, mask_wgs)

    append_log([
        "Script: process_mangrove_mask_phase2.py",
        f"Input GMW file: {gpkg}",
        f"Selected source layer: {layer}",
        f"Source CRS: {src_crs}",
        f"Clipped source feature count: {len(clipped):,}",
        f"Output WGS84 mask: {mask_wgs_path}",
        f"Output NISAR-CRS mask: {mask_utm_path}",
        f"Projected AOI area: {aoi_area_km2:.3f} km2",
        f"Mangrove mask area: {mangrove_area_km2:.3f} km2",
        f"Mangrove share of AOI: {mangrove_pct:.2f}%",
        "Mask role: spatial restriction only; not included in predictor matrix.",
        "Exact raster alignment deferred to Phase 3 after the NISAR GCOV AOI grid is extracted.",
        f"Python: {sys.version.split()[0]}",
        f"geopandas: {package_version('geopandas')}; pyogrio: {package_version('pyogrio')}; "
        f"shapely: {package_version('shapely')}; pyproj: {package_version('pyproj')}; "
        f"matplotlib: {package_version('matplotlib')}",
        "Decision: Phase 2 processing finished; review QC outputs before Phase 3.",
    ])

    print("\n" + "=" * 72)
    print("PHASE 2 PROCESSING COMPLETE — REVIEW REQUIRED")
    print("=" * 72)
    print(f"Clipped GMW features:     {len(clipped):,}")
    print(f"Projected AOI area:       {aoi_area_km2:.3f} km2")
    print(f"Mangrove mask area:       {mangrove_area_km2:.3f} km2")
    print(f"Mangrove share of AOI:    {mangrove_pct:.2f}%")
    print("\nCreated:")
    print(f" - {features_wgs_path}")
    print(f" - {mask_wgs_path}")
    print(f" - {mask_utm_path}")
    print(f" - {OUT_DIR / 'mask_metadata.csv'}")
    print(f" - {QC_DIR / 'GMW_2025_mask_QC.png'}")
    print(f" - {QC_DIR / 'GMW_2025_mask_QC.pdf'}")
    print(f" - {QC_DIR / 'GMW_2025_mask_QC_summary.txt'}")
    print("\nIMPORTANT:")
    print("Do not start Phase 3 until the mask area and QC map are reviewed.")
    print("Upload mask_metadata.csv, GMW_2025_mask_QC.png, and the terminal output to ChatGPT.")
    None  # interactive pause removed for portable execution

if __name__ == "__main__":
    main()
