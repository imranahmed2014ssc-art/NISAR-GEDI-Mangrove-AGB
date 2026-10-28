from pathlib import Path
import sys

import numpy as np
import pandas as pd

try:
    import ee
except ImportError:
    raise SystemExit(
        "Earth Engine Python API is not installed.\n"
        "Install with: py -m pip install earthengine-api"
    )

try:
    from pyproj import Transformer
except ImportError:
    raise SystemExit(
        "pyproj is not installed.\n"
        "Install with: py -m pip install pyproj"
    )

try:
    from shapely.geometry import MultiPoint
except ImportError:
    raise SystemExit(
        "shapely is not installed.\n"
        "Install with: py -m pip install shapely"
    )


# ============================================================
# PATHS
# ============================================================

ROOT = Path(r"D:\NISAR_Mangrove_Paper")

MASTER = (
    ROOT
    / "04_Matched_Data"
    / "Predictor_Expansion"
    / "Phase8C2"
    / "Tables"
    / "GEDI_NISAR_expanded_predictors_1641.csv"
)

OUT = (
    ROOT
    / "06_Results"
    / "Reviewer_Validation"
    / "C3_Temporal_Mismatch"
    / "C3_03A_Hydrology"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

TABLE_FILE = (
    OUT
    / "C3_03A_SCENE_HYDROLOGY.csv"
)

DIFF_FILE = (
    OUT
    / "C3_03A_JUNE_MINUS_SEPTEMBER.csv"
)

REPORT = (
    OUT
    / "C3_03A_HYDROLOGY_REPORT.txt"
)


# ============================================================
# FIXED SCENE DATES
# ============================================================

SCENES = {
    "June25_2026":
        "2026-06-25T23:29:21",

    "Sep17_2026":
        "2026-09-17T23:29:17",
}


# ============================================================
# INITIALIZE EARTH ENGINE
# ============================================================

try:

    ee.Initialize(project='effortless-cat-459606-b8')

except Exception as exc:

    print("")
    print("Earth Engine initialization failed.")
    print("")
    print("Run once:")
    print("earthengine authenticate")
    print("")
    print("Then rerun this script.")
    print("")
    raise


# ============================================================
# LOAD FROZEN GEDI TABLE
# ============================================================

df = pd.read_csv(
    MASTER
)

if len(df) != 1641:

    raise RuntimeError(
        f"Expected 1641 rows; "
        f"found {len(df)}."
    )

required = [
    "shot_number",
    "utm_x",
    "utm_y",
]

missing = [
    c
    for c in required
    if c not in df.columns
]

if missing:

    raise RuntimeError(
        f"Missing columns: {missing}"
    )


# ============================================================
# BUILD STUDY FOOTPRINT FROM GEDI COORDINATES
#
# EPSG:32645 -> EPSG:4326
# Convex hull is used only for coarse hydrometeorological
# context, not for biomass modelling.
# ============================================================

transformer = Transformer.from_crs(
    "EPSG:32645",
    "EPSG:4326",
    always_xy=True,
)

lon, lat = transformer.transform(
    df["utm_x"].to_numpy(float),
    df["utm_y"].to_numpy(float),
)

points = [
    (float(x), float(y))
    for x, y
    in zip(lon, lat)
]

hull = MultiPoint(
    points
).convex_hull


if hull.geom_type != "Polygon":

    raise RuntimeError(
        "GEDI footprint convex hull "
        "did not produce a polygon."
    )


coords = [
    [float(x), float(y)]
    for x, y
    in hull.exterior.coords
]

aoi = ee.Geometry.Polygon(
    [coords],
    proj="EPSG:4326",
    geodesic=False,
)


# ============================================================
# DATASETS
# ============================================================

chirps = ee.ImageCollection(
    "UCSB-CHG/CHIRPS/DAILY"
)

era5 = ee.ImageCollection(
    "ECMWF/ERA5_LAND/HOURLY"
)


# ============================================================
# REDUCTION HELPERS
# ============================================================

def reduce_mean(
    image,
    band,
    scale,
):

    value = (
        image
        .select(band)
        .reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=scale,
            bestEffort=True,
            maxPixels=1e9,
        )
        .get(band)
        .getInfo()
    )

    if value is None:
        return np.nan

    return float(value)


def collection_mean(
    collection,
    band,
    start,
    end,
    scale,
):

    image = (
        collection
        .filterDate(
            start,
            end
        )
        .select(
            band
        )
        .mean()
    )

    return reduce_mean(
        image,
        band,
        scale,
    )


def collection_sum(
    collection,
    band,
    start,
    end,
    scale,
    multiplier=1.0,
):

    image = (
        collection
        .filterDate(
            start,
            end
        )
        .select(
            band
        )
        .sum()
        .multiply(
            multiplier
        )
        .rename(
            band
        )
    )

    return reduce_mean(
        image,
        band,
        scale,
    )


# ============================================================
# PROCESS BOTH NISAR SCENES
# ============================================================

rows = []


for scene_name, iso_time in SCENES.items():

    acquisition = ee.Date(
        iso_time
    )

    start_24h = acquisition.advance(
        -24,
        "hour"
    )

    start_7d = acquisition.advance(
        -7,
        "day"
    )

    start_30d = acquisition.advance(
        -30,
        "day"
    )

    # ----------------------------------------
    # CHIRPS
    # ----------------------------------------

    chirps_7d_mm = collection_sum(
        chirps,
        "precipitation",
        start_7d,
        acquisition,
        5566,
        1.0,
    )

    chirps_30d_mm = collection_sum(
        chirps,
        "precipitation",
        start_30d,
        acquisition,
        5566,
        1.0,
    )

    # ----------------------------------------
    # ERA5-LAND SOIL MOISTURE
    # ----------------------------------------

    soil_24h = collection_mean(
        era5,
        "volumetric_soil_water_layer_1",
        start_24h,
        acquisition,
        11132,
    )

    soil_7d = collection_mean(
        era5,
        "volumetric_soil_water_layer_1",
        start_7d,
        acquisition,
        11132,
    )

    # ----------------------------------------
    # ERA5-LAND PRECIPITATION
    #
    # hourly precipitation is metres;
    # multiply by 1000 -> mm.
    # ----------------------------------------

    era5_precip_7d_mm = collection_sum(
        era5,
        "total_precipitation_hourly",
        start_7d,
        acquisition,
        11132,
        1000.0,
    )

    era5_precip_30d_mm = collection_sum(
        era5,
        "total_precipitation_hourly",
        start_30d,
        acquisition,
        11132,
        1000.0,
    )

    # ----------------------------------------
    # ERA5-LAND SURFACE RUNOFF
    #
    # hourly runoff is metres;
    # multiply by 1000 -> mm.
    # ----------------------------------------

    runoff_7d_mm = collection_sum(
        era5,
        "surface_runoff_hourly",
        start_7d,
        acquisition,
        11132,
        1000.0,
    )

    runoff_30d_mm = collection_sum(
        era5,
        "surface_runoff_hourly",
        start_30d,
        acquisition,
        11132,
        1000.0,
    )

    # ----------------------------------------
    # TEMPERATURE
    # ----------------------------------------

    temp_k = collection_mean(
        era5,
        "temperature_2m",
        start_24h,
        acquisition,
        11132,
    )

    temp_c = (
        temp_k - 273.15
        if np.isfinite(temp_k)
        else np.nan
    )

    rows.append({
        "scene":
            scene_name,

        "nisar_datetime_utc":
            iso_time,

        "CHIRPS_precip_7d_mm":
            chirps_7d_mm,

        "CHIRPS_precip_30d_mm":
            chirps_30d_mm,

        "ERA5_soil_water_layer1_24h_mean":
            soil_24h,

        "ERA5_soil_water_layer1_7d_mean":
            soil_7d,

        "ERA5_precip_7d_mm":
            era5_precip_7d_mm,

        "ERA5_precip_30d_mm":
            era5_precip_30d_mm,

        "ERA5_surface_runoff_7d_mm":
            runoff_7d_mm,

        "ERA5_surface_runoff_30d_mm":
            runoff_30d_mm,

        "ERA5_temperature_2m_24h_C":
            temp_c,
    })


result = pd.DataFrame(
    rows
)

result.to_csv(
    TABLE_FILE,
    index=False
)


# ============================================================
# JUNE MINUS SEPTEMBER DIFFERENCES
# ============================================================

june = (
    result[
        result[
            "scene"
        ]
        == "June25_2026"
    ]
    .iloc[0]
)

sep = (
    result[
        result[
            "scene"
        ]
        == "Sep17_2026"
    ]
    .iloc[0]
)


numeric_cols = [
    c
    for c in result.columns
    if c not in [
        "scene",
        "nisar_datetime_utc",
    ]
]


diff_rows = []


for col in numeric_cols:

    j = float(
        june[col]
    )

    s = float(
        sep[col]
    )

    difference = (
        j - s
    )

    percent_difference = (
        difference
        / abs(s)
        * 100.0
        if (
            np.isfinite(s)
            and s != 0
        )
        else np.nan
    )

    diff_rows.append({
        "variable":
            col,

        "June25":
            j,

        "Sep17":
            s,

        "June_minus_Sep":
            difference,

        "percent_difference_relative_to_Sep":
            percent_difference,
    })


diff_df = pd.DataFrame(
    diff_rows
)

diff_df.to_csv(
    DIFF_FILE,
    index=False
)


# ============================================================
# REPORT
# ============================================================

with REPORT.open(
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "=" * 100
        + "\n"
    )

    f.write(
        "C3 STEP 3A - "
        "HYDROMETEOROLOGICAL SCENE-CONDITION AUDIT\n"
    )

    f.write(
        "=" * 100
        + "\n\n"
    )

    f.write(
        "STATUS\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    f.write(
        "NO AGBD MODEL WAS FITTED.\n"
    )

    f.write(
        "C1 and C2 remain frozen.\n"
    )

    f.write(
        "C3 temporal-gap strata remain frozen.\n"
    )

    f.write(
        "Hydrometeorological variables are used "
        "only to characterize the two NISAR scenes.\n\n"
    )


    f.write(
        "STUDY GEOMETRY\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    f.write(
        "Convex hull of the 1,641 frozen GEDI "
        "footprint coordinates.\n"
    )

    f.write(
        "This geometry is used only because "
        "CHIRPS and ERA5-Land are coarse-resolution "
        "environmental datasets.\n\n"
    )


    f.write(
        "DATASETS\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    f.write(
        "CHIRPS Daily: UCSB-CHG/CHIRPS/DAILY\n"
    )

    f.write(
        "ERA5-Land Hourly: ECMWF/ERA5_LAND/HOURLY\n\n"
    )


    f.write(
        "SCENE CONDITIONS\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    f.write(
        result.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )


    f.write(
        "JUNE MINUS SEPTEMBER\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    f.write(
        diff_df.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )


    f.write(
        "INTERPRETATION BOUNDARY\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    f.write(
        "These variables characterize broad "
        "hydrometeorological scene conditions.\n"
    )

    f.write(
        "They do not provide a local tidal-stage "
        "measurement at individual GEDI footprints.\n"
    )

    f.write(
        "Therefore tide remains an unresolved "
        "scene-level confounder unless a reliable "
        "independent tide record or tide model is added.\n"
    )

    f.write(
        "Because only two NISAR scenes are available, "
        "differences in rainfall, soil moisture, runoff, "
        "temperature, acquisition season, and radar "
        "response cannot be causally separated.\n"
    )

    f.write(
        "This analysis is intended to document "
        "environmental differences between the June "
        "and September sensitivity scenes, not to "
        "claim hydrological causation of model error.\n"
    )


# ============================================================
# CONSOLE
# ============================================================

print("")
print("=" * 84)
print("C3 STEP 3A COMPLETE")
print("HYDROMETEOROLOGICAL SCENE-CONDITION AUDIT")
print("=" * 84)

print("")
print(result.to_string(index=False))

print("")
print("JUNE MINUS SEPTEMBER")
print(
    diff_df.to_string(
        index=False
    )
)

print("")
print("Report:")
print(REPORT)

print("")
print("NO AGBD MODEL WAS FITTED.")
print("C1 AND C2 REMAIN FROZEN.")
print("=" * 84)

