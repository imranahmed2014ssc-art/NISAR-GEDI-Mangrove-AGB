from pathlib import Path
import warnings

import ee
import numpy as np
import pandas as pd

from pyproj import Transformer

warnings.filterwarnings("ignore")


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
    / "C3_03B_Stability"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

POINT_FILE = (
    OUT
    / "C3_03B_POINT_STABILITY_SCREEN.csv"
)

SUMMARY_FILE = (
    OUT
    / "C3_03B_STABILITY_SUMMARY.csv"
)

PASS_FILE = (
    OUT
    / "C3_03B_STABILITY_BY_GEDI_PASS.csv"
)

REPORT = (
    OUT
    / "C3_03B_STABILITY_REPORT.txt"
)


# ============================================================
# EARTH ENGINE
# ============================================================

PROJECT = "effortless-cat-459606-b8"

ee.Initialize(
    project=PROJECT
)


# ============================================================
# FROZEN SCREENING SETTINGS
# ============================================================

NISAR_DATE = "2026-09-17T23:29:17"

WINDOW_DAYS = 365

GEDI_BUFFER_M = 15

S2_SCALE_M = 20

MIN_VALID_OBSERVATIONS = 3


# ------------------------------------------------------------
# THESE ARE SCREENING CUT-OFFS ONLY.
# THEY ARE NOT CLAIMED AS ECOLOGICALLY VALIDATED
# DISTURBANCE THRESHOLDS.
# ------------------------------------------------------------

NDVI_CHANGE_THRESHOLD = 0.15

NBR_CHANGE_THRESHOLD = 0.15

VEG_PROB_CHANGE_THRESHOLD = 0.25

WATER_PROB_CHANGE_THRESHOLD = 0.25


# ============================================================
# LOAD FROZEN MASTER
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
    "source_file",
    "acquisition_date",
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


if (
    df[
        "shot_number"
    ].nunique()
    != 1641
):

    raise RuntimeError(
        "shot_number is not unique."
    )


if (
    df[
        "source_file"
    ].nunique()
    != 8
):

    raise RuntimeError(
        "Expected eight GEDI source-file "
        "acquisition groups."
    )


# ============================================================
# COORDINATES
# ============================================================

transformer = Transformer.from_crs(
    "EPSG:32645",
    "EPSG:4326",
    always_xy=True,
)

lon, lat = transformer.transform(
    df[
        "utm_x"
    ].to_numpy(float),
    df[
        "utm_y"
    ].to_numpy(float),
)

df[
    "_lon"
] = lon

df[
    "_lat"
] = lat


# ============================================================
# SENTINEL-2 PREPROCESSING
# ============================================================

def prepare_s2(
    image
):

    scl = image.select(
        "SCL"
    )

    # Retain clear:
    # 2 = dark area pixels
    # 4 = vegetation
    # 5 = bare soil
    # 6 = water
    #
    # Exclude:
    # no-data, saturated, cloud shadow,
    # unclassified/clouds/cirrus/snow.
    clear = (
        scl.eq(2)
        .Or(
            scl.eq(4)
        )
        .Or(
            scl.eq(5)
        )
        .Or(
            scl.eq(6)
        )
    )

    image = image.updateMask(
        clear
    )

    ndvi = (
        image
        .normalizedDifference(
            [
                "B8",
                "B4",
            ]
        )
        .rename(
            "NDVI"
        )
    )

    nbr = (
        image
        .normalizedDifference(
            [
                "B8",
                "B12",
            ]
        )
        .rename(
            "NBR"
        )
    )

    return (
        ndvi
        .addBands(
            nbr
        )
        .copyProperties(
            image,
            [
                "system:time_start"
            ]
        )
    )


S2 = (
    ee.ImageCollection(
        "COPERNICUS/S2_SR_HARMONIZED"
    )
    .filter(
        ee.Filter.lt(
            "CLOUDY_PIXEL_PERCENTAGE",
            80
        )
    )
    .map(
        prepare_s2
    )
)


DW = ee.ImageCollection(
    "GOOGLE/DYNAMICWORLD/V1"
)


# ============================================================
# BUILD ONE ANNUAL COMPOSITE
# ============================================================

def build_composite(
    end_date,
    prefix,
):

    end = ee.Date(
        end_date
    ).advance(
        1,
        "day"
    )

    start = end.advance(
        -WINDOW_DAYS,
        "day"
    )

    s2_col = (
        S2
        .filterDate(
            start,
            end
        )
    )

    dw_col = (
        DW
        .filterDate(
            start,
            end
        )
    )

    s2_median = (
        s2_col
        .select(
            [
                "NDVI",
                "NBR",
            ]
        )
        .median()
        .rename(
            [
                prefix + "_NDVI",
                prefix + "_NBR",
            ]
        )
    )

    s2_count = (
        s2_col
        .select(
            "NDVI"
        )
        .count()
        .rename(
            prefix
            + "_S2_count"
        )
    )

    dw_mean = (
        dw_col
        .select(
            [
                "trees",
                "flooded_vegetation",
                "water",
            ]
        )
        .mean()
        .rename(
            [
                prefix
                + "_DW_trees",

                prefix
                + "_DW_flooded",

                prefix
                + "_DW_water",
            ]
        )
    )

    dw_count = (
        dw_col
        .select(
            "trees"
        )
        .count()
        .rename(
            prefix
            + "_DW_count"
        )
    )

    composite = (
        s2_median
        .addBands(
            s2_count
        )
        .addBands(
            dw_mean
        )
        .addBands(
            dw_count
        )
    )

    return (
        composite,
        int(
            s2_col.size().getInfo()
        ),
        int(
            dw_col.size().getInfo()
        ),
    )


# ============================================================
# FOLLOW-UP COMPOSITE
#
# Same follow-up composite for all GEDI observations:
# trailing 365 days preceding Sep-17-2026 NISAR.
# ============================================================

(
    follow_image,
    follow_s2_images,
    follow_dw_images,
) = build_composite(
    NISAR_DATE,
    "follow"
)


# ============================================================
# FEATURE COLLECTION HELPER
# ============================================================

def dataframe_to_fc(
    frame
):

    features = []

    for _, row in frame.iterrows():

        geom = (
            ee.Geometry.Point(
                [
                    float(
                        row["_lon"]
                    ),
                    float(
                        row["_lat"]
                    ),
                ]
            )
            .buffer(
                GEDI_BUFFER_M
            )
        )

        features.append(
            ee.Feature(
                geom,
                {
                    "shot_number":
                        str(
                            row["shot_number"]
                        ),

                    "source_file":
                        str(
                            row["source_file"]
                        ),

                    "acquisition_date":
                        str(
                            row["acquisition_date"]
                        ),
                },
            )
        )

    return ee.FeatureCollection(
        features
    )


# ============================================================
# REDUCE IMAGE TO GEDI FOOTPRINT BUFFERS
# ============================================================

def reduce_image(
    image,
    fc,
):

    reduced = (
        image
        .reduceRegions(
            collection=fc,
            reducer=ee.Reducer.mean(),
            scale=S2_SCALE_M,
            tileScale=4,
        )
        .getInfo()
    )

    rows = []

    for feat in reduced[
        "features"
    ]:

        props = feat[
            "properties"
        ]

        rows.append(
            props
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# PROCESS EIGHT GEDI ACQUISITION PASSES
# ============================================================

parts = []

pass_inventory_rows = []


for source_file, group in df.groupby(
    "source_file",
    sort=True
):

    dates = (
        group[
            "acquisition_date"
        ]
        .dropna()
        .astype(str)
        .unique()
    )

    if len(dates) != 1:

        raise RuntimeError(
            f"{source_file}: expected one "
            "acquisition date."
        )

    gedi_date = dates[0]

    print("")
    print(
        "=" * 80
    )

    print(
        "Processing GEDI pass:"
    )

    print(
        source_file
    )

    print(
        "GEDI acquisition:",
        gedi_date
    )

    print(
        "Observations:",
        len(group)
    )


    (
        baseline_image,
        baseline_s2_images,
        baseline_dw_images,
    ) = build_composite(
        gedi_date,
        "baseline"
    )


    combined = (
        baseline_image
        .addBands(
            follow_image
        )
    )


    fc = dataframe_to_fc(
        group
    )


    sampled = reduce_image(
        combined,
        fc
    )


    if len(
        sampled
    ) != len(
        group
    ):

        raise RuntimeError(
            f"{source_file}: expected "
            f"{len(group)} sampled rows, "
            f"found {len(sampled)}."
        )


    sampled[
        "baseline_window_end"
    ] = gedi_date

    sampled[
        "follow_window_end"
    ] = NISAR_DATE

    sampled[
        "baseline_S2_collection_images"
    ] = baseline_s2_images

    sampled[
        "baseline_DW_collection_images"
    ] = baseline_dw_images

    sampled[
        "follow_S2_collection_images"
    ] = follow_s2_images

    sampled[
        "follow_DW_collection_images"
    ] = follow_dw_images


    parts.append(
        sampled
    )


    pass_inventory_rows.append({
        "source_file":
            source_file,

        "gedi_acquisition_date":
            gedi_date,

        "n":
            len(group),

        "baseline_S2_images":
            baseline_s2_images,

        "baseline_DW_images":
            baseline_dw_images,

        "follow_S2_images":
            follow_s2_images,

        "follow_DW_images":
            follow_dw_images,
    })


out = pd.concat(
    parts,
    ignore_index=True
)


# ============================================================
# SHOT-ID QC
# ============================================================

if len(out) != 1641:

    raise RuntimeError(
        f"Expected 1641 output rows; "
        f"found {len(out)}."
    )


if (
    out[
        "shot_number"
    ].nunique()
    != 1641
):

    raise RuntimeError(
        "Output shot_number is not unique."
    )


if (
    set(
        out[
            "shot_number"
        ].astype(str)
    )
    !=
    set(
        df[
            "shot_number"
        ].astype(str)
    )
):

    raise RuntimeError(
        "Output shot set differs from "
        "frozen master."
    )


# ============================================================
# NUMERIC CONVERSION
# ============================================================

numeric_columns = [
    "baseline_NDVI",
    "baseline_NBR",
    "baseline_S2_count",
    "baseline_DW_trees",
    "baseline_DW_flooded",
    "baseline_DW_water",
    "baseline_DW_count",
    "follow_NDVI",
    "follow_NBR",
    "follow_S2_count",
    "follow_DW_trees",
    "follow_DW_flooded",
    "follow_DW_water",
    "follow_DW_count",
]


for col in numeric_columns:

    if col not in out.columns:

        out[
            col
        ] = np.nan

    out[
        col
    ] = pd.to_numeric(
        out[
            col
        ],
        errors="coerce",
    )


# ============================================================
# COMBINED MANGROVE-RELEVANT VEGETATION PROBABILITY
# ============================================================

out[
    "baseline_DW_vegprob"
] = (
    out[
        "baseline_DW_trees"
    ]
    +
    out[
        "baseline_DW_flooded"
    ]
)


out[
    "follow_DW_vegprob"
] = (
    out[
        "follow_DW_trees"
    ]
    +
    out[
        "follow_DW_flooded"
    ]
)


# ============================================================
# CHANGE METRICS
# ============================================================

out[
    "delta_NDVI"
] = (
    out[
        "follow_NDVI"
    ]
    -
    out[
        "baseline_NDVI"
    ]
)


out[
    "delta_NBR"
] = (
    out[
        "follow_NBR"
    ]
    -
    out[
        "baseline_NBR"
    ]
)


out[
    "delta_DW_vegprob"
] = (
    out[
        "follow_DW_vegprob"
    ]
    -
    out[
        "baseline_DW_vegprob"
    ]
)


out[
    "delta_DW_water"
] = (
    out[
        "follow_DW_water"
    ]
    -
    out[
        "baseline_DW_water"
    ]
)


# ============================================================
# SUFFICIENT OBSERVATION SUPPORT
# ============================================================

out[
    "sufficient_optical_support"
] = (
    out[
        "baseline_S2_count"
    ].ge(
        MIN_VALID_OBSERVATIONS
    )
    &
    out[
        "follow_S2_count"
    ].ge(
        MIN_VALID_OBSERVATIONS
    )
    &
    out[
        "baseline_DW_count"
    ].ge(
        MIN_VALID_OBSERVATIONS
    )
    &
    out[
        "follow_DW_count"
    ].ge(
        MIN_VALID_OBSERVATIONS
    )
)


# ============================================================
# FROZEN CHANGE FLAGS
# ============================================================

out[
    "flag_NDVI_change"
] = (
    out[
        "delta_NDVI"
    ].abs()
    >=
    NDVI_CHANGE_THRESHOLD
)


out[
    "flag_NBR_change"
] = (
    out[
        "delta_NBR"
    ].abs()
    >=
    NBR_CHANGE_THRESHOLD
)


out[
    "flag_DW_vegprob_change"
] = (
    out[
        "delta_DW_vegprob"
    ].abs()
    >=
    VEG_PROB_CHANGE_THRESHOLD
)


out[
    "flag_DW_water_change"
] = (
    out[
        "delta_DW_water"
    ].abs()
    >=
    WATER_PROB_CHANGE_THRESHOLD
)


flag_columns = [
    "flag_NDVI_change",
    "flag_NBR_change",
    "flag_DW_vegprob_change",
    "flag_DW_water_change",
]


out[
    "n_change_indicators"
] = (
    out[
        flag_columns
    ]
    .sum(
        axis=1
    )
    .astype(int)
)


# ============================================================
# STABILITY CLASS
# ============================================================

def classify(
    row
):

    if not row[
        "sufficient_optical_support"
    ]:

        return (
            "INSUFFICIENT_DATA"
        )

    n = int(
        row[
            "n_change_indicators"
        ]
    )

    if n == 0:

        return (
            "STABLE_SCREEN"
        )

    if n == 1:

        return (
            "SINGLE_INDICATOR_CHANGE"
        )

    return (
        "MULTI_INDICATOR_POTENTIAL_CHANGE"
    )


out[
    "stability_class"
] = out.apply(
    classify,
    axis=1
)


# ============================================================
# DIRECTIONAL DESCRIPTION
# ============================================================

def direction(
    value,
    threshold
):

    if pd.isna(
        value
    ):

        return (
            "NA"
        )

    if value <= -threshold:

        return (
            "DECREASE"
        )

    if value >= threshold:

        return (
            "INCREASE"
        )

    return (
        "WITHIN_THRESHOLD"
    )


out[
    "NDVI_direction"
] = [
    direction(
        x,
        NDVI_CHANGE_THRESHOLD
    )
    for x
    in out[
        "delta_NDVI"
    ]
]


out[
    "NBR_direction"
] = [
    direction(
        x,
        NBR_CHANGE_THRESHOLD
    )
    for x
    in out[
        "delta_NBR"
    ]
]


out[
    "DW_vegprob_direction"
] = [
    direction(
        x,
        VEG_PROB_CHANGE_THRESHOLD
    )
    for x
    in out[
        "delta_DW_vegprob"
    ]
]


out[
    "DW_water_direction"
] = [
    direction(
        x,
        WATER_PROB_CHANGE_THRESHOLD
    )
    for x
    in out[
        "delta_DW_water"
    ]
]


# ============================================================
# SUMMARIES
# ============================================================

summary = (
    out[
        "stability_class"
    ]
    .value_counts(
        dropna=False
    )
    .rename_axis(
        "stability_class"
    )
    .reset_index(
        name="n"
    )
)


summary[
    "percent"
] = (
    summary[
        "n"
    ]
    / len(out)
    * 100.0
)


pass_summary = (
    out.groupby(
        [
            "source_file",
            "acquisition_date",
            "stability_class",
        ],
        dropna=False,
    )
    .size()
    .reset_index(
        name="n"
    )
)


pass_totals = (
    out.groupby(
        "source_file"
    )
    .size()
    .rename(
        "pass_total"
    )
)


pass_summary = (
    pass_summary
    .merge(
        pass_totals,
        on="source_file",
        how="left",
    )
)


pass_summary[
    "percent_within_pass"
] = (
    pass_summary[
        "n"
    ]
    /
    pass_summary[
        "pass_total"
    ]
    * 100.0
)


# ============================================================
# SAVE
# ============================================================

out.to_csv(
    POINT_FILE,
    index=False
)


summary.to_csv(
    SUMMARY_FILE,
    index=False
)


pass_summary.to_csv(
    PASS_FILE,
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
        "C3 STEP 3B - "
        "SENTINEL-2 / DYNAMIC WORLD "
        "DISTURBANCE-STABILITY SCREEN\n"
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
        "AGBD and model performance were not used "
        "to define stability classes.\n"
    )

    f.write(
        "C1 and C2 remain frozen.\n"
    )

    f.write(
        "C3 temporal-gap strata and previous "
        "C3 results remain frozen.\n\n"
    )


    f.write(
        "DESIGN\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    f.write(
        "Each GEDI observation was screened using a "
        "15-m-radius footprint buffer.\n"
    )

    f.write(
        "Baseline optical conditions were represented "
        "by the trailing 365 days ending at the GEDI "
        "acquisition date of its source-file pass.\n"
    )

    f.write(
        "Follow-up conditions were represented by the "
        "trailing 365 days ending at the Sep-17-2026 "
        "primary NISAR acquisition.\n"
    )

    f.write(
        "Sentinel-2 SR Harmonized supplied median "
        "NDVI and NBR.\n"
    )

    f.write(
        "Dynamic World supplied mean tree, flooded "
        "vegetation, and water probabilities.\n"
    )

    f.write(
        "tree + flooded_vegetation probability was "
        "used as a combined mangrove-relevant "
        "vegetation probability.\n\n"
    )


    f.write(
        "FROZEN SCREENING THRESHOLDS\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    f.write(
        f"|delta NDVI| >= "
        f"{NDVI_CHANGE_THRESHOLD}\n"
    )

    f.write(
        f"|delta NBR| >= "
        f"{NBR_CHANGE_THRESHOLD}\n"
    )

    f.write(
        "|delta Dynamic World "
        "(trees + flooded vegetation)| >= "
        f"{VEG_PROB_CHANGE_THRESHOLD}\n"
    )

    f.write(
        "|delta Dynamic World water| >= "
        f"{WATER_PROB_CHANGE_THRESHOLD}\n"
    )

    f.write(
        f"Minimum valid optical observations per "
        f"window = {MIN_VALID_OBSERVATIONS}\n\n"
    )

    f.write(
        "These are conservative study-specific "
        "screening cut-offs, not independently "
        "validated ecological disturbance thresholds.\n\n"
    )


    f.write(
        "CLASSIFICATION RULE\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    f.write(
        "0 indicators: STABLE_SCREEN\n"
    )

    f.write(
        "1 indicator: SINGLE_INDICATOR_CHANGE\n"
    )

    f.write(
        "2-4 indicators: "
        "MULTI_INDICATOR_POTENTIAL_CHANGE\n"
    )

    f.write(
        "Insufficient observation support: "
        "INSUFFICIENT_DATA\n\n"
    )


    f.write(
        "OVERALL RESULT\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    f.write(
        summary.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )


    f.write(
        "RESULT BY GEDI ACQUISITION PASS\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    f.write(
        pass_summary.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )


    f.write(
        "COLLECTION IMAGE COUNTS BY PASS\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    f.write(
        pd.DataFrame(
            pass_inventory_rows
        ).to_string(
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
        "This is a conservative optical/LULC "
        "stability screen, not a validated "
        "disturbance map.\n"
    )

    f.write(
        "Differences may reflect structural change, "
        "hydrology, phenology, atmospheric residuals, "
        "classification uncertainty, or mixed "
        "mangrove-water pixels.\n"
    )

    f.write(
        "Annual composites and multiple indicators "
        "reduce but do not eliminate these confounders.\n"
    )

    f.write(
        "The next C3 analysis will evaluate existing "
        "frozen OOF model performance in the "
        "STABLE_SCREEN subset without refitting or "
        "reselecting any model.\n"
    )


# ============================================================
# CONSOLE
# ============================================================

print("")
print(
    "=" * 84
)

print(
    "C3 STEP 3B COMPLETE"
)

print(
    "DISTURBANCE / STABILITY SCREEN"
)

print(
    "=" * 84
)

print("")
print(
    "Overall stability classes:"
)

print(
    summary.to_string(
        index=False
    )
)

print("")
print(
    "By GEDI pass:"
)

print(
    pass_summary.to_string(
        index=False
    )
)

print("")
print(
    "Output rows:",
    len(out)
)

print(
    "Unique shots:",
    out[
        "shot_number"
    ].nunique()
)

print("")
print(
    "Report:"
)

print(
    REPORT
)

print("")
print(
    "NO AGBD MODEL WAS FITTED."
)

print(
    "C1 AND C2 REMAIN FROZEN."
)

print(
    "=" * 84
)

