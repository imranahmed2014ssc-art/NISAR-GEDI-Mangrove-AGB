from pathlib import Path
import numpy as np
import pandas as pd
import rasterio


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

OLD_SUPPORT = (
    ROOT
    / "06_Results"
    / "Reviewer_Validation"
    / "C6_Support_Mixing"
    / "C6_01_Support_Metrics"
    / "C6_01_POINT_SUPPORT_METRICS.csv"
)

GMW = (
    ROOT
    / "01_AOI"
    / "Mangrove_Mask"
    / "Processed"
    / "GMW_2025_NISAR_grid_mask.tif"
)

OUT = (
    ROOT
    / "06_Results"
    / "Reviewer_Validation"
    / "C6_Support_Mixing"
    / "C6_01B_Corrected_GMW"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

POINT_FILE = (
    OUT
    / "C6_01B_CORRECTED_SUPPORT.csv"
)

THRESHOLD_FILE = (
    OUT
    / "C6_01B_THRESHOLD_COUNTS.csv"
)

COMPARISON_FILE = (
    OUT
    / "C6_01B_MASTER_COMPARISON.csv"
)

REPORT = (
    OUT
    / "C6_01B_CORRECTED_GMW_REPORT.txt"
)


# ============================================================
# FROZEN DESIGN
# ============================================================

SUPPORTS = {
    "3x3": 1,
    "5x5": 2,
    "9x9": 4,
}

WIDTH_M = {
    "3x3": 30,
    "5x5": 50,
    "9x9": 90,
}

PRIMARY_THRESHOLD = 0.95

SENSITIVITY_THRESHOLDS = [
    1.00,
    0.95,
    0.90,
    0.75,
]


# ============================================================
# LOAD
# ============================================================

master = pd.read_csv(
    MASTER
)

old = pd.read_csv(
    OLD_SUPPORT
)


if len(master) != 1641:
    raise RuntimeError(
        f"Expected 1641 master rows; found {len(master)}."
    )


if len(old) != 1641:
    raise RuntimeError(
        f"Expected 1641 prior support rows; found {len(old)}."
    )


if master["shot_number"].nunique() != 1641:
    raise RuntimeError(
        "Master shot_number is not unique."
    )


# Preserve already-validated GCOV metrics.
keep_old = [
    "shot_number",
    "h5_center_offset_m",
]

for support in SUPPORTS:

    keep_old += [
        f"gcov_valid_fraction_{support}",
        f"exception_clean_fraction_{support}",
        f"positive_looks_fraction_{support}",
        f"numberOfLooks_mean_{support}",
        f"numberOfLooks_min_{support}",
        f"numberOfLooks_max_{support}",
        f"numberOfLooks_cv_{support}",
        f"qc_clean_{support}",
    ]


old_subset = old[
    keep_old
].copy()


df = master.merge(
    old_subset,
    on="shot_number",
    how="inner",
    validate="one_to_one",
)


if len(df) != 1641:
    raise RuntimeError(
        "Merge did not preserve 1641 rows."
    )


# ============================================================
# HELPER
# ============================================================

def window_bounds(
    center,
    half_width,
    maximum,
):

    start = (
        center
        -
        half_width
    )

    stop = (
        center
        +
        half_width
        +
        1
    )

    if (
        start < 0
        or
        stop > maximum
    ):

        return None

    return (
        start,
        stop,
    )


# ============================================================
# RAW BINARY RASTER
#
# IMPORTANT:
# masked=False is deliberate.
# Raw 1 = mangrove.
# Raw 0 = background/non-mangrove for support occupancy.
# ============================================================

with rasterio.open(
    GMW
) as src:

    raw = src.read(
        1,
        masked=False,
    )

    crs = (
        src.crs.to_epsg()
        if src.crs is not None
        else None
    )

    res_x = float(
        abs(
            src.transform.a
        )
    )

    res_y = float(
        abs(
            src.transform.e
        )
    )

    nodata = src.nodata


    unique_values = np.unique(
        raw
    )


    if not set(
        unique_values.tolist()
    ).issubset(
        {
            0,
            1,
        }
    ):

        raise RuntimeError(
            f"Expected raw binary 0/1 raster; "
            f"found {unique_values}."
        )


    if crs != 32645:

        raise RuntimeError(
            f"Unexpected CRS EPSG:{crs}"
        )


    if not (
        np.isclose(
            res_x,
            10.0
        )
        and
        np.isclose(
            res_y,
            10.0
        )
    ):

        raise RuntimeError(
            f"Unexpected raster resolution: "
            f"{res_x}, {res_y}"
        )


    rows = []


    for _, row in df.iterrows():

        x = float(
            row[
                "utm_x"
            ]
        )

        y = float(
            row[
                "utm_y"
            ]
        )


        raster_row, raster_col = (
            src.index(
                x,
                y,
            )
        )


        center_value = int(
            raw[
                raster_row,
                raster_col
            ]
        )


        result = {
            "shot_number":
                row[
                    "shot_number"
                ],

            "utm_x":
                x,

            "utm_y":
                y,

            "center_mangrove_master":
                row[
                    "center_mangrove"
                ],

            "raw_GMW_center_value":
                center_value,
        }


        # --------------------------------------------
        # Preserve validated GCOV metrics
        # --------------------------------------------

        for col in keep_old:

            if col == "shot_number":
                continue

            result[col] = row[col]


        # --------------------------------------------
        # Correct support fractions
        # --------------------------------------------

        for support, half_width in (
            SUPPORTS.items()
        ):

            rb = window_bounds(
                raster_row,
                half_width,
                src.height,
            )

            cb = window_bounds(
                raster_col,
                half_width,
                src.width,
            )


            if (
                rb is None
                or
                cb is None
            ):

                raise RuntimeError(
                    f"Shot {row['shot_number']}: "
                    f"{support} outside raster bounds."
                )


            rs, re = rb
            cs, ce = cb


            window = raw[
                rs:re,
                cs:ce
            ]


            expected = (
                2 * half_width + 1
            ) ** 2


            if window.size != expected:

                raise RuntimeError(
                    f"{support}: unexpected "
                    f"window size {window.size}."
                )


            if not set(
                np.unique(
                    window
                ).tolist()
            ).issubset(
                {
                    0,
                    1,
                }
            ):

                raise RuntimeError(
                    "Non-binary value encountered "
                    "inside support window."
                )


            mangrove_count = int(
                np.sum(
                    window
                    ==
                    1
                )
            )


            background_count = int(
                np.sum(
                    window
                    ==
                    0
                )
            )


            fraction = float(
                mangrove_count
                /
                expected
            )


            result[
                f"mangrove_count_{support}"
            ] = mangrove_count

            result[
                f"background_count_{support}"
            ] = background_count

            result[
                f"corrected_mangrove_fraction_{support}"
            ] = fraction


        # --------------------------------------------
        # Primary frozen class
        # --------------------------------------------

        m90 = result[
            "corrected_mangrove_fraction_9x9"
        ]


        result[
            "primary_90m_support_class"
        ] = (
            "INTERIOR_95"
            if m90
            >=
            PRIMARY_THRESHOLD
            else
            "EDGE_MIXED"
        )


        for threshold in (
            SENSITIVITY_THRESHOLDS
        ):

            suffix = (
                str(
                    threshold
                )
                .replace(
                    ".",
                    "p"
                )
            )


            result[
                f"interior90_ge_{suffix}"
            ] = bool(
                m90
                >=
                threshold
            )


        rows.append(
            result
        )


out = pd.DataFrame(
    rows
)


# ============================================================
# QC
# ============================================================

if len(out) != 1641:

    raise RuntimeError(
        f"Expected 1641 rows; found {len(out)}."
    )


if (
    out[
        "shot_number"
    ].nunique()
    !=
    1641
):

    raise RuntimeError(
        "Output shot_number not unique."
    )


center_match = (
    pd.to_numeric(
        out[
            "center_mangrove_master"
        ],
        errors="coerce",
    )
    ==
    pd.to_numeric(
        out[
            "raw_GMW_center_value"
        ],
        errors="coerce",
    )
)


center_match_n = int(
    center_match.sum()
)


if center_match_n != 1641:

    raise RuntimeError(
        f"Raw GMW center disagrees with "
        f"master center_mangrove for "
        f"{1641-center_match_n} shots."
    )


# ============================================================
# THRESHOLD COUNTS
# ============================================================

m90 = out[
    "corrected_mangrove_fraction_9x9"
]


threshold_rows = []


for threshold in (
    SENSITIVITY_THRESHOLDS
):

    interior = (
        m90
        >=
        threshold
    )


    threshold_rows.append({
        "threshold":
            threshold,

        "interior_n":
            int(
                interior.sum()
            ),

        "interior_percent":
            float(
                interior.mean()
                *
                100.0
            ),

        "edge_mixed_n":
            int(
                (
                    ~interior
                ).sum()
            ),

        "edge_mixed_percent":
            float(
                (
                    ~interior
                ).mean()
                *
                100.0
            ),
    })


threshold_df = pd.DataFrame(
    threshold_rows
)


# ============================================================
# COMPARE EXISTING MASTER FRACTION AGAINST
# 3x3 / 5x5 / 9x9 CORRECTED SUPPORTS
#
# This is diagnostic only.
# ============================================================

comparison_rows = []


if "mangrove_fraction" in df.columns:

    master_fraction = pd.to_numeric(
        df[
            "mangrove_fraction"
        ],
        errors="coerce",
    )


    for support in SUPPORTS:

        corrected = pd.to_numeric(
            out[
                f"corrected_mangrove_fraction_{support}"
            ],
            errors="coerce",
        )


        valid = (
            master_fraction.notna()
            &
            corrected.notna()
        )


        difference = (
            master_fraction[
                valid
            ].to_numpy(float)
            -
            corrected[
                valid
            ].to_numpy(float)
        )


        x = master_fraction[
            valid
        ].to_numpy(float)

        y = corrected[
            valid
        ].to_numpy(float)


        correlation = (
            float(
                np.corrcoef(
                    x,
                    y
                )[0, 1]
            )
            if (
                np.std(x) > 0
                and
                np.std(y) > 0
            )
            else np.nan
        )


        comparison_rows.append({
            "support":
                support,

            "n":
                int(
                    valid.sum()
                ),

            "mean_abs_difference":
                float(
                    np.mean(
                        np.abs(
                            difference
                        )
                    )
                ),

            "max_abs_difference":
                float(
                    np.max(
                        np.abs(
                            difference
                        )
                    )
                ),

            "pearson_correlation":
                correlation,
        })


comparison_df = pd.DataFrame(
    comparison_rows
)


# ============================================================
# DISTRIBUTION SUMMARY
# ============================================================

distribution_rows = []


for support in SUPPORTS:

    x = out[
        f"corrected_mangrove_fraction_{support}"
    ]


    distribution_rows.append({
        "support":
            support,

        "nominal_width_m":
            WIDTH_M[
                support
            ],

        "minimum":
            float(
                x.min()
            ),

        "q01":
            float(
                x.quantile(
                    0.01
                )
            ),

        "q05":
            float(
                x.quantile(
                    0.05
                )
            ),

        "q25":
            float(
                x.quantile(
                    0.25
                )
            ),

        "median":
            float(
                x.median()
            ),

        "q75":
            float(
                x.quantile(
                    0.75
                )
            ),

        "q95":
            float(
                x.quantile(
                    0.95
                )
            ),

        "maximum":
            float(
                x.max()
            ),

        "n_fraction_1":
            int(
                np.isclose(
                    x,
                    1.0
                ).sum()
            ),
    })


distribution_df = pd.DataFrame(
    distribution_rows
)


# ============================================================
# SAVE
# ============================================================

out.to_csv(
    POINT_FILE,
    index=False,
)

threshold_df.to_csv(
    THRESHOLD_FILE,
    index=False,
)

comparison_df.to_csv(
    COMPARISON_FILE,
    index=False,
)


# ============================================================
# REPORT
# ============================================================

with REPORT.open(
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "=" * 108
        + "\n"
    )

    f.write(
        "C6 STEP 1B - CORRECTED RAW-BINARY "
        "GMW SUPPORT OCCUPANCY\n"
    )

    f.write(
        "=" * 108
        + "\n\n"
    )


    f.write(
        "STATUS\n"
    )

    f.write(
        "-" * 108
        + "\n"
    )

    f.write(
        "NO MODEL WAS FITTED.\n"
    )

    f.write(
        "NO SUPPORT THRESHOLD WAS CHANGED.\n"
    )

    f.write(
        "This step supersedes ONLY the GMW fraction "
        "calculation from C6 Step 1.\n"
    )

    f.write(
        "The C6 Step 1 GCOV mask / exception / "
        "numberOfLooks results remain valid.\n"
    )

    f.write(
        "C1-C5 remain frozen.\n\n"
    )


    f.write(
        "WHY CORRECTION WAS REQUIRED\n"
    )

    f.write(
        "-" * 108
        + "\n"
    )

    f.write(
        f"GeoTIFF NoData tag = {nodata}.\n"
    )

    f.write(
        "Rasterio masked arrays therefore excluded "
        "raw zero/background cells from the original "
        "fraction denominator.\n"
    )

    f.write(
        "Corrected support fractions use all raw "
        "binary 0/1 cells within each complete "
        "support window.\n\n"
    )


    f.write(
        "GRID\n"
    )

    f.write(
        "-" * 108
        + "\n"
    )

    f.write(
        f"CRS: EPSG:{crs}\n"
    )

    f.write(
        f"Resolution: "
        f"{res_x:.3f} x "
        f"{res_y:.3f} m\n"
    )

    f.write(
        f"Raw raster values: "
        f"{unique_values.tolist()}\n\n"
    )


    f.write(
        "CENTER-PIXEL CONSISTENCY\n"
    )

    f.write(
        "-" * 108
        + "\n"
    )

    f.write(
        f"Raw GMW center matches master "
        f"center_mangrove: "
        f"{center_match_n}/1641\n\n"
    )


    f.write(
        "CORRECTED SUPPORT DISTRIBUTIONS\n"
    )

    f.write(
        "-" * 108
        + "\n"
    )

    f.write(
        distribution_df.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )


    f.write(
        "FROZEN 90-M THRESHOLD COUNTS\n"
    )

    f.write(
        "-" * 108
        + "\n"
    )

    f.write(
        threshold_df.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )


    f.write(
        "PRIMARY 95% CLASS\n"
    )

    f.write(
        "-" * 108
        + "\n"
    )

    class_counts = (
        out[
            "primary_90m_support_class"
        ]
        .value_counts()
        .rename_axis(
            "class"
        )
        .reset_index(
            name="n"
        )
    )

    class_counts[
        "percent"
    ] = (
        class_counts[
            "n"
        ]
        /
        len(out)
        *
        100.0
    )


    f.write(
        class_counts.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )


    f.write(
        "MASTER MANGROVE_FRACTION COMPARISON\n"
    )

    f.write(
        "-" * 108
        + "\n"
    )

    if len(
        comparison_df
    ):

        f.write(
            comparison_df.to_string(
                index=False
            )
        )

    else:

        f.write(
            "Master comparison unavailable."
        )

    f.write(
        "\n\n"
    )


    f.write(
        "GCOV QC STATUS RETAINED FROM STEP 1\n"
    )

    f.write(
        "-" * 108
        + "\n"
    )

    for support in SUPPORTS:

        clean = out[
            f"qc_clean_{support}"
        ]

        f.write(
            f"{support}: "
            f"{int(clean.sum())}/{len(clean)} "
            f"QC-clean "
            f"({clean.mean()*100:.3f}%)\n"
        )


    f.write(
        "\n"
    )


    f.write(
        "INTERPRETATION BOUNDARY\n"
    )

    f.write(
        "-" * 108
        + "\n"
    )

    f.write(
        "These fractions characterize binary GMW "
        "occupancy within the nominal NISAR predictor "
        "support windows.\n"
    )

    f.write(
        "They do not imply that non-mangrove cells are "
        "necessarily open water; background may include "
        "water or other non-GMW cover.\n"
    )

    f.write(
        "No causal relationship with biomass-model "
        "error is inferred in this step.\n"
    )


# ============================================================
# CONSOLE
# ============================================================

print("")
print("=" * 88)
print("C6 STEP 1B COMPLETE")
print("CORRECTED RAW-BINARY GMW SUPPORT")
print("=" * 88)

print("")
print(
    "Center matches:",
    center_match_n,
    "/ 1641"
)

print("")
print("Corrected support distributions:")
print(
    distribution_df.to_string(
        index=False
    )
)

print("")
print("90-m threshold counts:")
print(
    threshold_df.to_string(
        index=False
    )
)

print("")
print("Primary support class:")
print(
    out[
        "primary_90m_support_class"
    ]
    .value_counts()
)

print("")
print("Master comparison:")
print(
    comparison_df.to_string(
        index=False
    )
)

print("")
print("Report:")
print(REPORT)

print("")
print("NO MODEL WAS FITTED.")
print("NO THRESHOLD WAS CHANGED.")
print("C1-C5 REMAIN FROZEN.")
print("=" * 88)
