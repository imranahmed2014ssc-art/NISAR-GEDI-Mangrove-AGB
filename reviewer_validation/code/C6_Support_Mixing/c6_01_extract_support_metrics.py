from pathlib import Path
import hashlib
import warnings

import h5py
import numpy as np
import pandas as pd
import rasterio

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

H5_FILE = (
    ROOT
    / "02_NISAR_GCOV"
    / "Original"
    / "NISAR_L2_PR_GCOV_030_170_A_013_4005_DHDH_A_20260917T232917_20260917T232952_P05023_F_F_J_001.h5"
)

GMW_FILE = (
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
    / "C6_01_Support_Metrics"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

POINT_FILE = (
    OUT
    / "C6_01_POINT_SUPPORT_METRICS.csv"
)

SUMMARY_FILE = (
    OUT
    / "C6_01_SUPPORT_SUMMARY.csv"
)

THRESHOLD_FILE = (
    OUT
    / "C6_01_INTERIOR_THRESHOLD_COUNTS.csv"
)

REPORT = (
    OUT
    / "C6_01_SUPPORT_METRICS_REPORT.txt"
)


# ============================================================
# FROZEN DESIGN
# ============================================================

SUPPORTS = {
    "3x3": 1,
    "5x5": 2,
    "9x9": 4,
}

SUPPORT_WIDTH_M = {
    "3x3": 30,
    "5x5": 50,
    "9x9": 90,
}

PRIMARY_INTERIOR_THRESHOLD = 0.95

SENSITIVITY_THRESHOLDS = [
    1.00,
    0.95,
    0.90,
    0.75,
]


# ============================================================
# LOAD MASTER
# ============================================================

df = pd.read_csv(
    MASTER
)

if len(df) != 1641:
    raise RuntimeError(
        f"Expected 1641 rows; found {len(df)}."
    )

if df["shot_number"].nunique() != 1641:
    raise RuntimeError(
        "shot_number is not unique."
    )

required = [
    "shot_number",
    "utm_x",
    "utm_y",
    "center_mangrove",
]

missing = [
    c
    for c in required
    if c not in df.columns
]

if missing:
    raise RuntimeError(
        f"Missing master columns: {missing}"
    )


# ============================================================
# MASTER HASH
# ============================================================

sha = hashlib.sha256()

with MASTER.open(
    "rb"
) as f:

    while True:

        chunk = f.read(
            1024 * 1024
        )

        if not chunk:
            break

        sha.update(
            chunk
        )

master_hash = sha.hexdigest()


# ============================================================
# HELPERS
# ============================================================

def nearest_index(
    array,
    value,
):

    array = np.asarray(
        array,
        dtype=float,
    )

    if array[0] <= array[-1]:

        idx = np.searchsorted(
            array,
            value,
        )

        candidates = [
            max(
                0,
                min(
                    len(array) - 1,
                    idx - 1,
                ),
            ),
            max(
                0,
                min(
                    len(array) - 1,
                    idx,
                ),
            ),
        ]

    else:

        reverse = array[::-1]

        rev_idx = np.searchsorted(
            reverse,
            value,
        )

        rev_candidates = [
            max(
                0,
                min(
                    len(reverse) - 1,
                    rev_idx - 1,
                ),
            ),
            max(
                0,
                min(
                    len(reverse) - 1,
                    rev_idx,
                ),
            ),
        ]

        candidates = [
            len(array) - 1 - x
            for x in rev_candidates
        ]


    return min(
        candidates,
        key=lambda i:
            abs(
                array[i]
                -
                value
            ),
    )


def slice_bounds(
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


def safe_fraction(
    numerator,
    denominator,
):

    if denominator == 0:
        return np.nan

    return float(
        numerator
        /
        denominator
    )


# ============================================================
# OPEN NISAR GCOV
# ============================================================

H5_BASE = (
    "science/LSAR/GCOV/grids/frequencyA"
)


with h5py.File(
    H5_FILE,
    "r",
) as h5:

    x_coords = np.asarray(
        h5[
            f"{H5_BASE}/xCoordinates"
        ][:],
        dtype=float,
    )

    y_coords = np.asarray(
        h5[
            f"{H5_BASE}/yCoordinates"
        ][:],
        dtype=float,
    )

    projection = int(
        h5[
            f"{H5_BASE}/projection"
        ][()]
    )

    x_spacing = float(
        h5[
            f"{H5_BASE}/xCoordinateSpacing"
        ][()]
    )

    y_spacing = float(
        h5[
            f"{H5_BASE}/yCoordinateSpacing"
        ][()]
    )

    mask_ds = h5[
        f"{H5_BASE}/mask"
    ]

    looks_ds = h5[
        f"{H5_BASE}/numberOfLooks"
    ]

    exception_ds = h5[
        f"{H5_BASE}/inputDataExceptionMask"
    ]


    # ========================================================
    # OPEN ALIGNED GMW RASTER
    # ========================================================

    with rasterio.open(
        GMW_FILE
    ) as gmw:

        gmw_crs = (
            gmw.crs.to_epsg()
            if gmw.crs is not None
            else None
        )

        gmw_pixel_x = float(
            abs(
                gmw.transform.a
            )
        )

        gmw_pixel_y = float(
            abs(
                gmw.transform.e
            )
        )


        # ----------------------------------------------------
        # Raster-value inventory
        # ----------------------------------------------------

        full_masked = gmw.read(
            1,
            masked=True,
        )

        valid_values = (
            full_masked
            .compressed()
        )

        unique_gmw_values = np.unique(
            valid_values
        )


        if not set(
            unique_gmw_values.tolist()
        ).issubset(
            {
                0,
                1,
            }
        ):

            raise RuntimeError(
                "Expected binary GMW raster values "
                f"0/1; found {unique_gmw_values[:20]}."
            )


        # ====================================================
        # GRID QC
        # ====================================================

        if projection != 32645:

            raise RuntimeError(
                f"Unexpected NISAR projection: "
                f"EPSG:{projection}"
            )


        if gmw_crs != 32645:

            raise RuntimeError(
                f"Unexpected GMW projection: "
                f"EPSG:{gmw_crs}"
            )


        if not np.isclose(
            abs(x_spacing),
            10.0,
            atol=1e-6,
        ):

            raise RuntimeError(
                f"Unexpected NISAR X spacing: "
                f"{x_spacing}"
            )


        if not np.isclose(
            abs(y_spacing),
            10.0,
            atol=1e-6,
        ):

            raise RuntimeError(
                f"Unexpected NISAR Y spacing: "
                f"{y_spacing}"
            )


        if not np.isclose(
            gmw_pixel_x,
            10.0,
            atol=1e-6,
        ):

            raise RuntimeError(
                f"Unexpected GMW X resolution: "
                f"{gmw_pixel_x}"
            )


        if not np.isclose(
            gmw_pixel_y,
            10.0,
            atol=1e-6,
        ):

            raise RuntimeError(
                f"Unexpected GMW Y resolution: "
                f"{gmw_pixel_y}"
            )


        # ====================================================
        # EXTRACT EACH GEDI POINT
        # ====================================================

        rows = []


        for index, row in df.iterrows():

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


            h5_col = nearest_index(
                x_coords,
                x,
            )

            h5_row = nearest_index(
                y_coords,
                y,
            )


            h5_center_x = float(
                x_coords[
                    h5_col
                ]
            )

            h5_center_y = float(
                y_coords[
                    h5_row
                ]
            )


            h5_offset_m = float(
                np.hypot(
                    h5_center_x - x,
                    h5_center_y - y,
                )
            )


            gmw_row, gmw_col = gmw.index(
                x,
                y,
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

                "h5_row":
                    h5_row,

                "h5_col":
                    h5_col,

                "h5_center_offset_m":
                    h5_offset_m,

                "gmw_row":
                    int(
                        gmw_row
                    ),

                "gmw_col":
                    int(
                        gmw_col
                    ),
            }


            # ------------------------------------------------
            # Master context values retained for comparison
            # ------------------------------------------------

            for col in [
                "mangrove_fraction",
                "valid_NISAR_fraction",
                "expanded_all_valid_fraction",
                "HH_SD_dB_3x3_valid_fraction",
                "HH_MeanLin_dB_5x5_valid_fraction",
                "HH_MeanLin_dB_9x9_valid_fraction",
            ]:

                if col in df.columns:

                    result[
                        "master_"
                        +
                        col
                    ] = row[
                        col
                    ]


            # ------------------------------------------------
            # Support-specific metrics
            # ------------------------------------------------

            for support, half_width in SUPPORTS.items():

                h5_r = slice_bounds(
                    h5_row,
                    half_width,
                    len(
                        y_coords
                    ),
                )

                h5_c = slice_bounds(
                    h5_col,
                    half_width,
                    len(
                        x_coords
                    ),
                )


                if (
                    h5_r is None
                    or
                    h5_c is None
                ):

                    raise RuntimeError(
                        f"Shot {row['shot_number']}: "
                        f"{support} HDF5 window "
                        f"outside product bounds."
                    )


                rs, re = h5_r
                cs, ce = h5_c


                mask_window = np.asarray(
                    mask_ds[
                        rs:re,
                        cs:ce,
                    ]
                )


                exception_window = np.asarray(
                    exception_ds[
                        rs:re,
                        cs:ce,
                    ]
                )


                looks_window = np.asarray(
                    looks_ds[
                        rs:re,
                        cs:ce,
                    ],
                    dtype=float,
                )


                expected_pixels = int(
                    (
                        2 * half_width + 1
                    )
                    ** 2
                )


                if mask_window.size != expected_pixels:

                    raise RuntimeError(
                        f"Shot {row['shot_number']}: "
                        f"{support} unexpected "
                        f"HDF5 pixel count."
                    )


                gcov_valid = (
                    (mask_window != 0)
                    &
                    (mask_window != 255)
                )


                exception_clean = (
                    exception_window
                    ==
                    0
                )


                positive_looks = (
                    np.isfinite(
                        looks_window
                    )
                    &
                    (
                        looks_window
                        >
                        0
                    )
                )


                result[
                    f"gcov_valid_fraction_{support}"
                ] = float(
                    gcov_valid.mean()
                )


                result[
                    f"exception_clean_fraction_{support}"
                ] = float(
                    exception_clean.mean()
                )


                result[
                    f"positive_looks_fraction_{support}"
                ] = float(
                    positive_looks.mean()
                )


                usable_looks = looks_window[
                    gcov_valid
                    &
                    positive_looks
                ]


                result[
                    f"numberOfLooks_mean_{support}"
                ] = (
                    float(
                        np.mean(
                            usable_looks
                        )
                    )
                    if len(
                        usable_looks
                    )
                    else np.nan
                )


                result[
                    f"numberOfLooks_min_{support}"
                ] = (
                    float(
                        np.min(
                            usable_looks
                        )
                    )
                    if len(
                        usable_looks
                    )
                    else np.nan
                )


                result[
                    f"numberOfLooks_max_{support}"
                ] = (
                    float(
                        np.max(
                            usable_looks
                        )
                    )
                    if len(
                        usable_looks
                    )
                    else np.nan
                )


                result[
                    f"numberOfLooks_cv_{support}"
                ] = (
                    float(
                        np.std(
                            usable_looks,
                            ddof=1,
                        )
                        /
                        np.mean(
                            usable_looks
                        )
                    )
                    if (
                        len(
                            usable_looks
                        )
                        >
                        1
                        and
                        np.mean(
                            usable_looks
                        )
                        !=
                        0
                    )
                    else np.nan
                )


                result[
                    f"qc_clean_{support}"
                ] = bool(
                    np.isclose(
                        result[
                            f"gcov_valid_fraction_{support}"
                        ],
                        1.0,
                    )
                    and
                    np.isclose(
                        result[
                            f"exception_clean_fraction_{support}"
                        ],
                        1.0,
                    )
                    and
                    np.isclose(
                        result[
                            f"positive_looks_fraction_{support}"
                        ],
                        1.0,
                    )
                )


                # --------------------------------------------
                # GMW window from aligned 10-m raster
                # --------------------------------------------

                gmw_r = slice_bounds(
                    int(
                        gmw_row
                    ),
                    half_width,
                    gmw.height,
                )

                gmw_c = slice_bounds(
                    int(
                        gmw_col
                    ),
                    half_width,
                    gmw.width,
                )


                if (
                    gmw_r is None
                    or
                    gmw_c is None
                ):

                    raise RuntimeError(
                        f"Shot {row['shot_number']}: "
                        f"{support} GMW window "
                        f"outside raster bounds."
                    )


                grs, gre = gmw_r
                gcs, gce = gmw_c


                gmw_window = gmw.read(
                    1,
                    window=(
                        (
                            grs,
                            gre,
                        ),
                        (
                            gcs,
                            gce,
                        ),
                    ),
                    masked=True,
                )


                gmw_valid = (
                    ~np.ma.getmaskarray(
                        gmw_window
                    )
                )


                valid_count = int(
                    gmw_valid.sum()
                )


                if valid_count == 0:

                    mangrove_fraction = np.nan

                else:

                    values = np.asarray(
                        gmw_window
                    )

                    mangrove_fraction = safe_fraction(
                        int(
                            (
                                (
                                    values
                                    ==
                                    1
                                )
                                &
                                gmw_valid
                            ).sum()
                        ),
                        valid_count,
                    )


                result[
                    f"gmw_valid_fraction_{support}"
                ] = safe_fraction(
                    valid_count,
                    expected_pixels,
                )


                result[
                    f"gmw_mangrove_fraction_{support}"
                ] = (
                    mangrove_fraction
                )


            # ------------------------------------------------
            # Primary 90-m classification
            # ------------------------------------------------

            m90 = result[
                "gmw_mangrove_fraction_9x9"
            ]


            if pd.isna(
                m90
            ):

                result[
                    "primary_90m_support_class"
                ] = (
                    "INSUFFICIENT_GMW"
                )

            elif (
                m90
                >=
                PRIMARY_INTERIOR_THRESHOLD
            ):

                result[
                    "primary_90m_support_class"
                ] = (
                    "INTERIOR_95"
                )

            else:

                result[
                    "primary_90m_support_class"
                ] = (
                    "EDGE_MIXED"
                )


            # ------------------------------------------------
            # Threshold sensitivity flags
            # ------------------------------------------------

            for threshold in (
                SENSITIVITY_THRESHOLDS
            ):

                name = (
                    str(
                        threshold
                    )
                    .replace(
                        ".",
                        "p"
                    )
                )


                result[
                    f"interior90_ge_{name}"
                ] = bool(
                    pd.notna(
                        m90
                    )
                    and
                    m90
                    >=
                    threshold
                )


            rows.append(
                result
            )


support_df = pd.DataFrame(
    rows
)


# ============================================================
# OUTPUT QC
# ============================================================

if len(
    support_df
) != 1641:

    raise RuntimeError(
        f"Expected 1641 extracted rows; "
        f"found {len(support_df)}."
    )


if (
    support_df[
        "shot_number"
    ].nunique()
    !=
    1641
):

    raise RuntimeError(
        "Extracted shot_number is not unique."
    )


if (
    support_df[
        "h5_center_offset_m"
    ].max()
    >
    8.0
):

    raise RuntimeError(
        "Unexpectedly large GEDI-to-NISAR "
        "nearest-pixel offset."
    )


# ============================================================
# SUMMARY
# ============================================================

summary_rows = []


continuous_metrics = []


for support in SUPPORTS:

    continuous_metrics += [
        f"gmw_mangrove_fraction_{support}",
        f"gmw_valid_fraction_{support}",
        f"gcov_valid_fraction_{support}",
        f"exception_clean_fraction_{support}",
        f"positive_looks_fraction_{support}",
        f"numberOfLooks_mean_{support}",
        f"numberOfLooks_min_{support}",
        f"numberOfLooks_max_{support}",
    ]


for col in continuous_metrics:

    x = pd.to_numeric(
        support_df[
            col
        ],
        errors="coerce",
    )


    summary_rows.append({
        "variable":
            col,

        "n_valid":
            int(
                x.notna().sum()
            ),

        "n_missing":
            int(
                x.isna().sum()
            ),

        "min":
            float(
                x.min()
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

        "max":
            float(
                x.max()
            ),
    })


summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# THRESHOLD COUNTS
# ============================================================

threshold_rows = []


m90 = support_df[
    "gmw_mangrove_fraction_9x9"
]


for threshold in SENSITIVITY_THRESHOLDS:

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
# MASTER-COLUMN COMPARISON
# ============================================================

comparison_text = []


if (
    "master_mangrove_fraction"
    in
    support_df.columns
):

    x = pd.to_numeric(
        support_df[
            "master_mangrove_fraction"
        ],
        errors="coerce",
    )

    y = pd.to_numeric(
        support_df[
            "gmw_mangrove_fraction_9x9"
        ],
        errors="coerce",
    )

    valid = (
        x.notna()
        &
        y.notna()
    )


    if valid.any():

        comparison_text.append(
            "Master mangrove_fraction vs "
            "recomputed 9x9 GMW:"
        )

        comparison_text.append(
            f"  Mean absolute difference = "
            f"{np.mean(np.abs(x[valid]-y[valid])):.8f}"
        )

        comparison_text.append(
            f"  Maximum absolute difference = "
            f"{np.max(np.abs(x[valid]-y[valid])):.8f}"
        )

        comparison_text.append(
            f"  Pearson correlation = "
            f"{np.corrcoef(x[valid],y[valid])[0,1]:.8f}"
        )


# ============================================================
# SAVE
# ============================================================

support_df.to_csv(
    POINT_FILE,
    index=False,
)

summary_df.to_csv(
    SUMMARY_FILE,
    index=False,
)

threshold_df.to_csv(
    THRESHOLD_FILE,
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
        "C6 STEP 1 - FROZEN SUPPORT / "
        "EDGE-MIXING METRIC EXTRACTION\n"
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
        "NO AGBD MODEL WAS FITTED.\n"
    )

    f.write(
        "NO MODEL ERROR OR PERFORMANCE RESULT "
        "WAS USED TO DEFINE SUPPORT CLASSES.\n"
    )

    f.write(
        "C1-C5 remain frozen.\n\n"
    )


    f.write(
        "INPUT INTEGRITY\n"
    )

    f.write(
        "-" * 108
        + "\n"
    )

    f.write(
        f"Master rows: {len(df)}\n"
    )

    f.write(
        f"Unique shots: "
        f"{df['shot_number'].nunique()}\n"
    )

    f.write(
        f"Master SHA256: "
        f"{master_hash}\n\n"
    )


    f.write(
        "GRID GEOMETRY\n"
    )

    f.write(
        "-" * 108
        + "\n"
    )

    f.write(
        f"NISAR Frequency-A projection: "
        f"EPSG:{projection}\n"
    )

    f.write(
        f"NISAR grid spacing: "
        f"{abs(x_spacing):.3f} m x "
        f"{abs(y_spacing):.3f} m\n"
    )

    f.write(
        f"GMW raster projection: "
        f"EPSG:{gmw_crs}\n"
    )

    f.write(
        f"GMW raster resolution: "
        f"{gmw_pixel_x:.3f} m x "
        f"{gmw_pixel_y:.3f} m\n\n"
    )

    for support in SUPPORTS:

        f.write(
            f"{support} nominal support width = "
            f"{SUPPORT_WIDTH_M[support]} m\n"
        )

    f.write(
        "\n"
    )


    f.write(
        "FROZEN PRIMARY EDGE DEFINITION\n"
    )

    f.write(
        "-" * 108
        + "\n"
    )

    f.write(
        "Primary 90-m interior support: "
        "9x9 GMW mangrove fraction >= 0.95.\n"
    )

    f.write(
        "Primary edge/mixed support: "
        "9x9 GMW mangrove fraction < 0.95.\n"
    )

    f.write(
        "Sensitivity thresholds: "
        "1.00, 0.95, 0.90, 0.75.\n"
    )

    f.write(
        "These are support-screening thresholds, "
        "not ecological disturbance thresholds.\n\n"
    )


    f.write(
        "FROZEN GCOV QC-CLEAN DEFINITION\n"
    )

    f.write(
        "-" * 108
        + "\n"
    )

    f.write(
        "For each support window, QC_CLEAN requires:\n"
    )

    f.write(
        "1) GCOV mask valid fraction = 1.0;\n"
    )

    f.write(
        "2) inputDataExceptionMask clean fraction = 1.0;\n"
    )

    f.write(
        "3) finite positive numberOfLooks fraction = 1.0.\n\n"
    )


    f.write(
        "GMW VALUE INVENTORY\n"
    )

    f.write(
        "-" * 108
        + "\n"
    )

    f.write(
        f"Unique valid raster values: "
        f"{unique_gmw_values.tolist()}\n\n"
    )


    f.write(
        "SUPPORT METRIC DISTRIBUTIONS\n"
    )

    f.write(
        "-" * 108
        + "\n"
    )

    f.write(
        summary_df.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )


    f.write(
        "90-M INTERIOR THRESHOLD COUNTS\n"
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
        "PRIMARY 90-M SUPPORT CLASS\n"
    )

    f.write(
        "-" * 108
        + "\n"
    )

    f.write(
        support_df[
            "primary_90m_support_class"
        ]
        .value_counts(
            dropna=False
        )
        .rename_axis(
            "class"
        )
        .reset_index(
            name="n"
        )
        .to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )


    f.write(
        "QC-CLEAN COUNTS\n"
    )

    f.write(
        "-" * 108
        + "\n"
    )

    for support in SUPPORTS:

        clean = support_df[
            f"qc_clean_{support}"
        ]

        f.write(
            f"{support}: "
            f"{int(clean.sum())}/{len(clean)} "
            f"({clean.mean()*100:.3f}%)\n"
        )

    f.write(
        "\n"
    )


    f.write(
        "MASTER-COLUMN COMPARISON\n"
    )

    f.write(
        "-" * 108
        + "\n"
    )

    if comparison_text:

        for line in comparison_text:

            f.write(
                line
                +
                "\n"
            )

    else:

        f.write(
            "No master mangrove_fraction comparison "
            "available.\n"
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
        "This step characterizes predictor support, "
        "mangrove occupancy, and GCOV validity only.\n"
    )

    f.write(
        "It does not establish that edge mixing causes "
        "model error.\n"
    )

    f.write(
        "The next step will use already-frozen OOF "
        "predictions to test whether performance differs "
        "between interior and edge/mixed support without "
        "refitting or reselecting a model.\n"
    )


# ============================================================
# CONSOLE
# ============================================================

print("")
print("=" * 88)
print("C6 STEP 1 COMPLETE")
print("SUPPORT / EDGE-MIXING METRICS")
print("=" * 88)

print("")
print(
    "NISAR Frequency-A grid:",
    f"{abs(x_spacing):.1f} m x "
    f"{abs(y_spacing):.1f} m"
)

print(
    "GMW grid:",
    f"{gmw_pixel_x:.1f} m x "
    f"{gmw_pixel_y:.1f} m"
)

print("")
print("90-m threshold counts:")
print(
    threshold_df.to_string(
        index=False
    )
)

print("")
print("Primary 90-m support classes:")
print(
    support_df[
        "primary_90m_support_class"
    ]
    .value_counts(
        dropna=False
    )
)

print("")
print("QC-clean counts:")

for support in SUPPORTS:

    clean = support_df[
        f"qc_clean_{support}"
    ]

    print(
        support,
        int(clean.sum()),
        "/",
        len(clean),
        f"({clean.mean()*100:.3f}%)"
    )

print("")
print(
    "Maximum GEDI-to-NISAR center offset:",
    f"{support_df['h5_center_offset_m'].max():.4f} m"
)

print("")
print("Report:")
print(REPORT)

print("")
print("NO MODEL WAS FITTED.")
print("C1-C5 REMAIN FROZEN.")
print("=" * 88)
