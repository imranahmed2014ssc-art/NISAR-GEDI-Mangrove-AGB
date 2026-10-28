from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(r"D:\NISAR_Mangrove_Paper")

POINT_FILE = (
    ROOT
    / "06_Results"
    / "Reviewer_Validation"
    / "C3_Temporal_Mismatch"
    / "C3_03B_Stability"
    / "C3_03B_POINT_STABILITY_SCREEN.csv"
)

OUT = (
    ROOT
    / "06_Results"
    / "Reviewer_Validation"
    / "C3_Temporal_Mismatch"
    / "C3_03C_Stability_QC"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

DIST_FILE = (
    OUT
    / "C3_03C_CHANGE_DISTRIBUTIONS.csv"
)

SUPPORT_FILE = (
    OUT
    / "C3_03C_OPTICAL_SUPPORT.csv"
)

PASS_FILE = (
    OUT
    / "C3_03C_PASS_CHANGE_QC.csv"
)

TOP_FILE = (
    OUT
    / "C3_03C_TOP_CHANGE_POINTS.csv"
)

REPORT = (
    OUT
    / "C3_03C_STABILITY_QC_REPORT.txt"
)


# ============================================================
# FROZEN THRESHOLDS — DO NOT CHANGE
# ============================================================

THRESHOLDS = {
    "delta_NDVI":
        0.15,

    "delta_NBR":
        0.15,

    "delta_DW_vegprob":
        0.25,

    "delta_DW_water":
        0.25,
}

MIN_VALID_OBSERVATIONS = 3


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(
    POINT_FILE
)

if len(df) != 1641:

    raise RuntimeError(
        f"Expected 1641 rows; found {len(df)}."
    )

if (
    df["shot_number"].nunique()
    != 1641
):

    raise RuntimeError(
        "shot_number is not unique."
    )


required = [
    "shot_number",
    "source_file",
    "acquisition_date",
    "stability_class",
    "sufficient_optical_support",
    "n_change_indicators",
    "baseline_S2_count",
    "follow_S2_count",
    "baseline_DW_count",
    "follow_DW_count",
    "delta_NDVI",
    "delta_NBR",
    "delta_DW_vegprob",
    "delta_DW_water",
]

missing = [
    c
    for c in required
    if c not in df.columns
]

if missing:

    raise RuntimeError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# BOOL NORMALIZER
# ============================================================

def as_bool(series):

    if pd.api.types.is_bool_dtype(
        series
    ):
        return series.fillna(False)

    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": True,
            "false": False,
            "1": True,
            "0": False,
        })
        .fillna(False)
    )


support_bool = as_bool(
    df[
        "sufficient_optical_support"
    ]
)


# ============================================================
# CHANGE DISTRIBUTIONS
# ============================================================

distribution_rows = []


for variable, threshold in THRESHOLDS.items():

    x = pd.to_numeric(
        df[
            variable
        ],
        errors="coerce",
    )

    valid = x.dropna()

    abs_x = valid.abs()

    if len(valid) == 0:

        raise RuntimeError(
            f"{variable}: no valid values."
        )

    distribution_rows.append({
        "variable":
            variable,

        "threshold":
            threshold,

        "n_total":
            len(df),

        "n_valid":
            len(valid),

        "n_missing":
            int(
                x.isna().sum()
            ),

        "signed_min":
            float(
                valid.min()
            ),

        "signed_q025":
            float(
                valid.quantile(
                    0.025
                )
            ),

        "signed_median":
            float(
                valid.median()
            ),

        "signed_q975":
            float(
                valid.quantile(
                    0.975
                )
            ),

        "signed_max":
            float(
                valid.max()
            ),

        "abs_median":
            float(
                abs_x.median()
            ),

        "abs_q90":
            float(
                abs_x.quantile(
                    0.90
                )
            ),

        "abs_q95":
            float(
                abs_x.quantile(
                    0.95
                )
            ),

        "abs_q99":
            float(
                abs_x.quantile(
                    0.99
                )
            ),

        "abs_max":
            float(
                abs_x.max()
            ),

        "max_abs_fraction_of_threshold":
            float(
                abs_x.max()
                / threshold
            ),

        "n_at_or_above_threshold":
            int(
                (
                    abs_x >= threshold
                ).sum()
            ),

        "pct_at_or_above_threshold":
            float(
                (
                    abs_x >= threshold
                ).mean()
                * 100.0
            ),
    })


distribution_df = pd.DataFrame(
    distribution_rows
)

distribution_df.to_csv(
    DIST_FILE,
    index=False
)


# ============================================================
# OBSERVATION SUPPORT
# ============================================================

support_columns = [
    "baseline_S2_count",
    "follow_S2_count",
    "baseline_DW_count",
    "follow_DW_count",
]

support_rows = []


for col in support_columns:

    x = pd.to_numeric(
        df[
            col
        ],
        errors="coerce",
    )

    valid = x.dropna()

    support_rows.append({
        "variable":
            col,

        "n_valid":
            len(valid),

        "n_missing":
            int(
                x.isna().sum()
            ),

        "minimum":
            (
                float(
                    valid.min()
                )
                if len(valid)
                else np.nan
            ),

        "q05":
            (
                float(
                    valid.quantile(
                        0.05
                    )
                )
                if len(valid)
                else np.nan
            ),

        "median":
            (
                float(
                    valid.median()
                )
                if len(valid)
                else np.nan
            ),

        "q95":
            (
                float(
                    valid.quantile(
                        0.95
                    )
                )
                if len(valid)
                else np.nan
            ),

        "maximum":
            (
                float(
                    valid.max()
                )
                if len(valid)
                else np.nan
            ),

        "n_below_minimum_support":
            int(
                (
                    x
                    <
                    MIN_VALID_OBSERVATIONS
                )
                .fillna(True)
                .sum()
            ),
    })


support_df = pd.DataFrame(
    support_rows
)

support_df.to_csv(
    SUPPORT_FILE,
    index=False
)


# ============================================================
# RANGE / PHYSICAL SANITY CHECKS
# ============================================================

range_checks = []

for col in [
    "baseline_NDVI",
    "follow_NDVI",
    "baseline_NBR",
    "follow_NBR",
]:

    x = pd.to_numeric(
        df[
            col
        ],
        errors="coerce",
    )

    range_checks.append({
        "variable":
            col,

        "expected_range":
            "[-1, 1]",

        "n_missing":
            int(
                x.isna().sum()
            ),

        "n_outside_expected_range":
            int(
                (
                    (x < -1.0)
                    |
                    (x > 1.0)
                )
                .fillna(False)
                .sum()
            ),
    })


for col in [
    "baseline_DW_trees",
    "baseline_DW_flooded",
    "baseline_DW_water",
    "baseline_DW_vegprob",
    "follow_DW_trees",
    "follow_DW_flooded",
    "follow_DW_water",
    "follow_DW_vegprob",
]:

    x = pd.to_numeric(
        df[
            col
        ],
        errors="coerce",
    )

    range_checks.append({
        "variable":
            col,

        "expected_range":
            "[0, 1]",

        "n_missing":
            int(
                x.isna().sum()
            ),

        "n_outside_expected_range":
            int(
                (
                    (x < -1e-6)
                    |
                    (x > 1.000001)
                )
                .fillna(False)
                .sum()
            ),
    })


range_df = pd.DataFrame(
    range_checks
)


# ============================================================
# RECOMPUTE FLAGS FROM FROZEN THRESHOLDS
# ============================================================

recomputed_flags = pd.DataFrame(
    index=df.index
)


for variable, threshold in THRESHOLDS.items():

    recomputed_flags[
        variable
    ] = (
        pd.to_numeric(
            df[
                variable
            ],
            errors="coerce",
        )
        .abs()
        >= threshold
    )


recomputed_n = (
    recomputed_flags
    .sum(
        axis=1
    )
    .astype(int)
)


def recompute_class(
    support,
    n_flags
):

    if not support:

        return (
            "INSUFFICIENT_DATA"
        )

    if n_flags == 0:

        return (
            "STABLE_SCREEN"
        )

    if n_flags == 1:

        return (
            "SINGLE_INDICATOR_CHANGE"
        )

    return (
        "MULTI_INDICATOR_POTENTIAL_CHANGE"
    )


recomputed_class = [
    recompute_class(
        support,
        int(n_flags),
    )
    for support, n_flags
    in zip(
        support_bool,
        recomputed_n,
    )
]


class_mismatch = (
    pd.Series(
        recomputed_class,
        index=df.index,
    )
    !=
    df[
        "stability_class"
    ].astype(str)
)


indicator_count_mismatch = (
    recomputed_n
    !=
    pd.to_numeric(
        df[
            "n_change_indicators"
        ],
        errors="coerce",
    ).fillna(-999).astype(int)
)


# ============================================================
# PASS-LEVEL CHANGE QC
# ============================================================

pass_rows = []


for (
    source_file,
    acquisition_date
), g in df.groupby(
    [
        "source_file",
        "acquisition_date",
    ],
    dropna=False,
):

    row = {
        "source_file":
            source_file,

        "acquisition_date":
            acquisition_date,

        "n":
            len(g),

        "sufficient_support_n":
            int(
                as_bool(
                    g[
                        "sufficient_optical_support"
                    ]
                ).sum()
            ),

        "stable_n":
            int(
                (
                    g[
                        "stability_class"
                    ]
                    ==
                    "STABLE_SCREEN"
                ).sum()
            ),
    }

    for variable, threshold in THRESHOLDS.items():

        x = pd.to_numeric(
            g[
                variable
            ],
            errors="coerce",
        ).abs()

        row[
            "max_abs_"
            + variable
        ] = (
            float(
                x.max()
            )
            if x.notna().any()
            else np.nan
        )

        row[
            "max_fraction_threshold_"
            + variable
        ] = (
            float(
                x.max()
                / threshold
            )
            if x.notna().any()
            else np.nan
        )

    pass_rows.append(
        row
    )


pass_df = pd.DataFrame(
    pass_rows
)

pass_df.to_csv(
    PASS_FILE,
    index=False
)


# ============================================================
# TOP CHANGE POINTS
# ============================================================

top_parts = []


for variable, threshold in THRESHOLDS.items():

    temp = df[
        [
            "shot_number",
            "source_file",
            "acquisition_date",
            "stability_class",
            variable,
        ]
    ].copy()

    temp[
        "variable"
    ] = variable

    temp[
        "threshold"
    ] = threshold

    temp[
        "absolute_change"
    ] = pd.to_numeric(
        temp[
            variable
        ],
        errors="coerce",
    ).abs()

    temp[
        "fraction_of_threshold"
    ] = (
        temp[
            "absolute_change"
        ]
        / threshold
    )

    temp = (
        temp
        .sort_values(
            "absolute_change",
            ascending=False,
        )
        .head(20)
    )

    top_parts.append(
        temp
    )


top_df = pd.concat(
    top_parts,
    ignore_index=True,
)

top_df.to_csv(
    TOP_FILE,
    index=False
)


# ============================================================
# GLOBAL COLLECTION COUNT NOTE
# ============================================================

collection_count_columns = [
    "baseline_S2_collection_images",
    "baseline_DW_collection_images",
    "follow_S2_collection_images",
    "follow_DW_collection_images",
]

collection_count_max = {}

for col in collection_count_columns:

    if col in df.columns:

        x = pd.to_numeric(
            df[
                col
            ],
            errors="coerce",
        )

        collection_count_max[
            col
        ] = (
            float(
                x.max()
            )
            if x.notna().any()
            else np.nan
        )


global_count_warning = any(
    np.isfinite(v)
    and v > 100000
    for v in collection_count_max.values()
)


# ============================================================
# FINAL QC STATUS
# ============================================================

all_required_deltas_complete = bool(
    (
        distribution_df[
            "n_missing"
        ]
        == 0
    ).all()
)

all_support_complete = bool(
    (
        support_df[
            "n_missing"
        ]
        == 0
    ).all()
)

all_support_at_least_3 = bool(
    (
        support_df[
            "n_below_minimum_support"
        ]
        == 0
    ).all()
)

ranges_clean = bool(
    (
        range_df[
            "n_outside_expected_range"
        ]
        == 0
    ).all()
)

class_reproduction_exact = (
    int(
        class_mismatch.sum()
    )
    == 0
)

indicator_reproduction_exact = (
    int(
        indicator_count_mismatch.sum()
    )
    == 0
)

qc_pass = all([
    all_required_deltas_complete,
    all_support_complete,
    all_support_at_least_3,
    ranges_clean,
    class_reproduction_exact,
    indicator_reproduction_exact,
])


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
        "C3 STEP 3C - "
        "LOCAL QC OF DISTURBANCE / STABILITY SCREEN\n"
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
        "NO EARTH ENGINE REQUEST WAS MADE.\n"
    )

    f.write(
        "NO AGBD MODEL WAS FITTED.\n"
    )

    f.write(
        "NO SCREENING THRESHOLD WAS CHANGED.\n"
    )

    f.write(
        "This step audits the already-generated "
        "C3 Step 3B point table only.\n\n"
    )


    f.write(
        "INTEGRITY\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    f.write(
        f"Rows: {len(df)}\n"
    )

    f.write(
        f"Unique shots: "
        f"{df['shot_number'].nunique()}\n"
    )

    f.write(
        f"Class reproduction mismatches: "
        f"{int(class_mismatch.sum())}\n"
    )

    f.write(
        f"Indicator-count reproduction mismatches: "
        f"{int(indicator_count_mismatch.sum())}\n\n"
    )


    f.write(
        "CHANGE DISTRIBUTIONS\n"
    )

    f.write(
        "-" * 100
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
        "OPTICAL OBSERVATION SUPPORT\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    f.write(
        support_df.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )


    f.write(
        "PHYSICAL RANGE CHECKS\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    f.write(
        range_df.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )


    f.write(
        "PASS-LEVEL MAXIMUM CHANGES\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    f.write(
        pass_df.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )


    f.write(
        "COLLECTION-COUNT METADATA CHECK\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    for col, value in (
        collection_count_max.items()
    ):

        f.write(
            f"{col}: {value}\n"
        )

    f.write(
        "\n"
    )

    if global_count_warning:

        f.write(
            "WARNING: collection-level image counts "
            "are global/unfiltered collection counts "
            "and must NOT be reported as local "
            "study-area image availability.\n"
        )

        f.write(
            "Point-level baseline/follow *_count bands "
            "are the relevant optical-support quantities "
            "used for the stability classification.\n"
        )

    else:

        f.write(
            "No obviously global-scale collection-count "
            "values detected.\n"
        )

    f.write(
        "\n"
    )


    f.write(
        "OVERALL QC DECISION\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    f.write(
        f"Required delta fields complete: "
        f"{all_required_deltas_complete}\n"
    )

    f.write(
        f"Optical-support fields complete: "
        f"{all_support_complete}\n"
    )

    f.write(
        f"All point-level support >= "
        f"{MIN_VALID_OBSERVATIONS}: "
        f"{all_support_at_least_3}\n"
    )

    f.write(
        f"Physical ranges clean: "
        f"{ranges_clean}\n"
    )

    f.write(
        f"Class reproduction exact: "
        f"{class_reproduction_exact}\n"
    )

    f.write(
        f"Indicator reproduction exact: "
        f"{indicator_reproduction_exact}\n"
    )

    f.write(
        f"QC PASS: {qc_pass}\n\n"
    )


    f.write(
        "INTERPRETATION BOUNDARY\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    f.write(
        "If QC passes and no footprint exceeds the "
        "predeclared screening thresholds, the supported "
        "statement is: 'No threshold-exceeding optical/"
        "land-cover change was detected at the retained "
        "GEDI footprints under the conservative "
        "stability screen.'\n"
    )

    f.write(
        "This must NOT be rewritten as 'no disturbance "
        "occurred.' The annual-composite screen may miss "
        "subtle, short-lived, sub-pixel, or spectrally "
        "ambiguous change.\n"
    )


# ============================================================
# CONSOLE
# ============================================================

print("")
print("=" * 84)
print("C3 STEP 3C COMPLETE")
print("STABILITY-SCREEN LOCAL QC")
print("=" * 84)

print("")
print("Change distributions:")
print(
    distribution_df.to_string(
        index=False
    )
)

print("")
print("Optical support:")
print(
    support_df.to_string(
        index=False
    )
)

print("")
print("Range checks:")
print(
    range_df.to_string(
        index=False
    )
)

print("")
print(
    "Class mismatches:",
    int(
        class_mismatch.sum()
    )
)

print(
    "Indicator-count mismatches:",
    int(
        indicator_count_mismatch.sum()
    )
)

print("")
print(
    "Global collection-count warning:",
    global_count_warning
)

print("")
print(
    "QC PASS:",
    qc_pass
)

print("")
print("Report:")
print(REPORT)

print("")
print("NO MODEL WAS FITTED.")
print("NO THRESHOLD WAS CHANGED.")
print("=" * 84)
