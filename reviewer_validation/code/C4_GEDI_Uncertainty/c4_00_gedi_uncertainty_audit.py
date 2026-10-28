from pathlib import Path
import numpy as np
import pandas as pd

from scipy.stats import (
    pearsonr,
    spearmanr,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(r"D:\NISAR_Mangrove_Paper")

DATA = (
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
    / "C4_GEDI_Uncertainty"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

QUANTILE_FILE = (
    OUT
    / "C4_00_AGBD_SE_QUANTILES.csv"
)

REGIME_FILE = (
    OUT
    / "C4_00_AGBD_SE_BY_BIOMASS_REGIME.csv"
)

STRATUM_FILE = (
    OUT
    / "C4_00_UNCERTAINTY_STRATA_SUMMARY.csv"
)

MANIFEST_FILE = (
    OUT
    / "C4_00_UNCERTAINTY_MANIFEST.csv"
)

ASSOCIATION_FILE = (
    OUT
    / "C4_00_UNCERTAINTY_ASSOCIATIONS.csv"
)

REPORT = (
    OUT
    / "C4_00_GEDI_UNCERTAINTY_AUDIT_REPORT.txt"
)


# ============================================================
# LOAD FROZEN MASTER
# ============================================================

df = pd.read_csv(
    DATA
)

if len(df) != 1641:

    raise RuntimeError(
        f"Expected 1641 rows; found {len(df)}."
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


required = [
    "shot_number",
    "agbd",
    "agbd_se",
    "cv_block_id",
    "source_file",
    "acquisition_date",
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
# NUMERIC QC
# ============================================================

for col in [
    "agbd",
    "agbd_se",
]:

    df[
        col
    ] = pd.to_numeric(
        df[
            col
        ],
        errors="coerce",
    )


if (
    df[
        "agbd"
    ].isna().any()
):

    raise RuntimeError(
        "Missing/non-numeric AGBD values."
    )


if (
    df[
        "agbd_se"
    ].isna().any()
):

    raise RuntimeError(
        "Missing/non-numeric agbd_se values."
    )


if (
    df[
        "agbd_se"
    ] < 0
).any():

    raise RuntimeError(
        "Negative agbd_se values found."
    )


# ============================================================
# RELATIVE GEDI UNCERTAINTY
#
# Diagnostic only.
# Not used as the primary uncertainty stratum.
# ============================================================

df[
    "agbd_relative_se_percent"
] = np.where(
    df[
        "agbd"
    ] > 0,
    (
        df[
            "agbd_se"
        ]
        /
        df[
            "agbd"
        ]
        * 100.0
    ),
    np.nan,
)


# ============================================================
# BIOMASS REGIMES
# ============================================================

df[
    "AGBD_regime"
] = pd.cut(
    df[
        "agbd"
    ],
    bins=[
        -np.inf,
        50.0,
        100.0,
        np.inf,
    ],
    labels=[
        "<50",
        "50-<100",
        ">=100",
    ],
    right=False,
)


# ============================================================
# DISTRIBUTION QUANTILES
# ============================================================

quantiles = [
    0.00,
    0.01,
    0.05,
    0.10,
    0.25,
    1.0 / 3.0,
    0.50,
    2.0 / 3.0,
    0.75,
    0.90,
    0.95,
    0.99,
    1.00,
]


quantile_rows = []


for variable in [
    "agbd_se",
    "agbd_relative_se_percent",
]:

    x = (
        df[
            variable
        ]
        .dropna()
        .to_numpy(float)
    )

    for q in quantiles:

        quantile_rows.append({
            "variable":
                variable,

            "quantile":
                q,

            "value":
                float(
                    np.quantile(
                        x,
                        q
                    )
                ),

            "n":
                len(x),
        })


quantile_df = pd.DataFrame(
    quantile_rows
)

quantile_df.to_csv(
    QUANTILE_FILE,
    index=False
)


# ============================================================
# FREEZE PRIMARY UNCERTAINTY STRATA
#
# Primary stratification uses absolute GEDI agbd_se only.
#
# Cut points are the 1/3 and 2/3 quantiles.
# No model predictions or model errors are used.
# ============================================================

SE_Q1 = float(
    df[
        "agbd_se"
    ].quantile(
        1.0 / 3.0
    )
)

SE_Q2 = float(
    df[
        "agbd_se"
    ].quantile(
        2.0 / 3.0
    )
)


def absolute_uncertainty_stratum(
    x
):

    if x <= SE_Q1:

        return (
            "U1_LOWEST_SE"
        )

    if x <= SE_Q2:

        return (
            "U2_MIDDLE_SE"
        )

    return (
        "U3_HIGHEST_SE"
    )


df[
    "GEDI_SE_stratum"
] = (
    df[
        "agbd_se"
    ]
    .apply(
        absolute_uncertainty_stratum
    )
)


# ============================================================
# SECONDARY RELATIVE-UNCERTAINTY STRATA
#
# Diagnostic only because relative SE includes AGBD
# in the denominator.
# ============================================================

relative_valid = (
    df[
        "agbd_relative_se_percent"
    ]
    .dropna()
)


REL_Q1 = float(
    relative_valid.quantile(
        1.0 / 3.0
    )
)

REL_Q2 = float(
    relative_valid.quantile(
        2.0 / 3.0
    )
)


def relative_uncertainty_stratum(
    x
):

    if pd.isna(
        x
    ):

        return (
            "RELATIVE_SE_UNDEFINED"
        )

    if x <= REL_Q1:

        return (
            "R1_LOWEST_RELATIVE_SE"
        )

    if x <= REL_Q2:

        return (
            "R2_MIDDLE_RELATIVE_SE"
        )

    return (
        "R3_HIGHEST_RELATIVE_SE"
    )


df[
    "GEDI_relative_SE_stratum"
] = (
    df[
        "agbd_relative_se_percent"
    ]
    .apply(
        relative_uncertainty_stratum
    )
)


# ============================================================
# BIOMASS-REGIME SUMMARY
# ============================================================

regime_rows = []


for regime, g in df.groupby(
    "AGBD_regime",
    observed=True,
):

    x = (
        g[
            "agbd_se"
        ]
        .to_numpy(float)
    )

    r = (
        g[
            "agbd_relative_se_percent"
        ]
        .dropna()
        .to_numpy(float)
    )

    regime_rows.append({
        "AGBD_regime":
            str(regime),

        "n":
            len(g),

        "AGBD_mean":
            float(
                g[
                    "agbd"
                ].mean()
            ),

        "agbd_se_mean":
            float(
                np.mean(x)
            ),

        "agbd_se_median":
            float(
                np.median(x)
            ),

        "agbd_se_q25":
            float(
                np.quantile(
                    x,
                    0.25
                )
            ),

        "agbd_se_q75":
            float(
                np.quantile(
                    x,
                    0.75
                )
            ),

        "agbd_se_q90":
            float(
                np.quantile(
                    x,
                    0.90
                )
            ),

        "agbd_se_q95":
            float(
                np.quantile(
                    x,
                    0.95
                )
            ),

        "relative_se_percent_median":
            (
                float(
                    np.median(r)
                )
                if len(r)
                else np.nan
            ),

        "relative_se_percent_q90":
            (
                float(
                    np.quantile(
                        r,
                        0.90
                    )
                )
                if len(r)
                else np.nan
            ),
    })


regime_df = pd.DataFrame(
    regime_rows
)

regime_df.to_csv(
    REGIME_FILE,
    index=False
)


# ============================================================
# PRIMARY STRATUM SUMMARY
# ============================================================

stratum_rows = []


for stratum, g in df.groupby(
    "GEDI_SE_stratum",
    sort=True,
):

    stratum_rows.append({
        "GEDI_SE_stratum":
            stratum,

        "n":
            len(g),

        "blocks":
            int(
                g[
                    "cv_block_id"
                ].nunique()
            ),

        "agbd_se_min":
            float(
                g[
                    "agbd_se"
                ].min()
            ),

        "agbd_se_median":
            float(
                g[
                    "agbd_se"
                ].median()
            ),

        "agbd_se_max":
            float(
                g[
                    "agbd_se"
                ].max()
            ),

        "AGBD_mean":
            float(
                g[
                    "agbd"
                ].mean()
            ),

        "AGBD_sd":
            float(
                g[
                    "agbd"
                ].std(
                    ddof=1
                )
            ),

        "relative_se_percent_median":
            float(
                g[
                    "agbd_relative_se_percent"
                ].median()
            ),
    })


stratum_df = pd.DataFrame(
    stratum_rows
)

stratum_df.to_csv(
    STRATUM_FILE,
    index=False
)


# ============================================================
# ASSOCIATIONS
# ============================================================

association_rows = []


pairs = [
    (
        "AGBD_vs_absolute_SE",
        "agbd",
        "agbd_se",
    ),
    (
        "AGBD_vs_relative_SE_percent",
        "agbd",
        "agbd_relative_se_percent",
    ),
]


for label, xcol, ycol in pairs:

    temp = (
        df[
            [
                xcol,
                ycol,
            ]
        ]
        .dropna()
    )

    x = temp[
        xcol
    ].to_numpy(float)

    y = temp[
        ycol
    ].to_numpy(float)

    pr = pearsonr(
        x,
        y
    )

    sr = spearmanr(
        x,
        y
    )

    association_rows.append({
        "relationship":
            label,

        "n":
            len(temp),

        "Pearson_r":
            float(
                pr.statistic
            ),

        "Pearson_p":
            float(
                pr.pvalue
            ),

        "Spearman_rho":
            float(
                sr.statistic
            ),

        "Spearman_p":
            float(
                sr.pvalue
            ),
    })


association_df = pd.DataFrame(
    association_rows
)

association_df.to_csv(
    ASSOCIATION_FILE,
    index=False
)


# ============================================================
# SAVE FROZEN MANIFEST
# ============================================================

manifest_columns = [
    "shot_number",
    "agbd",
    "agbd_se",
    "agbd_relative_se_percent",
    "AGBD_regime",
    "GEDI_SE_stratum",
    "GEDI_relative_SE_stratum",
    "cv_block_id",
    "source_file",
    "acquisition_date",
]

if "fold" in df.columns:

    manifest_columns.append(
        "fold"
    )


df[
    manifest_columns
].to_csv(
    MANIFEST_FILE,
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
        "C4 STEP 0 - GEDI L4A AGBD "
        "UNCERTAINTY AUDIT AND PRE-MODEL STRATIFICATION\n"
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
        "NO MODEL WAS FITTED.\n"
    )

    f.write(
        "NO EXISTING MODEL PREDICTIONS WERE "
        "USED TO DEFINE UNCERTAINTY STRATA.\n"
    )

    f.write(
        "C1, C2 and C3 remain frozen.\n\n"
    )


    f.write(
        "DATASET\n"
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
        f"Missing agbd_se: "
        f"{df['agbd_se'].isna().sum()}\n\n"
    )


    f.write(
        "PRIMARY ABSOLUTE-SE STRATA\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    f.write(
        f"Lower tertile cut point: "
        f"{SE_Q1:.6f} Mg/ha\n"
    )

    f.write(
        f"Upper tertile cut point: "
        f"{SE_Q2:.6f} Mg/ha\n"
    )

    f.write(
        "These cut points are frozen before "
        "examining model errors by uncertainty.\n\n"
    )

    f.write(
        stratum_df.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )


    f.write(
        "AGBD_SE QUANTILES\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    f.write(
        quantile_df.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )


    f.write(
        "UNCERTAINTY BY BIOMASS REGIME\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    f.write(
        regime_df.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )


    f.write(
        "AGBD / UNCERTAINTY ASSOCIATIONS\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    f.write(
        association_df.to_string(
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
        "GEDI agbd_se is treated as the product-reported "
        "uncertainty associated with the GEDI AGBD estimate.\n"
    )

    f.write(
        "It is not interpreted as the full error of the "
        "reference observation and does not capture every "
        "systematic, spatial-support, temporal, or "
        "representativeness error source.\n"
    )

    f.write(
        "Inverse-variance weighting is NOT assumed to be "
        "automatically appropriate.\n"
    )

    f.write(
        "Subsequent C4 analyses will first evaluate existing "
        "frozen OOF predictions across the predeclared "
        "uncertainty strata before any uncertainty-aware "
        "sensitivity analysis is considered.\n"
    )


# ============================================================
# CONSOLE
# ============================================================

print("")
print("=" * 84)
print("C4 STEP 0 COMPLETE")
print("GEDI AGBD UNCERTAINTY AUDIT")
print("=" * 84)

print("")
print(
    "Absolute-SE tertile cut points:"
)

print(
    "  Q1 =",
    f"{SE_Q1:.6f}",
    "Mg/ha"
)

print(
    "  Q2 =",
    f"{SE_Q2:.6f}",
    "Mg/ha"
)

print("")
print(
    "Primary uncertainty strata:"
)

print(
    stratum_df.to_string(
        index=False
    )
)

print("")
print(
    "By biomass regime:"
)

print(
    regime_df.to_string(
        index=False
    )
)

print("")
print(
    "Associations:"
)

print(
    association_df.to_string(
        index=False
    )
)

print("")
print("Report:")
print(REPORT)

print("")
print("NO MODEL WAS FITTED.")
print("C1, C2 AND C3 REMAIN FROZEN.")
print("=" * 84)
