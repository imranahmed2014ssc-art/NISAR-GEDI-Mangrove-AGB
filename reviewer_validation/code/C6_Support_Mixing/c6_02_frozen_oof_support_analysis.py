from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from scipy.stats import (
    pearsonr,
    spearmanr,
)

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

ROOT = Path(r"D:\NISAR_Mangrove_Paper")

C5_OOF = (
    ROOT
    / "06_Results"
    / "Reviewer_Validation"
    / "C5_Multiscale_Ablation"
    / "C5_01_ExtraTrees_Ablation"
    / "C5_01_ALL_FAMILY_OOF.csv"
)

SUPPORT_FILE = (
    ROOT
    / "06_Results"
    / "Reviewer_Validation"
    / "C6_Support_Mixing"
    / "C6_01B_Corrected_GMW"
    / "C6_01B_CORRECTED_SUPPORT.csv"
)

OUT = (
    ROOT
    / "06_Results"
    / "Reviewer_Validation"
    / "C6_Support_Mixing"
    / "C6_02_Frozen_OOF"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

FULL14_FILE = (
    OUT
    / "C6_02_FULL14_BY_SUPPORT_CLASS.csv"
)

THRESHOLD_FILE = (
    OUT
    / "C6_02_THRESHOLD_SENSITIVITY.csv"
)

SUPPORT_COMPARE_FILE = (
    OUT
    / "C6_02_9X9_VS_5X5_BY_SUPPORT_CLASS.csv"
)

ASSOCIATION_FILE = (
    OUT
    / "C6_02_CONTINUOUS_ASSOCIATIONS.csv"
)

BOOT_FILE = (
    OUT
    / "C6_02_BLOCK_BOOTSTRAP.csv"
)

MERGED_FILE = (
    OUT
    / "C6_02_ANALYSIS_TABLE.csv"
)

REPORT = (
    OUT
    / "C6_02_FROZEN_OOF_SUPPORT_REPORT.txt"
)


# ============================================================
# FROZEN SETTINGS
# ============================================================

PRIMARY_THRESHOLD = 0.95

SENSITIVITY_THRESHOLDS = [
    1.00,
    0.95,
    0.90,
    0.75,
]

BOOTSTRAP_REPS = 2000

BOOTSTRAP_SEED = 20261029


# ============================================================
# LOAD
# ============================================================

oof = pd.read_csv(
    C5_OOF
)

support = pd.read_csv(
    SUPPORT_FILE
)


# ============================================================
# OOF INTEGRITY
# ============================================================

required_families = [
    "A08_3P_PLUS_5X5_CONTEXT",
    "A09_3P_PLUS_9X9_CONTEXT",
    "A11_FULL_14P",
]


for family in required_families:

    g = oof.loc[
        oof[
            "family"
        ]
        ==
        family
    ]

    if len(g) != 1641:

        raise RuntimeError(
            f"{family}: expected 1641 rows; "
            f"found {len(g)}."
        )

    if (
        g[
            "shot_number"
        ].nunique()
        !=
        1641
    ):

        raise RuntimeError(
            f"{family}: shot_number not unique."
        )


if len(support) != 1641:

    raise RuntimeError(
        f"Support table expected 1641 rows; "
        f"found {len(support)}."
    )


if (
    support[
        "shot_number"
    ].nunique()
    !=
    1641
):

    raise RuntimeError(
        "Support shot_number not unique."
    )


# ============================================================
# STANDARDIZE SHOT ID
# ============================================================

oof[
    "shot_number"
] = (
    oof[
        "shot_number"
    ]
    .astype(str)
)


support[
    "shot_number"
] = (
    support[
        "shot_number"
    ]
    .astype(str)
)


# ============================================================
# BUILD WIDE OOF TABLE
# ============================================================

base = (
    oof.loc[
        oof[
            "family"
        ]
        ==
        "A11_FULL_14P",
        [
            "shot_number",
            "cv_block_id",
            "outer_fold",
            "observed_agbd",
        ],
    ]
    .copy()
)


pred_wide = (
    oof.loc[
        oof[
            "family"
        ].isin(
            required_families
        )
    ]
    .pivot(
        index="shot_number",
        columns="family",
        values="predicted_agbd",
    )
    .reset_index()
)


df = (
    base
    .merge(
        pred_wide,
        on="shot_number",
        how="inner",
        validate="one_to_one",
    )
    .merge(
        support,
        on="shot_number",
        how="inner",
        validate="one_to_one",
    )
)


if len(df) != 1641:

    raise RuntimeError(
        f"Merged expected 1641 rows; "
        f"found {len(df)}."
    )


# ============================================================
# NUMERIC
# ============================================================

numeric = [
    "observed_agbd",
    "corrected_mangrove_fraction_9x9",
    "numberOfLooks_mean_9x9",
    "h5_center_offset_m",
    "A08_3P_PLUS_5X5_CONTEXT",
    "A09_3P_PLUS_9X9_CONTEXT",
    "A11_FULL_14P",
]


for col in numeric:

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
        numeric
    ].isna().any().any()
):

    raise RuntimeError(
        "Missing numeric values found."
    )


# ============================================================
# VERIFY FROZEN PRIMARY CLASS
# ============================================================

recomputed_primary = np.where(
    df[
        "corrected_mangrove_fraction_9x9"
    ]
    >=
    PRIMARY_THRESHOLD,
    "INTERIOR_95",
    "EDGE_MIXED",
)


class_mismatch = int(
    np.sum(
        recomputed_primary
        !=
        df[
            "primary_90m_support_class"
        ].astype(str)
    )
)


if class_mismatch != 0:

    raise RuntimeError(
        f"Primary support class mismatch: "
        f"{class_mismatch}"
    )


# ============================================================
# ERRORS
# ============================================================

df[
    "A11_residual"
] = (
    df[
        "A11_FULL_14P"
    ]
    -
    df[
        "observed_agbd"
    ]
)


df[
    "A11_absolute_error"
] = (
    df[
        "A11_residual"
    ].abs()
)


# ============================================================
# METRICS
# ============================================================

def metrics(
    g,
    prediction_col,
):

    y = (
        g[
            "observed_agbd"
        ]
        .to_numpy(float)
    )

    p = (
        g[
            prediction_col
        ]
        .to_numpy(float)
    )

    e = (
        p
        -
        y
    )


    rmse = float(
        np.sqrt(
            np.mean(
                e ** 2
            )
        )
    )


    mae = float(
        np.mean(
            np.abs(e)
        )
    )


    bias = float(
        np.mean(e)
    )


    ss_res = float(
        np.sum(
            e ** 2
        )
    )


    ss_tot = float(
        np.sum(
            (
                y
                -
                np.mean(y)
            )
            ** 2
        )
    )


    r2 = (
        float(
            1.0
            -
            ss_res
            /
            ss_tot
        )
        if ss_tot > 0
        else np.nan
    )


    if (
        len(y) >= 3
        and
        np.std(y) > 0
        and
        np.std(p) > 0
    ):

        pr = float(
            pearsonr(
                y,
                p,
            ).statistic
        )

        sr = float(
            spearmanr(
                y,
                p,
            ).statistic
        )

    else:

        pr = np.nan
        sr = np.nan


    observed_mean = float(
        np.mean(y)
    )


    observed_sd = float(
        np.std(
            y,
            ddof=1,
        )
    )


    predicted_sd = float(
        np.std(
            p,
            ddof=1,
        )
    )


    sd_ratio = (
        predicted_sd
        /
        observed_sd
        if observed_sd > 0
        else np.nan
    )


    rrmse = (
        rmse
        /
        observed_mean
        *
        100.0
        if observed_mean != 0
        else np.nan
    )


    return {
        "n":
            len(g),

        "blocks":
            int(
                g[
                    "cv_block_id"
                ].nunique()
            ),

        "observed_mean":
            observed_mean,

        "observed_sd":
            observed_sd,

        "RMSE":
            rmse,

        "MAE":
            mae,

        "bias":
            bias,

        "R2":
            r2,

        "Pearson_r":
            pr,

        "Spearman_rho":
            sr,

        "predicted_SD":
            predicted_sd,

        "predicted_to_observed_SD_ratio":
            sd_ratio,

        "rRMSE_percent":
            rrmse,
    }


# ============================================================
# FULL14 PRIMARY SUPPORT CLASSES
# ============================================================

full14_rows = []


for support_class in [
    "ALL",
    "INTERIOR_95",
    "EDGE_MIXED",
]:

    if support_class == "ALL":

        g = df

    else:

        g = df.loc[
            df[
                "primary_90m_support_class"
            ]
            ==
            support_class
        ]


    full14_rows.append({
        "support_class":
            support_class,

        **metrics(
            g,
            "A11_FULL_14P",
        ),
    })


full14_df = pd.DataFrame(
    full14_rows
)

full14_df.to_csv(
    FULL14_FILE,
    index=False,
)


# ============================================================
# THRESHOLD SENSITIVITY
# ============================================================

threshold_rows = []


for threshold in (
    SENSITIVITY_THRESHOLDS
):

    interior = (
        df[
            "corrected_mangrove_fraction_9x9"
        ]
        >=
        threshold
    )


    groups = [
        (
            "INTERIOR",
            df.loc[
                interior
            ],
        ),
        (
            "EDGE_MIXED",
            df.loc[
                ~interior
            ],
        ),
    ]


    for label, g in groups:

        if len(g) == 0:

            continue


        threshold_rows.append({
            "threshold":
                threshold,

            "support_class":
                label,

            **metrics(
                g,
                "A11_FULL_14P",
            ),
        })


threshold_df = pd.DataFrame(
    threshold_rows
)

threshold_df.to_csv(
    THRESHOLD_FILE,
    index=False,
)


# ============================================================
# 9X9 VERSUS 5X5 WITHIN SUPPORT CLASS
# ============================================================

compare_rows = []


for support_class in [
    "ALL",
    "INTERIOR_95",
    "EDGE_MIXED",
]:

    if support_class == "ALL":

        g = df

    else:

        g = df.loc[
            df[
                "primary_90m_support_class"
            ]
            ==
            support_class
        ]


    m9 = metrics(
        g,
        "A09_3P_PLUS_9X9_CONTEXT",
    )


    m5 = metrics(
        g,
        "A08_3P_PLUS_5X5_CONTEXT",
    )


    compare_rows.append({
        "support_class":
            support_class,

        "n":
            len(g),

        "blocks":
            int(
                g[
                    "cv_block_id"
                ].nunique()
            ),

        "A09_9x9_RMSE":
            m9[
                "RMSE"
            ],

        "A08_5x5_RMSE":
            m5[
                "RMSE"
            ],

        "delta_RMSE_9x9_minus_5x5":
            (
                m9[
                    "RMSE"
                ]
                -
                m5[
                    "RMSE"
                ]
            ),

        "A09_9x9_MAE":
            m9[
                "MAE"
            ],

        "A08_5x5_MAE":
            m5[
                "MAE"
            ],

        "delta_MAE_9x9_minus_5x5":
            (
                m9[
                    "MAE"
                ]
                -
                m5[
                    "MAE"
                ]
            ),

        "A09_9x9_R2":
            m9[
                "R2"
            ],

        "A08_5x5_R2":
            m5[
                "R2"
            ],

        "delta_R2_9x9_minus_5x5":
            (
                m9[
                    "R2"
                ]
                -
                m5[
                    "R2"
                ]
            ),
    })


compare_df = pd.DataFrame(
    compare_rows
)

compare_df.to_csv(
    SUPPORT_COMPARE_FILE,
    index=False,
)


# ============================================================
# CONTINUOUS ASSOCIATIONS
# ============================================================

association_rows = []


def association(
    name,
    x,
    y,
):

    temp = pd.DataFrame({
        "x":
            x,

        "y":
            y,
    }).dropna()


    if (
        len(temp) < 3
        or
        temp[
            "x"
        ].std() == 0
        or
        temp[
            "y"
        ].std() == 0
    ):

        return {
            "relationship":
                name,

            "n":
                len(temp),

            "Pearson_r":
                np.nan,

            "Pearson_p":
                np.nan,

            "Spearman_rho":
                np.nan,

            "Spearman_p":
                np.nan,

            "status":
                "NOT_TESTABLE_CONSTANT_OR_INSUFFICIENT",
        }


    pr = pearsonr(
        temp[
            "x"
        ],
        temp[
            "y"
        ],
    )


    sr = spearmanr(
        temp[
            "x"
        ],
        temp[
            "y"
        ],
    )


    return {
        "relationship":
            name,

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

        "status":
            "OK",
    }


association_rows.append(
    association(
        "90m_mangrove_fraction_vs_A11_absolute_error",
        df[
            "corrected_mangrove_fraction_9x9"
        ],
        df[
            "A11_absolute_error"
        ],
    )
)


association_rows.append(
    association(
        "90m_mangrove_fraction_vs_A11_residual",
        df[
            "corrected_mangrove_fraction_9x9"
        ],
        df[
            "A11_residual"
        ],
    )
)


association_rows.append(
    association(
        "numberOfLooks_mean_9x9_vs_A11_absolute_error",
        df[
            "numberOfLooks_mean_9x9"
        ],
        df[
            "A11_absolute_error"
        ],
    )
)


association_rows.append(
    association(
        "numberOfLooks_mean_9x9_vs_A11_residual",
        df[
            "numberOfLooks_mean_9x9"
        ],
        df[
            "A11_residual"
        ],
    )
)


association_rows.append(
    association(
        "GEDI_to_NISAR_center_offset_vs_A11_absolute_error",
        df[
            "h5_center_offset_m"
        ],
        df[
            "A11_absolute_error"
        ],
    )
)


# GCOV fractions are expected to be constant 1.0.
for col in [
    "gcov_valid_fraction_9x9",
    "exception_clean_fraction_9x9",
    "positive_looks_fraction_9x9",
]:

    association_rows.append(
        association(
            f"{col}_vs_A11_absolute_error",
            df[
                col
            ],
            df[
                "A11_absolute_error"
            ],
        )
    )


association_df = pd.DataFrame(
    association_rows
)

association_df.to_csv(
    ASSOCIATION_FILE,
    index=False,
)


# ============================================================
# BLOCK BOOTSTRAP
#
# Full spatial blocks are resampled.
# No model fitting occurs.
# ============================================================

rng = np.random.default_rng(
    BOOTSTRAP_SEED
)


blocks = (
    df[
        "cv_block_id"
    ]
    .drop_duplicates()
    .to_numpy()
)


bootstrap_rows = []


for rep in range(
    BOOTSTRAP_REPS
):

    sampled_blocks = rng.choice(
        blocks,
        size=len(blocks),
        replace=True,
    )


    pieces = []


    for block in sampled_blocks:

        pieces.append(
            df.loc[
                df[
                    "cv_block_id"
                ]
                ==
                block
            ]
        )


    boot = pd.concat(
        pieces,
        ignore_index=True,
    )


    interior = boot.loc[
        boot[
            "primary_90m_support_class"
        ]
        ==
        "INTERIOR_95"
    ]


    edge = boot.loc[
        boot[
            "primary_90m_support_class"
        ]
        ==
        "EDGE_MIXED"
    ]


    if (
        len(
            interior
        )
        >
        1
        and
        len(
            edge
        )
        >
        1
    ):

        mi = metrics(
            interior,
            "A11_FULL_14P",
        )

        me = metrics(
            edge,
            "A11_FULL_14P",
        )


        bootstrap_rows.append({
            "rep":
                rep,

            "analysis":
                "A11_EDGE_MINUS_INTERIOR",

            "delta_RMSE":
                (
                    me[
                        "RMSE"
                    ]
                    -
                    mi[
                        "RMSE"
                    ]
                ),

            "delta_MAE":
                (
                    me[
                        "MAE"
                    ]
                    -
                    mi[
                        "MAE"
                    ]
                ),

            "delta_bias":
                (
                    me[
                        "bias"
                    ]
                    -
                    mi[
                        "bias"
                    ]
                ),

            "delta_R2":
                (
                    me[
                        "R2"
                    ]
                    -
                    mi[
                        "R2"
                    ]
                ),
        })


    # --------------------------------------------------------
    # Paired 9x9 - 5x5 within interior and edge.
    # --------------------------------------------------------

    for label, g in [
        (
            "INTERIOR_95",
            interior,
        ),
        (
            "EDGE_MIXED",
            edge,
        ),
    ]:

        if len(g) <= 1:
            continue


        m9 = metrics(
            g,
            "A09_3P_PLUS_9X9_CONTEXT",
        )


        m5 = metrics(
            g,
            "A08_3P_PLUS_5X5_CONTEXT",
        )


        bootstrap_rows.append({
            "rep":
                rep,

            "analysis":
                f"A09_MINUS_A08_{label}",

            "delta_RMSE":
                (
                    m9[
                        "RMSE"
                    ]
                    -
                    m5[
                        "RMSE"
                    ]
                ),

            "delta_MAE":
                (
                    m9[
                        "MAE"
                    ]
                    -
                    m5[
                        "MAE"
                    ]
                ),

            "delta_bias":
                (
                    m9[
                        "bias"
                    ]
                    -
                    m5[
                        "bias"
                    ]
                ),

            "delta_R2":
                (
                    m9[
                        "R2"
                    ]
                    -
                    m5[
                        "R2"
                    ]
                ),
        })


boot_raw = pd.DataFrame(
    bootstrap_rows
)


# ============================================================
# BOOTSTRAP SUMMARY
# ============================================================

boot_summary_rows = []


for analysis_name, g in (
    boot_raw.groupby(
        "analysis"
    )
):

    for metric_name in [
        "delta_RMSE",
        "delta_MAE",
        "delta_bias",
        "delta_R2",
    ]:

        x = (
            g[
                metric_name
            ]
            .dropna()
            .to_numpy(float)
        )


        if len(x) == 0:
            continue


        boot_summary_rows.append({
            "analysis":
                analysis_name,

            "metric":
                metric_name,

            "valid_bootstrap_reps":
                len(x),

            "median":
                float(
                    np.median(x)
                ),

            "CI_2.5":
                float(
                    np.quantile(
                        x,
                        0.025
                    )
                ),

            "CI_97.5":
                float(
                    np.quantile(
                        x,
                        0.975
                    )
                ),
        })


boot_df = pd.DataFrame(
    boot_summary_rows
)

boot_df.to_csv(
    BOOT_FILE,
    index=False,
)


# ============================================================
# SAVE ANALYSIS TABLE
# ============================================================

df.to_csv(
    MERGED_FILE,
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
        "=" * 110
        + "\n"
    )

    f.write(
        "C6 STEP 2 - FROZEN OOF SUPPORT / "
        "EDGE-MIXING ANALYSIS\n"
    )

    f.write(
        "=" * 110
        + "\n\n"
    )


    f.write(
        "STATUS\n"
    )

    f.write(
        "-" * 110
        + "\n"
    )

    f.write(
        "NO MODEL WAS FITTED.\n"
    )

    f.write(
        "NO MODEL OR PREDICTOR FAMILY WAS RESELECTED.\n"
    )

    f.write(
        "NO SUPPORT THRESHOLD WAS CHANGED.\n"
    )

    f.write(
        "Existing frozen C5 OOF predictions were used.\n"
    )

    f.write(
        "C1-C5 and C6 Step 1B remain frozen.\n\n"
    )


    f.write(
        "INTEGRITY\n"
    )

    f.write(
        "-" * 110
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
        f"Spatial blocks: "
        f"{df['cv_block_id'].nunique()}\n"
    )

    f.write(
        f"Primary-class mismatches: "
        f"{class_mismatch}\n\n"
    )


    f.write(
        "FULL14P PERFORMANCE BY PRIMARY 90-M SUPPORT CLASS\n"
    )

    f.write(
        "-" * 110
        + "\n"
    )

    f.write(
        full14_df.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )


    f.write(
        "FULL14P THRESHOLD SENSITIVITY\n"
    )

    f.write(
        "-" * 110
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
        "9X9 VERSUS 5X5 CONTEXT BY SUPPORT CLASS\n"
    )

    f.write(
        "-" * 110
        + "\n"
    )

    f.write(
        compare_df.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )


    f.write(
        "CONTINUOUS SUPPORT / ERROR ASSOCIATIONS\n"
    )

    f.write(
        "-" * 110
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
        "BLOCK-BOOTSTRAP SENSITIVITY\n"
    )

    f.write(
        "-" * 110
        + "\n"
    )

    f.write(
        boot_df.to_string(
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
        "-" * 110
        + "\n"
    )

    f.write(
        "The primary edge/mixed group is small and "
        "must be treated as a sensitivity subset rather "
        "than an equally powered validation stratum.\n"
    )

    f.write(
        "Differences in RMSE or R2 between interior and "
        "edge groups can also reflect differences in the "
        "AGBD distribution, not only support mixing.\n"
    )

    f.write(
        "Binary GMW background is not interpreted as "
        "open water specifically; it may represent "
        "water or other non-GMW cover.\n"
    )

    f.write(
        "GCOV-valid, exception-clean, and positive-look "
        "fractions are constant at 1.0 across retained "
        "9x9 windows and therefore cannot explain "
        "within-sample error variation.\n"
    )


# ============================================================
# CONSOLE
# ============================================================

print("")
print("=" * 92)
print("C6 STEP 2 COMPLETE")
print("FROZEN OOF SUPPORT / EDGE-MIXING ANALYSIS")
print("=" * 92)

print("")
print("Full14P by primary class:")
print(
    full14_df.to_string(
        index=False
    )
)

print("")
print("Threshold sensitivity:")
print(
    threshold_df.to_string(
        index=False
    )
)

print("")
print("9x9 versus 5x5:")
print(
    compare_df.to_string(
        index=False
    )
)

print("")
print("Continuous associations:")
print(
    association_df.to_string(
        index=False
    )
)

print("")
print("Block-bootstrap:")
print(
    boot_df.to_string(
        index=False
    )
)

print("")
print("Report:")
print(REPORT)

print("")
print("NO MODEL WAS FITTED.")
print("NO THRESHOLD WAS CHANGED.")
print("C1-C5 AND C6 STEP 1B REMAIN FROZEN.")
print("=" * 92)
