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

OOF_FILE = (
    ROOT
    / "06_Results"
    / "Reviewer_Validation"
    / "C1_Independent_Validation"
    / "C1B_Full_Nested_Selection"
    / "C1B_SELECTION_AWARE_OOF.csv"
)

MANIFEST_FILE = (
    ROOT
    / "06_Results"
    / "Reviewer_Validation"
    / "C4_GEDI_Uncertainty"
    / "C4_00_UNCERTAINTY_MANIFEST.csv"
)

OUT = (
    ROOT
    / "06_Results"
    / "Reviewer_Validation"
    / "C4_GEDI_Uncertainty"
    / "C4_01_Frozen_OOF"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

MERGED_FILE = (
    OUT
    / "C4_01_OOF_WITH_GEDI_UNCERTAINTY.csv"
)

STRATUM_FILE = (
    OUT
    / "C4_01_METRICS_BY_SE_STRATUM.csv"
)

REGIME_FILE = (
    OUT
    / "C4_01_METRICS_BY_AGBD_REGIME.csv"
)

ASSOCIATION_FILE = (
    OUT
    / "C4_01_ERROR_UNCERTAINTY_ASSOCIATIONS.csv"
)

NORMALIZED_FILE = (
    OUT
    / "C4_01_SE_NORMALIZED_ERROR.csv"
)

WEIGHT_FILE = (
    OUT
    / "C4_01_INVERSE_VARIANCE_SENSITIVITY.csv"
)

BOOT_FILE = (
    OUT
    / "C4_01_BLOCK_BOOTSTRAP_CI.csv"
)

REPORT = (
    OUT
    / "C4_01_FROZEN_OOF_UNCERTAINTY_REPORT.txt"
)


# ============================================================
# SETTINGS
# ============================================================

BOOTSTRAP_REPS = 2000

BOOTSTRAP_SEED = 20261028


# ============================================================
# LOAD
# ============================================================

oof = pd.read_csv(
    OOF_FILE
)

manifest = pd.read_csv(
    MANIFEST_FILE
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

oof_required = [
    "shot_number",
    "cv_block_id",
    "outer_fold",
    "observed_agbd",
    "predicted_agbd",
]

manifest_required = [
    "shot_number",
    "agbd",
    "agbd_se",
    "agbd_relative_se_percent",
    "AGBD_regime",
    "GEDI_SE_stratum",
    "GEDI_relative_SE_stratum",
    "cv_block_id",
]


for name, frame, required in [
    (
        "OOF",
        oof,
        oof_required,
    ),
    (
        "manifest",
        manifest,
        manifest_required,
    ),
]:

    missing = [
        c
        for c in required
        if c not in frame.columns
    ]

    if missing:

        raise RuntimeError(
            f"{name}: missing columns {missing}"
        )


# ============================================================
# BASIC INTEGRITY
# ============================================================

if len(oof) != 1641:

    raise RuntimeError(
        f"OOF expected 1641 rows; found {len(oof)}."
    )

if len(manifest) != 1641:

    raise RuntimeError(
        f"Manifest expected 1641 rows; found {len(manifest)}."
    )


oof[
    "shot_number"
] = oof[
    "shot_number"
].astype(str)


manifest[
    "shot_number"
] = manifest[
    "shot_number"
].astype(str)


if (
    oof[
        "shot_number"
    ].nunique()
    != 1641
):

    raise RuntimeError(
        "OOF shot_number is not unique."
    )


if (
    manifest[
        "shot_number"
    ].nunique()
    != 1641
):

    raise RuntimeError(
        "Manifest shot_number is not unique."
    )


# ============================================================
# MERGE
# ============================================================

manifest_merge = (
    manifest
    .drop(
        columns=[
            "cv_block_id",
        ]
    )
)


df = oof.merge(
    manifest_merge,
    on="shot_number",
    how="inner",
    validate="one_to_one",
)


if len(df) != 1641:

    raise RuntimeError(
        f"Merged expected 1641 rows; found {len(df)}."
    )


# ============================================================
# OBSERVED AGBD CONSISTENCY
# ============================================================

obs_diff = np.abs(
    pd.to_numeric(
        df[
            "observed_agbd"
        ],
        errors="coerce",
    )
    -
    pd.to_numeric(
        df[
            "agbd"
        ],
        errors="coerce",
    )
)


max_obs_difference = float(
    obs_diff.max()
)


if max_obs_difference > 1e-8:

    raise RuntimeError(
        "Observed AGBD mismatch between frozen "
        f"OOF and C4 manifest. Max diff = "
        f"{max_obs_difference}"
    )


# ============================================================
# BLOCK CONSISTENCY
# ============================================================

if (
    "cv_block_id"
    not in df.columns
):

    raise RuntimeError(
        "cv_block_id missing after merge."
    )


# ============================================================
# NUMERIC
# ============================================================

numeric_columns = [
    "observed_agbd",
    "predicted_agbd",
    "agbd",
    "agbd_se",
    "agbd_relative_se_percent",
]


for col in numeric_columns:

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
        numeric_columns
    ].isna().any().any()
):

    raise RuntimeError(
        "Missing numeric values found."
    )


# ============================================================
# ERROR TERMS
# ============================================================

df[
    "residual"
] = (
    df[
        "predicted_agbd"
    ]
    -
    df[
        "observed_agbd"
    ]
)


df[
    "absolute_error"
] = (
    df[
        "residual"
    ].abs()
)


df[
    "squared_error"
] = (
    df[
        "residual"
    ] ** 2
)


df[
    "abs_error_over_agbd_se"
] = (
    df[
        "absolute_error"
    ]
    /
    df[
        "agbd_se"
    ]
)


# ============================================================
# METRIC FUNCTION
# ============================================================

def metric_row(
    g,
    label,
):

    y = (
        g[
            "observed_agbd"
        ]
        .to_numpy(float)
    )

    p = (
        g[
            "predicted_agbd"
        ]
        .to_numpy(float)
    )

    e = p - y

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
            ) ** 2
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

    pearson = (
        float(
            pearsonr(
                y,
                p
            ).statistic
        )
        if (
            len(y) >= 3
            and np.std(y) > 0
            and np.std(p) > 0
        )
        else np.nan
    )

    spearman = (
        float(
            spearmanr(
                y,
                p
            ).statistic
        )
        if (
            len(y) >= 3
            and np.std(y) > 0
            and np.std(p) > 0
        )
        else np.nan
    )

    obs_mean = float(
        np.mean(y)
    )

    obs_sd = float(
        np.std(
            y,
            ddof=1
        )
    )

    pred_sd = float(
        np.std(
            p,
            ddof=1
        )
    )

    sd_ratio = (
        pred_sd
        /
        obs_sd
        if obs_sd > 0
        else np.nan
    )

    rrmse = (
        rmse
        /
        obs_mean
        * 100.0
        if obs_mean != 0
        else np.nan
    )

    return {
        "group":
            label,

        "n":
            len(g),

        "blocks":
            int(
                g[
                    "cv_block_id"
                ].nunique()
            ),

        "observed_mean":
            obs_mean,

        "observed_sd":
            obs_sd,

        "agbd_se_mean":
            float(
                g[
                    "agbd_se"
                ].mean()
            ),

        "agbd_se_median":
            float(
                g[
                    "agbd_se"
                ].median()
            ),

        "RMSE":
            rmse,

        "MAE":
            mae,

        "bias":
            bias,

        "R2":
            r2,

        "Pearson_r":
            pearson,

        "Spearman_rho":
            spearman,

        "predicted_to_observed_SD_ratio":
            sd_ratio,

        "rRMSE_percent":
            rrmse,

        "median_abs_error_over_agbd_se":
            float(
                g[
                    "abs_error_over_agbd_se"
                ].median()
            ),

        "pct_abs_error_le_1SE":
            float(
                (
                    g[
                        "absolute_error"
                    ]
                    <=
                    g[
                        "agbd_se"
                    ]
                ).mean()
                * 100.0
            ),

        "pct_abs_error_le_2SE":
            float(
                (
                    g[
                        "absolute_error"
                    ]
                    <=
                    2.0
                    *
                    g[
                        "agbd_se"
                    ]
                ).mean()
                * 100.0
            ),
    }


# ============================================================
# OVERALL + PRIMARY SE STRATA
# ============================================================

stratum_rows = [
    metric_row(
        df,
        "ALL",
    )
]


stratum_order = [
    "U1_LOWEST_SE",
    "U2_MIDDLE_SE",
    "U3_HIGHEST_SE",
]


for stratum in stratum_order:

    g = df.loc[
        df[
            "GEDI_SE_stratum"
        ]
        ==
        stratum
    ]

    stratum_rows.append(
        metric_row(
            g,
            stratum,
        )
    )


stratum_df = pd.DataFrame(
    stratum_rows
)

stratum_df.to_csv(
    STRATUM_FILE,
    index=False
)


# ============================================================
# BIOMASS REGIME METRICS
# ============================================================

regime_rows = []


for regime in [
    "<50",
    "50-<100",
    ">=100",
]:

    g = df.loc[
        df[
            "AGBD_regime"
        ].astype(str)
        ==
        regime
    ]

    regime_rows.append(
        metric_row(
            g,
            regime,
        )
    )


regime_df = pd.DataFrame(
    regime_rows
)

regime_df.to_csv(
    REGIME_FILE,
    index=False
)


# ============================================================
# ERROR / UNCERTAINTY ASSOCIATIONS
#
# IMPORTANT:
# These are descriptive associations.
# They are NOT interpreted causally because agbd_se
# is strongly associated with observed AGBD.
# ============================================================

association_rows = []


def add_association(
    subset_name,
    g,
    xcol,
    ycol,
):

    temp = (
        g[
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

    if (
        len(temp) < 3
        or
        np.std(x) == 0
        or
        np.std(y) == 0
    ):

        pr = np.nan
        pp = np.nan
        sr = np.nan
        sp = np.nan

    else:

        p_res = pearsonr(
            x,
            y
        )

        s_res = spearmanr(
            x,
            y
        )

        pr = float(
            p_res.statistic
        )

        pp = float(
            p_res.pvalue
        )

        sr = float(
            s_res.statistic
        )

        sp = float(
            s_res.pvalue
        )

    association_rows.append({
        "subset":
            subset_name,

        "x":
            xcol,

        "y":
            ycol,

        "n":
            len(temp),

        "Pearson_r":
            pr,

        "Pearson_p":
            pp,

        "Spearman_rho":
            sr,

        "Spearman_p":
            sp,
    })


# Overall
for ycol in [
    "absolute_error",
    "squared_error",
]:

    add_association(
        "ALL",
        df,
        "agbd_se",
        ycol,
    )


# Within AGBD regimes
for regime in [
    "<50",
    "50-<100",
    ">=100",
]:

    g = df.loc[
        df[
            "AGBD_regime"
        ].astype(str)
        ==
        regime
    ]

    for ycol in [
        "absolute_error",
        "squared_error",
    ]:

        add_association(
            f"AGBD_{regime}",
            g,
            "agbd_se",
            ycol,
        )


# AGBD itself versus error for confounding context
for ycol in [
    "absolute_error",
    "squared_error",
]:

    add_association(
        "ALL",
        df,
        "observed_agbd",
        ycol,
    )


association_df = pd.DataFrame(
    association_rows
)

association_df.to_csv(
    ASSOCIATION_FILE,
    index=False
)


# ============================================================
# SE-NORMALIZED ERROR SUMMARY
# ============================================================

ratio = (
    df[
        "abs_error_over_agbd_se"
    ]
    .to_numpy(float)
)


normalized_df = pd.DataFrame([
    {
        "n":
            len(df),

        "median_abs_error_over_SE":
            float(
                np.median(ratio)
            ),

        "q75_abs_error_over_SE":
            float(
                np.quantile(
                    ratio,
                    0.75
                )
            ),

        "q90_abs_error_over_SE":
            float(
                np.quantile(
                    ratio,
                    0.90
                )
            ),

        "q95_abs_error_over_SE":
            float(
                np.quantile(
                    ratio,
                    0.95
                )
            ),

        "pct_abs_error_le_1SE":
            float(
                np.mean(
                    ratio <= 1.0
                )
                * 100.0
            ),

        "pct_abs_error_le_2SE":
            float(
                np.mean(
                    ratio <= 2.0
                )
                * 100.0
            ),

        "pct_abs_error_le_3SE":
            float(
                np.mean(
                    ratio <= 3.0
                )
                * 100.0
            ),
    }
])


normalized_df.to_csv(
    NORMALIZED_FILE,
    index=False
)


# ============================================================
# INVERSE-VARIANCE SENSITIVITY
#
# Existing OOF errors only.
# NO MODEL IS REFIT.
# ============================================================

raw_w = (
    1.0
    /
    (
        df[
            "agbd_se"
        ].to_numpy(float)
        ** 2
    )
)


w = (
    raw_w
    /
    np.mean(
        raw_w
    )
)


errors = (
    df[
        "residual"
    ]
    .to_numpy(float)
)


weighted_rmse = float(
    np.sqrt(
        np.sum(
            w
            *
            errors ** 2
        )
        /
        np.sum(w)
    )
)


weighted_mae = float(
    np.sum(
        w
        *
        np.abs(errors)
    )
    /
    np.sum(w)
)


unweighted_rmse = float(
    np.sqrt(
        np.mean(
            errors ** 2
        )
    )
)


unweighted_mae = float(
    np.mean(
        np.abs(errors)
    )
)


weight_df = pd.DataFrame([
    {
        "agbd_se_min":
            float(
                df[
                    "agbd_se"
                ].min()
            ),

        "agbd_se_max":
            float(
                df[
                    "agbd_se"
                ].max()
            ),

        "agbd_se_total_range":
            float(
                df[
                    "agbd_se"
                ].max()
                -
                df[
                    "agbd_se"
                ].min()
            ),

        "normalized_weight_min":
            float(
                np.min(w)
            ),

        "normalized_weight_max":
            float(
                np.max(w)
            ),

        "max_to_min_weight_ratio":
            float(
                np.max(w)
                /
                np.min(w)
            ),

        "unweighted_RMSE":
            unweighted_rmse,

        "inverse_variance_weighted_RMSE":
            weighted_rmse,

        "weighted_minus_unweighted_RMSE":
            weighted_rmse
            -
            unweighted_rmse,

        "unweighted_MAE":
            unweighted_mae,

        "inverse_variance_weighted_MAE":
            weighted_mae,

        "weighted_minus_unweighted_MAE":
            weighted_mae
            -
            unweighted_mae,
    }
])


weight_df.to_csv(
    WEIGHT_FILE,
    index=False
)


# ============================================================
# BLOCK BOOTSTRAP
#
# Resample the 54 spatial blocks.
# Existing OOF predictions only.
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


bootstrap_records = []


for rep in range(
    BOOTSTRAP_REPS
):

    sampled_blocks = rng.choice(
        blocks,
        size=len(blocks),
        replace=True,
    )

    pieces = []

    for b in sampled_blocks:

        pieces.append(
            df.loc[
                df[
                    "cv_block_id"
                ]
                ==
                b
            ]
        )

    boot = pd.concat(
        pieces,
        ignore_index=True,
    )


    for stratum in stratum_order:

        g = boot.loc[
            boot[
                "GEDI_SE_stratum"
            ]
            ==
            stratum
        ]

        if len(g) < 2:

            continue

        y = g[
            "observed_agbd"
        ].to_numpy(float)

        p = g[
            "predicted_agbd"
        ].to_numpy(float)

        e = p - y

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

        ss_tot = float(
            np.sum(
                (
                    y
                    -
                    np.mean(y)
                ) ** 2
            )
        )

        r2 = (
            float(
                1
                -
                np.sum(
                    e ** 2
                )
                /
                ss_tot
            )
            if ss_tot > 0
            else np.nan
        )

        bootstrap_records.append({
            "rep":
                rep,

            "group":
                stratum,

            "RMSE":
                rmse,

            "MAE":
                mae,

            "bias":
                bias,

            "R2":
                r2,
        })


boot_raw = pd.DataFrame(
    bootstrap_records
)


ci_rows = []


for group, g in boot_raw.groupby(
    "group"
):

    for metric in [
        "RMSE",
        "MAE",
        "bias",
        "R2",
    ]:

        x = (
            g[
                metric
            ]
            .dropna()
            .to_numpy(float)
        )

        ci_rows.append({
            "group":
                group,

            "metric":
                metric,

            "bootstrap_reps":
                len(x),

            "estimate":
                float(
                    stratum_df.loc[
                        stratum_df[
                            "group"
                        ]
                        ==
                        group,
                        metric,
                    ].iloc[0]
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


boot_ci_df = pd.DataFrame(
    ci_rows
)

boot_ci_df.to_csv(
    BOOT_FILE,
    index=False
)


# ============================================================
# SAVE MERGED DATA
# ============================================================

df.to_csv(
    MERGED_FILE,
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
        "C4 STEP 1 - FROZEN C1B OOF PERFORMANCE "
        "VERSUS GEDI AGBD_SE\n"
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
        "NO MODEL OR PREDICTOR FAMILY WAS RESELECTED.\n"
    )

    f.write(
        "Existing frozen C1B selection-aware OOF "
        "predictions were used.\n"
    )

    f.write(
        "C1, C2, C3 and C4 Step 0 remain frozen.\n\n"
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
        f"Unique spatial blocks: "
        f"{df['cv_block_id'].nunique()}\n"
    )

    f.write(
        f"Maximum observed-AGBD merge difference: "
        f"{max_obs_difference:.16g}\n\n"
    )


    f.write(
        "IMPORTANT INTERPRETATION CONSTRAINT\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    f.write(
        "GEDI agbd_se varies over a very narrow absolute "
        "range and is strongly associated with AGBD.\n"
    )

    f.write(
        "Therefore U1/U2/U3 comparisons are descriptive "
        "and must NOT be interpreted as an independent "
        "causal effect of GEDI reference uncertainty.\n"
    )

    f.write(
        "Relative SE is additionally denominator-coupled "
        "to AGBD and is not treated as an independent "
        "reference-quality measure.\n\n"
    )


    f.write(
        "FROZEN OOF PERFORMANCE BY ABSOLUTE-SE STRATUM\n"
    )

    f.write(
        "-" * 100
        + "\n"
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
        "FROZEN OOF PERFORMANCE BY AGBD REGIME\n"
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
        "ERROR / AGBD_SE ASSOCIATIONS\n"
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
        "ERROR MAGNITUDE RELATIVE TO GEDI AGBD_SE\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    f.write(
        normalized_df.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )


    f.write(
        "INVERSE-VARIANCE EVALUATION SENSITIVITY\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    f.write(
        weight_df.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )


    f.write(
        "BLOCK-BOOTSTRAP 95% INTERVALS\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    f.write(
        boot_ci_df.to_string(
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
        "GEDI agbd_se represents product-reported "
        "uncertainty and does not contain all reference, "
        "representativeness, temporal, or spatial-support "
        "error sources.\n"
    )

    f.write(
        "Differences in model error across agbd_se strata "
        "cannot be attributed solely to GEDI uncertainty "
        "because agbd_se is strongly confounded with "
        "observed biomass.\n"
    )

    f.write(
        "Inverse-variance weighted evaluation is reported "
        "only as a sensitivity diagnostic; no weighted "
        "model has been fitted in this step.\n"
    )


# ============================================================
# CONSOLE
# ============================================================

print("")
print("=" * 84)
print("C4 STEP 1 COMPLETE")
print("FROZEN OOF × GEDI UNCERTAINTY ANALYSIS")
print("=" * 84)

print("")
print("Performance by absolute-SE stratum:")
print(
    stratum_df.to_string(
        index=False
    )
)

print("")
print("Performance by AGBD regime:")
print(
    regime_df.to_string(
        index=False
    )
)

print("")
print("Error / uncertainty associations:")
print(
    association_df.to_string(
        index=False
    )
)

print("")
print("SE-normalized error:")
print(
    normalized_df.to_string(
        index=False
    )
)

print("")
print("Inverse-variance sensitivity:")
print(
    weight_df.to_string(
        index=False
    )
)

print("")
print("Report:")
print(REPORT)

print("")
print("NO MODEL WAS FITTED.")
print("C1, C2, C3 AND C4 STEP 0 REMAIN FROZEN.")
print("=" * 84)
