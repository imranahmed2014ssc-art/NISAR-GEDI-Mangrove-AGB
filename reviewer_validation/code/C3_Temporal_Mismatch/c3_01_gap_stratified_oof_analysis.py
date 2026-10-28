from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from scipy.stats import (
    pearsonr,
    spearmanr,
)

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

ROOT = Path(r"D:\NISAR_Mangrove_Paper")

C3_ROOT = (
    ROOT
    / "06_Results"
    / "Reviewer_Validation"
    / "C3_Temporal_Mismatch"
)

MANIFEST = (
    C3_ROOT
    / "C3_00_SHOT_GAP_STRATUM_MANIFEST.csv"
)

C1B_OOF = (
    ROOT
    / "06_Results"
    / "Reviewer_Validation"
    / "C1_Independent_Validation"
    / "C1B_Full_Nested_Selection"
    / "C1B_SELECTION_AWARE_OOF.csv"
)

PASS_OOF = (
    ROOT
    / "06_Results"
    / "Reviewer_Validation"
    / "C2_Spatial_Transferability"
    / "C2_02B_Pass_Holdout"
    / "C2_02B_PASS_OOF.csv"
)

OUT = (
    C3_ROOT
    / "C3_01_Gap_Stratified_OOF"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# OUTPUT FILES
# ============================================================

C1B_STRATA = (
    OUT
    / "C3_01_C1B_PERFORMANCE_BY_GAP_STRATUM.csv"
)

PASS_STRATA = (
    OUT
    / "C3_01_PASS_HOLDOUT_BY_GAP_STRATUM.csv"
)

SHORTEST = (
    OUT
    / "C3_01_SHORTEST_GAP_COMPARISON.csv"
)

PASS_METRICS = (
    OUT
    / "C3_01_PER_PASS_TEMPORAL_METRICS.csv"
)

PASS_ASSOC = (
    OUT
    / "C3_01_PASS_GAP_ERROR_ASSOCIATION.csv"
)

BOOT_SUMMARY = (
    OUT
    / "C3_01_BLOCK_BOOTSTRAP_STRATUM_SUMMARY.csv"
)

MERGED_C1B = (
    OUT
    / "C3_01_C1B_TEMPORAL_OOF.csv"
)

MERGED_PASS = (
    OUT
    / "C3_01_PASS_TEMPORAL_OOF.csv"
)

REPORT = (
    OUT
    / "C3_01_TEMPORAL_GAP_PERFORMANCE_REPORT.txt"
)


# ============================================================
# SETTINGS
# ============================================================

BOOT_REPS = 5000
BOOT_SEED = 20261028

STRATUM_ORDER = [
    "G1_SHORTEST",
    "G2_SHORT_MID",
    "G3_LONG_MID",
    "G4_LONGEST",
]


# ============================================================
# METRIC FUNCTION
# ============================================================

def metric_row(
    y,
    pred,
    baseline,
    label,
    n_blocks,
):

    y = np.asarray(
        y,
        dtype=float
    )

    pred = np.asarray(
        pred,
        dtype=float
    )

    baseline = np.asarray(
        baseline,
        dtype=float
    )

    rmse = float(
        np.sqrt(
            mean_squared_error(
                y,
                pred
            )
        )
    )

    base_rmse = float(
        np.sqrt(
            mean_squared_error(
                y,
                baseline
            )
        )
    )

    mae = float(
        mean_absolute_error(
            y,
            pred
        )
    )

    base_mae = float(
        mean_absolute_error(
            y,
            baseline
        )
    )

    r2 = (
        float(
            r2_score(
                y,
                pred
            )
        )
        if len(y) >= 2
        else np.nan
    )

    base_r2 = (
        float(
            r2_score(
                y,
                baseline
            )
        )
        if len(y) >= 2
        else np.nan
    )

    bias = float(
        np.mean(
            pred - y
        )
    )

    if (
        len(y) >= 3
        and np.std(y) > 0
        and np.std(pred) > 0
    ):

        pr = float(
            pearsonr(
                y,
                pred
            ).statistic
        )

        sr = float(
            spearmanr(
                y,
                pred
            ).statistic
        )

    else:

        pr = np.nan
        sr = np.nan

    return {
        "subset":
            label,

        "n":
            len(y),

        "n_blocks":
            n_blocks,

        "observed_mean":
            float(
                np.mean(y)
            ),

        "RMSE_Mg_ha":
            rmse,

        "MAE_Mg_ha":
            mae,

        "R2":
            r2,

        "Bias_Mg_ha":
            bias,

        "Pearson_r":
            pr,

        "Spearman_rho":
            sr,

        "baseline_RMSE":
            base_rmse,

        "baseline_MAE":
            base_mae,

        "baseline_R2":
            base_r2,

        "delta_RMSE_model_minus_mean":
            rmse - base_rmse,

        "delta_MAE_model_minus_mean":
            mae - base_mae,

        "delta_R2_model_minus_mean":
            r2 - base_r2,
    }


# ============================================================
# BLOCK BOOTSTRAP
# ============================================================

def block_bootstrap(
    df,
    label,
):

    block_rows = []

    for block_id, g in df.groupby(
        "cv_block_id"
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

        b = (
            g[
                "mean_only_prediction"
            ]
            .to_numpy(float)
        )

        block_rows.append({
            "cv_block_id":
                block_id,

            "n":
                len(g),

            "sum_y":
                float(
                    np.sum(y)
                ),

            "sum_y2":
                float(
                    np.sum(
                        y ** 2
                    )
                ),

            "sse_model":
                float(
                    np.sum(
                        (p - y) ** 2
                    )
                ),

            "sse_mean":
                float(
                    np.sum(
                        (b - y) ** 2
                    )
                ),

            "sae_model":
                float(
                    np.sum(
                        np.abs(
                            p - y
                        )
                    )
                ),

            "sae_mean":
                float(
                    np.sum(
                        np.abs(
                            b - y
                        )
                    )
                ),
        })

    source = pd.DataFrame(
        block_rows
    )

    B = len(source)

    if B < 2:

        return []

    seed_offset = sum(
        ord(x)
        for x in str(label)
    )

    rng = np.random.default_rng(
        BOOT_SEED
        + seed_offset
    )

    rows = []

    for rep in range(
        BOOT_REPS
    ):

        idx = rng.integers(
            0,
            B,
            size=B,
        )

        s = source.iloc[
            idx
        ]

        n = float(
            s[
                "n"
            ].sum()
        )

        sy = float(
            s[
                "sum_y"
            ].sum()
        )

        sy2 = float(
            s[
                "sum_y2"
            ].sum()
        )

        sse_m = float(
            s[
                "sse_model"
            ].sum()
        )

        sse_b = float(
            s[
                "sse_mean"
            ].sum()
        )

        sae_m = float(
            s[
                "sae_model"
            ].sum()
        )

        sae_b = float(
            s[
                "sae_mean"
            ].sum()
        )

        rmse_m = np.sqrt(
            sse_m / n
        )

        rmse_b = np.sqrt(
            sse_b / n
        )

        mae_m = (
            sae_m / n
        )

        mae_b = (
            sae_b / n
        )

        sst = (
            sy2
            - sy ** 2 / n
        )

        r2_m = (
            1.0
            - sse_m / sst
            if sst > 0
            else np.nan
        )

        r2_b = (
            1.0
            - sse_b / sst
            if sst > 0
            else np.nan
        )

        rows.append({
            "delta_RMSE":
                rmse_m - rmse_b,

            "delta_MAE":
                mae_m - mae_b,

            "delta_R2":
                r2_m - r2_b,
        })

    boot = pd.DataFrame(
        rows
    )

    output = []

    for metric in [
        "delta_RMSE",
        "delta_MAE",
        "delta_R2",
    ]:

        v = (
            boot[
                metric
            ]
            .to_numpy(float)
        )

        output.append({
            "subset":
                label,

            "metric":
                metric,

            "n_blocks":
                B,

            "mean":
                float(
                    np.nanmean(v)
                ),

            "median":
                float(
                    np.nanmedian(v)
                ),

            "CI_2.5":
                float(
                    np.nanquantile(
                        v,
                        0.025
                    )
                ),

            "CI_97.5":
                float(
                    np.nanquantile(
                        v,
                        0.975
                    )
                ),

            "n_replicates":
                BOOT_REPS,
        })

    return output


# ============================================================
# LOAD DATA
# ============================================================

manifest = pd.read_csv(
    MANIFEST
)

c1b = pd.read_csv(
    C1B_OOF
)

pass_oof = pd.read_csv(
    PASS_OOF
)


for name, d in [
    ("manifest", manifest),
    ("C1B", c1b),
    ("pass OOF", pass_oof),
]:

    if len(d) != 1641:

        raise RuntimeError(
            f"{name}: expected 1641 rows; "
            f"found {len(d)}."
        )

    if (
        d[
            "shot_number"
        ].nunique()
        != 1641
    ):

        raise RuntimeError(
            f"{name}: shot_number not unique."
        )


# ============================================================
# TEMPORAL METADATA
# ============================================================

temporal_cols = [
    "shot_number",
    "source_file",
    "acquisition_date",
    "delta_days_to_NISAR",
    "gap_rank_shortest_first",
    "gap_stratum",
    "shortest_gap_subset",
]

temporal = (
    manifest[
        temporal_cols
    ]
    .copy()
)


# ============================================================
# MERGE C1B
# ============================================================

c1 = (
    c1b.merge(
        temporal,
        on="shot_number",
        how="left",
        validate="one_to_one",
    )
)

if (
    c1[
        "gap_stratum"
    ].isna().any()
):

    raise RuntimeError(
        "C1B merge lost temporal metadata."
    )


# ============================================================
# MERGE PASS-HOLDOUT
# ============================================================

ph = (
    pass_oof.merge(
        temporal,
        on="shot_number",
        how="left",
        validate="one_to_one",
        suffixes=(
            "",
            "_temporal"
        )
    )
)

if (
    ph[
        "gap_stratum"
    ].isna().any()
):

    raise RuntimeError(
        "Pass-holdout merge lost "
        "temporal metadata."
    )


c1.to_csv(
    MERGED_C1B,
    index=False
)

ph.to_csv(
    MERGED_PASS,
    index=False
)


# ============================================================
# C1B STRATIFIED PERFORMANCE
# ============================================================

c1_rows = []

bootstrap_rows = []


for stratum in STRATUM_ORDER:

    g = (
        c1[
            c1[
                "gap_stratum"
            ] == stratum
        ]
        .copy()
    )

    c1_rows.append(
        metric_row(
            g[
                "observed_agbd"
            ],
            g[
                "predicted_agbd"
            ],
            g[
                "mean_only_prediction"
            ],
            stratum,
            g[
                "cv_block_id"
            ].nunique(),
        )
    )

    bootstrap_rows.extend(
        block_bootstrap(
            g,
            "C1B_" + stratum
        )
    )


# ============================================================
# PASS-HOLDOUT STRATIFIED PERFORMANCE
# ============================================================

ph_rows = []


for stratum in STRATUM_ORDER:

    g = (
        ph[
            ph[
                "gap_stratum"
            ] == stratum
        ]
        .copy()
    )

    ph_rows.append(
        metric_row(
            g[
                "observed_agbd"
            ],
            g[
                "predicted_agbd"
            ],
            g[
                "mean_only_prediction"
            ],
            stratum,
            g[
                "cv_block_id"
            ].nunique(),
        )
    )

    bootstrap_rows.extend(
        block_bootstrap(
            g,
            "PASS_" + stratum
        )
    )


c1_strata_df = pd.DataFrame(
    c1_rows
)

ph_strata_df = pd.DataFrame(
    ph_rows
)


# ============================================================
# SHORTEST-GAP SUBSET VS REST
# ============================================================

short_rows = []


for analysis_name, d in [
    (
        "C1B_GEOGRAPHIC_OOF",
        c1
    ),
    (
        "PASS_HOLDOUT_OOF",
        ph
    ),
]:

    for subset_name, mask in [
        (
            "G1_SHORTEST_TWO_PASSES",
            d[
                "shortest_gap_subset"
            ].astype(bool)
        ),
        (
            "ALL_OTHER_PASSES",
            ~d[
                "shortest_gap_subset"
            ].astype(bool)
        ),
    ]:

        g = (
            d[
                mask
            ]
            .copy()
        )

        row = metric_row(
            g[
                "observed_agbd"
            ],
            g[
                "predicted_agbd"
            ],
            g[
                "mean_only_prediction"
            ],
            subset_name,
            g[
                "cv_block_id"
            ].nunique(),
        )

        row[
            "analysis"
        ] = (
            analysis_name
        )

        row[
            "gap_min_days"
        ] = float(
            g[
                "delta_days_to_NISAR"
            ].min()
        )

        row[
            "gap_median_days"
        ] = float(
            g[
                "delta_days_to_NISAR"
            ].median()
        )

        row[
            "gap_max_days"
        ] = float(
            g[
                "delta_days_to_NISAR"
            ].max()
        )

        short_rows.append(
            row
        )


short_df = pd.DataFrame(
    short_rows
)


# ============================================================
# PER-PASS PERFORMANCE FROM PASS-HOLDOUT OOF
#
# This is the cleanest temporal-pass diagnostic because each
# acquisition pass was absent from training.
# ============================================================

pass_rows = []


for source_file, g in ph.groupby(
    "source_file"
):

    row = metric_row(
        g[
            "observed_agbd"
        ],
        g[
            "predicted_agbd"
        ],
        g[
            "mean_only_prediction"
        ],
        source_file,
        g[
            "cv_block_id"
        ].nunique(),
    )

    row[
        "source_file"
    ] = source_file

    row[
        "acquisition_date"
    ] = (
        g[
            "acquisition_date_temporal"
        ].iloc[0]
        if "acquisition_date_temporal"
        in g.columns
        else
        g[
            "acquisition_date"
        ].iloc[0]
    )

    row[
        "gap_days"
    ] = float(
        g[
            "delta_days_to_NISAR"
        ].iloc[0]
    )

    row[
        "gap_stratum"
    ] = (
        g[
            "gap_stratum"
        ].iloc[0]
    )

    row[
        "low_power_n_lt_30"
    ] = (
        len(g) < 30
    )

    pass_rows.append(
        row
    )


pass_df = (
    pd.DataFrame(
        pass_rows
    )
    .sort_values(
        "gap_days"
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# PASS-LEVEL GAP VS ERROR ASSOCIATION
#
# n = 8 acquisition passes.
# Treat as descriptive because of small n and unequal pass size.
# ============================================================

assoc_rows = []


for metric in [
    "RMSE_Mg_ha",
    "MAE_Mg_ha",
    "Bias_Mg_ha",
    "delta_RMSE_model_minus_mean",
]:

    x = (
        pass_df[
            "gap_days"
        ]
        .to_numpy(float)
    )

    y = (
        pass_df[
            metric
        ]
        .to_numpy(float)
    )

    sr = spearmanr(
        x,
        y
    )

    pr = pearsonr(
        x,
        y
    )

    assoc_rows.append({
        "error_metric":
            metric,

        "n_passes":
            len(pass_df),

        "Spearman_rho":
            float(
                sr.statistic
            ),

        "Spearman_p":
            float(
                sr.pvalue
            ),

        "Pearson_r":
            float(
                pr.statistic
            ),

        "Pearson_p":
            float(
                pr.pvalue
            ),

        "interpretation":
            (
                "DESCRIPTIVE_ONLY_"
                "8_UNEQUAL_PASS_UNITS"
            ),
    })


assoc_df = pd.DataFrame(
    assoc_rows
)

boot_summary_df = pd.DataFrame(
    bootstrap_rows
)


# ============================================================
# SAVE
# ============================================================

c1_strata_df.to_csv(
    C1B_STRATA,
    index=False
)

ph_strata_df.to_csv(
    PASS_STRATA,
    index=False
)

short_df.to_csv(
    SHORTEST,
    index=False
)

pass_df.to_csv(
    PASS_METRICS,
    index=False
)

assoc_df.to_csv(
    PASS_ASSOC,
    index=False
)

boot_summary_df.to_csv(
    BOOT_SUMMARY,
    index=False
)


# ============================================================
# REPORT
# ============================================================

with REPORT.open(
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "=" * 100
        + "\n"
    )

    f.write(
        "C3 STEP 1 — TEMPORAL-GAP "
        "STRATIFIED OOF PERFORMANCE\n"
    )

    f.write(
        "=" * 100
        + "\n\n"
    )

    f.write(
        "DESIGN\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    f.write(
        "NO NEW MODEL WAS FITTED.\n"
    )

    f.write(
        "Temporal strata were frozen in C3 Step 0 "
        "using GEDI-to-NISAR gap only.\n"
    )

    f.write(
        "Primary analysis uses the already-frozen "
        "C1B geographic OOF predictions.\n"
    )

    f.write(
        "Secondary analysis uses the already-frozen "
        "C2 leave-complete-GEDI-pass-out OOF predictions.\n"
    )

    f.write(
        "Thus the temporal analysis does not select "
        "models or predictors using temporal-group outcomes.\n\n"
    )

    f.write(
        "A. C1B GEOGRAPHIC OOF BY GAP STRATUM\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    f.write(
        c1_strata_df.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )

    f.write(
        "B. LEAVE-PASS-OUT OOF BY GAP STRATUM\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    f.write(
        ph_strata_df.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )

    f.write(
        "C. SHORTEST-GAP SUBSET VS ALL OTHER PASSES\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    f.write(
        short_df.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )

    f.write(
        "D. PER-PASS LEAVE-PASS-OUT TEMPORAL METRICS\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    f.write(
        pass_df[
            [
                "gap_days",
                "gap_stratum",
                "n",
                "n_blocks",
                "low_power_n_lt_30",
                "RMSE_Mg_ha",
                "MAE_Mg_ha",
                "R2",
                "Bias_Mg_ha",
                "delta_RMSE_model_minus_mean",
                "source_file",
            ]
        ].to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )

    f.write(
        "E. PASS-LEVEL GAP–ERROR ASSOCIATION\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    f.write(
        assoc_df.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )

    f.write(
        "F. 3-KM BLOCK BOOTSTRAP BY GAP STRATUM\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    f.write(
        boot_summary_df.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )

    f.write(
        "INTERPRETATION RULES\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    f.write(
        "1. G1 contains substantially more observations "
        "than G4; stratum metrics therefore have unequal precision.\n"
    )

    f.write(
        "2. Per-pass gap associations use only eight "
        "unequal acquisition units and are descriptive.\n"
    )

    f.write(
        "3. A monotonic deterioration with larger gap would "
        "support temporal mismatch as a performance limitation.\n"
    )

    f.write(
        "4. Absence of such a trend would NOT prove temporal "
        "mismatch irrelevant because gap is confounded with "
        "acquisition date, season, geography, hydrology, "
        "and possible disturbance.\n"
    )

    f.write(
        "5. The existing June-2026 scene must therefore still "
        "be evaluated separately after this step.\n"
    )


# ============================================================
# CONSOLE
# ============================================================

print("")
print("=" * 84)
print("C3 STEP 1 COMPLETE")
print("TEMPORAL-GAP STRATIFIED OOF ANALYSIS")
print("=" * 84)

print("")
print("C1B GEOGRAPHIC OOF BY GAP STRATUM")
print(
    c1_strata_df.to_string(
        index=False
    )
)

print("")
print("PASS-HOLDOUT OOF BY GAP STRATUM")
print(
    ph_strata_df.to_string(
        index=False
    )
)

print("")
print("SHORTEST-GAP SUBSET COMPARISON")
print(
    short_df.to_string(
        index=False
    )
)

print("")
print("PER-PASS GAP ASSOCIATION")
print(
    assoc_df.to_string(
        index=False
    )
)

print("")
print("Report:")
print(REPORT)

print("")
print("NO NEW MODEL WAS FITTED.")
print("C1 AND C2 REMAIN FROZEN.")
print("C3 TEMPORAL STRATA REMAIN FROZEN.")
print("=" * 84)
