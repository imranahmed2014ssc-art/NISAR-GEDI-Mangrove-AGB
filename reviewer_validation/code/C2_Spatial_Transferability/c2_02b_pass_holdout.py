from pathlib import Path
import hashlib
import json
import warnings

import numpy as np
import pandas as pd

from scipy.stats import pearsonr, spearmanr

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
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

DATA = (
    ROOT
    / "04_Matched_Data"
    / "Predictor_Expansion"
    / "Phase8C2"
    / "Tables"
    / "GEDI_NISAR_expanded_predictors_1641.csv"
)

C2_ROOT = (
    ROOT
    / "06_Results"
    / "Reviewer_Validation"
    / "C2_Spatial_Transferability"
)

GEOM_ROOT = (
    C2_ROOT
    / "Frozen_Geometries"
    / "Pass_Holdout"
)

TRACK_MAPPING = (
    C2_ROOT
    / "C2_01_TRACK_GROUP_MAPPING.csv"
)

OUT = (
    C2_ROOT
    / "C2_02B_Pass_Holdout"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

C1B_OOF = (
    ROOT
    / "06_Results"
    / "Reviewer_Validation"
    / "C1_Independent_Validation"
    / "C1B_Full_Nested_Selection"
    / "C1B_SELECTION_AWARE_OOF.csv"
)


# ============================================================
# OUTPUTS
# ============================================================

PASS_METRICS = (
    OUT
    / "C2_02B_PASS_METRICS.csv"
)

POOLED_METRICS = (
    OUT
    / "C2_02B_POOLED_METRICS.csv"
)

OOF_FILE = (
    OUT
    / "C2_02B_PASS_OOF.csv"
)

PARAM_FILE = (
    OUT
    / "C2_02B_ET_PARAMETERS.csv"
)

BOOT_FILE = (
    OUT
    / "C2_02B_PASS_BOOTSTRAP.csv"
)

BOOT_SUMMARY = (
    OUT
    / "C2_02B_PASS_BOOTSTRAP_SUMMARY.csv"
)

MACRO_FILE = (
    OUT
    / "C2_02B_PASS_MACRO_SUMMARY.csv"
)

REPORT = (
    OUT
    / "C2_02B_PASS_HOLDOUT_REPORT.txt"
)


# ============================================================
# FROZEN MODEL
# ============================================================

FEATURES = [
    "HH_dB",
    "HV_dB",
    "HV_SD",
    "HH_SD_dB_3x3",
    "HH_MeanLin_dB_5x5",
    "HV_MeanLin_dB_5x5",
    "HH_SD_dB_5x5",
    "HV_SD_dB_5x5",
    "PolContrast_SD_dB_5x5",
    "HH_MeanLin_dB_9x9",
    "HV_MeanLin_dB_9x9",
    "HH_SD_dB_9x9",
    "HV_SD_dB_9x9",
    "PolContrast_SD_dB_9x9",
]

RESPONSE = "agbd"

INNER_FOLDS = 4
RANDOM_SEED = 42
N_SEARCH_ITER = 18
N_JOBS = 1

BOOT_REPS = 5000
BOOT_SEED = 20261028


ET_PARAMS = {
    "n_estimators": [
        300,
        500,
        700,
        900,
    ],
    "max_depth": [
        None,
        8,
        12,
        18,
        24,
    ],
    "min_samples_split": [
        2,
        5,
        10,
    ],
    "min_samples_leaf": [
        1,
        2,
        4,
        6,
    ],
    "max_features": [
        "sqrt",
        0.6,
        0.8,
        1.0,
    ],
    "bootstrap": [
        False,
        True,
    ],
}


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        for chunk in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b""
        ):
            h.update(chunk)

    return h.hexdigest()


def json_safe(v):

    if isinstance(v, np.integer):
        return int(v)

    if isinstance(v, np.floating):
        return float(v)

    if isinstance(v, tuple):
        return list(v)

    return v


def params_json(params):

    return json.dumps(
        {
            k: json_safe(v)
            for k, v
            in params.items()
        },
        sort_keys=True
    )


def metric_row(
    y,
    p,
    model,
    pass_id,
    train_n=None,
    test_n=None,
    low_power=False,
):

    y = np.asarray(
        y,
        dtype=float
    )

    p = np.asarray(
        p,
        dtype=float
    )

    rmse = float(
        np.sqrt(
            mean_squared_error(
                y,
                p
            )
        )
    )

    mae = float(
        mean_absolute_error(
            y,
            p
        )
    )

    if len(y) >= 2:

        r2 = float(
            r2_score(
                y,
                p
            )
        )

    else:

        r2 = np.nan

    bias = float(
        np.mean(
            p - y
        )
    )

    obs_mean = float(
        np.mean(y)
    )

    rrmse = (
        rmse
        / obs_mean
        * 100.0
        if obs_mean != 0
        else np.nan
    )

    if (
        len(y) >= 3
        and np.std(y) > 0
        and np.std(p) > 0
    ):

        pr = float(
            pearsonr(
                y,
                p
            ).statistic
        )

        sr = float(
            spearmanr(
                y,
                p
            ).statistic
        )

    else:

        pr = np.nan
        sr = np.nan

    obs_sd = (
        float(
            np.std(
                y,
                ddof=1
            )
        )
        if len(y) > 1
        else np.nan
    )

    pred_sd = (
        float(
            np.std(
                p,
                ddof=1
            )
        )
        if len(p) > 1
        else np.nan
    )

    ratio = (
        pred_sd / obs_sd
        if (
            np.isfinite(obs_sd)
            and obs_sd > 0
        )
        else np.nan
    )

    return {
        "model":
            model,

        "pass_id":
            pass_id,

        "n":
            len(y),

        "train_n":
            train_n,

        "test_n":
            test_n,

        "low_power_n_lt_30":
            bool(low_power),

        "RMSE_Mg_ha":
            rmse,

        "MAE_Mg_ha":
            mae,

        "R2":
            r2,

        "Bias_Mg_ha":
            bias,

        "rRMSE_percent":
            rrmse,

        "Pearson_r":
            pr,

        "Spearman_rho":
            sr,

        "observed_mean":
            obs_mean,

        "predicted_mean":
            float(
                np.mean(p)
            ),

        "observed_sd":
            obs_sd,

        "predicted_sd":
            pred_sd,

        "pred_to_obs_SD_ratio":
            ratio,
    }


def load_membership(
    path,
    master
):

    membership = pd.read_csv(
        path
    )

    ids = set(
        membership[
            "shot_number"
        ]
    )

    out = (
        master[
            master[
                "shot_number"
            ].isin(ids)
        ]
        .copy()
    )

    if len(out) != len(
        membership
    ):

        raise RuntimeError(
            f"Membership mismatch: "
            f"{path}"
        )

    return out


def fit_et(
    train,
    test,
):

    X_train = (
        train[
            FEATURES
        ]
        .to_numpy(float)
    )

    y_train = (
        train[
            RESPONSE
        ]
        .to_numpy(float)
    )

    X_test = (
        test[
            FEATURES
        ]
        .to_numpy(float)
    )

    y_test = (
        test[
            RESPONSE
        ]
        .to_numpy(float)
    )

    groups = (
        train[
            "cv_block_id"
        ]
        .to_numpy()
    )

    n_groups = int(
        pd.Series(
            groups
        ).nunique()
    )

    if n_groups < INNER_FOLDS:

        raise RuntimeError(
            "Too few training blocks "
            "for GroupKFold."
        )

    base = ExtraTreesRegressor(
        random_state=
            RANDOM_SEED,

        n_jobs=
            N_JOBS,

        criterion=
            "squared_error",
    )

    cv = GroupKFold(
        n_splits=
            INNER_FOLDS
    )

    search = RandomizedSearchCV(
        estimator=
            base,

        param_distributions=
            ET_PARAMS,

        n_iter=
            N_SEARCH_ITER,

        scoring=
            "neg_root_mean_squared_error",

        cv=
            cv,

        random_state=
            RANDOM_SEED,

        n_jobs=
            N_JOBS,

        refit=
            True,

        return_train_score=
            False,

        verbose=
            0,

        error_score=
            "raise",
    )

    search.fit(
        X_train,
        y_train,
        groups=groups,
    )

    pred = (
        search
        .best_estimator_
        .predict(
            X_test
        )
    )

    return (
        pred,
        y_test,
        search.best_params_,
        float(
            -search.best_score_
        ),
        n_groups,
    )


# ============================================================
# LOAD INPUTS
# ============================================================

master = pd.read_csv(
    DATA
)

mapping = pd.read_csv(
    TRACK_MAPPING
)

if len(master) != 1641:

    raise RuntimeError(
        f"Expected 1641 rows; "
        f"found {len(master)}."
    )

if (
    master[
        "shot_number"
    ].nunique()
    != 1641
):

    raise RuntimeError(
        "shot_number not unique."
    )

if len(mapping) != 8:

    raise RuntimeError(
        f"Expected 8 passes; "
        f"found {len(mapping)}."
    )


# ============================================================
# LEAVE-ONE-PASS-OUT LOOP
# ============================================================

metric_rows = []
param_rows = []
oof_parts = []


for _, info in mapping.iterrows():

    pass_id = str(
        info[
            "pass_id"
        ]
    )

    source_file = str(
        info[
            "source_file"
        ]
    )

    acquisition_date = str(
        info[
            "acquisition_date"
        ]
    )

    fold_dir = (
        GEOM_ROOT
        / pass_id
    )

    train_path = (
        fold_dir
        / "TRAIN_MEMBERSHIP.csv"
    )

    test_path = (
        fold_dir
        / "TEST_MEMBERSHIP.csv"
    )

    train = load_membership(
        train_path,
        master
    )

    test = load_membership(
        test_path,
        master
    )

    if (
        test[
            "source_file"
        ]
        .nunique()
        != 1
    ):

        raise RuntimeError(
            f"{pass_id}: "
            "test contains multiple "
            "source files."
        )

    if (
        str(
            test[
                "source_file"
            ].iloc[0]
        )
        != source_file
    ):

        raise RuntimeError(
            f"{pass_id}: "
            "source-file mismatch."
        )

    if (
        source_file
        in set(
            train[
                "source_file"
            ].astype(str)
        )
    ):

        raise RuntimeError(
            f"{pass_id}: held-out "
            "source file found in training."
        )

    if (
        set(
            train[
                "shot_number"
            ]
        )
        .intersection(
            set(
                test[
                    "shot_number"
                ]
            )
        )
    ):

        raise RuntimeError(
            f"{pass_id}: "
            "train/test shot overlap."
        )

    low_power = (
        len(test) < 30
    )

    print("")
    print(
        "=" * 76
    )

    print(
        f"{pass_id} — "
        f"{acquisition_date}"
    )

    print(
        f"test n={len(test)}, "
        f"train n={len(train)}, "
        f"test blocks="
        f"{test['cv_block_id'].nunique()}"
    )

    print(
        "=" * 76
    )

    (
        pred,
        y_test,
        best_params,
        inner_rmse,
        train_blocks,
    ) = fit_et(
        train,
        test,
    )

    mean_value = float(
        train[
            RESPONSE
        ].mean()
    )

    mean_pred = np.full(
        len(test),
        mean_value,
        dtype=float,
    )

    model_metric = metric_row(
        y_test,
        pred,
        "ET14P_PASS_HOLDOUT",
        pass_id,
        train_n=
            len(train),
        test_n=
            len(test),
        low_power=
            low_power,
    )

    mean_metric = metric_row(
        y_test,
        mean_pred,
        "MEAN_ONLY",
        pass_id,
        train_n=
            len(train),
        test_n=
            len(test),
        low_power=
            low_power,
    )

    metric_rows.extend(
        [
            model_metric,
            mean_metric,
        ]
    )

    param_rows.append({
        "pass_id":
            pass_id,

        "source_file":
            source_file,

        "acquisition_date":
            acquisition_date,

        "train_n":
            len(train),

        "train_blocks":
            train_blocks,

        "test_n":
            len(test),

        "test_blocks":
            int(
                test[
                    "cv_block_id"
                ].nunique()
            ),

        "low_power_n_lt_30":
            low_power,

        "best_inner_RMSE":
            inner_rmse,

        "best_params_json":
            params_json(
                best_params
            ),
    })

    oof_parts.append(
        pd.DataFrame({

            "shot_number":
                test[
                    "shot_number"
                ].to_numpy(),

            "cv_block_id":
                test[
                    "cv_block_id"
                ].to_numpy(),

            "pass_id":
                pass_id,

            "source_file":
                source_file,

            "acquisition_date":
                acquisition_date,

            "low_power_n_lt_30":
                low_power,

            "observed_agbd":
                y_test,

            "predicted_agbd":
                pred,

            "mean_only_prediction":
                mean_pred,
        })
    )

    print(
        f"inner RMSE="
        f"{inner_rmse:.3f}"
    )

    print(
        f"outer RMSE="
        f"{model_metric['RMSE_Mg_ha']:.3f}, "
        f"MAE="
        f"{model_metric['MAE_Mg_ha']:.3f}, "
        f"R2="
        f"{model_metric['R2']:.4f}"
    )

    if low_power:

        print(
            "LOW-POWER PASS: "
            "n < 30; individual "
            "R2/correlation should "
            "not be interpreted alone."
        )


# ============================================================
# COMBINE
# ============================================================

pass_metrics_df = pd.DataFrame(
    metric_rows
)

param_df = pd.DataFrame(
    param_rows
)

oof = pd.concat(
    oof_parts,
    ignore_index=True
)


# ============================================================
# OOF INTEGRITY
# ============================================================

if len(oof) != 1641:

    raise RuntimeError(
        f"Expected 1641 OOF rows; "
        f"found {len(oof)}."
    )

if (
    oof[
        "shot_number"
    ].nunique()
    != 1641
):

    raise RuntimeError(
        "Pass-holdout OOF shots "
        "are not unique."
    )

if (
    set(
        oof[
            "shot_number"
        ].astype(str)
    )
    !=
    set(
        master[
            "shot_number"
        ].astype(str)
    )
):

    raise RuntimeError(
        "Pass-holdout OOF shot set "
        "does not equal frozen master."
    )


# ============================================================
# POOLED METRICS
# ============================================================

pooled_model = metric_row(
    oof[
        "observed_agbd"
    ],
    oof[
        "predicted_agbd"
    ],
    "ET14P_PASS_HOLDOUT",
    "POOLED",
    train_n=np.nan,
    test_n=1641,
    low_power=False,
)

pooled_mean = metric_row(
    oof[
        "observed_agbd"
    ],
    oof[
        "mean_only_prediction"
    ],
    "MEAN_ONLY",
    "POOLED",
    train_n=np.nan,
    test_n=1641,
    low_power=False,
)

pooled_df = pd.DataFrame(
    [
        pooled_model,
        pooled_mean,
    ]
)


# ============================================================
# PASS-MACRO SUMMARIES
# ============================================================

model_passes = (
    pass_metrics_df[
        pass_metrics_df[
            "model"
        ]
        == "ET14P_PASS_HOLDOUT"
    ]
    .copy()
)

baseline_passes = (
    pass_metrics_df[
        pass_metrics_df[
            "model"
        ]
        == "MEAN_ONLY"
    ]
    .copy()
)


macro_rows = []

for label, subset in [
    (
        "all_8_passes",
        model_passes
    ),
    (
        "passes_n_ge_30",
        model_passes[
            ~model_passes[
                "low_power_n_lt_30"
            ]
        ]
    ),
]:

    macro_rows.append({
        "model":
            "ET14P_PASS_HOLDOUT",

        "subset":
            label,

        "n_passes":
            len(subset),

        "mean_pass_RMSE":
            float(
                subset[
                    "RMSE_Mg_ha"
                ].mean()
            ),

        "median_pass_RMSE":
            float(
                subset[
                    "RMSE_Mg_ha"
                ].median()
            ),

        "mean_pass_MAE":
            float(
                subset[
                    "MAE_Mg_ha"
                ].mean()
            ),

        "mean_pass_bias":
            float(
                subset[
                    "Bias_Mg_ha"
                ].mean()
            ),

        "mean_pass_R2":
            float(
                subset[
                    "R2"
                ].mean()
            ),
    })


for label, subset in [
    (
        "all_8_passes",
        baseline_passes
    ),
    (
        "passes_n_ge_30",
        baseline_passes[
            ~baseline_passes[
                "low_power_n_lt_30"
            ]
        ]
    ),
]:

    macro_rows.append({
        "model":
            "MEAN_ONLY",

        "subset":
            label,

        "n_passes":
            len(subset),

        "mean_pass_RMSE":
            float(
                subset[
                    "RMSE_Mg_ha"
                ].mean()
            ),

        "median_pass_RMSE":
            float(
                subset[
                    "RMSE_Mg_ha"
                ].median()
            ),

        "mean_pass_MAE":
            float(
                subset[
                    "MAE_Mg_ha"
                ].mean()
            ),

        "mean_pass_bias":
            float(
                subset[
                    "Bias_Mg_ha"
                ].mean()
            ),

        "mean_pass_R2":
            float(
                subset[
                    "R2"
                ].mean()
            ),
    })


macro_df = pd.DataFrame(
    macro_rows
)


# ============================================================
# PASS-CLUSTER BOOTSTRAP
#
# Resampling unit = whole GEDI acquisition pass.
# There are only 8 nominal passes, so this uncertainty
# estimate is explicitly diagnostic and should be interpreted
# cautiously.
# ============================================================

pass_parts = []

for pass_id, g in oof.groupby(
    "pass_id"
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

    m = (
        g[
            "mean_only_prediction"
        ]
        .to_numpy(float)
    )

    pass_parts.append({
        "pass_id":
            pass_id,

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
                    (m - y) ** 2
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
                        m - y
                    )
                )
            ),
    })


pass_boot_source = pd.DataFrame(
    pass_parts
)

if len(pass_boot_source) != 8:

    raise RuntimeError(
        f"Expected 8 pass units; "
        f"found {len(pass_boot_source)}."
    )


rng = np.random.default_rng(
    BOOT_SEED
)

boot_rows = []

P = len(
    pass_boot_source
)

for rep in range(
    1,
    BOOT_REPS + 1
):

    idx = rng.integers(
        0,
        P,
        size=P,
    )

    s = pass_boot_source.iloc[
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

    sse_model = float(
        s[
            "sse_model"
        ].sum()
    )

    sse_mean = float(
        s[
            "sse_mean"
        ].sum()
    )

    sae_model = float(
        s[
            "sae_model"
        ].sum()
    )

    sae_mean = float(
        s[
            "sae_mean"
        ].sum()
    )

    rmse_model = np.sqrt(
        sse_model / n
    )

    rmse_mean = np.sqrt(
        sse_mean / n
    )

    mae_model = (
        sae_model / n
    )

    mae_mean = (
        sae_mean / n
    )

    sst = (
        sy2
        - sy ** 2 / n
    )

    r2_model = (
        1.0
        - sse_model / sst
        if sst > 0
        else np.nan
    )

    r2_mean = (
        1.0
        - sse_mean / sst
        if sst > 0
        else np.nan
    )

    boot_rows.append({
        "replicate":
            rep,

        "model_RMSE":
            rmse_model,

        "mean_only_RMSE":
            rmse_mean,

        "delta_RMSE_model_minus_mean":
            (
                rmse_model
                - rmse_mean
            ),

        "model_MAE":
            mae_model,

        "mean_only_MAE":
            mae_mean,

        "delta_MAE_model_minus_mean":
            (
                mae_model
                - mae_mean
            ),

        "model_R2":
            r2_model,

        "mean_only_R2":
            r2_mean,

        "delta_R2_model_minus_mean":
            (
                r2_model
                - r2_mean
            ),
    })


boot_df = pd.DataFrame(
    boot_rows
)


# ============================================================
# BOOTSTRAP SUMMARY
# ============================================================

summary_rows = []

for col in [
    "model_RMSE",
    "model_MAE",
    "model_R2",
    "delta_RMSE_model_minus_mean",
    "delta_MAE_model_minus_mean",
    "delta_R2_model_minus_mean",
]:

    v = (
        boot_df[
            col
        ]
        .to_numpy(float)
    )

    summary_rows.append({
        "metric":
            col,

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

        "nominal_pass_units":
            8,
    })


boot_summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# OPTIONAL REFERENCE TO C1B
# ============================================================

reference_lines = []

if C1B_OOF.exists():

    c1b = pd.read_csv(
        C1B_OOF
    )

    if (
        len(c1b) == 1641
        and
        "predicted_agbd"
        in c1b.columns
    ):

        ref_rmse = float(
            np.sqrt(
                mean_squared_error(
                    c1b[
                        "observed_agbd"
                    ],
                    c1b[
                        "predicted_agbd"
                    ],
                )
            )
        )

        reference_lines.append(
            "C1B contiguous-geographic "
            f"500-m pooled RMSE: "
            f"{ref_rmse:.6f} Mg/ha"
        )

        reference_lines.append(
            "Difference in pooled RMSE "
            "(pass-holdout minus C1B): "
            f"{pooled_model['RMSE_Mg_ha'] - ref_rmse:.6f} Mg/ha"
        )


# ============================================================
# SAVE
# ============================================================

pass_metrics_df.to_csv(
    PASS_METRICS,
    index=False
)

pooled_df.to_csv(
    POOLED_METRICS,
    index=False
)

oof.to_csv(
    OOF_FILE,
    index=False
)

param_df.to_csv(
    PARAM_FILE,
    index=False
)

macro_df.to_csv(
    MACRO_FILE,
    index=False
)

boot_df.to_csv(
    BOOT_FILE,
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
        "=" * 96
        + "\n"
    )

    f.write(
        "C2 STEP 2B — LEAVE-ONE-COMPLETE-GEDI-PASS-OUT VALIDATION\n"
    )

    f.write(
        "=" * 96
        + "\n\n"
    )

    f.write(
        "DESIGN\n"
    )

    f.write(
        "-" * 96
        + "\n"
    )

    f.write(
        "Track-group proxy: complete GEDI L4A source_file/acquisition pass.\n"
    )

    f.write(
        "All observations from one acquisition pass were withheld together.\n"
    )

    f.write(
        "The held-out source file was absent from the active training data.\n"
    )

    f.write(
        "A 500-m spatial exclusion buffer was additionally applied to training observations.\n"
    )

    f.write(
        "Frozen model family: Extra Trees.\n"
    )

    f.write(
        "Frozen predictor family: full 14P multiscale predictor set.\n"
    )

    f.write(
        "Hyperparameters were retuned only inside each active training sample using\n"
        "four-fold GroupKFold grouped by 3-km cv_block_id.\n"
    )

    f.write(
        "No C2 model-family or predictor-family selection occurred.\n\n"
    )

    f.write(
        "INPUT SHA256\n"
    )

    f.write(
        "-" * 96
        + "\n"
    )

    f.write(
        sha256_file(
            DATA
        )
        + "\n\n"
    )

    f.write(
        "POOLED LEAVE-PASS-OUT PERFORMANCE\n"
    )

    f.write(
        "-" * 96
        + "\n"
    )

    f.write(
        pooled_df.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )

    f.write(
        "PER-PASS PERFORMANCE\n"
    )

    f.write(
        "-" * 96
        + "\n"
    )

    f.write(
        pass_metrics_df.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )

    f.write(
        "PASS-MACRO DIAGNOSTICS\n"
    )

    f.write(
        "-" * 96
        + "\n"
    )

    f.write(
        macro_df.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )

    f.write(
        "SELECTED ET HYPERPARAMETERS\n"
    )

    f.write(
        "-" * 96
        + "\n"
    )

    f.write(
        param_df.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )

    f.write(
        "8-PASS CLUSTER BOOTSTRAP\n"
    )

    f.write(
        "-" * 96
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
        "IMPORTANT UNCERTAINTY NOTE\n"
    )

    f.write(
        "-" * 96
        + "\n"
    )

    f.write(
        "Only eight nominal acquisition-pass units exist, and pass sizes are highly unequal.\n"
    )

    f.write(
        "Therefore the pass-level bootstrap is a diagnostic uncertainty analysis rather than\n"
        "evidence of broad external transferability. Individual pass results with n < 30 are\n"
        "explicitly flagged as low-power and must not be interpreted in isolation.\n\n"
    )

    if reference_lines:

        f.write(
            "REFERENCE TO C1B CONTIGUOUS-GEOGRAPHIC VALIDATION\n"
        )

        f.write(
            "-" * 96
            + "\n"
        )

        for line in reference_lines:

            f.write(
                line
                + "\n"
            )

        f.write(
            "\n"
        )

    f.write(
        "INTERPRETATION BOUNDARY\n"
    )

    f.write(
        "-" * 96
        + "\n"
    )

    f.write(
        "This is a within-study leave-acquisition-pass-out transferability test.\n"
    )

    f.write(
        "It does not constitute external-site validation.\n"
    )


# ============================================================
# CONSOLE
# ============================================================

print("")
print(
    "=" * 80
)

print(
    "C2 STEP 2B COMPLETE"
)

print(
    "LEAVE-ONE-COMPLETE-GEDI-PASS-OUT"
)

print(
    "=" * 80
)

print("")
print(
    "POOLED PERFORMANCE"
)

print(
    pooled_df.to_string(
        index=False
    )
)

print("")
print(
    "PER-PASS ET14P PERFORMANCE"
)

print(
    model_passes[
        [
            "pass_id",
            "n",
            "low_power_n_lt_30",
            "RMSE_Mg_ha",
            "MAE_Mg_ha",
            "R2",
            "Bias_Mg_ha",
            "Pearson_r",
            "Spearman_rho",
        ]
    ].to_string(
        index=False
    )
)

print("")
print(
    "PASS-MACRO DIAGNOSTICS"
)

print(
    macro_df.to_string(
        index=False
    )
)

print("")
print(
    "PASS-CLUSTER BOOTSTRAP"
)

print(
    boot_summary_df.to_string(
        index=False
    )
)

if reference_lines:

    print("")

    for line in reference_lines:

        print(line)

print("")
print(
    "OOF rows:",
    len(oof)
)

print(
    "Unique OOF shots:",
    oof[
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
    "C1 REMAINS FROZEN."
)

print(
    "C2 GEOMETRIES REMAIN FROZEN."
)

print(
    "=" * 80
)
