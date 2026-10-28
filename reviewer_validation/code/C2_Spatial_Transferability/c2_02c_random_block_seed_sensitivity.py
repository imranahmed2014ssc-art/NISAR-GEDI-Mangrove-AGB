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
    / "Random_Block_Assignments"
)

HASH_FILE = (
    C2_ROOT
    / "C2_01B_ASSIGNMENT_HASHES.csv"
)

C1B_METRICS = (
    ROOT
    / "06_Results"
    / "Reviewer_Validation"
    / "C1_Independent_Validation"
    / "C1B_Full_Nested_Selection"
    / "C1B_POOLED_METRICS.csv"
)

OUT = (
    C2_ROOT
    / "C2_02C_Random_Block_Seeds"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# OUTPUTS
# ============================================================

FOLD_FILE = (
    OUT
    / "C2_02C_FOLD_METRICS.csv"
)

SEED_FILE = (
    OUT
    / "C2_02C_SEED_POOLED_METRICS.csv"
)

DELTA_FILE = (
    OUT
    / "C2_02C_SEED_DELTAS.csv"
)

SUMMARY_FILE = (
    OUT
    / "C2_02C_SEED_VARIABILITY_SUMMARY.csv"
)

PARAM_FILE = (
    OUT
    / "C2_02C_ET_PARAMETERS.csv"
)

OOF_FILE = (
    OUT
    / "C2_02C_OOF_ALL_SEEDS.csv"
)

REPORT = (
    OUT
    / "C2_02C_RANDOM_BLOCK_SEED_REPORT.txt"
)


# ============================================================
# FROZEN MODEL / DESIGN
# ============================================================

SEEDS = [
    20261001,
    20261002,
    20261003,
    20261004,
    20261005,
    20261006,
    20261007,
    20261008,
    20261009,
    20261010,
]

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
RANDOM_SEED_MODEL = 42
N_SEARCH_ITER = 18
N_JOBS = 1


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


def safe_value(v):

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
            k: safe_value(v)
            for k, v
            in params.items()
        },
        sort_keys=True
    )


def metric_row(
    y,
    p,
    model,
    seed,
    fold,
    train_n=None,
    test_n=None,
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

    r2 = float(
        r2_score(
            y,
            p
        )
    )

    bias = float(
        np.mean(
            p - y
        )
    )

    obs_mean = float(
        np.mean(y)
    )

    rrmse = (
        rmse / obs_mean * 100.0
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

    return {
        "model":
            model,

        "seed":
            seed,

        "fold":
            fold,

        "n":
            len(y),

        "train_n":
            train_n,

        "test_n":
            test_n,

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
            (
                pred_sd / obs_sd
                if obs_sd > 0
                else np.nan
            ),
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
            f"Membership mismatch: {path}"
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
            "Insufficient spatial blocks "
            "for inner GroupKFold."
        )

    base = ExtraTreesRegressor(
        random_state=
            RANDOM_SEED_MODEL,

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
            RANDOM_SEED_MODEL,

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

hashes = pd.read_csv(
    HASH_FILE
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
        "shot_number is not unique."
    )

if len(hashes) != 10:

    raise RuntimeError(
        f"Expected 10 frozen "
        f"assignment hashes; "
        f"found {len(hashes)}."
    )

if (
    hashes[
        "assignment_sha256"
    ].nunique()
    != 10
):

    raise RuntimeError(
        "Frozen randomized assignments "
        "are not all distinct."
    )

if (
    set(
        hashes[
            "seed"
        ].astype(int)
    )
    != set(SEEDS)
):

    raise RuntimeError(
        "Seed list differs from "
        "frozen protocol."
    )


# ============================================================
# RUN 10 SEEDS x 5 FOLDS
# ============================================================

fold_rows = []
param_rows = []
oof_parts = []


for repeat_index, seed in enumerate(
    SEEDS,
    start=1
):

    print("")
    print(
        "=" * 80
    )

    print(
        f"C2 RANDOM-BLOCK SEED "
        f"{seed} "
        f"({repeat_index}/10)"
    )

    print(
        "=" * 80
    )

    seed_dir = (
        GEOM_ROOT
        / f"Seed_{seed}"
    )

    for fold in range(
        1,
        6
    ):

        print(
            f"  Fold {fold}...",
            flush=True
        )

        fold_dir = (
            seed_dir
            / f"Fold_{fold}"
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

        overlap = (
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
        )

        if overlap:

            raise RuntimeError(
                f"Seed {seed}, fold {fold}: "
                f"train/test overlap."
            )

        train_blocks_set = set(
            train[
                "cv_block_id"
            ]
        )

        test_blocks_set = set(
            test[
                "cv_block_id"
            ]
        )

        if (
            train_blocks_set
            .intersection(
                test_blocks_set
            )
        ):

            raise RuntimeError(
                f"Seed {seed}, fold {fold}: "
                f"spatial block split across "
                f"train and test."
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
            "ET14P",
            seed,
            fold,
            train_n=
                len(train),
            test_n=
                len(test),
        )

        baseline_metric = metric_row(
            y_test,
            mean_pred,
            "MEAN_ONLY",
            seed,
            fold,
            train_n=
                len(train),
            test_n=
                len(test),
        )

        fold_rows.extend(
            [
                model_metric,
                baseline_metric,
            ]
        )

        param_rows.append({
            "repeat_index":
                repeat_index,

            "seed":
                seed,

            "fold":
                fold,

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

                "repeat_index":
                    repeat_index,

                "seed":
                    seed,

                "fold":
                    fold,

                "observed_agbd":
                    y_test,

                "predicted_agbd":
                    pred,

                "mean_only_prediction":
                    mean_pred,
            })
        )

        print(
            f"    train={len(train):4d}, "
            f"train blocks={train_blocks:2d}, "
            f"test={len(test):3d}, "
            f"inner RMSE={inner_rmse:.3f}, "
            f"outer RMSE="
            f"{model_metric['RMSE_Mg_ha']:.3f}, "
            f"R2="
            f"{model_metric['R2']:.4f}"
        )


# ============================================================
# COMBINE
# ============================================================

fold_df = pd.DataFrame(
    fold_rows
)

param_df = pd.DataFrame(
    param_rows
)

oof = pd.concat(
    oof_parts,
    ignore_index=True
)


# ============================================================
# INTEGRITY: EVERY SEED MUST PROVIDE COMPLETE OOF COVERAGE
# ============================================================

for seed in SEEDS:

    g = (
        oof[
            oof[
                "seed"
            ] == seed
        ]
    )

    if len(g) != 1641:

        raise RuntimeError(
            f"Seed {seed}: "
            f"expected 1641 OOF rows; "
            f"found {len(g)}."
        )

    if (
        g[
            "shot_number"
        ].nunique()
        != 1641
    ):

        raise RuntimeError(
            f"Seed {seed}: "
            f"OOF shots not unique."
        )

    if (
        set(
            g[
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
            f"Seed {seed}: "
            f"OOF shot set differs "
            f"from frozen master."
        )


# ============================================================
# POOLED PERFORMANCE FOR EACH SEED
# ============================================================

seed_rows = []


for seed in SEEDS:

    g = (
        oof[
            oof[
                "seed"
            ] == seed
        ]
        .copy()
    )

    seed_rows.append(
        metric_row(
            g[
                "observed_agbd"
            ],
            g[
                "predicted_agbd"
            ],
            "ET14P",
            seed,
            "POOLED",
            train_n=np.nan,
            test_n=1641,
        )
    )

    seed_rows.append(
        metric_row(
            g[
                "observed_agbd"
            ],
            g[
                "mean_only_prediction"
            ],
            "MEAN_ONLY",
            seed,
            "POOLED",
            train_n=np.nan,
            test_n=1641,
        )
    )


seed_metrics = pd.DataFrame(
    seed_rows
)


# ============================================================
# ET MINUS BASELINE DELTA BY SEED
# ============================================================

delta_rows = []


for seed in SEEDS:

    m = (
        seed_metrics[
            (
                seed_metrics[
                    "seed"
                ] == seed
            )
            &
            (
                seed_metrics[
                    "model"
                ] == "ET14P"
            )
        ]
        .iloc[0]
    )

    b = (
        seed_metrics[
            (
                seed_metrics[
                    "seed"
                ] == seed
            )
            &
            (
                seed_metrics[
                    "model"
                ] == "MEAN_ONLY"
            )
        ]
        .iloc[0]
    )

    delta_rows.append({
        "seed":
            seed,

        "ET_RMSE":
            m[
                "RMSE_Mg_ha"
            ],

        "baseline_RMSE":
            b[
                "RMSE_Mg_ha"
            ],

        "delta_RMSE_ET_minus_mean":
            (
                m[
                    "RMSE_Mg_ha"
                ]
                -
                b[
                    "RMSE_Mg_ha"
                ]
            ),

        "ET_MAE":
            m[
                "MAE_Mg_ha"
            ],

        "baseline_MAE":
            b[
                "MAE_Mg_ha"
            ],

        "delta_MAE_ET_minus_mean":
            (
                m[
                    "MAE_Mg_ha"
                ]
                -
                b[
                    "MAE_Mg_ha"
                ]
            ),

        "ET_R2":
            m[
                "R2"
            ],

        "baseline_R2":
            b[
                "R2"
            ],

        "delta_R2_ET_minus_mean":
            (
                m[
                    "R2"
                ]
                -
                b[
                    "R2"
                ]
            ),

        "ET_Pearson_r":
            m[
                "Pearson_r"
            ],

        "ET_Spearman_rho":
            m[
                "Spearman_rho"
            ],

        "ET_SD_ratio":
            m[
                "pred_to_obs_SD_ratio"
            ],
    })


delta_df = pd.DataFrame(
    delta_rows
)


# ============================================================
# DESCRIPTIVE VARIABILITY ACROSS 10 FROZEN SEEDS
#
# These are 10 validation-design realizations.
# Quantiles are descriptive, not confidence intervals.
# ============================================================

summary_rows = []

summary_columns = [
    "ET_RMSE",
    "ET_MAE",
    "ET_R2",
    "ET_Pearson_r",
    "ET_Spearman_rho",
    "ET_SD_ratio",
    "delta_RMSE_ET_minus_mean",
    "delta_MAE_ET_minus_mean",
    "delta_R2_ET_minus_mean",
]


for col in summary_columns:

    v = (
        delta_df[
            col
        ]
        .to_numpy(float)
    )

    summary_rows.append({
        "metric":
            col,

        "mean_across_10_seeds":
            float(
                np.mean(v)
            ),

        "sd_across_10_seeds":
            float(
                np.std(
                    v,
                    ddof=1
                )
            ),

        "median":
            float(
                np.median(v)
            ),

        "min":
            float(
                np.min(v)
            ),

        "max":
            float(
                np.max(v)
            ),

        "descriptive_q025":
            float(
                np.quantile(
                    v,
                    0.025
                )
            ),

        "descriptive_q975":
            float(
                np.quantile(
                    v,
                    0.975
                )
            ),

        "n_seeds":
            len(v),
    })


summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# C1B REFERENCE
# ============================================================

reference_lines = []

if C1B_METRICS.exists():

    c1b = pd.read_csv(
        C1B_METRICS
    )

    ref = (
        c1b[
            c1b[
                "model"
            ]
            == "C1B_NESTED_SELECTED"
        ]
    )

    if len(ref) == 1:

        ref_rmse = float(
            ref.iloc[0][
                "RMSE_Mg_ha"
            ]
        )

        ref_r2 = float(
            ref.iloc[0][
                "R2"
            ]
        )

        reference_lines.append(
            f"C1B contiguous-geographic "
            f"RMSE: {ref_rmse:.6f} Mg/ha"
        )

        reference_lines.append(
            f"C1B contiguous-geographic "
            f"R2: {ref_r2:.6f}"
        )

        reference_lines.append(
            "Mean randomized-block RMSE "
            "minus C1B contiguous RMSE: "
            f"{delta_df['ET_RMSE'].mean() - ref_rmse:.6f} Mg/ha"
        )

        reference_lines.append(
            "Randomized seeds with RMSE "
            "lower than C1B: "
            f"{int((delta_df['ET_RMSE'] < ref_rmse).sum())}/10"
        )


# ============================================================
# CONSISTENCY COUNTS
# ============================================================

n_better_rmse = int(
    (
        delta_df[
            "delta_RMSE_ET_minus_mean"
        ] < 0
    ).sum()
)

n_better_mae = int(
    (
        delta_df[
            "delta_MAE_ET_minus_mean"
        ] < 0
    ).sum()
)

n_better_r2 = int(
    (
        delta_df[
            "delta_R2_ET_minus_mean"
        ] > 0
    ).sum()
)


# ============================================================
# SAVE
# ============================================================

fold_df.to_csv(
    FOLD_FILE,
    index=False
)

seed_metrics.to_csv(
    SEED_FILE,
    index=False
)

delta_df.to_csv(
    DELTA_FILE,
    index=False
)

summary_df.to_csv(
    SUMMARY_FILE,
    index=False
)

param_df.to_csv(
    PARAM_FILE,
    index=False
)

oof.to_csv(
    OOF_FILE,
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
        "=" * 98
        + "\n"
    )

    f.write(
        "C2 STEP 2C — REPEATED RANDOMIZED "
        "SPATIAL-BLOCK ASSIGNMENT SENSITIVITY\n"
    )

    f.write(
        "=" * 98
        + "\n\n"
    )

    f.write(
        "DESIGN\n"
    )

    f.write(
        "-" * 98
        + "\n"
    )

    f.write(
        "Ten block-to-fold assignments were frozen "
        "before any C2 model performance was observed.\n"
    )

    f.write(
        "All 54 pre-existing 3-km spatial blocks remained "
        "indivisible units.\n"
    )

    f.write(
        "Each random assignment used five outer folds and "
        "a 500-m training exclusion buffer.\n"
    )

    f.write(
        "No AGBD or predictor values were used when assigning "
        "blocks to folds.\n"
    )

    f.write(
        "Frozen model family: Extra Trees.\n"
    )

    f.write(
        "Frozen predictor family: full 14P multiscale set.\n"
    )

    f.write(
        "ET hyperparameters were retuned only within each active "
        "training sample using four-fold GroupKFold by cv_block_id.\n"
    )

    f.write(
        "No model-family or feature-family selection occurred.\n\n"
    )

    f.write(
        "INPUT SHA256\n"
    )

    f.write(
        "-" * 98
        + "\n"
    )

    f.write(
        sha256_file(
            DATA
        )
        + "\n\n"
    )

    f.write(
        "FROZEN ASSIGNMENT HASHES\n"
    )

    f.write(
        "-" * 98
        + "\n"
    )

    f.write(
        hashes.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )

    f.write(
        "POOLED PERFORMANCE BY SEED\n"
    )

    f.write(
        "-" * 98
        + "\n"
    )

    f.write(
        seed_metrics.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )

    f.write(
        "ET14P MINUS MEAN-ONLY BY SEED\n"
    )

    f.write(
        "-" * 98
        + "\n"
    )

    f.write(
        delta_df.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )

    f.write(
        "DESCRIPTIVE VARIABILITY ACROSS TEN "
        "FROZEN ASSIGNMENTS\n"
    )

    f.write(
        "-" * 98
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
        "CONSISTENCY COUNTS\n"
    )

    f.write(
        "-" * 98
        + "\n"
    )

    f.write(
        f"ET14P lower RMSE than fold-specific mean baseline: "
        f"{n_better_rmse}/10 seeds\n"
    )

    f.write(
        f"ET14P lower MAE than fold-specific mean baseline: "
        f"{n_better_mae}/10 seeds\n"
    )

    f.write(
        f"ET14P higher R2 than fold-specific mean baseline: "
        f"{n_better_r2}/10 seeds\n\n"
    )

    if reference_lines:

        f.write(
            "REFERENCE TO CONTIGUOUS-GEOGRAPHIC C1B\n"
        )

        f.write(
            "-" * 98
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
        "-" * 98
        + "\n"
    )

    f.write(
        "The ten seed-level results quantify sensitivity to "
        "block-to-fold allocation.\n"
    )

    f.write(
        "They are repeated validation-design realizations over "
        "the same 1,641 observations, not ten independent datasets.\n"
    )

    f.write(
        "The descriptive seed quantiles are therefore not presented "
        "as inferential confidence intervals.\n"
    )


# ============================================================
# CONSOLE
# ============================================================

print("")
print(
    "=" * 80
)

print(
    "C2 STEP 2C COMPLETE"
)

print(
    "RANDOMIZED BLOCK-SEED SENSITIVITY"
)

print(
    "=" * 80
)

print("")
print(
    "ET14P VS MEAN-ONLY BY SEED"
)

print(
    delta_df.to_string(
        index=False
    )
)

print("")
print(
    "VARIABILITY ACROSS 10 SEEDS"
)

print(
    summary_df.to_string(
        index=False
    )
)

print("")
print(
    "Consistency:"
)

print(
    f"  RMSE better than baseline: "
    f"{n_better_rmse}/10"
)

print(
    f"  MAE better than baseline: "
    f"{n_better_mae}/10"
)

print(
    f"  R2 better than baseline: "
    f"{n_better_r2}/10"
)

if reference_lines:

    print("")

    for line in reference_lines:

        print(line)

print("")
print(
    "Total OOF rows:",
    len(oof)
)

print(
    "Expected:",
    1641 * 10
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
    "ALL C2 GEOMETRIES REMAIN FROZEN."
)

print(
    "=" * 80
)
