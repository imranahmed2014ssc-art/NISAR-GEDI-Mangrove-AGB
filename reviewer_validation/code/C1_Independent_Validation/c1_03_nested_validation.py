from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, ParameterSampler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================

ROOT = Path(r"D:\NISAR_Mangrove_Paper")

DATA = (
    ROOT / "04_Matched_Data" /
    "Predictor_Expansion" /
    "Phase8C2" /
    "Tables" /
    "GEDI_NISAR_expanded_predictors_1641.csv"
)

BASE = (
    ROOT / "06_Results" /
    "Reviewer_Validation" /
    "C1_Independent_Validation"
)

SPLITDIR = BASE / "Independent_Outer_Folds"

OUTDIR = BASE / "Nested_Validation"

OUTDIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# OUTPUTS
# ============================================================

LOCKED_FOLD_FILE = (
    OUTDIR /
    "C1_03_LOCKED_ET3P_ET9P_FOLD_METRICS.csv"
)

LOCKED_OOF_FILE = (
    OUTDIR /
    "C1_03_LOCKED_ET3P_ET9P_OOF.csv"
)

NESTED_FOLD_FILE = (
    OUTDIR /
    "C1_03_NESTED_SELECTED_FOLD_METRICS.csv"
)

NESTED_OOF_FILE = (
    OUTDIR /
    "C1_03_NESTED_SELECTED_OOF.csv"
)

SELECTED_FILE = (
    OUTDIR /
    "C1_03_SELECTED_CONFIGURATIONS.csv"
)

SEARCH_FILE = (
    OUTDIR /
    "C1_03_INNER_SEARCH_TRACE.csv"
)

POOLED_FILE = (
    OUTDIR /
    "C1_03_POOLED_METRICS.csv"
)

BOOT_FILE = (
    OUTDIR /
    "C1_03_BLOCK_BOOTSTRAP_5000.csv"
)

BOOT_SUMMARY_FILE = (
    OUTDIR /
    "C1_03_BLOCK_BOOTSTRAP_SUMMARY.csv"
)

REPORT = (
    OUTDIR /
    "C1_03_NESTED_VALIDATION_REPORT.txt"
)

# ============================================================
# LOCKED PROTOCOL
# ============================================================

RANDOM_SEED = 42
INNER_SPLITS = 4
BOOT_REPS = 5000
BOOT_SEED = 20261027

FEATURES_3P = [
    "HH_dB",
    "HV_dB",
    "HV_SD",
]

FEATURES_9P = [
    "HH_dB",
    "HV_dB",
    "HV_SD",
    "HH_SD_dB_3x3",
    "HH_MeanLin_dB_9x9",
    "HV_MeanLin_dB_9x9",
    "HH_SD_dB_9x9",
    "HV_SD_dB_9x9",
    "PolContrast_SD_dB_9x9",
]

ET_PARAM_SPACE = {
    "n_estimators": [400, 700, 1000],
    "max_depth": [6, 8, 12, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 3, 6, 10],
    "max_features": ["sqrt", 0.5, 0.8, 1.0],
    "bootstrap": [True, False],
}

MLP_PARAM_SPACE = {
    "hidden_layer_sizes": [
        (32,),
        (64,),
        (32, 16),
        (64, 32),
    ],
    "activation": ["relu", "tanh"],
    "alpha": [0.001, 0.01, 0.1, 1.0],
    "learning_rate_init": [
        0.0005,
        0.001,
        0.003,
    ],
}

ET_N_ITER = 18
MLP_N_ITER = 12

# ============================================================
# HELPERS
# ============================================================

def metric_row(
    y,
    p,
    model,
    fold=None,
    train_n=None,
    test_n=None,
):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)

    rmse = float(
        np.sqrt(mean_squared_error(y, p))
    )

    mae = float(
        mean_absolute_error(y, p)
    )

    r2 = float(
        r2_score(y, p)
    )

    bias = float(
        np.mean(p - y)
    )

    mean_y = float(np.mean(y))

    rrmse = (
        float(rmse / mean_y * 100.0)
        if mean_y != 0 else np.nan
    )

    if (
        len(y) >= 3
        and np.std(y) > 0
        and np.std(p) > 0
    ):
        pr = float(
            pearsonr(y, p).statistic
        )

        sr = float(
            spearmanr(y, p).statistic
        )

    else:
        pr = np.nan
        sr = np.nan

    obs_sd = float(
        np.std(y, ddof=1)
    )

    pred_sd = float(
        np.std(p, ddof=1)
    )

    return {
        "model": model,
        "fold": fold,
        "n": len(y),
        "train_n": train_n,
        "test_n": test_n,
        "RMSE_Mg_ha": rmse,
        "MAE_Mg_ha": mae,
        "R2": r2,
        "Bias_Mg_ha": bias,
        "rRMSE_percent": rrmse,
        "Pearson_r": pr,
        "Spearman_rho": sr,
        "observed_mean": mean_y,
        "predicted_mean": float(np.mean(p)),
        "observed_sd": obs_sd,
        "predicted_sd": pred_sd,
        "pred_to_obs_SD_ratio":
            pred_sd / obs_sd
            if obs_sd > 0
            else np.nan,
    }


def make_linear():
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LinearRegression()),
    ])


def make_mlp(params):
    return Pipeline([
        ("scale", StandardScaler()),
        (
            "mlp",
            MLPRegressor(
                hidden_layer_sizes=
                    params[
                        "hidden_layer_sizes"
                    ],
                activation=
                    params["activation"],
                alpha=
                    params["alpha"],
                learning_rate_init=
                    params[
                        "learning_rate_init"
                    ],
                solver="adam",
                max_iter=1500,
                early_stopping=False,
                random_state=RANDOM_SEED,
            ),
        ),
    ])


def grouped_linear_score(
    X,
    y,
    groups,
):
    cv = GroupKFold(
        n_splits=INNER_SPLITS
    )

    scores = []

    for tr, va in cv.split(
        X,
        y,
        groups
    ):
        model = make_linear()

        model.fit(
            X[tr],
            y[tr]
        )

        pred = model.predict(
            X[va]
        )

        scores.append(
            np.sqrt(
                mean_squared_error(
                    y[va],
                    pred
                )
            )
        )

    return float(
        np.mean(scores)
    )


def tune_et(
    X,
    y,
    groups,
    fold,
    feature_label,
):
    # Same candidate list for 3P and 9P
    # within a given outer fold.
    candidates = list(
        ParameterSampler(
            ET_PARAM_SPACE,
            n_iter=ET_N_ITER,
            random_state=
                RANDOM_SEED + fold,
        )
    )

    cv = GroupKFold(
        n_splits=INNER_SPLITS
    )

    trace = []

    best = None

    for ci, params in enumerate(
        candidates,
        start=1
    ):
        rmses = []

        for tr, va in cv.split(
            X,
            y,
            groups
        ):
            model = ExtraTreesRegressor(
                **params,
                random_state=RANDOM_SEED,
                n_jobs=1,
            )

            model.fit(
                X[tr],
                y[tr]
            )

            pred = model.predict(
                X[va]
            )

            rmses.append(
                np.sqrt(
                    mean_squared_error(
                        y[va],
                        pred
                    )
                )
            )

        score = float(
            np.mean(rmses)
        )

        safe = {
            k: (
                list(v)
                if isinstance(v, tuple)
                else v
            )
            for k, v in params.items()
        }

        row = {
            "outer_fold": fold,
            "model_family":
                "ExtraTrees",
            "feature_set":
                feature_label,
            "candidate": ci,
            "mean_inner_RMSE":
                score,
            "params_json":
                json.dumps(
                    safe,
                    sort_keys=True
                ),
        }

        trace.append(row)

        if (
            best is None
            or score <
            best["mean_inner_RMSE"]
        ):
            best = {
                **row,
                "params": params,
            }

    return best, trace


def tune_mlp(
    X,
    y,
    groups,
    fold,
    feature_label,
):
    candidates = list(
        ParameterSampler(
            MLP_PARAM_SPACE,
            n_iter=MLP_N_ITER,
            random_state=
                RANDOM_SEED +
                100 +
                fold,
        )
    )

    cv = GroupKFold(
        n_splits=INNER_SPLITS
    )

    trace = []
    best = None

    for ci, params in enumerate(
        candidates,
        start=1
    ):
        rmses = []

        for tr, va in cv.split(
            X,
            y,
            groups
        ):
            model = make_mlp(
                params
            )

            model.fit(
                X[tr],
                y[tr]
            )

            pred = model.predict(
                X[va]
            )

            rmses.append(
                np.sqrt(
                    mean_squared_error(
                        y[va],
                        pred
                    )
                )
            )

        score = float(
            np.mean(rmses)
        )

        safe = {
            k: (
                list(v)
                if isinstance(v, tuple)
                else v
            )
            for k, v in params.items()
        }

        row = {
            "outer_fold": fold,
            "model_family":
                "MLP",
            "feature_set":
                feature_label,
            "candidate": ci,
            "mean_inner_RMSE":
                score,
            "params_json":
                json.dumps(
                    safe,
                    sort_keys=True
                ),
        }

        trace.append(row)

        if (
            best is None
            or score <
            best["mean_inner_RMSE"]
        ):
            best = {
                **row,
                "params": params,
            }

    return best, trace


def fit_candidate(
    family,
    params,
    X,
    y,
):
    if family == "Linear":
        model = make_linear()

    elif family == "ExtraTrees":
        model = ExtraTreesRegressor(
            **params,
            random_state=RANDOM_SEED,
            n_jobs=1,
        )

    elif family == "MLP":
        model = make_mlp(
            params
        )

    else:
        raise ValueError(
            family
        )

    model.fit(X, y)

    return model


# ============================================================
# LOAD AND VERIFY DATA
# ============================================================

df = pd.read_csv(DATA)

if len(df) != 1641:
    raise RuntimeError(
        f"Expected 1641 rows; "
        f"found {len(df)}"
    )

required = (
    [
        "shot_number",
        "agbd",
        "cv_block_id",
    ]
    + FEATURES_9P
)

missing = [
    c for c in required
    if c not in df.columns
]

if missing:
    raise RuntimeError(
        f"Missing required columns: "
        f"{missing}"
    )

# ============================================================
# RESULT CONTAINERS
# ============================================================

locked_metrics = []
locked_oof = []

nested_metrics = []
nested_oof = []

selected_configs = []
search_trace = []

# ============================================================
# OUTER LOOP
# ============================================================

for fold in range(1, 6):

    print("")
    print("=" * 72)
    print(
        f"C1 NESTED VALIDATION - "
        f"OUTER FOLD {fold}"
    )
    print("=" * 72)

    fold_dir = (
        SPLITDIR /
        f"Fold_{fold}"
    )

    train_path = (
        fold_dir /
        f"C1_NEW_Fold_{fold}_"
        f"TRAIN_BUFFERED_500m.csv"
    )

    test_path = (
        fold_dir /
        f"C1_NEW_Fold_{fold}_"
        f"TEST.csv"
    )

    train = pd.read_csv(
        train_path
    )

    test = pd.read_csv(
        test_path
    )

    if len(
        set(
            train["shot_number"]
            .astype(str)
        ).intersection(
            set(
                test["shot_number"]
                .astype(str)
            )
        )
    ) != 0:
        raise RuntimeError(
            f"Fold {fold}: "
            f"train/test overlap."
        )

    ytr = train[
        "agbd"
    ].to_numpy(float)

    yte = test[
        "agbd"
    ].to_numpy(float)

    groups = train[
        "cv_block_id"
    ].to_numpy()

    feature_sets = {
        "3P": FEATURES_3P,
        "9P": FEATURES_9P,
    }

    inner_candidates = []

    # --------------------------------------------------------
    # INNER MODEL/PREDICTOR SELECTION
    # --------------------------------------------------------

    for feature_label, features \
            in feature_sets.items():

        Xtr = train[
            features
        ].to_numpy(float)

        # Linear
        lin_score = (
            grouped_linear_score(
                Xtr,
                ytr,
                groups,
            )
        )

        lin_row = {
            "outer_fold": fold,
            "model_family":
                "Linear",
            "feature_set":
                feature_label,
            "mean_inner_RMSE":
                lin_score,
            "best_params_json":
                "{}",
            "params": {},
        }

        inner_candidates.append(
            lin_row
        )

        search_trace.append({
            "outer_fold": fold,
            "model_family":
                "Linear",
            "feature_set":
                feature_label,
            "candidate": 1,
            "mean_inner_RMSE":
                lin_score,
            "params_json":
                "{}",
        })

        print(
            f"  Linear {feature_label}: "
            f"inner RMSE="
            f"{lin_score:.3f}"
        )

        # Extra Trees
        et_best, et_trace = (
            tune_et(
                Xtr,
                ytr,
                groups,
                fold,
                feature_label,
            )
        )

        search_trace.extend(
            et_trace
        )

        inner_candidates.append({
            "outer_fold": fold,
            "model_family":
                "ExtraTrees",
            "feature_set":
                feature_label,
            "mean_inner_RMSE":
                et_best[
                    "mean_inner_RMSE"
                ],
            "best_params_json":
                et_best[
                    "params_json"
                ],
            "params":
                et_best["params"],
        })

        print(
            f"  ExtraTrees "
            f"{feature_label}: "
            f"inner RMSE="
            f"{et_best['mean_inner_RMSE']:.3f}"
        )

        # MLP
        mlp_best, mlp_trace = (
            tune_mlp(
                Xtr,
                ytr,
                groups,
                fold,
                feature_label,
            )
        )

        search_trace.extend(
            mlp_trace
        )

        inner_candidates.append({
            "outer_fold": fold,
            "model_family":
                "MLP",
            "feature_set":
                feature_label,
            "mean_inner_RMSE":
                mlp_best[
                    "mean_inner_RMSE"
                ],
            "best_params_json":
                mlp_best[
                    "params_json"
                ],
            "params":
                mlp_best["params"],
        })

        print(
            f"  MLP {feature_label}: "
            f"inner RMSE="
            f"{mlp_best['mean_inner_RMSE']:.3f}"
        )

    # --------------------------------------------------------
    # SELECT CONFIGURATION USING INNER TRAINING ONLY
    # --------------------------------------------------------

    selected = min(
        inner_candidates,
        key=lambda x:
            x["mean_inner_RMSE"]
    )

    selected_configs.append({
        "outer_fold": fold,
        "selected_model_family":
            selected[
                "model_family"
            ],
        "selected_feature_set":
            selected[
                "feature_set"
            ],
        "best_inner_RMSE":
            selected[
                "mean_inner_RMSE"
            ],
        "best_params_json":
            selected[
                "best_params_json"
            ],
    })

    print("")
    print(
        "  SELECTED INSIDE TRAINING ONLY:"
    )
    print(
        f"  {selected['model_family']} "
        f"{selected['feature_set']}"
    )
    print(
        f"  inner RMSE="
        f"{selected['mean_inner_RMSE']:.3f}"
    )

    # --------------------------------------------------------
    # OUTER TEST OF SELECTED CONFIGURATION
    # --------------------------------------------------------

    selected_features = (
        feature_sets[
            selected[
                "feature_set"
            ]
        ]
    )

    Xtr_sel = train[
        selected_features
    ].to_numpy(float)

    Xte_sel = test[
        selected_features
    ].to_numpy(float)

    selected_model = fit_candidate(
        selected[
            "model_family"
        ],
        selected["params"],
        Xtr_sel,
        ytr,
    )

    pred_sel = (
        selected_model.predict(
            Xte_sel
        )
    )

    nm = metric_row(
        yte,
        pred_sel,
        (
            "NESTED_SELECTED_"
            f"{selected['model_family']}_"
            f"{selected['feature_set']}"
        ),
        fold=fold,
        train_n=len(train),
        test_n=len(test),
    )

    nested_metrics.append(
        nm
    )

    nested_oof.append(
        pd.DataFrame({
            "shot_number":
                test[
                    "shot_number"
                ].to_numpy(),
            "cv_block_id":
                test[
                    "cv_block_id"
                ].to_numpy(),
            "outer_fold":
                fold,
            "observed_agbd":
                yte,
            "predicted_agbd":
                pred_sel,
            "selected_model_family":
                selected[
                    "model_family"
                ],
            "selected_feature_set":
                selected[
                    "feature_set"
                ],
        })
    )

    print(
        f"  OUTER selected RMSE="
        f"{nm['RMSE_Mg_ha']:.3f}, "
        f"R2={nm['R2']:.4f}"
    )

    # --------------------------------------------------------
    # LOCKED ET3P AND ET9P OUTER TESTS
    #
    # These were specified before seeing these new
    # outer-fold results.
    # --------------------------------------------------------

    for feature_label, features \
            in feature_sets.items():

        matching = [
            x for x
            in inner_candidates
            if (
                x["model_family"]
                == "ExtraTrees"
                and
                x["feature_set"]
                == feature_label
            )
        ]

        if len(matching) != 1:
            raise RuntimeError(
                "ExtraTrees inner result "
                "not uniquely found."
            )

        et_info = matching[0]

        Xtr = train[
            features
        ].to_numpy(float)

        Xte = test[
            features
        ].to_numpy(float)

        model = ExtraTreesRegressor(
            **et_info["params"],
            random_state=RANDOM_SEED,
            n_jobs=1,
        )

        model.fit(
            Xtr,
            ytr
        )

        pred = model.predict(
            Xte
        )

        label = (
            f"LOCKED_ET_{feature_label}"
        )

        lm = metric_row(
            yte,
            pred,
            label,
            fold=fold,
            train_n=len(train),
            test_n=len(test),
        )

        locked_metrics.append(
            lm
        )

        locked_oof.append(
            pd.DataFrame({
                "shot_number":
                    test[
                        "shot_number"
                    ].to_numpy(),
                "cv_block_id":
                    test[
                        "cv_block_id"
                    ].to_numpy(),
                "outer_fold":
                    fold,
                "observed_agbd":
                    yte,
                "predicted_agbd":
                    pred,
                "model":
                    label,
            })
        )

        print(
            f"  {label}: "
            f"RMSE="
            f"{lm['RMSE_Mg_ha']:.3f}, "
            f"R2={lm['R2']:.4f}"
        )

    # --------------------------------------------------------
    # MEAN-ONLY BASELINE
    # --------------------------------------------------------

    baseline_pred = np.full(
        len(test),
        np.mean(ytr),
        dtype=float,
    )

    bm = metric_row(
        yte,
        baseline_pred,
        "MEAN_ONLY",
        fold=fold,
        train_n=len(train),
        test_n=len(test),
    )

    locked_metrics.append(
        bm
    )

    locked_oof.append(
        pd.DataFrame({
            "shot_number":
                test[
                    "shot_number"
                ].to_numpy(),
            "cv_block_id":
                test[
                    "cv_block_id"
                ].to_numpy(),
            "outer_fold":
                fold,
            "observed_agbd":
                yte,
            "predicted_agbd":
                baseline_pred,
            "model":
                "MEAN_ONLY",
        })
    )


# ============================================================
# SAVE RAW RESULTS
# ============================================================

locked_metrics_df = pd.DataFrame(
    locked_metrics
)

locked_oof_df = pd.concat(
    locked_oof,
    ignore_index=True
)

nested_metrics_df = pd.DataFrame(
    nested_metrics
)

nested_oof_df = pd.concat(
    nested_oof,
    ignore_index=True
)

selected_df = pd.DataFrame(
    selected_configs
)

search_df = pd.DataFrame(
    search_trace
)

locked_metrics_df.to_csv(
    LOCKED_FOLD_FILE,
    index=False
)

locked_oof_df.to_csv(
    LOCKED_OOF_FILE,
    index=False
)

nested_metrics_df.to_csv(
    NESTED_FOLD_FILE,
    index=False
)

nested_oof_df.to_csv(
    NESTED_OOF_FILE,
    index=False
)

selected_df.to_csv(
    SELECTED_FILE,
    index=False
)

search_df.to_csv(
    SEARCH_FILE,
    index=False
)

# ============================================================
# POOLED OOF METRICS
# ============================================================

pooled = []

for model in [
    "LOCKED_ET_3P",
    "LOCKED_ET_9P",
    "MEAN_ONLY",
]:

    d = locked_oof_df[
        locked_oof_df[
            "model"
        ] == model
    ]

    pooled.append(
        metric_row(
            d["observed_agbd"],
            d["predicted_agbd"],
            model,
            fold="POOLED",
            train_n=np.nan,
            test_n=len(d),
        )
    )

pooled.append(
    metric_row(
        nested_oof_df[
            "observed_agbd"
        ],
        nested_oof_df[
            "predicted_agbd"
        ],
        "NESTED_SELECTION_AWARE",
        fold="POOLED",
        train_n=np.nan,
        test_n=len(
            nested_oof_df
        ),
    )
)

pooled_df = pd.DataFrame(
    pooled
)

pooled_df.to_csv(
    POOLED_FILE,
    index=False
)

# ============================================================
# BLOCK BOOTSTRAP:
# LOCKED ET9P MINUS ET3P
# ============================================================

p3 = (
    locked_oof_df[
        locked_oof_df["model"]
        == "LOCKED_ET_3P"
    ][
        [
            "shot_number",
            "cv_block_id",
            "observed_agbd",
            "predicted_agbd"
        ]
    ]
    .rename(
        columns={
            "predicted_agbd":
                "pred3"
        }
    )
)

p9 = (
    locked_oof_df[
        locked_oof_df["model"]
        == "LOCKED_ET_9P"
    ][
        [
            "shot_number",
            "predicted_agbd"
        ]
    ]
    .rename(
        columns={
            "predicted_agbd":
                "pred9"
        }
    )
)

pair = p3.merge(
    p9,
    on="shot_number",
    how="inner",
    validate="one_to_one",
)

if len(pair) != 1641:
    raise RuntimeError(
        "Locked ET3P/ET9P OOF merge "
        "did not produce 1641 rows."
    )

block_stats = []

for bid, g in pair.groupby(
    "cv_block_id"
):

    y = g[
        "observed_agbd"
    ].to_numpy(float)

    e3 = (
        g["pred3"].to_numpy(float)
        - y
    )

    e9 = (
        g["pred9"].to_numpy(float)
        - y
    )

    block_stats.append({
        "block": bid,
        "n": len(g),
        "sum_y": float(
            np.sum(y)
        ),
        "sum_y2": float(
            np.sum(y ** 2)
        ),
        "sse3": float(
            np.sum(e3 ** 2)
        ),
        "sse9": float(
            np.sum(e9 ** 2)
        ),
        "sae3": float(
            np.sum(np.abs(e3))
        ),
        "sae9": float(
            np.sum(np.abs(e9))
        ),
    })

bs = pd.DataFrame(
    block_stats
)

rng = np.random.default_rng(
    BOOT_SEED
)

B = len(bs)

boot_rows = []

for rep in range(
    1,
    BOOT_REPS + 1
):

    idx = rng.integers(
        0,
        B,
        size=B
    )

    s = bs.iloc[idx]

    n = float(
        s["n"].sum()
    )

    sse3 = float(
        s["sse3"].sum()
    )

    sse9 = float(
        s["sse9"].sum()
    )

    sae3 = float(
        s["sae3"].sum()
    )

    sae9 = float(
        s["sae9"].sum()
    )

    sy = float(
        s["sum_y"].sum()
    )

    sy2 = float(
        s["sum_y2"].sum()
    )

    rmse3 = np.sqrt(
        sse3 / n
    )

    rmse9 = np.sqrt(
        sse9 / n
    )

    mae3 = sae3 / n
    mae9 = sae9 / n

    sst = (
        sy2
        - (sy ** 2) / n
    )

    r23 = (
        1.0 - sse3 / sst
        if sst > 0
        else np.nan
    )

    r29 = (
        1.0 - sse9 / sst
        if sst > 0
        else np.nan
    )

    boot_rows.append({
        "replicate": rep,
        "delta_RMSE_ET9P_minus_ET3P":
            rmse9 - rmse3,
        "delta_MAE_ET9P_minus_ET3P":
            mae9 - mae3,
        "delta_R2_ET9P_minus_ET3P":
            r29 - r23,
    })

boot_df = pd.DataFrame(
    boot_rows
)

boot_df.to_csv(
    BOOT_FILE,
    index=False
)

summary_rows = []

for metric in [
    "delta_RMSE_ET9P_minus_ET3P",
    "delta_MAE_ET9P_minus_ET3P",
    "delta_R2_ET9P_minus_ET3P",
]:

    vals = boot_df[
        metric
    ].to_numpy(float)

    summary_rows.append({
        "metric": metric,
        "mean":
            float(np.mean(vals)),
        "median":
            float(np.median(vals)),
        "CI_2.5":
            float(
                np.quantile(
                    vals,
                    0.025
                )
            ),
        "CI_97.5":
            float(
                np.quantile(
                    vals,
                    0.975
                )
            ),
        "n_replicates":
            BOOT_REPS,
        "n_blocks":
            B,
    })

boot_summary_df = pd.DataFrame(
    summary_rows
)

boot_summary_df.to_csv(
    BOOT_SUMMARY_FILE,
    index=False
)

# ============================================================
# REPORT
# ============================================================

fold_compare = (
    locked_metrics_df[
        locked_metrics_df[
            "model"
        ].isin([
            "LOCKED_ET_3P",
            "LOCKED_ET_9P",
        ])
    ]
    .pivot(
        index="fold",
        columns="model",
        values="RMSE_Mg_ha",
    )
)

if (
    "LOCKED_ET_3P"
    in fold_compare.columns
    and
    "LOCKED_ET_9P"
    in fold_compare.columns
):
    fold_compare[
        "Delta_RMSE_9P_minus_3P"
    ] = (
        fold_compare[
            "LOCKED_ET_9P"
        ]
        -
        fold_compare[
            "LOCKED_ET_3P"
        ]
    )

with REPORT.open(
    "w",
    encoding="utf-8"
) as f:

    f.write("=" * 90 + "\n")
    f.write(
        "C1 FULLY NESTED "
        "SELECTION-INDEPENDENT "
        "VALIDATION REPORT\n"
    )
    f.write("=" * 90 + "\n\n")

    f.write("DESIGN\n")
    f.write("-" * 90 + "\n")

    f.write(
        "New geographic outer folds were "
        "frozen before this modeling run.\n"
    )

    f.write(
        "Outer test observations were never "
        "used for model-family selection, "
        "predictor-set selection, or "
        "hyperparameter tuning.\n"
    )

    f.write(
        "All selection occurred within the "
        "500-m-buffered outer training data "
        "using four-fold GroupKFold grouped "
        "by 3-km cv_block_id.\n"
    )

    f.write(
        "Candidate predictor sets: locked 3P "
        "and locked 9P.\n"
    )

    f.write(
        "Candidate model families: Linear, "
        "Extra Trees, and MLP.\n"
    )

    f.write(
        "Extra Trees search: 18 candidates "
        "from the historical Phase 8C4 "
        "search space.\n"
    )

    f.write(
        "MLP search: 12 candidates from the "
        "historical Phase 8C4 search space.\n\n"
    )

    f.write("POOLED OUTER-TEST METRICS\n")
    f.write("-" * 90 + "\n")
    f.write(
        pooled_df.to_string(
            index=False
        )
    )
    f.write("\n\n")

    f.write("LOCKED ET3P VS ET9P BY OUTER FOLD\n")
    f.write("-" * 90 + "\n")
    f.write(
        fold_compare.to_string()
    )
    f.write("\n\n")

    f.write("NESTED SELECTED CONFIGURATION BY FOLD\n")
    f.write("-" * 90 + "\n")
    f.write(
        selected_df.to_string(
            index=False
        )
    )
    f.write("\n\n")

    f.write("BLOCK BOOTSTRAP SUMMARY\n")
    f.write("-" * 90 + "\n")
    f.write(
        boot_summary_df.to_string(
            index=False
        )
    )
    f.write("\n\n")

    f.write("INTERPRETIVE BOUNDARY\n")
    f.write("-" * 90 + "\n")

    f.write(
        "This analysis provides "
        "model-selection-aware nested spatial "
        "re-evaluation using a newly frozen "
        "outer validation geometry. It is "
        "stronger than reusing the original "
        "Phase 9 outer-fold scores.\n"
    )

    f.write(
        "However, it is not an external "
        "independent dataset: the same 1,641 "
        "observations formed the historical "
        "development dataset. Therefore the "
        "result should be described as nested "
        "selection-aware confirmation rather "
        "than external geographic validation.\n"
    )

    f.write("\n" + "=" * 90 + "\n")

# ============================================================
# CONSOLE
# ============================================================

print("")
print("=" * 80)
print("C1 STEP 3 COMPLETE")
print("FULLY NESTED VALIDATION FINISHED")
print("=" * 80)

print("")
print("POOLED METRICS")
print(
    pooled_df.to_string(
        index=False
    )
)

print("")
print(
    "LOCKED ET3P VS ET9P "
    "BY NEW OUTER FOLD"
)
print(
    fold_compare.to_string()
)

print("")
print(
    "NESTED SELECTED "
    "CONFIGURATIONS"
)
print(
    selected_df.to_string(
        index=False
    )
)

print("")
print(
    "BLOCK BOOTSTRAP SUMMARY"
)
print(
    boot_summary_df.to_string(
        index=False
    )
)

print("")
print("Report:")
print(REPORT)

print("")
print(
    "IMPORTANT: DO NOT ALTER "
    "THE FROZEN OUTER SPLITS "
    "AFTER SEEING THESE RESULTS."
)

print("=" * 80)
