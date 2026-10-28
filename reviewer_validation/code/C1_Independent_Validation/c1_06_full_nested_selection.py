from pathlib import Path
from datetime import datetime
import hashlib
import json
import warnings

import numpy as np
import pandas as pd

from scipy.stats import (
    pearsonr,
    spearmanr,
    loguniform,
)

from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
    clone,
)

from sklearn.model_selection import (
    GroupKFold,
    RandomizedSearchCV,
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    SplineTransformer,
)

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
)

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
)

from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from xgboost import XGBRegressor


warnings.filterwarnings("ignore")


# ============================================================
# PROJECT PATHS
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

BASE = (
    ROOT
    / "06_Results"
    / "Reviewer_Validation"
    / "C1_Independent_Validation"
)

SPLIT_DIR = (
    BASE
    / "Independent_Outer_Folds"
)

OUTDIR = (
    BASE
    / "C1B_Full_Nested_Selection"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# OUTPUT FILES
# ============================================================

PROTOCOL = (
    OUTDIR
    / "C1B_PROTOCOL_LOCKED_BEFORE_MODELING.txt"
)

INNER_WINNERS_FILE = (
    OUTDIR
    / "C1B_INNER_MODEL_FEATURE_WINNERS.csv"
)

SEARCH_TRACE_FILE = (
    OUTDIR
    / "C1B_INNER_SEARCH_TRACE.csv"
)

SELECTED_FILE = (
    OUTDIR
    / "C1B_SELECTED_CONFIGURATIONS.csv"
)

FOLD_METRICS_FILE = (
    OUTDIR
    / "C1B_OUTER_FOLD_METRICS.csv"
)

OOF_FILE = (
    OUTDIR
    / "C1B_SELECTION_AWARE_OOF.csv"
)

POOLED_FILE = (
    OUTDIR
    / "C1B_POOLED_METRICS.csv"
)

BOOT_FILE = (
    OUTDIR
    / "C1B_BLOCK_BOOTSTRAP_5000.csv"
)

BOOT_SUMMARY_FILE = (
    OUTDIR
    / "C1B_BLOCK_BOOTSTRAP_SUMMARY.csv"
)

REPORT = (
    OUTDIR
    / "C1B_FINAL_REPORT.txt"
)


# ============================================================
# FIXED SETTINGS
# ============================================================

RESPONSE = "agbd"

RANDOM_SEED = 42

INNER_FOLDS = 4

BOOT_REPS = 5000

BOOT_SEED = 20261028

# Runtime parallelism only.
# Historical candidate spaces remain unchanged.
N_JOBS_REVIEWER = 1


# ============================================================
# PREDECLARED FEATURE FAMILIES
#
# These families come from the existing Phase 8C predictor
# space and physical scale grouping.
#
# No outer-test outcome is used to construct these families.
# ============================================================

BASE_3P = [
    "HH_dB",
    "HV_dB",
    "HV_SD",
]

FINE_3X3 = [
    "HH_SD_dB_3x3",
]

CONTEXT_5X5 = [
    "HH_MeanLin_dB_5x5",
    "HV_MeanLin_dB_5x5",
    "HH_SD_dB_5x5",
    "HV_SD_dB_5x5",
    "PolContrast_SD_dB_5x5",
]

CONTEXT_9X9 = [
    "HH_MeanLin_dB_9x9",
    "HV_MeanLin_dB_9x9",
    "HH_SD_dB_9x9",
    "HV_SD_dB_9x9",
    "PolContrast_SD_dB_9x9",
]

FEATURE_FAMILIES = {

    "F01_3P":
        BASE_3P,

    "F02_3P_PLUS_3X3":
        BASE_3P
        + FINE_3X3,

    "F03_3P_PLUS_5X5":
        BASE_3P
        + CONTEXT_5X5,

    "F04_3P_PLUS_9X9":
        BASE_3P
        + CONTEXT_9X9,

    "F05_3P_PLUS_3X3_PLUS_5X5":
        BASE_3P
        + FINE_3X3
        + CONTEXT_5X5,

    "F06_HISTORICAL_9P":
        BASE_3P
        + FINE_3X3
        + CONTEXT_9X9,

    "F07_FULL_14P":
        BASE_3P
        + FINE_3X3
        + CONTEXT_5X5
        + CONTEXT_9X9,
}


# ============================================================
# HISTORICAL MODEL SEARCH SPACES
# ============================================================

RF_PARAMS = {
    "n_estimators":
        [300, 500, 700, 900],

    "max_depth":
        [None, 8, 12, 18, 24],

    "min_samples_split":
        [2, 5, 10],

    "min_samples_leaf":
        [1, 2, 4, 6],

    "max_features":
        ["sqrt", 0.6, 0.8, 1.0],

    "bootstrap":
        [True],
}

RF_N_ITER = 18


XGB_PARAMS = {
    "n_estimators":
        [200, 350, 500, 700],

    "learning_rate":
        [0.02, 0.05, 0.08, 0.10],

    "max_depth":
        [2, 3, 4, 5],

    "min_child_weight":
        [1, 3, 5, 10],

    "subsample":
        [0.70, 0.85, 1.00],

    "colsample_bytree":
        [0.70, 0.85, 1.00],

    "reg_alpha":
        [0.0, 0.1, 0.5, 1.0],

    "reg_lambda":
        [1.0, 2.0, 5.0, 10.0],

    "gamma":
        [0.0, 0.1, 0.5],
}

XGB_N_ITER = 12


SVR_PARAMS = {
    "svr__C":
        loguniform(1e-1, 1e3),

    "svr__epsilon":
        loguniform(1e-2, 2e1),

    "svr__gamma":
        loguniform(1e-3, 1e1),
}

SVR_N_ITER = 24


ET_PARAMS = {
    "n_estimators":
        [300, 500, 700, 900],

    "max_depth":
        [None, 8, 12, 18, 24],

    "min_samples_split":
        [2, 5, 10],

    "min_samples_leaf":
        [1, 2, 4, 6],

    "max_features":
        ["sqrt", 0.6, 0.8, 1.0],

    "bootstrap":
        [False, True],
}

ET_N_ITER = 18


MLP_PARAMS = {
    "mlp__hidden_layer_sizes": [
        (16,),
        (32,),
        (64,),
        (32, 16),
        (64, 32),
        (32, 16, 8),
    ],

    "mlp__activation":
        ["relu", "tanh"],

    "mlp__alpha":
        [1e-4, 1e-3, 1e-2, 1e-1],

    "mlp__learning_rate_init":
        [1e-4, 5e-4, 1e-3, 5e-3],
}

MLP_N_ITER = 12


GAM_PARAMS = {
    "spline__n_knots":
        [4, 5, 6, 7, 8, 10],

    "spline__degree":
        [2, 3],

    "ridge__alpha":
        [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
}

GAM_N_ITER = 16


MODEL_ORDER = [
    "Linear",
    "RandomForest",
    "XGBoost",
    "SVR",
    "ExtraTrees",
    "MLP",
    "GAM",
]


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            h.update(chunk)

    return h.hexdigest()


def json_safe_value(v):

    if isinstance(
        v,
        (
            np.integer,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
        )
    ):
        return int(v)

    if isinstance(
        v,
        (
            np.floating,
            np.float16,
            np.float32,
            np.float64,
        )
    ):
        return float(v)

    if isinstance(v, tuple):
        return [
            json_safe_value(x)
            for x in v
        ]

    if isinstance(v, list):
        return [
            json_safe_value(x)
            for x in v
        ]

    if isinstance(v, dict):
        return {
            str(k): json_safe_value(val)
            for k, val in v.items()
        }

    return v


def params_json(params):

    return json.dumps(
        json_safe_value(params),
        sort_keys=True,
    )


def metric_row(
    y,
    pred,
    label,
    fold=None,
    train_n=None,
    test_n=None,
):

    y = np.asarray(
        y,
        dtype=float
    )

    pred = np.asarray(
        pred,
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

    mae = float(
        mean_absolute_error(
            y,
            pred
        )
    )

    r2 = float(
        r2_score(
            y,
            pred
        )
    )

    bias = float(
        np.mean(
            pred - y
        )
    )

    mean_y = float(
        np.mean(y)
    )

    rrmse = (
        rmse / mean_y * 100.0
        if mean_y != 0
        else np.nan
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

    obs_sd = float(
        np.std(
            y,
            ddof=1
        )
    )

    pred_sd = float(
        np.std(
            pred,
            ddof=1
        )
    )

    return {
        "model": label,
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
        "observed_mean":
            mean_y,
        "predicted_mean":
            float(np.mean(pred)),
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


# ============================================================
# GAM TRANSFORMER
# ============================================================

class AdditiveSplineTransformer(
    BaseEstimator,
    TransformerMixin
):

    def __init__(
        self,
        n_knots=6,
        degree=3
    ):
        self.n_knots = n_knots
        self.degree = degree

    def fit(
        self,
        X,
        y=None
    ):

        X = np.asarray(
            X,
            dtype=float
        )

        self.transformers_ = []

        for j in range(
            X.shape[1]
        ):

            st = SplineTransformer(
                n_knots=self.n_knots,
                degree=self.degree,
                include_bias=False,
                knots="quantile",
                extrapolation="linear",
            )

            st.fit(
                X[:, [j]]
            )

            self.transformers_.append(
                st
            )

        return self

    def transform(
        self,
        X
    ):

        X = np.asarray(
            X,
            dtype=float
        )

        out = []

        for j, st in enumerate(
            self.transformers_
        ):

            out.append(
                st.transform(
                    X[:, [j]]
                )
            )

        return np.hstack(out)


# ============================================================
# MODEL CONSTRUCTORS
# ============================================================

def build_model(
    model_name,
    params=None
):

    params = (
        {}
        if params is None
        else params
    )

    if model_name == "Linear":

        model = Pipeline([
            (
                "scale",
                StandardScaler()
            ),
            (
                "linear",
                LinearRegression()
            ),
        ])

    elif model_name == "RandomForest":

        model = RandomForestRegressor(
            random_state=RANDOM_SEED,
            n_jobs=N_JOBS_REVIEWER,
            criterion="squared_error",
        )

        model.set_params(
            **params
        )

    elif model_name == "XGBoost":

        model = XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            tree_method="hist",
            random_state=RANDOM_SEED,
            n_jobs=N_JOBS_REVIEWER,
            verbosity=0,
        )

        model.set_params(
            **params
        )

    elif model_name == "SVR":

        model = Pipeline([
            (
                "scale",
                StandardScaler()
            ),
            (
                "svr",
                SVR(
                    kernel="rbf"
                )
            ),
        ])

        model.set_params(
            **params
        )

    elif model_name == "ExtraTrees":

        model = ExtraTreesRegressor(
            random_state=RANDOM_SEED,
            n_jobs=N_JOBS_REVIEWER,
            criterion="squared_error",
        )

        model.set_params(
            **params
        )

    elif model_name == "MLP":

        model = Pipeline([
            (
                "scale",
                StandardScaler()
            ),
            (
                "mlp",
                MLPRegressor(
                    solver="adam",
                    max_iter=1500,
                    early_stopping=False,
                    learning_rate="adaptive",
                    random_state=RANDOM_SEED,
                    tol=1e-4,
                )
            ),
        ])

        model.set_params(
            **params
        )

    elif model_name == "GAM":

        model = Pipeline([
            (
                "spline",
                AdditiveSplineTransformer()
            ),
            (
                "scale",
                StandardScaler()
            ),
            (
                "ridge",
                Ridge()
            ),
        ])

        model.set_params(
            **params
        )

    else:

        raise ValueError(
            f"Unknown model: "
            f"{model_name}"
        )

    return model


def search_definition(
    model_name
):

    if model_name == "RandomForest":
        return (
            RF_PARAMS,
            RF_N_ITER
        )

    if model_name == "XGBoost":
        return (
            XGB_PARAMS,
            XGB_N_ITER
        )

    if model_name == "SVR":
        return (
            SVR_PARAMS,
            SVR_N_ITER
        )

    if model_name == "ExtraTrees":
        return (
            ET_PARAMS,
            ET_N_ITER
        )

    if model_name == "MLP":
        return (
            MLP_PARAMS,
            MLP_N_ITER
        )

    if model_name == "GAM":
        return (
            GAM_PARAMS,
            GAM_N_ITER
        )

    raise ValueError(
        model_name
    )


# ============================================================
# INNER CV FOR LINEAR
# ============================================================

def linear_inner_score(
    X,
    y,
    groups
):

    cv = GroupKFold(
        n_splits=INNER_FOLDS
    )

    fold_rmse = []

    for tr, va in cv.split(
        X,
        y,
        groups
    ):

        model = build_model(
            "Linear"
        )

        model.fit(
            X[tr],
            y[tr]
        )

        pred = model.predict(
            X[va]
        )

        fold_rmse.append(
            np.sqrt(
                mean_squared_error(
                    y[va],
                    pred
                )
            )
        )

    return float(
        np.mean(
            fold_rmse
        )
    )


# ============================================================
# INNER MODEL TUNING
# ============================================================

def tune_candidate(
    model_name,
    feature_name,
    X,
    y,
    groups,
    outer_fold,
):

    if model_name == "Linear":

        score = linear_inner_score(
            X,
            y,
            groups
        )

        winner = {
            "outer_fold":
                outer_fold,

            "model_family":
                model_name,

            "feature_family":
                feature_name,

            "best_inner_RMSE":
                score,

            "best_params_json":
                "{}",

            "best_params_obj":
                {},
        }

        trace = [{
            "outer_fold":
                outer_fold,

            "model_family":
                model_name,

            "feature_family":
                feature_name,

            "candidate":
                1,

            "mean_inner_RMSE":
                score,

            "rank":
                1,

            "params_json":
                "{}",
        }]

        return (
            winner,
            trace
        )

    param_space, n_iter = (
        search_definition(
            model_name
        )
    )

    base = build_model(
        model_name
    )

    inner_cv = GroupKFold(
        n_splits=INNER_FOLDS
    )

    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_space,
        n_iter=n_iter,
        scoring=
            "neg_root_mean_squared_error",
        cv=inner_cv,
        random_state=RANDOM_SEED,
        n_jobs=N_JOBS_REVIEWER,
        refit=True,
        return_train_score=False,
        verbose=0,
        error_score="raise",
    )

    search.fit(
        X,
        y,
        groups=groups,
    )

    best_rmse = float(
        -search.best_score_
    )

    winner = {
        "outer_fold":
            outer_fold,

        "model_family":
            model_name,

        "feature_family":
            feature_name,

        "best_inner_RMSE":
            best_rmse,

        "best_params_json":
            params_json(
                search.best_params_
            ),

        "best_params_obj":
            search.best_params_,
    }

    trace = []

    cvres = search.cv_results_

    for i in range(
        len(cvres["params"])
    ):

        trace.append({
            "outer_fold":
                outer_fold,

            "model_family":
                model_name,

            "feature_family":
                feature_name,

            "candidate":
                i + 1,

            "mean_inner_RMSE":
                float(
                    -cvres[
                        "mean_test_score"
                    ][i]
                ),

            "rank":
                int(
                    cvres[
                        "rank_test_score"
                    ][i]
                ),

            "params_json":
                params_json(
                    cvres["params"][i]
                ),
        })

    return (
        winner,
        trace
    )


# ============================================================
# LOAD FROZEN MASTER TABLE
# ============================================================

master = pd.read_csv(
    DATA
)

if len(master) != 1641:

    raise RuntimeError(
        f"Expected 1641 frozen rows, "
        f"found {len(master)}."
    )

if master[
    "shot_number"
].nunique() != 1641:

    raise RuntimeError(
        "shot_number is not unique."
    )


# ============================================================
# VERIFY FEATURE SPACE
# ============================================================

all_features = sorted(
    set(
        f
        for values
        in FEATURE_FAMILIES.values()
        for f in values
    )
)

missing = [
    f
    for f in all_features
    if f not in master.columns
]

if missing:

    raise RuntimeError(
        f"Missing feature columns: "
        f"{missing}"
    )


# ============================================================
# WRITE PROTOCOL BEFORE MODEL FITTING
# ============================================================

manifest_path = (
    BASE
    / "C1_02_NEW_SPLIT_MANIFEST.csv"
)

manifest_hash = (
    sha256_file(
        manifest_path
    )
)

data_hash = (
    sha256_file(
        DATA
    )
)

script_hash = (
    sha256_file(
        Path(__file__)
    )
)

with PROTOCOL.open(
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "=" * 90
        + "\n"
    )

    f.write(
        "C1B FULL NESTED MODEL/PREDICTOR "
        "SELECTION PROTOCOL\n"
    )

    f.write(
        "=" * 90
        + "\n\n"
    )

    f.write(
        "THIS FILE WAS WRITTEN BEFORE "
        "ANY C1B MODEL FITTING.\n\n"
    )

    f.write(
        "Frozen observations: 1641\n"
    )

    f.write(
        "New outer validation geometry: "
        "C1 Step 2 frozen folds\n"
    )

    f.write(
        "Outer buffer: 500 m\n"
    )

    f.write(
        "Inner CV: 4-fold GroupKFold "
        "by cv_block_id\n"
    )

    f.write(
        "Selection metric: inner RMSE\n"
    )

    f.write(
        "Random seed: 42\n"
    )

    f.write(
        "Outer test files are not opened "
        "until all inner model/predictor "
        "selection for that fold is complete.\n\n"
    )

    f.write(
        "FEATURE FAMILIES\n"
    )

    f.write(
        "-" * 90
        + "\n"
    )

    for name, features in (
        FEATURE_FAMILIES.items()
    ):

        f.write(
            f"{name} "
            f"({len(features)} predictors):\n"
        )

        for p in features:
            f.write(
                f"  {p}\n"
            )

    f.write("\n")

    f.write(
        "MODEL FAMILIES\n"
    )

    f.write(
        "-" * 90
        + "\n"
    )

    for name in MODEL_ORDER:
        f.write(
            f"  {name}\n"
        )

    f.write("\n")

    f.write(
        "Search spaces and search iteration "
        "counts reproduce the historical "
        "Phase 7 model scripts.\n"
    )

    f.write(
        "Runtime parallelism is standardized "
        "to n_jobs=1 for reviewer validation; "
        "this does not alter candidate sets, "
        "folds, seeds, or scoring.\n\n"
    )

    f.write(
        f"Data SHA256:\n"
        f"{data_hash}\n\n"
    )

    f.write(
        f"Split-manifest SHA256:\n"
        f"{manifest_hash}\n\n"
    )

    f.write(
        f"C1B script SHA256:\n"
        f"{script_hash}\n"
    )


# ============================================================
# RESULT CONTAINERS
# ============================================================

inner_winners = []

search_trace = []

selected_rows = []

fold_metric_rows = []

oof_parts = []


# ============================================================
# FULL NESTED OUTER LOOP
# ============================================================

for outer_fold in range(
    1,
    6
):

    print("")
    print(
        "=" * 80
    )

    print(
        f"C1B FULL NESTED "
        f"OUTER FOLD {outer_fold}"
    )

    print(
        "=" * 80
    )

    fold_dir = (
        SPLIT_DIR
        / f"Fold_{outer_fold}"
    )

    train_path = (
        fold_dir
        / (
            f"C1_NEW_Fold_"
            f"{outer_fold}_"
            f"TRAIN_BUFFERED_500m.csv"
        )
    )

    test_path = (
        fold_dir
        / (
            f"C1_NEW_Fold_"
            f"{outer_fold}_TEST.csv"
        )
    )

    # --------------------------------------------------------
    # TRAINING DATA ONLY AT THIS STAGE
    # --------------------------------------------------------

    train = pd.read_csv(
        train_path
    )

    y_train = (
        train[
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

    if (
        pd.Series(groups)
        .nunique()
        < INNER_FOLDS
    ):

        raise RuntimeError(
            f"Fold {outer_fold}: "
            f"insufficient inner groups."
        )

    outer_candidates = []

    # --------------------------------------------------------
    # FEATURE-FAMILY LOOP
    # --------------------------------------------------------

    for feature_name, features in (
        FEATURE_FAMILIES.items()
    ):

        print("")
        print(
            f"  FEATURE FAMILY: "
            f"{feature_name} "
            f"({len(features)} predictors)"
        )

        X_train = (
            train[
                features
            ]
            .to_numpy(float)
        )

        if not np.isfinite(
            X_train
        ).all():

            raise RuntimeError(
                f"Non-finite predictors "
                f"in {feature_name}, "
                f"outer fold {outer_fold}."
            )

        # ----------------------------------------------------
        # MODEL-FAMILY LOOP
        # ----------------------------------------------------

        for model_name in MODEL_ORDER:

            print(
                f"    {model_name}...",
                flush=True
            )

            winner, trace = (
                tune_candidate(
                    model_name=
                        model_name,

                    feature_name=
                        feature_name,

                    X=
                        X_train,

                    y=
                        y_train,

                    groups=
                        groups,

                    outer_fold=
                        outer_fold,
                )
            )

            inner_winners.append({
                k: v
                for k, v
                in winner.items()
                if k !=
                "best_params_obj"
            })

            search_trace.extend(
                trace
            )

            outer_candidates.append(
                winner
            )

            print(
                f"      best inner "
                f"RMSE = "
                f"{winner['best_inner_RMSE']:.3f}"
            )

    # --------------------------------------------------------
    # SELECT MODEL + FEATURE FAMILY USING TRAINING ONLY
    # --------------------------------------------------------

    selected = min(
        outer_candidates,
        key=lambda d:
            d[
                "best_inner_RMSE"
            ]
    )

    selected_features = (
        FEATURE_FAMILIES[
            selected[
                "feature_family"
            ]
        ]
    )

    print("")
    print(
        "  SELECTED USING OUTER-TRAINING "
        "DATA ONLY:"
    )

    print(
        f"    Model: "
        f"{selected['model_family']}"
    )

    print(
        f"    Feature family: "
        f"{selected['feature_family']}"
    )

    print(
        f"    Inner RMSE: "
        f"{selected['best_inner_RMSE']:.3f}"
    )

    # --------------------------------------------------------
    # ONLY NOW OPEN OUTER TEST FILE
    # --------------------------------------------------------

    test = pd.read_csv(
        test_path
    )

    train_ids = set(
        train[
            "shot_number"
        ].astype(str)
    )

    test_ids = set(
        test[
            "shot_number"
        ].astype(str)
    )

    overlap = len(
        train_ids
        .intersection(
            test_ids
        )
    )

    if overlap != 0:

        raise RuntimeError(
            f"Fold {outer_fold}: "
            f"train/test overlap "
            f"= {overlap}"
        )

    X_train_selected = (
        train[
            selected_features
        ]
        .to_numpy(float)
    )

    X_test_selected = (
        test[
            selected_features
        ]
        .to_numpy(float)
    )

    y_test = (
        test[
            RESPONSE
        ]
        .to_numpy(float)
    )

    # --------------------------------------------------------
    # REFIT SELECTED CONFIGURATION
    # --------------------------------------------------------

    final_model = build_model(
        selected[
            "model_family"
        ],
        selected[
            "best_params_obj"
        ],
    )

    final_model.fit(
        X_train_selected,
        y_train
    )

    pred = final_model.predict(
        X_test_selected
    )

    # --------------------------------------------------------
    # TRAINING-MEAN BASELINE
    # --------------------------------------------------------

    baseline_pred = np.full(
        len(test),
        np.mean(
            y_train
        ),
        dtype=float,
    )

    # --------------------------------------------------------
    # OUTER METRICS
    # --------------------------------------------------------

    selected_metric = metric_row(
        y_test,
        pred,
        "C1B_NESTED_SELECTED",
        fold=
            outer_fold,
        train_n=
            len(train),
        test_n=
            len(test),
    )

    baseline_metric = metric_row(
        y_test,
        baseline_pred,
        "C1B_MEAN_ONLY",
        fold=
            outer_fold,
        train_n=
            len(train),
        test_n=
            len(test),
    )

    fold_metric_rows.append(
        selected_metric
    )

    fold_metric_rows.append(
        baseline_metric
    )

    selected_rows.append({
        "outer_fold":
            outer_fold,

        "selected_model_family":
            selected[
                "model_family"
            ],

        "selected_feature_family":
            selected[
                "feature_family"
            ],

        "selected_n_predictors":
            len(
                selected_features
            ),

        "selected_predictors":
            " | ".join(
                selected_features
            ),

        "best_inner_RMSE":
            selected[
                "best_inner_RMSE"
            ],

        "best_params_json":
            selected[
                "best_params_json"
            ],

        "train_n":
            len(train),

        "test_n":
            len(test),

        "train_blocks":
            train[
                "cv_block_id"
            ].nunique(),

        "test_blocks":
            test[
                "cv_block_id"
            ].nunique(),

        "outer_RMSE":
            selected_metric[
                "RMSE_Mg_ha"
            ],

        "outer_MAE":
            selected_metric[
                "MAE_Mg_ha"
            ],

        "outer_R2":
            selected_metric[
                "R2"
            ],
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

            "outer_fold":
                outer_fold,

            "observed_agbd":
                y_test,

            "predicted_agbd":
                pred,

            "mean_only_prediction":
                baseline_pred,

            "selected_model_family":
                selected[
                    "model_family"
                ],

            "selected_feature_family":
                selected[
                    "feature_family"
                ],
        })
    )

    print("")
    print(
        f"  OUTER TEST RMSE = "
        f"{selected_metric['RMSE_Mg_ha']:.3f}"
    )

    print(
        f"  OUTER TEST R2 = "
        f"{selected_metric['R2']:.4f}"
    )

    print(
        f"  MEAN-ONLY RMSE = "
        f"{baseline_metric['RMSE_Mg_ha']:.3f}"
    )


# ============================================================
# SAVE RAW TABLES
# ============================================================

inner_winners_df = pd.DataFrame(
    inner_winners
)

search_trace_df = pd.DataFrame(
    search_trace
)

selected_df = pd.DataFrame(
    selected_rows
)

fold_metrics_df = pd.DataFrame(
    fold_metric_rows
)

oof = pd.concat(
    oof_parts,
    ignore_index=True
)


inner_winners_df.to_csv(
    INNER_WINNERS_FILE,
    index=False
)

search_trace_df.to_csv(
    SEARCH_TRACE_FILE,
    index=False
)

selected_df.to_csv(
    SELECTED_FILE,
    index=False
)

fold_metrics_df.to_csv(
    FOLD_METRICS_FILE,
    index=False
)

oof.to_csv(
    OOF_FILE,
    index=False
)


# ============================================================
# OOF INTEGRITY
# ============================================================

if len(oof) != 1641:

    raise RuntimeError(
        f"Expected 1641 OOF rows, "
        f"found {len(oof)}."
    )

if (
    oof[
        "shot_number"
    ].nunique()
    != 1641
):

    raise RuntimeError(
        "OOF shot numbers "
        "are not unique."
    )

master_ids = set(
    master[
        "shot_number"
    ].astype(str)
)

oof_ids = set(
    oof[
        "shot_number"
    ].astype(str)
)

if oof_ids != master_ids:

    raise RuntimeError(
        "OOF IDs do not exactly "
        "match frozen observations."
    )


# ============================================================
# POOLED METRICS
# ============================================================

pooled_selected = metric_row(
    oof[
        "observed_agbd"
    ],
    oof[
        "predicted_agbd"
    ],
    "C1B_NESTED_SELECTED",
    fold="POOLED",
    train_n=np.nan,
    test_n=1641,
)

pooled_mean = metric_row(
    oof[
        "observed_agbd"
    ],
    oof[
        "mean_only_prediction"
    ],
    "C1B_MEAN_ONLY",
    fold="POOLED",
    train_n=np.nan,
    test_n=1641,
)

pooled_df = pd.DataFrame([
    pooled_selected,
    pooled_mean,
])

pooled_df.to_csv(
    POOLED_FILE,
    index=False
)


# ============================================================
# BLOCK BOOTSTRAP
#
# Uses the 54 frozen 3-km blocks as resampling units.
# ============================================================

block_rows = []

for block_id, g in oof.groupby(
    "cv_block_id"
):

    y = (
        g[
            "observed_agbd"
        ]
        .to_numpy(float)
    )

    ps = (
        g[
            "predicted_agbd"
        ]
        .to_numpy(float)
    )

    pm = (
        g[
            "mean_only_prediction"
        ]
        .to_numpy(float)
    )

    block_rows.append({

        "block_id":
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

        "sse_selected":
            float(
                np.sum(
                    (ps - y) ** 2
                )
            ),

        "sse_mean":
            float(
                np.sum(
                    (pm - y) ** 2
                )
            ),

        "sae_selected":
            float(
                np.sum(
                    np.abs(
                        ps - y
                    )
                )
            ),

        "sae_mean":
            float(
                np.sum(
                    np.abs(
                        pm - y
                    )
                )
            ),
    })


block_df = pd.DataFrame(
    block_rows
)

if len(block_df) != 54:

    raise RuntimeError(
        f"Expected 54 blocks, "
        f"found {len(block_df)}."
    )


rng = np.random.default_rng(
    BOOT_SEED
)

B = len(
    block_df
)

boot_rows = []

for rep in range(
    1,
    BOOT_REPS + 1
):

    idx = rng.integers(
        0,
        B,
        size=B,
    )

    s = block_df.iloc[
        idx
    ]

    n = float(
        s["n"].sum()
    )

    sy = float(
        s["sum_y"].sum()
    )

    sy2 = float(
        s["sum_y2"].sum()
    )

    sse_s = float(
        s[
            "sse_selected"
        ].sum()
    )

    sse_m = float(
        s[
            "sse_mean"
        ].sum()
    )

    sae_s = float(
        s[
            "sae_selected"
        ].sum()
    )

    sae_m = float(
        s[
            "sae_mean"
        ].sum()
    )

    rmse_s = np.sqrt(
        sse_s / n
    )

    rmse_m = np.sqrt(
        sse_m / n
    )

    mae_s = (
        sae_s / n
    )

    mae_m = (
        sae_m / n
    )

    sst = (
        sy2
        - (
            sy ** 2
        ) / n
    )

    r2_s = (
        1.0
        - sse_s / sst
        if sst > 0
        else np.nan
    )

    r2_m = (
        1.0
        - sse_m / sst
        if sst > 0
        else np.nan
    )

    boot_rows.append({

        "replicate":
            rep,

        "selected_RMSE":
            rmse_s,

        "mean_only_RMSE":
            rmse_m,

        "delta_RMSE_selected_minus_mean":
            rmse_s - rmse_m,

        "selected_MAE":
            mae_s,

        "mean_only_MAE":
            mae_m,

        "delta_MAE_selected_minus_mean":
            mae_s - mae_m,

        "selected_R2":
            r2_s,

        "mean_only_R2":
            r2_m,

        "delta_R2_selected_minus_mean":
            r2_s - r2_m,
    })


boot_df = pd.DataFrame(
    boot_rows
)

boot_df.to_csv(
    BOOT_FILE,
    index=False
)


# ============================================================
# BOOTSTRAP SUMMARY
# ============================================================

boot_summary_rows = []

bootstrap_metrics = [

    "selected_RMSE",

    "selected_MAE",

    "selected_R2",

    "delta_RMSE_selected_minus_mean",

    "delta_MAE_selected_minus_mean",

    "delta_R2_selected_minus_mean",
]

for col in bootstrap_metrics:

    values = (
        boot_df[
            col
        ]
        .to_numpy(float)
    )

    boot_summary_rows.append({

        "metric":
            col,

        "mean":
            float(
                np.nanmean(
                    values
                )
            ),

        "median":
            float(
                np.nanmedian(
                    values
                )
            ),

        "CI_2.5":
            float(
                np.nanquantile(
                    values,
                    0.025
                )
            ),

        "CI_97.5":
            float(
                np.nanquantile(
                    values,
                    0.975
                )
            ),

        "n_replicates":
            BOOT_REPS,

        "n_blocks":
            B,
    })


boot_summary_df = pd.DataFrame(
    boot_summary_rows
)

boot_summary_df.to_csv(
    BOOT_SUMMARY_FILE,
    index=False
)


# ============================================================
# SELECTION FREQUENCIES
# ============================================================

model_frequency = (
    selected_df[
        "selected_model_family"
    ]
    .value_counts()
    .rename_axis(
        "model_family"
    )
    .reset_index(
        name="n_outer_folds"
    )
)

feature_frequency = (
    selected_df[
        "selected_feature_family"
    ]
    .value_counts()
    .rename_axis(
        "feature_family"
    )
    .reset_index(
        name="n_outer_folds"
    )
)


# ============================================================
# FINAL REPORT
# ============================================================

with REPORT.open(
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "=" * 90
        + "\n"
    )

    f.write(
        "C1B FULL NESTED "
        "MODEL/PREDICTOR-SELECTION REPORT\n"
    )

    f.write(
        "=" * 90
        + "\n\n"
    )

    f.write(
        "DESIGN\n"
    )

    f.write(
        "-" * 90
        + "\n"
    )

    f.write(
        "Five geographic outer folds were "
        "created and frozen before C1 "
        "reviewer-validation modeling.\n"
    )

    f.write(
        "For every outer fold, all "
        "model-family selection, predictor-"
        "family selection, preprocessing, "
        "and hyperparameter tuning occurred "
        "inside the buffered outer-training "
        "sample only.\n"
    )

    f.write(
        "The outer test file was not opened "
        "by this script until the complete "
        "inner selection procedure for that "
        "fold was finished.\n"
    )

    f.write(
        "Seven predictor families and seven "
        "historical model families were "
        "predeclared.\n"
    )

    f.write(
        "Inner tuning used four-fold "
        "GroupKFold by 3-km cv_block_id.\n"
    )

    f.write(
        "Selection criterion: minimum "
        "mean inner RMSE.\n\n"
    )

    f.write(
        "SELECTED CONFIGURATION BY "
        "OUTER FOLD\n"
    )

    f.write(
        "-" * 90
        + "\n"
    )

    f.write(
        selected_df.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )

    f.write(
        "POOLED OUTER-TEST PERFORMANCE\n"
    )

    f.write(
        "-" * 90
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
        "OUTER-FOLD ERRORS\n"
    )

    f.write(
        "-" * 90
        + "\n"
    )

    f.write(
        fold_metrics_df.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )

    f.write(
        "MODEL SELECTION FREQUENCY\n"
    )

    f.write(
        "-" * 90
        + "\n"
    )

    f.write(
        model_frequency.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )

    f.write(
        "FEATURE-FAMILY SELECTION "
        "FREQUENCY\n"
    )

    f.write(
        "-" * 90
        + "\n"
    )

    f.write(
        feature_frequency.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )

    f.write(
        "54-BLOCK BOOTSTRAP SUMMARY\n"
    )

    f.write(
        "-" * 90
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
        "INTERPRETATION BOUNDARY\n"
    )

    f.write(
        "-" * 90
        + "\n"
    )

    f.write(
        "This experiment explicitly accounts "
        "for model-family, predictor-family, "
        "and hyperparameter selection within "
        "nested spatial validation.\n"
    )

    f.write(
        "It remains an internal nested "
        "confirmation using the same frozen "
        "1,641-observation development "
        "dataset; it is not an external "
        "independent geographic dataset.\n"
    )

    f.write(
        "No outer folds were modified in "
        "response to C1A or C1B model "
        "performance.\n"
    )

    f.write(
        "\n"
        + "=" * 90
        + "\n"
    )


# ============================================================
# CONSOLE SUMMARY
# ============================================================

print("")
print(
    "=" * 80
)

print(
    "C1B FULL NESTED SELECTION COMPLETE"
)

print(
    "=" * 80
)

print("")
print(
    "SELECTED CONFIGURATIONS"
)

print(
    selected_df.to_string(
        index=False
    )
)

print("")
print(
    "POOLED METRICS"
)

print(
    pooled_df.to_string(
        index=False
    )
)

print("")
print(
    "MODEL SELECTION FREQUENCY"
)

print(
    model_frequency.to_string(
        index=False
    )
)

print("")
print(
    "FEATURE-FAMILY SELECTION FREQUENCY"
)

print(
    feature_frequency.to_string(
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
print(
    "OOF records:",
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
    "Protocol:"
)

print(
    PROTOCOL
)

print("")
print(
    "Final report:"
)

print(
    REPORT
)

print("")
print(
    "DO NOT CHANGE THE FROZEN "
    "OUTER FOLDS AFTER THESE RESULTS."
)

print(
    "=" * 80
)
