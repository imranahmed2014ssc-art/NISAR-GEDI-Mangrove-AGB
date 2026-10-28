from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import (
    GroupKFold,
    RandomizedSearchCV,
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

MASTER = (
    ROOT
    / "04_Matched_Data"
    / "Predictor_Expansion"
    / "Phase8C2"
    / "Tables"
    / "GEDI_NISAR_expanded_predictors_1641.csv"
)

C1B_OOF = (
    ROOT
    / "06_Results"
    / "Reviewer_Validation"
    / "C1_Independent_Validation"
    / "C1B_Full_Nested_Selection"
    / "C1B_SELECTION_AWARE_OOF.csv"
)

FAMILY_FILE = (
    ROOT
    / "06_Results"
    / "Reviewer_Validation"
    / "C5_Multiscale_Ablation"
    / "C5_00_ABLATION_FAMILIES.csv"
)

OUT = (
    ROOT
    / "06_Results"
    / "Reviewer_Validation"
    / "C5_Multiscale_Ablation"
    / "C5_01_ExtraTrees_Ablation"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

OOF_FILE = (
    OUT
    / "C5_01_ALL_FAMILY_OOF.csv"
)

FOLD_METRIC_FILE = (
    OUT
    / "C5_01_FOLD_METRICS.csv"
)

SUMMARY_FILE = (
    OUT
    / "C5_01_POOLED_METRICS.csv"
)

PARAM_FILE = (
    OUT
    / "C5_01_BEST_PARAMETERS.csv"
)

SPLIT_FILE = (
    OUT
    / "C5_01_OUTER_SPLIT_QC.csv"
)

CONTRAST_FILE = (
    OUT
    / "C5_01_PREDECLARED_CONTRASTS.csv"
)

REPRO_FILE = (
    OUT
    / "C5_01_A11_C1B_REPRODUCTION.csv"
)

REPORT = (
    OUT
    / "C5_01_EXTRATREES_ABLATION_REPORT.txt"
)


# ============================================================
# FROZEN PROTOCOL
# ============================================================

RANDOM_SEED = 42

INNER_SPLITS = 4

N_SEARCH_ITER = 18

OUTER_BUFFER_M = 500.0

BOOTSTRAP_REPS = 2000

BOOTSTRAP_SEED = 20261028


# Historical ExtraTrees randomized-search space.
PARAM_DISTRIBUTIONS = {
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
# PREDECLARED PAIRED CONTRASTS
#
# candidate - reference
# Negative delta RMSE / MAE = improvement.
# Positive delta R2 = improvement.
# ============================================================

CONTRASTS = [
    (
        "A02_vs_A01",
        "A02_HH_HV9_MEANS",
        "A01_HV9_MEAN_ONLY",
    ),

    (
        "A04_vs_A03",
        "A04_3P_PLUS_HV9_MEAN",
        "A03_3P_BASE",
    ),

    (
        "A05_vs_A03",
        "A05_3P_PLUS_9X9_MEANS",
        "A03_3P_BASE",
    ),

    (
        "A06_vs_A03",
        "A06_3P_PLUS_9X9_HETEROGENEITY",
        "A03_3P_BASE",
    ),

    (
        "A07_vs_A03",
        "A07_3P_PLUS_HH3_SD",
        "A03_3P_BASE",
    ),

    (
        "A08_vs_A03",
        "A08_3P_PLUS_5X5_CONTEXT",
        "A03_3P_BASE",
    ),

    (
        "A09_vs_A03",
        "A09_3P_PLUS_9X9_CONTEXT",
        "A03_3P_BASE",
    ),

    (
        "A09_vs_A08",
        "A09_3P_PLUS_9X9_CONTEXT",
        "A08_3P_PLUS_5X5_CONTEXT",
    ),

    (
        "A10_vs_A03",
        "A10_HISTORICAL_9P",
        "A03_3P_BASE",
    ),

    (
        "A11_vs_A03",
        "A11_FULL_14P",
        "A03_3P_BASE",
    ),

    (
        "A11_vs_A10",
        "A11_FULL_14P",
        "A10_HISTORICAL_9P",
    ),

    (
        "A11_vs_A00",
        "A11_FULL_14P",
        "A00_COORD_ONLY",
    ),
]


# ============================================================
# LOAD DATA
# ============================================================

master = pd.read_csv(
    MASTER
)

c1b = pd.read_csv(
    C1B_OOF
)

families = pd.read_csv(
    FAMILY_FILE
)


# ============================================================
# MASTER QC
# ============================================================

if len(master) != 1641:
    raise RuntimeError(
        f"Master expected 1641 rows; "
        f"found {len(master)}."
    )

if master["shot_number"].nunique() != 1641:
    raise RuntimeError(
        "Master shot_number is not unique."
    )

required_master = [
    "shot_number",
    "agbd",
    "cv_block_id",
    "utm_x",
    "utm_y",
]

missing = [
    c
    for c in required_master
    if c not in master.columns
]

if missing:
    raise RuntimeError(
        f"Master missing columns: {missing}"
    )


# ============================================================
# C1B OOF QC
# ============================================================

required_c1b = [
    "shot_number",
    "outer_fold",
    "observed_agbd",
    "predicted_agbd",
]

missing = [
    c
    for c in required_c1b
    if c not in c1b.columns
]

if missing:
    raise RuntimeError(
        f"C1B OOF missing columns: {missing}"
    )

if len(c1b) != 1641:
    raise RuntimeError(
        f"C1B OOF expected 1641 rows; "
        f"found {len(c1b)}."
    )

if c1b["shot_number"].nunique() != 1641:
    raise RuntimeError(
        "C1B OOF shot_number is not unique."
    )


# ============================================================
# STANDARDIZE SHOT IDs
# ============================================================

master["shot_number"] = (
    master["shot_number"]
    .astype(str)
)

c1b["shot_number"] = (
    c1b["shot_number"]
    .astype(str)
)


# ============================================================
# MERGE FROZEN C1B OUTER FOLDS
# ============================================================

fold_map = (
    c1b[
        [
            "shot_number",
            "outer_fold",
            "observed_agbd",
            "predicted_agbd",
        ]
    ]
    .rename(
        columns={
            "outer_fold":
                "c1b_outer_fold",

            "observed_agbd":
                "c1b_observed_agbd",

            "predicted_agbd":
                "c1b_predicted_agbd",
        }
    )
)


df = master.merge(
    fold_map,
    on="shot_number",
    how="inner",
    validate="one_to_one",
)


if len(df) != 1641:
    raise RuntimeError(
        f"Merged dataset expected 1641 rows; "
        f"found {len(df)}."
    )


# ============================================================
# OBSERVED RESPONSE CONSISTENCY
# ============================================================

response_diff = np.abs(
    pd.to_numeric(
        df["agbd"],
        errors="coerce",
    )
    -
    pd.to_numeric(
        df["c1b_observed_agbd"],
        errors="coerce",
    )
)


MAX_RESPONSE_DIFF = float(
    response_diff.max()
)


if MAX_RESPONSE_DIFF > 1e-8:
    raise RuntimeError(
        "AGBD differs between master and C1B OOF. "
        f"Max difference = {MAX_RESPONSE_DIFF}"
    )


# ============================================================
# FAMILY QC
# ============================================================

if len(families) != 12:
    raise RuntimeError(
        f"Expected 12 frozen C5 families; "
        f"found {len(families)}."
    )


family_predictors = {}

for row in families.itertuples():

    predictors = [
        p
        for p in str(
            row.predictors
        ).split(";")
        if p
    ]

    family_predictors[
        row.family
    ] = predictors

    missing = [
        p
        for p in predictors
        if p not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"{row.family}: missing predictors "
            f"{missing}"
        )


# ============================================================
# OUTER FOLDS
# ============================================================

outer_folds = sorted(
    pd.unique(
        df["c1b_outer_fold"]
    )
)


if len(outer_folds) != 5:
    raise RuntimeError(
        f"Expected five frozen outer folds; "
        f"found {outer_folds}"
    )


# ============================================================
# PRECOMPUTE FROZEN OUTER SPLITS
# ============================================================

split_data = {}

split_rows = []

mean_oof_parts = []


for fold in outer_folds:

    test_mask = (
        df["c1b_outer_fold"]
        ==
        fold
    )

    test_df = (
        df.loc[
            test_mask
        ]
        .copy()
    )

    candidate_train = (
        df.loc[
            ~test_mask
        ]
        .copy()
    )

    test_xy = (
        test_df[
            [
                "utm_x",
                "utm_y",
            ]
        ]
        .to_numpy(float)
    )

    train_xy = (
        candidate_train[
            [
                "utm_x",
                "utm_y",
            ]
        ]
        .to_numpy(float)
    )


    # --------------------------------------------------------
    # 500 m train-to-test exclusion.
    # --------------------------------------------------------

    tree = cKDTree(
        test_xy
    )

    distances, _ = tree.query(
        train_xy,
        k=1,
    )

    keep = (
        distances
        >=
        OUTER_BUFFER_M
    )

    train_df = (
        candidate_train
        .loc[
            keep
        ]
        .copy()
    )


    if len(train_df) == 0:
        raise RuntimeError(
            f"Fold {fold}: no active training rows."
        )


    # --------------------------------------------------------
    # Verify exclusion.
    # --------------------------------------------------------

    active_train_xy = (
        train_df[
            [
                "utm_x",
                "utm_y",
            ]
        ]
        .to_numpy(float)
    )

    min_active_distance = float(
        tree.query(
            active_train_xy,
            k=1,
        )[0].min()
    )


    if (
        min_active_distance
        <
        OUTER_BUFFER_M
        - 1e-8
    ):
        raise RuntimeError(
            f"Fold {fold}: 500-m exclusion violated. "
            f"Minimum = {min_active_distance}"
        )


    n_train_blocks = int(
        train_df[
            "cv_block_id"
        ].nunique()
    )

    if n_train_blocks < INNER_SPLITS:
        raise RuntimeError(
            f"Fold {fold}: only "
            f"{n_train_blocks} training blocks."
        )


    split_data[
        fold
    ] = {
        "train":
            train_df,

        "test":
            test_df,
    }


    split_rows.append({
        "outer_fold":
            fold,

        "test_n":
            len(test_df),

        "candidate_train_n":
            len(candidate_train),

        "active_train_n":
            len(train_df),

        "excluded_within_500m_n":
            int(
                len(candidate_train)
                -
                len(train_df)
            ),

        "train_blocks":
            n_train_blocks,

        "test_blocks":
            int(
                test_df[
                    "cv_block_id"
                ].nunique()
            ),

        "minimum_active_train_to_test_distance_m":
            min_active_distance,
    })


    # --------------------------------------------------------
    # Mean-only baseline using active training mean.
    # --------------------------------------------------------

    train_mean = float(
        train_df[
            "agbd"
        ].mean()
    )

    temp = test_df[
        [
            "shot_number",
            "cv_block_id",
            "c1b_outer_fold",
            "agbd",
        ]
    ].copy()

    temp[
        "mean_only_prediction"
    ] = train_mean

    mean_oof_parts.append(
        temp
    )


split_df = pd.DataFrame(
    split_rows
)

split_df.to_csv(
    SPLIT_FILE,
    index=False
)


mean_oof = pd.concat(
    mean_oof_parts,
    ignore_index=True,
)


# ============================================================
# METRICS
# ============================================================

def compute_metrics(
    y,
    p,
):

    y = np.asarray(
        y,
        dtype=float,
    )

    p = np.asarray(
        p,
        dtype=float,
    )

    rmse = float(
        np.sqrt(
            mean_squared_error(
                y,
                p,
            )
        )
    )

    mae = float(
        mean_absolute_error(
            y,
            p,
        )
    )

    bias = float(
        np.mean(
            p - y
        )
    )

    r2 = float(
        r2_score(
            y,
            p,
        )
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
        * 100.0
        if observed_mean != 0
        else np.nan
    )


    return {
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

        "observed_mean":
            observed_mean,

        "observed_sd":
            observed_sd,

        "predicted_sd":
            predicted_sd,

        "predicted_to_observed_SD_ratio":
            sd_ratio,

        "rRMSE_percent":
            rrmse,
    }


# ============================================================
# RUN FROZEN ABLATION
# ============================================================

oof_parts = []

fold_metric_rows = []

parameter_rows = []


for family_row in families.itertuples():

    family = family_row.family

    predictors = (
        family_predictors[
            family
        ]
    )


    print("")
    print("=" * 88)
    print(
        "C5 FAMILY:",
        family
    )
    print(
        "Predictors:",
        len(predictors)
    )
    print("=" * 88)


    for fold in outer_folds:

        train_df = (
            split_data[
                fold
            ][
                "train"
            ]
        )

        test_df = (
            split_data[
                fold
            ][
                "test"
            ]
        )


        X_train = (
            train_df[
                predictors
            ]
            .to_numpy(float)
        )

        y_train = (
            train_df[
                "agbd"
            ]
            .to_numpy(float)
        )

        groups_train = (
            train_df[
                "cv_block_id"
            ]
            .to_numpy()
        )

        X_test = (
            test_df[
                predictors
            ]
            .to_numpy(float)
        )

        y_test = (
            test_df[
                "agbd"
            ]
            .to_numpy(float)
        )


        inner_cv = GroupKFold(
            n_splits=INNER_SPLITS
        )


        base_model = ExtraTreesRegressor(
            random_state=RANDOM_SEED,
            n_jobs=1,
        )


        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=PARAM_DISTRIBUTIONS,
            n_iter=N_SEARCH_ITER,
            scoring="neg_root_mean_squared_error",
            cv=inner_cv,
            random_state=RANDOM_SEED,
            n_jobs=1,
            refit=True,
            return_train_score=False,
        )


        search.fit(
            X_train,
            y_train,
            groups=groups_train,
        )


        pred = search.predict(
            X_test
        )


        fold_metrics = compute_metrics(
            y_test,
            pred,
        )


        fold_metric_rows.append({
            "family":
                family,

            "outer_fold":
                fold,

            "n_predictors":
                len(predictors),

            "predictors":
                ";".join(
                    predictors
                ),

            "train_n":
                len(train_df),

            "test_n":
                len(test_df),

            "train_blocks":
                int(
                    train_df[
                        "cv_block_id"
                    ].nunique()
                ),

            "test_blocks":
                int(
                    test_df[
                        "cv_block_id"
                    ].nunique()
                ),

            "best_inner_RMSE":
                float(
                    -search.best_score_
                ),

            **fold_metrics,
        })


        parameter_rows.append({
            "family":
                family,

            "outer_fold":
                fold,

            "best_inner_RMSE":
                float(
                    -search.best_score_
                ),

            "best_params_json":
                json.dumps(
                    search.best_params_,
                    sort_keys=True,
                ),
        })


        temp = test_df[
            [
                "shot_number",
                "cv_block_id",
                "c1b_outer_fold",
                "agbd",
                "c1b_predicted_agbd",
            ]
        ].copy()


        temp = temp.rename(
            columns={
                "c1b_outer_fold":
                    "outer_fold",

                "agbd":
                    "observed_agbd",
            }
        )


        temp[
            "family"
        ] = family

        temp[
            "predicted_agbd"
        ] = pred


        oof_parts.append(
            temp
        )


        print(
            f"Fold {fold}: "
            f"train={len(train_df)}, "
            f"test={len(test_df)}, "
            f"inner RMSE="
            f"{-search.best_score_:.4f}, "
            f"outer RMSE="
            f"{fold_metrics['RMSE']:.4f}"
        )


# ============================================================
# SAVE OOF / FOLD RESULTS
# ============================================================

oof_all = pd.concat(
    oof_parts,
    ignore_index=True,
)

fold_metrics_df = pd.DataFrame(
    fold_metric_rows
)

parameters_df = pd.DataFrame(
    parameter_rows
)


expected_oof_rows = (
    1641
    *
    len(families)
)

if len(oof_all) != expected_oof_rows:

    raise RuntimeError(
        f"Expected {expected_oof_rows} total OOF rows; "
        f"found {len(oof_all)}."
    )


for family in family_predictors:

    g = oof_all.loc[
        oof_all[
            "family"
        ]
        ==
        family
    ]

    if len(g) != 1641:
        raise RuntimeError(
            f"{family}: expected 1641 OOF rows; "
            f"found {len(g)}."
        )

    if (
        g[
            "shot_number"
        ].nunique()
        != 1641
    ):
        raise RuntimeError(
            f"{family}: OOF shots not unique."
        )


oof_all.to_csv(
    OOF_FILE,
    index=False
)

fold_metrics_df.to_csv(
    FOLD_METRIC_FILE,
    index=False
)

parameters_df.to_csv(
    PARAM_FILE,
    index=False
)


# ============================================================
# POOLED FAMILY METRICS
# ============================================================

summary_rows = []


# Mean-only evaluation baseline
baseline_metrics = compute_metrics(
    mean_oof[
        "agbd"
    ],
    mean_oof[
        "mean_only_prediction"
    ],
)


summary_rows.append({
    "family":
        "EVAL_BASELINE_MEAN_ONLY",

    "role":
        "evaluation_baseline",

    "n_predictors":
        0,

    "n":
        len(mean_oof),

    **baseline_metrics,
})


role_map = dict(
    zip(
        families[
            "family"
        ],
        families[
            "role"
        ],
    )
)


for family in families[
    "family"
]:

    g = oof_all.loc[
        oof_all[
            "family"
        ]
        ==
        family
    ]

    metrics = compute_metrics(
        g[
            "observed_agbd"
        ],
        g[
            "predicted_agbd"
        ],
    )

    summary_rows.append({
        "family":
            family,

        "role":
            role_map[
                family
            ],

        "n_predictors":
            len(
                family_predictors[
                    family
                ]
            ),

        "n":
            len(g),

        **metrics,
    })


summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    SUMMARY_FILE,
    index=False
)


# ============================================================
# A11 REPRODUCTION GATE AGAINST FROZEN C1B OOF
# ============================================================

a11 = (
    oof_all.loc[
        oof_all[
            "family"
        ]
        ==
        "A11_FULL_14P"
    ]
    .copy()
)


prediction_difference = (
    a11[
        "predicted_agbd"
    ]
    -
    a11[
        "c1b_predicted_agbd"
    ]
)


max_abs_prediction_difference = float(
    np.abs(
        prediction_difference
    ).max()
)


a11_metrics = compute_metrics(
    a11[
        "observed_agbd"
    ],
    a11[
        "predicted_agbd"
    ],
)


c1b_metrics = compute_metrics(
    a11[
        "observed_agbd"
    ],
    a11[
        "c1b_predicted_agbd"
    ],
)


reproduction_pass = bool(
    max_abs_prediction_difference
    <=
    1e-8
)


repro_df = pd.DataFrame([
    {
        "n":
            len(a11),

        "max_abs_prediction_difference":
            max_abs_prediction_difference,

        "C5_A11_RMSE":
            a11_metrics[
                "RMSE"
            ],

        "C1B_frozen_RMSE":
            c1b_metrics[
                "RMSE"
            ],

        "RMSE_difference":
            (
                a11_metrics[
                    "RMSE"
                ]
                -
                c1b_metrics[
                    "RMSE"
                ]
            ),

        "C5_A11_MAE":
            a11_metrics[
                "MAE"
            ],

        "C1B_frozen_MAE":
            c1b_metrics[
                "MAE"
            ],

        "C5_A11_R2":
            a11_metrics[
                "R2"
            ],

        "C1B_frozen_R2":
            c1b_metrics[
                "R2"
            ],

        "reproduction_tolerance":
            1e-8,

        "REPRODUCTION_PASS":
            reproduction_pass,
    }
])


repro_df.to_csv(
    REPRO_FILE,
    index=False
)


# ============================================================
# PREDECLARED PAIRED BLOCK BOOTSTRAP
# ============================================================

prediction_wide = (
    oof_all.pivot(
        index="shot_number",
        columns="family",
        values="predicted_agbd",
    )
)


base_info = (
    df[
        [
            "shot_number",
            "cv_block_id",
            "agbd",
        ]
    ]
    .copy()
)


base_info = (
    base_info
    .set_index(
        "shot_number"
    )
    .join(
        prediction_wide,
        how="inner",
    )
    .reset_index()
)


if len(base_info) != 1641:
    raise RuntimeError(
        "Paired bootstrap table does not contain "
        "1641 shots."
    )


blocks = (
    base_info[
        "cv_block_id"
    ]
    .drop_duplicates()
    .to_numpy()
)


rng = np.random.default_rng(
    BOOTSTRAP_SEED
)


def contrast_metrics(
    data,
    candidate,
    reference,
):

    y = (
        data[
            "agbd"
        ]
        .to_numpy(float)
    )

    pc = (
        data[
            candidate
        ]
        .to_numpy(float)
    )

    pr = (
        data[
            reference
        ]
        .to_numpy(float)
    )


    candidate_metrics = compute_metrics(
        y,
        pc,
    )

    reference_metrics = compute_metrics(
        y,
        pr,
    )


    return {
        "delta_RMSE_candidate_minus_reference":
            (
                candidate_metrics[
                    "RMSE"
                ]
                -
                reference_metrics[
                    "RMSE"
                ]
            ),

        "delta_MAE_candidate_minus_reference":
            (
                candidate_metrics[
                    "MAE"
                ]
                -
                reference_metrics[
                    "MAE"
                ]
            ),

        "delta_R2_candidate_minus_reference":
            (
                candidate_metrics[
                    "R2"
                ]
                -
                reference_metrics[
                    "R2"
                ]
            ),
    }


contrast_rows = []


for (
    contrast_name,
    candidate,
    reference,
) in CONTRASTS:

    full = contrast_metrics(
        base_info,
        candidate,
        reference,
    )


    boot_values = {
        "delta_RMSE_candidate_minus_reference":
            [],

        "delta_MAE_candidate_minus_reference":
            [],

        "delta_R2_candidate_minus_reference":
            [],
    }


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
                base_info.loc[
                    base_info[
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


        metrics = contrast_metrics(
            boot,
            candidate,
            reference,
        )


        for key in boot_values:

            boot_values[
                key
            ].append(
                metrics[
                    key
                ]
            )


    for metric_name in [
        "delta_RMSE_candidate_minus_reference",
        "delta_MAE_candidate_minus_reference",
        "delta_R2_candidate_minus_reference",
    ]:

        values = np.asarray(
            boot_values[
                metric_name
            ],
            dtype=float,
        )

        contrast_rows.append({
            "contrast":
                contrast_name,

            "candidate":
                candidate,

            "reference":
                reference,

            "metric":
                metric_name,

            "estimate":
                float(
                    full[
                        metric_name
                    ]
                ),

            "CI_2.5":
                float(
                    np.quantile(
                        values,
                        0.025,
                    )
                ),

            "CI_97.5":
                float(
                    np.quantile(
                        values,
                        0.975,
                    )
                ),

            "bootstrap_reps":
                BOOTSTRAP_REPS,
        })


contrast_df = pd.DataFrame(
    contrast_rows
)

contrast_df.to_csv(
    CONTRAST_FILE,
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
        "=" * 110
        + "\n"
    )

    f.write(
        "C5 STEP 1 - PREDECLARED EXTRATREES "
        "MULTISCALE ABLATION\n"
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
        "All predictor families were frozen in C5 Step 0 "
        "before performance was examined.\n"
    )

    f.write(
        "Extra Trees only.\n"
    )

    f.write(
        "Five frozen C1B geographic outer folds.\n"
    )

    f.write(
        "500-m train-to-test exclusion buffer.\n"
    )

    f.write(
        "Inner 4-fold GroupKFold by cv_block_id.\n"
    )

    f.write(
        "RandomizedSearchCV iterations = 18.\n"
    )

    f.write(
        "Random seed = 42.\n"
    )

    f.write(
        "C1-C4 remain frozen.\n\n"
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
        f"Outer folds: {outer_folds}\n"
    )

    f.write(
        f"Maximum response merge difference: "
        f"{MAX_RESPONSE_DIFF:.16g}\n\n"
    )


    f.write(
        "OUTER SPLIT / BUFFER QC\n"
    )

    f.write(
        "-" * 110
        + "\n"
    )

    f.write(
        split_df.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )


    f.write(
        "A11 FULL14P REPRODUCTION GATE\n"
    )

    f.write(
        "-" * 110
        + "\n"
    )

    f.write(
        repro_df.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )


    f.write(
        "POOLED PERFORMANCE\n"
    )

    f.write(
        "-" * 110
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
        "FOLD PERFORMANCE\n"
    )

    f.write(
        "-" * 110
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
        "BEST HYPERPARAMETERS\n"
    )

    f.write(
        "-" * 110
        + "\n"
    )

    f.write(
        parameters_df.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )


    f.write(
        "PREDECLARED PAIRED BLOCK-BOOTSTRAP CONTRASTS\n"
    )

    f.write(
        "-" * 110
        + "\n"
    )

    f.write(
        contrast_df.to_string(
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
        "-" * 110
        + "\n"
    )

    f.write(
        "Negative candidate-minus-reference delta RMSE "
        "or delta MAE indicates candidate improvement.\n"
    )

    f.write(
        "Positive candidate-minus-reference delta R2 "
        "indicates candidate improvement.\n"
    )

    f.write(
        "A09 versus A08 is the primary matched "
        "9x9-versus-5x5 support comparison.\n"
    )

    f.write(
        "A04/A05/A06 versus A03 separate contextual HV, "
        "contextual means, and contextual heterogeneity.\n"
    )

    f.write(
        "A00 is a spatial-location diagnostic only.\n"
    )

    f.write(
        "A11 must reproduce frozen C1B Full14P OOF "
        "predictions within tolerance before C5 is "
        "interpreted as protocol-consistent.\n"
    )

    f.write(
        "No new predictor family will be created after "
        "viewing these results.\n"
    )


# ============================================================
# CONSOLE
# ============================================================

print("")
print("=" * 92)
print("C5 STEP 1 COMPLETE")
print("PREDECLARED EXTRATREES MULTISCALE ABLATION")
print("=" * 92)

print("")
print("Outer split QC:")
print(
    split_df.to_string(
        index=False
    )
)

print("")
print("A11 reproduction gate:")
print(
    repro_df.to_string(
        index=False
    )
)

print("")
print("Pooled metrics:")
print(
    summary_df.to_string(
        index=False
    )
)

print("")
print("Predeclared contrasts:")
print(
    contrast_df.to_string(
        index=False
    )
)

print("")
print("Report:")
print(REPORT)

print("")
if reproduction_pass:
    print(
        "A11 C1B REPRODUCTION: PASS"
    )
else:
    print(
        "A11 C1B REPRODUCTION: FAIL"
    )
    print(
        "DO NOT INTERPRET C5 UNTIL THE "
        "PROTOCOL DISCREPANCY IS RESOLVED."
    )

print("")
print(
    "C1-C4 REMAIN FROZEN."
)

print("=" * 92)
