from pathlib import Path
import hashlib
import json
import warnings

import numpy as np
import pandas as pd

from scipy.stats import pearsonr, spearmanr

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
    / "Buffer_Sensitivity"
)

OUT = (
    C2_ROOT
    / "C2_02A_Buffer_Sensitivity"
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

FOLD_METRICS = (
    OUT
    / "C2_02A_BUFFER_FOLD_METRICS.csv"
)

POOLED_METRICS = (
    OUT
    / "C2_02A_BUFFER_POOLED_METRICS.csv"
)

OOF_FILE = (
    OUT
    / "C2_02A_BUFFER_OOF.csv"
)

PARAM_FILE = (
    OUT
    / "C2_02A_ET_PARAMETERS.csv"
)

BOOT_FILE = (
    OUT
    / "C2_02A_BUFFER_DELTA_BOOTSTRAP.csv"
)

BOOT_SUMMARY = (
    OUT
    / "C2_02A_BUFFER_DELTA_BOOTSTRAP_SUMMARY.csv"
)

REPORT = (
    OUT
    / "C2_02A_BUFFER_SENSITIVITY_REPORT.txt"
)


# ============================================================
# FROZEN C2 MODEL
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

BUFFERS = [
    500,
    1000,
    2000,
    3000,
]

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


def json_safe(obj):

    if isinstance(
        obj,
        np.integer
    ):
        return int(obj)

    if isinstance(
        obj,
        np.floating
    ):
        return float(obj)

    if isinstance(
        obj,
        tuple
    ):
        return list(obj)

    return obj


def params_json(params):

    return json.dumps(
        {
            k: json_safe(v)
            for k, v
            in params.items()
        },
        sort_keys=True
    )


def metrics(
    y,
    p,
    label,
    buffer_m,
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
            label,

        "buffer_m":
            buffer_m,

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
    membership_path,
    master
):

    m = pd.read_csv(
        membership_path
    )

    ids = set(
        m[
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

    if len(out) != len(m):

        raise RuntimeError(
            f"Membership mismatch for "
            f"{membership_path}"
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

    n_groups = (
        pd.Series(
            groups
        )
        .nunique()
    )

    if n_groups < INNER_FOLDS:

        raise RuntimeError(
            "Too few training blocks "
            "for four-fold GroupKFold."
        )

    base = ExtraTreesRegressor(
        random_state=RANDOM_SEED,
        n_jobs=N_JOBS,
        criterion="squared_error",
    )

    inner = GroupKFold(
        n_splits=INNER_FOLDS
    )

    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=ET_PARAMS,
        n_iter=N_SEARCH_ITER,
        scoring=
            "neg_root_mean_squared_error",
        cv=inner,
        random_state=RANDOM_SEED,
        n_jobs=N_JOBS,
        refit=True,
        return_train_score=False,
        verbose=0,
        error_score="raise",
    )

    search.fit(
        X_train,
        y_train,
        groups=groups,
    )

    best = search.best_estimator_

    pred = best.predict(
        X_test
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
# LOAD FROZEN MASTER TABLE
# ============================================================

master = pd.read_csv(
    DATA
)

if len(master) != 1641:

    raise RuntimeError(
        f"Expected 1641 rows, "
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


# ============================================================
# MAIN BUFFER LOOP
# ============================================================

fold_rows = []

param_rows = []

oof_parts = []


for buffer_m in BUFFERS:

    print("")
    print(
        "=" * 78
    )

    print(
        f"C2 BUFFER SENSITIVITY: "
        f"{buffer_m} m"
    )

    print(
        "=" * 78
    )

    buffer_dir = (
        GEOM_ROOT
        / f"Buffer_{buffer_m}m"
    )

    for fold in range(
        1,
        6
    ):

        print(
            f"  Fold {fold}...",
            flush=True
        )

        train_path = (
            buffer_dir
            / (
                f"Fold_{fold}_"
                f"TRAIN_MEMBERSHIP.csv"
            )
        )

        test_path = (
            buffer_dir
            / (
                f"Fold_{fold}_"
                f"TEST_MEMBERSHIP.csv"
            )
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
                f"Buffer {buffer_m}, "
                f"fold {fold}: "
                "train/test overlap."
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

        # Training-mean baseline
        train_mean = float(
            train[
                RESPONSE
            ].mean()
        )

        mean_pred = np.full(
            len(test),
            train_mean,
            dtype=float,
        )

        m = metrics(
            y_test,
            pred,
            "ET14P",
            buffer_m,
            fold,
            train_n=len(train),
            test_n=len(test),
        )

        mb = metrics(
            y_test,
            mean_pred,
            "MEAN_ONLY",
            buffer_m,
            fold,
            train_n=len(train),
            test_n=len(test),
        )

        fold_rows.extend(
            [
                m,
                mb,
            ]
        )

        param_rows.append({
            "buffer_m":
                buffer_m,

            "fold":
                fold,

            "train_n":
                len(train),

            "train_blocks":
                train_blocks,

            "test_n":
                len(test),

            "test_blocks":
                test[
                    "cv_block_id"
                ].nunique(),

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

                "buffer_m":
                    buffer_m,

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
            f"blocks={train_blocks:2d}, "
            f"inner RMSE={inner_rmse:.3f}, "
            f"outer RMSE="
            f"{m['RMSE_Mg_ha']:.3f}, "
            f"R2={m['R2']:.4f}"
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
# OOF INTEGRITY
# ============================================================

for buffer_m in BUFFERS:

    g = (
        oof[
            oof[
                "buffer_m"
            ] == buffer_m
        ]
    )

    if len(g) != 1641:

        raise RuntimeError(
            f"Buffer {buffer_m}: "
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
            f"Buffer {buffer_m}: "
            "OOF shots not unique."
        )


# ============================================================
# POOLED METRICS
# ============================================================

pooled_rows = []

for buffer_m in BUFFERS:

    g = (
        oof[
            oof[
                "buffer_m"
            ] == buffer_m
        ]
        .copy()
    )

    pooled_rows.append(
        metrics(
            g[
                "observed_agbd"
            ],
            g[
                "predicted_agbd"
            ],
            "ET14P",
            buffer_m,
            "POOLED",
            train_n=np.nan,
            test_n=1641,
        )
    )

    pooled_rows.append(
        metrics(
            g[
                "observed_agbd"
            ],
            g[
                "mean_only_prediction"
            ],
            "MEAN_ONLY",
            buffer_m,
            "POOLED",
            train_n=np.nan,
            test_n=1641,
        )
    )


pooled_df = pd.DataFrame(
    pooled_rows
)


# ============================================================
# QC: 500-M RUN SHOULD REPRODUCE C1B
# ============================================================

qc_text = []

if C1B_OOF.exists():

    c1b = pd.read_csv(
        C1B_OOF
    )

    current500 = (
        oof[
            oof[
                "buffer_m"
            ] == 500
        ]
        [
            [
                "shot_number",
                "predicted_agbd",
            ]
        ]
        .rename(
            columns={
                "predicted_agbd":
                    "pred_c2"
            }
        )
    )

    if (
        "predicted_agbd"
        not in c1b.columns
    ):

        raise RuntimeError(
            "C1B OOF file lacks "
            "predicted_agbd."
        )

    comp = (
        current500.merge(
            c1b[
                [
                    "shot_number",
                    "predicted_agbd",
                ]
            ].rename(
                columns={
                    "predicted_agbd":
                        "pred_c1b"
                }
            ),
            on="shot_number",
            how="inner",
        )
    )

    if len(comp) != 1641:

        raise RuntimeError(
            "C1B/C2 500-m merge "
            "did not produce 1641 rows."
        )

    comp[
        "abs_diff"
    ] = np.abs(
        comp[
            "pred_c2"
        ]
        - comp[
            "pred_c1b"
        ]
    )

    max_diff = float(
        comp[
            "abs_diff"
        ].max()
    )

    mean_diff = float(
        comp[
            "abs_diff"
        ].mean()
    )

    qc_text.append(
        f"C1B 500-m reproduction "
        f"max abs prediction difference: "
        f"{max_diff:.12g}"
    )

    qc_text.append(
        f"C1B 500-m reproduction "
        f"mean abs prediction difference: "
        f"{mean_diff:.12g}"
    )

else:

    qc_text.append(
        "C1B OOF file not found; "
        "prediction-level reproduction "
        "check skipped."
    )


# ============================================================
# PAIRED 54-BLOCK BOOTSTRAP
#
# Compare 1000/2000/3000 m predictions against 500 m
# on exactly the same 1641 observations.
# ============================================================

wide = (
    oof.pivot(
        index=[
            "shot_number",
            "cv_block_id",
            "observed_agbd",
        ],
        columns="buffer_m",
        values="predicted_agbd",
    )
    .reset_index()
)

required_pred_cols = BUFFERS

for b in required_pred_cols:

    if b not in wide.columns:

        raise RuntimeError(
            f"Missing prediction column "
            f"for buffer {b}."
        )


block_parts = []

for block_id, g in wide.groupby(
    "cv_block_id"
):

    y = (
        g[
            "observed_agbd"
        ]
        .to_numpy(float)
    )

    row = {
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
    }

    for b in BUFFERS:

        p = (
            g[b]
            .to_numpy(float)
        )

        row[
            f"sse_{b}"
        ] = float(
            np.sum(
                (p - y) ** 2
            )
        )

        row[
            f"sae_{b}"
        ] = float(
            np.sum(
                np.abs(
                    p - y
                )
            )
        )

    block_parts.append(
        row
    )


block_df = pd.DataFrame(
    block_parts
)

if len(block_df) != 54:

    raise RuntimeError(
        f"Expected 54 blocks, "
        f"found {len(block_df)}."
    )


rng = np.random.default_rng(
    BOOT_SEED
)

boot_rows = []

B = len(
    block_df
)

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

    sst = (
        sy2
        - sy ** 2 / n
    )

    rmse = {}

    mae = {}

    r2 = {}

    for b in BUFFERS:

        sse = float(
            s[
                f"sse_{b}"
            ].sum()
        )

        sae = float(
            s[
                f"sae_{b}"
            ].sum()
        )

        rmse[b] = np.sqrt(
            sse / n
        )

        mae[b] = (
            sae / n
        )

        r2[b] = (
            1.0
            - sse / sst
            if sst > 0
            else np.nan
        )

    for b in [
        1000,
        2000,
        3000,
    ]:

        boot_rows.append({
            "replicate":
                rep,

            "comparison_buffer_m":
                b,

            "delta_RMSE_vs_500m":
                (
                    rmse[b]
                    - rmse[500]
                ),

            "delta_MAE_vs_500m":
                (
                    mae[b]
                    - mae[500]
                ),

            "delta_R2_vs_500m":
                (
                    r2[b]
                    - r2[500]
                ),
        })


boot_df = pd.DataFrame(
    boot_rows
)


# ============================================================
# BOOTSTRAP SUMMARY
# ============================================================

summary_rows = []

for b in [
    1000,
    2000,
    3000,
]:

    g = (
        boot_df[
            boot_df[
                "comparison_buffer_m"
            ] == b
        ]
    )

    for metric in [
        "delta_RMSE_vs_500m",
        "delta_MAE_vs_500m",
        "delta_R2_vs_500m",
    ]:

        v = (
            g[
                metric
            ]
            .to_numpy(float)
        )

        summary_rows.append({
            "comparison":
                f"{b}m_minus_500m",

            "metric":
                metric,

            "mean":
                float(
                    np.mean(v)
                ),

            "median":
                float(
                    np.median(v)
                ),

            "CI_2.5":
                float(
                    np.quantile(
                        v,
                        0.025
                    )
                ),

            "CI_97.5":
                float(
                    np.quantile(
                        v,
                        0.975
                    )
                ),

            "n_replicates":
                BOOT_REPS,

            "n_blocks":
                54,
        })


boot_summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# SAVE TABLES
# ============================================================

fold_df.to_csv(
    FOLD_METRICS,
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
        "=" * 94
        + "\n"
    )

    f.write(
        "C2 STEP 2A — "
        "TRAIN–TEST EXCLUSION-BUFFER SENSITIVITY\n"
    )

    f.write(
        "=" * 94
        + "\n\n"
    )

    f.write(
        "DESIGN\n"
    )

    f.write(
        "-" * 94
        + "\n"
    )

    f.write(
        "Frozen contiguous C1 geographic "
        "outer-test folds were retained exactly.\n"
    )

    f.write(
        "Only the minimum train-to-test "
        "exclusion distance varied: "
        "500, 1000, 2000, or 3000 m.\n"
    )

    f.write(
        "The model family was frozen before "
        "C2 performance was observed: "
        "Extra Trees with the full 14P "
        "multiscale predictor set.\n"
    )

    f.write(
        "Within each active training sample, "
        "Extra Trees hyperparameters were "
        "retuned using four-fold GroupKFold "
        "by 3-km cv_block_id.\n"
    )

    f.write(
        "No model-family or predictor-family "
        "selection occurred in C2.\n\n"
    )

    f.write(
        "INPUT SHA256\n"
    )

    f.write(
        "-" * 94
        + "\n"
    )

    f.write(
        sha256_file(
            DATA
        )
        + "\n\n"
    )

    f.write(
        "500-M C1B REPRODUCTION QC\n"
    )

    f.write(
        "-" * 94
        + "\n"
    )

    for line in qc_text:

        f.write(
            line
            + "\n"
        )

    f.write(
        "\n"
    )

    f.write(
        "POOLED PERFORMANCE\n"
    )

    f.write(
        "-" * 94
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
        "PER-FOLD PERFORMANCE\n"
    )

    f.write(
        "-" * 94
        + "\n"
    )

    f.write(
        fold_df.to_string(
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
        "-" * 94
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
        "PAIRED 54-BLOCK BOOTSTRAP: "
        "LARGER BUFFER MINUS 500 M\n"
    )

    f.write(
        "-" * 94
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
        "INTERPRETIVE RULE\n"
    )

    f.write(
        "-" * 94
        + "\n"
    )

    f.write(
        "Positive delta_RMSE indicates worse "
        "prediction after increasing spatial "
        "separation; negative delta_RMSE "
        "indicates improvement.\n"
    )

    f.write(
        "The direction, magnitude, and confidence "
        "intervals should be interpreted together; "
        "no buffer will be chosen retrospectively "
        "as a new primary validation design.\n"
    )


# ============================================================
# CONSOLE
# ============================================================

print("")
print(
    "=" * 80
)

print(
    "C2 STEP 2A COMPLETE"
)

print(
    "BUFFER SENSITIVITY"
)

print(
    "=" * 80
)

print("")
print(
    "500-M C1B REPRODUCTION QC"
)

for line in qc_text:

    print(line)

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
    "PAIRED BLOCK-BOOTSTRAP SUMMARY"
)

print(
    boot_summary_df.to_string(
        index=False
    )
)

print("")
print(
    "OOF rows:",
    len(oof)
)

print(
    "Expected OOF rows:",
    1641 * len(BUFFERS)
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
