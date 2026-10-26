from pathlib import Path
from datetime import datetime
import hashlib
import json
import os
import sys
import traceback
import warnings

# Windows-safe / low-memory execution
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, ParameterSampler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# PHASE 8C.4 — CONTROLLED SPATIAL-CV RERUN WITH LOCKED 9P FEATURE SET
# ============================================================
PROJECT = Path(os.environ.get("NISAR_PROJECT_ROOT", r"D:\NISAR_Mangrove_Paper"))

EXPANDED_TABLE = (
    PROJECT
    / "04_Matched_Data"
    / "Predictor_Expansion"
    / "Phase8C2"
    / "Tables"
    / "GEDI_NISAR_expanded_predictors_1641.csv"
)
EXPECTED_EXPANDED_SHA256 = (
    "de9fc736ec77d38dae527020b2d037c4567f3c814f874d9843da230c6bf67dc0"
)

BUFFER_ROOT = (
    PROJECT
    / "05_ML_DL"
    / "Spatial_CV"
    / "Buffered_500m"
)

OUT_ROOT = (
    PROJECT
    / "06_Results"
    / "Predictor_Expansion"
    / "Phase8C4_Expanded9P"
)
TABLE_DIR = OUT_ROOT / "Tables"
FIG_DIR = OUT_ROOT / "Figures"
QC_DIR = OUT_ROOT / "QC"
for d in [TABLE_DIR, FIG_DIR, QC_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Original September 3P OOF predictions for the benchmark.
OLD_BASELINE_OOF = (
    PROJECT / "06_Results" / "Baseline" / "Tables"
    / "baseline_OOF_predictions.csv"
)
OLD_ET_OOF = (
    PROJECT / "06_Results" / "Extra_Trees" / "Tables"
    / "ET_OOF_predictions.csv"
)
OLD_MLP_OOF = (
    PROJECT / "06_Results" / "DNN_MLP" / "Tables"
    / "MLP_OOF_predictions.csv"
)

EXPECTED_N = 1641
RANDOM_SEED = 42
INNER_SPLITS = 4

# ============================================================
# LOCKED PREDICTOR STRATEGY
# ============================================================
# This 9P set is locked BEFORE Phase 8C.4 response-based modelling.
# Selection rationale:
# - retain accepted old 3P benchmark
# - retain HH fine-scale heterogeneity to complement legacy HV_SD
# - use 9x9 contextual means and heterogeneity as the broad spatial scale
# - retain 9x9 polarization-contrast heterogeneity
# - omit 5x5 family because it carried the highest VIF-like redundancy in the
#   full 14-variable candidate space and represents an intermediate version of
#   the same constructs
# - omit HH_minus_HV exact algebraic redundancy
FEATURES = [
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

# Extra Trees search space: modest, low-memory, training-only.
ET_PARAM_SPACE = {
    "n_estimators": [400, 700, 1000],
    "max_depth": [6, 8, 12, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 3, 6, 10],
    "max_features": ["sqrt", 0.5, 0.8, 1.0],
    "bootstrap": [True, False],
}

# MLP search space: modest and training-only.
MLP_PARAM_SPACE = {
    "hidden_layer_sizes": [(32,), (64,), (32, 16), (64, 32)],
    "activation": ["relu", "tanh"],
    "alpha": [0.001, 0.01, 0.1, 1.0],
    "learning_rate_init": [0.0005, 0.001, 0.003],
}

ET_N_ITER = 18
MLP_N_ITER = 12

# ============================================================
# HELPERS
# ============================================================
def sha256_file(path, chunk_size=8 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def metric_row(y, p, model, fold=None, train_n=None, test_n=None):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    rmse = float(np.sqrt(mean_squared_error(y, p)))
    mae = float(mean_absolute_error(y, p))
    r2 = float(r2_score(y, p))
    bias = float(np.mean(p - y))
    mean_y = float(np.mean(y))
    rrmse = float(rmse / mean_y * 100.0) if mean_y != 0 else np.nan

    if len(y) >= 3 and np.std(y) > 0 and np.std(p) > 0:
        pr = float(pearsonr(y, p).statistic)
        sr = float(spearmanr(y, p).statistic)
    else:
        pr = np.nan
        sr = np.nan

    obs_sd = float(np.std(y, ddof=1))
    pred_sd = float(np.std(p, ddof=1))

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
        "pred_to_obs_SD_ratio": pred_sd / obs_sd if obs_sd > 0 else np.nan,
    }

def shot_set(path):
    d = pd.read_csv(path)
    if "shot_number" not in d.columns:
        raise RuntimeError(f"shot_number missing: {path}")
    return set(pd.to_numeric(d["shot_number"], errors="raise").astype("int64"))

def fold_files(fold):
    fd = BUFFER_ROOT / f"Fold_{fold}"
    train = fd / f"Fold_{fold}_TRAIN_BUFFERED_500m.csv"
    test = fd / f"Fold_{fold}_TEST.csv"
    if not train.exists() or not test.exists():
        raise FileNotFoundError(
            f"Missing fold {fold} buffered files:\n{train}\n{test}"
        )
    return train, test

def grouped_rmse_linear(X, y, groups):
    cv = GroupKFold(n_splits=INNER_SPLITS)
    vals = []
    for tr, va in cv.split(X, y, groups):
        m = LinearRegression()
        m.fit(X[tr], y[tr])
        pred = m.predict(X[va])
        vals.append(np.sqrt(mean_squared_error(y[va], pred)))
    return float(np.mean(vals))

def tune_et(X, y, groups, fold):
    candidates = list(ParameterSampler(
        ET_PARAM_SPACE, n_iter=ET_N_ITER,
        random_state=RANDOM_SEED + fold
    ))
    cv = GroupKFold(n_splits=INNER_SPLITS)

    best = None
    rows = []

    for ci, params in enumerate(candidates, start=1):
        rmses = []
        for tr, va in cv.split(X, y, groups):
            m = ExtraTreesRegressor(
                **params,
                random_state=RANDOM_SEED,
                n_jobs=1,
            )
            m.fit(X[tr], y[tr])
            pred = m.predict(X[va])
            rmses.append(np.sqrt(mean_squared_error(y[va], pred)))

        mean_rmse = float(np.mean(rmses))
        row = {
            "outer_fold": fold,
            "candidate": ci,
            "mean_inner_RMSE": mean_rmse,
            "params_json": json.dumps(params, sort_keys=True),
        }
        rows.append(row)

        if best is None or mean_rmse < best["mean_inner_RMSE"]:
            best = {**row, "params": params}

    return best, pd.DataFrame(rows)

def make_mlp(params):
    return Pipeline([
        ("scale", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=params["hidden_layer_sizes"],
            activation=params["activation"],
            alpha=params["alpha"],
            learning_rate_init=params["learning_rate_init"],
            solver="adam",
            max_iter=1500,
            early_stopping=False,
            random_state=RANDOM_SEED,
        )),
    ])

def tune_mlp(X, y, groups, fold):
    candidates = list(ParameterSampler(
        MLP_PARAM_SPACE, n_iter=MLP_N_ITER,
        random_state=RANDOM_SEED + 100 + fold
    ))
    cv = GroupKFold(n_splits=INNER_SPLITS)

    best = None
    rows = []

    for ci, params in enumerate(candidates, start=1):
        rmses = []
        for tr, va in cv.split(X, y, groups):
            m = make_mlp(params)
            m.fit(X[tr], y[tr])
            pred = m.predict(X[va])
            rmses.append(np.sqrt(mean_squared_error(y[va], pred)))

        mean_rmse = float(np.mean(rmses))
        safe_params = {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in params.items()
        }
        row = {
            "outer_fold": fold,
            "candidate": ci,
            "mean_inner_RMSE": mean_rmse,
            "params_json": json.dumps(safe_params, sort_keys=True),
        }
        rows.append(row)

        if best is None or mean_rmse < best["mean_inner_RMSE"]:
            best = {**row, "params": params}

    return best, pd.DataFrame(rows)

def old_oof_reference():
    parts = []

    if OLD_BASELINE_OOF.exists():
        d = pd.read_csv(OLD_BASELINE_OOF)
        if {"shot_number", "observed_agbd", "predicted_agbd_baseline"}.issubset(d.columns):
            x = d[["shot_number", "observed_agbd", "predicted_agbd_baseline"]].copy()
            x["model"] = "OLD_Linear_3P"
            x = x.rename(columns={"predicted_agbd_baseline": "predicted_agbd"})
            parts.append(x)

    if OLD_ET_OOF.exists():
        d = pd.read_csv(OLD_ET_OOF)
        if {"model", "shot_number", "observed_agbd", "predicted_agbd"}.issubset(d.columns):
            x = d[d["model"] == "ET_3P_SENSITIVITY"][
                ["shot_number", "observed_agbd", "predicted_agbd"]
            ].copy()
            x["model"] = "OLD_ET_3P"
            parts.append(x)

    if OLD_MLP_OOF.exists():
        d = pd.read_csv(OLD_MLP_OOF)
        if {"model", "shot_number", "observed_agbd", "predicted_agbd"}.issubset(d.columns):
            x = d[d["model"] == "MLP_3P_SENSITIVITY"][
                ["shot_number", "observed_agbd", "predicted_agbd"]
            ].copy()
            x["model"] = "OLD_MLP_3P"
            parts.append(x)

    if not parts:
        raise RuntimeError("Could not load old 3P OOF references.")

    out = pd.concat(parts, ignore_index=True)
    return out

def class_metrics(oof):
    bins = [0, 50, 100, np.inf]
    labels = ["<50", "50-<100", ">=100"]
    d = oof.copy()
    d["AGBD_regime"] = pd.cut(
        d["observed_agbd"], bins=bins, labels=labels,
        right=False, include_lowest=True
    )
    rows = []
    for (model, reg), g in d.groupby(["model", "AGBD_regime"], observed=False):
        if len(g) == 0:
            continue
        r = metric_row(g["observed_agbd"], g["predicted_agbd"], model)
        r["AGBD_regime"] = str(reg)
        rows.append(r)
    return pd.DataFrame(rows)

def plot_comparison(comp):
    order = [
        "Linear", "Extra Trees", "MLP"
    ]
    x = np.arange(3)
    width = 0.35

    # RMSE
    fig, ax = plt.subplots(figsize=(8.2, 5.3))
    ax.bar(x - width/2, comp["old_RMSE"], width, label="Original 3P")
    ax.bar(x + width/2, comp["expanded_RMSE"], width, label="Expanded 9P")
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("RMSE (Mg/ha)")
    ax.set_title("Original 3P vs expanded 9P — spatial-CV RMSE", pad=12)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR/"old3P_vs_expanded9P_RMSE.png", dpi=450, bbox_inches="tight")
    fig.savefig(FIG_DIR/"old3P_vs_expanded9P_RMSE.pdf", bbox_inches="tight")
    plt.close(fig)

    # R2
    fig, ax = plt.subplots(figsize=(8.2, 5.3))
    ax.bar(x - width/2, comp["old_R2"], width, label="Original 3P")
    ax.bar(x + width/2, comp["expanded_R2"], width, label="Expanded 9P")
    ax.axhline(0, linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("Pooled out-of-fold R²")
    ax.set_title("Original 3P vs expanded 9P — spatial-CV R²", pad=12)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR/"old3P_vs_expanded9P_R2.png", dpi=450, bbox_inches="tight")
    fig.savefig(FIG_DIR/"old3P_vs_expanded9P_R2.pdf", bbox_inches="tight")
    plt.close(fig)

    # Predicted SD / observed SD
    fig, ax = plt.subplots(figsize=(8.2, 5.3))
    ax.bar(x - width/2, comp["old_SD_ratio"], width, label="Original 3P")
    ax.bar(x + width/2, comp["expanded_SD_ratio"], width, label="Expanded 9P")
    ax.axhline(1.0, linestyle="--", linewidth=1.0, label="Ideal = 1")
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("Predicted SD / observed SD")
    ax.set_title("Prediction dynamic-range recovery", pad=12)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR/"old3P_vs_expanded9P_SD_ratio.png", dpi=450, bbox_inches="tight")
    fig.savefig(FIG_DIR/"old3P_vs_expanded9P_SD_ratio.pdf", bbox_inches="tight")
    plt.close(fig)

def plot_obs_pred(oof, model_name):
    g = oof[oof["model"] == model_name]
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    ax.scatter(g["observed_agbd"], g["predicted_agbd"], s=12, alpha=0.45)
    lo = min(g["observed_agbd"].min(), g["predicted_agbd"].min())
    hi = max(g["observed_agbd"].max(), g["predicted_agbd"].max())
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.1, label="1:1")
    ax.set_xlabel("Observed GEDI AGBD (Mg/ha)")
    ax.set_ylabel("Predicted AGBD (Mg/ha)")
    ax.set_title(f"{model_name} — pooled spatial-CV predictions", pad=12)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.tight_layout()
    safe = model_name.replace(" ", "_")
    fig.savefig(FIG_DIR/f"{safe}_observed_vs_predicted.png", dpi=450, bbox_inches="tight")
    fig.savefig(FIG_DIR/f"{safe}_observed_vs_predicted.pdf", bbox_inches="tight")
    plt.close(fig)

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 88)
    print("PHASE 8C.4 — EXPANDED 9P CONTROLLED SPATIAL-CV MODEL RERUN")
    print("=" * 88)
    print("Locked 9-predictor set; same 3 km folds + same 500 m buffer.")
    print("ET/MLP tuning occurs ONLY inside each buffered training set.\n")

    if not EXPANDED_TABLE.exists():
        raise FileNotFoundError(EXPANDED_TABLE)

    actual_hash = sha256_file(EXPANDED_TABLE)
    print("Expanded table SHA256:", actual_hash)
    if actual_hash != EXPECTED_EXPANDED_SHA256:
        raise RuntimeError(
            "Expanded table hash mismatch.\n"
            f"Expected: {EXPECTED_EXPANDED_SHA256}\n"
            f"Found:    {actual_hash}"
        )

    df = pd.read_csv(EXPANDED_TABLE)
    if len(df) != EXPECTED_N:
        raise RuntimeError(f"Expected {EXPECTED_N} rows; found {len(df)}.")

    required = {"shot_number", "agbd", "cv_block_id"} | set(FEATURES)
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    df["shot_number"] = pd.to_numeric(df["shot_number"], errors="raise").astype("int64")
    if df["shot_number"].duplicated().any():
        raise RuntimeError("Duplicate shot_number in expanded table.")

    for p in FEATURES + ["agbd"]:
        df[p] = pd.to_numeric(df[p], errors="raise")

    oof_parts = []
    fold_rows = []
    et_tune_rows = []
    mlp_tune_rows = []
    selected_rows = []

    for fold in range(1, 6):
        print("\n" + "-" * 88)
        print(f"OUTER FOLD {fold}")
        print("-" * 88)

        train_file, test_file = fold_files(fold)
        train_ids = shot_set(train_file)
        test_ids = shot_set(test_file)

        train = df[df["shot_number"].isin(train_ids)].copy()
        test = df[df["shot_number"].isin(test_ids)].copy()

        if len(train) != len(train_ids) or len(test) != len(test_ids):
            raise RuntimeError(f"Fold {fold}: membership mismatch.")
        if set(train["shot_number"]) & set(test["shot_number"]):
            raise RuntimeError(f"Fold {fold}: train/test overlap.")

        print(f"Train n={len(train):,}; Test n={len(test):,}")

        Xtr = train[FEATURES].to_numpy(float)
        Xte = test[FEATURES].to_numpy(float)
        ytr = train["agbd"].to_numpy(float)
        yte = test["agbd"].to_numpy(float)
        groups = train["cv_block_id"].to_numpy()

        # Linear
        print("  Linear 9P...")
        lin = Pipeline([
            ("scale", StandardScaler()),
            ("model", LinearRegression())
        ])
        lin.fit(Xtr, ytr)
        pred = lin.predict(Xte)
        m = metric_row(
            yte, pred, "EXPANDED_Linear_9P",
            fold=fold, train_n=len(train), test_n=len(test)
        )
        fold_rows.append(m)
        oof_parts.append(pd.DataFrame({
            "shot_number": test["shot_number"].to_numpy(),
            "outer_fold": fold,
            "observed_agbd": yte,
            "predicted_agbd": pred,
            "model": "EXPANDED_Linear_9P",
        }))
        print(f"    RMSE={m['RMSE_Mg_ha']:.3f}, R2={m['R2']:.4f}")

        # Extra Trees
        print(f"  Extra Trees 9P — inner grouped search ({ET_N_ITER} candidates)...")
        et_best, et_trace = tune_et(Xtr, ytr, groups, fold)
        et_tune_rows.append(et_trace)
        et = ExtraTreesRegressor(
            **et_best["params"],
            random_state=RANDOM_SEED,
            n_jobs=1,
        )
        et.fit(Xtr, ytr)
        pred = et.predict(Xte)
        m = metric_row(
            yte, pred, "EXPANDED_ET_9P",
            fold=fold, train_n=len(train), test_n=len(test)
        )
        fold_rows.append(m)
        oof_parts.append(pd.DataFrame({
            "shot_number": test["shot_number"].to_numpy(),
            "outer_fold": fold,
            "observed_agbd": yte,
            "predicted_agbd": pred,
            "model": "EXPANDED_ET_9P",
        }))
        selected_rows.append({
            "outer_fold": fold,
            "model": "EXPANDED_ET_9P",
            "best_inner_RMSE": et_best["mean_inner_RMSE"],
            "best_params_json": et_best["params_json"],
        })
        print(
            f"    RMSE={m['RMSE_Mg_ha']:.3f}, R2={m['R2']:.4f}; "
            f"best inner RMSE={et_best['mean_inner_RMSE']:.3f}"
        )

        # MLP
        print(f"  MLP 9P — inner grouped search ({MLP_N_ITER} candidates)...")
        mlp_best, mlp_trace = tune_mlp(Xtr, ytr, groups, fold)
        mlp_tune_rows.append(mlp_trace)
        mlp = make_mlp(mlp_best["params"])
        mlp.fit(Xtr, ytr)
        pred = mlp.predict(Xte)
        m = metric_row(
            yte, pred, "EXPANDED_MLP_9P",
            fold=fold, train_n=len(train), test_n=len(test)
        )
        fold_rows.append(m)
        oof_parts.append(pd.DataFrame({
            "shot_number": test["shot_number"].to_numpy(),
            "outer_fold": fold,
            "observed_agbd": yte,
            "predicted_agbd": pred,
            "model": "EXPANDED_MLP_9P",
        }))
        safe_params = {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in mlp_best["params"].items()
        }
        selected_rows.append({
            "outer_fold": fold,
            "model": "EXPANDED_MLP_9P",
            "best_inner_RMSE": mlp_best["mean_inner_RMSE"],
            "best_params_json": json.dumps(safe_params, sort_keys=True),
        })
        print(
            f"    RMSE={m['RMSE_Mg_ha']:.3f}, R2={m['R2']:.4f}; "
            f"best inner RMSE={mlp_best['mean_inner_RMSE']:.3f}"
        )

    oof = pd.concat(oof_parts, ignore_index=True)
    chk = oof.groupby("model")["shot_number"].agg(["count", "nunique"])
    if not ((chk["count"] == EXPECTED_N) & (chk["nunique"] == EXPECTED_N)).all():
        raise RuntimeError("OOF completeness check failed.")

    pooled_rows = []
    for model, g in oof.groupby("model"):
        pooled_rows.append(metric_row(
            g["observed_agbd"], g["predicted_agbd"], model
        ))
    pooled = pd.DataFrame(pooled_rows)

    cls = class_metrics(oof)

    # Old reference
    old = old_oof_reference()
    old_pooled_rows = []
    for model, g in old.groupby("model"):
        if len(g) != EXPECTED_N:
            raise RuntimeError(f"{model}: old OOF row count != 1641.")
        old_pooled_rows.append(metric_row(
            g["observed_agbd"], g["predicted_agbd"], model
        ))
    old_pooled = pd.DataFrame(old_pooled_rows)

    map_pairs = [
        ("Linear", "OLD_Linear_3P", "EXPANDED_Linear_9P"),
        ("Extra Trees", "OLD_ET_3P", "EXPANDED_ET_9P"),
        ("MLP", "OLD_MLP_3P", "EXPANDED_MLP_9P"),
    ]

    comp_rows = []
    for family, old_name, new_name in map_pairs:
        o = old_pooled[old_pooled["model"] == old_name].iloc[0]
        n = pooled[pooled["model"] == new_name].iloc[0]
        comp_rows.append({
            "model_family": family,
            "old_model": old_name,
            "expanded_model": new_name,
            "old_RMSE": o["RMSE_Mg_ha"],
            "expanded_RMSE": n["RMSE_Mg_ha"],
            "RMSE_change": n["RMSE_Mg_ha"] - o["RMSE_Mg_ha"],
            "RMSE_percent_change": (
                (n["RMSE_Mg_ha"] - o["RMSE_Mg_ha"]) / o["RMSE_Mg_ha"] * 100
            ),
            "old_MAE": o["MAE_Mg_ha"],
            "expanded_MAE": n["MAE_Mg_ha"],
            "old_R2": o["R2"],
            "expanded_R2": n["R2"],
            "R2_change": n["R2"] - o["R2"],
            "old_Pearson_r": o["Pearson_r"],
            "expanded_Pearson_r": n["Pearson_r"],
            "old_SD_ratio": o["pred_to_obs_SD_ratio"],
            "expanded_SD_ratio": n["pred_to_obs_SD_ratio"],
        })
    comp = pd.DataFrame(comp_rows)

    # Save outputs
    oof_path = TABLE_DIR/"expanded9P_OOF_predictions.csv"
    fold_path = TABLE_DIR/"expanded9P_fold_metrics.csv"
    pooled_path = TABLE_DIR/"expanded9P_pooled_metrics.csv"
    class_path = TABLE_DIR/"expanded9P_performance_by_AGBD_regime.csv"
    old_path = TABLE_DIR/"original3P_reference_metrics_recalculated.csv"
    comp_path = TABLE_DIR/"original3P_vs_expanded9P_comparison.csv"
    params_path = TABLE_DIR/"expanded9P_selected_hyperparameters.csv"
    et_trace_path = TABLE_DIR/"ET9P_inner_search_trace.csv"
    mlp_trace_path = TABLE_DIR/"MLP9P_inner_search_trace.csv"

    oof.to_csv(oof_path, index=False)
    pd.DataFrame(fold_rows).to_csv(fold_path, index=False)
    pooled.to_csv(pooled_path, index=False)
    cls.to_csv(class_path, index=False)
    old_pooled.to_csv(old_path, index=False)
    comp.to_csv(comp_path, index=False)
    pd.DataFrame(selected_rows).to_csv(params_path, index=False)
    pd.concat(et_tune_rows, ignore_index=True).to_csv(et_trace_path, index=False)
    pd.concat(mlp_tune_rows, ignore_index=True).to_csv(mlp_trace_path, index=False)

    plot_comparison(comp)
    plot_obs_pred(oof, "EXPANDED_ET_9P")
    plot_obs_pred(oof, "EXPANDED_MLP_9P")

    report = [
        "PHASE 8C.4 — EXPANDED 9P CONTROLLED SPATIAL-CV MODEL RERUN",
        "=" * 88,
        f"Run time: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"Expanded table: {EXPANDED_TABLE}",
        f"Expanded table SHA256: {actual_hash}",
        "",
        "LOCKED 9P FEATURE SET",
        ", ".join(FEATURES),
        "",
        "FEATURE-SET RATIONALE",
        "- Old HH_dB, HV_dB, HV_SD retained as accepted benchmark predictors.",
        "- HH_SD_dB_3x3 retained as complementary fine-scale HH heterogeneity.",
        "- 9x9 mean/SD predictors retained as broader spatial context.",
        "- PolContrast_SD_dB_9x9 retained as broad polarization-contrast heterogeneity.",
        "- 5x5 variants omitted because Phase 8C.3 showed they were among the highest",
        "  VIF-like variables in the full candidate set and represent intermediate",
        "  versions of the same physical constructs.",
        "- HH_minus_HV excluded because it is an exact algebraic combination of HH and HV.",
        "",
        "VALIDATION",
        "- Same frozen five outer spatial folds.",
        "- Same 3 km block design.",
        "- Same 500 m training exclusion buffer.",
        "- Outer test folds untouched.",
        "- ET and MLP hyperparameters tuned only within buffered training data using",
        "  4-fold GroupKFold by cv_block_id.",
        "- Linear model has no response-based feature selection.",
        "",
        "EXPANDED 9P POOLED PERFORMANCE",
        pooled.to_string(index=False),
        "",
        "ORIGINAL 3P REFERENCE PERFORMANCE",
        old_pooled.to_string(index=False),
        "",
        "ORIGINAL 3P VS EXPANDED 9P",
        comp.to_string(index=False),
        "",
        "EXPANDED 9P PERFORMANCE BY AGBD REGIME",
        cls.to_string(index=False),
        "",
        "FINAL DECISION RULE",
        "A useful predictor-expansion result should improve more than one metric and should",
        "also reduce prediction compression / biomass-regime bias. A tiny numerical RMSE",
        "gain alone is not sufficient to justify wall-to-wall biomass mapping.",
        "",
        "NEXT STEP",
        "Upload this report and comparison outputs for the final defensibility review.",
        "Do not create a final AGBD map yet.",
    ]

    report_path = QC_DIR/"PHASE8C4_EXPANDED9P_MODEL_REPORT.txt"
    report_path.write_text("\n".join(report), encoding="utf-8")

    print("\n" + "=" * 88)
    print("PHASE 8C.4 COMPLETE — FINAL REVIEW REQUIRED")
    print("=" * 88)

    print("\nExpanded 9P pooled performance:")
    print(
        pooled[[
            "model","RMSE_Mg_ha","MAE_Mg_ha","R2",
            "Bias_Mg_ha","Pearson_r","pred_to_obs_SD_ratio"
        ]].to_string(index=False)
    )

    print("\nOriginal 3P vs expanded 9P:")
    print(
        comp[[
            "model_family","old_RMSE","expanded_RMSE","RMSE_percent_change",
            "old_R2","expanded_R2","R2_change",
            "old_SD_ratio","expanded_SD_ratio"
        ]].to_string(index=False)
    )

    print("\nCreated:")
    print(report_path)
    print(comp_path)
    print(oof_path)
    print(params_path)
    print(f"Figures: {FIG_DIR}")

    print("\nIMPORTANT:")
    print("Do not create a final biomass map yet.")
    print("Send the Phase 8C.4 report and comparison table to ChatGPT.")
    None  # interactive pause removed for portable execution

if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n" + "=" * 88)
        print("PHASE 8C.4 FAILED")
        print("=" * 88)
        traceback.print_exc()
        None  # interactive pause removed for portable execution
        sys.exit(1)
