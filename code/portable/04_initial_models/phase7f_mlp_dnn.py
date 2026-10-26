from pathlib import Path
from datetime import datetime
import hashlib
import os
import sys
import warnings

# Windows-safe / low-memory settings before NumPy / sklearn imports
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.exceptions import ConvergenceWarning
import matplotlib.pyplot as plt
import joblib

# ============================================================
# PROJECT SETTINGS
# ============================================================
PROJECT_ROOT = Path(os.environ.get("NISAR_PROJECT_ROOT", r"D:\NISAR_Mangrove_Paper"))

BUFFER_ROOT = (
    PROJECT_ROOT
    / "05_ML_DL"
    / "Spatial_CV"
    / "Buffered_500m"
)

MODEL_DIR = PROJECT_ROOT / "05_ML_DL" / "DNN"
OUT_DIR = PROJECT_ROOT / "06_Results" / "DNN_MLP"
TABLE_DIR = OUT_DIR / "Tables"
FIG_DIR = OUT_DIR / "Figures"
LOG_PATH = PROJECT_ROOT / "00_Documentation" / "analysis_log.txt"

RESPONSE = "agbd"

FEATURE_SETS = {
    "MLP_4P_MAIN": ["HH_dB", "HV_dB", "HH_minus_HV", "HV_SD"],
    "MLP_3P_SENSITIVITY": ["HH_dB", "HV_dB", "HV_SD"],
}

N_OUTER_FOLDS = 5
N_INNER_FOLDS = 4
RANDOM_SEED = 42
N_SEARCH_ITER = 12
N_JOBS = 1

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# MLP SEARCH SPACE
# ============================================================
PARAM_DISTRIBUTIONS = {
    "mlp__hidden_layer_sizes": [
        (16,),
        (32,),
        (64,),
        (32, 16),
        (64, 32),
        (32, 16, 8),
    ],
    "mlp__activation": ["relu", "tanh"],
    "mlp__alpha": [1e-4, 1e-3, 1e-2, 1e-1],
    "mlp__learning_rate_init": [1e-4, 5e-4, 1e-3, 5e-3],
}

# ============================================================
# HELPERS
# ============================================================
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def append_log(lines):
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write("\n[" + stamp + "] PHASE 7F — MLP / DNN\n")
        for line in lines:
            f.write(line + "\n")
        f.write("-" * 72 + "\n")

def calc_metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    bias = float(np.mean(y_pred - y_true))
    mean_obs = float(np.mean(y_true))
    nrmse = float(100.0 * rmse / mean_obs) if mean_obs != 0 else np.nan

    if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        r = float(pearsonr(y_true, y_pred).statistic)
    else:
        r = np.nan

    return {
        "RMSE_Mg_ha": rmse,
        "MAE_Mg_ha": mae,
        "R2": r2,
        "Bias_Mg_ha": bias,
        "nRMSE_percent_of_mean": nrmse,
        "Pearson_r": r,
    }

def check_columns(df, features, path):
    required = [RESPONSE, "cv_block_id"] + features
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns in {path.name}: {missing}")

def finite_check(df, cols, label):
    arr = df[cols].to_numpy(dtype=float)
    if not np.isfinite(arr).all():
        raise RuntimeError(f"{label}: non-finite values found.")

def fit_one_outer_fold(model_name, features, fold, train, test):
    X_train = train[features].to_numpy(dtype=float)
    y_train = train[RESPONSE].to_numpy(dtype=float)
    X_test = test[features].to_numpy(dtype=float)
    y_test = test[RESPONSE].to_numpy(dtype=float)

    groups = train["cv_block_id"].astype(str).to_numpy()
    unique_groups = np.unique(groups)

    if len(unique_groups) < N_INNER_FOLDS:
        raise RuntimeError(
            f"{model_name} fold {fold}: only {len(unique_groups)} inner groups."
        )

    inner_cv = GroupKFold(n_splits=N_INNER_FOLDS)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            solver="adam",
            max_iter=1500,
            early_stopping=False,
            learning_rate="adaptive",
            random_state=RANDOM_SEED,
            tol=1e-4,
        )),
    ])

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=N_SEARCH_ITER,
        scoring="neg_root_mean_squared_error",
        cv=inner_cv,
        random_state=RANDOM_SEED,
        n_jobs=N_JOBS,
        refit=True,
        return_train_score=False,
        verbose=0,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        search.fit(X_train, y_train, groups=groups)

    best = search.best_estimator_
    pred = best.predict(X_test)

    m = calc_metrics(y_test, pred)
    m.update({
        "model": model_name,
        "fold": fold,
        "train_n": len(train),
        "test_n": len(test),
        "inner_groups_n": len(unique_groups),
        "inner_best_RMSE_Mg_ha": float(-search.best_score_),
    })

    # Test-fold permutation importance is descriptive only.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pi = permutation_importance(
            best,
            X_test,
            y_test,
            scoring="neg_root_mean_squared_error",
            n_repeats=20,
            random_state=RANDOM_SEED,
            n_jobs=1,
        )

    perm = pd.DataFrame({
        "model": model_name,
        "fold": fold,
        "predictor": features,
        "mean_permutation_RMSE_increase": pi.importances_mean,
        "sd_permutation_RMSE_increase": pi.importances_std,
    })

    params = {
        "model": model_name,
        "fold": fold,
        "best_inner_RMSE_Mg_ha": float(-search.best_score_),
        **{k.replace("mlp__", ""): v for k, v in search.best_params_.items()},
    }

    # Add fitted iteration count / training-loss metadata.
    fitted_mlp = best.named_steps["mlp"]
    params["n_iter_fit"] = int(getattr(fitted_mlp, "n_iter_", -1))
    params["final_training_loss"] = float(getattr(fitted_mlp, "loss_", np.nan))

    pred_df = test.copy()
    pred_df["outer_fold"] = fold
    pred_df["model"] = model_name
    pred_df["observed_agbd"] = y_test
    pred_df["predicted_agbd"] = pred
    pred_df["residual_observed_minus_predicted"] = y_test - pred

    return m, perm, params, pred_df, best

def pooled_metrics_from_oof(oof):
    rows = []
    for model_name, g in oof.groupby("model"):
        m = calc_metrics(
            g["observed_agbd"].to_numpy(dtype=float),
            g["predicted_agbd"].to_numpy(dtype=float),
        )
        m.update({
            "model": model_name,
            "evaluation": "pooled_out_of_fold",
            "n": len(g),
        })
        rows.append(m)
    return pd.DataFrame(rows)

def fold_metric_summary(fold_metrics):
    rows = []
    metric_cols = [
        "RMSE_Mg_ha", "MAE_Mg_ha", "R2", "Bias_Mg_ha",
        "nRMSE_percent_of_mean", "Pearson_r", "inner_best_RMSE_Mg_ha",
    ]
    for model_name, g in fold_metrics.groupby("model"):
        mean_row = {
            "model": model_name,
            "evaluation": "mean_across_folds",
            "n": int(g["test_n"].sum()),
        }
        sd_row = {
            "model": model_name,
            "evaluation": "sd_across_folds",
            "n": int(g["test_n"].sum()),
        }
        for c in metric_cols:
            mean_row[c] = float(g[c].mean())
            sd_row[c] = float(g[c].std(ddof=1))
        rows.extend([mean_row, sd_row])
    return pd.DataFrame(rows)

def comparison_table(mlp_pooled):
    rows = []

    baseline_file = PROJECT_ROOT / "06_Results" / "Baseline" / "Tables" / "baseline_overall_metrics.csv"
    rf_file = PROJECT_ROOT / "06_Results" / "Random_Forest" / "Tables" / "RF_overall_metrics.csv"
    xgb_file = PROJECT_ROOT / "06_Results" / "XGBoost" / "Tables" / "XGB_overall_metrics.csv"
    svr_file = PROJECT_ROOT / "06_Results" / "SVR" / "Tables" / "SVR_overall_metrics.csv"
    et_file = PROJECT_ROOT / "06_Results" / "Extra_Trees" / "Tables" / "ET_overall_metrics.csv"

    if baseline_file.exists():
        b = pd.read_csv(baseline_file)
        b = b[b["evaluation"] == "pooled_out_of_fold"]
        if len(b) == 1:
            r = b.iloc[0]
            rows.append({
                "model": "Linear_baseline_3P",
                "RMSE_Mg_ha": r["RMSE_Mg_ha"],
                "MAE_Mg_ha": r["MAE_Mg_ha"],
                "R2": r["R2"],
                "Bias_Mg_ha": r["Bias_Mg_ha"],
                "Pearson_r": r["Pearson_r"],
            })

    for f in [rf_file, xgb_file, svr_file, et_file]:
        if f.exists():
            t = pd.read_csv(f)
            t = t[t["evaluation"] == "pooled_out_of_fold"]
            for _, r in t.iterrows():
                rows.append({
                    "model": r["model"],
                    "RMSE_Mg_ha": r["RMSE_Mg_ha"],
                    "MAE_Mg_ha": r["MAE_Mg_ha"],
                    "R2": r["R2"],
                    "Bias_Mg_ha": r["Bias_Mg_ha"],
                    "Pearson_r": r["Pearson_r"],
                })

    for _, r in mlp_pooled.iterrows():
        rows.append({
            "model": r["model"],
            "RMSE_Mg_ha": r["RMSE_Mg_ha"],
            "MAE_Mg_ha": r["MAE_Mg_ha"],
            "R2": r["R2"],
            "Bias_Mg_ha": r["Bias_Mg_ha"],
            "Pearson_r": r["Pearson_r"],
        })

    out = pd.DataFrame(rows)
    if len(out):
        out.to_csv(TABLE_DIR / "final_model_comparison_through_MLP.csv", index=False)
    return out

def save_figures(oof, fold_metrics, perm_summary, comparison):
    main = oof[oof["model"] == "MLP_4P_MAIN"].copy()

    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    ax.scatter(main["observed_agbd"], main["predicted_agbd"], s=16, alpha=0.45)
    lo = float(min(main["observed_agbd"].min(), main["predicted_agbd"].min()))
    hi = float(max(main["observed_agbd"].max(), main["predicted_agbd"].max()))
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2)
    ax.set_xlabel("Observed GEDI AGBD (Mg/ha)")
    ax.set_ylabel("Predicted AGBD (Mg/ha)")
    ax.set_title("MLP neural network (4 predictors) — pooled spatial-CV predictions", pad=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "MLP4_observed_vs_predicted.png", dpi=450, bbox_inches="tight")
    fig.savefig(FIG_DIR / "MLP4_observed_vs_predicted.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 6.0))
    ax.scatter(
        main["predicted_agbd"],
        main["residual_observed_minus_predicted"],
        s=16,
        alpha=0.45,
    )
    ax.axhline(0.0, linestyle="--", linewidth=1.2)
    ax.set_xlabel("Predicted AGBD (Mg/ha)")
    ax.set_ylabel("Residual: observed − predicted (Mg/ha)")
    ax.set_title("MLP neural network (4 predictors) — residual diagnostic", pad=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "MLP4_residuals_vs_predicted.png", dpi=450, bbox_inches="tight")
    fig.savefig(FIG_DIR / "MLP4_residuals_vs_predicted.pdf", bbox_inches="tight")
    plt.close(fig)

    pivot = fold_metrics.pivot(index="fold", columns="model", values="RMSE_Mg_ha")
    fig, ax = plt.subplots(figsize=(7.4, 5.3))
    x = np.arange(len(pivot.index))
    width = 0.36
    cols = list(pivot.columns)
    for i, c in enumerate(cols):
        offset = (i - (len(cols) - 1) / 2) * width
        ax.bar(x + offset, pivot[c].values, width=width, label=c)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in pivot.index])
    ax.set_xlabel("Outer spatial fold")
    ax.set_ylabel("RMSE (Mg/ha)")
    ax.set_title("MLP sensitivity — RMSE by outer fold", pad=12)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "MLP_4P_vs_3P_RMSE_by_fold.png", dpi=450, bbox_inches="tight")
    fig.savefig(FIG_DIR / "MLP_4P_vs_3P_RMSE_by_fold.pdf", bbox_inches="tight")
    plt.close(fig)

    pfi = perm_summary[
        perm_summary["model"] == "MLP_4P_MAIN"
    ].sort_values("mean_permutation_RMSE_increase", ascending=True)

    if len(pfi):
        fig, ax = plt.subplots(figsize=(7.0, 5.2))
        ax.barh(
            pfi["predictor"],
            pfi["mean_permutation_RMSE_increase"],
            xerr=pfi["sd_across_folds"].fillna(0.0),
        )
        ax.set_xlabel("Mean permutation importance (RMSE increase)")
        ax.set_ylabel("Predictor")
        ax.set_title("MLP neural network (4 predictors) — spatial-CV permutation importance", pad=12)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "MLP4_permutation_importance.png", dpi=450, bbox_inches="tight")
        fig.savefig(FIG_DIR / "MLP4_permutation_importance.pdf", bbox_inches="tight")
        plt.close(fig)

    if len(comparison) >= 2:
        fig, ax = plt.subplots(figsize=(11.2, 5.8))
        ax.bar(comparison["model"], comparison["R2"])
        ax.axhline(0.0, linestyle="--", linewidth=1.0)
        ax.set_ylabel("Pooled out-of-fold R²")
        ax.set_title("Final model comparison under identical spatial validation", pad=12)
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "final_model_comparison_R2.png", dpi=450, bbox_inches="tight")
        fig.savefig(FIG_DIR / "final_model_comparison_R2.pdf", bbox_inches="tight")
        plt.close(fig)

def main():
    print("=" * 72)
    print("PHASE 7F — MLP / DNN")
    print("=" * 72)
    print("Implementation: scikit-learn MLPRegressor neural network")
    print("Outer validation: frozen 5-fold spatial CV + 500 m training buffer")
    print(f"Inner tuning: {N_INNER_FOLDS}-fold GroupKFold by 3 km cv_block_id")
    print(f"Randomized search iterations: {N_SEARCH_ITER}")
    print(f"Random seed: {RANDOM_SEED}")
    print("StandardScaler is fitted inside each training fold only.")
    print("Windows-safe execution: n_jobs=1")

    metrics_all = []
    perm_all = []
    params_all = []
    oof_all = []
    provenance_all = []

    for model_name, features in FEATURE_SETS.items():
        print("\n" + "-" * 72)
        print(model_name)
        print(f"Predictors: {', '.join(features)}")
        print("-" * 72)

        for fold in range(1, N_OUTER_FOLDS + 1):
            fold_dir = BUFFER_ROOT / f"Fold_{fold}"
            train_path = fold_dir / f"Fold_{fold}_TRAIN_BUFFERED_500m.csv"
            test_path = fold_dir / f"Fold_{fold}_TEST.csv"

            if not train_path.exists() or not test_path.exists():
                raise FileNotFoundError(
                    f"Missing Phase-6B buffered input for outer fold {fold}."
                )

            train = pd.read_csv(train_path)
            test = pd.read_csv(test_path)

            check_columns(train, features, train_path)
            check_columns(test, features, test_path)
            finite_check(train, [RESPONSE] + features, f"{model_name} fold {fold} training")
            finite_check(test, [RESPONSE] + features, f"{model_name} fold {fold} test")

            if "shot_number" in train.columns and "shot_number" in test.columns:
                overlap = set(train["shot_number"]).intersection(set(test["shot_number"]))
                if overlap:
                    raise RuntimeError(
                        f"{model_name} fold {fold}: train/test shot_number overlap."
                    )

            m, perm, params, pred_df, best = fit_one_outer_fold(
                model_name, features, fold, train, test
            )

            metrics_all.append(m)
            perm_all.append(perm)
            params_all.append(params)
            oof_all.append(pred_df)

            provenance_all.append({
                "model": model_name,
                "fold": fold,
                "train_file": str(train_path),
                "train_sha256": sha256_file(train_path),
                "test_file": str(test_path),
                "test_sha256": sha256_file(test_path),
            })

            joblib.dump(best, MODEL_DIR / f"{model_name}_outer_fold_{fold}.joblib")

            print(
                f"Fold {fold}: train={len(train):,}, test={len(test):,}, "
                f"inner RMSE={m['inner_best_RMSE_Mg_ha']:.2f}, "
                f"outer RMSE={m['RMSE_Mg_ha']:.2f}, "
                f"MAE={m['MAE_Mg_ha']:.2f}, "
                f"R2={m['R2']:.3f}, r={m['Pearson_r']:.3f}"
            )

    fold_metrics = pd.DataFrame(metrics_all).sort_values(["model", "fold"])
    permutation = pd.concat(perm_all, ignore_index=True)
    best_params = pd.DataFrame(params_all).sort_values(["model", "fold"])
    oof = pd.concat(oof_all, ignore_index=True)
    provenance = pd.DataFrame(provenance_all).sort_values(["model", "fold"])

    pooled = pooled_metrics_from_oof(oof)
    fold_summary = fold_metric_summary(fold_metrics)
    overall = pd.concat([pooled, fold_summary], ignore_index=True, sort=False)

    perm_summary = (
        permutation.groupby(["model", "predictor"], as_index=False)
        .agg(
            mean_permutation_RMSE_increase=("mean_permutation_RMSE_increase", "mean"),
            sd_across_folds=("mean_permutation_RMSE_increase", "std"),
            min_across_folds=("mean_permutation_RMSE_increase", "min"),
            max_across_folds=("mean_permutation_RMSE_increase", "max"),
        )
    )

    fold_metrics.to_csv(TABLE_DIR / "MLP_fold_metrics.csv", index=False)
    overall.to_csv(TABLE_DIR / "MLP_overall_metrics.csv", index=False)
    best_params.to_csv(TABLE_DIR / "MLP_best_parameters_by_fold.csv", index=False)
    oof.to_csv(TABLE_DIR / "MLP_OOF_predictions.csv", index=False)
    permutation.to_csv(TABLE_DIR / "MLP_permutation_importance_by_fold.csv", index=False)
    perm_summary.to_csv(TABLE_DIR / "MLP_permutation_importance_summary.csv", index=False)
    provenance.to_csv(TABLE_DIR / "MLP_input_provenance.csv", index=False)

    comparison = comparison_table(pooled)
    save_figures(oof, fold_metrics, perm_summary, comparison)

    main_row = pooled[pooled["model"] == "MLP_4P_MAIN"].iloc[0]
    sens_row = pooled[pooled["model"] == "MLP_3P_SENSITIVITY"].iloc[0]

    report = [
        "PHASE 7F — MLP / DNN",
        "=" * 72,
        "Implementation: scikit-learn MLPRegressor.",
        "Outer validation: frozen 5-fold spatial block CV with 500 m training exclusion buffer.",
        f"Inner tuning: {N_INNER_FOLDS}-fold GroupKFold grouped by cv_block_id.",
        f"Randomized search iterations per outer fold: {N_SEARCH_ITER}.",
        f"Random seed: {RANDOM_SEED}.",
        "Scaling: StandardScaler fitted inside each training fold via Pipeline.",
        "Early stopping: disabled, avoiding an extra random internal validation split.",
        "Execution: n_jobs=1 / low-memory Windows-safe.",
        "",
        "Primary MLP model:",
        "MLP_4P_MAIN = HH_dB + HV_dB + HH_minus_HV + HV_SD",
        "",
        "Sensitivity MLP model:",
        "MLP_3P_SENSITIVITY = HH_dB + HV_dB + HV_SD",
        "",
        "Pooled out-of-fold performance:",
        pooled.to_string(index=False),
        "",
        "Per-fold performance:",
        fold_metrics.to_string(index=False),
        "",
        "Best architecture / hyperparameters by outer fold:",
        best_params.to_string(index=False),
        "",
        "Permutation importance summary:",
        perm_summary.to_string(index=False),
        "",
        "Rules:",
        "- Outer folds and 500 m buffer are unchanged.",
        "- Scaling and hyperparameter tuning are training-only.",
        "- Test-fold permutation importance is descriptive only.",
        "- This completes the planned algorithm-comparison phase.",
    ]

    (OUT_DIR / "MLP_QC_REPORT.txt").write_text("\n".join(report), encoding="utf-8")

    append_log([
        "Script: phase7f_mlp_dnn.py",
        "Implementation: scikit-learn MLPRegressor",
        "Primary model: MLP_4P_MAIN",
        "Sensitivity model: MLP_3P_SENSITIVITY",
        f"Inner tuning: {N_INNER_FOLDS}-fold GroupKFold by cv_block_id",
        f"Search iterations: {N_SEARCH_ITER}",
        "StandardScaler fitted in training-only Pipeline.",
        "Early stopping disabled.",
        f"MLP_4P pooled OOF RMSE: {main_row['RMSE_Mg_ha']:.4f} Mg/ha",
        f"MLP_4P pooled OOF R2: {main_row['R2']:.6f}",
        f"MLP_3P pooled OOF RMSE: {sens_row['RMSE_Mg_ha']:.4f} Mg/ha",
        f"MLP_3P pooled OOF R2: {sens_row['R2']:.6f}",
    ])

    print("\n" + "=" * 72)
    print("PHASE 7F COMPLETE — MLP/DNN REVIEW REQUIRED")
    print("=" * 72)

    print("\nPrimary MLP_4P_MAIN pooled OOF:")
    print(f"  RMSE: {main_row['RMSE_Mg_ha']:.2f} Mg/ha")
    print(f"  MAE:  {main_row['MAE_Mg_ha']:.2f} Mg/ha")
    print(f"  R2:   {main_row['R2']:.3f}")
    print(f"  r:    {main_row['Pearson_r']:.3f}")

    print("\nMLP_3P_SENSITIVITY pooled OOF:")
    print(f"  RMSE: {sens_row['RMSE_Mg_ha']:.2f} Mg/ha")
    print(f"  MAE:  {sens_row['MAE_Mg_ha']:.2f} Mg/ha")
    print(f"  R2:   {sens_row['R2']:.3f}")
    print(f"  r:    {sens_row['Pearson_r']:.3f}")

    print("\nCreated:")
    print(f" - {TABLE_DIR / 'MLP_fold_metrics.csv'}")
    print(f" - {TABLE_DIR / 'MLP_overall_metrics.csv'}")
    print(f" - {TABLE_DIR / 'MLP_best_parameters_by_fold.csv'}")
    print(f" - {TABLE_DIR / 'MLP_OOF_predictions.csv'}")
    print(f" - {TABLE_DIR / 'MLP_permutation_importance_summary.csv'}")
    print(f" - {TABLE_DIR / 'final_model_comparison_through_MLP.csv'}")
    print(f" - {OUT_DIR / 'MLP_QC_REPORT.txt'}")
    print(f" - {FIG_DIR / 'MLP4_observed_vs_predicted.png'}")
    print(f" - {FIG_DIR / 'MLP4_residuals_vs_predicted.png'}")
    print(f" - {FIG_DIR / 'MLP_4P_vs_3P_RMSE_by_fold.png'}")
    print(f" - {FIG_DIR / 'MLP4_permutation_importance.png'}")
    print(f" - {FIG_DIR / 'final_model_comparison_R2.png'}")

    print("\nIMPORTANT:")
    print("After this run, upload the Phase-7F outputs for final model-comparison review.")
    None  # interactive pause removed for portable execution

if __name__ == "__main__":
    main()
