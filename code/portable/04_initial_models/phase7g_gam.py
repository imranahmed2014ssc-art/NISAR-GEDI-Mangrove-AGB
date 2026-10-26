from pathlib import Path
from datetime import datetime
import hashlib
import os
import sys
import warnings

# Windows-safe / low-memory settings
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
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

MODEL_DIR = PROJECT_ROOT / "05_ML_DL" / "GAM"
OUT_DIR = PROJECT_ROOT / "06_Results" / "GAM"
TABLE_DIR = OUT_DIR / "Tables"
FIG_DIR = OUT_DIR / "Figures"
CURVE_DIR = OUT_DIR / "Smooth_Response_Curves"
LOG_PATH = PROJECT_ROOT / "00_Documentation" / "analysis_log.txt"

RESPONSE = "agbd"

FEATURE_SETS = {
    "GAM_4P_MAIN": ["HH_dB", "HV_dB", "HH_minus_HV", "HV_SD"],
    "GAM_3P_SENSITIVITY": ["HH_dB", "HV_dB", "HV_SD"],
}

N_OUTER_FOLDS = 5
N_INNER_FOLDS = 4
RANDOM_SEED = 42
N_SEARCH_ITER = 16
N_JOBS = 1

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
CURVE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# SEARCH SPACE
# ============================================================
# This is a GAM-style additive penalized spline model implemented with:
# one spline basis per predictor + Ridge penalty.
#
# No predictor interactions are included.
PARAM_DISTRIBUTIONS = {
    "spline__n_knots": [4, 5, 6, 7, 8, 10],
    "spline__degree": [2, 3],
    "ridge__alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
}

# ============================================================
# CUSTOM ADDITIVE SPLINE TRANSFORMER
# ============================================================
class AdditiveSplineTransformer(BaseEstimator, TransformerMixin):
    """
    Applies a separate univariate SplineTransformer to every input column.
    The transformed bases are concatenated with NO cross-feature interactions.
    """
    def __init__(self, n_knots=6, degree=3):
        self.n_knots = n_knots
        self.degree = degree

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.transformers_ = []
        for j in range(X.shape[1]):
            st = SplineTransformer(
                n_knots=self.n_knots,
                degree=self.degree,
                include_bias=False,
                knots="quantile",
                extrapolation="linear",
            )
            st.fit(X[:, [j]])
            self.transformers_.append(st)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        parts = [
            st.transform(X[:, [j]])
            for j, st in enumerate(self.transformers_)
        ]
        return np.hstack(parts)

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
        f.write("\n[" + stamp + "] PHASE 7G — GAM-STYLE PENALIZED SPLINE REGRESSION\n")
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

def make_pipeline():
    return Pipeline([
        ("spline", AdditiveSplineTransformer()),
        ("scale", StandardScaler()),
        ("ridge", Ridge()),
    ])

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

    search = RandomizedSearchCV(
        estimator=make_pipeline(),
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

    # Outer-test permutation importance is descriptive only.
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
        **{
            k.replace("spline__", "").replace("ridge__", ""): v
            for k, v in search.best_params_.items()
        },
    }

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

def comparison_table(gam_pooled):
    rows = []

    sources = [
        ("baseline", PROJECT_ROOT / "06_Results" / "Baseline" / "Tables" / "baseline_overall_metrics.csv"),
        ("rf", PROJECT_ROOT / "06_Results" / "Random_Forest" / "Tables" / "RF_overall_metrics.csv"),
        ("xgb", PROJECT_ROOT / "06_Results" / "XGBoost" / "Tables" / "XGB_overall_metrics.csv"),
        ("svr", PROJECT_ROOT / "06_Results" / "SVR" / "Tables" / "SVR_overall_metrics.csv"),
        ("et", PROJECT_ROOT / "06_Results" / "Extra_Trees" / "Tables" / "ET_overall_metrics.csv"),
        ("mlp", PROJECT_ROOT / "06_Results" / "DNN_MLP" / "Tables" / "MLP_overall_metrics.csv"),
    ]

    # Baseline
    bfile = sources[0][1]
    if bfile.exists():
        b = pd.read_csv(bfile)
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

    # Prior model families
    for _, f in sources[1:]:
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

    for _, r in gam_pooled.iterrows():
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
        out.to_csv(TABLE_DIR / "final_model_comparison_through_GAM.csv", index=False)
    return out

def make_main_smooth_curves(fitted_models, train_refs, features):
    """
    For the 4P MAIN GAM-style model only:
    vary one predictor at a time from its pooled 5th to 95th percentile,
    hold all other predictors at pooled training medians,
    predict with each outer-fold fitted model, then summarize mean ± SD.
    """
    pooled_train = pd.concat(train_refs, ignore_index=True)
    medians = pooled_train[features].median()
    curves = []

    for feat in features:
        q05 = float(pooled_train[feat].quantile(0.05))
        q95 = float(pooled_train[feat].quantile(0.95))
        grid = np.linspace(q05, q95, 120)

        fold_preds = []
        for fold, model in fitted_models.items():
            X = pd.DataFrame(
                np.tile(medians.to_numpy(dtype=float), (len(grid), 1)),
                columns=features,
            )
            X[feat] = grid
            pred = model.predict(X[features].to_numpy(dtype=float))
            fold_preds.append(pred)

        fold_preds = np.vstack(fold_preds)
        mean_pred = fold_preds.mean(axis=0)
        sd_pred = fold_preds.std(axis=0, ddof=1)

        cdf = pd.DataFrame({
            "predictor": feat,
            "predictor_value": grid,
            "mean_predicted_agbd": mean_pred,
            "sd_across_outer_folds": sd_pred,
            "lower_1sd": mean_pred - sd_pred,
            "upper_1sd": mean_pred + sd_pred,
        })
        curves.append(cdf)

        fig, ax = plt.subplots(figsize=(7.2, 5.4))
        ax.plot(grid, mean_pred, linewidth=1.8, label="Mean GAM response")
        ax.fill_between(
            grid,
            mean_pred - sd_pred,
            mean_pred + sd_pred,
            alpha=0.20,
            label="±1 SD across outer-fold fits",
        )
        ax.set_xlabel(feat)
        ax.set_ylabel("Predicted AGBD (Mg/ha)")
        ax.set_title(f"GAM smooth response — {feat}", pad=12)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
        fig.tight_layout()
        fig.savefig(CURVE_DIR / f"GAM_smooth_{feat}.png", dpi=450, bbox_inches="tight")
        fig.savefig(CURVE_DIR / f"GAM_smooth_{feat}.pdf", bbox_inches="tight")
        plt.close(fig)

    curves = pd.concat(curves, ignore_index=True)
    curves.to_csv(TABLE_DIR / "GAM_main_smooth_response_curves.csv", index=False)
    return curves

def save_figures(oof, fold_metrics, perm_summary, comparison):
    main = oof[oof["model"] == "GAM_4P_MAIN"].copy()

    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    ax.scatter(main["observed_agbd"], main["predicted_agbd"], s=16, alpha=0.45)
    lo = float(min(main["observed_agbd"].min(), main["predicted_agbd"].min()))
    hi = float(max(main["observed_agbd"].max(), main["predicted_agbd"].max()))
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2)
    ax.set_xlabel("Observed GEDI AGBD (Mg/ha)")
    ax.set_ylabel("Predicted AGBD (Mg/ha)")
    ax.set_title("GAM-style penalized spline (4 predictors) — pooled spatial-CV predictions", pad=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "GAM4_observed_vs_predicted.png", dpi=450, bbox_inches="tight")
    fig.savefig(FIG_DIR / "GAM4_observed_vs_predicted.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 6.0))
    ax.scatter(main["predicted_agbd"], main["residual_observed_minus_predicted"], s=16, alpha=0.45)
    ax.axhline(0.0, linestyle="--", linewidth=1.2)
    ax.set_xlabel("Predicted AGBD (Mg/ha)")
    ax.set_ylabel("Residual: observed − predicted (Mg/ha)")
    ax.set_title("GAM-style penalized spline (4 predictors) — residual diagnostic", pad=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "GAM4_residuals_vs_predicted.png", dpi=450, bbox_inches="tight")
    fig.savefig(FIG_DIR / "GAM4_residuals_vs_predicted.pdf", bbox_inches="tight")
    plt.close(fig)

    pivot = fold_metrics.pivot(index="fold", columns="model", values="RMSE_Mg_ha")
    fig, ax = plt.subplots(figsize=(7.4, 5.3))
    x = np.arange(len(pivot.index))
    width = 0.36
    cols = list(pivot.columns)
    for i, c in enumerate(cols):
        offset = (i - (len(cols)-1)/2) * width
        ax.bar(x + offset, pivot[c].values, width=width, label=c)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in pivot.index])
    ax.set_xlabel("Outer spatial fold")
    ax.set_ylabel("RMSE (Mg/ha)")
    ax.set_title("GAM sensitivity — RMSE by outer fold", pad=12)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "GAM_4P_vs_3P_RMSE_by_fold.png", dpi=450, bbox_inches="tight")
    fig.savefig(FIG_DIR / "GAM_4P_vs_3P_RMSE_by_fold.pdf", bbox_inches="tight")
    plt.close(fig)

    pfi = perm_summary[
        perm_summary["model"] == "GAM_4P_MAIN"
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
        ax.set_title("GAM-style model (4 predictors) — spatial-CV permutation importance", pad=12)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "GAM4_permutation_importance.png", dpi=450, bbox_inches="tight")
        fig.savefig(FIG_DIR / "GAM4_permutation_importance.pdf", bbox_inches="tight")
        plt.close(fig)

    if len(comparison) >= 2:
        fig, ax = plt.subplots(figsize=(12.0, 5.8))
        ax.bar(comparison["model"], comparison["R2"])
        ax.axhline(0.0, linestyle="--", linewidth=1.0)
        ax.set_ylabel("Pooled out-of-fold R²")
        ax.set_title("Final model comparison through GAM under identical spatial validation", pad=12)
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "final_model_comparison_through_GAM_R2.png", dpi=450, bbox_inches="tight")
        fig.savefig(FIG_DIR / "final_model_comparison_through_GAM_R2.pdf", bbox_inches="tight")
        plt.close(fig)

def main():
    print("=" * 72)
    print("PHASE 7G — GAM-STYLE PENALIZED ADDITIVE SPLINE REGRESSION")
    print("=" * 72)
    print("Implementation: separate univariate spline basis per predictor + Ridge penalty")
    print("No predictor interactions are included.")
    print("Outer validation: frozen 5-fold spatial CV + 500 m training buffer")
    print(f"Inner tuning: {N_INNER_FOLDS}-fold GroupKFold by 3 km cv_block_id")
    print(f"Randomized search iterations: {N_SEARCH_ITER}")
    print(f"Random seed: {RANDOM_SEED}")
    print("Windows-safe execution: n_jobs=1")

    metrics_all = []
    perm_all = []
    params_all = []
    oof_all = []
    provenance_all = []

    # Store main 4P outer-fold fits to generate smooth response curves later.
    main_models = {}
    main_train_refs = []

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

            if model_name == "GAM_4P_MAIN":
                main_models[fold] = best
                main_train_refs.append(train[features].copy())

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

    fold_metrics.to_csv(TABLE_DIR / "GAM_fold_metrics.csv", index=False)
    overall.to_csv(TABLE_DIR / "GAM_overall_metrics.csv", index=False)
    best_params.to_csv(TABLE_DIR / "GAM_best_parameters_by_fold.csv", index=False)
    oof.to_csv(TABLE_DIR / "GAM_OOF_predictions.csv", index=False)
    permutation.to_csv(TABLE_DIR / "GAM_permutation_importance_by_fold.csv", index=False)
    perm_summary.to_csv(TABLE_DIR / "GAM_permutation_importance_summary.csv", index=False)
    provenance.to_csv(TABLE_DIR / "GAM_input_provenance.csv", index=False)

    comparison = comparison_table(pooled)
    save_figures(oof, fold_metrics, perm_summary, comparison)

    main_features = FEATURE_SETS["GAM_4P_MAIN"]
    make_main_smooth_curves(main_models, main_train_refs, main_features)

    main_row = pooled[pooled["model"] == "GAM_4P_MAIN"].iloc[0]
    sens_row = pooled[pooled["model"] == "GAM_3P_SENSITIVITY"].iloc[0]

    report = [
        "PHASE 7G — GAM-STYLE PENALIZED ADDITIVE SPLINE REGRESSION",
        "=" * 72,
        "Implementation: separate univariate spline basis per predictor + Ridge penalty.",
        "No cross-predictor interactions are included.",
        "This is a GAM-style penalized additive spline model implemented entirely in scikit-learn.",
        "Outer validation: frozen 5-fold spatial block CV with 500 m training exclusion buffer.",
        f"Inner tuning: {N_INNER_FOLDS}-fold GroupKFold grouped by cv_block_id.",
        f"Randomized search iterations per outer fold: {N_SEARCH_ITER}.",
        f"Random seed: {RANDOM_SEED}.",
        "",
        "Primary model:",
        "GAM_4P_MAIN = s(HH_dB) + s(HV_dB) + s(HH_minus_HV) + s(HV_SD)",
        "",
        "Sensitivity model:",
        "GAM_3P_SENSITIVITY = s(HH_dB) + s(HV_dB) + s(HV_SD)",
        "",
        "Pooled out-of-fold performance:",
        pooled.to_string(index=False),
        "",
        "Per-fold performance:",
        fold_metrics.to_string(index=False),
        "",
        "Best spline/penalty hyperparameters by outer fold:",
        best_params.to_string(index=False),
        "",
        "Permutation importance summary:",
        perm_summary.to_string(index=False),
        "",
        "Smooth response curves:",
        "Main-model predictor response curves were generated by varying one predictor at a time",
        "from its pooled training 5th to 95th percentile while holding other predictors at",
        "their pooled training medians; curves summarize mean ±1 SD across the five outer-fold fits.",
        "",
        "Rules:",
        "- Outer folds and 500 m buffer are unchanged.",
        "- Spline complexity and Ridge penalty are tuned using training data only.",
        "- Test-fold permutation importance is descriptive only.",
        "- GAM response curves are interpretive diagnostics, not an additional tuning step.",
        "- This is the final additional model before saturation/temporal-mismatch diagnosis.",
    ]

    (OUT_DIR / "GAM_QC_REPORT.txt").write_text("\n".join(report), encoding="utf-8")

    append_log([
        "Script: phase7g_gam.py",
        "Implementation: additive univariate spline bases + Ridge penalty",
        "Primary model: GAM_4P_MAIN",
        "Sensitivity model: GAM_3P_SENSITIVITY",
        f"Inner tuning: {N_INNER_FOLDS}-fold GroupKFold by cv_block_id",
        f"Search iterations: {N_SEARCH_ITER}",
        f"Random seed: {RANDOM_SEED}",
        f"GAM_4P pooled OOF RMSE: {main_row['RMSE_Mg_ha']:.4f} Mg/ha",
        f"GAM_4P pooled OOF R2: {main_row['R2']:.6f}",
        f"GAM_3P pooled OOF RMSE: {sens_row['RMSE_Mg_ha']:.4f} Mg/ha",
        f"GAM_3P pooled OOF R2: {sens_row['R2']:.6f}",
    ])

    print("\n" + "=" * 72)
    print("PHASE 7G COMPLETE — GAM REVIEW REQUIRED")
    print("=" * 72)

    print("\nPrimary GAM_4P_MAIN pooled OOF:")
    print(f"  RMSE: {main_row['RMSE_Mg_ha']:.2f} Mg/ha")
    print(f"  MAE:  {main_row['MAE_Mg_ha']:.2f} Mg/ha")
    print(f"  R2:   {main_row['R2']:.3f}")
    print(f"  r:    {main_row['Pearson_r']:.3f}")

    print("\nGAM_3P_SENSITIVITY pooled OOF:")
    print(f"  RMSE: {sens_row['RMSE_Mg_ha']:.2f} Mg/ha")
    print(f"  MAE:  {sens_row['MAE_Mg_ha']:.2f} Mg/ha")
    print(f"  R2:   {sens_row['R2']:.3f}")
    print(f"  r:    {sens_row['Pearson_r']:.3f}")

    print("\nCreated:")
    print(f" - {TABLE_DIR / 'GAM_fold_metrics.csv'}")
    print(f" - {TABLE_DIR / 'GAM_overall_metrics.csv'}")
    print(f" - {TABLE_DIR / 'GAM_best_parameters_by_fold.csv'}")
    print(f" - {TABLE_DIR / 'GAM_OOF_predictions.csv'}")
    print(f" - {TABLE_DIR / 'GAM_permutation_importance_summary.csv'}")
    print(f" - {TABLE_DIR / 'GAM_main_smooth_response_curves.csv'}")
    print(f" - {TABLE_DIR / 'final_model_comparison_through_GAM.csv'}")
    print(f" - {OUT_DIR / 'GAM_QC_REPORT.txt'}")
    print(f" - {FIG_DIR / 'GAM4_observed_vs_predicted.png'}")
    print(f" - {FIG_DIR / 'GAM4_residuals_vs_predicted.png'}")
    print(f" - {FIG_DIR / 'GAM_4P_vs_3P_RMSE_by_fold.png'}")
    print(f" - {FIG_DIR / 'GAM4_permutation_importance.png'}")
    print(f" - {FIG_DIR / 'final_model_comparison_through_GAM_R2.png'}")
    print(f" - Smooth-response PNG/PDF files in {CURVE_DIR}")

    print("\nIMPORTANT:")
    print("After this run, upload the Phase-7G outputs for review.")
    None  # interactive pause removed for portable execution

if __name__ == "__main__":
    main()
