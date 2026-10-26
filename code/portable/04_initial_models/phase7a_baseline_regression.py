import os
from pathlib import Path
from datetime import datetime
import hashlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

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

OUT_DIR = PROJECT_ROOT / "06_Results" / "Baseline"
TABLE_DIR = OUT_DIR / "Tables"
FIG_DIR = OUT_DIR / "Figures"
MODEL_DIR = PROJECT_ROOT / "05_ML_DL" / "Baseline"
LOG_PATH = PROJECT_ROOT / "00_Documentation" / "analysis_log.txt"

RESPONSE = "agbd"
PREDICTORS = ["HH_dB", "HV_dB", "HV_SD"]
N_FOLDS = 5

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

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
        f.write("\n[" + stamp + "] PHASE 7A — BASELINE LINEAR REGRESSION\n")
        for line in lines:
            f.write(line + "\n")
        f.write("-" * 72 + "\n")

def metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    bias = float(np.mean(y_pred - y_true))
    mean_obs = float(np.mean(y_true))
    nrmse_pct = float(100.0 * rmse / mean_obs) if mean_obs != 0 else np.nan

    if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        r = float(pearsonr(y_true, y_pred).statistic)
    else:
        r = np.nan

    return {
        "RMSE_Mg_ha": rmse,
        "MAE_Mg_ha": mae,
        "R2": r2,
        "Bias_Mg_ha": bias,
        "nRMSE_percent_of_mean": nrmse_pct,
        "Pearson_r": r,
    }

def check_columns(df, path):
    required = [RESPONSE] + PREDICTORS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns in {path.name}: {missing}")

def main():
    print("=" * 72)
    print("PHASE 7A — BASELINE LINEAR REGRESSION")
    print("=" * 72)
    print(f"Response: {RESPONSE}")
    print(f"Predictors: {', '.join(PREDICTORS)}")
    print("Validation: frozen 5-fold spatial CV + 500 m training buffer")

    fold_metrics = []
    fold_coefficients = []
    oof_parts = []
    provenance = []

    for fold in range(1, N_FOLDS + 1):
        fold_dir = BUFFER_ROOT / f"Fold_{fold}"
        train_path = fold_dir / f"Fold_{fold}_TRAIN_BUFFERED_500m.csv"
        test_path = fold_dir / f"Fold_{fold}_TEST.csv"

        if not train_path.exists():
            raise FileNotFoundError(f"Missing buffered training file:\n{train_path}")
        if not test_path.exists():
            raise FileNotFoundError(f"Missing test file:\n{test_path}")

        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        check_columns(train, train_path)
        check_columns(test, test_path)

        # Strict NA/finite check; no silent imputation in baseline.
        train_valid = np.isfinite(train[[RESPONSE] + PREDICTORS].to_numpy(dtype=float)).all(axis=1)
        test_valid = np.isfinite(test[[RESPONSE] + PREDICTORS].to_numpy(dtype=float)).all(axis=1)

        if not train_valid.all():
            raise RuntimeError(f"Fold {fold}: non-finite values found in training predictors/response.")
        if not test_valid.all():
            raise RuntimeError(f"Fold {fold}: non-finite values found in test predictors/response.")

        # Optional identity leakage check.
        if "shot_number" in train.columns and "shot_number" in test.columns:
            overlap = set(train["shot_number"]).intersection(set(test["shot_number"]))
            if overlap:
                raise RuntimeError(f"Fold {fold}: train/test shot_number overlap detected ({len(overlap)} records).")

        X_train = train[PREDICTORS].to_numpy(dtype=float)
        y_train = train[RESPONSE].to_numpy(dtype=float)
        X_test = test[PREDICTORS].to_numpy(dtype=float)
        y_test = test[RESPONSE].to_numpy(dtype=float)

        # Scaling is fitted on TRAINING DATA ONLY.
        # For ordinary least squares, scaling does not change fitted predictions;
        # it improves numerical conditioning and provides comparable standardized coefficients.
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression())
        ])
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        m = metrics(y_test, pred)
        m.update({
            "fold": fold,
            "train_n": len(train),
            "test_n": len(test),
            "train_agbd_mean": float(np.mean(y_train)),
            "test_agbd_mean": float(np.mean(y_test)),
        })
        fold_metrics.append(m)

        scaler = model.named_steps["scaler"]
        reg = model.named_steps["regressor"]

        # Convert standardized-fit coefficients back to original predictor units.
        raw_coef = reg.coef_ / scaler.scale_
        raw_intercept = reg.intercept_ - np.sum(reg.coef_ * scaler.mean_ / scaler.scale_)

        fold_coefficients.append({
            "fold": fold,
            "intercept_raw_units": float(raw_intercept),
            **{f"coef_raw_{p}": float(c) for p, c in zip(PREDICTORS, raw_coef)},
            **{f"coef_standardizedX_{p}": float(c) for p, c in zip(PREDICTORS, reg.coef_)},
        })

        pred_df = test.copy()
        pred_df["outer_fold"] = fold
        pred_df["observed_agbd"] = y_test
        pred_df["predicted_agbd_baseline"] = pred
        pred_df["residual_observed_minus_predicted"] = y_test - pred
        oof_parts.append(pred_df)

        provenance.append({
            "fold": fold,
            "train_file": str(train_path),
            "train_sha256": sha256_file(train_path),
            "test_file": str(test_path),
            "test_sha256": sha256_file(test_path),
        })

        print(
            f"Fold {fold}: train={len(train):,}, test={len(test):,}, "
            f"RMSE={m['RMSE_Mg_ha']:.2f}, MAE={m['MAE_Mg_ha']:.2f}, "
            f"R2={m['R2']:.3f}, r={m['Pearson_r']:.3f}"
        )

    fm = pd.DataFrame(fold_metrics).sort_values("fold")
    fc = pd.DataFrame(fold_coefficients).sort_values("fold")
    oof = pd.concat(oof_parts, ignore_index=True)
    prov = pd.DataFrame(provenance).sort_values("fold")

    # Pooled out-of-fold metrics: each record predicted only when held out.
    pooled = metrics(
        oof["observed_agbd"].to_numpy(dtype=float),
        oof["predicted_agbd_baseline"].to_numpy(dtype=float)
    )
    pooled.update({
        "evaluation": "pooled_out_of_fold",
        "n": len(oof),
    })

    fold_mean = {
        "evaluation": "mean_of_5_outer_folds",
        "n": len(oof),
        "RMSE_Mg_ha": float(fm["RMSE_Mg_ha"].mean()),
        "MAE_Mg_ha": float(fm["MAE_Mg_ha"].mean()),
        "R2": float(fm["R2"].mean()),
        "Bias_Mg_ha": float(fm["Bias_Mg_ha"].mean()),
        "nRMSE_percent_of_mean": float(fm["nRMSE_percent_of_mean"].mean()),
        "Pearson_r": float(fm["Pearson_r"].mean()),
    }

    fold_sd = {
        "evaluation": "sd_across_5_outer_folds",
        "n": len(oof),
        "RMSE_Mg_ha": float(fm["RMSE_Mg_ha"].std(ddof=1)),
        "MAE_Mg_ha": float(fm["MAE_Mg_ha"].std(ddof=1)),
        "R2": float(fm["R2"].std(ddof=1)),
        "Bias_Mg_ha": float(fm["Bias_Mg_ha"].std(ddof=1)),
        "nRMSE_percent_of_mean": float(fm["nRMSE_percent_of_mean"].std(ddof=1)),
        "Pearson_r": float(fm["Pearson_r"].std(ddof=1)),
    }

    overall = pd.DataFrame([pooled, fold_mean, fold_sd])

    # Coefficient stability summary
    numeric_coef_cols = [c for c in fc.columns if c != "fold"]
    coef_summary = pd.DataFrame({
        "parameter": numeric_coef_cols,
        "mean_across_folds": [fc[c].mean() for c in numeric_coef_cols],
        "sd_across_folds": [fc[c].std(ddof=1) for c in numeric_coef_cols],
        "min_across_folds": [fc[c].min() for c in numeric_coef_cols],
        "max_across_folds": [fc[c].max() for c in numeric_coef_cols],
    })

    fm.to_csv(TABLE_DIR / "baseline_fold_metrics.csv", index=False)
    overall.to_csv(TABLE_DIR / "baseline_overall_metrics.csv", index=False)
    fc.to_csv(TABLE_DIR / "baseline_coefficients_by_fold.csv", index=False)
    coef_summary.to_csv(TABLE_DIR / "baseline_coefficient_stability.csv", index=False)
    oof.to_csv(TABLE_DIR / "baseline_OOF_predictions.csv", index=False)
    prov.to_csv(TABLE_DIR / "baseline_input_provenance.csv", index=False)

    # --------------------------------------------------------
    # FIGURES
    # --------------------------------------------------------
    # Observed vs predicted
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    ax.scatter(
        oof["observed_agbd"],
        oof["predicted_agbd_baseline"],
        s=16,
        alpha=0.45
    )
    lo = float(min(oof["observed_agbd"].min(), oof["predicted_agbd_baseline"].min()))
    hi = float(max(oof["observed_agbd"].max(), oof["predicted_agbd_baseline"].max()))
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2)
    ax.set_xlabel("Observed GEDI AGBD (Mg/ha)")
    ax.set_ylabel("Predicted AGBD (Mg/ha)")
    ax.set_title("Baseline linear regression — pooled spatial-CV predictions", pad=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "baseline_observed_vs_predicted.png", dpi=450, bbox_inches="tight")
    fig.savefig(FIG_DIR / "baseline_observed_vs_predicted.pdf", bbox_inches="tight")
    plt.close(fig)

    # Residual vs predicted
    fig, ax = plt.subplots(figsize=(7.4, 6.0))
    ax.scatter(
        oof["predicted_agbd_baseline"],
        oof["residual_observed_minus_predicted"],
        s=16,
        alpha=0.45
    )
    ax.axhline(0.0, linestyle="--", linewidth=1.2)
    ax.set_xlabel("Predicted AGBD (Mg/ha)")
    ax.set_ylabel("Residual: observed − predicted (Mg/ha)")
    ax.set_title("Baseline linear regression — residual diagnostic", pad=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "baseline_residuals_vs_predicted.png", dpi=450, bbox_inches="tight")
    fig.savefig(FIG_DIR / "baseline_residuals_vs_predicted.pdf", bbox_inches="tight")
    plt.close(fig)

    # Fold RMSE
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    ax.bar(fm["fold"].astype(str), fm["RMSE_Mg_ha"])
    ax.set_xlabel("Outer spatial fold")
    ax.set_ylabel("RMSE (Mg/ha)")
    ax.set_title("Baseline RMSE by buffered spatial fold", pad=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "baseline_RMSE_by_fold.png", dpi=450, bbox_inches="tight")
    fig.savefig(FIG_DIR / "baseline_RMSE_by_fold.pdf", bbox_inches="tight")
    plt.close(fig)

    report = [
        "PHASE 7A — BASELINE LINEAR REGRESSION",
        "=" * 72,
        f"Response: {RESPONSE}",
        f"Predictors: {', '.join(PREDICTORS)}",
        "Excluded from linear baseline: HH_minus_HV",
        "Reason: HH_minus_HV = HH_dB - HV_dB, so simultaneous inclusion with HH_dB and HV_dB creates exact algebraic redundancy.",
        "Validation: frozen 5-fold spatial block CV with 500 m training exclusion buffer.",
        "Preprocessing: StandardScaler fitted independently within each buffered training fold only.",
        "",
        "Pooled out-of-fold metrics:",
        pd.DataFrame([pooled]).to_string(index=False),
        "",
        "Per-fold metrics:",
        fm.to_string(index=False),
        "",
        "Coefficient stability:",
        coef_summary.to_string(index=False),
        "",
        "Interpretation rule:",
        "- Treat this model as the transparent linear benchmark.",
        "- Do not tune or transform the baseline after seeing test-fold performance.",
        "- RF, XGBoost, and DNN must use the same frozen outer test folds.",
    ]
    (OUT_DIR / "BASELINE_QC_REPORT.txt").write_text("\n".join(report), encoding="utf-8")

    append_log([
        "Script: phase7a_baseline_regression.py",
        f"Response: {RESPONSE}",
        f"Predictors: {', '.join(PREDICTORS)}",
        "HH_minus_HV excluded from baseline because it is algebraically dependent on HH_dB and HV_dB.",
        "Validation: frozen Phase-6 folds with Phase-6B 500 m buffered training sets.",
        f"Pooled OOF RMSE: {pooled['RMSE_Mg_ha']:.4f} Mg/ha",
        f"Pooled OOF MAE: {pooled['MAE_Mg_ha']:.4f} Mg/ha",
        f"Pooled OOF R2: {pooled['R2']:.6f}",
        f"Pooled OOF Pearson r: {pooled['Pearson_r']:.6f}",
    ])

    print("\n" + "=" * 72)
    print("PHASE 7A COMPLETE — BASELINE REVIEW REQUIRED")
    print("=" * 72)
    print(f"Pooled OOF RMSE: {pooled['RMSE_Mg_ha']:.2f} Mg/ha")
    print(f"Pooled OOF MAE:  {pooled['MAE_Mg_ha']:.2f} Mg/ha")
    print(f"Pooled OOF R2:   {pooled['R2']:.3f}")
    print(f"Pooled OOF r:    {pooled['Pearson_r']:.3f}")

    print("\nCreated:")
    print(f" - {TABLE_DIR / 'baseline_fold_metrics.csv'}")
    print(f" - {TABLE_DIR / 'baseline_overall_metrics.csv'}")
    print(f" - {TABLE_DIR / 'baseline_coefficients_by_fold.csv'}")
    print(f" - {TABLE_DIR / 'baseline_coefficient_stability.csv'}")
    print(f" - {TABLE_DIR / 'baseline_OOF_predictions.csv'}")
    print(f" - {OUT_DIR / 'BASELINE_QC_REPORT.txt'}")
    print(f" - {FIG_DIR / 'baseline_observed_vs_predicted.png'}")
    print(f" - {FIG_DIR / 'baseline_residuals_vs_predicted.png'}")
    print(f" - {FIG_DIR / 'baseline_RMSE_by_fold.png'}")

    print("\nIMPORTANT:")
    print("Do not start Random Forest yet. Upload the baseline QC outputs for review.")
    None  # interactive pause removed for portable execution

if __name__ == "__main__":
    main()
