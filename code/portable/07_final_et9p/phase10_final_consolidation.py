import os
from pathlib import Path
from datetime import datetime
import hashlib
import sys
import traceback

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# PHASE 10 — FINAL MANUSCRIPT / FIGURE CONSOLIDATION
# ============================================================
PROJECT = Path(os.environ.get("NISAR_PROJECT_ROOT", r"D:\NISAR_Mangrove_Paper"))

PH8 = (
    PROJECT / "06_Results" / "Predictor_Expansion"
    / "Phase8C4_Expanded9P" / "Tables"
)
PH9 = (
    PROJECT / "06_Results" / "Predictor_Expansion"
    / "Phase9_ET9P_Final_Defensibility" / "Tables"
)

OUT = (
    PROJECT / "07_Manuscript"
    / "Phase10_Final_Consolidation"
)
FIG = OUT / "Figures"
TAB = OUT / "Tables"
TXT = OUT / "Text"
QC = OUT / "QC"
for d in [FIG, TAB, TXT, QC]:
    d.mkdir(parents=True, exist_ok=True)

FILES = {
    "pooled9p": PH8 / "expanded9P_pooled_metrics.csv",
    "regimes": PH8 / "expanded9P_performance_by_AGBD_regime.csv",
    "compare": PH8 / "original3P_vs_expanded9P_comparison.csv",
    "core": PH9 / "ET3P_ET9P_core_metrics.csv",
    "boot": PH9 / "ET9P_vs_ET3P_block_bootstrap_summary.csv",
    "folds": PH9 / "ET9P_vs_ET3P_fold_comparison.csv",
    "perm": PH9 / "ET9P_permutation_importance_aggregated.csv",
    "moran": PH9 / "ET9P_residual_MoransI.csv",
}

def sha256(path, chunk=8*1024*1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

def require():
    missing = [str(p) for p in FILES.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

def save_fig(fig, stem):
    fig.tight_layout()
    fig.savefig(FIG / f"{stem}.png", dpi=450, bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)

def main():
    print("="*92)
    print("PHASE 10 — FINAL MANUSCRIPT / FIGURE CONSOLIDATION")
    print("="*92)
    print("No models are fit. No predictors are changed. No biomass map is created.\n")

    require()

    pooled = pd.read_csv(FILES["pooled9p"])
    regimes = pd.read_csv(FILES["regimes"])
    compare = pd.read_csv(FILES["compare"])
    core = pd.read_csv(FILES["core"])
    boot = pd.read_csv(FILES["boot"])
    folds = pd.read_csv(FILES["folds"])
    perm = pd.read_csv(FILES["perm"])
    moran = pd.read_csv(FILES["moran"])

    # ------------------------------------------------------------
    # Locked final-model table
    # ------------------------------------------------------------
    et9 = pooled[pooled["model"] == "EXPANDED_ET_9P"].iloc[0]
    et3 = core[core["model"] == "ET_3P"].iloc[0]
    et9core = core[core["model"] == "ET_9P"].iloc[0]

    final_metrics = pd.DataFrame([{
        "final_model": "Extra Trees 9P",
        "n": int(et9["n"]),
        "RMSE_Mg_ha": float(et9["RMSE_Mg_ha"]),
        "MAE_Mg_ha": float(et9["MAE_Mg_ha"]),
        "R2": float(et9["R2"]),
        "Pearson_r": float(et9["Pearson_r"]),
        "Spearman_rho": float(et9["Spearman_rho"]),
        "Bias_Mg_ha": float(et9["Bias_Mg_ha"]),
        "rRMSE_percent": float(et9["rRMSE_percent"]),
        "predicted_SD": float(et9["predicted_sd"]),
        "observed_SD": float(et9["observed_sd"]),
        "pred_to_obs_SD_ratio": float(et9["pred_to_obs_SD_ratio"]),
    }])
    final_metrics.to_csv(TAB / "FINAL_ET9P_METRICS.csv", index=False)

    # ------------------------------------------------------------
    # Final comparison table
    # ------------------------------------------------------------
    et_cmp = compare[compare["model_family"] == "Extra Trees"].copy()
    et_cmp.to_csv(TAB / "FINAL_ET3P_vs_ET9P_COMPARISON.csv", index=False)

    # ------------------------------------------------------------
    # Biomass-regime table
    # ------------------------------------------------------------
    et_reg = regimes[regimes["model"] == "EXPANDED_ET_9P"].copy()
    et_reg.to_csv(TAB / "FINAL_ET9P_AGBD_REGIME_PERFORMANCE.csv", index=False)

    # ------------------------------------------------------------
    # Final bootstrap table
    # ------------------------------------------------------------
    boot.to_csv(TAB / "FINAL_ET9P_BOOTSTRAP_CI.csv", index=False)

    # ------------------------------------------------------------
    # Final permutation table
    # ------------------------------------------------------------
    perm.to_csv(TAB / "FINAL_ET9P_PERMUTATION_IMPORTANCE.csv", index=False)

    # ------------------------------------------------------------
    # Final Moran table
    # ------------------------------------------------------------
    moran.to_csv(TAB / "FINAL_ET9P_RESIDUAL_MORANS_I.csv", index=False)

    # ------------------------------------------------------------
    # Figure 1 — ET3P vs ET9P RMSE/R2/SD ratio
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.8, 5.1))
    vals = [float(et3["RMSE"]), float(et9core["RMSE"])]
    ax.bar(["ET 3P", "ET 9P"], vals)
    ax.set_ylabel("RMSE (Mg/ha)")
    ax.set_title("Held-out RMSE: original vs expanded Extra Trees", pad=12)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.25, f"{v:.2f}", ha="center", va="bottom")
    save_fig(fig, "Fig_Final_ET_RMSE")

    fig, ax = plt.subplots(figsize=(7.8, 5.1))
    vals = [float(et3["R2"]), float(et9core["R2"])]
    ax.bar(["ET 3P", "ET 9P"], vals)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_ylabel("Pooled out-of-fold R²")
    ax.set_title("Held-out R²: original vs expanded Extra Trees", pad=12)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.003, f"{v:.3f}", ha="center", va="bottom")
    save_fig(fig, "Fig_Final_ET_R2")

    fig, ax = plt.subplots(figsize=(7.8, 5.1))
    vals = [float(et3["pred_to_obs_SD_ratio"]), float(et9core["pred_to_obs_SD_ratio"])]
    ax.bar(["ET 3P", "ET 9P"], vals)
    ax.axhline(1.0, linestyle="--", linewidth=1, label="Ideal = 1")
    ax.set_ylabel("Predicted SD / observed SD")
    ax.set_title("Prediction dynamic-range recovery", pad=12)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.015, f"{v:.3f}", ha="center", va="bottom")
    save_fig(fig, "Fig_Final_ET_SD_ratio")

    # ------------------------------------------------------------
    # Figure 2 — Fold RMSE consistency
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    x = folds["outer_fold"].to_numpy()
    ax.plot(x, folds["old_RMSE"], marker="o", label="ET 3P")
    ax.plot(x, folds["new_RMSE"], marker="o", label="ET 9P")
    ax.set_xticks(x)
    ax.set_xlabel("Outer spatial fold")
    ax.set_ylabel("RMSE (Mg/ha)")
    ax.set_title("Fold-by-fold held-out RMSE", pad=12)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    save_fig(fig, "Fig_Final_Fold_RMSE")

    # ------------------------------------------------------------
    # Figure 3 — Permutation importance
    # ------------------------------------------------------------
    d = perm.sort_values("mean_delta_RMSE", ascending=True)
    fig, ax = plt.subplots(figsize=(8.8, 6.3))
    ax.barh(d["predictor"], d["mean_delta_RMSE"])
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Mean increase in held-out RMSE after permutation (Mg/ha)")
    ax.set_title("Held-out permutation importance of ET 9P", pad=12)
    save_fig(fig, "Fig_Final_Permutation_Importance")

    # ------------------------------------------------------------
    # Figure 4 — Residual Moran
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    ax.plot(moran["distance_m"]/1000.0, moran["Morans_I_residual"], marker="o")
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Distance threshold (km)")
    ax.set_ylabel("Moran's I of out-of-fold residuals")
    ax.set_title("Residual spatial autocorrelation of ET 9P", pad=12)
    save_fig(fig, "Fig_Final_Residual_MoransI")

    # ------------------------------------------------------------
    # Figure 5 — Bias by AGBD regime
    # ------------------------------------------------------------
    order = ["<50", "50-<100", ">=100"]
    er = et_reg.set_index("AGBD_regime").loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    ax.bar(er["AGBD_regime"], er["Bias_Mg_ha"])
    ax.axhline(0, linewidth=1)
    ax.set_xlabel("Observed GEDI AGBD regime (Mg/ha)")
    ax.set_ylabel("Mean bias (Mg/ha)")
    ax.set_title("ET 9P bias across biomass regimes", pad=12)
    save_fig(fig, "Fig_Final_ET9P_Bias_by_AGBD_Regime")

    # ------------------------------------------------------------
    # Manuscript-ready text
    # ------------------------------------------------------------
    rmse_boot = boot[boot["metric"] == "delta_RMSE_new_minus_old"].iloc[0]
    mae_boot = boot[boot["metric"] == "delta_MAE_new_minus_old"].iloc[0]
    r2_boot = boot[boot["metric"] == "delta_R2_new_minus_old"].iloc[0]

    top3 = perm.sort_values("mean_delta_RMSE", ascending=False).head(3)
    top3_text = "; ".join(
        f"{r.predictor} (ΔRMSE={r.mean_delta_RMSE:.2f} Mg/ha)"
        for r in top3.itertuples()
    )

    results_text = f"""FINAL RESULTS TEXT — EXTRA TREES 9P

Under the frozen five-fold spatial cross-validation design with 3-km blocks and a 500-m training exclusion buffer, the expanded nine-predictor Extra Trees model produced the best held-out performance. The model achieved RMSE = {float(et9["RMSE_Mg_ha"]):.2f} Mg/ha, MAE = {float(et9["MAE_Mg_ha"]):.2f} Mg/ha, R² = {float(et9["R2"]):.3f}, Pearson r = {float(et9["Pearson_r"]):.3f}, and Spearman ρ = {float(et9["Spearman_rho"]):.3f}. Relative RMSE was {float(et9["rRMSE_percent"]):.1f}% of the mean observed AGBD.

Compared with the original three-predictor Extra Trees model, predictor expansion reduced RMSE from {float(et3["RMSE"]):.2f} to {float(et9core["RMSE"]):.2f} Mg/ha and increased R² from {float(et3["R2"]):.3f} to {float(et9core["R2"]):.3f}. A paired 3-km spatial-block bootstrap with 5,000 replicates supported this improvement: the mean change in RMSE (ET9P − ET3P) was {float(rmse_boot["mean"]):.2f} Mg/ha with a 95% interval of {float(rmse_boot["CI2.5"]):.2f} to {float(rmse_boot["CI97.5"]):.2f} Mg/ha. The mean change in MAE was {float(mae_boot["mean"]):.2f} Mg/ha (95% interval {float(mae_boot["CI2.5"]):.2f} to {float(mae_boot["CI97.5"]):.2f}), while the mean change in R² was +{float(r2_boot["mean"]):.3f} (95% interval {float(r2_boot["CI2.5"]):.3f} to {float(r2_boot["CI97.5"]):.3f}). ET9P had lower RMSE than ET3P in all five outer folds.

Held-out permutation importance indicated that the strongest predictive contributions came from {top3_text}. These importance values represent predictive contribution rather than ecological causation.

Despite the improvement, prediction compression remained substantial. The ET9P predicted standard deviation was {float(et9core["predicted_SD"]):.2f} Mg/ha compared with an observed standard deviation of {float(et9core["observed_SD"]):.2f} Mg/ha, corresponding to a predicted-to-observed SD ratio of {float(et9core["pred_to_obs_SD_ratio"]):.3f}. Biomass-regime diagnostics also showed systematic bias: the model overpredicted low-biomass observations (<50 Mg/ha) and underpredicted moderate and high biomass, with the strongest underprediction in observations >=100 Mg/ha.

Residual spatial autocorrelation remained detectable after model fitting. Moran's I declined from {float(moran.iloc[0]["Morans_I_residual"]):.3f} at 500 m to {float(moran.iloc[-1]["Morans_I_residual"]):.3f} at 3 km, with permutation p = {float(moran.iloc[-1]["permutation_p_two_sided"]):.3f} at 3 km.

Accordingly, the expanded multiscale predictor set yielded a robust and reproducible improvement in NISAR-based AGBD sensitivity, but the absolute held-out predictive power remained limited and predictions remained strongly compressed. The model is therefore retained as the final analytical model for sensitivity and methodological evaluation, rather than as an operational wall-to-wall AGBD estimator.
"""

    methods_text = """FINAL METHODS TEXT — VALIDATION AND MODEL SELECTION

Model development used a frozen set of 1,641 GEDI footprints. Spatial dependence was addressed with five-fold spatial cross-validation using 3-km blocks. For each outer test fold, training observations within 500 m of the held-out fold were excluded. Extra Trees hyperparameters were tuned only within each buffered training set using grouped inner cross-validation by spatial block. The outer test folds were never used for predictor selection or tuning.

Predictor expansion was pre-specified and based on physically interpretable multiscale spatial context derived from NISAR HH and HV backscatter. The final nine-predictor set retained the original HH_dB, HV_dB, and HV_SD variables and added HH_SD_dB_3x3, HH_MeanLin_dB_9x9, HV_MeanLin_dB_9x9, HH_SD_dB_9x9, HV_SD_dB_9x9, and PolContrast_SD_dB_9x9. The algebraically redundant HH_minus_HV term was excluded.

Final model defensibility was assessed using a paired spatial-block bootstrap in which the occupied 3-km blocks were resampled with replacement. Held-out permutation importance was computed by refitting the already-selected Extra Trees model within each buffered training fold and permuting one predictor at a time only in the corresponding outer test fold. Residual spatial dependence was evaluated using Moran's I of pooled out-of-fold residuals at distance thresholds of 500 m, 1 km, 2 km, and 3 km.
"""

    conclusion_text = """FINAL INTERPRETATION / CONCLUSION TEXT

Multiscale contextual NISAR predictors improved spatially validated GEDI-derived mangrove AGBD prediction relative to the original dual-polarization predictor set, and the improvement was robust across all five spatial folds and under 3-km block bootstrap resampling. Nevertheless, overall model skill remained low, with R² below 0.10, substantial compression of the predicted biomass distribution, systematic overprediction at low biomass and underprediction at high biomass, and residual spatial autocorrelation. These results indicate that single-date dual-polarization NISAR GCOV contains measurable but limited information for stand-level AGBD retrieval in this setting. The final Extra Trees model is therefore interpreted as a sensitivity-analysis model rather than an operational biomass-mapping model.
"""

    (TXT / "FINAL_RESULTS_TEXT.txt").write_text(results_text, encoding="utf-8")
    (TXT / "FINAL_METHODS_TEXT.txt").write_text(methods_text, encoding="utf-8")
    (TXT / "FINAL_INTERPRETATION_TEXT.txt").write_text(conclusion_text, encoding="utf-8")

    # ------------------------------------------------------------
    # QC report
    # ------------------------------------------------------------
    hashes = {k: sha256(v) for k, v in FILES.items()}
    qc_text = [
        "PHASE 10 — FINAL MANUSCRIPT / FIGURE CONSOLIDATION",
        "="*92,
        f"Run time: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "NO MODELS WERE FIT.",
        "NO PREDICTORS WERE CHANGED.",
        "NO FINAL AGBD MAP WAS CREATED.",
        "",
        "FINAL MODEL: Extra Trees 9P",
        f"RMSE: {float(et9['RMSE_Mg_ha']):.6f}",
        f"MAE: {float(et9['MAE_Mg_ha']):.6f}",
        f"R2: {float(et9['R2']):.6f}",
        f"Pearson r: {float(et9['Pearson_r']):.6f}",
        f"Spearman rho: {float(et9['Spearman_rho']):.6f}",
        f"Prediction SD ratio: {float(et9['pred_to_obs_SD_ratio']):.6f}",
        "",
        "INPUT HASHES",
    ]
    for k, h in hashes.items():
        qc_text.append(f"{k}: {h}")

    qc_text += [
        "",
        "OUTPUTS",
        str(TAB / "FINAL_ET9P_METRICS.csv"),
        str(TAB / "FINAL_ET3P_vs_ET9P_COMPARISON.csv"),
        str(TAB / "FINAL_ET9P_AGBD_REGIME_PERFORMANCE.csv"),
        str(TAB / "FINAL_ET9P_BOOTSTRAP_CI.csv"),
        str(TAB / "FINAL_ET9P_PERMUTATION_IMPORTANCE.csv"),
        str(TAB / "FINAL_ET9P_RESIDUAL_MORANS_I.csv"),
        str(TXT / "FINAL_RESULTS_TEXT.txt"),
        str(TXT / "FINAL_METHODS_TEXT.txt"),
        str(TXT / "FINAL_INTERPRETATION_TEXT.txt"),
        "",
        "STATUS: PHASE 10 COMPLETE",
        "FINAL QUANTITATIVE WALL-TO-WALL AGBD MAP: NOT RECOMMENDED.",
    ]

    report = QC / "PHASE10_FINAL_CONSOLIDATION_REPORT.txt"
    report.write_text("\n".join(qc_text), encoding="utf-8")

    print("\n" + "="*92)
    print("PHASE 10 COMPLETE")
    print("="*92)
    print("\nFinal model:")
    print(final_metrics.to_string(index=False))
    print("\nCreated manuscript-ready tables, figures, and text under:")
    print(OUT)
    print("\nIMPORTANT:")
    print("This phase does NOT authorize a quantitative final AGBD map.")
    print("Use the consolidated outputs for manuscript drafting.")
    None  # interactive pause removed for portable execution

if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\nPHASE 10 FAILED")
        traceback.print_exc()
        None  # interactive pause removed for portable execution
        sys.exit(1)
