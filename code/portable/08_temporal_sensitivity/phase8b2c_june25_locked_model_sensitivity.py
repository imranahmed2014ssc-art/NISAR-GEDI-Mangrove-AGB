from pathlib import Path
from datetime import datetime
import hashlib
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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# ISOLATED JUNE WORKSPACE
# ============================================================
JUNE_ROOT = Path(os.environ.get("NISAR_JUNE_PHASE_ROOT", str(Path(os.environ.get("NISAR_JUNE_ROOT", r"D:\new_nisar_data")) / "Phase8B2_June25_Sensitivity_V2")))
MATCH_TABLE = (
    JUNE_ROOT
    / "02_GEDI_Matching"
    / "Tables"
    / "GEDI_June25_matched_all_1641.csv"
)
MATCH_QC = (
    JUNE_ROOT
    / "02_GEDI_Matching"
    / "QC"
    / "JUNE25_GEDI_MATCHING_QC.txt"
)

OUT_ROOT = JUNE_ROOT / "03_Temporal_Model_Sensitivity"
TABLE_DIR = OUT_ROOT / "Tables"
FIG_DIR = OUT_ROOT / "Figures"
REPORT_DIR = OUT_ROOT / "QC"
for d in [TABLE_DIR, FIG_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Existing project is READ ONLY.
MAIN_ROOT = Path(os.environ.get("NISAR_PROJECT_ROOT", r"D:\NISAR_Mangrove_Paper"))
BUFFER_ROOT = (
    MAIN_ROOT / "05_ML_DL" / "Spatial_CV" / "Buffered_500m"
)

SEP_BASELINE_OOF = (
    MAIN_ROOT / "06_Results" / "Baseline" / "Tables"
    / "baseline_OOF_predictions.csv"
)
SEP_ET_OOF = (
    MAIN_ROOT / "06_Results" / "Extra_Trees" / "Tables"
    / "ET_OOF_predictions.csv"
)
SEP_MLP_OOF = (
    MAIN_ROOT / "06_Results" / "DNN_MLP" / "Tables"
    / "MLP_OOF_predictions.csv"
)

EXPECTED_FROZEN_HASH = (
    "c624d8c3c29e295778012b2998058cd454d82d7e9655d0e128bdd8e8779db24e"
)
EXPECTED_N = 1641
RANDOM_SEED = 42

PRED_JUNE = ["June25_HH_dB", "June25_HV_dB", "June25_HV_SD"]
PRED_RENAME = {
    "June25_HH_dB": "HH_dB",
    "June25_HV_dB": "HV_dB",
    "June25_HV_SD": "HV_SD",
}

# ============================================================
# LOCKED SEPTEMBER-SELECTED HYPERPARAMETERS
# These are NOT re-tuned on the June scene.
# This isolates the effect of changing NISAR acquisition date.
# ============================================================
ET_PARAMS = {
    1: dict(n_estimators=700, min_samples_split=10, min_samples_leaf=6,
            max_features=0.8, max_depth=8, bootstrap=True),
    2: dict(n_estimators=700, min_samples_split=10, min_samples_leaf=6,
            max_features=0.8, max_depth=8, bootstrap=True),
    3: dict(n_estimators=700, min_samples_split=10, min_samples_leaf=6,
            max_features=0.8, max_depth=8, bootstrap=True),
    4: dict(n_estimators=700, min_samples_split=5, min_samples_leaf=6,
            max_features=0.8, max_depth=8, bootstrap=False),
    5: dict(n_estimators=700, min_samples_split=10, min_samples_leaf=6,
            max_features=0.8, max_depth=8, bootstrap=True),
}

MLP_PARAMS = {
    1: dict(learning_rate_init=0.0010, hidden_layer_sizes=(32,),
            alpha=0.1000, activation="relu"),
    2: dict(learning_rate_init=0.0010, hidden_layer_sizes=(32,),
            alpha=0.1000, activation="relu"),
    3: dict(learning_rate_init=0.0005, hidden_layer_sizes=(64,),
            alpha=0.1000, activation="tanh"),
    4: dict(learning_rate_init=0.0005, hidden_layer_sizes=(64,),
            alpha=0.1000, activation="tanh"),
    5: dict(learning_rate_init=0.0005, hidden_layer_sizes=(64,),
            alpha=0.1000, activation="tanh"),
}

# September pooled reference values already established in the project.
# The script also reads actual September OOF tables and recalculates them.
KNOWN_SEP = {
    "Linear_baseline_3P": {
        "RMSE": 41.32, "MAE": 34.94, "R2": 0.0098, "Pearson_r": 0.102
    },
    "ET_3P_SENSITIVITY": {
        "RMSE": 41.262069, "MAE": 34.980262, "R2": 0.012656, "Pearson_r": 0.112509
    },
    "MLP_3P_SENSITIVITY": {
        "RMSE": 41.219863, "MAE": 34.808311, "R2": 0.014675, "Pearson_r": 0.124057
    },
}

# ============================================================
# HELPERS
# ============================================================
def sha256_file(path, chunk_size=8 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def metric_row(y, p, model, scene, fold=None, train_n=None, test_n=None):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
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
    return {
        "scene": scene,
        "model": model,
        "fold": fold,
        "n": int(len(y)),
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
        "observed_sd": float(np.std(y, ddof=1)),
        "predicted_sd": float(np.std(p, ddof=1)),
    }

def shot_set_from_csv(path):
    d = pd.read_csv(path)
    if "shot_number" not in d.columns:
        raise RuntimeError(f"shot_number missing from {path}")
    return set(pd.to_numeric(d["shot_number"], errors="raise").astype("int64").tolist())

def find_fold_files(fold):
    fd = BUFFER_ROOT / f"Fold_{fold}"
    train = fd / f"Fold_{fold}_TRAIN_BUFFERED_500m.csv"
    test = fd / f"Fold_{fold}_TEST.csv"
    if not train.exists() or not test.exists():
        raise FileNotFoundError(
            f"Missing frozen buffered fold files for fold {fold}:\n{train}\n{test}"
        )
    return train, test

def make_mlp(params):
    return Pipeline([
        ("scale", StandardScaler()),
        ("mlp", MLPRegressor(
            **params,
            solver="adam",
            max_iter=1500,
            early_stopping=False,
            random_state=RANDOM_SEED,
        ))
    ])

def read_sep_oof():
    parts = []

    # Linear baseline
    if SEP_BASELINE_OOF.exists():
        d = pd.read_csv(SEP_BASELINE_OOF)
        pred = "predicted_agbd_baseline"
        if {"shot_number", "observed_agbd", pred}.issubset(d.columns):
            x = d[["shot_number", "observed_agbd", pred] +
                  (["outer_fold"] if "outer_fold" in d.columns else [])].copy()
            x["model"] = "Linear_baseline_3P"
            x = x.rename(columns={pred: "predicted_agbd"})
            parts.append(x)

    for path, model_name in [
        (SEP_ET_OOF, "ET_3P_SENSITIVITY"),
        (SEP_MLP_OOF, "MLP_3P_SENSITIVITY"),
    ]:
        if path.exists():
            d = pd.read_csv(path)
            if {"model", "shot_number", "observed_agbd", "predicted_agbd"}.issubset(d.columns):
                x = d[d["model"] == model_name].copy()
                keep = ["shot_number", "observed_agbd", "predicted_agbd", "model"]
                if "outer_fold" in x.columns:
                    keep.append("outer_fold")
                elif "fold" in x.columns:
                    keep.append("fold")
                parts.append(x[keep])

    if not parts:
        raise RuntimeError("Could not load September OOF prediction tables.")

    sep = pd.concat(parts, ignore_index=True, sort=False)
    if "outer_fold" not in sep.columns and "fold" in sep.columns:
        sep = sep.rename(columns={"fold": "outer_fold"})
    return sep

def plot_scene_comparison(comp):
    models = ["Linear_baseline_3P", "ET_3P_SENSITIVITY", "MLP_3P_SENSITIVITY"]
    g = comp.set_index("model").reindex(models)

    x = np.arange(len(models))
    width = 0.34

    # RMSE
    fig, ax = plt.subplots(figsize=(8.4, 5.3))
    ax.bar(x - width/2, g["Sep17_RMSE"].values, width, label="Sep 17, 2026")
    ax.bar(x + width/2, g["June25_RMSE"].values, width, label="Jun 25, 2026")
    ax.set_xticks(x)
    ax.set_xticklabels(["Linear", "Extra Trees 3P", "MLP 3P"])
    ax.set_ylabel("RMSE (Mg/ha)")
    ax.set_title("Temporal sensitivity: pooled spatial-CV RMSE", pad=12)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "June25_vs_Sep17_RMSE.png", dpi=450, bbox_inches="tight")
    fig.savefig(FIG_DIR / "June25_vs_Sep17_RMSE.pdf", bbox_inches="tight")
    plt.close(fig)

    # R2
    fig, ax = plt.subplots(figsize=(8.4, 5.3))
    ax.bar(x - width/2, g["Sep17_R2"].values, width, label="Sep 17, 2026")
    ax.bar(x + width/2, g["June25_R2"].values, width, label="Jun 25, 2026")
    ax.axhline(0, linewidth=1.0, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(["Linear", "Extra Trees 3P", "MLP 3P"])
    ax.set_ylabel("Pooled out-of-fold R²")
    ax.set_title("Temporal sensitivity: pooled spatial-CV R²", pad=12)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "June25_vs_Sep17_R2.png", dpi=450, bbox_inches="tight")
    fig.savefig(FIG_DIR / "June25_vs_Sep17_R2.pdf", bbox_inches="tight")
    plt.close(fig)

def plot_mlp_obs_pred(oof):
    g = oof[oof["model"] == "MLP_3P_SENSITIVITY"].copy()
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    ax.scatter(g["observed_agbd"], g["predicted_agbd"], s=12, alpha=0.45)
    lo = min(g["observed_agbd"].min(), g["predicted_agbd"].min())
    hi = max(g["observed_agbd"].max(), g["predicted_agbd"].max())
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.1, label="1:1")
    ax.set_xlabel("Observed GEDI AGBD (Mg/ha)")
    ax.set_ylabel("Predicted AGBD (Mg/ha)")
    ax.set_title("June 25 MLP 3P — pooled spatial-CV predictions", pad=12)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "June25_MLP3_observed_vs_predicted.png",
                dpi=450, bbox_inches="tight")
    fig.savefig(FIG_DIR / "June25_MLP3_observed_vs_predicted.pdf",
                bbox_inches="tight")
    plt.close(fig)

def class_metrics(oof):
    edges = [0, 50, 100, np.inf]
    labels = ["<50", "50-<100", ">=100"]
    x = oof.copy()
    x["AGBD_regime"] = pd.cut(
        x["observed_agbd"], bins=edges, labels=labels,
        right=False, include_lowest=True
    )
    rows = []
    for (model, reg), g in x.groupby(["model", "AGBD_regime"], observed=False):
        if len(g) == 0:
            continue
        r = metric_row(
            g["observed_agbd"], g["predicted_agbd"],
            model=model, scene="June25", fold=None
        )
        r["AGBD_regime"] = str(reg)
        rows.append(r)
    return pd.DataFrame(rows)

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 82)
    print("PHASE 8B.2C — JUNE 25 LOCKED-MODEL TEMPORAL SENSITIVITY")
    print("=" * 82)
    print("No hyperparameter tuning is performed.")
    print("September-selected fold-specific model settings are reused.")
    print("This isolates the effect of changing the NISAR acquisition date.")
    print("Existing project remains READ ONLY.\n")

    if not MATCH_TABLE.exists():
        raise FileNotFoundError(f"Missing June matched table:\n{MATCH_TABLE}")
    if not MATCH_QC.exists():
        raise FileNotFoundError(f"Missing June matching QC:\n{MATCH_QC}")

    qctxt = MATCH_QC.read_text(encoding="utf-8", errors="replace")
    if "PHASE 8B.2B STATUS: PASS" not in qctxt:
        raise RuntimeError("June matching QC is not PASS.")

    df = pd.read_csv(MATCH_TABLE)
    if len(df) != EXPECTED_N:
        raise RuntimeError(f"Expected 1,641 June matched rows; found {len(df):,}.")

    required = {
        "shot_number", "agbd", "spatial_fold", "cv_block_id",
        "June25_HH_dB", "June25_HV_dB", "June25_HV_SD",
        "June25_valid_ge_80"
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Missing required columns in June table: {missing}")

    if not df["June25_valid_ge_80"].astype(bool).all():
        raise RuntimeError("Not all 1,641 June records pass >=80% valid radar.")

    df["shot_number"] = pd.to_numeric(df["shot_number"], errors="raise").astype("int64")
    if df["shot_number"].duplicated().any():
        raise RuntimeError("Duplicate shot_number found in June matched table.")

    # Rename June features in-memory only for convenience.
    for src, dst in PRED_RENAME.items():
        df["J_" + dst] = pd.to_numeric(df[src], errors="raise")

    df["agbd"] = pd.to_numeric(df["agbd"], errors="raise")

    june_oof_parts = []
    fold_metric_rows = []
    membership_rows = []

    for fold in range(1, 6):
        print("\n" + "-" * 82)
        print(f"OUTER FOLD {fold}")
        print("-" * 82)

        train_file, test_file = find_fold_files(fold)
        train_ids = shot_set_from_csv(train_file)
        test_ids = shot_set_from_csv(test_file)

        train = df[df["shot_number"].isin(train_ids)].copy()
        test = df[df["shot_number"].isin(test_ids)].copy()

        expected_test_fold = set(df.loc[df["spatial_fold"] == fold, "shot_number"].tolist())

        if set(test["shot_number"]) != expected_test_fold:
            raise RuntimeError(
                f"Fold {fold}: frozen TEST membership does not match spatial_fold."
            )
        if set(train["shot_number"]) != train_ids:
            missing_train = len(train_ids - set(train["shot_number"]))
            raise RuntimeError(
                f"Fold {fold}: {missing_train} buffered training shots missing from June table."
            )
        if set(test["shot_number"]) & set(train["shot_number"]):
            raise RuntimeError(f"Fold {fold}: train/test overlap detected.")

        membership_rows.append({
            "fold": fold,
            "train_n": len(train),
            "test_n": len(test),
            "expected_train_n": len(train_ids),
            "expected_test_n": len(test_ids),
            "membership_pass": True,
        })

        print(f"Train n={len(train):,}; Test n={len(test):,}")

        Xtr = train[["J_HH_dB", "J_HV_dB", "J_HV_SD"]].to_numpy(float)
        Xte = test[["J_HH_dB", "J_HV_dB", "J_HV_SD"]].to_numpy(float)
        ytr = train["agbd"].to_numpy(float)
        yte = test["agbd"].to_numpy(float)

        models = {
            "Linear_baseline_3P": LinearRegression(),
            "ET_3P_SENSITIVITY": ExtraTreesRegressor(
                **ET_PARAMS[fold],
                random_state=RANDOM_SEED,
                n_jobs=1,
            ),
            "MLP_3P_SENSITIVITY": make_mlp(MLP_PARAMS[fold]),
        }

        for model_name, model in models.items():
            print(f"  Fitting {model_name}...")
            model.fit(Xtr, ytr)
            pred = model.predict(Xte)

            met = metric_row(
                yte, pred, model_name, "June25",
                fold=fold, train_n=len(train), test_n=len(test)
            )
            fold_metric_rows.append(met)

            part = pd.DataFrame({
                "shot_number": test["shot_number"].to_numpy(),
                "outer_fold": fold,
                "observed_agbd": yte,
                "predicted_agbd": pred,
                "model": model_name,
                "scene": "June25_2026",
            })
            june_oof_parts.append(part)

            print(
                f"    RMSE={met['RMSE_Mg_ha']:.3f}, "
                f"R2={met['R2']:.4f}, "
                f"r={met['Pearson_r']:.3f}"
            )

    june_oof = pd.concat(june_oof_parts, ignore_index=True)
    if len(june_oof) != EXPECTED_N * 3:
        raise RuntimeError(
            f"Expected {EXPECTED_N*3:,} June OOF rows; found {len(june_oof):,}."
        )

    # Every model must have every shot exactly once.
    chk = june_oof.groupby("model")["shot_number"].agg(["count", "nunique"])
    if not ((chk["count"] == EXPECTED_N) & (chk["nunique"] == EXPECTED_N)).all():
        raise RuntimeError("June OOF completeness check failed.")

    pooled_rows = []
    for model, g in june_oof.groupby("model"):
        pooled_rows.append(metric_row(
            g["observed_agbd"], g["predicted_agbd"],
            model=model, scene="June25"
        ))
    june_pooled = pd.DataFrame(pooled_rows)

    # September OOF actual recalculation
    sep_oof = read_sep_oof()
    sep_pooled_rows = []
    for model in ["Linear_baseline_3P", "ET_3P_SENSITIVITY", "MLP_3P_SENSITIVITY"]:
        g = sep_oof[sep_oof["model"] == model].copy()
        if len(g) != EXPECTED_N:
            raise RuntimeError(
                f"September {model} OOF expected 1,641 rows; found {len(g):,}."
            )
        sep_pooled_rows.append(metric_row(
            g["observed_agbd"], g["predicted_agbd"],
            model=model, scene="Sep17"
        ))
    sep_pooled = pd.DataFrame(sep_pooled_rows)

    # Side-by-side comparison
    comp_rows = []
    for model in ["Linear_baseline_3P", "ET_3P_SENSITIVITY", "MLP_3P_SENSITIVITY"]:
        s = sep_pooled[sep_pooled["model"] == model].iloc[0]
        j = june_pooled[june_pooled["model"] == model].iloc[0]
        comp_rows.append({
            "model": model,
            "Sep17_RMSE": s["RMSE_Mg_ha"],
            "June25_RMSE": j["RMSE_Mg_ha"],
            "June_minus_Sep_RMSE": j["RMSE_Mg_ha"] - s["RMSE_Mg_ha"],
            "RMSE_percent_change": (
                (j["RMSE_Mg_ha"] - s["RMSE_Mg_ha"]) / s["RMSE_Mg_ha"] * 100.0
            ),
            "Sep17_MAE": s["MAE_Mg_ha"],
            "June25_MAE": j["MAE_Mg_ha"],
            "Sep17_R2": s["R2"],
            "June25_R2": j["R2"],
            "June_minus_Sep_R2": j["R2"] - s["R2"],
            "Sep17_Pearson_r": s["Pearson_r"],
            "June25_Pearson_r": j["Pearson_r"],
            "Sep17_predicted_SD": s["predicted_sd"],
            "June25_predicted_SD": j["predicted_sd"],
            "observed_SD": j["observed_sd"],
            "Sep17_pred_to_obs_SD_ratio": s["predicted_sd"] / s["observed_sd"],
            "June25_pred_to_obs_SD_ratio": j["predicted_sd"] / j["observed_sd"],
        })
    comp = pd.DataFrame(comp_rows)

    # Paired shot-wise error comparison between scenes.
    paired_rows = []
    for model in ["Linear_baseline_3P", "ET_3P_SENSITIVITY", "MLP_3P_SENSITIVITY"]:
        j = june_oof[june_oof["model"] == model][
            ["shot_number", "observed_agbd", "predicted_agbd"]
        ].rename(columns={"predicted_agbd": "June_pred"})
        s = sep_oof[sep_oof["model"] == model][
            ["shot_number", "observed_agbd", "predicted_agbd"]
        ].rename(columns={"predicted_agbd": "Sep_pred"})

        m = j.merge(
            s[["shot_number", "Sep_pred"]],
            on="shot_number",
            how="inner",
            validate="one_to_one"
        )
        if len(m) != EXPECTED_N:
            raise RuntimeError(f"Paired OOF merge failed for {model}.")

        m["June_abs_error"] = np.abs(m["June_pred"] - m["observed_agbd"])
        m["Sep_abs_error"] = np.abs(m["Sep_pred"] - m["observed_agbd"])
        m["June_minus_Sep_abs_error"] = m["June_abs_error"] - m["Sep_abs_error"]
        m["June_sq_error"] = (m["June_pred"] - m["observed_agbd"]) ** 2
        m["Sep_sq_error"] = (m["Sep_pred"] - m["observed_agbd"]) ** 2

        paired_rows.append({
            "model": model,
            "n": len(m),
            "mean_June_minus_Sep_abs_error": float(m["June_minus_Sep_abs_error"].mean()),
            "median_June_minus_Sep_abs_error": float(m["June_minus_Sep_abs_error"].median()),
            "fraction_shots_June_lower_abs_error": float(
                np.mean(m["June_abs_error"] < m["Sep_abs_error"])
            ),
            "mean_June_minus_Sep_squared_error": float(
                (m["June_sq_error"] - m["Sep_sq_error"]).mean()
            ),
        })

    paired = pd.DataFrame(paired_rows)

    # Broad biomass performance for June only
    cls = class_metrics(june_oof)

    # Save tables
    june_oof_path = TABLE_DIR / "June25_locked_models_OOF_predictions.csv"
    fold_path = TABLE_DIR / "June25_locked_models_fold_metrics.csv"
    pooled_path = TABLE_DIR / "June25_locked_models_pooled_metrics.csv"
    sep_path = TABLE_DIR / "Sep17_reference_pooled_metrics_recalculated.csv"
    comp_path = TABLE_DIR / "June25_vs_Sep17_locked_model_comparison.csv"
    paired_path = TABLE_DIR / "June25_vs_Sep17_paired_error_comparison.csv"
    class_path = TABLE_DIR / "June25_locked_models_performance_by_AGBD_regime.csv"
    membership_path = TABLE_DIR / "frozen_fold_membership_check.csv"

    june_oof.to_csv(june_oof_path, index=False)
    pd.DataFrame(fold_metric_rows).to_csv(fold_path, index=False)
    june_pooled.to_csv(pooled_path, index=False)
    sep_pooled.to_csv(sep_path, index=False)
    comp.to_csv(comp_path, index=False)
    paired.to_csv(paired_path, index=False)
    cls.to_csv(class_path, index=False)
    pd.DataFrame(membership_rows).to_csv(membership_path, index=False)

    plot_scene_comparison(comp)
    plot_mlp_obs_pred(june_oof)

    # Decision logic: improvement must be consistent, not one metric only.
    # This is descriptive; no formal threshold is used to "pass" a model.
    mlp = comp[comp["model"] == "MLP_3P_SENSITIVITY"].iloc[0]
    et = comp[comp["model"] == "ET_3P_SENSITIVITY"].iloc[0]
    lin = comp[comp["model"] == "Linear_baseline_3P"].iloc[0]

    report = [
        "PHASE 8B.2C — JUNE 25 LOCKED-MODEL TEMPORAL SENSITIVITY",
        "=" * 82,
        f"Run time: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "SCIENTIFIC DESIGN",
        "- Same frozen 1,641 GEDI observations.",
        "- Same five outer spatial folds.",
        "- Same 500 m buffered training membership.",
        "- Same three predictors: HH_dB, HV_dB, HV_SD.",
        "- Same September-selected fold-specific ET and MLP hyperparameters.",
        "- StandardScaler for MLP is refit using June training predictors within each fold.",
        "- No June hyperparameter tuning and no outer-test selection.",
        "- Therefore the main experimental change is NISAR acquisition date / predictor values.",
        "",
        "SEPTEMBER-SELECTED EXTRA TREES 3P SETTINGS",
        str(ET_PARAMS),
        "",
        "SEPTEMBER-SELECTED MLP 3P SETTINGS",
        str(MLP_PARAMS),
        "",
        "JUNE POOLED OUT-OF-FOLD PERFORMANCE",
        june_pooled.to_string(index=False),
        "",
        "SEPTEMBER REFERENCE POOLED PERFORMANCE RECALCULATED FROM OOF TABLES",
        sep_pooled.to_string(index=False),
        "",
        "JUNE VS SEPTEMBER COMPARISON",
        comp.to_string(index=False),
        "",
        "PAIRED SHOT-WISE ERROR COMPARISON",
        paired.to_string(index=False),
        "",
        "JUNE PERFORMANCE BY BROAD AGBD REGIME",
        cls.to_string(index=False),
        "",
        "INTERPRETATION RULES",
        "- A modest numerical change is not enough to claim temporal mismatch was the main limitation.",
        "- Evidence for a meaningful temporal-alignment benefit should be consistent across",
        "  RMSE/R2/correlation, folds, biomass regimes, and prediction dynamic range.",
        "- The June scene is only 84 days closer than the September scene in median temporal gap;",
        "  its median GEDI gap remains about 592 days.",
        "- Do not select a new final model from this sensitivity test alone.",
        "",
        "KEY COMPARISON — MLP 3P",
        f"Sep RMSE: {mlp['Sep17_RMSE']:.6f}",
        f"June RMSE: {mlp['June25_RMSE']:.6f}",
        f"RMSE change (%): {mlp['RMSE_percent_change']:.4f}",
        f"Sep R2: {mlp['Sep17_R2']:.6f}",
        f"June R2: {mlp['June25_R2']:.6f}",
        f"R2 change: {mlp['June_minus_Sep_R2']:.6f}",
        f"Sep predicted/observed SD ratio: {mlp['Sep17_pred_to_obs_SD_ratio']:.6f}",
        f"June predicted/observed SD ratio: {mlp['June25_pred_to_obs_SD_ratio']:.6f}",
        "",
        "NEXT DECISION",
        "Review whether June materially and consistently improves held-out evidence.",
        "If improvement remains small/weak, do not chase more acquisition dates; proceed to",
        "Phase 8C predictor expansion and keep temporal mismatch as a documented limitation.",
    ]

    report_path = REPORT_DIR / "JUNE25_LOCKED_MODEL_TEMPORAL_SENSITIVITY_REPORT.txt"
    report_path.write_text("\n".join(report), encoding="utf-8")

    print("\n" + "=" * 82)
    print("PHASE 8B.2C COMPLETE")
    print("=" * 82)
    print("\nJune pooled spatial-CV performance:")
    print(june_pooled[[
        "model", "RMSE_Mg_ha", "MAE_Mg_ha", "R2",
        "Bias_Mg_ha", "Pearson_r", "predicted_sd"
    ]].to_string(index=False))

    print("\nJune vs September:")
    print(comp[[
        "model", "Sep17_RMSE", "June25_RMSE", "RMSE_percent_change",
        "Sep17_R2", "June25_R2", "June_minus_Sep_R2",
        "Sep17_Pearson_r", "June25_Pearson_r"
    ]].to_string(index=False))

    print("\nCreated:")
    print(report_path)
    print(comp_path)
    print(paired_path)
    print(june_oof_path)
    print(f"Figures: {FIG_DIR}")

    print("\nIMPORTANT:")
    print("Upload the report, comparison CSV, and figures for final temporal review.")
    print("Do not produce a final biomass map yet.")
    None  # interactive pause removed for portable execution

if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n" + "=" * 82)
        print("PHASE 8B.2C FAILED")
        print("=" * 82)
        traceback.print_exc()
        None  # interactive pause removed for portable execution
        sys.exit(1)
