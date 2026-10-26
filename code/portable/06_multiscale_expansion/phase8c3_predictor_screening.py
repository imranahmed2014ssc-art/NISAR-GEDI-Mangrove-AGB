import os
from pathlib import Path
from datetime import datetime
import hashlib
import sys
import traceback
import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# PHASE 8C.3 — REDUNDANCY / DESCRIPTIVE SENSITIVITY SCREENING
# ============================================================
PROJECT = Path(os.environ.get("NISAR_PROJECT_ROOT", r"D:\NISAR_Mangrove_Paper"))

INPUT_TABLE = (
    PROJECT
    / "04_Matched_Data"
    / "Predictor_Expansion"
    / "Phase8C2"
    / "Tables"
    / "GEDI_NISAR_expanded_predictors_1641.csv"
)

EXPECTED_FROZEN_N = 1641

OUT_ROOT = (
    PROJECT
    / "05_ML_DL"
    / "Predictor_Expansion"
    / "Phase8C3_Screening"
)
TABLE_DIR = OUT_ROOT / "Tables"
FIG_DIR = OUT_ROOT / "Figures"
QC_DIR = OUT_ROOT / "QC"
for d in [TABLE_DIR, FIG_DIR, QC_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Existing 3P benchmark predictors.
BASE_3P = [
    "HH_dB",
    "HV_dB",
    "HV_SD",
]

# Fixed expanded candidates from Phase 8C.1 / 8C.2.
NEW_11 = [
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

# Primary expanded candidate space.
# HH_minus_HV is intentionally excluded because HH_minus_HV = HH_dB - HV_dB exactly.
CANDIDATES = BASE_3P + NEW_11

# HH_minus_HV is included in descriptive correlation tables only, if present.
DESCRIPTIVE_EXTRA = ["HH_minus_HV"]

HIGH_RHO = 0.85
VERY_HIGH_RHO = 0.95

# ============================================================
# HELPERS
# ============================================================
def sha256_file(path, chunk_size=8 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def numeric_complete(df, cols):
    x = df[cols].apply(pd.to_numeric, errors="coerce")
    if x.isna().any().any():
        bad = x.columns[x.isna().any()].tolist()
        raise RuntimeError(f"Missing/non-numeric values in predictors: {bad}")
    return x

def vif_table(X):
    """
    VIF-like redundancy diagnostic using each standardized predictor as response
    and all other standardized predictors as covariates.
    This uses predictor values only; AGBD is not involved.
    """
    scaler = StandardScaler()
    Z = scaler.fit_transform(X.to_numpy(float))
    rows = []
    for j, name in enumerate(X.columns):
        y = Z[:, j]
        idx = [k for k in range(Z.shape[1]) if k != j]
        reg = LinearRegression().fit(Z[:, idx], y)
        r2 = float(reg.score(Z[:, idx], y))
        if r2 >= 1.0 - 1e-12:
            vif = np.inf
        else:
            vif = 1.0 / (1.0 - r2)
        rows.append({
            "predictor": name,
            "R2_explained_by_other_predictors": r2,
            "VIF_like": vif,
        })
    return pd.DataFrame(rows).sort_values(
        ["VIF_like", "R2_explained_by_other_predictors"],
        ascending=False
    )

def high_corr_pairs(corr, method):
    rows = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = float(corr.iloc[i, j])
            ar = abs(r)
            if ar >= HIGH_RHO:
                rows.append({
                    "method": method,
                    "predictor_1": cols[i],
                    "predictor_2": cols[j],
                    "correlation": r,
                    "abs_correlation": ar,
                    "redundancy_level": (
                        "VERY_HIGH_>=0.95" if ar >= VERY_HIGH_RHO
                        else "HIGH_0.85_to_<0.95"
                    ),
                })
    return pd.DataFrame(rows).sort_values(
        "abs_correlation", ascending=False
    ) if rows else pd.DataFrame(
        columns=[
            "method", "predictor_1", "predictor_2",
            "correlation", "abs_correlation", "redundancy_level"
        ]
    )

def agbd_descriptive(df, predictors):
    """
    Descriptive only. These results MUST NOT be used to choose predictors from
    the full dataset before outer spatial CV.
    """
    y = pd.to_numeric(df["agbd"], errors="coerce").to_numpy(float)
    rows = []
    for p in predictors:
        if p not in df.columns:
            continue
        x = pd.to_numeric(df[p], errors="coerce").to_numpy(float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 3:
            continue
        pe = pearsonr(x[m], y[m])
        sp = spearmanr(x[m], y[m])
        rows.append({
            "predictor": p,
            "n": int(m.sum()),
            "Pearson_r_with_AGBD_DESCRIPTIVE_ONLY": float(pe.statistic),
            "Spearman_rho_with_AGBD_DESCRIPTIVE_ONLY": float(sp.statistic),
        })
    return pd.DataFrame(rows)

def variance_table(X):
    rows = []
    for p in X.columns:
        v = X[p].to_numpy(float)
        sd = float(np.std(v, ddof=1))
        rows.append({
            "predictor": p,
            "mean": float(np.mean(v)),
            "sd": sd,
            "cv_abs": float(abs(sd / np.mean(v))) if np.mean(v) != 0 else np.nan,
            "min": float(np.min(v)),
            "median": float(np.median(v)),
            "max": float(np.max(v)),
            "near_zero_variance": bool(sd < 1e-8),
        })
    return pd.DataFrame(rows)

def plot_corr_heatmap(corr, path_png, path_pdf, title):
    # Reorder with hierarchical clustering using 1-|rho| as distance.
    abs_corr = np.abs(corr.to_numpy(float))
    dist = 1.0 - abs_corr
    np.fill_diagonal(dist, 0.0)

    if len(corr) > 2:
        condensed = squareform(dist, checks=False)
        Z = linkage(condensed, method="average")
        order = leaves_list(Z)
    else:
        order = np.arange(len(corr))

    ordered = corr.iloc[order, order]

    fig, ax = plt.subplots(figsize=(11.5, 9.5))
    im = ax.imshow(ordered.to_numpy(float), vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(ordered.columns)))
    ax.set_yticks(np.arange(len(ordered.index)))
    ax.set_xticklabels(ordered.columns, rotation=70, ha="right", fontsize=8)
    ax.set_yticklabels(ordered.index, fontsize=8)
    ax.set_title(title, pad=12)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("Correlation")
    fig.tight_layout()
    fig.savefig(path_png, dpi=450, bbox_inches="tight")
    fig.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)

def plot_vif(vif_df):
    d = vif_df.copy()
    # Replace inf for plotting only.
    finite = d["VIF_like"].replace([np.inf, -np.inf], np.nan)
    cap = np.nanmax(finite.to_numpy()) if finite.notna().any() else 100.0
    cap = max(cap, 20.0)
    d["VIF_plot"] = d["VIF_like"].replace(np.inf, cap * 1.15)

    d = d.sort_values("VIF_plot")
    fig, ax = plt.subplots(figsize=(8.5, 6.8))
    ax.barh(d["predictor"], d["VIF_plot"])
    ax.axvline(5, linestyle="--", linewidth=1.0, label="VIF = 5")
    ax.axvline(10, linestyle=":", linewidth=1.0, label="VIF = 10")
    ax.set_xlabel("VIF-like predictor redundancy diagnostic")
    ax.set_title("Expanded predictor multicollinearity diagnostic", pad=12)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "predictor_VIF_like.png", dpi=450, bbox_inches="tight")
    fig.savefig(FIG_DIR / "predictor_VIF_like.pdf", bbox_inches="tight")
    plt.close(fig)

def plot_descriptive_agbd_corr(desc):
    d = desc.copy()
    d["abs_spearman"] = d[
        "Spearman_rho_with_AGBD_DESCRIPTIVE_ONLY"
    ].abs()
    d = d.sort_values("abs_spearman")

    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    ax.barh(
        d["predictor"],
        d["Spearman_rho_with_AGBD_DESCRIPTIVE_ONLY"]
    )
    ax.axvline(0, linewidth=1.0)
    ax.set_xlabel("Spearman ρ with AGBD (descriptive only)")
    ax.set_title(
        "Predictor–AGBD associations\nNOT used for full-dataset feature selection",
        pad=12
    )
    fig.tight_layout()
    fig.savefig(
        FIG_DIR / "descriptive_predictor_AGBD_spearman.png",
        dpi=450, bbox_inches="tight"
    )
    fig.savefig(
        FIG_DIR / "descriptive_predictor_AGBD_spearman.pdf",
        bbox_inches="tight"
    )
    plt.close(fig)

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 86)
    print("PHASE 8C.3 — EXPANDED PREDICTOR REDUNDANCY / SCREENING DIAGNOSTICS")
    print("=" * 86)
    print("No model fitting. No full-dataset response-based feature selection.\n")

    if not INPUT_TABLE.exists():
        raise FileNotFoundError(INPUT_TABLE)

    df = pd.read_csv(INPUT_TABLE)
    print(f"Expanded table records: {len(df):,}")
    print("Input SHA256:", sha256_file(INPUT_TABLE))

    if len(df) != EXPECTED_FROZEN_N:
        raise RuntimeError(
            f"Expected {EXPECTED_FROZEN_N:,} frozen records; found {len(df):,}"
        )

    required = {"agbd"} | set(CANDIDATES)
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    X = numeric_complete(df, CANDIDATES)

    # Exact identity check for original algebraic contrast if available.
    algebraic_check = None
    if "HH_minus_HV" in df.columns:
        hh = pd.to_numeric(df["HH_dB"], errors="coerce").to_numpy(float)
        hv = pd.to_numeric(df["HV_dB"], errors="coerce").to_numpy(float)
        diff = pd.to_numeric(df["HH_minus_HV"], errors="coerce").to_numpy(float)
        max_abs = float(np.nanmax(np.abs(diff - (hh - hv))))
        algebraic_check = max_abs

    pear = X.corr(method="pearson")
    spear = X.corr(method="spearman")

    pear_pairs = high_corr_pairs(pear, "Pearson")
    spear_pairs = high_corr_pairs(spear, "Spearman")
    pairs = pd.concat([pear_pairs, spear_pairs], ignore_index=True)

    vif = vif_table(X)
    var = variance_table(X)

    desc_predictors = CANDIDATES + [
        p for p in DESCRIPTIVE_EXTRA if p in df.columns
    ]
    desc = agbd_descriptive(df, desc_predictors)

    # Condition number after standardization, predictor-only diagnostic.
    Z = StandardScaler().fit_transform(X.to_numpy(float))
    cond = float(np.linalg.cond(Z))

    # Save tables
    pear_path = TABLE_DIR / "predictor_Pearson_correlation_matrix.csv"
    spear_path = TABLE_DIR / "predictor_Spearman_correlation_matrix.csv"
    pairs_path = TABLE_DIR / "high_redundancy_predictor_pairs.csv"
    vif_path = TABLE_DIR / "predictor_VIF_like.csv"
    var_path = TABLE_DIR / "predictor_variance_summary.csv"
    desc_path = TABLE_DIR / "predictor_AGBD_correlations_DESCRIPTIVE_ONLY.csv"

    pear.to_csv(pear_path)
    spear.to_csv(spear_path)
    pairs.to_csv(pairs_path, index=False)
    vif.to_csv(vif_path, index=False)
    var.to_csv(var_path, index=False)
    desc.to_csv(desc_path, index=False)

    # Figures
    plot_corr_heatmap(
        spear,
        FIG_DIR / "predictor_Spearman_heatmap.png",
        FIG_DIR / "predictor_Spearman_heatmap.pdf",
        "Expanded NISAR predictor redundancy — Spearman correlation"
    )
    plot_vif(vif)
    plot_descriptive_agbd_corr(desc)

    very_high_spear = spear_pairs[
        spear_pairs["abs_correlation"] >= VERY_HIGH_RHO
    ].copy()

    report = [
        "PHASE 8C.3 — EXPANDED PREDICTOR REDUNDANCY / SCREENING DIAGNOSTICS",
        "=" * 86,
        f"Run time: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"Input table: {INPUT_TABLE}",
        f"Input SHA256: {sha256_file(INPUT_TABLE)}",
        f"Records: {len(df):,}",
        "",
        "PRIMARY CANDIDATE SPACE",
        ", ".join(CANDIDATES),
        "",
        "IMPORTANT EXCLUSION",
        "HH_minus_HV is not part of the primary expanded candidate space because",
        "HH_minus_HV = HH_dB - HV_dB exactly. It is retained only for descriptive audit.",
        f"Maximum absolute algebraic check error: {algebraic_check}",
        "",
        "SCREENING PRINCIPLE",
        "- Predictor-vs-predictor redundancy may be assessed on all 1,641 records because",
        "  no AGBD response is used.",
        "- Predictor-vs-AGBD correlations below are DESCRIPTIVE ONLY.",
        "- They must NOT be used to cherry-pick a final feature subset before outer CV.",
        "- Any response-based feature selection/tuning must occur inside buffered training",
        "  data within the spatial cross-validation framework.",
        "",
        f"Standardized predictor matrix condition number: {cond:.6f}",
        "",
        "VERY HIGH SPEARMAN REDUNDANCY PAIRS (|rho| >= 0.95)",
        (
            very_high_spear.to_string(index=False)
            if len(very_high_spear)
            else "None"
        ),
        "",
        "ALL HIGH REDUNDANCY PAIRS (|r/rho| >= 0.85)",
        (
            pairs.to_string(index=False)
            if len(pairs)
            else "None"
        ),
        "",
        "VIF-LIKE REDUNDANCY DIAGNOSTIC",
        vif.to_string(index=False),
        "",
        "DESCRIPTIVE PREDICTOR–AGBD CORRELATIONS — DO NOT USE FOR GLOBAL SELECTION",
        desc.to_string(index=False),
        "",
        "DECISION RULE FOR PHASE 8C.4",
        "Do not automatically drop variables based on AGBD correlation from this full table.",
        "First review very-high predictor-only redundancy. Then define a compact expanded",
        "feature set using physical interpretation + predictor-only redundancy, or place",
        "any response-based selection strictly inside the training folds.",
        "",
        "PHASE 8C.3 STATUS: COMPLETE — REVIEW REQUIRED",
    ]

    report_path = QC_DIR / "PHASE8C3_PREDICTOR_SCREENING_REPORT.txt"
    report_path.write_text("\n".join(report), encoding="utf-8")

    print("\n" + "=" * 86)
    print("PHASE 8C.3 COMPLETE — REVIEW REQUIRED")
    print("=" * 86)

    print("\nVery-high Spearman redundancy pairs (|rho| >= 0.95):")
    if len(very_high_spear):
        print(very_high_spear.to_string(index=False))
    else:
        print("None")

    print("\nVIF-like redundancy diagnostic:")
    print(vif.to_string(index=False))

    print("\nDescriptive predictor–AGBD Spearman correlations:")
    print(
        desc[[
            "predictor",
            "Spearman_rho_with_AGBD_DESCRIPTIVE_ONLY"
        ]].sort_values(
            "Spearman_rho_with_AGBD_DESCRIPTIVE_ONLY",
            key=lambda s: s.abs(),
            ascending=False
        ).to_string(index=False)
    )

    print("\nCreated:")
    print(report_path)
    print(pairs_path)
    print(vif_path)
    print(desc_path)
    print(f"Figures: {FIG_DIR}")

    print("\nIMPORTANT:")
    print("Do NOT fit the Phase 8C.4 models yet.")
    print("Upload the screening report, high-redundancy table, and VIF table for review.")
    None  # interactive pause removed for portable execution

if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n" + "=" * 86)
        print("PHASE 8C.3 FAILED")
        print("=" * 86)
        traceback.print_exc()
        None  # interactive pause removed for portable execution
        sys.exit(1)
