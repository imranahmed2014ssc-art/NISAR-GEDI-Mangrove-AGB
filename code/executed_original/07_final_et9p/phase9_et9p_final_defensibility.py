from pathlib import Path
from datetime import datetime
import hashlib
import json
import math
import os
import sys
import traceback
import warnings

# Low-memory / reproducible Windows execution
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from pyproj import Transformer

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# PHASE 9 — FINAL DEFENSIBILITY OF EXPANDED EXTRA TREES 9P
# ============================================================
PROJECT = Path(r"D:\NISAR_Mangrove_Paper")

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

NEW_OOF = (
    PROJECT
    / "06_Results"
    / "Predictor_Expansion"
    / "Phase8C4_Expanded9P"
    / "Tables"
    / "expanded9P_OOF_predictions.csv"
)

SELECTED_PARAMS = (
    PROJECT
    / "06_Results"
    / "Predictor_Expansion"
    / "Phase8C4_Expanded9P"
    / "Tables"
    / "expanded9P_selected_hyperparameters.csv"
)

OLD_ET_OOF = (
    PROJECT
    / "06_Results"
    / "Extra_Trees"
    / "Tables"
    / "ET_OOF_predictions.csv"
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
    / "Phase9_ET9P_Final_Defensibility"
)
TABLE_DIR = OUT_ROOT / "Tables"
FIG_DIR = OUT_ROOT / "Figures"
QC_DIR = OUT_ROOT / "QC"
for d in [TABLE_DIR, FIG_DIR, QC_DIR]:
    d.mkdir(parents=True, exist_ok=True)

EXPECTED_N = 1641
RANDOM_SEED = 42
BOOTSTRAP_REPS = 5000
PERMUTATION_REPEATS = 50
MORAN_PERMUTATIONS = 499
MORAN_DISTANCES_M = [500, 1000, 2000, 3000]

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

# ============================================================
# HELPERS
# ============================================================
def sha256_file(path, chunk_size=8 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))

def mae(y, p):
    return float(mean_absolute_error(y, p))

def r2(y, p):
    return float(r2_score(y, p))

def detect_xy(df):
    projected_pairs = [
        ("utm_x", "utm_y"),
        ("easting", "northing"),
        ("utm_easting", "utm_northing"),
        ("x_utm", "y_utm"),
        ("utm_x_m", "utm_y_m"),
    ]
    for xc, yc in projected_pairs:
        if xc in df.columns and yc in df.columns:
            return (
                pd.to_numeric(df[xc], errors="raise").to_numpy(float),
                pd.to_numeric(df[yc], errors="raise").to_numpy(float),
                f"{xc}/{yc}",
            )

    lon_candidates = ["longitude", "lon", "gedi_lon"]
    lat_candidates = ["latitude", "lat", "gedi_lat"]
    lon = next((c for c in lon_candidates if c in df.columns), None)
    lat = next((c for c in lat_candidates if c in df.columns), None)
    if lon and lat:
        tr = Transformer.from_crs(4326, 32645, always_xy=True)
        xx, yy = tr.transform(
            pd.to_numeric(df[lon], errors="raise").to_numpy(float),
            pd.to_numeric(df[lat], errors="raise").to_numpy(float),
        )
        return np.asarray(xx), np.asarray(yy), f"{lon}/{lat}->EPSG:32645"

    raise RuntimeError("Could not detect coordinates for residual Moran's I.")

def load_old_et():
    if not OLD_ET_OOF.exists():
        raise FileNotFoundError(OLD_ET_OOF)
    d = pd.read_csv(OLD_ET_OOF)

    required = {"shot_number", "observed_agbd", "predicted_agbd"}
    if not required.issubset(d.columns):
        raise RuntimeError(
            f"Old ET OOF missing columns: {sorted(required - set(d.columns))}"
        )

    if "model" in d.columns:
        if (d["model"] == "ET_3P_SENSITIVITY").any():
            d = d[d["model"] == "ET_3P_SENSITIVITY"].copy()
        elif (d["model"] == "ET_3P").any():
            d = d[d["model"] == "ET_3P"].copy()
        else:
            raise RuntimeError(
                "Could not identify ET 3P rows in old ET OOF 'model' column."
            )

    d = d[["shot_number", "observed_agbd", "predicted_agbd"]].copy()
    d = d.rename(columns={
        "observed_agbd": "observed_agbd_old",
        "predicted_agbd": "predicted_ET3P",
    })
    d["shot_number"] = pd.to_numeric(
        d["shot_number"], errors="raise"
    ).astype("int64")

    if len(d) != EXPECTED_N or d["shot_number"].nunique() != EXPECTED_N:
        raise RuntimeError(
            f"Old ET 3P OOF must contain {EXPECTED_N} unique shots; found {len(d)}."
        )
    return d

def load_new_et():
    if not NEW_OOF.exists():
        raise FileNotFoundError(NEW_OOF)
    d = pd.read_csv(NEW_OOF)

    required = {"shot_number", "outer_fold", "observed_agbd", "predicted_agbd", "model"}
    if not required.issubset(d.columns):
        raise RuntimeError(
            f"New OOF missing columns: {sorted(required - set(d.columns))}"
        )

    d = d[d["model"] == "EXPANDED_ET_9P"].copy()
    d = d[[
        "shot_number", "outer_fold", "observed_agbd", "predicted_agbd"
    ]].copy()
    d = d.rename(columns={
        "observed_agbd": "observed_agbd_new",
        "predicted_agbd": "predicted_ET9P",
    })
    d["shot_number"] = pd.to_numeric(
        d["shot_number"], errors="raise"
    ).astype("int64")

    if len(d) != EXPECTED_N or d["shot_number"].nunique() != EXPECTED_N:
        raise RuntimeError(
            f"New ET 9P OOF must contain {EXPECTED_N} unique shots; found {len(d)}."
        )
    return d

def paired_block_bootstrap(merged):
    """
    Resample occupied 3-km cv_block_id blocks with replacement.
    Every time a block is sampled, all shots in that block are included.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    blocks = pd.unique(merged["cv_block_id"])
    block_parts = {
        b: merged[merged["cv_block_id"] == b].copy()
        for b in blocks
    }

    rows = []
    for rep in range(BOOTSTRAP_REPS):
        sampled = rng.choice(blocks, size=len(blocks), replace=True)
        y_parts = []
        old_parts = []
        new_parts = []

        for b in sampled:
            g = block_parts[b]
            y_parts.append(g["observed_agbd"].to_numpy(float))
            old_parts.append(g["predicted_ET3P"].to_numpy(float))
            new_parts.append(g["predicted_ET9P"].to_numpy(float))

        y = np.concatenate(y_parts)
        oldp = np.concatenate(old_parts)
        newp = np.concatenate(new_parts)

        old_rmse = rmse(y, oldp)
        new_rmse = rmse(y, newp)
        old_mae = mae(y, oldp)
        new_mae = mae(y, newp)
        old_r2 = r2(y, oldp)
        new_r2 = r2(y, newp)

        rows.append({
            "rep": rep + 1,
            "old_RMSE": old_rmse,
            "new_RMSE": new_rmse,
            "delta_RMSE_new_minus_old": new_rmse - old_rmse,
            "old_MAE": old_mae,
            "new_MAE": new_mae,
            "delta_MAE_new_minus_old": new_mae - old_mae,
            "old_R2": old_r2,
            "new_R2": new_r2,
            "delta_R2_new_minus_old": new_r2 - old_r2,
        })

    boot = pd.DataFrame(rows)

    def summarize(col):
        v = boot[col].to_numpy(float)
        return {
            "metric": col,
            "mean": float(np.mean(v)),
            "median": float(np.median(v)),
            "CI2.5": float(np.percentile(v, 2.5)),
            "CI97.5": float(np.percentile(v, 97.5)),
            "fraction_below_zero": float(np.mean(v < 0)),
            "fraction_above_zero": float(np.mean(v > 0)),
        }

    summary = pd.DataFrame([
        summarize("delta_RMSE_new_minus_old"),
        summarize("delta_MAE_new_minus_old"),
        summarize("delta_R2_new_minus_old"),
    ])
    return boot, summary

def fold_comparison(merged):
    rows = []
    for fold, g in merged.groupby("outer_fold"):
        y = g["observed_agbd"].to_numpy(float)
        oldp = g["predicted_ET3P"].to_numpy(float)
        newp = g["predicted_ET9P"].to_numpy(float)

        old_rmse = rmse(y, oldp)
        new_rmse = rmse(y, newp)
        old_mae = mae(y, oldp)
        new_mae = mae(y, newp)

        rows.append({
            "outer_fold": int(fold),
            "n": len(g),
            "old_RMSE": old_rmse,
            "new_RMSE": new_rmse,
            "delta_RMSE_new_minus_old": new_rmse - old_rmse,
            "old_MAE": old_mae,
            "new_MAE": new_mae,
            "delta_MAE_new_minus_old": new_mae - old_mae,
            "old_R2": r2(y, oldp),
            "new_R2": r2(y, newp),
        })
    return pd.DataFrame(rows).sort_values("outer_fold")

def parse_params_table():
    if not SELECTED_PARAMS.exists():
        raise FileNotFoundError(SELECTED_PARAMS)
    p = pd.read_csv(SELECTED_PARAMS)
    p = p[p["model"] == "EXPANDED_ET_9P"].copy()
    if len(p) != 5:
        raise RuntimeError(
            f"Expected 5 selected ET9P parameter rows; found {len(p)}."
        )

    out = {}
    for _, r in p.iterrows():
        fold = int(r["outer_fold"])
        params = json.loads(r["best_params_json"])
        out[fold] = params
    return out

def load_fold_ids(fold):
    fd = BUFFER_ROOT / f"Fold_{fold}"
    train_path = fd / f"Fold_{fold}_TRAIN_BUFFERED_500m.csv"
    test_path = fd / f"Fold_{fold}_TEST.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Missing buffered fold files for fold {fold}."
        )

    tr = pd.read_csv(train_path)
    te = pd.read_csv(test_path)
    for d, name in [(tr, "train"), (te, "test")]:
        if "shot_number" not in d.columns:
            raise RuntimeError(f"shot_number missing from fold {fold} {name} file.")

    train_ids = set(
        pd.to_numeric(tr["shot_number"], errors="raise").astype("int64")
    )
    test_ids = set(
        pd.to_numeric(te["shot_number"], errors="raise").astype("int64")
    )
    return train_ids, test_ids

def heldout_permutation_importance(df, saved_oof):
    """
    Refit the already-selected ET9P model in each buffered training fold using
    the already-selected fold-specific hyperparameters, then permute one
    predictor at a time only in the untouched outer test fold.

    This is post-hoc diagnostic interpretation. It is not used to retune models.
    """
    params_by_fold = parse_params_table()
    rng = np.random.default_rng(RANDOM_SEED + 9000)

    rows = []
    reproduction_rows = []

    for fold in range(1, 6):
        print(f"  Permutation importance — fold {fold}/5")
        train_ids, test_ids = load_fold_ids(fold)
        tr = df[df["shot_number"].isin(train_ids)].copy()
        te = df[df["shot_number"].isin(test_ids)].copy()

        Xtr = tr[FEATURES].to_numpy(float)
        ytr = tr["agbd"].to_numpy(float)
        Xte = te[FEATURES].to_numpy(float)
        yte = te["agbd"].to_numpy(float)

        params = params_by_fold[fold]
        model = ExtraTreesRegressor(
            **params,
            random_state=RANDOM_SEED,
            n_jobs=1,
        )
        model.fit(Xtr, ytr)
        baseline_pred = model.predict(Xte)
        baseline_rmse = rmse(yte, baseline_pred)

        # Reproduction audit against saved Phase 8C.4 ET9P OOF
        saved = saved_oof[saved_oof["outer_fold"] == fold].copy()
        lookup = dict(
            zip(
                saved["shot_number"].astype("int64"),
                saved["predicted_ET9P"].astype(float),
            )
        )
        saved_pred = np.array([lookup[int(s)] for s in te["shot_number"]], float)
        max_abs_diff = float(np.max(np.abs(saved_pred - baseline_pred)))

        reproduction_rows.append({
            "outer_fold": fold,
            "train_n": len(tr),
            "test_n": len(te),
            "baseline_RMSE_recomputed": baseline_rmse,
            "max_abs_prediction_diff_vs_saved_OOF": max_abs_diff,
        })

        for j, feat in enumerate(FEATURES):
            deltas = []
            for rep in range(PERMUTATION_REPEATS):
                Xperm = Xte.copy()
                perm_idx = rng.permutation(len(Xperm))
                Xperm[:, j] = Xperm[perm_idx, j]
                pred = model.predict(Xperm)
                deltas.append(rmse(yte, pred) - baseline_rmse)

            rows.append({
                "outer_fold": fold,
                "predictor": feat,
                "baseline_RMSE": baseline_rmse,
                "mean_delta_RMSE_when_permuted": float(np.mean(deltas)),
                "median_delta_RMSE_when_permuted": float(np.median(deltas)),
                "sd_delta_RMSE_when_permuted": float(np.std(deltas, ddof=1)),
                "p05_delta_RMSE": float(np.percentile(deltas, 5)),
                "p95_delta_RMSE": float(np.percentile(deltas, 95)),
                "n_repeats": PERMUTATION_REPEATS,
            })

    fold_imp = pd.DataFrame(rows)
    repro = pd.DataFrame(reproduction_rows)

    agg = (
        fold_imp.groupby("predictor")
        .agg(
            mean_delta_RMSE=("mean_delta_RMSE_when_permuted", "mean"),
            median_fold_delta_RMSE=("mean_delta_RMSE_when_permuted", "median"),
            min_fold_delta_RMSE=("mean_delta_RMSE_when_permuted", "min"),
            max_fold_delta_RMSE=("mean_delta_RMSE_when_permuted", "max"),
            folds_positive=("mean_delta_RMSE_when_permuted", lambda s: int((s > 0).sum())),
        )
        .reset_index()
        .sort_values("mean_delta_RMSE", ascending=False)
    )

    return fold_imp, agg, repro

def morans_i_binary_pairs(values, pairs):
    """
    Symmetric binary weights represented by undirected pairs.
    Equivalent S0 = 2 * number_of_pairs.
    """
    z = np.asarray(values, float)
    z = z - np.mean(z)
    denom = float(np.sum(z * z))
    if denom <= 0 or len(pairs) == 0:
        return np.nan

    num_undirected = float(np.sum(z[pairs[:, 0]] * z[pairs[:, 1]]))
    n = len(z)
    # numerator and S0 both doubled for symmetric weights, so doubles cancel.
    return float((n / len(pairs)) * (num_undirected / denom))

def residual_moran(merged):
    x, y, coord_source = detect_xy(merged)
    coords = np.column_stack([x, y])
    tree = cKDTree(coords)

    residual = (
        merged["observed_agbd"].to_numpy(float)
        - merged["predicted_ET9P"].to_numpy(float)
    )

    rng = np.random.default_rng(RANDOM_SEED + 777)
    rows = []

    for dist in MORAN_DISTANCES_M:
        pairs = tree.query_pairs(r=dist, output_type="ndarray")
        obs_i = morans_i_binary_pairs(residual, pairs)

        perm_i = np.empty(MORAN_PERMUTATIONS, dtype=float)
        for k in range(MORAN_PERMUTATIONS):
            perm_i[k] = morans_i_binary_pairs(
                rng.permutation(residual), pairs
            )

        # Two-sided permutation p-value around zero.
        p_two = (
            1.0 + np.sum(np.abs(perm_i) >= abs(obs_i))
        ) / (MORAN_PERMUTATIONS + 1.0)

        rows.append({
            "distance_m": dist,
            "n_pairs": int(len(pairs)),
            "Morans_I_residual": obs_i,
            "permutation_p_two_sided": float(p_two),
            "perm_mean_I": float(np.nanmean(perm_i)),
            "perm_sd_I": float(np.nanstd(perm_i, ddof=1)),
            "coordinate_source": coord_source,
            "n_permutations": MORAN_PERMUTATIONS,
        })

    return pd.DataFrame(rows)

# ============================================================
# FIGURES
# ============================================================
def figure_bootstrap(boot):
    vals = boot["delta_RMSE_new_minus_old"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    ax.hist(vals, bins=45)
    ax.axvline(0, linestyle="--", linewidth=1.1, label="No RMSE change")
    ax.axvline(np.median(vals), linestyle=":", linewidth=1.2, label="Bootstrap median")
    ax.set_xlabel("ΔRMSE = ET9P − ET3P (Mg/ha)")
    ax.set_ylabel("Spatial-block bootstrap frequency")
    ax.set_title("Paired 3-km block bootstrap of ET predictor expansion", pad=12)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR/"ET9P_vs_ET3P_block_bootstrap_delta_RMSE.png",
                dpi=450, bbox_inches="tight")
    fig.savefig(FIG_DIR/"ET9P_vs_ET3P_block_bootstrap_delta_RMSE.pdf",
                bbox_inches="tight")
    plt.close(fig)

def figure_fold_comparison(folds):
    fig, ax = plt.subplots(figsize=(7.6, 5.1))
    x = folds["outer_fold"].to_numpy()
    ax.plot(x, folds["old_RMSE"], marker="o", label="ET 3P")
    ax.plot(x, folds["new_RMSE"], marker="o", label="ET 9P")
    ax.set_xticks(x)
    ax.set_xlabel("Outer spatial fold")
    ax.set_ylabel("RMSE (Mg/ha)")
    ax.set_title("Fold-by-fold held-out RMSE", pad=12)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR/"ET9P_vs_ET3P_fold_RMSE.png",
                dpi=450, bbox_inches="tight")
    fig.savefig(FIG_DIR/"ET9P_vs_ET3P_fold_RMSE.pdf",
                bbox_inches="tight")
    plt.close(fig)

def figure_importance(agg):
    d = agg.sort_values("mean_delta_RMSE", ascending=True)
    fig, ax = plt.subplots(figsize=(8.5, 6.1))
    ax.barh(d["predictor"], d["mean_delta_RMSE"])
    ax.axvline(0, linewidth=1.0)
    ax.set_xlabel("Mean increase in held-out RMSE after permutation (Mg/ha)")
    ax.set_title("ET9P held-out permutation importance", pad=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR/"ET9P_heldout_permutation_importance.png",
                dpi=450, bbox_inches="tight")
    fig.savefig(FIG_DIR/"ET9P_heldout_permutation_importance.pdf",
                bbox_inches="tight")
    plt.close(fig)

def figure_moran(moran):
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(
        moran["distance_m"] / 1000.0,
        moran["Morans_I_residual"],
        marker="o"
    )
    ax.axhline(0, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Distance threshold (km)")
    ax.set_ylabel("Moran's I of ET9P OOF residuals")
    ax.set_title("Residual spatial autocorrelation", pad=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR/"ET9P_residual_MoransI.png",
                dpi=450, bbox_inches="tight")
    fig.savefig(FIG_DIR/"ET9P_residual_MoransI.pdf",
                bbox_inches="tight")
    plt.close(fig)

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 92)
    print("PHASE 9 — FINAL DEFENSIBILITY OF EXPANDED EXTRA TREES 9P")
    print("=" * 92)
    print("No new algorithm search. No new feature search. No final map is produced.")
    print()

    for p in [EXPANDED_TABLE, NEW_OOF, SELECTED_PARAMS, OLD_ET_OOF]:
        if not p.exists():
            raise FileNotFoundError(p)

    table_hash = sha256_file(EXPANDED_TABLE)
    print("Expanded table SHA256:", table_hash)
    if table_hash != EXPECTED_EXPANDED_SHA256:
        raise RuntimeError(
            "Expanded table hash mismatch.\n"
            f"Expected: {EXPECTED_EXPANDED_SHA256}\n"
            f"Found:    {table_hash}"
        )

    df = pd.read_csv(EXPANDED_TABLE)
    if len(df) != EXPECTED_N:
        raise RuntimeError(f"Expected {EXPECTED_N} rows; found {len(df)}.")

    required = {"shot_number", "agbd", "cv_block_id"} | set(FEATURES)
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Expanded table missing columns: {missing}")

    df["shot_number"] = pd.to_numeric(
        df["shot_number"], errors="raise"
    ).astype("int64")

    old = load_old_et()
    new = load_new_et()

    merged = (
        df[["shot_number", "agbd", "cv_block_id"]].copy()
        .merge(old, on="shot_number", how="inner", validate="one_to_one")
        .merge(new, on="shot_number", how="inner", validate="one_to_one")
    )

    if len(merged) != EXPECTED_N:
        raise RuntimeError("Old/new OOF merge did not preserve all 1,641 shots.")

    merged = merged.rename(columns={"agbd": "observed_agbd"})
    obs_diff_old = np.max(np.abs(
        merged["observed_agbd"].to_numpy(float)
        - merged["observed_agbd_old"].to_numpy(float)
    ))
    obs_diff_new = np.max(np.abs(
        merged["observed_agbd"].to_numpy(float)
        - merged["observed_agbd_new"].to_numpy(float)
    ))
    if obs_diff_old > 1e-6 or obs_diff_new > 1e-6:
        raise RuntimeError("Observed AGBD mismatch between frozen table and OOF files.")

    merged = merged.drop(columns=["observed_agbd_old", "observed_agbd_new"])

    print(f"Matched old/new OOF shots: {len(merged):,}")
    print(f"Occupied 3-km blocks: {merged['cv_block_id'].nunique()}")

    # ------------------------------------------------------------------
    # 1) Paired spatial-block bootstrap
    # ------------------------------------------------------------------
    print("\n1/4 Paired 3-km block bootstrap...")
    boot, boot_summary = paired_block_bootstrap(merged)
    boot.to_csv(TABLE_DIR/"ET9P_vs_ET3P_block_bootstrap_5000.csv", index=False)
    boot_summary.to_csv(
        TABLE_DIR/"ET9P_vs_ET3P_block_bootstrap_summary.csv", index=False
    )

    # ------------------------------------------------------------------
    # 2) Fold-level consistency
    # ------------------------------------------------------------------
    print("2/4 Fold-by-fold consistency...")
    folds = fold_comparison(merged)
    folds.to_csv(TABLE_DIR/"ET9P_vs_ET3P_fold_comparison.csv", index=False)

    # ------------------------------------------------------------------
    # 3) Held-out permutation importance
    # ------------------------------------------------------------------
    print("3/4 Held-out ET9P permutation importance...")
    # Supply coordinates to the merged table for residual Moran later.
    # We keep a copy of all original df columns for training.
    fold_imp, agg_imp, repro = heldout_permutation_importance(df, new)
    fold_imp.to_csv(
        TABLE_DIR/"ET9P_permutation_importance_by_fold.csv", index=False
    )
    agg_imp.to_csv(
        TABLE_DIR/"ET9P_permutation_importance_aggregated.csv", index=False
    )
    repro.to_csv(
        QC_DIR/"ET9P_reproduction_audit.csv", index=False
    )

    # ------------------------------------------------------------------
    # 4) Residual Moran's I
    # ------------------------------------------------------------------
    print("4/4 Residual Moran's I...")
    coord_cols = [
        c for c in df.columns
        if c.lower() in {
            "utm_x","utm_y","easting","northing","utm_easting","utm_northing",
            "x_utm","y_utm","utm_x_m","utm_y_m",
            "longitude","latitude","lon","lat","gedi_lon","gedi_lat"
        }
    ]
    moran_input = merged.merge(
        df[["shot_number"] + coord_cols],
        on="shot_number", how="left", validate="one_to_one"
    )
    moran = residual_moran(moran_input)
    moran.to_csv(TABLE_DIR/"ET9P_residual_MoransI.csv", index=False)

    # ------------------------------------------------------------------
    # Core metrics + automatic interpretation flags
    # ------------------------------------------------------------------
    y = merged["observed_agbd"].to_numpy(float)
    oldp = merged["predicted_ET3P"].to_numpy(float)
    newp = merged["predicted_ET9P"].to_numpy(float)

    core = pd.DataFrame([
        {
            "model": "ET_3P",
            "RMSE": rmse(y, oldp),
            "MAE": mae(y, oldp),
            "R2": r2(y, oldp),
            "predicted_SD": float(np.std(oldp, ddof=1)),
            "observed_SD": float(np.std(y, ddof=1)),
            "pred_to_obs_SD_ratio": float(
                np.std(oldp, ddof=1) / np.std(y, ddof=1)
            ),
        },
        {
            "model": "ET_9P",
            "RMSE": rmse(y, newp),
            "MAE": mae(y, newp),
            "R2": r2(y, newp),
            "predicted_SD": float(np.std(newp, ddof=1)),
            "observed_SD": float(np.std(y, ddof=1)),
            "pred_to_obs_SD_ratio": float(
                np.std(newp, ddof=1) / np.std(y, ddof=1)
            ),
        },
    ])
    core.to_csv(TABLE_DIR/"ET3P_ET9P_core_metrics.csv", index=False)

    rmse_summary = boot_summary[
        boot_summary["metric"] == "delta_RMSE_new_minus_old"
    ].iloc[0]
    mae_summary = boot_summary[
        boot_summary["metric"] == "delta_MAE_new_minus_old"
    ].iloc[0]

    robust_rmse_improvement = bool(rmse_summary["CI97.5"] < 0)
    robust_mae_improvement = bool(mae_summary["CI97.5"] < 0)
    folds_rmse_better = int(
        (folds["delta_RMSE_new_minus_old"] < 0).sum()
    )

    max_repro_diff = float(
        repro["max_abs_prediction_diff_vs_saved_OOF"].max()
    )

    # Important: this does NOT authorize mapping.
    if robust_rmse_improvement and robust_mae_improvement:
        improvement_label = (
            "ROBUST IMPROVEMENT SUPPORTED BY 3-KM BLOCK BOOTSTRAP"
        )
    else:
        improvement_label = (
            "NUMERICAL IMPROVEMENT OBSERVED, BUT BLOCK-BOOTSTRAP "
            "ROBUSTNESS IS NOT FULLY SUPPORTED"
        )

    # Figures
    figure_bootstrap(boot)
    figure_fold_comparison(folds)
    figure_importance(agg_imp)
    figure_moran(moran)

    # Report
    report = [
        "PHASE 9 — FINAL DEFENSIBILITY OF EXPANDED EXTRA TREES 9P",
        "=" * 92,
        f"Run time: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"Expanded table SHA256: {table_hash}",
        f"Records: {len(merged):,}",
        f"Occupied 3-km blocks: {merged['cv_block_id'].nunique()}",
        "",
        "PURPOSE",
        "Assess whether the ET9P improvement over ET3P is spatially robust and",
        "scientifically interpretable without introducing a new algorithm or feature search.",
        "",
        "LOCKED DESIGN",
        "- Same frozen 1,641 GEDI records.",
        "- Same 3-km cv_block_id units.",
        "- Same five outer spatial folds.",
        "- Same 500-m training exclusion buffer.",
        "- No new model-family comparison.",
        "- No new feature selection.",
        "",
        "CORE OOF PERFORMANCE",
        core.to_string(index=False),
        "",
        "PAIRED 3-KM BLOCK BOOTSTRAP (5,000 replicates)",
        "Delta is defined as ET9P minus ET3P; negative RMSE/MAE means ET9P is better.",
        boot_summary.to_string(index=False),
        "",
        f"RMSE improvement 95% CI entirely below zero: {robust_rmse_improvement}",
        f"MAE improvement 95% CI entirely below zero: {robust_mae_improvement}",
        f"Interpretation: {improvement_label}",
        "",
        "FOLD-BY-FOLD CONSISTENCY",
        folds.to_string(index=False),
        f"Outer folds with lower RMSE for ET9P: {folds_rmse_better}/5",
        "",
        "HELD-OUT PERMUTATION IMPORTANCE",
        "Positive delta RMSE means model performance worsened when that predictor was permuted.",
        "These are predictive contributions, not ecological causal effects.",
        "Correlated predictors can share or dilute permutation importance.",
        agg_imp.to_string(index=False),
        "",
        "ET9P REPRODUCTION AUDIT",
        repro.to_string(index=False),
        f"Maximum absolute prediction difference vs saved OOF: {max_repro_diff}",
        "",
        "OUT-OF-FOLD RESIDUAL MORAN'S I",
        moran.to_string(index=False),
        "",
        "FINAL INTERPRETATION RULE",
        "A robust ET9P improvement can be reported if the paired 3-km block bootstrap",
        "supports lower held-out RMSE/MAE. This does NOT automatically make the model",
        "suitable for operational AGBD mapping. Mapping still requires review of the",
        "absolute held-out R2, residual biomass-regime bias, prediction compression,",
        "and residual spatial structure.",
        "",
        "STATUS",
        "PHASE 9 COMPLETE — FINAL SCIENTIFIC REVIEW REQUIRED",
        "FINAL AGBD MAP REMAINS ON HOLD.",
    ]

    report_path = QC_DIR/"PHASE9_ET9P_FINAL_DEFENSIBILITY_REPORT.txt"
    report_path.write_text("\n".join(report), encoding="utf-8")

    print("\n" + "=" * 92)
    print("PHASE 9 COMPLETE — FINAL SCIENTIFIC REVIEW REQUIRED")
    print("=" * 92)

    print("\nCore OOF performance:")
    print(core.to_string(index=False))

    print("\nPaired 3-km block-bootstrap summary:")
    print(boot_summary.to_string(index=False))

    print("\nFold-by-fold ET3P vs ET9P:")
    print(folds.to_string(index=False))

    print("\nHeld-out permutation importance:")
    print(agg_imp.to_string(index=False))

    print("\nResidual Moran's I:")
    print(moran.to_string(index=False))

    print("\nDecision:")
    print(improvement_label)
    print(f"ET9P lower RMSE in {folds_rmse_better}/5 outer folds.")
    print("FINAL AGBD MAP REMAINS ON HOLD until review.")

    print("\nCreated:")
    print(report_path)
    print(TABLE_DIR/"ET9P_vs_ET3P_block_bootstrap_summary.csv")
    print(TABLE_DIR/"ET9P_vs_ET3P_fold_comparison.csv")
    print(TABLE_DIR/"ET9P_permutation_importance_aggregated.csv")
    print(TABLE_DIR/"ET9P_residual_MoransI.csv")
    print(f"Figures: {FIG_DIR}")

    print("\nIMPORTANT:")
    print("Do not create a final biomass map yet.")
    print("Send the Phase 9 report and the four summary CSV files to ChatGPT.")
    input("\nPress Enter to close...")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n" + "=" * 92)
        print("PHASE 9 FAILED")
        print("=" * 92)
        traceback.print_exc()
        input("\nPress Enter to close...")
        sys.exit(1)
