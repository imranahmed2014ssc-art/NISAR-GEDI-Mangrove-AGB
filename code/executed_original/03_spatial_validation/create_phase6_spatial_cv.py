from pathlib import Path
from datetime import datetime
import sys
import hashlib

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial.distance import cdist
import geopandas as gpd
from shapely.geometry import box
import matplotlib.pyplot as plt

# ============================================================
# PROJECT SETTINGS
# ============================================================
PROJECT_ROOT = Path(r"D:\NISAR_Mangrove_Paper")
INPUT_CSV = PROJECT_ROOT / "04_Matched_Data" / "GEDI_NISAR_modeling_table_FINAL.csv"

CV_DIR = PROJECT_ROOT / "05_ML_DL" / "Spatial_CV"
RESULT_DIR = PROJECT_ROOT / "06_Results" / "Spatial_CV"
LOG_PATH = PROJECT_ROOT / "00_Documentation" / "analysis_log.txt"

BLOCK_SIZE_M = 3000
N_FOLDS = 5
SEED = 42

# Descriptive AGBD strata used only to BALANCE fold assignment.
# They do not alter or filter the response.
AGBD_EDGES = np.array([0, 25, 50, 75, 100, 125, 150, 175, 200, np.inf], dtype=float)

CV_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

def package_version(name):
    try:
        import importlib.metadata as md
        return md.version(name)
    except Exception:
        return "unknown"

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def append_log(lines):
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    block = ["", f"[{stamp}] PHASE 6 — 5-FOLD SPATIAL BLOCK CROSS-VALIDATION"] + lines + ["-" * 72]
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write("\n".join(block) + "\n")

def assign_agbd_bin(values):
    idx = np.digitize(values, AGBD_EDGES[1:-1], right=False)
    return idx.astype(int)

def objective(fold_n, fold_bins, fold_block_n, target_n, target_bins, target_blocks):
    # Balanced total sample count
    count_term = np.mean(((fold_n - target_n) / max(target_n, 1.0)) ** 2)

    # Balanced AGBD-bin counts. Ignore bins with zero target.
    denom = np.where(target_bins > 0, target_bins, 1.0)
    bin_term = np.mean(((fold_bins - target_bins) / denom) ** 2)

    # Avoid one fold receiving too many/few blocks.
    block_term = np.mean(((fold_block_n - target_blocks) / max(target_blocks, 1.0)) ** 2)

    return 1.0 * count_term + 1.25 * bin_term + 0.20 * block_term

def make_block_table(df):
    # Prefer already-computed NISAR/UTM coordinates when present.
    if {"nisar_x", "nisar_y"}.issubset(df.columns):
        x = pd.to_numeric(df["nisar_x"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(df["nisar_y"], errors="coerce").to_numpy(dtype=float)
    else:
        tr = Transformer.from_crs("EPSG:4326", "EPSG:32645", always_xy=True)
        x, y = tr.transform(
            df["longitude"].to_numpy(dtype=float),
            df["latitude"].to_numpy(dtype=float)
        )
        x = np.asarray(x)
        y = np.asarray(y)

    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise RuntimeError("Non-finite projected coordinates found.")

    # Global UTM-grid origin (0,0): deterministic and independent of sample extent.
    bx = np.floor(x / BLOCK_SIZE_M).astype(int)
    by = np.floor(y / BLOCK_SIZE_M).astype(int)
    block_id = np.array([f"{a}_{b}" for a, b in zip(bx, by)], dtype=object)

    out = df.copy()
    out["cv_x"] = x
    out["cv_y"] = y
    out["cv_block_x"] = bx
    out["cv_block_y"] = by
    out["cv_block_id"] = block_id
    out["agbd_balance_bin"] = assign_agbd_bin(
        pd.to_numeric(out["agbd"], errors="coerce").to_numpy(dtype=float)
    )

    # One row per occupied block.
    bin_cols = []
    agg_parts = []
    for b in range(len(AGBD_EDGES) - 1):
        name = f"bin_{b}_n"
        bin_cols.append(name)
        temp = (
            out.assign(_flag=(out["agbd_balance_bin"] == b).astype(int))
               .groupby("cv_block_id")["_flag"].sum()
               .rename(name)
        )
        agg_parts.append(temp)

    blocks = out.groupby("cv_block_id").agg(
        cv_block_x=("cv_block_x", "first"),
        cv_block_y=("cv_block_y", "first"),
        n=("agbd", "size"),
        mean_agbd=("agbd", "mean"),
        median_agbd=("agbd", "median"),
        min_agbd=("agbd", "min"),
        max_agbd=("agbd", "max"),
        x_mean=("cv_x", "mean"),
        y_mean=("cv_y", "mean"),
    )

    for temp in agg_parts:
        blocks = blocks.join(temp)

    blocks = blocks.reset_index()
    return out, blocks, bin_cols

def greedy_balance(blocks, bin_cols):
    rng = np.random.default_rng(SEED)

    B = len(blocks)
    K = len(bin_cols)
    target_n = blocks["n"].sum() / N_FOLDS
    target_bins = blocks[bin_cols].sum(axis=0).to_numpy(dtype=float) / N_FOLDS
    target_blocks = B / N_FOLDS

    # Hardest/largest blocks first.
    entropy = []
    for _, r in blocks.iterrows():
        v = r[bin_cols].to_numpy(dtype=float)
        p = v / max(v.sum(), 1)
        nz = p[p > 0]
        ent = -np.sum(nz * np.log(nz)) if len(nz) else 0.0
        entropy.append(ent)

    order = (
        blocks.assign(_entropy=entropy, _tie=rng.random(B))
              .sort_values(["n", "_entropy", "_tie"], ascending=[False, False, True])
              .index.to_list()
    )

    assignment = np.full(B, -1, dtype=int)
    fold_n = np.zeros(N_FOLDS, dtype=float)
    fold_bins = np.zeros((N_FOLDS, K), dtype=float)
    fold_block_n = np.zeros(N_FOLDS, dtype=float)

    # Seed first five blocks into different folds.
    for f, idx in enumerate(order[:N_FOLDS]):
        assignment[idx] = f
        fold_n[f] += blocks.loc[idx, "n"]
        fold_bins[f] += blocks.loc[idx, bin_cols].to_numpy(dtype=float)
        fold_block_n[f] += 1

    for idx in order[N_FOLDS:]:
        bn = float(blocks.loc[idx, "n"])
        bb = blocks.loc[idx, bin_cols].to_numpy(dtype=float)

        scores = []
        for f in range(N_FOLDS):
            n2 = fold_n.copy()
            b2 = fold_bins.copy()
            c2 = fold_block_n.copy()
            n2[f] += bn
            b2[f] += bb
            c2[f] += 1
            scores.append(objective(n2, b2, c2, target_n, target_bins, target_blocks))

        best = int(np.argmin(scores))
        assignment[idx] = best
        fold_n[best] += bn
        fold_bins[best] += bb
        fold_block_n[best] += 1

    # Deterministic local swap improvement.
    def current_obj(assign):
        fn = np.zeros(N_FOLDS)
        fb = np.zeros((N_FOLDS, K))
        fc = np.zeros(N_FOLDS)
        for i, f in enumerate(assign):
            fn[f] += blocks.loc[i, "n"]
            fb[f] += blocks.loc[i, bin_cols].to_numpy(dtype=float)
            fc[f] += 1
        return objective(fn, fb, fc, target_n, target_bins, target_blocks)

    best_obj = current_obj(assignment)
    for _ in range(25000):
        i, j = rng.integers(0, B, size=2)
        if i == j or assignment[i] == assignment[j]:
            continue
        test = assignment.copy()
        test[i], test[j] = test[j], test[i]
        obj = current_obj(test)
        if obj + 1e-12 < best_obj:
            assignment = test
            best_obj = obj

    # Human-facing fold IDs 1..5.
    return assignment + 1, best_obj

def fold_summary(df):
    rows = []
    for f in range(1, N_FOLDS + 1):
        s = df[df["spatial_fold"] == f]
        rows.append({
            "fold": f,
            "n": len(s),
            "occupied_blocks": s["cv_block_id"].nunique(),
            "agbd_mean": s["agbd"].mean(),
            "agbd_sd": s["agbd"].std(ddof=1),
            "agbd_median": s["agbd"].median(),
            "agbd_min": s["agbd"].min(),
            "agbd_max": s["agbd"].max(),
            "longitude_min": s["longitude"].min(),
            "longitude_max": s["longitude"].max(),
            "latitude_min": s["latitude"].min(),
            "latitude_max": s["latitude"].max(),
        })
    return pd.DataFrame(rows)

def fold_bin_counts(df):
    labels = [
        "[0,25)", "[25,50)", "[50,75)", "[75,100)", "[100,125)",
        "[125,150)", "[150,175)", "[175,200)", "[200,+inf)"
    ]
    rows = []
    for f in range(1, N_FOLDS + 1):
        s = df[df["spatial_fold"] == f]
        counts = s["agbd_balance_bin"].value_counts().reindex(range(len(labels)), fill_value=0)
        for b, lab in enumerate(labels):
            rows.append({
                "fold": f,
                "agbd_bin": lab,
                "count": int(counts.loc[b]),
                "percent_within_fold": float(100.0 * counts.loc[b] / len(s))
            })
    return pd.DataFrame(rows)

def near_duplicate_check(df, threshold_m=30.0):
    # Detect pairs closer than threshold and report whether any cross fold.
    xy = df[["cv_x", "cv_y"]].to_numpy(dtype=float)
    cross_rows = []
    for i in range(len(df)):
        # Small dataset: vectorized row-wise search is sufficient.
        d = np.sqrt(np.sum((xy[i+1:] - xy[i])**2, axis=1))
        hits = np.where(d < threshold_m)[0]
        for h in hits:
            j = i + 1 + int(h)
            if int(df.iloc[i]["spatial_fold"]) != int(df.iloc[j]["spatial_fold"]):
                cross_rows.append({
                    "row_i": i,
                    "row_j": j,
                    "shot_i": df.iloc[i].get("shot_number", i),
                    "shot_j": df.iloc[j].get("shot_number", j),
                    "distance_m": float(d[h]),
                    "fold_i": int(df.iloc[i]["spatial_fold"]),
                    "fold_j": int(df.iloc[j]["spatial_fold"]),
                })
    out = pd.DataFrame(cross_rows)
    out.to_csv(RESULT_DIR / "cross_fold_pairs_under_30m.csv", index=False)
    return len(out)

def train_test_distance_summary(df):
    rows = []
    for f in range(1, N_FOLDS + 1):
        test = df[df["spatial_fold"] == f][["cv_x", "cv_y"]].to_numpy(dtype=float)
        train = df[df["spatial_fold"] != f][["cv_x", "cv_y"]].to_numpy(dtype=float)
        D = cdist(test, train)
        nearest = np.min(D, axis=1)
        rows.append({
            "fold": f,
            "test_n": len(test),
            "train_n": len(train),
            "min_test_to_train_m": float(np.min(nearest)),
            "p05_test_to_train_m": float(np.quantile(nearest, 0.05)),
            "median_test_to_train_m": float(np.median(nearest)),
            "p95_test_to_train_m": float(np.quantile(nearest, 0.95)),
        })
    out = pd.DataFrame(rows)
    out.to_csv(RESULT_DIR / "fold_train_test_distance_summary.csv", index=False)
    return out

def make_block_gpkg(blocks):
    geoms = []
    for _, r in blocks.iterrows():
        bx = int(r["cv_block_x"])
        by = int(r["cv_block_y"])
        x0 = bx * BLOCK_SIZE_M
        y0 = by * BLOCK_SIZE_M
        geoms.append(box(x0, y0, x0 + BLOCK_SIZE_M, y0 + BLOCK_SIZE_M))
    gdf = gpd.GeoDataFrame(blocks.copy(), geometry=geoms, crs="EPSG:32645")
    out = CV_DIR / "spatial_blocks_3000m_with_folds.gpkg"
    if out.exists():
        out.unlink()
    gdf.to_file(out, layer="spatial_blocks", driver="GPKG")
    return gdf

def save_fold_map(df, block_gdf):
    fig, ax = plt.subplots(figsize=(9.2, 7.2))

    for f in range(1, N_FOLDS + 1):
        bg = block_gdf[block_gdf["spatial_fold"] == f]
        bg.boundary.plot(ax=ax, linewidth=0.8, alpha=0.45)

    for f in range(1, N_FOLDS + 1):
        s = df[df["spatial_fold"] == f]
        ax.scatter(
            s["cv_x"], s["cv_y"],
            s=11, alpha=0.70,
            label=f"Fold {f} (n={len(s)})"
        )

    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title("3 km spatial blocks and final 5-fold assignment", pad=12)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        frameon=False
    )
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "spatial_cv_folds_map.png", dpi=450, bbox_inches="tight")
    fig.savefig(RESULT_DIR / "spatial_cv_folds_map.pdf", bbox_inches="tight")
    plt.close(fig)

def save_fold_counts(summary):
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    ax.bar(summary["fold"].astype(str), summary["n"])
    ax.set_xlabel("Spatial CV fold")
    ax.set_ylabel("Footprint count")
    ax.set_title("Sample count by spatial fold", pad=12)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "spatial_cv_fold_counts.png", dpi=450, bbox_inches="tight")
    fig.savefig(RESULT_DIR / "spatial_cv_fold_counts.pdf", bbox_inches="tight")
    plt.close(fig)

def main():
    print("=" * 72)
    print("PHASE 6 — CREATE FINAL 5-FOLD SPATIAL BLOCK CROSS-VALIDATION")
    print("=" * 72)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Frozen modelling table not found:\n{INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    required = {"agbd", "longitude", "latitude"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {sorted(missing)}")

    print(f"Input records: {len(df):,}")
    print(f"Block size: {BLOCK_SIZE_M/1000:.1f} km")
    print(f"Folds: {N_FOLDS}")
    print(f"Seed: {SEED}")

    df2, blocks, bin_cols = make_block_table(df)
    fold_ids, balance_obj = greedy_balance(blocks, bin_cols)
    blocks["spatial_fold"] = fold_ids

    fold_map = dict(zip(blocks["cv_block_id"], blocks["spatial_fold"]))
    df2["spatial_fold"] = df2["cv_block_id"].map(fold_map).astype(int)

    # Verify strict block integrity.
    split_blocks = (
        df2.groupby("cv_block_id")["spatial_fold"].nunique().gt(1).sum()
    )
    if split_blocks != 0:
        raise RuntimeError("ERROR: at least one 3 km block was split across folds.")

    summary = fold_summary(df2)
    bins = fold_bin_counts(df2)
    cross30 = near_duplicate_check(df2, threshold_m=30.0)
    dist_summary = train_test_distance_summary(df2)

    # Save final immutable assignment products.
    final_path = CV_DIR / "GEDI_NISAR_modeling_table_with_spatial_folds.csv"
    assign_path = CV_DIR / "spatial_fold_assignment.csv"
    blocks_path = CV_DIR / "spatial_block_assignment.csv"

    df2.to_csv(final_path, index=False)

    id_cols = [c for c in ["shot_number", "longitude", "latitude", "cv_x", "cv_y",
                           "cv_block_id", "spatial_fold"] if c in df2.columns]
    df2[id_cols].to_csv(assign_path, index=False)
    blocks.to_csv(blocks_path, index=False)

    summary.to_csv(RESULT_DIR / "spatial_fold_summary.csv", index=False)
    bins.to_csv(RESULT_DIR / "spatial_fold_AGBD_bin_counts.csv", index=False)

    block_gdf = make_block_gpkg(blocks)
    save_fold_map(df2, block_gdf)
    save_fold_counts(summary)

    count_cv = float(summary["n"].std(ddof=1) / summary["n"].mean())
    mean_range = float(summary["agbd_mean"].max() - summary["agbd_mean"].min())
    all_have_low = bool(
        bins[bins["agbd_bin"] == "[0,25)"].groupby("fold")["count"].sum().reindex(range(1,6), fill_value=0).gt(0).all()
    )
    all_have_high = bool(
        bins[bins["agbd_bin"].isin(["[125,150)", "[150,175)", "[175,200)", "[200,+inf)"])]
        .groupby("fold")["count"].sum().reindex(range(1,6), fill_value=0).gt(0).all()
    )

    status = (
        "PASS"
        if split_blocks == 0 and cross30 == 0 and summary["n"].min() >= 200
        and all_have_low and all_have_high
        else "REVIEW"
    )

    report_lines = [
        "PHASE 6 — SPATIAL CV QC",
        "=" * 72,
        f"Input frozen table: {INPUT_CSV}",
        f"Input SHA256: {sha256_file(INPUT_CSV)}",
        f"Block size: {BLOCK_SIZE_M} m",
        f"Number of occupied blocks: {len(blocks)}",
        f"Number of folds: {N_FOLDS}",
        f"Random seed: {SEED}",
        f"Block split violations: {split_blocks}",
        f"Cross-fold pairs closer than 30 m: {cross30}",
        f"Fold sample-count CV: {count_cv:.4f}",
        f"Range of fold mean AGBD: {mean_range:.3f} Mg/ha",
        f"Every fold contains AGBD <25 Mg/ha: {all_have_low}",
        f"Every fold contains AGBD >=125 Mg/ha: {all_have_high}",
        f"QC status: {status}",
        "",
        "Fold summary:",
        summary.to_string(index=False),
        "",
        "Train-test nearest-distance summary:",
        dist_summary.to_string(index=False),
        "",
        "Rule: exactly the same spatial_fold assignment must be used for every model.",
        "Rule: all hyperparameter tuning must occur inside the training data of each outer fold.",
        "Rule: DNN scaling/statistics must be fitted on training data only.",
    ]
    (RESULT_DIR / "SPATIAL_CV_QC_REPORT.txt").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    append_log([
        "Script: create_phase6_spatial_cv.py",
        f"Frozen input: {INPUT_CSV}",
        f"Input SHA256: {sha256_file(INPUT_CSV)}",
        "Phase-5B decision: 3 km selected because raw Moran's I first falls below ~0.05 at 3 km "
        "while detrended Moran's I is near zero, and 3 km retains 51 occupied blocks for fold construction.",
        f"Spatial block size: {BLOCK_SIZE_M} m",
        f"Occupied blocks: {len(blocks)}",
        f"Folds: {N_FOLDS}",
        f"Seed: {SEED}",
        f"Block split violations: {split_blocks}",
        f"Cross-fold pairs <30 m: {cross30}",
        f"Fold count CV: {count_cv:.4f}",
        f"Fold mean AGBD range: {mean_range:.3f} Mg/ha",
        f"QC status: {status}",
        "This fold assignment is frozen and must be reused for baseline, RF, XGBoost, and DNN.",
        f"Python: {sys.version.split()[0]}; numpy={package_version('numpy')}; "
        f"pandas={package_version('pandas')}; scipy={package_version('scipy')}; "
        f"pyproj={package_version('pyproj')}; geopandas={package_version('geopandas')}; "
        f"matplotlib={package_version('matplotlib')}",
    ])

    print("\n" + "=" * 72)
    print("PHASE 6 COMPLETE — FINAL FOLD QC REQUIRED")
    print("=" * 72)
    print(f"Occupied 3 km blocks:             {len(blocks)}")
    print(f"Block split violations:          {split_blocks}")
    print(f"Cross-fold pairs <30 m:          {cross30}")
    print(f"Fold count CV:                   {count_cv:.4f}")
    print(f"Fold mean AGBD range:            {mean_range:.2f} Mg/ha")
    print(f"QC status:                       {status}")

    print("\nFold summary:")
    print(summary.to_string(index=False))

    print("\nCreated:")
    print(f" - {final_path}")
    print(f" - {assign_path}")
    print(f" - {blocks_path}")
    print(f" - {CV_DIR / 'spatial_blocks_3000m_with_folds.gpkg'}")
    print(f" - {RESULT_DIR / 'spatial_fold_summary.csv'}")
    print(f" - {RESULT_DIR / 'spatial_fold_AGBD_bin_counts.csv'}")
    print(f" - {RESULT_DIR / 'fold_train_test_distance_summary.csv'}")
    print(f" - {RESULT_DIR / 'SPATIAL_CV_QC_REPORT.txt'}")
    print(f" - {RESULT_DIR / 'spatial_cv_folds_map.png'}")

    print("\nIMPORTANT:")
    print("Do not start model fitting until this Phase-6 fold assignment is reviewed.")
    input("\nPress Enter to close...")

if __name__ == "__main__":
    main()
