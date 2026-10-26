
from pathlib import Path
from datetime import datetime
import hashlib, sys
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(r"D:\NISAR_Mangrove_Paper")
INPUT_CSV = PROJECT_ROOT / "05_ML_DL" / "Spatial_CV" / "GEDI_NISAR_modeling_table_with_spatial_folds.csv"
OUT_ROOT = PROJECT_ROOT / "05_ML_DL" / "Spatial_CV" / "Buffered_500m"
QC_DIR = PROJECT_ROOT / "06_Results" / "Spatial_CV_Buffered_500m"
LOG_PATH = PROJECT_ROOT / "00_Documentation" / "analysis_log.txt"
BUFFER_M = 500.0
N_FOLDS = 5

OUT_ROOT.mkdir(parents=True, exist_ok=True)
QC_DIR.mkdir(parents=True, exist_ok=True)

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def append_log(lines):
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write("\n[" + stamp + "] PHASE 6B — 500 M BUFFERED SPATIAL CV\n")
        for line in lines:
            f.write(line + "\n")
        f.write("-" * 72 + "\n")

def get_xy(df):
    if {"cv_x","cv_y"}.issubset(df.columns):
        x = pd.to_numeric(df["cv_x"], errors="coerce").to_numpy(float)
        y = pd.to_numeric(df["cv_y"], errors="coerce").to_numpy(float)
    elif {"nisar_x","nisar_y"}.issubset(df.columns):
        x = pd.to_numeric(df["nisar_x"], errors="coerce").to_numpy(float)
        y = pd.to_numeric(df["nisar_y"], errors="coerce").to_numpy(float)
    else:
        raise RuntimeError("Projected coordinates missing: expected cv_x/cv_y or nisar_x/nisar_y.")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise RuntimeError("Non-finite projected coordinates found.")
    return np.column_stack([x, y])

def agbd_stats(df, prefix):
    s = pd.to_numeric(df["agbd"], errors="coerce")
    return {
        f"{prefix}_agbd_mean": float(s.mean()),
        f"{prefix}_agbd_sd": float(s.std(ddof=1)),
        f"{prefix}_agbd_median": float(s.median()),
        f"{prefix}_agbd_min": float(s.min()),
        f"{prefix}_agbd_max": float(s.max()),
    }

def main():
    print("=" * 72)
    print("PHASE 6B — 500 M BUFFERED SPATIAL CROSS-VALIDATION AUDIT")
    print("=" * 72)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Phase 6 fold table not found:\n{INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    required = {"spatial_fold","agbd"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {sorted(missing)}")

    folds = set(df["spatial_fold"].dropna().astype(int).unique())
    if folds != set(range(1, N_FOLDS + 1)):
        raise RuntimeError(f"Expected spatial_fold 1..5, found {sorted(folds)}")

    xy = get_xy(df)

    print(f"Input records: {len(df):,}")
    print(f"Input SHA256: {sha256_file(INPUT_CSV)}")
    print(f"Buffer distance: {BUFFER_M:.0f} m")

    rows, removed_all = [], []

    for fold in range(1, N_FOLDS + 1):
        fold_dir = OUT_ROOT / f"Fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        test_mask = df["spatial_fold"].astype(int).eq(fold).to_numpy()
        train_mask = ~test_mask

        test_df = df.loc[test_mask].copy()
        train_df = df.loc[train_mask].copy()

        tree = cKDTree(xy[test_mask])
        dmin, _ = tree.query(xy[train_mask], k=1)

        keep = dmin >= BUFFER_M
        removed = ~keep

        train_final = train_df.loc[keep].copy()
        train_removed = train_df.loc[removed].copy()

        train_final["min_distance_to_test_m"] = dmin[keep]
        train_removed["min_distance_to_test_m"] = dmin[removed]

        test_df.to_csv(fold_dir / f"Fold_{fold}_TEST.csv", index=False)
        train_final.to_csv(fold_dir / f"Fold_{fold}_TRAIN_BUFFERED_500m.csv", index=False)
        train_removed.to_csv(fold_dir / f"Fold_{fold}_TRAIN_REMOVED_WITHIN_500m.csv", index=False)

        id_cols = [c for c in ["shot_number","cv_block_id","spatial_fold","longitude","latitude","cv_x","cv_y"] if c in df.columns]
        test_df[id_cols].to_csv(fold_dir / f"Fold_{fold}_TEST_INDEX.csv", index=False)
        train_final[id_cols + ["min_distance_to_test_m"]].to_csv(
            fold_dir / f"Fold_{fold}_TRAIN_INDEX_BUFFERED_500m.csv", index=False
        )

        if len(train_removed):
            tmp = train_removed[id_cols + ["min_distance_to_test_m"]].copy()
            tmp.insert(0, "outer_fold", fold)
            removed_all.append(tmp)

        row = {
            "fold": fold,
            "test_n": len(test_df),
            "train_candidate_n": len(train_df),
            "train_removed_within_500m_n": int(removed.sum()),
            "train_final_n": len(train_final),
            "train_retained_percent": float(100 * len(train_final) / len(train_df)),
            "min_train_to_test_distance_after_buffer_m": float(train_final["min_distance_to_test_m"].min()),
            "p05_train_to_test_distance_after_buffer_m": float(train_final["min_distance_to_test_m"].quantile(0.05)),
            "median_train_to_test_distance_after_buffer_m": float(train_final["min_distance_to_test_m"].median()),
        }
        row.update(agbd_stats(test_df, "test"))
        row.update(agbd_stats(train_final, "train"))
        rows.append(row)

        print(
            f"Fold {fold}: test={len(test_df):,}, train candidates={len(train_df):,}, "
            f"removed={int(removed.sum()):,}, final train={len(train_final):,}, "
            f"min distance={row['min_train_to_test_distance_after_buffer_m']:.1f} m"
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(QC_DIR / "buffered_500m_fold_summary.csv", index=False)

    if removed_all:
        pd.concat(removed_all, ignore_index=True).to_csv(
            QC_DIR / "all_buffer_removed_training_records.csv", index=False
        )
    else:
        pd.DataFrame().to_csv(QC_DIR / "all_buffer_removed_training_records.csv", index=False)

    min_after = float(summary["min_train_to_test_distance_after_buffer_m"].min())
    min_train_n = int(summary["train_final_n"].min())

    status = "PASS"
    notes = []
    if min_after < BUFFER_M - 1e-6:
        status = "FAIL"
        notes.append("At least one retained training record remains inside 500 m.")
    if min_train_n < 800 and status != "FAIL":
        status = "REVIEW"
        notes.append("A buffered training fold contains fewer than 800 records.")

    report = [
        "PHASE 6B — 500 M BUFFERED SPATIAL CV QC",
        "=" * 72,
        f"Input fold table: {INPUT_CSV}",
        f"Input SHA256: {sha256_file(INPUT_CSV)}",
        f"Buffer distance: {BUFFER_M:.0f} m",
        f"Number of outer folds: {N_FOLDS}",
        f"Minimum retained train-to-test distance: {min_after:.3f} m",
        f"Minimum buffered training sample size: {min_train_n}",
        f"QC status: {status}",
        "",
        "Fold summary:",
        summary.to_string(index=False),
        "",
        "Rules:",
        "- Test membership is unchanged from Phase 6.",
        "- Training records within 500 m of the active test fold are excluded only for that fold.",
        "- The same Phase-6 spatial_fold values remain the master outer-CV assignment.",
        "- Hyperparameter tuning must occur only inside each buffered training set.",
        "- DNN/MLP scaling must be fitted only on the buffered training data.",
    ]
    if notes:
        report += ["", "QC notes:"] + [f"- {x}" for x in notes]

    (QC_DIR / "BUFFERED_500m_CV_QC_REPORT.txt").write_text("\n".join(report), encoding="utf-8")

    append_log([
        f"Input fold table: {INPUT_CSV}",
        f"Input SHA256: {sha256_file(INPUT_CSV)}",
        f"Buffer distance: {BUFFER_M:.0f} m",
        f"Minimum retained train-to-test distance: {min_after:.3f} m",
        f"Minimum buffered training n: {min_train_n}",
        f"QC status: {status}",
        "Phase-6 folds remain frozen; Phase 6B only removes nearby training observations."
    ])

    print("\n" + "=" * 72)
    print("PHASE 6B COMPLETE — BUFFER QC REVIEW REQUIRED")
    print("=" * 72)
    print(f"Minimum retained train-to-test distance: {min_after:.1f} m")
    print(f"Minimum buffered training n:            {min_train_n}")
    print(f"QC status:                              {status}")
    print(f"\nBuffered datasets: {OUT_ROOT}")
    print(f"QC results:        {QC_DIR}")
    print("\nDo not fit models until Phase 6B QC is reviewed.")
    input("\nPress Enter to close...")

if __name__ == "__main__":
    main()
