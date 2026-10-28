from pathlib import Path
import pandas as pd
import json
from collections import Counter

ROOT = Path(r"D:\NISAR_Mangrove_Paper")

FROZEN = (
    ROOT / "04_Matched_Data" / "Predictor_Expansion" /
    "Phase8C2" / "Tables" /
    "GEDI_NISAR_expanded_predictors_1641.csv"
)

OUTDIR = (
    ROOT / "06_Results" / "Reviewer_Validation" /
    "C1_Independent_Validation"
)

OUTDIR.mkdir(parents=True, exist_ok=True)

REPORT = OUTDIR / "C1_01_PHASE9_DEPENDENCY_AUDIT.txt"
INVENTORY = OUTDIR / "C1_01_PHASE9_DEPENDENCY_INVENTORY.csv"

EXACT_TARGETS = [
    "ET_OOF_predictions.csv",
    "expanded9P_OOF_predictions.csv",
    "expanded9P_selected_hyperparameters.csv",
    "ET3P_ET9P_core_metrics.csv",
    "ET9P_vs_ET3P_fold_comparison.csv",
    "ET9P_reproduction_audit.csv",
]

SEARCH_ROOTS = [
    ROOT / "05_ML_DL",
    ROOT / "06_Results",
]

# ============================================================
# Load frozen table
# ============================================================

df = pd.read_csv(FROZEN)

if len(df) != 1641:
    raise RuntimeError(f"Frozen table has {len(df)} rows, expected 1641.")

required = [
    "shot_number",
    "agbd",
    "utm_x",
    "utm_y",
    "cv_block_id",
    "spatial_fold",
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

missing = [c for c in required if c not in df.columns]

if missing:
    raise RuntimeError(f"Missing required frozen fields: {missing}")

# ============================================================
# Locate Phase 9 dependency files
# ============================================================

found = []

for base in SEARCH_ROOTS:
    if not base.exists():
        continue

    for p in base.rglob("*.csv"):

        name = p.name

        match_type = None

        if name in EXACT_TARGETS:
            match_type = "exact"

        elif name.startswith("Fold_") and name.endswith("_TEST.csv"):
            match_type = "fold_test"

        elif (
            name.startswith("Fold_")
            and name.endswith("_TRAIN_BUFFERED_500m.csv")
        ):
            match_type = "fold_train_buffered"

        if match_type is None:
            continue

        try:
            x = pd.read_csv(p)

            found.append({
                "file": str(p),
                "name": name,
                "type": match_type,
                "rows": len(x),
                "columns": len(x.columns),
                "column_names": " | ".join(map(str, x.columns)),
            })

        except Exception as e:

            found.append({
                "file": str(p),
                "name": name,
                "type": match_type,
                "rows": None,
                "columns": None,
                "column_names": f"ERROR: {e}",
            })

inv = pd.DataFrame(found)

if len(inv):
    inv = inv.sort_values(["type", "name", "file"])

inv.to_csv(INVENTORY, index=False)

# ============================================================
# Frozen spatial-fold summary
# ============================================================

fold_summary = (
    df.groupby("spatial_fold")
      .agg(
          n=("shot_number", "size"),
          blocks=("cv_block_id", "nunique"),
          agbd_mean=("agbd", "mean"),
          agbd_sd=("agbd", "std"),
      )
      .reset_index()
)

# ============================================================
# Original fold-file integrity checks
# ============================================================

test_files = []
train_files = []

for rec in found:

    if rec["type"] == "fold_test":
        test_files.append(Path(rec["file"]))

    elif rec["type"] == "fold_train_buffered":
        train_files.append(Path(rec["file"]))

all_test_ids = []
fold_checks = []

for test_path in sorted(test_files):

    name = test_path.name

    try:
        fold_num = int(name.split("_")[1])
    except Exception:
        fold_num = None

    test = pd.read_csv(test_path)

    train_match = None

    if fold_num is not None:
        expected_train_name = (
            f"Fold_{fold_num}_TRAIN_BUFFERED_500m.csv"
        )

        candidates = [
            x for x in train_files
            if x.name == expected_train_name
        ]

        if candidates:
            train_match = candidates[0]

    test_ids = set(
        test["shot_number"].astype(str)
    ) if "shot_number" in test.columns else set()

    all_test_ids.extend(list(test_ids))

    overlap = None
    n_train = None
    min_dist = None

    if train_match is not None:

        train = pd.read_csv(train_match)

        n_train = len(train)

        train_ids = set(
            train["shot_number"].astype(str)
        ) if "shot_number" in train.columns else set()

        overlap = len(test_ids.intersection(train_ids))

    fold_checks.append({
        "fold": fold_num,
        "test_file": str(test_path),
        "n_test": len(test),
        "train_file": str(train_match) if train_match else None,
        "n_train_buffered": n_train,
        "train_test_shot_overlap": overlap,
    })

fold_checks_df = pd.DataFrame(fold_checks)

# ============================================================
# Test union checks
# ============================================================

frozen_ids = set(df["shot_number"].astype(str))

test_counter = Counter(all_test_ids)

test_union = set(all_test_ids)

test_union_matches_frozen = (
    test_union == frozen_ids
    if all_test_ids else False
)

test_ids_appearing_more_than_once = [
    sid for sid, count in test_counter.items()
    if count > 1
]

# ============================================================
# Hyperparameter and OOF file summaries
# ============================================================

special_contents = []

for rec in found:

    if rec["name"] in [
        "expanded9P_selected_hyperparameters.csv",
        "ET3P_ET9P_core_metrics.csv",
        "ET9P_vs_ET3P_fold_comparison.csv",
        "ET9P_reproduction_audit.csv",
    ]:

        p = Path(rec["file"])

        try:
            x = pd.read_csv(p)

            special_contents.append(
                (rec["name"], str(p), x.to_string(index=False))
            )

        except Exception as e:
            special_contents.append(
                (rec["name"], str(p), f"ERROR: {e}")
            )

# ============================================================
# OOF identity audit
# ============================================================

oof_checks = []

for rec in found:

    if rec["name"] not in [
        "ET_OOF_predictions.csv",
        "expanded9P_OOF_predictions.csv",
    ]:
        continue

    p = Path(rec["file"])

    try:
        x = pd.read_csv(p)

        check = {
            "file": str(p),
            "rows": len(x),
            "has_shot_number": "shot_number" in x.columns,
            "has_fold":
                "fold" in x.columns
                or "spatial_fold" in x.columns,
        }

        if "shot_number" in x.columns:

            ids = set(x["shot_number"].astype(str))

            check["unique_shots"] = len(ids)
            check["matches_frozen_shots"] = ids == frozen_ids

        else:
            check["unique_shots"] = None
            check["matches_frozen_shots"] = None

        oof_checks.append(check)

    except Exception as e:

        oof_checks.append({
            "file": str(p),
            "error": str(e),
        })

# ============================================================
# Write report
# ============================================================

with REPORT.open("w", encoding="utf-8") as f:

    f.write("=" * 90 + "\n")
    f.write("C1 PHASE 9 DEPENDENCY / FOLD INTEGRITY AUDIT\n")
    f.write("=" * 90 + "\n\n")

    f.write("FROZEN TABLE\n")
    f.write("-" * 90 + "\n")
    f.write(f"{FROZEN}\n")
    f.write(f"Rows: {len(df)}\n")
    f.write(f"Unique shots: {df['shot_number'].nunique()}\n")
    f.write(f"Unique CV blocks: {df['cv_block_id'].nunique()}\n")
    f.write(f"Spatial folds: {sorted(df['spatial_fold'].unique())}\n\n")

    f.write("FROZEN FOLD SUMMARY\n")
    f.write("-" * 90 + "\n")
    f.write(fold_summary.to_string(index=False))
    f.write("\n\n")

    f.write("PHASE 9 DEPENDENCY FILES FOUND\n")
    f.write("-" * 90 + "\n")

    if len(inv):
        for _, r in inv.iterrows():
            f.write(
                f"{r['type']:22s} | "
                f"{str(r['rows']):>6s} rows | "
                f"{r['file']}\n"
            )
    else:
        f.write("NONE FOUND\n")

    f.write("\n")

    f.write("ORIGINAL TRAIN/TEST FOLD CHECKS\n")
    f.write("-" * 90 + "\n")

    if len(fold_checks_df):
        f.write(fold_checks_df.to_string(index=False))
    else:
        f.write("No fold-specific TEST files were located.")

    f.write("\n\n")

    f.write("TEST-FOLD UNION CHECK\n")
    f.write("-" * 90 + "\n")
    f.write(f"Frozen shots: {len(frozen_ids)}\n")
    f.write(f"Union of located test shots: {len(test_union)}\n")
    f.write(
        f"Test union exactly matches frozen 1641 shots: "
        f"{test_union_matches_frozen}\n"
    )
    f.write(
        f"Test IDs appearing in more than one outer fold: "
        f"{len(test_ids_appearing_more_than_once)}\n"
    )

    f.write("\nOOF FILE CHECKS\n")
    f.write("-" * 90 + "\n")

    if oof_checks:
        for item in oof_checks:
            f.write(json.dumps(item, indent=2))
            f.write("\n")
    else:
        f.write("No OOF prediction files located.\n")

    f.write("\nSELECTED HYPERPARAMETERS / SAVED SUMMARY TABLES\n")
    f.write("-" * 90 + "\n")

    if special_contents:
        for name, path, content in special_contents:

            f.write(f"\nFILE: {name}\n")
            f.write(f"PATH: {path}\n")
            f.write(content)
            f.write("\n")
    else:
        f.write("None located.\n")

    f.write("\n" + "=" * 90 + "\n")
    f.write("END OF C1 DEPENDENCY AUDIT\n")
    f.write("=" * 90 + "\n")

# ============================================================
# Console
# ============================================================

print("")
print("=" * 80)
print("C1 PHASE 9 DEPENDENCY AUDIT COMPLETE")
print("=" * 80)

print("")
print("Frozen table:")
print(FROZEN)

print("")
print("Rows:", len(df))
print("Unique shots:", df["shot_number"].nunique())
print("Unique blocks:", df["cv_block_id"].nunique())

print("")
print("Fold counts:")
print(
    df["spatial_fold"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("")
print("Dependency files found:", len(found))

print("")
print("Fold TEST files found:", len(test_files))
print("Fold buffered TRAIN files found:", len(train_files))

print("")
print(
    "Test union exactly matches frozen dataset:",
    test_union_matches_frozen
)

print(
    "Test observations repeated across folds:",
    len(test_ids_appearing_more_than_once)
)

print("")
print("OOF checks:")
for x in oof_checks:
    print(x)

print("")
print("Inventory:")
print(INVENTORY)

print("")
print("Detailed report:")
print(REPORT)

print("")
print("NO MODEL WAS FITTED.")
print("NO PHASE 9 FILE WAS MODIFIED.")
print("=" * 80)
