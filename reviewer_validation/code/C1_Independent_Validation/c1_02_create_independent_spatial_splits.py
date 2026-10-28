from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree
import hashlib

# ============================================================
# PATHS
# ============================================================

ROOT = Path(r"D:\NISAR_Mangrove_Paper")

FROZEN = (
    ROOT / "04_Matched_Data" / "Predictor_Expansion" /
    "Phase8C2" / "Tables" /
    "GEDI_NISAR_expanded_predictors_1641.csv"
)

OUTDIR = (
    ROOT / "06_Results" /
    "Reviewer_Validation" /
    "C1_Independent_Validation"
)

SPLITDIR = OUTDIR / "Independent_Outer_Folds"

OUTDIR.mkdir(parents=True, exist_ok=True)
SPLITDIR.mkdir(parents=True, exist_ok=True)

MANIFEST = OUTDIR / "C1_02_NEW_SPLIT_MANIFEST.csv"
SUMMARY = OUTDIR / "C1_02_NEW_OUTER_SPLIT_SUMMARY.csv"
CROSSTAB = OUTDIR / "C1_02_OLD_vs_NEW_FOLD_CROSSTAB.csv"
BLOCKMAP = OUTDIR / "C1_02_BLOCK_ASSIGNMENTS.csv"
REPORT = OUTDIR / "C1_02_NEW_SPLIT_PROTOCOL.txt"

# ============================================================
# LOCKED PROTOCOL
# ============================================================

N_OUTER_FOLDS = 5

# Deliberately different from original Phase 6/9 seed 42.
NEW_SPLIT_SEED = 20261027

# Same local exclusion distance as original analysis for C1.
# Larger distances are tested separately under reviewer comment C2.
OUTER_BUFFER_M = 500.0

# ============================================================
# LOAD FROZEN DATA
# ============================================================

df = pd.read_csv(FROZEN)

required = [
    "shot_number",
    "utm_x",
    "utm_y",
    "cv_block_id",
    "spatial_fold",
    "agbd",
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise RuntimeError(
        f"Required columns missing: {missing}"
    )

if len(df) != 1641:
    raise RuntimeError(
        f"Expected 1641 rows, found {len(df)}"
    )

if df["shot_number"].nunique() != 1641:
    raise RuntimeError(
        "shot_number is not unique."
    )

if df["cv_block_id"].nunique() != 54:
    raise RuntimeError(
        f"Expected 54 blocks, found "
        f"{df['cv_block_id'].nunique()}"
    )

# ============================================================
# HASH FROZEN INPUT
# ============================================================

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

input_sha256 = sha256_file(FROZEN)

# ============================================================
# CREATE BLOCK-CENTROID TABLE
#
# IMPORTANT:
# No AGBD values are used here.
# Each existing 3-km block is treated as one spatial unit.
# ============================================================

block = (
    df.groupby("cv_block_id", as_index=False)
      .agg(
          centroid_x=("utm_x", "mean"),
          centroid_y=("utm_y", "mean"),
          n_observations=("shot_number", "size"),
      )
)

# ============================================================
# COORDINATE-ONLY SPATIAL CLUSTERING
#
# KMeans is applied to block centroids, not observations.
# This creates five geographically coherent holdout groups.
# ============================================================

coords = block[
    ["centroid_x", "centroid_y"]
].to_numpy()

km = KMeans(
    n_clusters=N_OUTER_FOLDS,
    random_state=NEW_SPLIT_SEED,
    n_init=100,
)

raw_cluster = km.fit_predict(coords)

block["raw_cluster"] = raw_cluster

# ============================================================
# STABLE FOLD NUMBERING
#
# KMeans labels are arbitrary.
# Re-number clusters west-to-east using centroid_x.
# This changes labels only, not membership.
# ============================================================

cluster_centers = (
    block.groupby("raw_cluster", as_index=False)
         .agg(
             cluster_x=("centroid_x", "mean"),
             cluster_y=("centroid_y", "mean"),
         )
         .sort_values(
             ["cluster_x", "cluster_y"]
         )
         .reset_index(drop=True)
)

cluster_centers["new_outer_fold"] = (
    np.arange(len(cluster_centers)) + 1
)

cluster_to_fold = dict(
    zip(
        cluster_centers["raw_cluster"],
        cluster_centers["new_outer_fold"]
    )
)

block["new_outer_fold"] = (
    block["raw_cluster"]
    .map(cluster_to_fold)
    .astype(int)
)

# ============================================================
# MERGE NEW BLOCK ASSIGNMENT TO ALL 1641 OBSERVATIONS
# ============================================================

df_new = df.merge(
    block[
        [
            "cv_block_id",
            "centroid_x",
            "centroid_y",
            "new_outer_fold"
        ]
    ],
    on="cv_block_id",
    how="left",
    validate="many_to_one",
)

if df_new["new_outer_fold"].isna().any():
    raise RuntimeError(
        "Some observations did not receive a new fold."
    )

# ============================================================
# VERIFY BLOCK INTEGRITY
# ============================================================

block_fold_counts = (
    df_new.groupby("cv_block_id")[
        "new_outer_fold"
    ].nunique()
)

if (block_fold_counts != 1).any():
    raise RuntimeError(
        "At least one 3-km block was split "
        "between new outer folds."
    )

# ============================================================
# SAVE FROZEN NEW SPLIT BEFORE ANY MODELING
# ============================================================

manifest_cols = [
    "shot_number",
    "cv_block_id",
    "spatial_fold",
    "new_outer_fold",
    "utm_x",
    "utm_y",
]

manifest = df_new[manifest_cols].copy()

manifest.rename(
    columns={
        "spatial_fold":
        "original_phase6_spatial_fold"
    },
    inplace=True
)

manifest.to_csv(
    MANIFEST,
    index=False
)

block_out = block[
    [
        "cv_block_id",
        "centroid_x",
        "centroid_y",
        "n_observations",
        "new_outer_fold"
    ]
].copy()

block_out.to_csv(
    BLOCKMAP,
    index=False
)

manifest_sha256 = sha256_file(MANIFEST)

# ============================================================
# OLD VS NEW FOLD CROSS-TAB
#
# Diagnostic only.
# No reassignment is allowed after this point.
# ============================================================

cross = pd.crosstab(
    manifest["original_phase6_spatial_fold"],
    manifest["new_outer_fold"],
)

cross.to_csv(CROSSTAB)

# ============================================================
# BUILD 500-m BUFFERED TRAINING SETS
#
# These memberships are now frozen for C1.
# ============================================================

summaries = []

all_test_ids = []

for fold in range(1, N_OUTER_FOLDS + 1):

    fold_dir = SPLITDIR / f"Fold_{fold}"
    fold_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    test = df_new[
        df_new["new_outer_fold"] == fold
    ].copy()

    candidate_train = df_new[
        df_new["new_outer_fold"] != fold
    ].copy()

    test_xy = test[
        ["utm_x", "utm_y"]
    ].to_numpy()

    train_xy = candidate_train[
        ["utm_x", "utm_y"]
    ].to_numpy()

    tree = cKDTree(test_xy)

    nearest_dist, _ = tree.query(
        train_xy,
        k=1
    )

    candidate_train[
        "distance_to_nearest_outer_test_m"
    ] = nearest_dist

    train = candidate_train[
        candidate_train[
            "distance_to_nearest_outer_test_m"
        ] >= OUTER_BUFFER_M
    ].copy()

    if len(train) == 0:
        raise RuntimeError(
            f"Fold {fold}: no training data "
            f"after buffer."
        )

    min_distance = float(
        train[
            "distance_to_nearest_outer_test_m"
        ].min()
    )

    test_ids = set(
        test["shot_number"].astype(str)
    )

    train_ids = set(
        train["shot_number"].astype(str)
    )

    overlap = len(
        test_ids.intersection(train_ids)
    )

    if overlap != 0:
        raise RuntimeError(
            f"Fold {fold}: train/test "
            f"shot overlap = {overlap}"
        )

    all_test_ids.extend(
        list(test_ids)
    )

    test_path = (
        fold_dir /
        f"C1_NEW_Fold_{fold}_TEST.csv"
    )

    train_path = (
        fold_dir /
        f"C1_NEW_Fold_{fold}_TRAIN_BUFFERED_500m.csv"
    )

    test.to_csv(
        test_path,
        index=False
    )

    train.to_csv(
        train_path,
        index=False
    )

    summaries.append({
        "new_outer_fold": fold,
        "test_n": len(test),
        "test_blocks":
            test["cv_block_id"].nunique(),
        "candidate_train_n":
            len(candidate_train),
        "buffered_train_n":
            len(train),
        "removed_by_500m_buffer":
            len(candidate_train) - len(train),
        "buffered_train_blocks":
            train["cv_block_id"].nunique(),
        "minimum_train_test_distance_m":
            min_distance,
        "train_test_shot_overlap":
            overlap,
    })

summary = pd.DataFrame(summaries)

summary.to_csv(
    SUMMARY,
    index=False
)

# ============================================================
# FINAL SPLIT-INTEGRITY CHECK
# ============================================================

all_test_ids = list(all_test_ids)

if len(all_test_ids) != 1641:
    raise RuntimeError(
        "Total test membership count is not 1641."
    )

if len(set(all_test_ids)) != 1641:
    raise RuntimeError(
        "At least one observation appears in "
        "multiple new outer test folds."
    )

frozen_ids = set(
    df["shot_number"].astype(str)
)

if set(all_test_ids) != frozen_ids:
    raise RuntimeError(
        "New outer-test union does not exactly "
        "match frozen dataset."
    )

# ============================================================
# REPORT
# ============================================================

with REPORT.open(
    "w",
    encoding="utf-8"
) as f:

    f.write("=" * 90 + "\n")
    f.write(
        "C1 INDEPENDENT SPATIAL VALIDATION "
        "- PRE-REGISTERED OUTER SPLITS\n"
    )
    f.write("=" * 90 + "\n\n")

    f.write("PURPOSE\n")
    f.write("-" * 90 + "\n")

    f.write(
        "Create a new selection-independent "
        "outer spatial validation geometry "
        "before any reviewer-validation model "
        "is fitted.\n\n"
    )

    f.write("LOCKED RULES\n")
    f.write("-" * 90 + "\n")

    f.write(
        "1. Frozen dataset: 1,641 observations.\n"
    )

    f.write(
        "2. Spatial unit: existing 3-km "
        "cv_block_id.\n"
    )

    f.write(
        "3. Number of new outer folds: 5.\n"
    )

    f.write(
        "4. Split generation uses only UTM "
        "coordinates and block identity.\n"
    )

    f.write(
        "5. AGBD is NOT used for fold creation "
        "or balancing.\n"
    )

    f.write(
        "6. Each 3-km block belongs to exactly "
        "one new outer fold.\n"
    )

    f.write(
        "7. New outer test folds are buffered "
        "by 500 m from training observations.\n"
    )

    f.write(
        "8. Seed: "
        f"{NEW_SPLIT_SEED}.\n"
    )

    f.write(
        "9. Split membership is frozen after "
        "creation and must not be altered based "
        "on subsequent model performance.\n\n"
    )

    f.write("INPUT HASH\n")
    f.write("-" * 90 + "\n")
    f.write(
        f"{input_sha256}  {FROZEN}\n\n"
    )

    f.write("SPLIT MANIFEST HASH\n")
    f.write("-" * 90 + "\n")
    f.write(
        f"{manifest_sha256}  {MANIFEST}\n\n"
    )

    f.write("NEW OUTER-FOLD SUMMARY\n")
    f.write("-" * 90 + "\n")
    f.write(
        summary.to_string(index=False)
    )
    f.write("\n\n")

    f.write("OLD VS NEW FOLD CROSS-TAB\n")
    f.write("-" * 90 + "\n")
    f.write(
        cross.to_string()
    )
    f.write("\n\n")

    f.write("INTEGRITY CHECKS\n")
    f.write("-" * 90 + "\n")

    f.write(
        f"Total test memberships: "
        f"{len(all_test_ids)}\n"
    )

    f.write(
        f"Unique test shots: "
        f"{len(set(all_test_ids))}\n"
    )

    f.write(
        "Test union equals frozen dataset: "
        f"{set(all_test_ids) == frozen_ids}\n"
    )

    f.write(
        "Blocks split across folds: "
        f"{int((block_fold_counts != 1).sum())}\n"
    )

    f.write(
        "Any train/test shot overlap: "
        f"{bool((summary['train_test_shot_overlap'] > 0).any())}\n"
    )

    f.write("\n")
    f.write(
        "NO MODEL WAS FITTED DURING THIS STEP.\n"
    )

    f.write(
        "THE NEW OUTER SPLITS ARE NOW FROZEN.\n"
    )

    f.write("=" * 90 + "\n")

# ============================================================
# CONSOLE
# ============================================================

print("")
print("=" * 80)
print("C1 STEP 2 COMPLETE")
print("NEW INDEPENDENT OUTER SPLITS CREATED")
print("=" * 80)

print("")
print("Frozen observations:", len(df))
print("Spatial blocks:", df["cv_block_id"].nunique())

print("")
print("New fold summary:")
print(summary.to_string(index=False))

print("")
print("Old-vs-new fold cross-tab:")
print(cross.to_string())

print("")
print("Manifest:")
print(MANIFEST)

print("")
print("Manifest SHA256:")
print(manifest_sha256)

print("")
print("Protocol report:")
print(REPORT)

print("")
print("NO MODEL WAS FITTED.")
print("NO AGBD VALUE WAS USED TO ASSIGN FOLDS.")
print("THE NEW SPLITS ARE NOW FROZEN.")
print("=" * 80)
