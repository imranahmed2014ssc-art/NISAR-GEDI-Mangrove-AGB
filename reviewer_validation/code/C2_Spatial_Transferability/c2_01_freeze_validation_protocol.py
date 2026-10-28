from pathlib import Path
from datetime import datetime
import hashlib
import json

import numpy as np
import pandas as pd

from scipy.spatial import cKDTree
from sklearn.cluster import KMeans


# ============================================================
# PATHS
# ============================================================

ROOT = Path(r"D:\NISAR_Mangrove_Paper")

DATA = (
    ROOT
    / "04_Matched_Data"
    / "Predictor_Expansion"
    / "Phase8C2"
    / "Tables"
    / "GEDI_NISAR_expanded_predictors_1641.csv"
)

C1_FOLD_ROOT = (
    ROOT
    / "06_Results"
    / "Reviewer_Validation"
    / "C1_Independent_Validation"
    / "Independent_Outer_Folds"
)

OUT = (
    ROOT
    / "06_Results"
    / "Reviewer_Validation"
    / "C2_Spatial_Transferability"
)

GEOM = OUT / "Frozen_Geometries"

BUFFER_DIR = GEOM / "Buffer_Sensitivity"
PASS_DIR = GEOM / "Pass_Holdout"
REPEAT_DIR = GEOM / "Repeated_Spatial_Seeds"

for d in [
    OUT,
    GEOM,
    BUFFER_DIR,
    PASS_DIR,
    REPEAT_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)


# ============================================================
# OUTPUTS
# ============================================================

REPORT = OUT / "C2_01_FROZEN_PROTOCOL_REPORT.txt"

BUFFER_SUMMARY = (
    OUT / "C2_01_BUFFER_GEOMETRY_SUMMARY.csv"
)

PASS_SUMMARY = (
    OUT / "C2_01_PASS_HOLDOUT_SUMMARY.csv"
)

REPEAT_SUMMARY = (
    OUT / "C2_01_REPEATED_SPLIT_SUMMARY.csv"
)

REPEAT_ASSIGNMENTS = (
    OUT / "C2_01_REPEATED_BLOCK_ASSIGNMENTS.csv"
)

EFFECTIVE_UNITS = (
    OUT / "C2_01_EFFECTIVE_UNIT_COUNTS.csv"
)

TRACK_MAPPING = (
    OUT / "C2_01_TRACK_GROUP_MAPPING.csv"
)

PROTOCOL_JSON = (
    OUT / "C2_01_LOCKED_PROTOCOL.json"
)


# ============================================================
# LOCKED C2 RULES
# ============================================================

BUFFER_DISTANCES_M = [
    500,
    1000,
    2000,
    3000,
]

PASS_BUFFER_M = 500

REPEAT_BUFFER_M = 500

REPEAT_SEEDS = [
    20261001,
    20261002,
    20261003,
    20261004,
    20261005,
    20261006,
    20261007,
    20261008,
    20261009,
    20261010,
]

N_REPEAT_FOLDS = 5

# C2 model is frozen BEFORE any C2 performance is seen.
C2_MODEL_FAMILY = "ExtraTrees"

C2_FEATURE_FAMILY = "FULL_14P"

C2_FEATURES = [
    "HH_dB",
    "HV_dB",
    "HV_SD",
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

INNER_CV = "4-fold GroupKFold by cv_block_id"
RANDOM_SEED_MODEL = 42


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            h.update(chunk)

    return h.hexdigest()


def numeric_xy(df):
    xy = df[
        ["utm_x", "utm_y"]
    ].to_numpy(float)

    if not np.isfinite(xy).all():
        raise RuntimeError(
            "Non-finite UTM coordinates found."
        )

    return xy


def buffered_training(
    candidate_train,
    test,
    buffer_m
):
    """
    Returns kept training rows, removed rows,
    and nearest-test distance for all candidates.
    """

    if len(test) == 0:
        raise RuntimeError(
            "Empty test set."
        )

    if len(candidate_train) == 0:
        raise RuntimeError(
            "Empty candidate training set."
        )

    test_tree = cKDTree(
        numeric_xy(test)
    )

    d, _ = test_tree.query(
        numeric_xy(candidate_train),
        k=1
    )

    work = candidate_train.copy()

    work[
        "nearest_test_distance_m"
    ] = d

    keep = (
        work[
            "nearest_test_distance_m"
        ]
        >= float(buffer_m)
    )

    kept = work.loc[
        keep
    ].copy()

    removed = work.loc[
        ~keep
    ].copy()

    return kept, removed


def save_memberships(
    train,
    test,
    train_path,
    test_path
):
    train_cols = [
        "shot_number",
        "cv_block_id",
        "utm_x",
        "utm_y",
        "nearest_test_distance_m",
    ]

    test_cols = [
        "shot_number",
        "cv_block_id",
        "utm_x",
        "utm_y",
    ]

    train[
        train_cols
    ].to_csv(
        train_path,
        index=False
    )

    test[
        test_cols
    ].to_csv(
        test_path,
        index=False
    )


def kish_effective_groups(
    counts
):
    counts = np.asarray(
        counts,
        dtype=float
    )

    if len(counts) == 0:
        return np.nan

    denominator = np.sum(
        counts ** 2
    )

    if denominator <= 0:
        return np.nan

    return float(
        np.sum(counts) ** 2
        / denominator
    )


def effective_unit_row(
    label,
    group_series
):
    counts = (
        group_series
        .value_counts(dropna=False)
        .to_numpy()
    )

    return {
        "unit_definition":
            label,

        "nominal_groups":
            int(len(counts)),

        "total_observations":
            int(np.sum(counts)),

        "min_group_n":
            int(np.min(counts)),

        "median_group_n":
            float(np.median(counts)),

        "mean_group_n":
            float(np.mean(counts)),

        "max_group_n":
            int(np.max(counts)),

        "kish_effective_groups":
            kish_effective_groups(
                counts
            ),

        "largest_group_fraction":
            float(
                np.max(counts)
                / np.sum(counts)
            ),
    }


# ============================================================
# LOAD FROZEN DATA
# ============================================================

df = pd.read_csv(
    DATA
)

if len(df) != 1641:
    raise RuntimeError(
        f"Expected 1641 rows; found {len(df)}."
    )

if (
    df["shot_number"].nunique()
    != 1641
):
    raise RuntimeError(
        "shot_number is not unique."
    )

required = [
    "shot_number",
    "cv_block_id",
    "utm_x",
    "utm_y",
    "source_file",
    "acquisition_date",
    "beam",
] + C2_FEATURES

missing = [
    c
    for c in required
    if c not in df.columns
]

if missing:
    raise RuntimeError(
        f"Missing columns: {missing}"
    )


# ============================================================
# VERIFY SOURCE FILE <-> ACQUISITION DATE
# ============================================================

file_date = (
    df.groupby("source_file")
      ["acquisition_date"]
      .nunique()
)

date_file = (
    df.groupby("acquisition_date")
      ["source_file"]
      .nunique()
)

one_date_per_file = bool(
    (file_date == 1).all()
)

one_file_per_date = bool(
    (date_file == 1).all()
)

if not one_date_per_file:
    raise RuntimeError(
        "A source_file maps to multiple "
        "acquisition dates."
    )

if not one_file_per_date:
    raise RuntimeError(
        "An acquisition date maps to multiple "
        "source files."
    )


# ============================================================
# TRACK GROUP MAPPING
# ============================================================

track_mapping = (
    df.groupby(
        [
            "source_file",
            "acquisition_date",
        ]
    )
    .agg(
        n=("shot_number", "size"),
        beams=("beam", "nunique"),
        blocks=("cv_block_id", "nunique"),
        x_min=("utm_x", "min"),
        x_max=("utm_x", "max"),
        y_min=("utm_y", "min"),
        y_max=("utm_y", "max"),
    )
    .reset_index()
    .sort_values(
        "acquisition_date"
    )
    .reset_index(drop=True)
)

track_mapping[
    "pass_id"
] = [
    f"PASS_{i:02d}"
    for i in range(
        1,
        len(track_mapping) + 1
    )
]

track_mapping.to_csv(
    TRACK_MAPPING,
    index=False
)


# ============================================================
# A. BUFFER SENSITIVITY
#
# Test sets = already-frozen C1 geographic outer folds.
# Only training exclusion distance changes.
# ============================================================

buffer_rows = []

all_ids = set(
    df["shot_number"]
)

for buffer_m in BUFFER_DISTANCES_M:

    this_dir = (
        BUFFER_DIR
        / f"Buffer_{buffer_m}m"
    )

    this_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for fold in range(
        1,
        6
    ):

        c1_test_path = (
            C1_FOLD_ROOT
            / f"Fold_{fold}"
            / f"C1_NEW_Fold_{fold}_TEST.csv"
        )

        if not c1_test_path.exists():
            raise FileNotFoundError(
                c1_test_path
            )

        c1_test = pd.read_csv(
            c1_test_path
        )

        test_ids = set(
            c1_test[
                "shot_number"
            ]
        )

        test = (
            df[
                df["shot_number"]
                .isin(test_ids)
            ]
            .copy()
        )

        candidate = (
            df[
                ~df["shot_number"]
                .isin(test_ids)
            ]
            .copy()
        )

        train, removed = (
            buffered_training(
                candidate,
                test,
                buffer_m,
            )
        )

        if (
            set(train["shot_number"])
            .intersection(
                set(test["shot_number"])
            )
        ):
            raise RuntimeError(
                "Train/test overlap."
            )

        min_distance = (
            float(
                train[
                    "nearest_test_distance_m"
                ].min()
            )
            if len(train)
            else np.nan
        )

        train_path = (
            this_dir
            / (
                f"Fold_{fold}_"
                f"TRAIN_MEMBERSHIP.csv"
            )
        )

        test_path = (
            this_dir
            / (
                f"Fold_{fold}_"
                f"TEST_MEMBERSHIP.csv"
            )
        )

        save_memberships(
            train,
            test,
            train_path,
            test_path,
        )

        buffer_rows.append({
            "buffer_m":
                buffer_m,

            "fold":
                fold,

            "test_n":
                len(test),

            "test_blocks":
                test[
                    "cv_block_id"
                ].nunique(),

            "candidate_train_n":
                len(candidate),

            "buffered_train_n":
                len(train),

            "removed_by_buffer_n":
                len(removed),

            "buffered_train_blocks":
                train[
                    "cv_block_id"
                ].nunique(),

            "min_train_test_distance_m":
                min_distance,

            "train_membership":
                str(train_path),

            "test_membership":
                str(test_path),
        })


buffer_summary = pd.DataFrame(
    buffer_rows
)

buffer_summary.to_csv(
    BUFFER_SUMMARY,
    index=False
)


# ============================================================
# B. LEAVE-ONE-GEDI-PASS / SOURCE-FILE OUT
#
# Entire granule/pass held out.
# 500-m buffer applied to remaining passes.
# ============================================================

pass_rows = []

for _, row in track_mapping.iterrows():

    pass_id = row[
        "pass_id"
    ]

    source_file = row[
        "source_file"
    ]

    acquisition_date = row[
        "acquisition_date"
    ]

    test = (
        df[
            df["source_file"]
            == source_file
        ]
        .copy()
    )

    candidate = (
        df[
            df["source_file"]
            != source_file
        ]
        .copy()
    )

    train, removed = (
        buffered_training(
            candidate,
            test,
            PASS_BUFFER_M,
        )
    )

    pass_dir = (
        PASS_DIR
        / pass_id
    )

    pass_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    train_path = (
        pass_dir
        / "TRAIN_MEMBERSHIP.csv"
    )

    test_path = (
        pass_dir
        / "TEST_MEMBERSHIP.csv"
    )

    save_memberships(
        train,
        test,
        train_path,
        test_path,
    )

    min_distance = (
        float(
            train[
                "nearest_test_distance_m"
            ].min()
        )
        if len(train)
        else np.nan
    )

    pass_rows.append({
        "pass_id":
            pass_id,

        "source_file":
            source_file,

        "acquisition_date":
            acquisition_date,

        "test_n":
            len(test),

        "test_beams":
            test["beam"].nunique(),

        "test_blocks":
            test["cv_block_id"].nunique(),

        "candidate_train_n":
            len(candidate),

        "buffered_train_n":
            len(train),

        "removed_by_500m_buffer_n":
            len(removed),

        "buffered_train_blocks":
            train["cv_block_id"].nunique(),

        "min_train_test_distance_m":
            min_distance,

        "low_power_test_n_lt_30":
            bool(len(test) < 30),

        "train_membership":
            str(train_path),

        "test_membership":
            str(test_path),
    })


pass_summary = pd.DataFrame(
    pass_rows
)

pass_summary.to_csv(
    PASS_SUMMARY,
    index=False
)


# ============================================================
# C. REPEATED SPATIAL ASSIGNMENTS
#
# 54 frozen 3-km blocks.
# K-means is based only on block centroids.
# No AGBD is used.
#
# 10 seeds x 5 folds.
# ============================================================

block_centroids = (
    df.groupby(
        "cv_block_id"
    )
    .agg(
        x=("utm_x", "mean"),
        y=("utm_y", "mean"),
        n=("shot_number", "size"),
    )
    .reset_index()
)

if len(block_centroids) != 54:
    raise RuntimeError(
        f"Expected 54 blocks; "
        f"found {len(block_centroids)}."
    )

assignment_parts = []

repeat_rows = []

for repeat_index, seed in enumerate(
    REPEAT_SEEDS,
    start=1
):

    X = block_centroids[
        ["x", "y"]
    ].to_numpy(float)

    km = KMeans(
        n_clusters=N_REPEAT_FOLDS,
        random_state=seed,
        n_init=50,
    )

    raw_labels = km.fit_predict(
        X
    )

    centers = km.cluster_centers_

    # Stable west-to-east ordering.
    order = sorted(
        range(N_REPEAT_FOLDS),
        key=lambda i: (
            centers[i, 0],
            centers[i, 1]
        )
    )

    label_map = {
        old: new + 1
        for new, old
        in enumerate(order)
    }

    bc = block_centroids.copy()

    bc[
        "repeat_index"
    ] = repeat_index

    bc[
        "seed"
    ] = seed

    bc[
        "outer_fold"
    ] = [
        label_map[x]
        for x in raw_labels
    ]

    assignment_parts.append(
        bc[
            [
                "repeat_index",
                "seed",
                "cv_block_id",
                "outer_fold",
                "x",
                "y",
                "n",
            ]
        ]
    )

    block_to_fold = dict(
        zip(
            bc["cv_block_id"],
            bc["outer_fold"]
        )
    )

    row_fold = (
        df["cv_block_id"]
        .map(block_to_fold)
    )

    repeat_dir = (
        REPEAT_DIR
        / f"Seed_{seed}"
    )

    repeat_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for fold in range(
        1,
        N_REPEAT_FOLDS + 1
    ):

        test = (
            df[
                row_fold == fold
            ]
            .copy()
        )

        candidate = (
            df[
                row_fold != fold
            ]
            .copy()
        )

        train, removed = (
            buffered_training(
                candidate,
                test,
                REPEAT_BUFFER_M,
            )
        )

        fold_dir = (
            repeat_dir
            / f"Fold_{fold}"
        )

        fold_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        train_path = (
            fold_dir
            / "TRAIN_MEMBERSHIP.csv"
        )

        test_path = (
            fold_dir
            / "TEST_MEMBERSHIP.csv"
        )

        save_memberships(
            train,
            test,
            train_path,
            test_path,
        )

        min_distance = (
            float(
                train[
                    "nearest_test_distance_m"
                ].min()
            )
            if len(train)
            else np.nan
        )

        repeat_rows.append({
            "repeat_index":
                repeat_index,

            "seed":
                seed,

            "fold":
                fold,

            "test_n":
                len(test),

            "test_blocks":
                test[
                    "cv_block_id"
                ].nunique(),

            "candidate_train_n":
                len(candidate),

            "buffered_train_n":
                len(train),

            "removed_by_500m_buffer_n":
                len(removed),

            "buffered_train_blocks":
                train[
                    "cv_block_id"
                ].nunique(),

            "min_train_test_distance_m":
                min_distance,

            "train_membership":
                str(train_path),

            "test_membership":
                str(test_path),
        })


repeat_assignments = pd.concat(
    assignment_parts,
    ignore_index=True
)

repeat_assignments.to_csv(
    REPEAT_ASSIGNMENTS,
    index=False
)

repeat_summary = pd.DataFrame(
    repeat_rows
)

repeat_summary.to_csv(
    REPEAT_SUMMARY,
    index=False
)


# ============================================================
# D. EFFECTIVE INDEPENDENT-UNIT COUNTS
# ============================================================

date_beam_key = (
    df["acquisition_date"]
    .astype(str)
    + "__"
    + df["beam"].astype(str)
)

effective_rows = [
    effective_unit_row(
        "3-km spatial blocks (cv_block_id)",
        df["cv_block_id"]
    ),

    effective_unit_row(
        "GEDI acquisition/source-file passes",
        df["source_file"]
    ),

    effective_unit_row(
        "GEDI acquisition-date x beam groups",
        date_beam_key
    ),

    effective_unit_row(
        "GEDI beam only",
        df["beam"]
    ),
]

effective_df = pd.DataFrame(
    effective_rows
)

effective_df.to_csv(
    EFFECTIVE_UNITS,
    index=False
)


# ============================================================
# LOCK PROTOCOL
# ============================================================

protocol = {

    "created_before_any_C2_model_fit":
        True,

    "input_table":
        str(DATA),

    "input_sha256":
        sha256_file(DATA),

    "n_observations":
        len(df),

    "c1_status":
        "FROZEN_CLOSED",

    "c2_model_family":
        C2_MODEL_FAMILY,

    "c2_feature_family":
        C2_FEATURE_FAMILY,

    "c2_predictors":
        C2_FEATURES,

    "model_random_seed":
        RANDOM_SEED_MODEL,

    "inner_cv":
        INNER_CV,

    "buffer_sensitivity_m":
        BUFFER_DISTANCES_M,

    "buffer_test_geometry":
        "Frozen C1 geographic outer-fold test sets",

    "pass_holdout_definition":
        "Leave one complete GEDI source_file/acquisition pass out",

    "pass_holdout_buffer_m":
        PASS_BUFFER_M,

    "repeat_spatial_seeds":
        REPEAT_SEEDS,

    "repeat_spatial_folds":
        N_REPEAT_FOLDS,

    "repeat_spatial_buffer_m":
        REPEAT_BUFFER_M,

    "repeat_split_method":
        (
            "KMeans on 54 3-km block centroids only; "
            "no AGBD balancing; labels stabilized west-to-east"
        ),

    "outer_test_data_used_for_split_selection":
        False,

    "agbd_used_to_define_C2_geometry":
        False,

    "date_beam_groups_modelled_as_primary_track_holdout":
        False,

    "reason_date_beam_secondary_only":
        (
            "40 groups are highly unbalanced and include groups with n=1; "
            "whole-source-file/pass holdout is stricter."
        ),
}

PROTOCOL_JSON.write_text(
    json.dumps(
        protocol,
        indent=2
    ),
    encoding="utf-8"
)


# ============================================================
# REPORT
# ============================================================

with REPORT.open(
    "w",
    encoding="utf-8"
) as f:

    f.write("=" * 94 + "\n")
    f.write(
        "C2 STEP 1 — FROZEN SPATIAL TRANSFERABILITY PROTOCOL\n"
    )
    f.write("=" * 94 + "\n\n")

    f.write("STATUS\n")
    f.write("-" * 94 + "\n")
    f.write(
        "NO MODEL WAS FITTED.\n"
        "C1 remains frozen and unchanged.\n"
        "All C2 validation geometries were defined before C2 performance was observed.\n\n"
    )

    f.write("INPUT\n")
    f.write("-" * 94 + "\n")
    f.write(f"{DATA}\n")
    f.write(
        f"SHA256: {sha256_file(DATA)}\n"
    )
    f.write(
        f"Observations: {len(df)}\n"
    )
    f.write(
        f"3-km blocks: {df['cv_block_id'].nunique()}\n\n"
    )

    f.write("FROZEN C2 MODEL\n")
    f.write("-" * 94 + "\n")
    f.write(
        "Model family: Extra Trees\n"
    )
    f.write(
        "Predictor family: full 14P multiscale set\n"
    )
    f.write(
        "Hyperparameters will be retuned only inside each active C2 training sample.\n"
    )
    f.write(
        "Inner validation: 4-fold GroupKFold by cv_block_id.\n"
    )
    f.write(
        "No new model-family or feature-family selection is permitted in C2.\n\n"
    )

    f.write("TEST A — BUFFER SENSITIVITY\n")
    f.write("-" * 94 + "\n")
    f.write(
        "Frozen C1 geographic test folds retained exactly.\n"
    )
    f.write(
        "Training exclusion distances: 500, 1000, 2000, 3000 m.\n\n"
    )

    f.write(
        buffer_summary.to_string(
            index=False
        )
    )
    f.write("\n\n")

    f.write("TEST B — LEAVE-ONE-PASS-OUT\n")
    f.write("-" * 94 + "\n")
    f.write(
        "Primary track group = complete GEDI source_file/acquisition pass.\n"
    )
    f.write(
        f"Unique passes: {df['source_file'].nunique()}.\n"
    )
    f.write(
        "source_file and acquisition_date are one-to-one: "
        f"{one_date_per_file and one_file_per_date}.\n"
    )
    f.write(
        "500-m spatial training exclusion buffer retained.\n\n"
    )

    f.write(
        pass_summary.to_string(
            index=False
        )
    )
    f.write("\n\n")

    f.write("TEST C — REPEATED SPATIAL ASSIGNMENTS\n")
    f.write("-" * 94 + "\n")
    f.write(
        f"Repeated seeds: {REPEAT_SEEDS}\n"
    )
    f.write(
        "Each repeat uses five geographic clusters of the 54 block centroids.\n"
    )
    f.write(
        "No response balancing or AGBD information was used.\n"
    )
    f.write(
        "Each active fold uses a 500-m training exclusion buffer.\n\n"
    )

    repeat_compact = (
        repeat_summary
        .groupby(
            ["repeat_index", "seed"]
        )
        .agg(
            min_test_n=("test_n", "min"),
            max_test_n=("test_n", "max"),
            min_train_n=("buffered_train_n", "min"),
            max_train_n=("buffered_train_n", "max"),
            min_distance_m=(
                "min_train_test_distance_m",
                "min"
            ),
        )
        .reset_index()
    )

    f.write(
        repeat_compact.to_string(
            index=False
        )
    )
    f.write("\n\n")

    f.write("EFFECTIVE INDEPENDENT-UNIT COUNTS\n")
    f.write("-" * 94 + "\n")
    f.write(
        effective_df.to_string(
            index=False
        )
    )
    f.write("\n\n")

    f.write("TRACK-GROUP DECISION\n")
    f.write("-" * 94 + "\n")
    f.write(
        "The primary leave-track-group-out test uses whole source-file/acquisition passes.\n"
    )
    f.write(
        "This is stricter than leaving individual date×beam fragments out because all beams\n"
        "from a GEDI acquisition pass are removed simultaneously. Date×beam structure is\n"
        "retained as a secondary independent-unit diagnostic.\n\n"
    )

    f.write("LOCK\n")
    f.write("-" * 94 + "\n")
    f.write(
        "Do not change buffers, pass definitions, repeated seeds, folds, predictor set,\n"
        "or model family after C2 model performance is observed.\n"
    )


# ============================================================
# CONSOLE
# ============================================================

print("")
print("=" * 80)
print("C2 STEP 1 COMPLETE")
print("ALL C2 VALIDATION GEOMETRIES FROZEN")
print("=" * 80)

print("")
print("BUFFER SENSITIVITY SUMMARY")
print(
    buffer_summary[
        [
            "buffer_m",
            "fold",
            "test_n",
            "buffered_train_n",
            "removed_by_buffer_n",
            "buffered_train_blocks",
            "min_train_test_distance_m",
        ]
    ].to_string(
        index=False
    )
)

print("")
print("PASS HOLDOUT SUMMARY")
print(
    pass_summary[
        [
            "pass_id",
            "acquisition_date",
            "test_n",
            "test_beams",
            "test_blocks",
            "buffered_train_n",
            "removed_by_500m_buffer_n",
            "min_train_test_distance_m",
            "low_power_test_n_lt_30",
        ]
    ].to_string(
        index=False
    )
)

print("")
print("EFFECTIVE INDEPENDENT-UNIT COUNTS")
print(
    effective_df.to_string(
        index=False
    )
)

print("")
print("REPEATED SPATIAL SEEDS:")
print(REPEAT_SEEDS)

print("")
print("Repeated split rows:")
print(len(repeat_summary))

print("")
print("Report:")
print(REPORT)

print("")
print("NO MODEL WAS FITTED.")
print("C1 REMAINS FROZEN.")
print("C2 GEOMETRIES ARE NOW FROZEN.")
print("=" * 80)
