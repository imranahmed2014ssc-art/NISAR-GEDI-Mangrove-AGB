from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


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

OUT = (
    ROOT
    / "06_Results"
    / "Reviewer_Validation"
    / "C2_Spatial_Transferability"
)

GEOM = (
    OUT
    / "Frozen_Geometries"
    / "Random_Block_Assignments"
)

GEOM.mkdir(
    parents=True,
    exist_ok=True
)

SUMMARY_FILE = (
    OUT
    / "C2_01B_RANDOM_BLOCK_SPLIT_SUMMARY.csv"
)

ASSIGNMENT_FILE = (
    OUT
    / "C2_01B_RANDOM_BLOCK_ASSIGNMENTS.csv"
)

HASH_FILE = (
    OUT
    / "C2_01B_ASSIGNMENT_HASHES.csv"
)

REPORT = (
    OUT
    / "C2_01B_RANDOM_BLOCK_PROTOCOL_REPORT.txt"
)


# ============================================================
# FROZEN RULES
# ============================================================

SEEDS = [
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

N_FOLDS = 5

BUFFER_M = 500


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        for chunk in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b""
        ):
            h.update(chunk)

    return h.hexdigest()


def text_hash(text):

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def xy(df):

    arr = (
        df[
            ["utm_x", "utm_y"]
        ]
        .to_numpy(float)
    )

    if not np.isfinite(
        arr
    ).all():

        raise RuntimeError(
            "Non-finite coordinates."
        )

    return arr


def apply_buffer(
    candidate,
    test,
    distance_m
):

    tree = cKDTree(
        xy(test)
    )

    dist, _ = tree.query(
        xy(candidate),
        k=1
    )

    out = candidate.copy()

    out[
        "nearest_test_distance_m"
    ] = dist

    kept = (
        out[
            out[
                "nearest_test_distance_m"
            ] >= distance_m
        ]
        .copy()
    )

    removed = (
        out[
            out[
                "nearest_test_distance_m"
            ] < distance_m
        ]
        .copy()
    )

    return (
        kept,
        removed
    )


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    DATA
)

if len(df) != 1641:

    raise RuntimeError(
        f"Expected 1641 rows, "
        f"found {len(df)}."
    )

if (
    df[
        "shot_number"
    ].nunique()
    != 1641
):

    raise RuntimeError(
        "shot_number not unique."
    )

if (
    df[
        "cv_block_id"
    ].nunique()
    != 54
):

    raise RuntimeError(
        "Expected 54 spatial blocks."
    )


# ============================================================
# BLOCK INVENTORY
# ============================================================

blocks = (
    df.groupby(
        "cv_block_id"
    )
    .agg(
        n=(
            "shot_number",
            "size"
        ),
        x=(
            "utm_x",
            "mean"
        ),
        y=(
            "utm_y",
            "mean"
        ),
    )
    .reset_index()
)

# Sort from largest to smallest.
# Assignment does NOT use AGBD.
blocks = (
    blocks
    .sort_values(
        [
            "n",
            "cv_block_id",
        ],
        ascending=[
            False,
            True,
        ]
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# RANDOMIZED SIZE-STRATIFIED BLOCK ASSIGNMENT
#
# Blocks are handled in groups of five according to block size.
# Inside each group, both block order and fold labels are
# randomized.
#
# Therefore:
# - no AGBD is used;
# - each fold receives a similar mixture of large/small blocks;
# - assignments vary genuinely by seed;
# - each spatial block always remains intact.
# ============================================================

assignment_parts = []

summary_rows = []

hash_rows = []


for repeat_index, seed in enumerate(
    SEEDS,
    start=1
):

    rng = np.random.default_rng(
        seed
    )

    assignment = {}

    for start in range(
        0,
        len(blocks),
        N_FOLDS
    ):

        batch = (
            blocks.iloc[
                start:
                start + N_FOLDS
            ]
            .copy()
        )

        block_ids = (
            batch[
                "cv_block_id"
            ]
            .to_numpy()
            .copy()
        )

        rng.shuffle(
            block_ids
        )

        folds = np.arange(
            1,
            N_FOLDS + 1
        )

        rng.shuffle(
            folds
        )

        # Final batch may contain <5 blocks.
        folds = folds[
            :len(block_ids)
        ]

        for block_id, fold in zip(
            block_ids,
            folds
        ):
            assignment[
                block_id
            ] = int(fold)

    if len(assignment) != 54:

        raise RuntimeError(
            f"Seed {seed}: "
            f"not all blocks assigned."
        )

    assign_df = (
        blocks.copy()
    )

    assign_df[
        "repeat_index"
    ] = repeat_index

    assign_df[
        "seed"
    ] = seed

    assign_df[
        "outer_fold"
    ] = (
        assign_df[
            "cv_block_id"
        ]
        .map(
            assignment
        )
        .astype(int)
    )

    assignment_parts.append(
        assign_df[
            [
                "repeat_index",
                "seed",
                "cv_block_id",
                "outer_fold",
                "n",
                "x",
                "y",
            ]
        ]
    )

    canonical = (
        assign_df[
            [
                "cv_block_id",
                "outer_fold",
            ]
        ]
        .sort_values(
            "cv_block_id"
        )
    )

    canonical_text = "\n".join(
        f"{row.cv_block_id}|{row.outer_fold}"
        for row in canonical.itertuples()
    )

    hash_rows.append({
        "repeat_index":
            repeat_index,

        "seed":
            seed,

        "assignment_sha256":
            text_hash(
                canonical_text
            ),
    })

    # --------------------------------------------------------
    # CREATE FIVE FOLDS
    # --------------------------------------------------------

    block_to_fold = dict(
        zip(
            assign_df[
                "cv_block_id"
            ],
            assign_df[
                "outer_fold"
            ]
        )
    )

    row_fold = (
        df[
            "cv_block_id"
        ]
        .map(
            block_to_fold
        )
    )

    seed_dir = (
        GEOM
        / f"Seed_{seed}"
    )

    seed_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for fold in range(
        1,
        N_FOLDS + 1
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
            apply_buffer(
                candidate,
                test,
                BUFFER_M,
            )
        )

        if (
            set(
                train[
                    "shot_number"
                ]
            )
            .intersection(
                set(
                    test[
                        "shot_number"
                    ]
                )
            )
        ):

            raise RuntimeError(
                "Train/test overlap."
            )

        fold_dir = (
            seed_dir
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

        train[
            [
                "shot_number",
                "cv_block_id",
                "utm_x",
                "utm_y",
                "nearest_test_distance_m",
            ]
        ].to_csv(
            train_path,
            index=False
        )

        test[
            [
                "shot_number",
                "cv_block_id",
                "utm_x",
                "utm_y",
            ]
        ].to_csv(
            test_path,
            index=False
        )

        summary_rows.append({

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

            "buffered_train_blocks":
                train[
                    "cv_block_id"
                ].nunique(),

            "removed_by_500m_buffer_n":
                len(removed),

            "min_train_test_distance_m":
                float(
                    train[
                        "nearest_test_distance_m"
                    ].min()
                ),

            "train_membership":
                str(train_path),

            "test_membership":
                str(test_path),
        })


# ============================================================
# SAVE
# ============================================================

assignments = pd.concat(
    assignment_parts,
    ignore_index=True
)

summary = pd.DataFrame(
    summary_rows
)

hashes = pd.DataFrame(
    hash_rows
)

assignments.to_csv(
    ASSIGNMENT_FILE,
    index=False
)

summary.to_csv(
    SUMMARY_FILE,
    index=False
)

hashes.to_csv(
    HASH_FILE,
    index=False
)


# ============================================================
# VERIFY DISTINCT ASSIGNMENTS
# ============================================================

unique_hashes = (
    hashes[
        "assignment_sha256"
    ]
    .nunique()
)

if unique_hashes != len(
    SEEDS
):

    raise RuntimeError(
        f"Expected {len(SEEDS)} "
        f"distinct assignments, "
        f"found {unique_hashes}."
    )


# ============================================================
# COMPACT SUMMARY
# ============================================================

compact = (
    summary.groupby(
        [
            "repeat_index",
            "seed",
        ]
    )
    .agg(
        min_test_n=(
            "test_n",
            "min"
        ),
        max_test_n=(
            "test_n",
            "max"
        ),
        min_test_blocks=(
            "test_blocks",
            "min"
        ),
        max_test_blocks=(
            "test_blocks",
            "max"
        ),
        min_train_n=(
            "buffered_train_n",
            "min"
        ),
        max_train_n=(
            "buffered_train_n",
            "max"
        ),
        min_train_blocks=(
            "buffered_train_blocks",
            "min"
        ),
        min_distance_m=(
            "min_train_test_distance_m",
            "min"
        ),
    )
    .reset_index()
)


# ============================================================
# REPORT
# ============================================================

with REPORT.open(
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "=" * 94
        + "\n"
    )

    f.write(
        "C2 STEP 1B — RANDOMIZED "
        "SPATIAL-BLOCK ASSIGNMENT PROTOCOL\n"
    )

    f.write(
        "=" * 94
        + "\n\n"
    )

    f.write(
        "STATUS\n"
    )

    f.write(
        "-" * 94
        + "\n"
    )

    f.write(
        "This protocol amendment was created "
        "BEFORE any C2 model performance was observed.\n"
    )

    f.write(
        "Existing contiguous KMeans geometries "
        "remain unchanged and frozen.\n"
    )

    f.write(
        "This new component addresses sensitivity "
        "to repeated block-to-fold allocation.\n"
    )

    f.write(
        "NO MODEL WAS FITTED.\n"
        "NO C1 RESULT WAS MODIFIED.\n\n"
    )

    f.write(
        "METHOD\n"
    )

    f.write(
        "-" * 94
        + "\n"
    )

    f.write(
        "54 pre-existing 3-km spatial blocks "
        "remain indivisible units.\n"
    )

    f.write(
        "Blocks are ranked by sample count and "
        "processed in batches of five.\n"
    )

    f.write(
        "Within each batch, block order and fold "
        "labels are randomized using the locked seed.\n"
    )

    f.write(
        "This approximately balances block sampling "
        "density without using AGBD or predictor values.\n"
    )

    f.write(
        "A 500-m train-test exclusion buffer is "
        "applied after each assignment.\n\n"
    )

    f.write(
        f"Seeds: {SEEDS}\n"
    )

    f.write(
        f"Unique assignment hashes: "
        f"{unique_hashes}\n\n"
    )

    f.write(
        "COMPACT GEOMETRY SUMMARY\n"
    )

    f.write(
        "-" * 94
        + "\n"
    )

    f.write(
        compact.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )

    f.write(
        "ASSIGNMENT HASHES\n"
    )

    f.write(
        "-" * 94
        + "\n"
    )

    f.write(
        hashes.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )

    f.write(
        "LOCK\n"
    )

    f.write(
        "-" * 94
        + "\n"
    )

    f.write(
        "These ten randomized assignments are now "
        "frozen. Do not regenerate them based on "
        "subsequent model performance.\n"
    )


# ============================================================
# CONSOLE
# ============================================================

print("")
print("=" * 80)
print("C2 STEP 1B COMPLETE")
print("RANDOM BLOCK ASSIGNMENTS FROZEN")
print("=" * 80)

print("")
print(
    "Unique assignment hashes:",
    unique_hashes
)

print("")
print(
    compact.to_string(
        index=False
    )
)

print("")
print("Assignment hashes:")
print(
    hashes.to_string(
        index=False
    )
)

print("")
print("Report:")
print(REPORT)

print("")
print("NO MODEL WAS FITTED.")
print("C1 REMAINS FROZEN.")
print(
    "EXISTING C2 GEOMETRIES "
    "REMAIN UNCHANGED."
)
print("=" * 80)
