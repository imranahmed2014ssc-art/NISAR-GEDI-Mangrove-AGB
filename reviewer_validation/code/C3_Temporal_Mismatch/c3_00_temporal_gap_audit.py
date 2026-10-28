from pathlib import Path
import hashlib

import numpy as np
import pandas as pd


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
    / "C3_Temporal_Mismatch"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

REPORT = (
    OUT
    / "C3_00_TEMPORAL_GAP_AUDIT_REPORT.txt"
)

PASS_TABLE = (
    OUT
    / "C3_00_ACQUISITION_PASS_GAPS.csv"
)

STRATUM_TABLE = (
    OUT
    / "C3_00_GAP_STRATA.csv"
)

SHOT_MANIFEST = (
    OUT
    / "C3_00_SHOT_GAP_STRATUM_MANIFEST.csv"
)

FILE_INVENTORY = (
    OUT
    / "C3_00_EXISTING_TEMPORAL_FILE_INVENTORY.csv"
)


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


# ============================================================
# LOAD FROZEN MASTER
# ============================================================

df = pd.read_csv(
    DATA
)

if len(df) != 1641:

    raise RuntimeError(
        f"Expected 1641 rows; "
        f"found {len(df)}."
    )

if (
    df[
        "shot_number"
    ].nunique()
    != 1641
):

    raise RuntimeError(
        "shot_number is not unique."
    )


required = [
    "shot_number",
    "source_file",
    "acquisition_date",
    "gedi_datetime",
    "nisar_datetime",
    "delta_days_to_NISAR",
    "cv_block_id",
    "utm_x",
    "utm_y",
]

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
# PARSE DATES
# ============================================================

df[
    "_gedi_dt"
] = pd.to_datetime(
    df[
        "gedi_datetime"
    ],
    utc=True,
    errors="coerce",
)

df[
    "_nisar_dt"
] = pd.to_datetime(
    df[
        "nisar_datetime"
    ],
    utc=True,
    errors="coerce",
)

if (
    df[
        "_gedi_dt"
    ].isna().any()
):

    raise RuntimeError(
        "Some GEDI datetimes could "
        "not be parsed."
    )

if (
    df[
        "_nisar_dt"
    ].isna().any()
):

    raise RuntimeError(
        "Some NISAR datetimes could "
        "not be parsed."
    )


# ============================================================
# VERIFY STORED GAP
# ============================================================

df[
    "_computed_gap_days"
] = (
    (
        df[
            "_nisar_dt"
        ]
        -
        df[
            "_gedi_dt"
        ]
    )
    .dt.total_seconds()
    / 86400.0
)

df[
    "_gap_abs_difference"
] = np.abs(
    df[
        "_computed_gap_days"
    ]
    -
    df[
        "delta_days_to_NISAR"
    ]
)

max_gap_difference = float(
    df[
        "_gap_abs_difference"
    ].max()
)

if max_gap_difference > 1e-6:

    raise RuntimeError(
        "Stored delta_days_to_NISAR "
        "does not agree with parsed dates. "
        f"Maximum difference = "
        f"{max_gap_difference}"
    )


# ============================================================
# VERIFY ONE SOURCE FILE PER ACQUISITION
# ============================================================

file_date_counts = (
    df.groupby(
        "source_file"
    )[
        "acquisition_date"
    ]
    .nunique()
)

date_file_counts = (
    df.groupby(
        "acquisition_date"
    )[
        "source_file"
    ]
    .nunique()
)

if not (
    file_date_counts == 1
).all():

    raise RuntimeError(
        "A source file maps to more "
        "than one acquisition date."
    )

if not (
    date_file_counts == 1
).all():

    raise RuntimeError(
        "An acquisition date maps to more "
        "than one source file."
    )


# ============================================================
# PASS-LEVEL GAP TABLE
# ============================================================

pass_table = (
    df.groupby(
        [
            "source_file",
            "acquisition_date",
        ]
    )
    .agg(
        n=(
            "shot_number",
            "size"
        ),

        blocks=(
            "cv_block_id",
            "nunique"
        ),

        gedi_datetime=(
            "_gedi_dt",
            "first"
        ),

        nisar_datetime=(
            "_nisar_dt",
            "first"
        ),

        gap_days_min=(
            "delta_days_to_NISAR",
            "min"
        ),

        gap_days_max=(
            "delta_days_to_NISAR",
            "max"
        ),

        gap_days_median=(
            "delta_days_to_NISAR",
            "median"
        ),
    )
    .reset_index()
)

if len(
    pass_table
) != 8:

    raise RuntimeError(
        f"Expected 8 GEDI acquisition "
        f"passes; found "
        f"{len(pass_table)}."
    )


# Every observation within an acquisition pass should
# have effectively the same temporal gap.
pass_table[
    "within_pass_gap_range_days"
] = (
    pass_table[
        "gap_days_max"
    ]
    -
    pass_table[
        "gap_days_min"
    ]
)


# ============================================================
# SORT SHORTEST TO LONGEST TEMPORAL GAP
# ============================================================

pass_table = (
    pass_table
    .sort_values(
        [
            "gap_days_median",
            "acquisition_date",
        ]
    )
    .reset_index(
        drop=True
    )
)

pass_table[
    "gap_rank_shortest_first"
] = (
    np.arange(
        1,
        len(pass_table) + 1
    )
)


# ============================================================
# PREDECLARE FOUR TEMPORAL-GAP STRATA
#
# Exactly two complete acquisition passes per stratum.
#
# This uses temporal distance ONLY.
# AGBD and model performance are not used.
# ============================================================

stratum_labels = [
    "G1_SHORTEST",
    "G1_SHORTEST",
    "G2_SHORT_MID",
    "G2_SHORT_MID",
    "G3_LONG_MID",
    "G3_LONG_MID",
    "G4_LONGEST",
    "G4_LONGEST",
]

pass_table[
    "gap_stratum"
] = stratum_labels

pass_table[
    "shortest_gap_subset"
] = (
    pass_table[
        "gap_stratum"
    ]
    == "G1_SHORTEST"
)


# ============================================================
# STRATUM SUMMARY
# ============================================================

stratum_table = (
    pass_table.groupby(
        "gap_stratum",
        sort=False
    )
    .agg(
        passes=(
            "source_file",
            "size"
        ),

        observations=(
            "n",
            "sum"
        ),

        blocks_min=(
            "blocks",
            "min"
        ),

        blocks_max=(
            "blocks",
            "max"
        ),

        gap_days_min=(
            "gap_days_min",
            "min"
        ),

        gap_days_max=(
            "gap_days_max",
            "max"
        ),

        gap_days_median_of_passes=(
            "gap_days_median",
            "median"
        ),
    )
    .reset_index()
)


# ============================================================
# SHOT-LEVEL FROZEN MANIFEST
# ============================================================

mapping = (
    pass_table[
        [
            "source_file",
            "gap_rank_shortest_first",
            "gap_stratum",
            "shortest_gap_subset",
        ]
    ]
)

shot_manifest = (
    df.merge(
        mapping,
        on="source_file",
        how="left",
        validate="many_to_one",
    )
)

if (
    shot_manifest[
        "gap_stratum"
    ].isna().any()
):

    raise RuntimeError(
        "Some shots were not assigned "
        "to a temporal stratum."
    )


shot_manifest[
    [
        "shot_number",
        "source_file",
        "acquisition_date",
        "gedi_datetime",
        "nisar_datetime",
        "delta_days_to_NISAR",
        "gap_rank_shortest_first",
        "gap_stratum",
        "shortest_gap_subset",
        "cv_block_id",
        "utm_x",
        "utm_y",
    ]
].to_csv(
    SHOT_MANIFEST,
    index=False
)


pass_table.to_csv(
    PASS_TABLE,
    index=False
)

stratum_table.to_csv(
    STRATUM_TABLE,
    index=False
)


# ============================================================
# INVENTORY EXISTING TEMPORAL / JUNE FILES
#
# READ-ONLY inventory.
# Do not process new NISAR here.
# ============================================================

scan_roots = [
    ROOT,
    Path(r"D:\new_nisar_data"),
]

keywords = [
    "june",
    "temporal",
    "20260625",
    "2026-06-25",
    "0625",
    "sensitivity",
]

allowed_suffixes = {
    ".csv",
    ".txt",
    ".json",
    ".parquet",
    ".gpkg",
    ".tif",
    ".tiff",
    ".h5",
    ".hdf5",
    ".py",
}

inventory_rows = []

for scan_root in scan_roots:

    if not scan_root.exists():

        inventory_rows.append({
            "scan_root":
                str(scan_root),

            "path":
                "",

            "suffix":
                "",

            "size_bytes":
                np.nan,

            "matched_keyword":
                "",

            "status":
                "ROOT_NOT_FOUND",
        })

        continue

    for path in scan_root.rglob("*"):

        if not path.is_file():

            continue

        if (
            path.suffix.lower()
            not in allowed_suffixes
        ):

            continue

        low_name = str(
            path
        ).lower()

        matched = [
            k
            for k in keywords
            if k in low_name
        ]

        if not matched:

            continue

        try:

            size = (
                path.stat()
                .st_size
            )

        except Exception:

            size = np.nan

        inventory_rows.append({
            "scan_root":
                str(scan_root),

            "path":
                str(path),

            "suffix":
                path.suffix.lower(),

            "size_bytes":
                size,

            "matched_keyword":
                " | ".join(
                    matched
                ),

            "status":
                "FOUND",
        })


inventory = pd.DataFrame(
    inventory_rows
)

inventory.to_csv(
    FILE_INVENTORY,
    index=False
)


# ============================================================
# REPORT
# ============================================================

with REPORT.open(
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "=" * 96
        + "\n"
    )

    f.write(
        "C3 STEP 0 — TEMPORAL-GAP AUDIT "
        "AND PRE-MODEL STRATIFICATION\n"
    )

    f.write(
        "=" * 96
        + "\n\n"
    )

    f.write(
        "STATUS\n"
    )

    f.write(
        "-" * 96
        + "\n"
    )

    f.write(
        "NO MODEL WAS FITTED.\n"
    )

    f.write(
        "C1 and C2 remain frozen.\n"
    )

    f.write(
        "Temporal strata were defined using "
        "GEDI-to-primary-NISAR time gap only.\n"
    )

    f.write(
        "AGBD and model performance were NOT "
        "used to define strata.\n\n"
    )

    f.write(
        "INPUT\n"
    )

    f.write(
        "-" * 96
        + "\n"
    )

    f.write(
        f"{DATA}\n"
    )

    f.write(
        f"SHA256: "
        f"{sha256_file(DATA)}\n"
    )

    f.write(
        f"Rows: "
        f"{len(df)}\n"
    )

    f.write(
        f"Unique acquisition passes: "
        f"{len(pass_table)}\n"
    )

    f.write(
        f"Maximum stored-vs-computed "
        f"gap difference: "
        f"{max_gap_difference:.12g} days\n\n"
    )

    f.write(
        "PRIMARY SEPTEMBER-2026 GAP RANGE\n"
    )

    f.write(
        "-" * 96
        + "\n"
    )

    f.write(
        f"Minimum: "
        f"{df['delta_days_to_NISAR'].min():.6f} days\n"
    )

    f.write(
        f"Median: "
        f"{df['delta_days_to_NISAR'].median():.6f} days\n"
    )

    f.write(
        f"Maximum: "
        f"{df['delta_days_to_NISAR'].max():.6f} days\n\n"
    )

    f.write(
        "ACQUISITION PASSES — "
        "SHORTEST GAP FIRST\n"
    )

    f.write(
        "-" * 96
        + "\n"
    )

    f.write(
        pass_table[
            [
                "gap_rank_shortest_first",
                "gap_stratum",
                "acquisition_date",
                "n",
                "blocks",
                "gap_days_median",
                "shortest_gap_subset",
                "source_file",
            ]
        ].to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )

    f.write(
        "PREDECLARED TEMPORAL-GAP STRATA\n"
    )

    f.write(
        "-" * 96
        + "\n"
    )

    f.write(
        stratum_table.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )

    f.write(
        "SHORTEST-GAP SUBSET\n"
    )

    f.write(
        "-" * 96
        + "\n"
    )

    shortest = (
        pass_table[
            pass_table[
                "shortest_gap_subset"
            ]
        ]
    )

    f.write(
        shortest[
            [
                "acquisition_date",
                "n",
                "blocks",
                "gap_days_median",
                "source_file",
            ]
        ].to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )

    f.write(
        "EXISTING JUNE / TEMPORAL FILE INVENTORY\n"
    )

    f.write(
        "-" * 96
        + "\n"
    )

    if len(inventory):

        f.write(
            inventory.to_string(
                index=False
            )
        )

    else:

        f.write(
            "No filename matches found."
        )

    f.write(
        "\n\n"
    )

    f.write(
        "C3 NEXT STEPS\n"
    )

    f.write(
        "-" * 96
        + "\n"
    )

    f.write(
        "1. Evaluate predictive skill by the "
        "four frozen acquisition-gap strata.\n"
    )

    f.write(
        "2. Evaluate the frozen shortest-gap "
        "subset separately.\n"
    )

    f.write(
        "3. Audit disturbance / land-cover "
        "stability between GEDI and NISAR dates.\n"
    )

    f.write(
        "4. Evaluate season/hydrology sensitivity.\n"
    )

    f.write(
        "5. Reconcile the existing June-2026 "
        "NISAR temporal-sensitivity experiment "
        "without mixing its workspace with "
        "the primary workflow.\n"
    )

    f.write(
        "6. Contemporary field/airborne "
        "reference data are unavailable and "
        "will not be invented.\n"
    )


# ============================================================
# CONSOLE
# ============================================================

print("")
print(
    "=" * 80
)

print(
    "C3 STEP 0 COMPLETE"
)

print(
    "TEMPORAL-GAP AUDIT"
)

print(
    "=" * 80
)

print("")
print(
    "Primary September NISAR gap:"
)

print(
    "  min   =",
    f"{df['delta_days_to_NISAR'].min():.3f}",
    "days"
)

print(
    "  median=",
    f"{df['delta_days_to_NISAR'].median():.3f}",
    "days"
)

print(
    "  max   =",
    f"{df['delta_days_to_NISAR'].max():.3f}",
    "days"
)

print("")
print(
    "Passes shortest-gap first:"
)

print(
    pass_table[
        [
            "gap_rank_shortest_first",
            "gap_stratum",
            "acquisition_date",
            "n",
            "blocks",
            "gap_days_median",
        ]
    ].to_string(
        index=False
    )
)

print("")
print(
    "Frozen strata:"
)

print(
    stratum_table.to_string(
        index=False
    )
)

print("")
print(
    "Temporal/JUNE inventory rows:",
    len(inventory)
)

if len(inventory):

    print(
        inventory.to_string(
            index=False
        )
    )

print("")
print(
    "Report:"
)

print(
    REPORT
)

print("")
print(
    "NO MODEL WAS FITTED."
)

print(
    "C1 AND C2 REMAIN FROZEN."
)

print(
    "=" * 80
)
