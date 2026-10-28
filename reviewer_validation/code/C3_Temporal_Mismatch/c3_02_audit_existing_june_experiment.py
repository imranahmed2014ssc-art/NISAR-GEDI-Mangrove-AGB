from pathlib import Path
import hashlib
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(r"D:\NISAR_Mangrove_Paper")
JUNE_ROOT = Path(r"D:\new_nisar_data")

MASTER = (
    ROOT
    / "04_Matched_Data"
    / "Predictor_Expansion"
    / "Phase8C2"
    / "Tables"
    / "GEDI_NISAR_expanded_predictors_1641.csv"
)

BASE = (
    JUNE_ROOT
    / "Phase8B2_June25_Sensitivity_V2"
)

MATCH_DIR = (
    BASE
    / "02_GEDI_Matching"
)

MODEL_DIR = (
    BASE
    / "03_Temporal_Model_Sensitivity"
)

QC_REPORT = (
    MODEL_DIR
    / "QC"
    / "JUNE25_LOCKED_MODEL_TEMPORAL_SENSITIVITY_REPORT.txt"
)

TABLE_DIR = (
    MODEL_DIR
    / "Tables"
)

JUNE_MATCH = (
    MATCH_DIR
    / "Tables"
    / "GEDI_June25_matched_all_1641.csv"
)

REFERENCE_SNAPSHOT = (
    MATCH_DIR
    / "Reference_Snapshot"
    / "GEDI_NISAR_modeling_table_with_spatial_folds_READONLY_COPY.csv"
)

FILES = {
    "frozen_fold_membership":
        TABLE_DIR
        / "frozen_fold_membership_check.csv",

    "june_fold_metrics":
        TABLE_DIR
        / "June25_locked_models_fold_metrics.csv",

    "june_oof":
        TABLE_DIR
        / "June25_locked_models_OOF_predictions.csv",

    "june_regimes":
        TABLE_DIR
        / "June25_locked_models_performance_by_AGBD_regime.csv",

    "june_pooled":
        TABLE_DIR
        / "June25_locked_models_pooled_metrics.csv",

    "june_vs_sep":
        TABLE_DIR
        / "June25_vs_Sep17_locked_model_comparison.csv",

    "paired_error":
        TABLE_DIR
        / "June25_vs_Sep17_paired_error_comparison.csv",

    "sep_reference":
        TABLE_DIR
        / "Sep17_reference_pooled_metrics_recalculated.csv",

    "june_match":
        JUNE_MATCH,

    "reference_snapshot":
        REFERENCE_SNAPSHOT,

    "original_qc_report":
        QC_REPORT,
}


OUT = (
    ROOT
    / "06_Results"
    / "Reviewer_Validation"
    / "C3_Temporal_Mismatch"
    / "C3_02_June_Existing_Experiment_Audit"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

REPORT = (
    OUT
    / "C3_02_JUNE_EXISTING_EXPERIMENT_AUDIT.txt"
)

INVENTORY = (
    OUT
    / "C3_02_FILE_INVENTORY.csv"
)

GAP_TABLE = (
    OUT
    / "C3_02_SEP_VS_JUNE_GAP_SUMMARY.csv"
)

OOF_QC = (
    OUT
    / "C3_02_JUNE_OOF_INTEGRITY.csv"
)


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


def read_text_fallback(path):

    for enc in [
        "utf-8",
        "utf-8-sig",
        "cp1252",
        "latin1",
    ]:

        try:
            return path.read_text(
                encoding=enc
            )

        except UnicodeDecodeError:
            pass

    return path.read_text(
        encoding="utf-8",
        errors="replace",
    )


def identify_shot_col(df):

    candidates = [
        "shot_number",
        "shot",
        "shot_id",
        "shotnumber",
    ]

    lower = {
        c.lower(): c
        for c in df.columns
    }

    for candidate in candidates:

        if candidate.lower() in lower:

            return lower[
                candidate.lower()
            ]

    for c in df.columns:

        if "shot" in c.lower():

            return c

    return None


def identify_gap_col(df):

    preferred = [
        "delta_days_to_NISAR",
        "delta_days",
        "gap_days",
        "temporal_gap_days",
    ]

    lower = {
        c.lower(): c
        for c in df.columns
    }

    for candidate in preferred:

        if candidate.lower() in lower:

            c = lower[
                candidate.lower()
            ]

            if pd.api.types.is_numeric_dtype(
                df[c]
            ):

                return c

    for c in df.columns:

        lc = c.lower()

        if (
            (
                "delta" in lc
                or "gap" in lc
            )
            and "day" in lc
            and pd.api.types.is_numeric_dtype(
                df[c]
            )
        ):

            return c

    return None


def identify_model_col(df):

    candidates = [
        "model",
        "model_name",
        "algorithm",
    ]

    lower = {
        c.lower(): c
        for c in df.columns
    }

    for candidate in candidates:

        if candidate in lower:

            return lower[
                candidate
            ]

    for c in df.columns:

        if "model" in c.lower():

            return c

    return None


def summarize_csv(path):

    d = pd.read_csv(
        path
    )

    return d


# ============================================================
# VERIFY ALL REQUIRED FILES EXIST
# ============================================================

missing = [
    str(path)
    for path in FILES.values()
    if not path.exists()
]

if missing:

    raise FileNotFoundError(
        "Required June experiment files missing:\n"
        + "\n".join(missing)
    )


# ============================================================
# INVENTORY + HASHES
# ============================================================

inventory_rows = []

for label, path in FILES.items():

    if path.suffix.lower() == ".csv":

        d = pd.read_csv(
            path
        )

        rows = len(d)
        cols = len(
            d.columns
        )

        columns = " | ".join(
            d.columns.astype(str)
        )

    else:

        rows = np.nan
        cols = np.nan
        columns = ""

    inventory_rows.append({
        "label":
            label,

        "path":
            str(path),

        "size_bytes":
            path.stat().st_size,

        "sha256":
            sha256_file(
                path
            ),

        "rows":
            rows,

        "columns":
            cols,

        "column_names":
            columns,
    })


inventory_df = pd.DataFrame(
    inventory_rows
)

inventory_df.to_csv(
    INVENTORY,
    index=False
)


# ============================================================
# LOAD PRIMARY TABLES
# ============================================================

master = pd.read_csv(
    MASTER
)

june_match = pd.read_csv(
    JUNE_MATCH
)

reference_snapshot = pd.read_csv(
    REFERENCE_SNAPSHOT
)

fold_check = summarize_csv(
    FILES[
        "frozen_fold_membership"
    ]
)

fold_metrics = summarize_csv(
    FILES[
        "june_fold_metrics"
    ]
)

june_oof = summarize_csv(
    FILES[
        "june_oof"
    ]
)

regimes = summarize_csv(
    FILES[
        "june_regimes"
    ]
)

june_pooled = summarize_csv(
    FILES[
        "june_pooled"
    ]
)

june_vs_sep = summarize_csv(
    FILES[
        "june_vs_sep"
    ]
)

paired_error = summarize_csv(
    FILES[
        "paired_error"
    ]
)

sep_reference = summarize_csv(
    FILES[
        "sep_reference"
    ]
)


# ============================================================
# MASTER / JUNE MATCH INTEGRITY
# ============================================================

if len(master) != 1641:

    raise RuntimeError(
        f"Primary master expected 1641 rows; "
        f"found {len(master)}."
    )

if len(june_match) != 1641:

    raise RuntimeError(
        f"June matched table expected 1641 rows; "
        f"found {len(june_match)}."
    )


master_shot_col = identify_shot_col(
    master
)

june_shot_col = identify_shot_col(
    june_match
)

snapshot_shot_col = identify_shot_col(
    reference_snapshot
)


shot_set_equal = np.nan
snapshot_set_equal = np.nan

if (
    master_shot_col is not None
    and june_shot_col is not None
):

    master_ids = set(
        master[
            master_shot_col
        ].astype(str)
    )

    june_ids = set(
        june_match[
            june_shot_col
        ].astype(str)
    )

    shot_set_equal = (
        master_ids == june_ids
    )


if (
    master_shot_col is not None
    and snapshot_shot_col is not None
):

    snapshot_ids = set(
        reference_snapshot[
            snapshot_shot_col
        ].astype(str)
    )

    snapshot_set_equal = (
        set(
            master[
                master_shot_col
            ].astype(str)
        )
        == snapshot_ids
    )


# ============================================================
# TEMPORAL GAP COMPARISON
# ============================================================

sep_gap_col = identify_gap_col(
    master
)

june_gap_col = identify_gap_col(
    june_match
)

gap_rows = []


if sep_gap_col is not None:

    x = pd.to_numeric(
        master[
            sep_gap_col
        ],
        errors="coerce",
    ).dropna()

    gap_rows.append({
        "scene":
            "Sep17_2026",

        "gap_column":
            sep_gap_col,

        "n":
            len(x),

        "min_days":
            float(
                x.min()
            ),

        "median_days":
            float(
                x.median()
            ),

        "mean_days":
            float(
                x.mean()
            ),

        "max_days":
            float(
                x.max()
            ),
    })


if june_gap_col is not None:

    x = pd.to_numeric(
        june_match[
            june_gap_col
        ],
        errors="coerce",
    ).dropna()

    gap_rows.append({
        "scene":
            "June25_2026",

        "gap_column":
            june_gap_col,

        "n":
            len(x),

        "min_days":
            float(
                x.min()
            ),

        "median_days":
            float(
                x.median()
            ),

        "mean_days":
            float(
                x.mean()
            ),

        "max_days":
            float(
                x.max()
            ),
    })


gap_df = pd.DataFrame(
    gap_rows
)


median_reduction = np.nan

if (
    len(gap_df) == 2
    and
    set(
        gap_df[
            "scene"
        ]
    )
    ==
    {
        "Sep17_2026",
        "June25_2026",
    }
):

    sep_median = float(
        gap_df.loc[
            gap_df[
                "scene"
            ]
            == "Sep17_2026",
            "median_days",
        ].iloc[0]
    )

    june_median = float(
        gap_df.loc[
            gap_df[
                "scene"
            ]
            == "June25_2026",
            "median_days",
        ].iloc[0]
    )

    median_reduction = (
        sep_median
        - june_median
    )


gap_df.to_csv(
    GAP_TABLE,
    index=False
)


# ============================================================
# JUNE OOF INTEGRITY
# ============================================================

oof_shot_col = identify_shot_col(
    june_oof
)

oof_model_col = identify_model_col(
    june_oof
)

oof_rows = []


if oof_model_col is not None:

    for model_name, g in june_oof.groupby(
        oof_model_col,
        dropna=False
    ):

        if oof_shot_col is not None:

            unique_shots = int(
                g[
                    oof_shot_col
                ].nunique()
            )

            shot_match = (
                set(
                    g[
                        oof_shot_col
                    ].astype(str)
                )
                ==
                set(
                    master[
                        master_shot_col
                    ].astype(str)
                )
                if master_shot_col
                is not None
                else np.nan
            )

        else:

            unique_shots = np.nan
            shot_match = np.nan

        oof_rows.append({
            "model":
                str(model_name),

            "rows":
                len(g),

            "unique_shots":
                unique_shots,

            "shot_set_matches_primary_1641":
                shot_match,
        })

else:

    unique_shots = (
        int(
            june_oof[
                oof_shot_col
            ].nunique()
        )
        if oof_shot_col
        is not None
        else np.nan
    )

    oof_rows.append({
        "model":
            "MODEL_COLUMN_NOT_IDENTIFIED",

        "rows":
            len(june_oof),

        "unique_shots":
            unique_shots,

        "shot_set_matches_primary_1641":
            np.nan,
    })


oof_qc_df = pd.DataFrame(
    oof_rows
)

oof_qc_df.to_csv(
    OOF_QC,
    index=False
)


# ============================================================
# ORIGINAL REPORT TEXT
# ============================================================

original_report = read_text_fallback(
    QC_REPORT
)


# ============================================================
# REPORT
# ============================================================

with REPORT.open(
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "=" * 100
        + "\n"
    )

    f.write(
        "C3 STEP 2 - READ-ONLY AUDIT OF EXISTING "
        "JUNE 25 2026 NISAR TEMPORAL EXPERIMENT\n"
    )

    f.write(
        "=" * 100
        + "\n\n"
    )

    f.write(
        "STATUS\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    f.write(
        "NO MODEL WAS FITTED.\n"
    )

    f.write(
        "NO FILE UNDER D:\\new_nisar_data "
        "WAS MODIFIED.\n"
    )

    f.write(
        "C1 and C2 remain frozen.\n"
    )

    f.write(
        "C3 temporal-gap strata remain frozen.\n\n"
    )


    f.write(
        "A. FILE INVENTORY AND HASHES\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    f.write(
        inventory_df[
            [
                "label",
                "size_bytes",
                "sha256",
                "rows",
                "columns",
                "path",
            ]
        ].to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )


    f.write(
        "B. MATCHED-SAMPLE INTEGRITY\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    f.write(
        f"Primary master rows: {len(master)}\n"
    )

    f.write(
        f"June matched rows: {len(june_match)}\n"
    )

    f.write(
        f"Primary shot column: {master_shot_col}\n"
    )

    f.write(
        f"June shot column: {june_shot_col}\n"
    )

    f.write(
        f"June shot set equals primary shot set: "
        f"{shot_set_equal}\n"
    )

    f.write(
        f"Reference snapshot shot set equals primary: "
        f"{snapshot_set_equal}\n\n"
    )


    f.write(
        "C. GEDI-TO-NISAR GAP COMPARISON\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    if len(
        gap_df
    ):

        f.write(
            gap_df.to_string(
                index=False
            )
        )

    else:

        f.write(
            "No compatible temporal-gap "
            "column could be identified."
        )

    f.write(
        "\n"
    )

    f.write(
        f"Median temporal-gap reduction "
        f"(Sep minus June): "
        f"{median_reduction}\n\n"
    )


    f.write(
        "D. FROZEN FOLD MEMBERSHIP CHECK\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    f.write(
        fold_check.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )


    f.write(
        "E. JUNE OOF INTEGRITY\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    f.write(
        oof_qc_df.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )


    f.write(
        "F. JUNE POOLED METRICS - ORIGINAL TABLE\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    f.write(
        june_pooled.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )


    f.write(
        "G. SEPTEMBER REFERENCE METRICS - "
        "ORIGINAL RECALCULATED TABLE\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    f.write(
        sep_reference.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )


    f.write(
        "H. JUNE VS SEPTEMBER LOCKED-MODEL COMPARISON\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    f.write(
        june_vs_sep.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )


    f.write(
        "I. PAIRED ERROR COMPARISON\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    f.write(
        paired_error.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )


    f.write(
        "J. ORIGINAL JUNE FOLD METRICS\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    f.write(
        fold_metrics.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )


    f.write(
        "K. ORIGINAL AGBD-REGIME RESULTS\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    f.write(
        regimes.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )


    f.write(
        "L. ORIGINAL JUNE QC REPORT - VERBATIM COPY\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    f.write(
        original_report
    )

    f.write(
        "\n\n"
    )


    f.write(
        "INTERPRETATION BOUNDARY\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    f.write(
        "This audit does not reinterpret the June experiment "
        "as a controlled temporal experiment.\n"
    )

    f.write(
        "June 25 and September 17 differ not only in "
        "GEDI-to-NISAR temporal gap but also in acquisition date, "
        "season, hydrology, moisture, tide, and potentially "
        "other scene conditions.\n"
    )

    f.write(
        "Therefore any similar model performance can support only "
        "the statement that the tested 84-day reduction did not "
        "materially alter nonlinear model performance.\n"
    )

    f.write(
        "It cannot establish that temporal mismatch is irrelevant.\n"
    )


# ============================================================
# CONSOLE
# ============================================================

print("")
print("=" * 84)
print("C3 STEP 2 COMPLETE")
print("READ-ONLY JUNE-2026 EXPERIMENT AUDIT")
print("=" * 84)

print("")
print("Matched sample integrity:")
print(
    "  primary rows:",
    len(master)
)
print(
    "  June rows:",
    len(june_match)
)
print(
    "  identical shot set:",
    shot_set_equal
)
print(
    "  reference snapshot identical:",
    snapshot_set_equal
)

print("")
print("Temporal-gap comparison:")

if len(gap_df):

    print(
        gap_df.to_string(
            index=False
        )
    )

print(
    "Median gap reduction:",
    median_reduction
)

print("")
print("Frozen fold check:")
print(
    fold_check.to_string(
        index=False
    )
)

print("")
print("June OOF integrity:")
print(
    oof_qc_df.to_string(
        index=False
    )
)

print("")
print("June pooled metrics:")
print(
    june_pooled.to_string(
        index=False
    )
)

print("")
print("June vs September comparison:")
print(
    june_vs_sep.to_string(
        index=False
    )
)

print("")
print("Paired error comparison:")
print(
    paired_error.to_string(
        index=False
    )
)

print("")
print("Report:")
print(REPORT)

print("")
print("NO MODEL WAS FITTED.")
print("D:\\new_nisar_data WAS NOT MODIFIED.")
print("C1 AND C2 REMAIN FROZEN.")
print("=" * 84)
