from pathlib import Path
import pandas as pd
import re

ROOT = Path(r"D:\NISAR_Mangrove_Paper")

MATCHED = ROOT / "04_Matched_Data"

PHASE9_SCRIPT = (
    ROOT
    / "05_ML_DL"
    / "Predictor_Expansion"
    / "Phase9_Scripts"
    / "phase9_et9p_final_defensibility.py"
)

OUTDIR = (
    ROOT
    / "06_Results"
    / "Reviewer_Validation"
    / "C1_Independent_Validation"
)

OUTDIR.mkdir(parents=True, exist_ok=True)

REPORT = OUTDIR / "C1_00_INPUT_AUDIT_REPORT.txt"
CSV_SUMMARY = OUTDIR / "C1_00_CSV_INVENTORY.csv"


# ============================================================
# 1. Scan all CSV files under 04_Matched_Data
# ============================================================

records = []

for f in MATCHED.rglob("*.csv"):
    try:
        df = pd.read_csv(f)

        records.append({
            "file": str(f),
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": " | ".join(df.columns.astype(str)),
        })

    except Exception as e:
        records.append({
            "file": str(f),
            "rows": None,
            "columns": None,
            "column_names": f"ERROR: {e}",
        })

inventory = pd.DataFrame(records)
inventory.to_csv(CSV_SUMMARY, index=False)


# ============================================================
# 2. Find all 1641-row candidate tables
# ============================================================

candidates = inventory[inventory["rows"] == 1641].copy()


# ============================================================
# 3. Read Phase 9 script
# ============================================================

if not PHASE9_SCRIPT.exists():
    raise FileNotFoundError(
        f"Phase 9 script not found:\n{PHASE9_SCRIPT}"
    )

phase9_text = PHASE9_SCRIPT.read_text(
    encoding="utf-8",
    errors="replace"
)


# ============================================================
# 4. Locked predictor definitions
# ============================================================

ET3P = [
    "HH_dB",
    "HV_dB",
    "HV_SD",
]

ET9P = [
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
# 5. Audit every 1641-row candidate
# ============================================================

candidate_details = []

for _, row in candidates.iterrows():

    f = Path(row["file"])
    df = pd.read_csv(f)

    cols = list(df.columns)

    missing_3p = [x for x in ET3P if x not in cols]
    missing_9p = [x for x in ET9P if x not in cols]

    coord_candidates = [
        c for c in cols
        if any(k in c.lower() for k in [
            "lat",
            "lon",
            "longitude",
            "latitude",
            "utm",
            "easting",
            "northing",
            "x_coord",
            "y_coord"
        ])
    ]

    fold_candidates = [
        c for c in cols
        if any(k in c.lower() for k in [
            "fold",
            "block",
            "group",
            "cluster"
        ])
    ]

    id_candidates = [
        c for c in cols
        if any(k in c.lower() for k in [
            "shot",
            "beam",
            "track",
            "orbit",
            "_id",
            "id_"
        ])
    ]

    target_candidates = [
        c for c in cols
        if any(k in c.lower() for k in [
            "agbd",
            "biomass"
        ])
    ]

    uncertainty_candidates = [
        c for c in cols
        if any(k in c.lower() for k in [
            "agbd_se",
            "uncert",
            "standard_error",
            "prediction_se"
        ])
    ]

    date_candidates = [
        c for c in cols
        if any(k in c.lower() for k in [
            "date",
            "time",
            "acquisition"
        ])
    ]

    candidate_details.append({
        "file": str(f),
        "rows": len(df),
        "columns": len(cols),
        "all_ET3P_present": len(missing_3p) == 0,
        "all_ET9P_present": len(missing_9p) == 0,
        "missing_ET3P": missing_3p,
        "missing_ET9P": missing_9p,
        "coordinate_candidates": coord_candidates,
        "fold_block_candidates": fold_candidates,
        "id_track_candidates": id_candidates,
        "target_candidates": target_candidates,
        "uncertainty_candidates": uncertainty_candidates,
        "date_candidates": date_candidates,
    })


# ============================================================
# 6. Search Phase 9 script for settings
# ============================================================

csv_mentions = sorted(set(
    re.findall(
        r'[^"\']+\.csv',
        phase9_text,
        flags=re.IGNORECASE
    )
))

seed_mentions = re.findall(
    r'(?:random_state|seed)\s*=\s*(\d+)',
    phase9_text,
    flags=re.IGNORECASE
)

feature_mentions = {
    feature: feature in phase9_text
    for feature in ET9P
}


# ============================================================
# 7. Write detailed report
# ============================================================

with REPORT.open("w", encoding="utf-8") as f:

    f.write("=" * 90 + "\n")
    f.write("C1 REVIEWER VALIDATION - FROZEN INPUT AUDIT\n")
    f.write("=" * 90 + "\n\n")

    f.write("PURPOSE\n")
    f.write("-" * 90 + "\n")
    f.write(
        "Identify and verify the frozen 1,641-record modeling table, "
        "locked ET3P/ET9P predictor fields, coordinates, spatial identifiers, "
        "GEDI target, uncertainty fields, acquisition information, and "
        "Phase 9 dependencies before new reviewer-validation analyses.\n\n"
    )

    f.write("RULES\n")
    f.write("-" * 90 + "\n")
    f.write("1. No model is fitted by this script.\n")
    f.write("2. No existing Phase 9 result is modified.\n")
    f.write("3. ET3P and ET9P predictor definitions remain frozen.\n")
    f.write("4. The audit is used only to identify the correct inputs.\n\n")

    f.write("=" * 90 + "\n")
    f.write("1641-ROW CSV CANDIDATES\n")
    f.write("=" * 90 + "\n")

    if len(candidate_details) == 0:
        f.write("\nNO 1641-ROW CSV FILE FOUND.\n")

    else:
        for i, d in enumerate(candidate_details, start=1):

            f.write("\n" + "-" * 90 + "\n")
            f.write(f"CANDIDATE {i}\n")
            f.write("-" * 90 + "\n")

            f.write(f"File:\n{d['file']}\n\n")

            f.write(f"Rows: {d['rows']}\n")
            f.write(f"Columns: {d['columns']}\n")

            f.write(
                f"All ET3P predictors present: "
                f"{d['all_ET3P_present']}\n"
            )

            f.write(
                f"All ET9P predictors present: "
                f"{d['all_ET9P_present']}\n"
            )

            f.write(
                f"Missing ET3P predictors: "
                f"{d['missing_ET3P']}\n"
            )

            f.write(
                f"Missing ET9P predictors: "
                f"{d['missing_ET9P']}\n"
            )

            f.write(
                f"Coordinate candidates: "
                f"{d['coordinate_candidates']}\n"
            )

            f.write(
                f"Fold/block candidates: "
                f"{d['fold_block_candidates']}\n"
            )

            f.write(
                f"ID/track candidates: "
                f"{d['id_track_candidates']}\n"
            )

            f.write(
                f"Target candidates: "
                f"{d['target_candidates']}\n"
            )

            f.write(
                f"Uncertainty candidates: "
                f"{d['uncertainty_candidates']}\n"
            )

            f.write(
                f"Date/time candidates: "
                f"{d['date_candidates']}\n"
            )

    f.write("\n" + "=" * 90 + "\n")
    f.write("PHASE 9 SCRIPT AUDIT\n")
    f.write("=" * 90 + "\n\n")

    f.write(f"Phase 9 script:\n{PHASE9_SCRIPT}\n\n")
    f.write(f"Script exists: {PHASE9_SCRIPT.exists()}\n")

    f.write("\nCSV file references found in Phase 9 script:\n")

    if csv_mentions:
        for x in csv_mentions:
            f.write(f"  {x}\n")
    else:
        f.write("  None detected by simple text scan.\n")

    f.write("\nSeed/random_state values found:\n")
    f.write(f"  {seed_mentions}\n")

    f.write("\nLocked ET9P feature names found in Phase 9 script:\n")

    for feature, found in feature_mentions.items():
        f.write(f"  {feature}: {found}\n")

    f.write("\n" + "=" * 90 + "\n")
    f.write("LOCKED REVIEWER-VALIDATION FEATURES\n")
    f.write("=" * 90 + "\n")

    f.write("\nET3P\n")
    for x in ET3P:
        f.write(f"  {x}\n")

    f.write("\nET9P\n")
    for x in ET9P:
        f.write(f"  {x}\n")

    f.write("\n" + "=" * 90 + "\n")
    f.write("END OF AUDIT\n")
    f.write("=" * 90 + "\n")


# ============================================================
# 8. Console summary
# ============================================================

print("")
print("=" * 80)
print("C1 INPUT AUDIT COMPLETE")
print("=" * 80)

print("")
print("CSV inventory saved to:")
print(CSV_SUMMARY)

print("")
print("Detailed report saved to:")
print(REPORT)

print("")
print(f"1641-row candidate files found: {len(candidate_details)}")

for i, d in enumerate(candidate_details, start=1):

    print("")
    print("-" * 80)
    print(f"CANDIDATE {i}")
    print("-" * 80)

    print(d["file"])
    print(f"Rows: {d['rows']}")
    print(f"Columns: {d['columns']}")

    print(
        f"All ET3P predictors present: "
        f"{d['all_ET3P_present']}"
    )

    print(
        f"All ET9P predictors present: "
        f"{d['all_ET9P_present']}"
    )

    print(
        f"Target candidates: "
        f"{d['target_candidates']}"
    )

    print(
        f"Coordinate candidates: "
        f"{d['coordinate_candidates']}"
    )

    print(
        f"Fold/block candidates: "
        f"{d['fold_block_candidates']}"
    )

    print(
        f"ID/track candidates: "
        f"{d['id_track_candidates']}"
    )

    print(
        f"Uncertainty candidates: "
        f"{d['uncertainty_candidates']}"
    )

    print(
        f"Date/time candidates: "
        f"{d['date_candidates']}"
    )

print("")
print("IMPORTANT:")
print("NO MODEL WAS FITTED.")
print("NO EXISTING FILE WAS MODIFIED.")
print("ET3P AND ET9P DEFINITIONS REMAIN FROZEN.")
print("=" * 80)
