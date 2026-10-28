from pathlib import Path
import re
import numpy as np
import pandas as pd


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

JUNE_MATCH = (
    JUNE_ROOT
    / "Phase8B2_June25_Sensitivity_V2"
    / "02_GEDI_Matching"
    / "Tables"
    / "GEDI_June25_matched_all_1641.csv"
)

OUT = (
    ROOT
    / "06_Results"
    / "Reviewer_Validation"
    / "C3_Temporal_Mismatch"
    / "C3_02_June_Existing_Experiment_Audit"
)

OUT.mkdir(parents=True, exist_ok=True)

SUMMARY_FILE = (
    OUT
    / "C3_02B_CORRECTED_GAP_SUMMARY.csv"
)

SHOT_FILE = (
    OUT
    / "C3_02B_CORRECTED_SHOT_GAPS.csv"
)

REPORT = (
    OUT
    / "C3_02B_CORRECTED_GAP_AUDIT.txt"
)


# ============================================================
# FIND THE ISOLATED JUNE NISAR PRODUCT
# ============================================================

products = list(
    JUNE_ROOT.glob(
        "NISAR_L2_PR_GCOV_*20260625T*.h5"
    )
)

if len(products) != 1:
    raise RuntimeError(
        "Expected exactly one June-25 NISAR H5 "
        f"at D:\\new_nisar_data root; found {len(products)}."
    )

june_product = products[0]


# ============================================================
# PARSE NISAR ACQUISITION START TIME FROM PRODUCT ID
# ============================================================

m = re.search(
    r"_(\d{8}T\d{6})_(\d{8}T\d{6})_P\d+_",
    june_product.name,
)

if m is None:
    raise RuntimeError(
        "Could not parse June acquisition time "
        "from NISAR product filename."
    )

june_start_string = m.group(1)

june_datetime = pd.to_datetime(
    june_start_string,
    format="%Y%m%dT%H%M%S",
    utc=True,
)


# ============================================================
# LOAD FROZEN TABLES
# ============================================================

master = pd.read_csv(MASTER)
june = pd.read_csv(JUNE_MATCH)

if len(master) != 1641:
    raise RuntimeError(
        f"Primary master has {len(master)} rows, expected 1641."
    )

if len(june) != 1641:
    raise RuntimeError(
        f"June matched table has {len(june)} rows, expected 1641."
    )

if master["shot_number"].nunique() != 1641:
    raise RuntimeError(
        "Primary shot_number is not unique."
    )

if june["shot_number"].nunique() != 1641:
    raise RuntimeError(
        "June shot_number is not unique."
    )

if set(master["shot_number"].astype(str)) != set(
    june["shot_number"].astype(str)
):
    raise RuntimeError(
        "June and primary shot sets differ."
    )


# ============================================================
# USE GEDI DATE FROM FROZEN PRIMARY TABLE
# ============================================================

required = [
    "shot_number",
    "gedi_datetime",
    "nisar_datetime",
    "delta_days_to_NISAR",
]

missing = [
    c for c in required
    if c not in master.columns
]

if missing:
    raise RuntimeError(
        f"Primary table missing columns: {missing}"
    )


work = master[
    required
].copy()

work["gedi_datetime_parsed"] = pd.to_datetime(
    work["gedi_datetime"],
    utc=True,
    errors="coerce",
)

work["sep_nisar_datetime_parsed"] = pd.to_datetime(
    work["nisar_datetime"],
    utc=True,
    errors="coerce",
)

if work["gedi_datetime_parsed"].isna().any():
    raise RuntimeError(
        "Some GEDI datetimes could not be parsed."
    )

if work["sep_nisar_datetime_parsed"].isna().any():
    raise RuntimeError(
        "Some September NISAR datetimes could not be parsed."
    )


# ============================================================
# RECOMPUTE BOTH TEMPORAL GAPS FROM DATES
# ============================================================

work["sep_gap_days_recomputed"] = (
    (
        work["sep_nisar_datetime_parsed"]
        - work["gedi_datetime_parsed"]
    )
    .dt.total_seconds()
    / 86400.0
)

work["june_nisar_datetime"] = june_datetime

work["june_gap_days_recomputed"] = (
    (
        work["june_nisar_datetime"]
        - work["gedi_datetime_parsed"]
    )
    .dt.total_seconds()
    / 86400.0
)

work["gap_reduction_days_sep_minus_june"] = (
    work["sep_gap_days_recomputed"]
    - work["june_gap_days_recomputed"]
)


# ============================================================
# QC SEPTEMBER AGAINST STORED FIELD
# ============================================================

work["sep_stored_vs_recomputed_absdiff"] = np.abs(
    work["delta_days_to_NISAR"]
    - work["sep_gap_days_recomputed"]
)

max_sep_diff = float(
    work[
        "sep_stored_vs_recomputed_absdiff"
    ].max()
)

if max_sep_diff > 1e-6:
    raise RuntimeError(
        "September recomputed gaps do not match "
        f"frozen stored gaps. Max difference={max_sep_diff}"
    )


# ============================================================
# SUMMARY
# ============================================================

summary = pd.DataFrame([
    {
        "scene": "Sep17_2026",
        "nisar_datetime": str(
            work["sep_nisar_datetime_parsed"].iloc[0]
        ),
        "n": len(work),
        "min_gap_days": float(
            work["sep_gap_days_recomputed"].min()
        ),
        "median_gap_days": float(
            work["sep_gap_days_recomputed"].median()
        ),
        "mean_gap_days": float(
            work["sep_gap_days_recomputed"].mean()
        ),
        "max_gap_days": float(
            work["sep_gap_days_recomputed"].max()
        ),
    },
    {
        "scene": "June25_2026",
        "nisar_datetime": str(june_datetime),
        "n": len(work),
        "min_gap_days": float(
            work["june_gap_days_recomputed"].min()
        ),
        "median_gap_days": float(
            work["june_gap_days_recomputed"].median()
        ),
        "mean_gap_days": float(
            work["june_gap_days_recomputed"].mean()
        ),
        "max_gap_days": float(
            work["june_gap_days_recomputed"].max()
        ),
    },
])

sep_median = float(
    summary.loc[
        summary["scene"] == "Sep17_2026",
        "median_gap_days",
    ].iloc[0]
)

june_median = float(
    summary.loc[
        summary["scene"] == "June25_2026",
        "median_gap_days",
    ].iloc[0]
)

median_reduction = (
    sep_median
    - june_median
)

reduction_min = float(
    work[
        "gap_reduction_days_sep_minus_june"
    ].min()
)

reduction_max = float(
    work[
        "gap_reduction_days_sep_minus_june"
    ].max()
)

reduction_sd = float(
    work[
        "gap_reduction_days_sep_minus_june"
    ].std(ddof=1)
)


# ============================================================
# SAVE — REVIEWER WORKSPACE ONLY
# ============================================================

summary.to_csv(
    SUMMARY_FILE,
    index=False,
)

work[
    [
        "shot_number",
        "gedi_datetime",
        "sep_nisar_datetime_parsed",
        "june_nisar_datetime",
        "sep_gap_days_recomputed",
        "june_gap_days_recomputed",
        "gap_reduction_days_sep_minus_june",
    ]
].to_csv(
    SHOT_FILE,
    index=False,
)


# ============================================================
# REPORT
# ============================================================

with REPORT.open(
    "w",
    encoding="utf-8",
) as f:

    f.write("=" * 96 + "\n")
    f.write(
        "C3 STEP 2B - CORRECTED READ-ONLY "
        "JUNE TEMPORAL-GAP AUDIT\n"
    )
    f.write("=" * 96 + "\n\n")

    f.write("STATUS\n")
    f.write("-" * 96 + "\n")
    f.write("NO MODEL WAS FITTED.\n")
    f.write(
        "NO FILE UNDER D:\\new_nisar_data WAS MODIFIED.\n"
    )
    f.write(
        "The previous Step-2 gap comparison used the "
        "stale September delta_days_to_NISAR field "
        "contained in the June matched table.\n"
    )
    f.write(
        "This audit recomputes June temporal gaps directly "
        "from GEDI acquisition datetimes and the June NISAR "
        "product acquisition timestamp.\n\n"
    )

    f.write("JUNE NISAR PRODUCT\n")
    f.write("-" * 96 + "\n")
    f.write(str(june_product) + "\n")
    f.write(
        f"Parsed acquisition start: {june_datetime}\n\n"
    )

    f.write("SEPTEMBER QC\n")
    f.write("-" * 96 + "\n")
    f.write(
        "Maximum stored-vs-recomputed September gap "
        f"difference: {max_sep_diff:.12g} days\n\n"
    )

    f.write("CORRECTED GAP SUMMARY\n")
    f.write("-" * 96 + "\n")
    f.write(
        summary.to_string(index=False)
    )
    f.write("\n\n")

    f.write("GAP REDUCTION\n")
    f.write("-" * 96 + "\n")
    f.write(
        f"Median reduction (Sep minus June): "
        f"{median_reduction:.6f} days\n"
    )
    f.write(
        f"Minimum per-shot reduction: "
        f"{reduction_min:.6f} days\n"
    )
    f.write(
        f"Maximum per-shot reduction: "
        f"{reduction_max:.6f} days\n"
    )
    f.write(
        f"SD of per-shot reduction: "
        f"{reduction_sd:.12g} days\n\n"
    )

    f.write("SCIENTIFIC INTERPRETATION\n")
    f.write("-" * 96 + "\n")
    f.write(
        "The June scene reduces the GEDI-to-NISAR temporal "
        "gap relative to the September scene, but the two "
        "scenes also differ in season, hydrology, moisture, "
        "tide, and potentially other environmental conditions.\n"
    )
    f.write(
        "The existing locked-model comparison showed nearly "
        "unchanged nonlinear model performance.\n"
    )
    f.write(
        "Therefore the supported conclusion is: "
        "'the tested temporal-gap reduction did not materially "
        "alter nonlinear model performance.'\n"
    )
    f.write(
        "This does not establish that temporal mismatch "
        "is irrelevant.\n"
    )


# ============================================================
# CONSOLE
# ============================================================

print("")
print("=" * 84)
print("C3 STEP 2B COMPLETE")
print("CORRECTED JUNE TEMPORAL-GAP AUDIT")
print("=" * 84)

print("")
print("June product:")
print(june_product)

print("")
print("Parsed June NISAR time:")
print(june_datetime)

print("")
print("Corrected gap summary:")
print(summary.to_string(index=False))

print("")
print(
    "Median gap reduction:",
    f"{median_reduction:.6f}",
    "days"
)

print(
    "Per-shot reduction range:",
    f"{reduction_min:.6f}",
    "to",
    f"{reduction_max:.6f}",
    "days"
)

print("")
print("Report:")
print(REPORT)

print("")
print("NO MODEL WAS FITTED.")
print("D:\\new_nisar_data WAS NOT MODIFIED.")
print("C1 AND C2 REMAIN FROZEN.")
print("=" * 84)
