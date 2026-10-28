from pathlib import Path
import hashlib
import json

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
    / "C5_Multiscale_Ablation"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

FAMILY_FILE = (
    OUT
    / "C5_00_ABLATION_FAMILIES.csv"
)

JSON_FILE = (
    OUT
    / "C5_00_ABLATION_FAMILIES.json"
)

REPORT = (
    OUT
    / "C5_00_ABLATION_DESIGN_REPORT.txt"
)


# ============================================================
# LOAD FROZEN TABLE
# ============================================================

df = pd.read_csv(DATA)

if len(df) != 1641:
    raise RuntimeError(
        f"Expected 1641 rows; found {len(df)}."
    )

if "shot_number" not in df.columns:
    raise RuntimeError(
        "shot_number missing."
    )

if df["shot_number"].nunique() != 1641:
    raise RuntimeError(
        "shot_number is not unique."
    )

if "cv_block_id" not in df.columns:
    raise RuntimeError(
        "cv_block_id missing."
    )


# ============================================================
# FROZEN BASIC PREDICTORS
# ============================================================

P3 = [
    "HH_dB",
    "HV_dB",
    "HV_SD",
]

CONTEXT_5 = [
    "HH_MeanLin_dB_5x5",
    "HV_MeanLin_dB_5x5",
    "HH_SD_dB_5x5",
    "HV_SD_dB_5x5",
    "PolContrast_SD_dB_5x5",
]

CONTEXT_9 = [
    "HH_MeanLin_dB_9x9",
    "HV_MeanLin_dB_9x9",
    "HH_SD_dB_9x9",
    "HV_SD_dB_9x9",
    "PolContrast_SD_dB_9x9",
]

HISTORICAL_9P = [
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

FULL_14P = [
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


# ============================================================
# PREDECLARED C5 FAMILIES
# ============================================================

families = [
    {
        "family":
            "A00_COORD_ONLY",

        "role":
            "spatial_diagnostic",

        "predictors": [
            "utm_x",
            "utm_y",
        ],

        "purpose":
            "Tests whether geographic location alone carries apparent predictive skill.",
    },

    {
        "family":
            "A01_HV9_MEAN_ONLY",

        "role":
            "radar_ablation",

        "predictors": [
            "HV_MeanLin_dB_9x9",
        ],

        "purpose":
            "Tests whether contextual HV alone carries most of the multiscale signal.",
    },

    {
        "family":
            "A02_HH_HV9_MEANS",

        "role":
            "radar_ablation",

        "predictors": [
            "HH_MeanLin_dB_9x9",
            "HV_MeanLin_dB_9x9",
        ],

        "purpose":
            "Tests dual-polarization 9x9 contextual means without local predictors.",
    },

    {
        "family":
            "A03_3P_BASE",

        "role":
            "reference_ablation",

        "predictors":
            P3,

        "purpose":
            "Frozen historical three-predictor baseline.",
    },

    {
        "family":
            "A04_3P_PLUS_HV9_MEAN",

        "role":
            "radar_ablation",

        "predictors":
            P3
            +
            [
                "HV_MeanLin_dB_9x9",
            ],

        "purpose":
            "Tests incremental value of contextual HV mean alone.",
    },

    {
        "family":
            "A05_3P_PLUS_9X9_MEANS",

        "role":
            "radar_ablation",

        "predictors":
            P3
            +
            [
                "HH_MeanLin_dB_9x9",
                "HV_MeanLin_dB_9x9",
            ],

        "purpose":
            "Tests contextual means separately from contextual heterogeneity.",
    },

    {
        "family":
            "A06_3P_PLUS_9X9_HETEROGENEITY",

        "role":
            "radar_ablation",

        "predictors":
            P3
            +
            [
                "HH_SD_dB_9x9",
                "HV_SD_dB_9x9",
                "PolContrast_SD_dB_9x9",
            ],

        "purpose":
            "Tests whether contextual heterogeneity rather than contextual mean drives the gain.",
    },

    {
        "family":
            "A07_3P_PLUS_HH3_SD",

        "role":
            "support_ablation",

        "predictors":
            P3
            +
            [
                "HH_SD_dB_3x3",
            ],

        "purpose":
            "Tests the incremental local 3x3 HH heterogeneity term.",
    },

    {
        "family":
            "A08_3P_PLUS_5X5_CONTEXT",

        "role":
            "support_ablation",

        "predictors":
            P3
            +
            CONTEXT_5,

        "purpose":
            "Eight-predictor 5x5 contextual model for matched support-size comparison.",
    },

    {
        "family":
            "A09_3P_PLUS_9X9_CONTEXT",

        "role":
            "support_ablation",

        "predictors":
            P3
            +
            CONTEXT_9,

        "purpose":
            "Eight-predictor 9x9 contextual model directly comparable with A08.",
    },

    {
        "family":
            "A10_HISTORICAL_9P",

        "role":
            "historical_reference",

        "predictors":
            HISTORICAL_9P,

        "purpose":
            "Exact previously locked historical 9P predictor family.",
    },

    {
        "family":
            "A11_FULL_14P",

        "role":
            "c1b_reference",

        "predictors":
            FULL_14P,

        "purpose":
            "Exact C1B Full14P family selected in all five C1B outer folds.",
    },
]


# ============================================================
# COLUMN VERIFICATION
# ============================================================

all_predictors = sorted(
    set(
        p
        for item in families
        for p in item["predictors"]
    )
)

missing = [
    p
    for p in all_predictors
    if p not in df.columns
]

if missing:
    raise RuntimeError(
        f"Required C5 predictor columns missing: {missing}"
    )


# ============================================================
# NUMERIC / MISSING QC
# ============================================================

qc_rows = []

for predictor in all_predictors:

    x = pd.to_numeric(
        df[predictor],
        errors="coerce",
    )

    qc_rows.append({
        "predictor":
            predictor,

        "n":
            len(x),

        "n_missing":
            int(
                x.isna().sum()
            ),

        "n_unique":
            int(
                x.nunique(
                    dropna=True
                )
            ),
    })


qc_df = pd.DataFrame(
    qc_rows
)

bad_missing = qc_df.loc[
    qc_df["n_missing"] > 0
]

if len(bad_missing):
    raise RuntimeError(
        "Missing values found in required "
        "C5 predictors:\n"
        +
        bad_missing.to_string(
            index=False
        )
    )


# ============================================================
# FAMILY TABLE
# ============================================================

family_rows = []

for item in families:

    predictors = item[
        "predictors"
    ]

    if len(predictors) != len(
        set(predictors)
    ):
        raise RuntimeError(
            f"Duplicate predictor inside "
            f"{item['family']}."
        )

    family_rows.append({
        "family":
            item[
                "family"
            ],

        "role":
            item[
                "role"
            ],

        "n_predictors":
            len(
                predictors
            ),

        "predictors":
            ";".join(
                predictors
            ),

        "purpose":
            item[
                "purpose"
            ],
    })


family_df = pd.DataFrame(
    family_rows
)

family_df.to_csv(
    FAMILY_FILE,
    index=False
)


# ============================================================
# INPUT HASH
# ============================================================

sha256 = hashlib.sha256()

with DATA.open(
    "rb"
) as f:

    while True:

        chunk = f.read(
            1024 * 1024
        )

        if not chunk:
            break

        sha256.update(
            chunk
        )

input_hash = sha256.hexdigest()


# ============================================================
# JSON FREEZE MANIFEST
# ============================================================

freeze_manifest = {
    "status":
        "FROZEN_BEFORE_C5_MODEL_RESULTS",

    "input_table":
        str(
            DATA
        ),

    "input_sha256":
        input_hash,

    "n_rows":
        int(
            len(df)
        ),

    "n_unique_shots":
        int(
            df[
                "shot_number"
            ].nunique()
        ),

    "n_spatial_blocks":
        int(
            df[
                "cv_block_id"
            ].nunique()
        ),

    "families":
        families,

    "analysis_rules": [
        "No predictor family may be added or removed after viewing C5 performance.",
        "No family may be modified after viewing C5 performance.",
        "A08 and A09 provide the matched 5x5 versus 9x9 support comparison.",
        "A00 is a spatial diagnostic and is not a radar biomass model.",
        "C1-C4 remain frozen.",
        "No operational biomass-map claim will be made.",
    ],
}


with JSON_FILE.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        freeze_manifest,
        f,
        indent=2,
    )


# ============================================================
# REPORT
# ============================================================

with REPORT.open(
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "=" * 100
        + "\n"
    )

    f.write(
        "C5 STEP 0 - PREDECLARED MULTISCALE "
        "ABLATION DESIGN FREEZE\n"
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
        "NO PERFORMANCE RESULT WAS USED TO "
        "DEFINE THESE FAMILIES.\n"
    )

    f.write(
        "C1, C2, C3 and C4 remain frozen.\n\n"
    )

    f.write(
        "INPUT\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    f.write(
        f"Table: {DATA}\n"
    )

    f.write(
        f"SHA256: {input_hash}\n"
    )

    f.write(
        f"Rows: {len(df)}\n"
    )

    f.write(
        f"Unique shots: "
        f"{df['shot_number'].nunique()}\n"
    )

    f.write(
        f"Spatial blocks: "
        f"{df['cv_block_id'].nunique()}\n\n"
    )

    f.write(
        "FROZEN FAMILIES\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    f.write(
        family_df.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )

    f.write(
        "PREDICTOR QC\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    f.write(
        qc_df.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )

    f.write(
        "INTERPRETATION RULES\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    f.write(
        "A08 versus A09 is the primary matched "
        "5x5-versus-9x9 support comparison.\n"
    )

    f.write(
        "A01-A06 isolate contextual mean, HV, "
        "and heterogeneity contributions.\n"
    )

    f.write(
        "A10 reproduces the historical 9P feature set.\n"
    )

    f.write(
        "A11 is the exact C1B Full14P reference family.\n"
    )

    f.write(
        "A00 is strictly a spatial-location diagnostic "
        "and must not be described as a radar model.\n"
    )

    f.write(
        "No predictor family will be changed after "
        "C5 model performance is examined.\n"
    )


# ============================================================
# CONSOLE
# ============================================================

print("")
print("=" * 84)
print("C5 STEP 0 COMPLETE")
print("MULTISCALE ABLATION DESIGN FROZEN")
print("=" * 84)

print("")
print(
    family_df.to_string(
        index=False
    )
)

print("")
print(
    "Rows:",
    len(df)
)

print(
    "Unique shots:",
    df[
        "shot_number"
    ].nunique()
)

print(
    "Spatial blocks:",
    df[
        "cv_block_id"
    ].nunique()
)

print("")
print(
    "Input SHA256:",
    input_hash
)

print("")
print("Report:")
print(REPORT)

print("")
print("NO MODEL WAS FITTED.")
print("C1-C4 REMAIN FROZEN.")
print("=" * 84)
