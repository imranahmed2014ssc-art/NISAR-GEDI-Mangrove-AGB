from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

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

OUT.mkdir(parents=True, exist_ok=True)

REPORT = OUT / "C2_00_UNIT_AUDIT_REPORT.txt"
CARD = OUT / "C2_00_COLUMN_CARDINALITY.csv"
TRACKCOLS = OUT / "C2_00_POTENTIAL_TRACK_COLUMNS.csv"
DATEBEAM = OUT / "C2_00_DATE_BEAM_COUNTS.csv"
BLOCKS = OUT / "C2_00_BLOCK_COUNTS.csv"
REPEATS = OUT / "C2_00_REPEAT_GROUP_COUNTS.csv"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


df = pd.read_csv(DATA)

if len(df) != 1641:
    raise RuntimeError(f"Expected 1641 rows; found {len(df)}")

if df["shot_number"].nunique() != 1641:
    raise RuntimeError("shot_number is not unique.")

# ------------------------------------------------------------
# Column cardinality
# ------------------------------------------------------------

rows = []

for c in df.columns:
    s = df[c]

    rows.append({
        "column": c,
        "dtype": str(s.dtype),
        "n_nonnull": int(s.notna().sum()),
        "n_missing": int(s.isna().sum()),
        "n_unique": int(s.nunique(dropna=True)),
    })

card = pd.DataFrame(rows).sort_values(
    ["n_unique", "column"],
    ascending=[False, True]
)

card.to_csv(CARD, index=False)

# ------------------------------------------------------------
# Potential track / acquisition / source identifiers
# ------------------------------------------------------------

keywords = [
    "track",
    "orbit",
    "beam",
    "granule",
    "file",
    "source",
    "acquisition",
    "date",
    "time",
    "delta",
    "shot",
    "segment",
    "group",
    "repeat",
    "cycle",
]

potential = []

for c in df.columns:
    lc = c.lower()

    if any(k in lc for k in keywords):
        potential.append({
            "column": c,
            "dtype": str(df[c].dtype),
            "n_nonnull": int(df[c].notna().sum()),
            "n_unique": int(df[c].nunique(dropna=True)),
            "example_1": (
                str(df[c].dropna().iloc[0])
                if df[c].notna().any()
                else ""
            ),
            "example_2": (
                str(df[c].dropna().iloc[1])
                if df[c].notna().sum() > 1
                else ""
            ),
        })

potential_df = pd.DataFrame(potential)

potential_df.to_csv(
    TRACKCOLS,
    index=False
)

# ------------------------------------------------------------
# Existing spatial-unit audit
# ------------------------------------------------------------

required_spatial = [
    "cv_block_id",
    "spatial_fold",
    "utm_x",
    "utm_y",
]

for c in required_spatial:
    if c not in df.columns:
        raise RuntimeError(f"Required spatial field missing: {c}")

block_counts = (
    df.groupby("cv_block_id")
      .agg(
          n=("shot_number", "size"),
          x_mean=("utm_x", "mean"),
          y_mean=("utm_y", "mean"),
          x_min=("utm_x", "min"),
          x_max=("utm_x", "max"),
          y_min=("utm_y", "min"),
          y_max=("utm_y", "max"),
          original_fold=("spatial_fold", "first"),
      )
      .reset_index()
      .sort_values("cv_block_id")
)

block_counts.to_csv(
    BLOCKS,
    index=False
)

# ------------------------------------------------------------
# Repeat-group audit
# ------------------------------------------------------------

if "repeat_group_id" in df.columns:

    repeat_counts = (
        df.groupby("repeat_group_id")
          .agg(
              n=("shot_number", "size"),
              unique_blocks=("cv_block_id", "nunique"),
              x_mean=("utm_x", "mean"),
              y_mean=("utm_y", "mean"),
          )
          .reset_index()
          .sort_values(
              ["n", "repeat_group_id"],
              ascending=[False, True]
          )
    )

    repeat_counts.to_csv(
        REPEATS,
        index=False
    )

else:

    repeat_counts = pd.DataFrame()

# ------------------------------------------------------------
# Date / beam combinations
# ------------------------------------------------------------

date_candidates = [
    c for c in [
        "acquisition_date",
        "gedi_datetime",
    ]
    if c in df.columns
]

beam_candidates = [
    c for c in [
        "beam",
        "beam_name",
    ]
    if c in df.columns
]

date_beam_table = pd.DataFrame()

if date_candidates and beam_candidates:

    date_col = date_candidates[0]
    beam_col = beam_candidates[0]

    work = df.copy()

    # Normalize acquisition day only.
    dt = pd.to_datetime(
        work[date_col],
        errors="coerce"
    )

    work["_acq_day"] = dt.dt.date.astype(str)

    date_beam_table = (
        work.groupby(
            ["_acq_day", beam_col],
            dropna=False
        )
        .agg(
            n=("shot_number", "size"),
            blocks=("cv_block_id", "nunique"),
            x_min=("utm_x", "min"),
            x_max=("utm_x", "max"),
            y_min=("utm_y", "min"),
            y_max=("utm_y", "max"),
        )
        .reset_index()
        .sort_values(
            ["_acq_day", beam_col]
        )
    )

    date_beam_table.to_csv(
        DATEBEAM,
        index=False
    )

# ------------------------------------------------------------
# Candidate composite track keys
# ------------------------------------------------------------

candidate_keys = []

if "acquisition_date" in df.columns and "beam" in df.columns:

    acq_day = pd.to_datetime(
        df["acquisition_date"],
        errors="coerce"
    ).dt.date.astype(str)

    key = (
        acq_day.astype(str)
        + "__"
        + df["beam"].astype(str)
    )

    candidate_keys.append({
        "candidate_track_key":
            "acquisition_date_day + beam",
        "n_groups":
            int(key.nunique()),
        "min_group_n":
            int(key.value_counts().min()),
        "median_group_n":
            float(key.value_counts().median()),
        "max_group_n":
            int(key.value_counts().max()),
    })

if "gedi_datetime" in df.columns and "beam" in df.columns:

    gedi_day = pd.to_datetime(
        df["gedi_datetime"],
        errors="coerce"
    ).dt.date.astype(str)

    key = (
        gedi_day.astype(str)
        + "__"
        + df["beam"].astype(str)
    )

    candidate_keys.append({
        "candidate_track_key":
            "gedi_datetime_day + beam",
        "n_groups":
            int(key.nunique()),
        "min_group_n":
            int(key.value_counts().min()),
        "median_group_n":
            float(key.value_counts().median()),
        "max_group_n":
            int(key.value_counts().max()),
    })

candidate_key_df = pd.DataFrame(candidate_keys)

candidate_key_path = (
    OUT
    / "C2_00_CANDIDATE_TRACK_KEYS.csv"
)

candidate_key_df.to_csv(
    candidate_key_path,
    index=False
)

# ------------------------------------------------------------
# Report
# ------------------------------------------------------------

with REPORT.open("w", encoding="utf-8") as f:

    f.write("=" * 90 + "\n")
    f.write("C2 STEP 0 — SPATIAL / TRACK UNIT AUDIT\n")
    f.write("=" * 90 + "\n\n")

    f.write("PURPOSE\n")
    f.write("-" * 90 + "\n")
    f.write(
        "Identify reliable independent spatial and GEDI acquisition/track units "
        "before any C2 model is fitted.\n\n"
    )

    f.write("INPUT\n")
    f.write("-" * 90 + "\n")
    f.write(f"{DATA}\n")
    f.write(f"SHA256: {sha256_file(DATA)}\n")
    f.write(f"Rows: {len(df)}\n")
    f.write(f"Unique shots: {df['shot_number'].nunique()}\n")
    f.write(f"Columns: {len(df.columns)}\n\n")

    f.write("CORE SPATIAL UNITS\n")
    f.write("-" * 90 + "\n")
    f.write(f"Unique 3-km cv_block_id: {df['cv_block_id'].nunique()}\n")
    f.write(f"Original spatial folds: {sorted(df['spatial_fold'].unique().tolist())}\n")

    if "repeat_group_id" in df.columns:
        f.write(
            f"Unique repeat_group_id: "
            f"{df['repeat_group_id'].nunique()}\n"
        )

    if "beam" in df.columns:
        f.write(
            f"Unique beam values: "
            f"{df['beam'].nunique()}\n"
        )

    if "acquisition_date" in df.columns:
        f.write(
            f"Unique acquisition_date values: "
            f"{df['acquisition_date'].nunique()}\n"
        )

    f.write("\n")

    f.write("POTENTIAL TRACK / SOURCE COLUMNS\n")
    f.write("-" * 90 + "\n")

    if len(potential_df):
        f.write(
            potential_df.to_string(
                index=False
            )
        )
    else:
        f.write("None detected.")

    f.write("\n\n")

    f.write("CANDIDATE COMPOSITE TRACK KEYS\n")
    f.write("-" * 90 + "\n")

    if len(candidate_key_df):
        f.write(
            candidate_key_df.to_string(
                index=False
            )
        )
    else:
        f.write(
            "No date+beam candidate key could be built."
        )

    f.write("\n\n")

    f.write("3-KM BLOCK SIZE DISTRIBUTION\n")
    f.write("-" * 90 + "\n")
    f.write(
        block_counts["n"]
        .describe()
        .to_string()
    )

    f.write("\n\n")

    if len(repeat_counts):

        f.write("REPEAT-GROUP SIZE DISTRIBUTION\n")
        f.write("-" * 90 + "\n")

        f.write(
            repeat_counts["n"]
            .describe()
            .to_string()
        )

        f.write("\n\n")

    if len(date_beam_table):

        f.write("DATE × BEAM GROUP SUMMARY\n")
        f.write("-" * 90 + "\n")

        f.write(
            date_beam_table.to_string(
                index=False
            )
        )

        f.write("\n\n")

    f.write("C2 STATUS\n")
    f.write("-" * 90 + "\n")

    f.write(
        "NO MODEL WAS FITTED.\n"
        "NO C1 FILE WAS MODIFIED.\n"
        "NO C2 VALIDATION GEOMETRY HAS YET BEEN CHOSEN FROM PERFORMANCE.\n"
    )

print("")
print("=" * 80)
print("C2 STEP 0 COMPLETE")
print("SPATIAL / TRACK UNIT AUDIT")
print("=" * 80)

print("")
print("Frozen observations:", len(df))
print("Unique shots:", df["shot_number"].nunique())
print("Unique 3-km blocks:", df["cv_block_id"].nunique())

if "repeat_group_id" in df.columns:
    print(
        "Unique repeat groups:",
        df["repeat_group_id"].nunique()
    )

if "beam" in df.columns:
    print(
        "Unique beams:",
        df["beam"].nunique()
    )

if "acquisition_date" in df.columns:
    print(
        "Unique acquisition dates:",
        df["acquisition_date"].nunique()
    )

print("")
print("Candidate track keys:")
print(
    candidate_key_df.to_string(
        index=False
    )
)

print("")
print("Potential track/source columns:")
print(
    potential_df.to_string(
        index=False
    )
)

print("")
print("Report:")
print(REPORT)

print("")
print("NO MODEL WAS FITTED.")
print("C1 REMAINS FROZEN.")
print("=" * 80)
