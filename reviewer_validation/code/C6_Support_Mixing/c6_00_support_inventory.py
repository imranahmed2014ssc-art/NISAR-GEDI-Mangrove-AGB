from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import h5py


ROOT = Path(r"D:\NISAR_Mangrove_Paper")

MASTER = (
    ROOT
    / "04_Matched_Data"
    / "Predictor_Expansion"
    / "Phase8C2"
    / "Tables"
    / "GEDI_NISAR_expanded_predictors_1641.csv"
)

NISAR_ROOT = (
    ROOT
    / "02_NISAR_GCOV"
)

OUT = (
    ROOT
    / "06_Results"
    / "Reviewer_Validation"
    / "C6_Support_Mixing"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

H5_OUT = OUT / "C6_00_HDF5_RELEVANT_DATASETS.csv"
GMW_OUT = OUT / "C6_00_GMW_CANDIDATES.csv"
MASTER_OUT = OUT / "C6_00_MASTER_QC_COLUMNS.csv"
REPORT = OUT / "C6_00_SUPPORT_INVENTORY_REPORT.txt"


# ============================================================
# MASTER
# ============================================================

df = pd.read_csv(MASTER)

if len(df) != 1641:
    raise RuntimeError(
        f"Expected 1641 master rows; found {len(df)}."
    )

if df["shot_number"].nunique() != 1641:
    raise RuntimeError(
        "shot_number is not unique."
    )


# ============================================================
# MASTER HASH
# ============================================================

sha = hashlib.sha256()

with MASTER.open("rb") as f:
    while True:
        chunk = f.read(1024 * 1024)

        if not chunk:
            break

        sha.update(chunk)

master_hash = sha.hexdigest()


# ============================================================
# MASTER QC / SUPPORT-LIKE COLUMNS
# ============================================================

keywords = [
    "mask",
    "valid",
    "water",
    "mangrove",
    "look",
    "incidence",
    "rtc",
    "shadow",
    "layover",
    "fraction",
    "edge",
    "support",
    "quality",
]

master_rows = []

for col in df.columns:

    lower = col.lower()

    if any(
        key in lower
        for key in keywords
    ):

        x = df[col]

        master_rows.append({
            "column":
                col,

            "dtype":
                str(x.dtype),

            "n":
                len(x),

            "n_missing":
                int(x.isna().sum()),

            "n_unique":
                int(x.nunique(dropna=True)),
        })


master_qc_df = pd.DataFrame(
    master_rows
)

master_qc_df.to_csv(
    MASTER_OUT,
    index=False,
)


# ============================================================
# FIND MAIN-PROJECT NISAR H5 FILES
#
# D:\new_nisar_data IS DELIBERATELY NOT SCANNED.
# ============================================================

h5_files = sorted(
    list(
        NISAR_ROOT.rglob("*.h5")
    )
    +
    list(
        NISAR_ROOT.rglob("*.hdf5")
    )
)


# ============================================================
# INVENTORY RELEVANT HDF5 DATASETS
# ============================================================

h5_keywords = [
    "numberoflooks",
    "number_of_looks",
    "mask",
    "valid",
    "water",
    "layover",
    "shadow",
    "rtc",
    "incidence",
    "quality",
    "xcoordinates",
    "ycoordinates",
    "spacing",
    "projection",
]


h5_rows = []

pixel_spacing_rows = []


for h5_path in h5_files:

    print("")
    print(
        "Scanning:",
        h5_path
    )

    with h5py.File(
        h5_path,
        "r",
    ) as h5:

        def visitor(
            name,
            obj,
        ):

            if not isinstance(
                obj,
                h5py.Dataset,
            ):
                return

            lower = name.lower()

            if any(
                key in lower
                for key in h5_keywords
            ):

                h5_rows.append({
                    "file":
                        str(h5_path),

                    "dataset":
                        name,

                    "shape":
                        str(obj.shape),

                    "dtype":
                        str(obj.dtype),

                    "units":
                        str(
                            obj.attrs.get(
                                "units",
                                "",
                            )
                        ),

                    "description":
                        str(
                            obj.attrs.get(
                                "description",
                                "",
                            )
                        )[:500],
                })


            # --------------------------------------------
            # Estimate coordinate spacing where feasible.
            # --------------------------------------------

            leaf = (
                name
                .split("/")[-1]
                .lower()
            )

            if (
                leaf
                in [
                    "xcoordinates",
                    "ycoordinates",
                ]
                and
                len(obj.shape) == 1
                and
                obj.shape[0] >= 2
                and
                obj.shape[0] <= 1000000
            ):

                try:

                    values = np.asarray(
                        obj[:],
                        dtype=float,
                    )

                    finite = values[
                        np.isfinite(values)
                    ]

                    if len(finite) >= 2:

                        diffs = np.diff(
                            finite
                        )

                        diffs = diffs[
                            np.isfinite(diffs)
                        ]

                        if len(diffs):

                            pixel_spacing_rows.append({
                                "file":
                                    str(h5_path),

                                "dataset":
                                    name,

                                "coordinate":
                                    leaf,

                                "n":
                                    len(finite),

                                "median_spacing":
                                    float(
                                        np.median(
                                            np.abs(diffs)
                                        )
                                    ),

                                "min_spacing":
                                    float(
                                        np.min(
                                            np.abs(diffs)
                                        )
                                    ),

                                "max_spacing":
                                    float(
                                        np.max(
                                            np.abs(diffs)
                                        )
                                    ),
                            })

                except Exception as exc:

                    pixel_spacing_rows.append({
                        "file":
                            str(h5_path),

                        "dataset":
                            name,

                        "coordinate":
                            leaf,

                        "n":
                            np.nan,

                        "median_spacing":
                            np.nan,

                        "min_spacing":
                            np.nan,

                        "max_spacing":
                            np.nan,

                        "error":
                            str(exc),
                    })


        h5.visititems(
            visitor
        )


h5_df = pd.DataFrame(
    h5_rows
)

h5_df.to_csv(
    H5_OUT,
    index=False,
)


spacing_df = pd.DataFrame(
    pixel_spacing_rows
)


# ============================================================
# FIND GMW / MANGROVE CANDIDATES
# ============================================================

gmw_candidates = []

search_roots = [
    ROOT / "01_AOI",
    ROOT / "03_GEDI_L4A_V3",
]


for search_root in search_roots:

    if not search_root.exists():
        continue

    for path in search_root.rglob("*"):

        if not path.is_file():
            continue

        lower = path.name.lower()

        if (
            "gmw" in lower
            or
            "mangrove" in lower
        ):

            gmw_candidates.append({
                "path":
                    str(path),

                "filename":
                    path.name,

                "suffix":
                    path.suffix,

                "size_bytes":
                    path.stat().st_size,
            })


gmw_df = pd.DataFrame(
    gmw_candidates
)

gmw_df.to_csv(
    GMW_OUT,
    index=False,
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
        "C6 STEP 0 - SUPPORT / EDGE-MIXING "
        "DATA INVENTORY\n"
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
        "NO SUPPORT OR EDGE THRESHOLD WAS DEFINED.\n"
    )

    f.write(
        "NO DATA WERE MODIFIED.\n"
    )

    f.write(
        "D:\\new_nisar_data was NOT scanned or touched.\n"
    )

    f.write(
        "C1-C5 remain frozen.\n\n"
    )


    f.write(
        "MASTER TABLE\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    f.write(
        f"Rows: {len(df)}\n"
    )

    f.write(
        f"Unique shots: "
        f"{df['shot_number'].nunique()}\n"
    )

    f.write(
        f"SHA256: {master_hash}\n\n"
    )


    f.write(
        "MASTER SUPPORT / QC-LIKE COLUMNS\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    if len(master_qc_df):

        f.write(
            master_qc_df.to_string(
                index=False
            )
        )

    else:

        f.write(
            "No matching support/QC-like columns found."
        )

    f.write(
        "\n\n"
    )


    f.write(
        "MAIN-PROJECT NISAR HDF5 FILES\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    f.write(
        f"HDF5 files found: {len(h5_files)}\n"
    )

    for path in h5_files:

        f.write(
            str(path)
            + "\n"
        )

    f.write(
        "\n"
    )


    f.write(
        "RELEVANT HDF5 DATASETS\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    if len(h5_df):

        f.write(
            h5_df.to_string(
                index=False
            )
        )

    else:

        f.write(
            "No keyword-matched HDF5 datasets found."
        )

    f.write(
        "\n\n"
    )


    f.write(
        "COORDINATE / PIXEL-SPACING AUDIT\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    if len(spacing_df):

        f.write(
            spacing_df.to_string(
                index=False
            )
        )

    else:

        f.write(
            "No readable x/y coordinate arrays found."
        )

    f.write(
        "\n\n"
    )


    f.write(
        "GMW / MANGROVE FILE CANDIDATES\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    if len(gmw_df):

        f.write(
            gmw_df.to_string(
                index=False
            )
        )

    else:

        f.write(
            "No GMW/mangrove filename candidates found."
        )

    f.write(
        "\n\n"
    )


    f.write(
        "NEXT C6 DECISION\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    f.write(
        "This inventory will determine whether C6 can "
        "directly use GCOV validity/number-of-looks masks "
        "and the existing GMW 2025 mangrove layer.\n"
    )

    f.write(
        "Only after this inventory is reviewed will "
        "interior/edge and support-mixing definitions "
        "be frozen.\n"
    )


# ============================================================
# CONSOLE
# ============================================================

print("")
print("=" * 84)
print("C6 STEP 0 COMPLETE")
print("SUPPORT / EDGE-MIXING INVENTORY")
print("=" * 84)

print("")
print(
    "Master rows:",
    len(df)
)

print(
    "Unique shots:",
    df["shot_number"].nunique()
)

print(
    "Main-project HDF5 files:",
    len(h5_files)
)

print(
    "Relevant HDF5 datasets:",
    len(h5_df)
)

print(
    "GMW/mangrove candidates:",
    len(gmw_df)
)

print("")
print("Pixel-spacing audit:")

if len(spacing_df):
    print(
        spacing_df.to_string(
            index=False
        )
    )
else:
    print(
        "No coordinate spacing results."
    )

print("")
print("Report:")
print(REPORT)

print("")
print("NO MODEL WAS FITTED.")
print("NO THRESHOLD WAS DEFINED.")
print("C1-C5 REMAIN FROZEN.")
print("=" * 84)
