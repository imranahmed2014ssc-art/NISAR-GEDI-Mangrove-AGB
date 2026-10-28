from pathlib import Path
import numpy as np
import pandas as pd
import rasterio


ROOT = Path(r"D:\NISAR_Mangrove_Paper")

GMW = (
    ROOT
    / "01_AOI"
    / "Mangrove_Mask"
    / "Processed"
    / "GMW_2025_NISAR_grid_mask.tif"
)

POINTS = (
    ROOT
    / "06_Results"
    / "Reviewer_Validation"
    / "C6_Support_Mixing"
    / "C6_01_Support_Metrics"
    / "C6_01_POINT_SUPPORT_METRICS.csv"
)

OUT = (
    ROOT
    / "06_Results"
    / "Reviewer_Validation"
    / "C6_Support_Mixing"
    / "C6_01A_GMW_QC"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT = (
    OUT
    / "C6_01A_GMW_ENCODING_QC_REPORT.txt"
)


# ============================================================
# READ SUPPORT TABLE
# ============================================================

df = pd.read_csv(
    POINTS
)

if len(df) != 1641:
    raise RuntimeError(
        f"Expected 1641 support rows; found {len(df)}."
    )


# ============================================================
# READ RASTER
# ============================================================

with rasterio.open(
    GMW
) as src:

    raw = src.read(
        1,
        masked=False,
    )

    masked = src.read(
        1,
        masked=True,
    )

    nodata = src.nodata

    raw_unique, raw_counts = np.unique(
        raw,
        return_counts=True,
    )

    masked_values = masked.compressed()

    valid_unique, valid_counts = np.unique(
        masked_values,
        return_counts=True,
    )


    raw_inventory = pd.DataFrame({
        "value":
            raw_unique,

        "count":
            raw_counts,

        "percent_all_pixels":
            raw_counts
            /
            raw.size
            *
            100.0,
    })


    valid_inventory = pd.DataFrame({
        "value":
            valid_unique,

        "count":
            valid_counts,

        "percent_valid_pixels":
            valid_counts
            /
            len(masked_values)
            *
            100.0,
    })


# ============================================================
# SUPPORT VALID-FRACTION QC
# ============================================================

support_rows = []

for support in [
    "3x3",
    "5x5",
    "9x9",
]:

    valid_col = (
        f"gmw_valid_fraction_{support}"
    )

    frac_col = (
        f"gmw_mangrove_fraction_{support}"
    )

    valid = pd.to_numeric(
        df[
            valid_col
        ],
        errors="coerce",
    )

    frac = pd.to_numeric(
        df[
            frac_col
        ],
        errors="coerce",
    )


    support_rows.append({
        "support":
            support,

        "valid_fraction_min":
            float(
                valid.min()
            ),

        "valid_fraction_q05":
            float(
                valid.quantile(
                    0.05
                )
            ),

        "valid_fraction_median":
            float(
                valid.median()
            ),

        "valid_fraction_q95":
            float(
                valid.quantile(
                    0.95
                )
            ),

        "valid_fraction_max":
            float(
                valid.max()
            ),

        "n_valid_fraction_lt_1":
            int(
                (
                    valid
                    <
                    1.0
                ).sum()
            ),

        "mangrove_fraction_min":
            float(
                frac.min()
            ),

        "mangrove_fraction_median":
            float(
                frac.median()
            ),

        "mangrove_fraction_max":
            float(
                frac.max()
            ),
    })


support_df = pd.DataFrame(
    support_rows
)


# ============================================================
# COMPARE MASTER MANGROVE FRACTION
# ============================================================

comparison = None

if (
    "master_mangrove_fraction"
    in
    df.columns
):

    master_fraction = pd.to_numeric(
        df[
            "master_mangrove_fraction"
        ],
        errors="coerce",
    )

    recomputed = pd.to_numeric(
        df[
            "gmw_mangrove_fraction_9x9"
        ],
        errors="coerce",
    )

    valid = (
        master_fraction.notna()
        &
        recomputed.notna()
    )


    comparison = {
        "master_min":
            float(
                master_fraction[
                    valid
                ].min()
            ),

        "master_median":
            float(
                master_fraction[
                    valid
                ].median()
            ),

        "master_max":
            float(
                master_fraction[
                    valid
                ].max()
            ),

        "recomputed_min":
            float(
                recomputed[
                    valid
                ].min()
            ),

        "recomputed_median":
            float(
                recomputed[
                    valid
                ].median()
            ),

        "recomputed_max":
            float(
                recomputed[
                    valid
                ].max()
            ),

        "mean_abs_difference":
            float(
                np.mean(
                    np.abs(
                        master_fraction[
                            valid
                        ]
                        -
                        recomputed[
                            valid
                        ]
                    )
                )
            ),

        "max_abs_difference":
            float(
                np.max(
                    np.abs(
                        master_fraction[
                            valid
                        ]
                        -
                        recomputed[
                            valid
                        ]
                    )
                )
            ),
    }


# ============================================================
# DECISION FLAGS
# ============================================================

nodata_is_zero = bool(
    nodata == 0
)

zero_present_raw = bool(
    np.any(
        raw_unique
        ==
        0
    )
)

zero_present_valid = bool(
    np.any(
        valid_unique
        ==
        0
    )
)

all_9x9_valid = bool(
    np.allclose(
        pd.to_numeric(
            df[
                "gmw_valid_fraction_9x9"
            ],
            errors="coerce",
        ),
        1.0,
    )
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
        "C6 STEP 1A - GMW RASTER ENCODING / "
        "DENOMINATOR QC\n"
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
        "NO C6 SUPPORT THRESHOLD WAS CHANGED.\n"
    )

    f.write(
        "Existing C6 Step 1 outputs were audited "
        "read-only.\n\n"
    )


    f.write(
        "RASTER METADATA\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    f.write(
        f"Raster: {GMW}\n"
    )

    f.write(
        f"NoData value: {nodata}\n"
    )

    f.write(
        f"NoData is zero: {nodata_is_zero}\n"
    )

    f.write(
        f"Zero present in raw raster: "
        f"{zero_present_raw}\n"
    )

    f.write(
        f"Zero present among valid pixels: "
        f"{zero_present_valid}\n\n"
    )


    f.write(
        "RAW RASTER VALUE INVENTORY\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    f.write(
        raw_inventory.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )


    f.write(
        "VALID/UNMASKED VALUE INVENTORY\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    f.write(
        valid_inventory.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )


    f.write(
        "SUPPORT-WINDOW VALIDITY\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    f.write(
        support_df.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )


    f.write(
        "MASTER VS RECOMPUTED 9X9 FRACTION\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    if comparison is None:

        f.write(
            "Comparison unavailable.\n"
        )

    else:

        for key, value in (
            comparison.items()
        ):

            f.write(
                f"{key}: {value}\n"
            )


    f.write(
        "\n"
    )


    f.write(
        "QC INTERPRETATION\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    if nodata_is_zero:

        f.write(
            "IMPORTANT: raster value 0 is configured "
            "as NoData. A masked-array denominator "
            "would therefore exclude those cells.\n"
        )

        f.write(
            "The C6 Step 1 mangrove fractions must NOT "
            "be frozen until the intended binary-mask "
            "semantics are reconstructed.\n"
        )

    elif (
        zero_present_valid
        and
        all_9x9_valid
    ):

        f.write(
            "Zero is a valid raster class and every "
            "9x9 support window has complete raster "
            "coverage. The 100% interior result is "
            "therefore structurally plausible.\n"
        )

    elif not all_9x9_valid:

        f.write(
            "Some 9x9 windows contain masked/NoData "
            "cells. Mangrove-fraction denominator "
            "semantics require review before C6 is "
            "frozen.\n"
        )

    else:

        f.write(
            "Raster encoding requires manual review "
            "before accepting the 100% interior result.\n"
        )


# ============================================================
# CONSOLE
# ============================================================

print("")
print("=" * 84)
print("C6 STEP 1A COMPLETE")
print("GMW ENCODING / DENOMINATOR QC")
print("=" * 84)

print("")
print(
    "NoData:",
    nodata
)

print(
    "Zero present raw:",
    zero_present_raw
)

print(
    "Zero present valid:",
    zero_present_valid
)

print("")
print("Support validity:")
print(
    support_df.to_string(
        index=False
    )
)

print("")
if comparison is not None:

    print(
        "Master vs recomputed:"
    )

    for key, value in (
        comparison.items()
    ):

        print(
            key,
            "=",
            value
        )

print("")
print("Report:")
print(REPORT)

print("")
print("NO MODEL WAS FITTED.")
print("NO THRESHOLD WAS CHANGED.")
print("=" * 84)
