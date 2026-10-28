from pathlib import Path
import shutil
import hashlib
import csv


ROOT = Path(r"D:\NISAR_Mangrove_Paper")

GITHUB = (
    ROOT
    / "github_repo"
)

SRC_CODE = (
    ROOT
    / "05_ML_DL"
    / "Reviewer_Validation"
)

SRC_RESULTS = (
    ROOT
    / "06_Results"
    / "Reviewer_Validation"
)

DEST = (
    GITHUB
    / "reviewer_validation"
)

CODE_DEST = (
    DEST
    / "code"
)

RESULT_DEST = (
    DEST
    / "results"
)

DOC_DEST = (
    DEST
    / "docs"
)


# ============================================================
# ALLOWED TEXT / REPRODUCIBILITY FILES
# ============================================================

ALLOWED_SUFFIXES = {
    ".py",
    ".csv",
    ".txt",
    ".json",
    ".md",
    ".yaml",
    ".yml",
}


# ============================================================
# CLEAN ONLY THE GENERATED REVIEWER_VALIDATION PACKAGE
# ============================================================

if DEST.exists():
    shutil.rmtree(
        DEST
    )


CODE_DEST.mkdir(
    parents=True,
    exist_ok=True,
)

RESULT_DEST.mkdir(
    parents=True,
    exist_ok=True,
)

DOC_DEST.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path):

    h = hashlib.sha256()

    with path.open("rb") as f:

        while True:

            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            h.update(
                chunk
            )

    return h.hexdigest()


def copy_allowed_tree(
    source,
    destination,
    include_prefixes=None,
):

    copied = []

    for path in source.rglob("*"):

        if not path.is_file():
            continue

        if (
            path.suffix.lower()
            not in
            ALLOWED_SUFFIXES
        ):
            continue

        rel = path.relative_to(
            source
        )

        if include_prefixes is not None:

            top = rel.parts[0]

            if top not in include_prefixes:
                continue

        target = (
            destination
            / rel
        )

        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copy2(
            path,
            target,
        )

        copied.append(
            target
        )

    return copied


# ============================================================
# C1-C6 CODE
# ============================================================

code_prefixes = {
    "C1_Independent_Validation",
    "C2_Spatial_Transferability",
    "C3_Temporal_Mismatch",
    "C4_GEDI_Uncertainty",
    "C5_Multiscale_Ablation",
    "C6_Support_Mixing",
    "C8_Reproducibility",
}


code_files = copy_allowed_tree(
    SRC_CODE,
    CODE_DEST,
    include_prefixes=code_prefixes,
)


# ============================================================
# C1-C6 + C8 RESULTS
# ============================================================

result_prefixes = {
    "C1_Independent_Validation",
    "C2_Spatial_Transferability",
    "C3_Temporal_Mismatch",
    "C4_GEDI_Uncertainty",
    "C5_Multiscale_Ablation",
    "C6_Support_Mixing",
    "C8_Reproducibility",
}


result_files = copy_allowed_tree(
    SRC_RESULTS,
    RESULT_DEST,
    include_prefixes=result_prefixes,
)


# ============================================================
# SOURCE PRODUCT MANIFEST
# ============================================================

manifest = DOC_DEST / "SOURCE_PRODUCTS.md"

manifest.write_text(
    """# Source Product Manifest

## Primary NISAR GCOV

`NISAR_L2_PR_GCOV_030_170_A_013_4005_DHDH_A_20260917T232917_20260917T232952_P05023_F_F_J_001.h5`

Acquisition start: 2026-09-17 23:29:17 UTC

Product level: NISAR L2 GCOV PROVISIONAL

Frequency used: Frequency A

Grid spacing used in analysis: 10 m

Primary covariance terms used: HHHH and HVHV-derived predictors.


## Temporal-sensitivity NISAR GCOV

`NISAR_L2_PR_GCOV_023_170_A_013_4005_DHDH_A_20260625T232921_20260625T232956_P05023_N_F_J_001.h5`

Acquisition start: 2026-06-25 23:29:21 UTC

This scene was used only in the isolated temporal-sensitivity experiment.


## GEDI

GEDI L4A Version 3 AGBD observations were used as reference data.

Twenty-two downloaded/raw GEDI granules were processed. The final 1,641 matched observations are represented by eight retained acquisition/source-file groups after filtering and matching.


## GMW

Global Mangrove Watch 2025 was used for the mangrove mask and support-mixing analysis.


## Data availability

Raw NISAR, GEDI, and Global Mangrove Watch source data are not redistributed in this repository. Users should obtain the source products from their original data providers using the identifiers and acquisition information given above.

The repository contains code, frozen validation memberships, model outputs, OOF predictions, QC tables, checksums, and documentation needed to reproduce the reported analytical workflow once the source data are obtained.
""",
    encoding="utf-8",
)


# ============================================================
# REVIEWER VALIDATION README
# ============================================================

rv_readme = DEST / "README.md"

rv_readme.write_text(
    """# Reviewer-validation reproducibility package

This directory contains the frozen analyses added during major revision of the NISAR–GEDI mangrove AGBD manuscript.

## Validation modules

- `C1_Independent_Validation` — selection-aware nested geographic validation
- `C2_Spatial_Transferability` — buffer, pass-holdout, and randomized block sensitivity
- `C3_Temporal_Mismatch` — temporal-gap, June-scene, hydrometeorological, and stability analyses
- `C4_GEDI_Uncertainty` — GEDI `agbd_se` uncertainty characterization
- `C5_Multiscale_Ablation` — predeclared multiscale predictor ablation
- `C6_Support_Mixing` — mangrove-support and GCOV-quality sensitivity
- `C8_Reproducibility` — reproducibility audit, environment snapshot, and checksums

## Frozen analytical principles

No C1–C6 predictor family, spatial split, support threshold, or uncertainty stratum was redefined after viewing the corresponding performance results.

C1–C6 are within-study validation analyses and do not constitute independent external field validation.

The manuscript does not present an operational wall-to-wall biomass product.

## Main frozen modeling table

Final matched sample: 1,641 GEDI–NISAR observations.

Frozen master-table SHA256:

`de9fc736ec77d38dae527020b2d037c4567f3c814f874d9843da230c6bf67dc0`

## GitHub release

This reviewer-validation package is intended for release as **v1.0.2**, superseding v1.0.1 for the revised manuscript reproducibility record.

See `docs/SOURCE_PRODUCTS.md` for source-product identifiers.
""",
    encoding="utf-8",
)


# ============================================================
# FILE MANIFEST + SHA256
# ============================================================

all_files = sorted(
    p
    for p in DEST.rglob("*")
    if p.is_file()
)


manifest_csv = (
    DOC_DEST
    / "REVIEWER_VALIDATION_SHA256.csv"
)


with manifest_csv.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.writer(
        f
    )

    writer.writerow([
        "relative_path",
        "size_bytes",
        "sha256",
    ])

    for path in all_files:

        writer.writerow([
            path.relative_to(
                GITHUB
            ).as_posix(),
            path.stat().st_size,
            sha256_file(
                path
            ),
        ])


# ============================================================
# PACKAGE SUMMARY
# ============================================================

total_bytes = sum(
    p.stat().st_size
    for p in all_files
)


summary = (
    DOC_DEST
    / "PACKAGE_SUMMARY.txt"
)


summary.write_text(
    "\n".join([
        "C8 v1.0.2 reviewer-validation package",
        "",
        f"Code files copied: {len(code_files)}",
        f"Result/document files copied: {len(result_files)}",
        f"Total packaged files: {len(all_files)}",
        f"Total package bytes: {total_bytes}",
        "",
        "Raw NISAR/GEDI/GMW data copied: NO",
        "D:\\new_nisar_data modified: NO",
        "Scientific results modified: NO",
        "",
        "Target GitHub release: v1.0.2",
    ]),
    encoding="utf-8",
)


print("")
print("=" * 84)
print("C8 STEP 2 PACKAGE BUILD COMPLETE")
print("=" * 84)

print(
    "Code files:",
    len(code_files)
)

print(
    "Result/document files:",
    len(result_files)
)

print(
    "Total package files:",
    len(all_files)
)

print(
    "Total bytes:",
    total_bytes
)

print("")
print(
    "Package:",
    DEST
)

print("")
print(
    "Raw source data copied: NO"
)

print(
    "Scientific results modified: NO"
)

print(
    "D:\\new_nisar_data modified: NO"
)

print("=" * 84)
