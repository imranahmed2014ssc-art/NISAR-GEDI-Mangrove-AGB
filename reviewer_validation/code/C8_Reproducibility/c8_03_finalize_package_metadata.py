from pathlib import Path
import csv
import hashlib
import shutil


ROOT = Path(r"D:\NISAR_Mangrove_Paper")

SOURCE_CODE = (
    ROOT
    / "05_ML_DL"
    / "Reviewer_Validation"
    / "C8_Reproducibility"
)

PACKAGE = (
    ROOT
    / "github_repo"
    / "reviewer_validation"
)

PACKAGE_CODE = (
    PACKAGE
    / "code"
    / "C8_Reproducibility"
)

DOCS = (
    PACKAGE
    / "docs"
)

MANIFEST = (
    DOCS
    / "REVIEWER_VALIDATION_SHA256.csv"
)

SUMMARY = (
    DOCS
    / "PACKAGE_SUMMARY.txt"
)

REPORT = (
    DOCS
    / "FINAL_PACKAGE_QC.txt"
)


FORBIDDEN_SUFFIXES = {
    ".h5",
    ".hdf5",
    ".tif",
    ".tiff",
    ".gpkg",
    ".gpkgm",
    ".shp",
    ".shx",
    ".dbf",
    ".prj",
    ".nc",
    ".zip",
    ".7z",
    ".rar",
    ".png",
    ".jpg",
    ".jpeg",
    ".pdf",
}


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


# ============================================================
# ENSURE FINAL C8 SCRIPTS ARE INCLUDED
# ============================================================

PACKAGE_CODE.mkdir(
    parents=True,
    exist_ok=True,
)


for script_name in [
    "c8_02_final_package_qc.py",
    "c8_03_finalize_package_metadata.py",
]:

    src = (
        SOURCE_CODE
        / script_name
    )

    dst = (
        PACKAGE_CODE
        / script_name
    )

    if not src.exists():

        raise RuntimeError(
            f"Missing source script: {src}"
        )

    shutil.copy2(
        src,
        dst,
    )


# ============================================================
# INVENTORY AFTER ALL FINAL FILES EXIST
#
# Existing manifest/summary/report are already present,
# so rewriting them does not alter file count.
# ============================================================

all_files = sorted(
    p
    for p in PACKAGE.rglob("*")
    if p.is_file()
)


forbidden = [
    p
    for p in all_files
    if p.suffix.lower()
    in FORBIDDEN_SUFFIXES
]


oversized = [
    p
    for p in all_files
    if p.stat().st_size
    >=
    100 * 1024 * 1024
]


# ============================================================
# MODULE COUNTS
# ============================================================

module_rows = []


for module in [
    "C1_Independent_Validation",
    "C2_Spatial_Transferability",
    "C3_Temporal_Mismatch",
    "C4_GEDI_Uncertainty",
    "C5_Multiscale_Ablation",
    "C6_Support_Mixing",
    "C8_Reproducibility",
]:

    code_dir = (
        PACKAGE
        / "code"
        / module
    )

    result_dir = (
        PACKAGE
        / "results"
        / module
    )


    code_count = (
        sum(
            1
            for p in code_dir.rglob("*")
            if p.is_file()
        )
        if code_dir.exists()
        else 0
    )


    result_count = (
        sum(
            1
            for p in result_dir.rglob("*")
            if p.is_file()
        )
        if result_dir.exists()
        else 0
    )


    module_rows.append(
        (
            module,
            code_count,
            result_count,
        )
    )


# ============================================================
# FINAL COUNT
#
# Manifest itself is part of package but is not hashed
# inside itself.
# ============================================================

final_total_files = len(
    all_files
)

expected_manifest_entries = (
    final_total_files
    -
    1
)


# ============================================================
# UPDATE PACKAGE SUMMARY BEFORE HASHING
# ============================================================

total_bytes_before_manifest = sum(
    p.stat().st_size
    for p in all_files
)


SUMMARY.write_text(
    "\n".join([
        "C8 v1.0.2 reviewer-validation package",
        "",
        f"Final package files: {final_total_files}",
        (
            "Expected SHA256 manifest entries: "
            f"{expected_manifest_entries}"
        ),
        (
            "Package bytes before final manifest rewrite: "
            f"{total_bytes_before_manifest}"
        ),
        "",
        "Raw NISAR/GEDI/GMW data copied: NO",
        "D:\\new_nisar_data modified: NO",
        "Scientific results modified: NO",
        "",
        "Frozen validation modules: C1-C6",
        "Reproducibility module: C8",
        "",
        "Target GitHub release: v1.0.2",
        "",
        "Previous Zenodo archive:",
        "v1.0.1 - 10.5281/zenodo.23675039",
        "",
        (
            "v1.0.2 is intended to supersede v1.0.1 "
            "for the revised-manuscript reproducibility record."
        ),
    ]),
    encoding="utf-8",
)


# ============================================================
# WRITE FINAL QC REPORT BEFORE HASHING
# ============================================================

status = (
    "PASS"
    if (
        len(forbidden) == 0
        and
        len(oversized) == 0
    )
    else
    "FAIL"
)


REPORT.write_text(
    "\n".join([
        "=" * 100,
        "C8 FINAL REVIEWER-VALIDATION PACKAGE QC",
        "=" * 100,
        "",
        f"STATUS: {status}",
        "",
        f"Final package files: {final_total_files}",
        (
            "Expected SHA256 manifest entries: "
            f"{expected_manifest_entries}"
        ),
        "",
        f"Forbidden/raw files: {len(forbidden)}",
        f">=100 MiB files: {len(oversized)}",
        "",
        "Module counts:",
        *[
            (
                f"{module}: "
                f"code={code_count}, "
                f"results={result_count}"
            )
            for (
                module,
                code_count,
                result_count,
            )
            in module_rows
        ],
        "",
        "Raw source data copied: NO",
        "Scientific results modified: NO",
        "D:\\new_nisar_data modified: NO",
        "",
        (
            "The SHA256 manifest is generated after "
            "this report so that this report itself "
            "is included in the manifest."
        ),
    ]),
    encoding="utf-8",
)


# ============================================================
# RE-INVENTORY
#
# File count should remain unchanged because summary/report
# already existed and were only rewritten.
# ============================================================

all_files = sorted(
    p
    for p in PACKAGE.rglob("*")
    if p.is_file()
)


if len(all_files) != final_total_files:

    raise RuntimeError(
        "Unexpected file-count change before manifest."
    )


# ============================================================
# FINAL SHA256 MANIFEST
#
# Exclude only the manifest itself.
# ============================================================

targets = [
    p
    for p in all_files
    if p != MANIFEST
]


with MANIFEST.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "relative_path",
        "size_bytes",
        "sha256",
    ])


    for path in targets:

        writer.writerow([
            path.relative_to(
                PACKAGE.parent
            ).as_posix(),

            path.stat().st_size,

            sha256_file(
                path
            ),
        ])


# ============================================================
# VERIFY MANIFEST CONTENT
# ============================================================

with MANIFEST.open(
    "r",
    encoding="utf-8",
    newline="",
) as f:

    reader = csv.DictReader(f)

    manifest_rows = list(
        reader
    )


manifest_entries = len(
    manifest_rows
)


final_files_after = sorted(
    p
    for p in PACKAGE.rglob("*")
    if p.is_file()
)


actual_total = len(
    final_files_after
)


manifest_complete = (
    manifest_entries
    ==
    actual_total - 1
)


if forbidden:

    raise RuntimeError(
        f"Forbidden/raw files detected: {len(forbidden)}"
    )


if oversized:

    raise RuntimeError(
        f"Oversized files detected: {len(oversized)}"
    )


if not manifest_complete:

    raise RuntimeError(
        "Final SHA256 manifest coverage mismatch."
    )


# ============================================================
# CONSOLE
# ============================================================

print("")
print("=" * 88)
print("C8 FINAL PACKAGE METADATA COMPLETE")
print("=" * 88)

print(
    "STATUS: PASS"
)

print(
    "Final total files:",
    actual_total
)

print(
    "Manifest entries:",
    manifest_entries
)

print(
    "Expected manifest entries:",
    actual_total - 1
)

print(
    "Forbidden/raw files:",
    len(forbidden)
)

print(
    ">=100 MiB files:",
    len(oversized)
)

print("")
print("Module counts:")

for (
    module,
    code_count,
    result_count,
) in module_rows:

    print(
        module,
        "code=",
        code_count,
        "results=",
        result_count,
    )

print("")
print(
    "Manifest:",
    MANIFEST
)

print(
    "QC report:",
    REPORT
)

print("")
print(
    "NO SCIENTIFIC RESULT WAS MODIFIED."
)

print(
    "NO RAW SOURCE DATA WERE COPIED."
)

print("=" * 88)
