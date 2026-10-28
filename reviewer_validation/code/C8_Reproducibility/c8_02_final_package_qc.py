from pathlib import Path
import hashlib
import csv


ROOT = Path(r"D:\NISAR_Mangrove_Paper")

PACKAGE = (
    ROOT
    / "github_repo"
    / "reviewer_validation"
)

DOCS = (
    PACKAGE
    / "docs"
)

MANIFEST = (
    DOCS
    / "REVIEWER_VALIDATION_SHA256.csv"
)

REPORT = (
    DOCS
    / "FINAL_PACKAGE_QC.txt"
)


# ============================================================
# POLICY
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

GITHUB_HARD_LIMIT = (
    100
    *
    1024
    *
    1024
)

WARNING_LIMIT = (
    50
    *
    1024
    *
    1024
)


REQUIRED_RELATIVE_PATHS = [
    "README.md",
    "docs/SOURCE_PRODUCTS.md",
    "docs/PACKAGE_SUMMARY.txt",

    "results/C1_Independent_Validation/"
    "C1B_Full_Nested_Selection/"
    "C1B_SELECTION_AWARE_OOF.csv",

    "results/C5_Multiscale_Ablation/"
    "C5_01_ExtraTrees_Ablation/"
    "C5_01_ALL_FAMILY_OOF.csv",

    "results/C6_Support_Mixing/"
    "C6_01B_Corrected_GMW/"
    "C6_01B_CORRECTED_SUPPORT.csv",

    "results/C6_Support_Mixing/"
    "C6_02_Frozen_OOF/"
    "C6_02_FROZEN_OOF_SUPPORT_REPORT.txt",

    "results/C8_Reproducibility/"
    "C8_00_KEY_FILE_HASHES.csv",

    "results/C8_Reproducibility/"
    "C8_00_CURRENT_ENVIRONMENT.json",
]


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path):

    h = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:

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
# INVENTORY
# ============================================================

if not PACKAGE.exists():

    raise RuntimeError(
        f"Package not found: {PACKAGE}"
    )


all_files = sorted(
    p
    for p in PACKAGE.rglob("*")
    if p.is_file()
)


forbidden_files = [
    p
    for p in all_files
    if p.suffix.lower()
    in FORBIDDEN_SUFFIXES
]


unexpected_suffixes = [
    p
    for p in all_files
    if (
        p != MANIFEST
        and
        p.suffix.lower()
        not in ALLOWED_SUFFIXES
    )
]


oversized = [
    p
    for p in all_files
    if p.stat().st_size
    >=
    GITHUB_HARD_LIMIT
]


large_warning = [
    p
    for p in all_files
    if (
        p.stat().st_size
        >=
        WARNING_LIMIT
        and
        p.stat().st_size
        <
        GITHUB_HARD_LIMIT
    )
]


# ============================================================
# REQUIRED FILES
# ============================================================

required_rows = []

missing_required = []


for relative in REQUIRED_RELATIVE_PATHS:

    path = (
        PACKAGE
        /
        Path(relative)
    )

    exists = path.exists()

    required_rows.append(
        (
            relative,
            exists,
        )
    )

    if not exists:

        missing_required.append(
            relative
        )


# ============================================================
# REQUIRED MODULE DIRECTORY COUNTS
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
        /
        "code"
        /
        module
    )

    result_dir = (
        PACKAGE
        /
        "results"
        /
        module
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
# LARGEST FILES
# ============================================================

largest = sorted(
    all_files,
    key=lambda p:
        p.stat().st_size,
    reverse=True,
)[:20]


# ============================================================
# REGENERATE CHECKSUM MANIFEST
#
# Manifest excludes itself by definition.
# Everything else is included.
# ============================================================

manifest_targets = sorted(
    p
    for p in PACKAGE.rglob("*")
    if (
        p.is_file()
        and
        p != MANIFEST
    )
)


with MANIFEST.open(
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


    for path in manifest_targets:

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
# FINAL COUNTS AFTER MANIFEST REGENERATION
# ============================================================

final_files = sorted(
    p
    for p in PACKAGE.rglob("*")
    if p.is_file()
)


total_bytes = sum(
    p.stat().st_size
    for p in final_files
)


manifest_entries = (
    len(
        manifest_targets
    )
)


expected_manifest_entries = (
    len(
        final_files
    )
    -
    1
)


manifest_complete = (
    manifest_entries
    ==
    expected_manifest_entries
)


# ============================================================
# PASS / FAIL
# ============================================================

failures = []


if forbidden_files:

    failures.append(
        "Forbidden/raw-data file types detected."
    )


if unexpected_suffixes:

    failures.append(
        "Unexpected file suffixes detected."
    )


if oversized:

    failures.append(
        "File(s) exceed GitHub 100 MiB hard limit."
    )


if missing_required:

    failures.append(
        "Required reproducibility files are missing."
    )


for (
    module,
    code_count,
    result_count,
) in module_rows:

    if module == "C8_Reproducibility":

        # C8 itself may have fewer code/result files.
        if code_count == 0:
            failures.append(
                f"{module}: no code files."
            )

    else:

        if code_count == 0:

            failures.append(
                f"{module}: no code files."
            )

        if result_count == 0:

            failures.append(
                f"{module}: no result files."
            )


if not manifest_complete:

    failures.append(
        "Checksum manifest coverage mismatch."
    )


status = (
    "PASS"
    if not failures
    else
    "FAIL"
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
        "C8 FINAL REVIEWER-VALIDATION PACKAGE QC\n"
    )

    f.write(
        "=" * 100
        + "\n\n"
    )


    f.write(
        f"STATUS: {status}\n\n"
    )


    f.write(
        "PACKAGE SUMMARY\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    f.write(
        f"Total files: "
        f"{len(final_files)}\n"
    )

    f.write(
        f"Total bytes: "
        f"{total_bytes}\n"
    )

    f.write(
        f"Checksum manifest entries: "
        f"{manifest_entries}\n"
    )

    f.write(
        f"Expected manifest entries: "
        f"{expected_manifest_entries}\n"
    )

    f.write(
        f"Manifest complete: "
        f"{manifest_complete}\n\n"
    )


    f.write(
        "RAW / FORBIDDEN DATA CHECK\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    f.write(
        f"Forbidden files: "
        f"{len(forbidden_files)}\n"
    )

    for p in forbidden_files:

        f.write(
            f"  {p}\n"
        )


    f.write(
        f"Unexpected suffixes: "
        f"{len(unexpected_suffixes)}\n"
    )

    for p in unexpected_suffixes:

        f.write(
            f"  {p}\n"
        )


    f.write(
        "\n"
    )


    f.write(
        "FILE-SIZE CHECK\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    f.write(
        f">=100 MiB: "
        f"{len(oversized)}\n"
    )

    f.write(
        f"50-100 MiB warnings: "
        f"{len(large_warning)}\n\n"
    )


    f.write(
        "REQUIRED FILES\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    for relative, exists in (
        required_rows
    ):

        f.write(
            f"{'PASS' if exists else 'MISSING'} "
            f"{relative}\n"
        )


    f.write(
        "\n"
    )


    f.write(
        "MODULE COUNTS\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    for (
        module,
        code_count,
        result_count,
    ) in module_rows:

        f.write(
            f"{module}: "
            f"code={code_count}, "
            f"results={result_count}\n"
        )


    f.write(
        "\n"
    )


    f.write(
        "20 LARGEST FILES\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    for path in largest:

        f.write(
            f"{path.stat().st_size:>12}  "
            f"{path.relative_to(PACKAGE)}\n"
        )


    f.write(
        "\n"
    )


    f.write(
        "FAILURES\n"
    )

    f.write(
        "-" * 100
        + "\n"
    )

    if failures:

        for failure in failures:

            f.write(
                f"- {failure}\n"
            )

    else:

        f.write(
            "None.\n"
        )


# ============================================================
# CONSOLE
# ============================================================

print("")
print("=" * 88)
print("C8 FINAL PACKAGE QC")
print("=" * 88)

print(
    "STATUS:",
    status
)

print(
    "Total files:",
    len(final_files)
)

print(
    "Total bytes:",
    total_bytes
)

print(
    "Manifest entries:",
    manifest_entries
)

print(
    "Expected manifest entries:",
    expected_manifest_entries
)

print(
    "Forbidden/raw files:",
    len(forbidden_files)
)

print(
    "Unexpected suffixes:",
    len(unexpected_suffixes)
)

print(
    ">=100 MiB files:",
    len(oversized)
)

print(
    "Missing required:",
    len(missing_required)
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
print("Largest files:")

for path in largest[:10]:

    print(
        path.stat().st_size,
        path.relative_to(
            PACKAGE
        ),
    )

print("")
print("Report:")
print(REPORT)

print("=" * 88)
