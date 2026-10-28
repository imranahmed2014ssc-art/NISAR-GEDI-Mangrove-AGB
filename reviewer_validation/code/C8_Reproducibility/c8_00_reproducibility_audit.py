from pathlib import Path
import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import sys

import pandas as pd


# ============================================================
# ROOTS
# ============================================================

ROOT = Path(r"D:\NISAR_Mangrove_Paper")
JUNE_ROOT = Path(r"D:\new_nisar_data")

SCRIPT_ROOT = (
    ROOT
    / "05_ML_DL"
    / "Reviewer_Validation"
)

RESULT_ROOT = (
    ROOT
    / "06_Results"
    / "Reviewer_Validation"
)

OUT = (
    RESULT_ROOT
    / "C8_Reproducibility"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


AUDIT_CSV = (
    OUT
    / "C8_00_REPRODUCIBILITY_AUDIT.csv"
)

HASH_CSV = (
    OUT
    / "C8_00_KEY_FILE_HASHES.csv"
)

SEED_CSV = (
    OUT
    / "C8_00_SEED_INVENTORY.csv"
)

OOF_CSV = (
    OUT
    / "C8_00_OOF_INVENTORY.csv"
)

ENV_JSON = (
    OUT
    / "C8_00_CURRENT_ENVIRONMENT.json"
)

TEXT_ID_CSV = (
    OUT
    / "C8_00_ARCHIVE_IDENTIFIERS.csv"
)

REPORT = (
    OUT
    / "C8_00_REPRODUCIBILITY_AUDIT_REPORT.txt"
)


# ============================================================
# KNOWN IDENTIFIERS
# ============================================================

MAIN_NISAR_ID = (
    "NISAR_L2_PR_GCOV_030_170_A_013_4005_DHDH_A_"
    "20260917T232917_20260917T232952_P05023_F_F_J_001.h5"
)

JUNE_NISAR_ID = (
    "NISAR_L2_PR_GCOV_023_170_A_013_4005_DHDH_A_"
    "20260625T232921_20260625T232956_P05023_N_F_J_001.h5"
)

EXPECTED_GITHUB = (
    "imranahmed2014ssc-art/NISAR-GEDI-Mangrove-AGB"
)

EXPECTED_ZENODO_DOI = (
    "10.5281/zenodo.23675039"
)


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path, chunk_size=1024 * 1024):

    h = hashlib.sha256()

    with path.open("rb") as f:

        while True:

            chunk = f.read(
                chunk_size
            )

            if not chunk:
                break

            h.update(
                chunk
            )

    return h.hexdigest()


def safe_glob(
    root,
    pattern,
):

    if not root.exists():
        return []

    return sorted(
        root.rglob(
            pattern
        )
    )


def find_filename(
    root,
    filename,
):

    matches = []

    if root.exists():

        for p in root.rglob(
            filename
        ):

            if p.is_file():
                matches.append(
                    p
                )

    return sorted(
        matches
    )


def status_row(
    item,
    requirement,
    status,
    evidence="",
    notes="",
):

    return {
        "item":
            item,

        "requirement":
            requirement,

        "status":
            status,

        "evidence":
            evidence,

        "notes":
            notes,
    }


audit_rows = []


# ============================================================
# 1. MASTER MODELING TABLE
# ============================================================

master = (
    ROOT
    / "04_Matched_Data"
    / "Predictor_Expansion"
    / "Phase8C2"
    / "Tables"
    / "GEDI_NISAR_expanded_predictors_1641.csv"
)

if master.exists():

    master_df = pd.read_csv(
        master
    )

    master_ok = (
        len(master_df) == 1641
        and
        "shot_number" in master_df.columns
        and
        master_df[
            "shot_number"
        ].nunique() == 1641
    )

    audit_rows.append(
        status_row(
            "Frozen modeling table",
            "1641-shot frozen master table",
            "PASS" if master_ok else "FAIL",
            str(master),
            (
                f"Rows={len(master_df)}; "
                f"unique shots="
                f"{master_df['shot_number'].nunique()}"
                if "shot_number" in master_df.columns
                else
                f"Rows={len(master_df)}; "
                "shot_number missing"
            ),
        )
    )

else:

    audit_rows.append(
        status_row(
            "Frozen modeling table",
            "1641-shot frozen master table",
            "MISSING",
        )
    )


# ============================================================
# 2. BOTH NISAR PRODUCT IDENTIFIERS
# ============================================================

main_matches = find_filename(
    ROOT,
    MAIN_NISAR_ID,
)

june_matches = find_filename(
    JUNE_ROOT,
    JUNE_NISAR_ID,
)


audit_rows.append(
    status_row(
        "Primary NISAR product",
        "Exact September 17 2026 NISAR GCOV product",
        (
            "PASS"
            if len(main_matches) >= 1
            else
            "MISSING"
        ),
        "; ".join(
            str(x)
            for x in main_matches
        ),
        MAIN_NISAR_ID,
    )
)


audit_rows.append(
    status_row(
        "June sensitivity NISAR product",
        "Exact June 25 2026 NISAR GCOV product",
        (
            "PASS"
            if len(june_matches) >= 1
            else
            "MISSING"
        ),
        "; ".join(
            str(x)
            for x in june_matches
        ),
        (
            "READ-ONLY INVENTORY. "
            "D:\\new_nisar_data was not modified."
        ),
    )
)


# ============================================================
# 3. C1-C6 REVIEWER-VALIDATION RESULTS
# ============================================================

expected_result_dirs = {
    "C1":
        RESULT_ROOT
        / "C1_Independent_Validation",

    "C2":
        RESULT_ROOT
        / "C2_Spatial_Transferability",

    "C3":
        RESULT_ROOT
        / "C3_Temporal_Mismatch",

    "C4":
        RESULT_ROOT
        / "C4_GEDI_Uncertainty",

    "C5":
        RESULT_ROOT
        / "C5_Multiscale_Ablation",

    "C6":
        RESULT_ROOT
        / "C6_Support_Mixing",
}


for label, path in (
    expected_result_dirs.items()
):

    if path.exists():

        n_files = sum(
            1
            for p in path.rglob("*")
            if p.is_file()
        )

        audit_rows.append(
            status_row(
                f"{label} validation results",
                f"{label} reviewer-validation outputs",
                (
                    "PASS"
                    if n_files > 0
                    else
                    "FAIL"
                ),
                str(path),
                f"Files={n_files}",
            )
        )

    else:

        audit_rows.append(
            status_row(
                f"{label} validation results",
                f"{label} reviewer-validation outputs",
                "MISSING",
                str(path),
            )
        )


# ============================================================
# 4. KEY C1/C5/C6 FILES
# ============================================================

expected_files = {
    "C1 selection-aware OOF":
        (
            RESULT_ROOT
            / "C1_Independent_Validation"
            / "C1B_Full_Nested_Selection"
            / "C1B_SELECTION_AWARE_OOF.csv"
        ),

    "C5 frozen family manifest":
        (
            RESULT_ROOT
            / "C5_Multiscale_Ablation"
            / "C5_00_ABLATION_FAMILIES.json"
        ),

    "C5 all-family OOF":
        (
            RESULT_ROOT
            / "C5_Multiscale_Ablation"
            / "C5_01_ExtraTrees_Ablation"
            / "C5_01_ALL_FAMILY_OOF.csv"
        ),

    "C5 best hyperparameters":
        (
            RESULT_ROOT
            / "C5_Multiscale_Ablation"
            / "C5_01_ExtraTrees_Ablation"
            / "C5_01_BEST_PARAMETERS.csv"
        ),

    "C5 contrasts":
        (
            RESULT_ROOT
            / "C5_Multiscale_Ablation"
            / "C5_01_ExtraTrees_Ablation"
            / "C5_01_PREDECLARED_CONTRASTS.csv"
        ),

    "C6 corrected support table":
        (
            RESULT_ROOT
            / "C6_Support_Mixing"
            / "C6_01B_Corrected_GMW"
            / "C6_01B_CORRECTED_SUPPORT.csv"
        ),

    "C6 frozen support report":
        (
            RESULT_ROOT
            / "C6_Support_Mixing"
            / "C6_02_Frozen_OOF"
            / "C6_02_FROZEN_OOF_SUPPORT_REPORT.txt"
        ),
}


for label, path in (
    expected_files.items()
):

    audit_rows.append(
        status_row(
            label,
            "Required frozen reproducibility artifact",
            (
                "PASS"
                if path.exists()
                else
                "MISSING"
            ),
            str(path),
        )
    )


# ============================================================
# 5. SOURCE SCRIPTS
# ============================================================

review_scripts = (
    safe_glob(
        SCRIPT_ROOT,
        "*.py",
    )
)


audit_rows.append(
    status_row(
        "Reviewer-validation scripts",
        "Scripts required to regenerate C1-C6 validation",
        (
            "PASS"
            if len(review_scripts) > 0
            else
            "MISSING"
        ),
        str(SCRIPT_ROOT),
        f"Python scripts={len(review_scripts)}",
    )
)


# ============================================================
# 6. OOF INVENTORY
# ============================================================

oof_rows = []


for root in [
    RESULT_ROOT,
    ROOT / "06_Results",
]:

    if not root.exists():
        continue

    for path in root.rglob(
        "*"
    ):

        if (
            path.is_file()
            and
            "oof" in path.name.lower()
            and
            path.suffix.lower()
            in [
                ".csv",
                ".txt",
                ".json",
            ]
        ):

            row = {
                "path":
                    str(path),

                "size_bytes":
                    path.stat().st_size,

                "rows":
                    "",
            }

            if path.suffix.lower() == ".csv":

                try:

                    temp = pd.read_csv(
                        path
                    )

                    row[
                        "rows"
                    ] = len(temp)

                    row[
                        "columns"
                    ] = ";".join(
                        temp.columns.astype(
                            str
                        ).tolist()
                    )

                except Exception as exc:

                    row[
                        "read_error"
                    ] = str(exc)

            oof_rows.append(
                row
            )


oof_df = pd.DataFrame(
    oof_rows
)

oof_df.to_csv(
    OOF_CSV,
    index=False,
)


audit_rows.append(
    status_row(
        "OOF predictions",
        "Archived out-of-fold predictions",
        (
            "PASS"
            if len(oof_df) > 0
            else
            "MISSING"
        ),
        str(OOF_CSV),
        f"OOF-like files={len(oof_df)}",
    )
)


# ============================================================
# 7. FOLD / MEMBERSHIP FILES
# ============================================================

fold_candidates = []


for pattern in [
    "*fold*.csv",
    "*split*.csv",
    "*membership*.csv",
    "*manifest*.csv",
]:

    fold_candidates.extend(
        safe_glob(
            RESULT_ROOT,
            pattern,
        )
    )


fold_candidates = sorted(
    set(
        fold_candidates
    )
)


audit_rows.append(
    status_row(
        "Fold and split memberships",
        (
            "Exact fold IDs and buffered "
            "train/test memberships"
        ),
        (
            "PASS"
            if len(fold_candidates) > 0
            else
            "MISSING"
        ),
        "; ".join(
            str(p)
            for p in fold_candidates[
                :25
            ]
        ),
        (
            f"Candidate membership files="
            f"{len(fold_candidates)}"
        ),
    )
)


# ============================================================
# 8. HYPERPARAMETER / CONFIG INVENTORY
# ============================================================

parameter_candidates = []


for pattern in [
    "*PARAM*.csv",
    "*parameter*.csv",
    "*hyper*.csv",
    "*config*.json",
    "*config*.yaml",
    "*config*.yml",
    "*best*.csv",
]:

    parameter_candidates.extend(
        safe_glob(
            ROOT,
            pattern,
        )
    )


parameter_candidates = sorted(
    set(
        parameter_candidates
    )
)


audit_rows.append(
    status_row(
        "Hyperparameters/configuration",
        "Exact model settings and configurations",
        (
            "PASS"
            if len(parameter_candidates) > 0
            else
            "MISSING"
        ),
        "; ".join(
            str(p)
            for p in parameter_candidates[
                :25
            ]
        ),
        (
            f"Candidate files="
            f"{len(parameter_candidates)}"
        ),
    )
)


# ============================================================
# 9. RANDOM SEED INVENTORY
# ============================================================

seed_patterns = [
    re.compile(
        r"\bRANDOM_SEED\s*=\s*(\d+)"
    ),

    re.compile(
        r"\bBOOTSTRAP_SEED\s*=\s*(\d+)"
    ),

    re.compile(
        r"\brandom_state\s*=\s*(\d+)"
    ),

    re.compile(
        r"\bseed\s*=\s*(\d+)",
        re.IGNORECASE,
    ),

    re.compile(
        r"\bSEED\s*=\s*(\d+)"
    ),
]


seed_rows = []


for path in review_scripts:

    try:

        text_data = path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

    except Exception:
        continue


    lines = text_data.splitlines()


    for line_no, line in enumerate(
        lines,
        start=1,
    ):

        for pattern in seed_patterns:

            match = pattern.search(
                line
            )

            if match:

                seed_rows.append({
                    "file":
                        str(path),

                    "line":
                        line_no,

                    "seed":
                        match.group(1),

                    "text":
                        line.strip(),
                })


seed_df = pd.DataFrame(
    seed_rows
)

seed_df.to_csv(
    SEED_CSV,
    index=False,
)


audit_rows.append(
    status_row(
        "Random seeds",
        "All analysis/tuning/bootstrap seeds archived",
        (
            "PASS"
            if len(seed_df) > 0
            else
            "MISSING"
        ),
        str(SEED_CSV),
        (
            f"Seed declarations found="
            f"{len(seed_df)}"
        ),
    )
)


# ============================================================
# 10. REPRODUCTION / PHASE 9 AUDIT
# ============================================================

repro_candidates = []


for pattern in [
    "*reproduction*",
    "*reproduc*",
    "*phase9*",
    "*audit*",
]:

    repro_candidates.extend(
        safe_glob(
            ROOT / "06_Results",
            pattern,
        )
    )


repro_candidates = [
    p
    for p in sorted(
        set(
            repro_candidates
        )
    )
    if p.is_file()
]


audit_rows.append(
    status_row(
        "Independent reproduction check",
        (
            "Rerun/reproduction evidence "
            "including Phase 9"
        ),
        (
            "PASS"
            if len(repro_candidates) > 0
            else
            "MISSING"
        ),
        "; ".join(
            str(p)
            for p in repro_candidates[
                :25
            ]
        ),
        (
            f"Candidate files="
            f"{len(repro_candidates)}"
        ),
    )
)


# ============================================================
# 11. README / REQUIREMENTS / LICENSE / ENVIRONMENT RECORDS
# ============================================================

documentation_candidates = []


for name in [
    "README.md",
    "README.txt",
    "requirements.txt",
    "environment.yml",
    "environment.yaml",
    "LICENSE",
    "LICENSE.txt",
    "CITATION.cff",
]:

    documentation_candidates.extend(
        find_filename(
            ROOT,
            name,
        )
    )


documentation_candidates = sorted(
    set(
        documentation_candidates
    )
)


audit_rows.append(
    status_row(
        "Repository documentation",
        (
            "README, dependencies/environment, "
            "license/citation metadata"
        ),
        (
            "PASS"
            if len(documentation_candidates) >= 2
            else
            "PARTIAL"
        ),
        "; ".join(
            str(p)
            for p in documentation_candidates
        ),
        (
            f"Documentation files="
            f"{len(documentation_candidates)}"
        ),
    )
)


# ============================================================
# 12. CURRENT ENVIRONMENT SNAPSHOT
#
# This is labelled CURRENT, not necessarily historical.
# ============================================================

packages_to_check = [
    "geopandas",
    "h5py",
    "joblib",
    "matplotlib",
    "numpy",
    "pandas",
    "pyogrio",
    "pyproj",
    "rasterio",
    "scipy",
    "shapely",
    "sklearn",
    "statsmodels",
    "xgboost",
]


env = {
    "label":
        (
            "CURRENT AUDIT ENVIRONMENT - "
            "not automatically assumed to be "
            "the exact historical modeling environment"
        ),

    "python":
        sys.version,

    "python_executable":
        sys.executable,

    "platform":
        platform.platform(),

    "packages":
        {},
}


for package in packages_to_check:

    try:

        module = __import__(
            package
        )

        version = getattr(
            module,
            "__version__",
            "UNKNOWN",
        )

        env[
            "packages"
        ][
            package
        ] = str(
            version
        )

    except Exception as exc:

        env[
            "packages"
        ][
            package
        ] = (
            "IMPORT_ERROR: "
            +
            str(exc)
        )


with ENV_JSON.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        env,
        f,
        indent=2,
    )


audit_rows.append(
    status_row(
        "Environment snapshot",
        "Python and package versions documented",
        "PASS",
        str(ENV_JSON),
        (
            "Snapshot is current. "
            "Historical environment file must "
            "still be identified if available."
        ),
    )
)


# ============================================================
# 13. SEARCH TEXT FILES FOR GITHUB / DOI IDENTIFIERS
# ============================================================

identifier_rows = []


text_extensions = {
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".tex",
    ".csv",
    ".cff",
}


doi_regex = re.compile(
    r"10\.5281/zenodo\.\d+",
    re.IGNORECASE,
)

github_regex = re.compile(
    r"github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+",
    re.IGNORECASE,
)


# Avoid scanning huge raw data directories.
search_roots = [
    ROOT / "00_Documentation",
    ROOT / "05_ML_DL",
    ROOT / "06_Results",
]


for search_root in search_roots:

    if not search_root.exists():
        continue


    for path in search_root.rglob("*"):

        if (
            not path.is_file()
            or
            path.suffix.lower()
            not in text_extensions
        ):
            continue


        # Skip huge text-like files.
        if path.stat().st_size > 20_000_000:
            continue


        try:

            content = path.read_text(
                encoding="utf-8",
                errors="ignore",
            )

        except Exception:
            continue


        for match in doi_regex.finditer(
            content
        ):

            identifier_rows.append({
                "type":
                    "ZENODO_DOI",

                "value":
                    match.group(0),

                "file":
                    str(path),
            })


        for match in github_regex.finditer(
            content
        ):

            identifier_rows.append({
                "type":
                    "GITHUB_URL",

                "value":
                    match.group(0),

                "file":
                    str(path),
            })


identifier_df = pd.DataFrame(
    identifier_rows
)


if len(identifier_df):

    identifier_df = (
        identifier_df
        .drop_duplicates()
        .reset_index(
            drop=True
        )
    )


identifier_df.to_csv(
    TEXT_ID_CSV,
    index=False,
)


doi_found = (
    len(identifier_df) > 0
    and
    (
        identifier_df[
            "value"
        ]
        .astype(str)
        .str.lower()
        ==
        EXPECTED_ZENODO_DOI.lower()
    ).any()
)


github_found = (
    len(identifier_df) > 0
    and
    identifier_df[
        "value"
    ]
    .astype(str)
    .str.lower()
    .str.contains(
        EXPECTED_GITHUB.lower(),
        regex=False,
    )
    .any()
)


audit_rows.append(
    status_row(
        "Zenodo DOI recorded locally",
        EXPECTED_ZENODO_DOI,
        (
            "PASS"
            if doi_found
            else
            "NOT_FOUND"
        ),
        str(TEXT_ID_CSV),
        (
            "This checks local documentation only; "
            "it does not verify DOI resolution."
        ),
    )
)


audit_rows.append(
    status_row(
        "GitHub repository recorded locally",
        EXPECTED_GITHUB,
        (
            "PASS"
            if github_found
            else
            "NOT_FOUND"
        ),
        str(TEXT_ID_CSV),
        (
            "This checks local documentation only."
        ),
    )
)


# ============================================================
# 14. GIT REPOSITORY CHECK
# ============================================================

git_status = {
    "detected":
        False,
}


if (
    ROOT
    / ".git"
).exists():

    git_status[
        "detected"
    ] = True

    try:

        remote = subprocess.run(
            [
                "git",
                "-C",
                str(ROOT),
                "remote",
                "-v",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        git_status[
            "remote"
        ] = (
            remote.stdout.strip()
        )


        head = subprocess.run(
            [
                "git",
                "-C",
                str(ROOT),
                "rev-parse",
                "HEAD",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        git_status[
            "HEAD"
        ] = (
            head.stdout.strip()
        )

    except Exception as exc:

        git_status[
            "error"
        ] = str(exc)


audit_rows.append(
    status_row(
        "Local Git repository",
        "Commit hash and remote should be archived",
        (
            "PASS"
            if git_status[
                "detected"
            ]
            else
            "NOT_DETECTED"
        ),
        str(ROOT),
        json.dumps(
            git_status,
        ),
    )
)


# ============================================================
# 15. KEY FILE HASHES
# ============================================================

key_files = []


if master.exists():
    key_files.append(
        (
            "master_modeling_table",
            master,
        )
    )


for label, path in (
    expected_files.items()
):

    if path.exists():

        key_files.append(
            (
                label,
                path,
            )
        )


for p in main_matches:

    key_files.append(
        (
            "primary_NISAR_GCOV",
            p,
        )
    )


# June H5 can be very large.
# It is still hashed read-only because reproducibility requires
# exact product identity; no file is modified.
for p in june_matches:

    key_files.append(
        (
            "June_NISAR_GCOV",
            p,
        )
    )


hash_rows = []


for label, path in key_files:

    print(
        "Hashing:",
        label,
        path,
    )

    hash_rows.append({
        "artifact":
            label,

        "path":
            str(path),

        "size_bytes":
            path.stat().st_size,

        "sha256":
            sha256_file(
                path
            ),
    })


hash_df = pd.DataFrame(
    hash_rows
)

hash_df.to_csv(
    HASH_CSV,
    index=False,
)


audit_rows.append(
    status_row(
        "Key artifact hashes",
        "SHA256 checksums for core reproducibility artifacts",
        (
            "PASS"
            if len(hash_df) > 0
            else
            "MISSING"
        ),
        str(HASH_CSV),
        f"Hashed artifacts={len(hash_df)}",
    )
)


# ============================================================
# 16. WRITE AUDIT TABLE
# ============================================================

audit_df = pd.DataFrame(
    audit_rows
)

audit_df.to_csv(
    AUDIT_CSV,
    index=False,
)


# ============================================================
# 17. OVERALL STATUS
# ============================================================

blocking_statuses = {
    "MISSING",
    "FAIL",
}


blocking = audit_df.loc[
    audit_df[
        "status"
    ].isin(
        blocking_statuses
    )
]


overall = (
    "PASS_WITH_DOCUMENTATION_GAPS"
    if len(blocking) == 0
    else
    "INCOMPLETE"
)


# ============================================================
# REPORT
# ============================================================

with REPORT.open(
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "=" * 110
        + "\n"
    )

    f.write(
        "C8 STEP 0 - REPRODUCIBILITY / ARCHIVE AUDIT\n"
    )

    f.write(
        "=" * 110
        + "\n\n"
    )


    f.write(
        "STATUS\n"
    )

    f.write(
        "-" * 110
        + "\n"
    )

    f.write(
        f"Overall audit state: {overall}\n"
    )

    f.write(
        "READ-ONLY INVENTORY OF EXISTING PROJECT "
        "AND JUNE SENSITIVITY FILES.\n"
    )

    f.write(
        "NO MODEL WAS FITTED.\n"
    )

    f.write(
        "NO PRIOR RESULT WAS MODIFIED.\n"
    )

    f.write(
        "D:\\new_nisar_data was READ ONLY and "
        "was not overwritten or rerun.\n"
    )

    f.write(
        "C1-C7 remain frozen.\n\n"
    )


    f.write(
        "AUDIT MATRIX\n"
    )

    f.write(
        "-" * 110
        + "\n"
    )

    f.write(
        audit_df[
            [
                "item",
                "status",
                "requirement",
                "evidence",
                "notes",
            ]
        ].to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )


    f.write(
        "BLOCKING ITEMS\n"
    )

    f.write(
        "-" * 110
        + "\n"
    )

    if len(blocking):

        f.write(
            blocking[
                [
                    "item",
                    "status",
                    "requirement",
                    "evidence",
                ]
            ].to_string(
                index=False
            )
        )

    else:

        f.write(
            "No missing/failing core local artifacts "
            "detected.\n"
        )

    f.write(
        "\n\n"
    )


    f.write(
        "ARCHIVE IDENTIFIERS EXPECTED\n"
    )

    f.write(
        "-" * 110
        + "\n"
    )

    f.write(
        f"GitHub: {EXPECTED_GITHUB}\n"
    )

    f.write(
        f"Zenodo DOI supplied for project: "
        f"{EXPECTED_ZENODO_DOI}\n"
    )

    f.write(
        "C8 finalization must verify whether the Zenodo "
        "DOI is the exact-version DOI or concept DOI "
        "for the release cited in the manuscript.\n\n"
    )


    f.write(
        "C8 FINAL PACKAGE REQUIREMENTS\n"
    )

    f.write(
        "-" * 110
        + "\n"
    )

    f.write(
        "1. Exact September and June NISAR product IDs.\n"
    )

    f.write(
        "2. Frozen master-table identity and checksum.\n"
    )

    f.write(
        "3. Exact geographic fold IDs and buffered "
        "training memberships.\n"
    )

    f.write(
        "4. All random seeds.\n"
    )

    f.write(
        "5. Exact selected/tuned hyperparameters.\n"
    )

    f.write(
        "6. C1-C6 OOF predictions and sensitivity tables.\n"
    )

    f.write(
        "7. Environment/dependency specification.\n"
    )

    f.write(
        "8. Reproduction/regeneration scripts.\n"
    )

    f.write(
        "9. README with exact run order.\n"
    )

    f.write(
        "10. Git commit/release identifier and "
        "Zenodo archived-release DOI.\n"
    )

    f.write(
        "11. Checksums for frozen source tables and "
        "critical outputs.\n"
    )

    f.write(
        "12. Clear statement that raw NISAR/GEDI data "
        "are not redistributed when licensing or file "
        "size prevents redistribution; instead provide "
        "product/granule identifiers and acquisition "
        "instructions.\n"
    )


# ============================================================
# CONSOLE
# ============================================================

print("")
print("=" * 90)
print("C8 STEP 0 COMPLETE")
print("REPRODUCIBILITY / ARCHIVE AUDIT")
print("=" * 90)

print("")
print(
    audit_df[
        [
            "item",
            "status",
        ]
    ].to_string(
        index=False
    )
)

print("")
print(
    "Overall:",
    overall
)

print("")
print("Report:")
print(REPORT)

print("")
print("Audit table:")
print(AUDIT_CSV)

print("")
print("Key hashes:")
print(HASH_CSV)

print("")
print("NO MODEL WAS FITTED.")
print("NO PRIOR RESULT WAS MODIFIED.")
print("C1-C7 REMAIN FROZEN.")
print("=" * 90)
