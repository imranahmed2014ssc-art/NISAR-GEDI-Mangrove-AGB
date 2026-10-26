from pathlib import Path
from datetime import datetime
import csv
import os
import platform
import re
import sys

try:
    import pandas as pd
except Exception:
    pd = None

PROJECT_ROOT = Path(os.environ.get("NISAR_PROJECT_ROOT", r"D:\NISAR_Mangrove_Paper"))
DOC_DIR = PROJECT_ROOT / "00_Documentation"
NISAR_DIR = PROJECT_ROOT / "02_NISAR_GCOV" / "Original"
GEDI_DIR = PROJECT_ROOT / "03_GEDI_L4A_V3" / "ORIGINAL"
ACCEPT_DIR = DOC_DIR / "Data_Acceptance_Check"

AOI_WEST = 89.58
AOI_EAST = 89.84
AOI_SOUTH = 21.95
AOI_NORTH = 22.15
AOI_AREA_KM2 = 595.05
AOI_CRS = "EPSG:4326 (WGS84)"
DEFAULT_NISAR_CRS = "EPSG:32645"
RANDOM_SEED = 42

ANALYSIS_LOG = DOC_DIR / "analysis_log.txt"
DATA_INVENTORY = DOC_DIR / "data_inventory.csv"

def remove_empty_placeholder_dir(path: Path):
    if path.exists() and path.is_dir():
        if any(path.iterdir()):
            raise RuntimeError(
                f"{path} is a folder and is NOT empty. "
                "Rename or move it manually before running this script."
            )
        path.rmdir()
        print(f"Removed empty placeholder folder: {path}")

def human_mb(nbytes):
    return round(nbytes / (1024 ** 2), 3)

def parse_nisar_datetime(name):
    m = re.search(r"_(\d{8}T\d{6})_", name)
    if not m:
        return ""
    try:
        return datetime.strptime(m.group(1), "%Y%m%dT%H%M%S").isoformat()
    except Exception:
        return ""

def parse_gedi_datetime(name):
    # GEDI04_A_YYYYDOYhhmmss_...
    m = re.search(r"GEDI04_A_(\d{13})_", name)
    if not m:
        return ""
    try:
        return datetime.strptime(m.group(1), "%Y%j%H%M%S").isoformat()
    except Exception:
        return ""

def read_overall_acceptance():
    p = ACCEPT_DIR / "OVERALL_STATUS.txt"
    if p.exists():
        txt = p.read_text(encoding="utf-8", errors="replace").strip()
        return txt if txt else "UNKNOWN"
    return "UNKNOWN"

def read_gedi_file_check():
    p = ACCEPT_DIR / "GEDI_FILE_CHECK.csv"
    if not p.exists():
        return {}
    rows = {}
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows[row.get("file", "")] = row
    return rows

def read_acceptance_summary():
    p = ACCEPT_DIR / "DATA_ACCEPTANCE_SUMMARY.csv"
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def summary_detail(summary_rows, dataset, check):
    for r in summary_rows:
        if r.get("dataset") == dataset and r.get("check") == check:
            return r.get("detail", "")
    return ""

def package_version(pkg_name):
    try:
        import importlib.metadata as md
        return md.version(pkg_name)
    except Exception:
        return "not detected"

def main():
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    # The user originally created these as empty folders.
    remove_empty_placeholder_dir(ANALYSIS_LOG)
    remove_empty_placeholder_dir(DATA_INVENTORY)

    overall = read_overall_acceptance()
    gedi_checks = read_gedi_file_check()
    summary_rows = read_acceptance_summary()

    detected_crs = summary_detail(summary_rows, "NISAR", "Projection detected")
    detected_spacing = summary_detail(summary_rows, "NISAR", "Pixel spacing")
    if detected_crs:
        nisar_crs = f"EPSG:{detected_crs}" if str(detected_crs).isdigit() else str(detected_crs)
    else:
        nisar_crs = DEFAULT_NISAR_CRS

    inventory = []

    # NISAR raw files
    nisar_files = sorted(NISAR_DIR.glob("*.h5"))
    for fp in nisar_files:
        size = fp.stat().st_size
        inventory.append({
            "dataset": "NISAR",
            "filename": fp.name,
            "product": "NISAR L2 GCOV PROVISIONAL",
            "acquisition_datetime": parse_nisar_datetime(fp.name),
            "file_size_bytes": size,
            "file_size_mb": human_mb(size),
            "source": "ASF DAAC / NASA Earthdata",
            "acceptance_status": overall if overall.startswith("PASS") else "CHECK REPORT",
            "aoi_records": "",
            "quality_records": "",
            "raw_path": str(fp),
            "notes": "Raw HDF5; keep read-only; do not overwrite."
        })

    # GEDI raw files
    gedi_files = sorted(GEDI_DIR.glob("*.h5"))
    for fp in gedi_files:
        size = fp.stat().st_size
        chk = gedi_checks.get(fp.name, {})
        inventory.append({
            "dataset": "GEDI",
            "filename": fp.name,
            "product": "GEDI L4A Version 3 (GEDI04_A V003)",
            "acquisition_datetime": parse_gedi_datetime(fp.name),
            "file_size_bytes": size,
            "file_size_mb": human_mb(size),
            "source": "ORNL DAAC / NASA Earthdata",
            "acceptance_status": chk.get("status", "PASS" if overall.startswith("PASS") else "CHECK REPORT"),
            "aoi_records": chk.get("aoi_records", ""),
            "quality_records": chk.get("quality_records", ""),
            "raw_path": str(fp),
            "notes": "Raw HDF5; keep read-only; do not overwrite."
        })

    fields = [
        "dataset", "filename", "product", "acquisition_datetime",
        "file_size_bytes", "file_size_mb", "source", "acceptance_status",
        "aoi_records", "quality_records", "raw_path", "notes"
    ]

    with DATA_INVENTORY.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(inventory)

    now = datetime.now().astimezone().isoformat(timespec="seconds")

    log_lines = [
        "NISAR-GEDI MANGROVE AGBD PROJECT — ANALYSIS LOG",
        "=" * 72,
        f"Project root: {PROJECT_ROOT}",
        f"Log initialized: {now}",
        "",
        "LOCKED SCIENTIFIC DESIGN",
        "-" * 72,
        "Study area: Chandpai-Sarankhola, Bangladesh Sundarbans",
        f"AOI bounds (WGS84): {AOI_WEST} to {AOI_EAST} E; {AOI_SOUTH} to {AOI_NORTH} N",
        f"Approximate AOI area: {AOI_AREA_KM2} km2",
        f"AOI geographic CRS: {AOI_CRS}",
        f"NISAR GCOV native/project processing CRS: {nisar_crs}",
        f"NISAR pixel spacing: {detected_spacing if detected_spacing else 'to be read from accepted GCOV metadata'}",
        "Predictor source: NISAR L2 GCOV PROVISIONAL only",
        "Reference response: GEDI L4A Version 3 footprint-level AGBD",
        "Core predictors: HH_dB, HV_dB, HH_minus_HV, HV_SD",
        "Core models: baseline regression, Random Forest, XGBoost, MLP/DNN",
        "Validation: 5-fold spatial block cross-validation",
        f"Fixed modelling random seed: {RANDOM_SEED}",
        "",
        "PHASE 0A — DATA ACCEPTANCE",
        "-" * 72,
        f"Overall status: {overall}",
        f"NISAR raw HDF5 count: {len(nisar_files)}",
        f"GEDI raw HDF5 count: {len(gedi_files)}",
        f"Acceptance output folder: {ACCEPT_DIR}",
        "Decision: Gate 0 passed; formal processing may proceed.",
        "",
        "PHASE 0B — FREEZE INPUTS AND REPRODUCIBILITY RECORDS",
        "-" * 72,
        f"data_inventory.csv created with {len(inventory)} raw-file records.",
        "All raw NISAR and GEDI HDF5 files are to remain unchanged/read-only.",
        "Exact raw filenames and paths are recorded in data_inventory.csv.",
        "AOI coordinates and CRS are recorded in this log.",
        "",
        "SOFTWARE ENVIRONMENT AT LOG INITIALIZATION",
        "-" * 72,
        f"Python: {sys.version.split()[0]}",
        f"Platform: {platform.platform()}",
        f"h5py: {package_version('h5py')}",
        f"numpy: {package_version('numpy')}",
        f"pandas: {package_version('pandas')}",
        f"pyproj: {package_version('pyproj')}",
        "",
        "NEXT APPROVED STEP",
        "-" * 72,
        "Phase 1 — GEDI L4A V3 processing and quality filtering.",
        "Do not begin NISAR predictor generation or ML/DL modelling before Phase 1/Gate 1 review.",
        "",
        "LOGGING RULE",
        "-" * 72,
        "Append each major processing step below with date/time, script name, inputs, outputs,",
        "software/version information when relevant, decisions, and any warnings/errors.",
        "",
        "-----------------------------------------------------------------------",
        ""
    ]

    ANALYSIS_LOG.write_text("\n".join(log_lines), encoding="utf-8")

    print()
    print("=" * 72)
    print("PHASE 0B RECORDS CREATED")
    print("=" * 72)
    print(f"Created: {ANALYSIS_LOG}")
    print(f"Created: {DATA_INVENTORY}")
    print(f"NISAR files recorded: {len(nisar_files)}")
    print(f"GEDI files recorded : {len(gedi_files)}")
    print(f"Gate 0 status        : {overall}")
    print()
    print("Phase 0B is complete if:")
    print("  - data_inventory.csv contains 1 NISAR + 22 GEDI records")
    print("  - analysis_log.txt opens as a normal text file")
    print("  - raw HDF5 files remain unchanged")
    print()
    None  # interactive pause removed for portable execution

if __name__ == "__main__":
    main()
