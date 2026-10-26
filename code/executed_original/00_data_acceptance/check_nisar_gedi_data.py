from pathlib import Path
from datetime import datetime
import csv, math, re, sys, traceback

import h5py
import numpy as np
import pandas as pd

try:
    from pyproj import Transformer
    HAVE_PYPROJ = True
except Exception:
    HAVE_PYPROJ = False

# ============================================================
# PROJECT SETTINGS
# ============================================================
NISAR_DIR = Path(r"D:\NISAR_Mangrove_Paper\02_NISAR_GCOV\Original")
GEDI_DIR  = Path(r"D:\NISAR_Mangrove_Paper\03_GEDI_L4A_V3\ORIGINAL")
OUT_DIR   = Path(r"D:\NISAR_Mangrove_Paper\00_Documentation\Data_Acceptance_Check")

# Locked AOI (WGS84)
WEST, EAST = 89.58, 89.84
SOUTH, NORTH = 21.95, 22.15

OUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = []
SUMMARY = []
NISAR_INV = []
GEDI_FILE_ROWS = []

def log(msg=""):
    print(msg)
    REPORT.append(str(msg))

def add_summary(dataset, check, status, detail):
    SUMMARY.append({
        "dataset": dataset,
        "check": check,
        "status": status,
        "detail": detail
    })

def safe_scalar(x):
    try:
        a = np.asarray(x)
        if a.size == 1:
            v = a.reshape(-1)[0]
            if isinstance(v, bytes):
                return v.decode("utf-8", errors="replace")
            if isinstance(v, np.generic):
                return v.item()
            return v
    except Exception:
        pass
    return x

def walk_h5(group, prefix=""):
    """Yield (full_path, object) recursively."""
    for key in group.keys():
        obj = group[key]
        path = f"{prefix}/{key}"
        yield path, obj
        if isinstance(obj, h5py.Group):
            yield from walk_h5(obj, path)

def find_dataset_paths(h5, exact_names):
    exact_names = {x.lower() for x in exact_names}
    found = []
    for path, obj in walk_h5(h5):
        if isinstance(obj, h5py.Dataset):
            name = path.split("/")[-1].lower()
            if name in exact_names:
                found.append(path)
    return found

def find_group_paths(h5, exact_names):
    exact_names = {x.lower() for x in exact_names}
    found = []
    for path, obj in walk_h5(h5):
        if isinstance(obj, h5py.Group):
            name = path.split("/")[-1].lower()
            if name in exact_names:
                found.append(path)
    return found

def sample_2d_dataset(dset, target=250_000):
    """Deterministic strided sample from 2D dataset without loading full raster."""
    if len(dset.shape) != 2:
        return np.asarray(dset[...]).reshape(-1)
    n = dset.shape[0] * dset.shape[1]
    step = max(1, int(math.ceil(math.sqrt(n / target))))
    try:
        arr = np.asarray(dset[::step, ::step])
    except Exception:
        # fallback: central window
        r0 = max(0, dset.shape[0]//2 - 256)
        c0 = max(0, dset.shape[1]//2 - 256)
        arr = np.asarray(dset[r0:r0+512, c0:c0+512])
    return arr.reshape(-1)

def numeric_stats(arr):
    arr = np.asarray(arr)
    if np.iscomplexobj(arr):
        arr = np.abs(arr)
    arr = arr.astype("float64", copy=False).reshape(-1)
    finite = np.isfinite(arr)
    vals = arr[finite]
    if len(vals) == 0:
        return {
            "n": int(arr.size), "finite_n": 0, "finite_pct": 0.0,
            "min": np.nan, "max": np.nan, "mean": np.nan,
            "positive_pct": 0.0, "zero_pct": 0.0
        }
    return {
        "n": int(arr.size),
        "finite_n": int(len(vals)),
        "finite_pct": float(100 * len(vals) / arr.size),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "mean": float(np.mean(vals)),
        "positive_pct": float(100 * np.mean(vals > 0)),
        "zero_pct": float(100 * np.mean(vals == 0)),
    }

def read_first_dataset(h5, candidate_names):
    paths = find_dataset_paths(h5, candidate_names)
    if not paths:
        return None, None
    p = paths[0]
    try:
        return p, np.asarray(h5[p][...])
    except Exception:
        return p, None

def dataset_value_or_attr(h5, candidate_names):
    paths = find_dataset_paths(h5, candidate_names)
    if paths:
        try:
            return safe_scalar(h5[paths[0]][()])
        except Exception:
            pass
    # Search attrs
    cands = {x.lower() for x in candidate_names}
    for path, obj in [("/", h5)] + list(walk_h5(h5)):
        for k, v in obj.attrs.items():
            if str(k).lower() in cands:
                return safe_scalar(v)
    return None

def check_aoi_coverage(x, y, epsg):
    if x is None or y is None or epsg is None or not HAVE_PYPROJ:
        return None
    try:
        x = np.asarray(x).astype(float).reshape(-1)
        y = np.asarray(y).astype(float).reshape(-1)
        xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
        ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
        transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{int(epsg)}", always_xy=True)
        corners = [(WEST,SOUTH),(WEST,NORTH),(EAST,SOUTH),(EAST,NORTH)]
        proj = [transformer.transform(lon,lat) for lon,lat in corners]
        ok = all((xmin <= xx <= xmax and ymin <= yy <= ymax) for xx,yy in proj)
        return {
            "ok": ok,
            "raster_bbox": (xmin, ymin, xmax, ymax),
            "aoi_proj_corners": proj
        }
    except Exception:
        return None

def parse_gedi_datetime(name):
    # GEDI04_A_YYYYDOYhhmmss_...
    m = re.search(r"GEDI04_A_(\d{13})_", name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%j%H%M%S")
    except Exception:
        return None

# ============================================================
# NISAR ACCEPTANCE TEST
# ============================================================
log("="*72)
log("NISAR + GEDI DATA ACCEPTANCE TEST")
log("="*72)
log(f"NISAR folder: {NISAR_DIR}")
log(f"GEDI folder : {GEDI_DIR}")
log(f"Output folder: {OUT_DIR}")
log(f"AOI WGS84: W={WEST}, E={EAST}, S={SOUTH}, N={NORTH}")
log("")

nisar_files = sorted(NISAR_DIR.glob("*.h5"))
log("NISAR ACCEPTANCE CHECK")
log("-"*72)
log(f"NISAR .h5 files found: {len(nisar_files)}")
add_summary("NISAR", "HDF5 file count", "PASS" if len(nisar_files)==1 else ("WARN" if len(nisar_files)>1 else "FAIL"),
            f"{len(nisar_files)} file(s) found in {NISAR_DIR}")

nisar_core_pass = False
nisar_aoi_pass = False
nisar_mask_found = False
nisar_projection = None
nisar_pixel_x = None
nisar_pixel_y = None

if not nisar_files:
    log("FAIL: No NISAR HDF5 file found.")
else:
    fp = nisar_files[0]
    log(f"Checking: {fp.name}")
    try:
        with h5py.File(fp, "r") as h5:
            log("PASS: HDF5 opens successfully.")
            add_summary("NISAR", "HDF5 readable", "PASS", fp.name)

            # inventory
            for path, obj in walk_h5(h5):
                if isinstance(obj, h5py.Dataset):
                    NISAR_INV.append({
                        "path": path,
                        "shape": str(obj.shape),
                        "dtype": str(obj.dtype),
                        "size": int(obj.size)
                    })

            # product metadata
            product_type = dataset_value_or_attr(h5, ["productType", "product_type", "product"])
            look_direction = dataset_value_or_attr(h5, ["lookDirection", "look_direction"])
            orbit_pass = dataset_value_or_attr(h5, ["orbitPassDirection", "orbit_pass_direction"])
            granule_id = dataset_value_or_attr(h5, ["granuleId", "granule_id"])
            if granule_id is None:
                granule_id = fp.stem

            log(f"Granule ID / filename: {granule_id}")
            if product_type is not None:
                log(f"Product type metadata: {product_type}")
            if orbit_pass is not None:
                log(f"Orbit pass direction: {orbit_pass}")
            if look_direction is not None:
                log(f"Look direction: {look_direction}")

            # frequencyA
            freq_groups = find_group_paths(h5, ["frequencyA"])
            if freq_groups:
                log(f"PASS: Frequency-A group found: {freq_groups[0]}")
                add_summary("NISAR", "Frequency-A group", "PASS", freq_groups[0])
            else:
                log("FAIL: Frequency-A group not found.")
                add_summary("NISAR", "Frequency-A group", "FAIL", "frequencyA not found")

            # HHHH/HVHV
            hhhh_paths = find_dataset_paths(h5, ["HHHH"])
            hvhv_paths = find_dataset_paths(h5, ["HVHV"])
            if hhhh_paths:
                log(f"PASS: HHHH found: {hhhh_paths[0]}")
                add_summary("NISAR", "HHHH available", "PASS", hhhh_paths[0])
            else:
                log("FAIL: HHHH not found.")
                add_summary("NISAR", "HHHH available", "FAIL", "HHHH not found")
            if hvhv_paths:
                log(f"PASS: HVHV found: {hvhv_paths[0]}")
                add_summary("NISAR", "HVHV available", "PASS", hvhv_paths[0])
            else:
                log("FAIL: HVHV not found.")
                add_summary("NISAR", "HVHV available", "FAIL", "HVHV not found")

            # basic sampled stats
            core_valid = True
            for label, paths in [("HHHH", hhhh_paths), ("HVHV", hvhv_paths)]:
                if not paths:
                    core_valid = False
                    continue
                ds = h5[paths[0]]
                smp = sample_2d_dataset(ds)
                st = numeric_stats(smp)
                log(f"{label} sampled stats: shape={ds.shape}, dtype={ds.dtype}, "
                    f"finite={st['finite_pct']:.2f}%, positive={st['positive_pct']:.2f}%, "
                    f"min={st['min']:.6g}, max={st['max']:.6g}, mean={st['mean']:.6g}")
                status = "PASS" if st["finite_n"] > 0 and st["positive_pct"] > 0 else "FAIL"
                add_summary("NISAR", f"{label} sampled values", status,
                            f"shape={ds.shape}; finite={st['finite_pct']:.2f}%; positive={st['positive_pct']:.2f}%")
                if status != "PASS":
                    core_valid = False

            # projection / x/y
            proj = dataset_value_or_attr(h5, ["projection", "epsg", "epsgCode", "epsg_code"])
            try:
                if isinstance(proj, str):
                    m = re.search(r"(\d{4,6})", proj)
                    if m: proj = int(m.group(1))
                elif proj is not None:
                    proj = int(proj)
            except Exception:
                proj = None
            nisar_projection = proj
            log(f"Projection / EPSG detected: {proj}")
            add_summary("NISAR", "Projection detected", "PASS" if proj else "WARN",
                        str(proj) if proj else "Could not automatically detect EPSG")

            xp, x = read_first_dataset(h5, ["xCoordinates", "x_coordinates", "xCoordinate", "x"])
            yp, y = read_first_dataset(h5, ["yCoordinates", "y_coordinates", "yCoordinate", "y"])
            if x is not None and y is not None:
                x1=np.asarray(x).reshape(-1); y1=np.asarray(y).reshape(-1)
                if len(x1)>1:
                    nisar_pixel_x=float(np.nanmedian(np.abs(np.diff(x1))))
                if len(y1)>1:
                    nisar_pixel_y=float(np.nanmedian(np.abs(np.diff(y1))))
                log(f"Coordinate datasets: x={xp}, y={yp}")
                log(f"Detected pixel spacing: dx={nisar_pixel_x}, dy={nisar_pixel_y}")
                add_summary("NISAR","Pixel spacing","PASS",
                            f"dx={nisar_pixel_x}, dy={nisar_pixel_y}")
            else:
                log("WARN: Could not locate 1D x/y coordinate datasets automatically.")
                add_summary("NISAR","Pixel spacing","WARN","x/y coordinate arrays not found automatically")

            cov = check_aoi_coverage(x, y, proj)
            if cov is not None:
                nisar_aoi_pass = bool(cov["ok"])
                log(f"AOI coverage check: {'PASS' if cov['ok'] else 'FAIL'}")
                log(f"NISAR raster bbox: {cov['raster_bbox']}")
                add_summary("NISAR","Full AOI coverage","PASS" if cov["ok"] else "FAIL",
                            f"raster_bbox={cov['raster_bbox']}")
            else:
                log("WARN: AOI coverage could not be verified automatically (projection/x/y unavailable).")
                add_summary("NISAR","Full AOI coverage","WARN",
                            "Could not verify automatically; inspect geospatial metadata manually")

            mask_paths = find_dataset_paths(h5, ["mask", "validityMask", "valid_data_mask"])
            nisar_mask_found = bool(mask_paths)
            if mask_paths:
                log(f"PASS: NISAR mask dataset found: {mask_paths[0]}")
                add_summary("NISAR","Valid-data mask","PASS",mask_paths[0])
            else:
                log("WARN: No dataset named mask/validityMask found automatically.")
                add_summary("NISAR","Valid-data mask","WARN","Mask not found by name search")

            nisar_core_pass = bool(freq_groups and hhhh_paths and hvhv_paths and core_valid)

    except Exception as e:
        log(f"FAIL: Could not open/check NISAR file: {e}")
        add_summary("NISAR","HDF5 readable","FAIL",repr(e))
        traceback.print_exc()

# ============================================================
# GEDI ACCEPTANCE TEST
# ============================================================
log("")
log("GEDI L4A V3 ACCEPTANCE CHECK")
log("-"*72)

gedi_files = sorted(GEDI_DIR.glob("*.h5"))
log(f"GEDI .h5 files found: {len(gedi_files)}")
add_summary("GEDI","HDF5 file count","PASS" if len(gedi_files)==22 else ("WARN" if len(gedi_files)>0 else "FAIL"),
            f"{len(gedi_files)} file(s) found; expected 22 from completed download")

required = [
    "shot_number","lat_lowestmode","lon_lowestmode","agbd","agbd_se",
    "l2_algrunflag","l2a_quality_flag_rel3","l4a_quality_flag_rel3",
    "degrade_include_flag","elev_highestreturn_outlier_flag",
    "predictor_limit_flag","response_limit_flag","sensitivity","delta_time"
]

all_aoi_agbd=[]
all_q_agbd=[]
all_q_agbd_se=[]
all_q_lons=[]
all_q_lats=[]
all_q_shots=[]
all_q_dates=[]
total_aoi=0
total_quality=0
readable_files=0
files_with_all_core=0
missing_counter={k:0 for k in required}
date_values=[]

for i, fp in enumerate(gedi_files, 1):
    row = {"file": fp.name, "status":"", "beam_count":0, "aoi_records":0, "quality_records":0, "missing_variables":""}
    dt=parse_gedi_datetime(fp.name)
    if dt: date_values.append(dt)

    try:
        with h5py.File(fp,"r") as h5:
            readable_files += 1
            beams=[k for k in h5.keys() if str(k).startswith("BEAM")]
            row["beam_count"]=len(beams)
            missing_in_file=set()
            file_aoi=0
            file_q=0

            for beam in beams:
                g=h5[beam]
                # required variable availability at beam level
                for v in required:
                    if v not in g:
                        missing_in_file.add(v)

                core_needed=["shot_number","lat_lowestmode","lon_lowestmode","agbd","agbd_se"]
                if not all(v in g for v in core_needed):
                    continue

                lon=np.asarray(g["lon_lowestmode"][:],dtype=float)
                lat=np.asarray(g["lat_lowestmode"][:],dtype=float)
                agbd=np.asarray(g["agbd"][:],dtype=float)
                agbd_se=np.asarray(g["agbd_se"][:],dtype=float)
                shot=np.asarray(g["shot_number"][:])

                m=(
                    np.isfinite(lon)&np.isfinite(lat)&
                    (lon>=WEST)&(lon<=EAST)&(lat>=SOUTH)&(lat<=NORTH)
                )
                n_aoi=int(m.sum())
                file_aoi += n_aoi
                if n_aoi == 0:
                    continue

                all_aoi_agbd.extend(agbd[m & np.isfinite(agbd)].tolist())

                # Primary proposal quality filter only if all needed flags exist
                q_needed=["l2_algrunflag","l2a_quality_flag_rel3","l4a_quality_flag_rel3",
                          "degrade_include_flag","elev_highestreturn_outlier_flag"]
                if all(v in g for v in q_needed):
                    q = (
                        m &
                        (np.asarray(g["l2_algrunflag"][:])==1) &
                        (np.asarray(g["l2a_quality_flag_rel3"][:])==1) &
                        (np.asarray(g["l4a_quality_flag_rel3"][:])==1) &
                        (np.asarray(g["degrade_include_flag"][:])==1) &
                        (np.asarray(g["elev_highestreturn_outlier_flag"][:])==0) &
                        np.isfinite(agbd) & (agbd>0)
                    )
                    nq=int(q.sum())
                    file_q += nq
                    if nq:
                        all_q_agbd.extend(agbd[q].tolist())
                        all_q_agbd_se.extend(agbd_se[q & np.isfinite(agbd_se)].tolist())
                        all_q_lons.extend(lon[q].tolist())
                        all_q_lats.extend(lat[q].tolist())
                        all_q_shots.extend([int(x) for x in shot[q]])
                        if dt:
                            all_q_dates.extend([dt.isoformat()]*nq)

            row["aoi_records"]=file_aoi
            row["quality_records"]=file_q
            row["missing_variables"]=";".join(sorted(missing_in_file))
            row["status"]="OK"
            total_aoi += file_aoi
            total_quality += file_q
            if not missing_in_file:
                files_with_all_core += 1
            for v in missing_in_file:
                missing_counter[v]+=1

    except Exception as e:
        row["status"]="ERROR: "+repr(e)

    GEDI_FILE_ROWS.append(row)
    print(f"GEDI [{i}/{len(gedi_files)}] {fp.name}: {row['status']}, AOI={row['aoi_records']}, Q={row['quality_records']}")

log(f"Readable GEDI files: {readable_files}/{len(gedi_files)}")
add_summary("GEDI","All files readable",
            "PASS" if len(gedi_files)>0 and readable_files==len(gedi_files) else "FAIL",
            f"{readable_files}/{len(gedi_files)} readable")

log(f"GEDI files with all required variables in every beam: {files_with_all_core}/{len(gedi_files)}")
if len(gedi_files):
    missing_any={k:v for k,v in missing_counter.items() if v>0}
else:
    missing_any={}
if missing_any:
    log(f"WARN/FAIL: Missing-variable counts by file: {missing_any}")
    add_summary("GEDI","Required variables present","FAIL",str(missing_any))
else:
    log("PASS: All required GEDI variables were found in all checked beam groups.")
    add_summary("GEDI","Required variables present","PASS","All required variables found")

if date_values:
    dmin=min(date_values); dmax=max(date_values)
    log(f"GEDI acquisition date range from filenames: {dmin.isoformat()} to {dmax.isoformat()}")
    add_summary("GEDI","Acquisition date range","PASS",f"{dmin.date()} to {dmax.date()}")

log(f"GEDI records inside AOI before quality filtering: {total_aoi:,}")
log(f"GEDI records passing primary proposal quality filter: {total_quality:,}")
add_summary("GEDI","AOI footprints present","PASS" if total_aoi>0 else "FAIL",f"{total_aoi:,} records")
add_summary("GEDI","Quality-filtered footprints present","PASS" if total_quality>0 else "FAIL",f"{total_quality:,} records")

# unique quality shots
uniq_q = len(set(all_q_shots)) if all_q_shots else 0
dupes = len(all_q_shots)-uniq_q
log(f"Quality-filtered shot_number records: {len(all_q_shots):,}")
log(f"Unique quality-filtered shot_number records: {uniq_q:,}")
log(f"Duplicate shot_number count in quality subset: {dupes:,}")
add_summary("GEDI","Duplicate shot_number check","PASS" if dupes==0 else "WARN",
            f"{dupes:,} duplicate shot_number records")

# stats
if all_q_agbd:
    a=np.asarray(all_q_agbd,float)
    a=a[np.isfinite(a)]
    log("Quality-filtered AGBD stats (Mg/ha): "
        f"n={len(a):,}, min={np.min(a):.2f}, median={np.median(a):.2f}, "
        f"mean={np.mean(a):.2f}, p95={np.percentile(a,95):.2f}, max={np.max(a):.2f}")
    add_summary("GEDI","AGBD range","PASS",
                f"n={len(a):,}; min={np.min(a):.2f}; median={np.median(a):.2f}; mean={np.mean(a):.2f}; max={np.max(a):.2f}")
if all_q_agbd_se:
    s=np.asarray(all_q_agbd_se,float)
    s=s[np.isfinite(s)]
    log("Quality-filtered AGBD_SE stats: "
        f"n={len(s):,}, median={np.median(s):.2f}, mean={np.mean(s):.2f}, p95={np.percentile(s,95):.2f}")
    add_summary("GEDI","AGBD_SE available","PASS",f"n={len(s):,}; median={np.median(s):.2f}; mean={np.mean(s):.2f}")

# spatial spread across AOI (5x5 grid)
grid_counts=np.zeros((5,5),dtype=int)
if all_q_lons:
    lons=np.asarray(all_q_lons); lats=np.asarray(all_q_lats)
    xi=np.clip(((lons-WEST)/(EAST-WEST)*5).astype(int),0,4)
    yi=np.clip(((lats-SOUTH)/(NORTH-SOUTH)*5).astype(int),0,4)
    for xg,yg in zip(xi,yi):
        grid_counts[yg,xg]+=1
    occupied=int(np.sum(grid_counts>0))
    log(f"Quality-footprint spatial coverage: {occupied}/25 coarse 5x5 AOI cells occupied")
    add_summary("GEDI","Spatial coverage quick check",
                "PASS" if occupied>=10 else "WARN",
                f"{occupied}/25 coarse grid cells occupied")

# ============================================================
# FINAL ACCEPTANCE
# ============================================================
log("")
log("FINAL ACCEPTANCE DECISION")
log("-"*72)

# Gate-0 logic for the locked one-scene / 22-granule design.
# A NISAR AOI-coverage WARN is allowed only as PASS_WITH_WARNINGS and requires
# manual metadata/map verification before scientific processing. An explicit
# AOI coverage FAIL is a Gate-0 FAIL.
coverage_warn = any(
    s["dataset"]=="NISAR" and s["check"]=="Full AOI coverage" and s["status"]=="WARN"
    for s in SUMMARY
)
nisar_required_pass = (
    len(nisar_files) == 1 and
    nisar_core_pass and
    (nisar_aoi_pass or coverage_warn)
)

gedi_required_pass = (
    len(gedi_files) == 22 and
    readable_files == 22 and
    files_with_all_core == 22 and
    total_aoi > 0 and
    total_quality > 0
)

if nisar_required_pass and gedi_required_pass:
    overall="PASS_WITH_WARNINGS" if any(s["status"]=="WARN" for s in SUMMARY) else "PASS"
else:
    overall="FAIL"

# Python typo guard replacement is done below before writing file.

log(f"OVERALL DATA ACCEPTANCE: {overall}")
if overall.startswith("PASS"):
    log("Core datasets are suitable to proceed to formal processing, subject to any WARN items listed below.")
else:
    log("Do NOT proceed to formal modelling until FAIL items are corrected.")

warns=[s for s in SUMMARY if s["status"]=="WARN"]
fails=[s for s in SUMMARY if s["status"]=="FAIL"]
if fails:
    log("")
    log("FAIL ITEMS:")
    for x in fails:
        log(f"- {x['dataset']} | {x['check']} | {x['detail']}")
if warns:
    log("")
    log("WARNING ITEMS:")
    for x in warns:
        log(f"- {x['dataset']} | {x['check']} | {x['detail']}")

# ============================================================
# SAVE OUTPUTS
# ============================================================
pd.DataFrame(SUMMARY).to_csv(OUT_DIR/"DATA_ACCEPTANCE_SUMMARY.csv", index=False)
pd.DataFrame(NISAR_INV).to_csv(OUT_DIR/"NISAR_HDF5_DATASET_INVENTORY.csv", index=False)
pd.DataFrame(GEDI_FILE_ROWS).to_csv(OUT_DIR/"GEDI_FILE_CHECK.csv", index=False)

if all_q_agbd:
    pd.DataFrame({
        "metric":["n","min","median","mean","p95","max"],
        "value":[len(a),np.min(a),np.median(a),np.mean(a),np.percentile(a,95),np.max(a)]
    }).to_csv(OUT_DIR/"GEDI_QUALITY_AGBD_QUICK_STATS.csv", index=False)

pd.DataFrame(grid_counts).to_csv(OUT_DIR/"GEDI_QUALITY_5x5_SPATIAL_COUNTS.csv", index=False, header=False)

with open(OUT_DIR/"DATA_ACCEPTANCE_REPORT.txt","w",encoding="utf-8") as f:
    f.write("\n".join(REPORT))

with open(OUT_DIR/"OVERALL_STATUS.txt","w",encoding="utf-8") as f:
    f.write(overall+"\n")

log("")
log(f"Saved acceptance-test outputs to: {OUT_DIR}")
log("Please send/upload DATA_ACCEPTANCE_REPORT.txt and DATA_ACCEPTANCE_SUMMARY.csv to ChatGPT for review.")
