from pathlib import Path
from datetime import datetime
import hashlib, json, sys, traceback
import h5py
import numpy as np
import pandas as pd

PROJECT = Path(r"D:\NISAR_Mangrove_Paper")
H5 = PROJECT / "02_NISAR_GCOV" / "Original" / (
    "NISAR_L2_PR_GCOV_030_170_A_013_4005_DHDH_A_"
    "20260917T232917_20260917T232952_P05023_F_F_J_001.h5"
)
OUT = PROJECT / "02_NISAR_GCOV" / "Predictor_Expansion" / "Phase8C0_HDF5_Inventory"
OUT.mkdir(parents=True, exist_ok=True)

FREQ = "/science/LSAR/GCOV/grids/frequencyA"

def sha256(path):
    h = hashlib.sha256()
    total = path.stat().st_size
    done = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(16*1024*1024), b""):
            h.update(chunk)
            done += len(chunk)
            print(f"\rSHA256: {100*done/total:5.1f}%", end="", flush=True)
    print()
    return h.hexdigest()

def clean(v):
    try:
        if isinstance(v, bytes):
            return v.decode("utf-8", errors="replace")
        if isinstance(v, np.ndarray):
            if v.size == 1:
                return clean(v.reshape(-1)[0])
            if v.size <= 10:
                return [clean(x) for x in v.tolist()]
            return f"array(shape={v.shape}, dtype={v.dtype})"
        if isinstance(v, np.generic):
            return v.item()
        return v
    except Exception:
        return str(v)

def attrs_json(obj):
    return json.dumps({str(k): clean(v) for k, v in obj.attrs.items()},
                      ensure_ascii=False, default=str)

def sample_2d(ds, nr=20, nc=20):
    rr = np.unique(np.linspace(0, ds.shape[0]-1, min(nr, ds.shape[0]), dtype=int))
    cc = np.unique(np.linspace(0, ds.shape[1]-1, min(nc, ds.shape[1]), dtype=int))
    vals = [np.asarray(ds[r, cc]) for r in rr]
    return np.concatenate(vals) if vals else np.array([])

def sample_stats(a):
    if a.size == 0:
        return {}
    out = {"sample_n": int(a.size), "sample_complex": bool(np.iscomplexobj(a))}
    if np.iscomplexobj(a):
        m = np.isfinite(a.real) & np.isfinite(a.imag)
        z = a[m]
        out["sample_finite_n"] = int(z.size)
        if z.size:
            mag = np.abs(z).astype(float)
            out.update({
                "sample_real_mean": float(np.mean(z.real)),
                "sample_imag_mean": float(np.mean(z.imag)),
                "sample_abs_min": float(np.min(mag)),
                "sample_abs_median": float(np.median(mag)),
                "sample_abs_mean": float(np.mean(mag)),
                "sample_abs_max": float(np.max(mag)),
            })
    else:
        try:
            x = a.astype(float, copy=False)
            x = x[np.isfinite(x)]
            out["sample_finite_n"] = int(x.size)
            if x.size:
                out.update({
                    "sample_min": float(np.min(x)),
                    "sample_median": float(np.median(x)),
                    "sample_mean": float(np.mean(x)),
                    "sample_max": float(np.max(x)),
                })
        except Exception:
            pass
    return out

def classify(name, path, ds):
    tags = []
    if path.startswith(FREQ + "/"): tags.append("frequencyA")
    tags.append(f"{ds.ndim}D")
    if ds.ndim == 2: tags.append("raster")
    if np.issubdtype(ds.dtype, np.complexfloating): tags.append("complex")
    elif np.issubdtype(ds.dtype, np.floating): tags.append("float")
    elif np.issubdtype(ds.dtype, np.integer): tags.append("integer")
    if name in {"HHHH","HVHV","VVVV"}: tags.append("diagonal_covariance_power")
    if name in {"HHHV","HHVV","HVVV"}: tags.append("offdiagonal_covariance_candidate")
    return ";".join(tags)

def reason(name, ds):
    r = []
    if name in {"HHHH","HVHV"}:
        r.append("already-used radar power layer")
    if name in {"HHHV","HHVV","HVVV"}:
        r.append("possible off-diagonal covariance term for normalized covariance/coherence-style features")
    if ds.ndim == 2 and np.issubdtype(ds.dtype, np.complexfloating):
        r.append("complex raster; inspect magnitude/phase interpretation")
    low = name.lower()
    if "incidence" in low:
        r.append("geometry diagnostic/covariate candidate")
    if "mask" in low or "valid" in low:
        r.append("quality/mask diagnostic")
    return " | ".join(r)

def main():
    print("="*80)
    print("PHASE 8C.0 — SEPTEMBER NISAR HDF5 INVENTORY")
    print("="*80)
    print("Inspection only. No expanded predictors are created.\n")

    if not H5.exists():
        raise FileNotFoundError(H5)

    print("Input:", H5)
    print(f"Size: {H5.stat().st_size/(1024**3):.3f} GiB")
    print("\nComputing SHA256...")
    digest = sha256(H5)
    print("SHA256:", digest)

    rows = []
    immediate = []
    group_attrs = []

    with h5py.File(H5, "r") as h:
        if FREQ not in h:
            raise RuntimeError(f"Missing group: {FREQ}")

        print("\nImmediate Frequency A contents:")
        for k in h[FREQ].keys():
            obj = h[FREQ][k]
            typ = "Dataset" if isinstance(obj, h5py.Dataset) else "Group"
            immediate.append((k, typ))
            print(f"  {k:35s} {typ}")

        for gp in ["/", "/science/LSAR/GCOV", "/science/LSAR/GCOV/grids", FREQ]:
            obj = h if gp == "/" else h[gp]
            group_attrs.append({"group_path": gp, "attributes_json": attrs_json(obj)})

        def visit(name, obj):
            if not isinstance(obj, h5py.Dataset):
                return
            path = "/" + name.lstrip("/")
            nm = Path(path).name
            row = {
                "path": path,
                "name": nm,
                "ndim": int(obj.ndim),
                "shape": str(tuple(obj.shape)),
                "dtype": str(obj.dtype),
                "size_elements": int(obj.size),
                "storage_size_bytes": int(obj.id.get_storage_size()),
                "compression": obj.compression,
                "chunks": str(obj.chunks),
                "classification": classify(nm, path, obj),
                "candidate_reason": reason(nm, obj),
                "attributes_json": attrs_json(obj),
            }
            if path.startswith(FREQ + "/") and obj.ndim == 2:
                try:
                    row.update(sample_stats(sample_2d(obj)))
                except Exception as e:
                    row["sample_error"] = repr(e)
            rows.append(row)

        h.visititems(visit)

    inv = pd.DataFrame(rows)
    fa = inv[inv["path"].str.startswith(FREQ + "/")].copy()
    ras = fa[fa["ndim"] == 2].copy()
    cand = fa[fa["candidate_reason"].fillna("").str.len() > 0].copy()

    inv.to_csv(OUT/"September_HDF5_dataset_inventory.csv", index=False)
    fa.to_csv(OUT/"September_frequencyA_dataset_inventory.csv", index=False)
    ras.to_csv(OUT/"September_frequencyA_2D_rasters.csv", index=False)
    cand.to_csv(OUT/"September_frequencyA_candidate_layers.csv", index=False)
    pd.DataFrame(group_attrs).to_csv(OUT/"September_HDF5_group_attributes.csv", index=False)

    lines = [
        "PHASE 8C.0 — SEPTEMBER NISAR HDF5 INVENTORY / PREDICTOR DESIGN",
        "="*80,
        f"Run time: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"Input: {H5}",
        f"SHA256: {digest}",
        "",
        "PURPOSE",
        "Inspect the original September GCOV before designing expanded predictors.",
        "No predictor raster was created.",
        "",
        "IMMEDIATE FREQUENCY A CONTENTS",
    ]
    lines += [f"- {k}: {t}" for k,t in immediate]
    lines += [
        "",
        f"Total datasets inventoried: {len(inv)}",
        f"Frequency A datasets: {len(fa)}",
        f"Frequency A 2D rasters: {len(ras)}",
        f"Flagged candidate/diagnostic layers: {len(cand)}",
        "",
        "FREQUENCY A 2D RASTERS",
    ]
    cols = [c for c in [
        "name","shape","dtype","classification","candidate_reason",
        "sample_n","sample_finite_n","sample_min","sample_median","sample_mean",
        "sample_max","sample_abs_median","sample_abs_mean"
    ] if c in ras.columns]
    lines.append(ras[cols].to_string(index=False))
    lines += ["", "FLAGGED CANDIDATES"]
    if len(cand):
        lines.append(cand[["name","shape","dtype","classification","candidate_reason"]].to_string(index=False))
    else:
        lines.append("None automatically flagged.")
    lines += [
        "",
        "NEXT RULE",
        "Do not create expanded predictors yet.",
        "Review this inventory first, then pre-specify a compact predictor set."
    ]

    report = OUT/"PHASE8C0_SEPTEMBER_HDF5_INVENTORY_REPORT.txt"
    report.write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "="*80)
    print("PHASE 8C.0 COMPLETE")
    print("="*80)
    print("\nFrequency A 2D raster names:")
    for n in ras["name"]:
        print(" -", n)
    print("\nCreated:")
    print(report)
    print(OUT/"September_frequencyA_dataset_inventory.csv")
    print(OUT/"September_frequencyA_candidate_layers.csv")
    print("\nUpload the report and the two Frequency A CSV files before Phase 8C.1.")
    input("\nPress Enter to close...")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\nPHASE 8C.0 FAILED")
        traceback.print_exc()
        input("\nPress Enter to close...")
        sys.exit(1)
