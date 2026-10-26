import os
from pathlib import Path
from datetime import datetime
import hashlib
import gc
import sys
import traceback

import h5py
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine
from rasterio.crs import CRS
from rasterio.windows import Window
from scipy.ndimage import convolve

# ============================================================
# ISOLATED WORKSPACE — NOTHING IS WRITTEN TO THE MAIN PROJECT
# ============================================================
ROOT = Path(os.environ.get("NISAR_JUNE_ROOT", r"D:\new_nisar_data"))
WORK = ROOT / "Phase8B2_June25_Sensitivity_V2"
QC_DIR = WORK / "00_QC"
PRED_DIR = WORK / "01_Predictors"
QC_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_PREFIX = (
    "NISAR_L2_PR_GCOV_023_170_A_013_4005_DHDH_A_"
    "20260625T232921_20260625T232956"
)

HH_PATH = "/science/LSAR/GCOV/grids/frequencyA/HHHH"
HV_PATH = "/science/LSAR/GCOV/grids/frequencyA/HVHV"
X_PATH  = "/science/LSAR/GCOV/grids/frequencyA/xCoordinates"
Y_PATH  = "/science/LSAR/GCOV/grids/frequencyA/yCoordinates"

EXPECTED_EPSG = 32645
EXPECTED_PIXEL = 10.0

# Deliberately small strip height for Windows memory safety.
ROWS_PER_BLOCK = 128

# Cap the deterministic QC sample used for approximate median/p95.
SAMPLE_PER_BLOCK = 2500

# ============================================================
# HELPERS
# ============================================================
def find_input_h5():
    candidates = [
        p for p in ROOT.iterdir()
        if p.is_file()
        and p.name.startswith(EXPECTED_PREFIX)
    ]
    if not candidates:
        candidates = [
            p for p in ROOT.iterdir()
            if p.is_file()
            and "GCOV_023_170_A_013_4005_DHDH" in p.name
            and "20260625T232921" in p.name
        ]
    if len(candidates) == 0:
        raise FileNotFoundError(
            "Could not find the intended June 25 NISAR H5 directly inside:\n"
            f"{ROOT}"
        )
    if len(candidates) > 1:
        raise RuntimeError(
            "More than one matching June-25 NISAR file was found directly under "
            "D:\\new_nisar_data. Keep only the intended scene there."
        )
    return candidates[0]

def sha256_stream(path, chunk_size=16 * 1024 * 1024):
    h = hashlib.sha256()
    total = path.stat().st_size
    done = 0
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
            done += len(chunk)
            pct = 100.0 * done / total if total else 100.0
            print(f"\rSHA256: {pct:5.1f}%", end="", flush=True)
    print()
    return h.hexdigest()

def find_projection_epsg(h5):
    found = []
    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset) and name.lower().endswith("projection"):
            try:
                v = obj[()]
                if np.isscalar(v):
                    found.append((name, v))
            except Exception:
                pass
    h5.visititems(visitor)
    for name, v in found:
        try:
            epsg = int(np.asarray(v).item())
            if 1000 <= epsg <= 999999:
                return epsg, "/" + name.lstrip("/")
        except Exception:
            pass
    return EXPECTED_EPSG, "expected exact-configuration CRS fallback"

def to_db(a):
    a = np.asarray(a, dtype=np.float32)
    out = np.full(a.shape, np.nan, dtype=np.float32)
    m = np.isfinite(a) & (a > 0)
    # Compute only on valid values; avoids log warnings.
    out[m] = 10.0 * np.log10(a[m])
    return out

def local_nanstd_3x3(a):
    """
    Population SD over a 3x3 window, ignoring invalid values.
    Input is only a small strip plus a 1-row halo.
    """
    a = np.asarray(a, dtype=np.float32)
    valid = np.isfinite(a)

    # float32 keeps the memory footprint small.
    x = np.where(valid, a, 0.0).astype(np.float32, copy=False)
    kernel = np.ones((3, 3), dtype=np.float32)

    count = convolve(valid.astype(np.float32), kernel, mode="constant", cval=0.0)
    s1 = convolve(x, kernel, mode="constant", cval=0.0)
    s2 = convolve(x * x, kernel, mode="constant", cval=0.0)

    mean = np.divide(s1, count, out=np.zeros_like(s1), where=count > 0)
    var = np.divide(s2, count, out=np.zeros_like(s2), where=count > 0) - mean * mean
    np.maximum(var, 0.0, out=var)
    sd = np.sqrt(var, dtype=np.float32)
    sd[count <= 0] = np.nan
    return sd.astype(np.float32, copy=False)

class StreamStats:
    def __init__(self, name):
        self.name = name
        self.n = 0
        self.sum = 0.0
        self.min = np.inf
        self.max = -np.inf
        self.samples = []

    def update(self, arr):
        v = np.asarray(arr, dtype=np.float32)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return
        self.n += int(v.size)
        self.sum += float(np.sum(v, dtype=np.float64))
        self.min = min(self.min, float(np.min(v)))
        self.max = max(self.max, float(np.max(v)))

        if v.size <= SAMPLE_PER_BLOCK:
            s = v
        else:
            idx = np.linspace(0, v.size - 1, SAMPLE_PER_BLOCK, dtype=np.int64)
            s = v[idx]
        self.samples.append(np.asarray(s, dtype=np.float32))

    def summary(self):
        if self.n == 0:
            return {
                "predictor": self.name, "n_finite": 0,
                "min": np.nan, "mean": np.nan,
                "median_approx": np.nan, "p95_approx": np.nan, "max": np.nan,
                "sample_n": 0,
            }
        sample = (
            np.concatenate(self.samples)
            if self.samples else np.empty(0, dtype=np.float32)
        )
        return {
            "predictor": self.name,
            "n_finite": self.n,
            "min": self.min,
            "mean": self.sum / self.n,
            "median_approx": float(np.median(sample)) if sample.size else np.nan,
            "p95_approx": float(np.percentile(sample, 95)) if sample.size else np.nan,
            "max": self.max,
            "sample_n": int(sample.size),
        }

def open_output(path, height, width, transform, crs):
    profile = {
        "driver": "GTiff",
        "height": int(height),
        "width": int(width),
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": -9999.0,
        "compress": "LZW",
        "tiled": True,
        "BIGTIFF": "YES",
        "blockxsize": 256,
        "blockysize": 256,
    }
    return rasterio.open(path, "w", **profile)

def write_strip(dst, arr, row0):
    out = np.where(np.isfinite(arr), arr, -9999.0).astype(np.float32, copy=False)
    dst.write(
        out,
        1,
        window=Window(
            col_off=0,
            row_off=int(row0),
            width=int(out.shape[1]),
            height=int(out.shape[0]),
        ),
    )

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 78)
    print("PHASE 8B.2A V2 — JUNE 25 NISAR CHUNKED / LOW-MEMORY PREPROCESSING")
    print("=" * 78)
    print("All outputs stay under D:\\new_nisar_data.")
    print("D:\\NISAR_Mangrove_Paper is NOT read from or written to in this phase.")
    print(f"Chunk height: {ROWS_PER_BLOCK} rows\n")

    h5_path = find_input_h5()
    print("Input:", h5_path)

    size_gib = h5_path.stat().st_size / (1024**3)
    print(f"Size: {size_gib:.3f} GiB")
    name_ok = h5_path.name.startswith(EXPECTED_PREFIX)
    print("Filename/configuration:", "PASS" if name_ok else "REVIEW")

    print("\nComputing SHA256...")
    sha = sha256_stream(h5_path)
    print("SHA256:", sha)

    out_hh = PRED_DIR / "June25_HH_dB.tif"
    out_hv = PRED_DIR / "June25_HV_dB.tif"
    out_diff = PRED_DIR / "June25_HH_minus_HV.tif"
    out_sd = PRED_DIR / "June25_HV_SD.tif"

    stats = {
        "HH_dB": StreamStats("HH_dB"),
        "HV_dB": StreamStats("HV_dB"),
        "HH_minus_HV": StreamStats("HH_minus_HV"),
        "HV_SD": StreamStats("HV_SD"),
    }

    with h5py.File(h5_path, "r") as h:
        required = [HH_PATH, HV_PATH, X_PATH, Y_PATH]
        missing = [p for p in required if p not in h]
        if missing:
            raise RuntimeError("Missing required datasets:\n" + "\n".join(missing))

        hh_ds = h[HH_PATH]
        hv_ds = h[HV_PATH]
        x = np.asarray(h[X_PATH][:], dtype=np.float64)
        y = np.asarray(h[Y_PATH][:], dtype=np.float64)

        if hh_ds.shape != hv_ds.shape:
            raise RuntimeError(f"HH/HV shapes differ: {hh_ds.shape} vs {hv_ds.shape}")

        height, width = hh_ds.shape
        if (len(y), len(x)) != (height, width):
            raise RuntimeError(
                f"Coordinate lengths do not match raster shape. "
                f"Raster={hh_ds.shape}; y={len(y)}, x={len(x)}"
            )

        dx = float(np.median(np.diff(x)))
        dy = float(np.median(np.diff(y)))
        epsg, epsg_source = find_projection_epsg(h)

        print(f"Grid: {height} rows x {width} cols")
        print(f"dx={dx}; dy={dy}; EPSG={epsg}")

        transform = Affine(
            dx, 0.0, x[0] - dx / 2.0,
            0.0, dy, y[0] - dy / 2.0,
        )
        crs = CRS.from_epsg(int(epsg))

        grid_pass = (
            int(epsg) == EXPECTED_EPSG
            and abs(abs(dx) - EXPECTED_PIXEL) < 1e-6
            and abs(abs(dy) - EXPECTED_PIXEL) < 1e-6
        )

        # ----------------------------------------------------
        # PASS 1: HH_dB, HV_dB, HH-HV
        # ----------------------------------------------------
        print("\nPASS 1/2 — HH_dB, HV_dB, HH-HV")
        with (
            open_output(out_hh, height, width, transform, crs) as dst_hh,
            open_output(out_hv, height, width, transform, crs) as dst_hv,
            open_output(out_diff, height, width, transform, crs) as dst_diff,
        ):
            block_no = 0
            total_blocks = (height + ROWS_PER_BLOCK - 1) // ROWS_PER_BLOCK

            for row0 in range(0, height, ROWS_PER_BLOCK):
                row1 = min(height, row0 + ROWS_PER_BLOCK)
                block_no += 1

                hhlin = np.asarray(hh_ds[row0:row1, :], dtype=np.float32)
                hvlin = np.asarray(hv_ds[row0:row1, :], dtype=np.float32)

                hhdb = to_db(hhlin)
                hvdb = to_db(hvlin)
                del hhlin, hvlin

                diff = np.full(hhdb.shape, np.nan, dtype=np.float32)
                m = np.isfinite(hhdb) & np.isfinite(hvdb)
                diff[m] = hhdb[m] - hvdb[m]
                del m

                write_strip(dst_hh, hhdb, row0)
                write_strip(dst_hv, hvdb, row0)
                write_strip(dst_diff, diff, row0)

                stats["HH_dB"].update(hhdb)
                stats["HV_dB"].update(hvdb)
                stats["HH_minus_HV"].update(diff)

                del hhdb, hvdb, diff
                gc.collect()

                if block_no == 1 or block_no % 10 == 0 or block_no == total_blocks:
                    pct = 100.0 * row1 / height
                    print(
                        f"\r  block {block_no}/{total_blocks} "
                        f"rows {row0}:{row1} ({pct:5.1f}%)",
                        end="", flush=True
                    )
            print()

        # ----------------------------------------------------
        # PASS 2: HV_SD with one-row halo
        # ----------------------------------------------------
        print("\nPASS 2/2 — HV_SD 3x3 local standard deviation")
        with open_output(out_sd, height, width, transform, crs) as dst_sd:
            block_no = 0
            total_blocks = (height + ROWS_PER_BLOCK - 1) // ROWS_PER_BLOCK

            for row0 in range(0, height, ROWS_PER_BLOCK):
                row1 = min(height, row0 + ROWS_PER_BLOCK)
                block_no += 1

                ext0 = max(0, row0 - 1)
                ext1 = min(height, row1 + 1)

                hvlin_ext = np.asarray(hv_ds[ext0:ext1, :], dtype=np.float32)
                hvdb_ext = to_db(hvlin_ext)
                del hvlin_ext

                sd_ext = local_nanstd_3x3(hvdb_ext)
                del hvdb_ext

                central0 = row0 - ext0
                central1 = central0 + (row1 - row0)
                sd = sd_ext[central0:central1, :]
                del sd_ext

                write_strip(dst_sd, sd, row0)
                stats["HV_SD"].update(sd)

                del sd
                gc.collect()

                if block_no == 1 or block_no % 10 == 0 or block_no == total_blocks:
                    pct = 100.0 * row1 / height
                    print(
                        f"\r  block {block_no}/{total_blocks} "
                        f"rows {row0}:{row1} ({pct:5.1f}%)",
                        end="", flush=True
                    )
            print()

    # --------------------------------------------------------
    # QC summaries
    # --------------------------------------------------------
    summaries = [stats[k].summary() for k in [
        "HH_dB", "HV_dB", "HH_minus_HV", "HV_SD"
    ]]
    summary_df = pd.DataFrame(summaries)
    summary_csv = QC_DIR / "June25_predictor_streaming_stats.csv"
    summary_df.to_csv(summary_csv, index=False)

    required_valid = all(int(s["n_finite"]) > 0 for s in summaries)
    overall = "PASS" if (name_ok and grid_pass and required_valid) else "REVIEW"

    report = [
        "PHASE 8B.2A V2 — JUNE 25, 2026 NISAR CHUNKED / LOW-MEMORY QC",
        "=" * 78,
        f"Run time: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "ISOLATION",
        f"Input root: {ROOT}",
        f"Output root: {WORK}",
        "No file under D:\\NISAR_Mangrove_Paper was read or written by this script.",
        "",
        "INPUT",
        f"File: {h5_path}",
        f"Size GiB: {size_gib:.6f}",
        f"SHA256: {sha}",
        f"Filename/configuration: {'PASS' if name_ok else 'REVIEW'}",
        "",
        "GRID",
        f"Raster shape: {height} rows x {width} cols",
        f"EPSG: {epsg}",
        f"EPSG source: {epsg_source}",
        f"dx: {dx}",
        f"dy: {dy}",
        f"EPSG / 10 m grid check: {'PASS' if grid_pass else 'REVIEW'}",
        "",
        "PROCESSING METHOD",
        f"Row-block processing: {ROWS_PER_BLOCK} rows per block",
        "Full 33,912 x 34,632 arrays were NEVER loaded into RAM at once.",
        "HH_dB = 10*log10(HHHH) for finite positive pixels.",
        "HV_dB = 10*log10(HVHV) for finite positive pixels.",
        "HH_minus_HV = HH_dB - HV_dB.",
        "HV_SD = population SD of HV_dB in a 3x3 window, ignoring invalid pixels.",
        "HV_SD block edges use a one-row halo so internal strip boundaries do not",
        "create artificial texture discontinuities.",
        "",
        "STREAMING PREDICTOR STATISTICS",
        "Median and p95 are deterministic sample approximations; n/min/mean/max are",
        "computed across all finite pixels.",
        summary_df.to_string(index=False),
        "",
        "OUTPUTS",
        str(out_hh),
        str(out_hv),
        str(out_diff),
        str(out_sd),
        str(summary_csv),
        "",
        f"PHASE 8B.2A V2 STATUS: {overall}",
        "",
        "NEXT RULE",
        "Do not perform GEDI matching or model reruns until this QC is reviewed.",
    ]

    report_path = QC_DIR / "JUNE25_NISAR_ACCEPTANCE_AND_PREDICTOR_QC_V2.txt"
    report_path.write_text("\n".join(report), encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"PHASE 8B.2A V2 STATUS: {overall}")
    print("=" * 78)
    print("\nStreaming predictor statistics:")
    print(summary_df.to_string(index=False))

    print("\nCreated:")
    print(report_path)
    print(summary_csv)
    print(out_hh)
    print(out_hv)
    print(out_diff)
    print(out_sd)

    print("\nIMPORTANT:")
    print("Keep all June-25 sensitivity files under D:\\new_nisar_data.")
    print("Send the V2 QC report and terminal output to ChatGPT.")
    None  # interactive pause removed for portable execution

if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n" + "=" * 78)
        print("PHASE 8B.2A V2 FAILED")
        print("=" * 78)
        traceback.print_exc()
        None  # interactive pause removed for portable execution
        sys.exit(1)
