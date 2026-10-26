from pathlib import Path
from datetime import datetime
import sys
import hashlib
import math

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial.distance import pdist, squareform
from scipy import sparse
import matplotlib.pyplot as plt

# ============================================================
# PROJECT SETTINGS
# ============================================================
PROJECT_ROOT = Path(r"D:\NISAR_Mangrove_Paper")
INPUT_CSV = PROJECT_ROOT / "04_Matched_Data" / "GEDI_NISAR_modeling_table_FINAL.csv"

OUT_DIR = PROJECT_ROOT / "06_Results" / "Spatial_Diagnostics"
TABLE_DIR = OUT_DIR / "Tables"
FIG_DIR = OUT_DIR / "Figures"
LOG_PATH = PROJECT_ROOT / "00_Documentation" / "analysis_log.txt"

RESPONSE = "agbd"
SEED = 42
N_PERM = 199

# Fine short-range bins, chosen because GEDI footprints are strongly clustered
# along tracks and the earlier 500 m variogram bins were too coarse.
BINS_M = np.array([0, 100, 250, 500, 750, 1000, 1500, 2000, 3000, 5000, 7500, 10000, 15000])
MORAN_THRESHOLDS_M = [100, 250, 500, 750, 1000, 1500, 2000, 3000, 5000]
CANDIDATE_BLOCK_SIZES_M = [500, 1000, 2000, 3000, 5000]

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

def package_version(name):
    try:
        import importlib.metadata as md
        return md.version(name)
    except Exception:
        return "unknown"

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def append_log(lines):
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    block = ["", f"[{stamp}] PHASE 5B — SPATIAL DEPENDENCE DIAGNOSTICS"] + lines + ["-" * 72]
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write("\n".join(block) + "\n")

def detrend_first_order(x, y, z):
    X = np.column_stack([
        np.ones(len(z)),
        (x - np.mean(x)) / 1000.0,
        (y - np.mean(y)) / 1000.0
    ])
    beta, *_ = np.linalg.lstsq(X, z, rcond=None)
    return z - X @ beta

def binned_semivariogram(dist_condensed, values, label):
    gamma_condensed = 0.5 * pdist(values.reshape(-1, 1), metric="sqeuclidean")
    var = float(np.var(values, ddof=1))
    rows = []

    for lo, hi in zip(BINS_M[:-1], BINS_M[1:]):
        m = (dist_condensed >= lo) & (dist_condensed < hi)
        n = int(np.sum(m))
        if n == 0:
            continue
        g = float(np.mean(gamma_condensed[m]))
        # Under second-order stationarity, Cov(h) ≈ Var - gamma(h).
        # This is only a diagnostic proxy, not a formal correlation estimate.
        rho_proxy = 1.0 - g / var if var > 0 else np.nan
        rows.append({
            "variable": label,
            "distance_low_m": lo,
            "distance_high_m": hi,
            "distance_mid_m": (lo + hi) / 2,
            "pair_count": n,
            "semivariance": g,
            "sample_variance": var,
            "covariance_proxy": var - g,
            "correlation_proxy": rho_proxy,
        })

    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / f"{label}_short_range_semivariogram.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.scatter(out["distance_mid_m"] / 1000.0, out["semivariance"], s=45)
    ax.axhline(var, linestyle="--", linewidth=1.2)
    ax.set_xlabel("Separation distance (km)")
    ax.set_ylabel("Semivariance")
    ax.set_title(f"Short-range empirical semivariogram — {label}", pad=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{label}_short_range_semivariogram.png", dpi=450, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{label}_short_range_semivariogram.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.plot(out["distance_mid_m"] / 1000.0, out["correlation_proxy"], marker="o")
    ax.axhline(0.0, linewidth=1.0)
    ax.axhline(0.05, linestyle="--", linewidth=1.0)
    ax.axhline(-0.05, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Separation distance (km)")
    ax.set_ylabel("Correlation proxy: 1 − γ(h)/Var")
    ax.set_title(f"Short-range dependence diagnostic — {label}", pad=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{label}_correlation_proxy.png", dpi=450, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{label}_correlation_proxy.pdf", bbox_inches="tight")
    plt.close(fig)

    return out

def morans_i_binary_distance(dist_square, values, threshold_m, rng):
    n = len(values)
    z = values - np.mean(values)
    denom = float(np.sum(z * z))

    # Symmetric binary distance-band weights; diagonal excluded.
    mask = (dist_square > 0) & (dist_square <= threshold_m)
    W = sparse.csr_matrix(mask.astype(np.float64))
    S0 = float(W.sum())
    if S0 == 0 or denom == 0:
        return np.nan, np.nan, 0, 0.0

    I_obs = (n / S0) * float(z @ (W @ z)) / denom

    perm = np.empty(N_PERM, dtype=float)
    for k in range(N_PERM):
        zp = rng.permutation(z)
        perm[k] = (n / S0) * float(zp @ (W @ zp)) / denom

    # Two-sided pseudo-p value around the permutation mean.
    center = float(np.mean(perm))
    extreme = np.sum(np.abs(perm - center) >= abs(I_obs - center))
    p = (extreme + 1) / (N_PERM + 1)

    pair_count_unique = int(S0 / 2.0)
    mean_neighbors = S0 / n
    return I_obs, p, pair_count_unique, mean_neighbors

def moran_table(dist_square, raw, detrended):
    rng = np.random.default_rng(SEED)
    rows = []

    for label, values in [("AGBD_raw", raw), ("AGBD_detrended", detrended)]:
        for d in MORAN_THRESHOLDS_M:
            I, p, pairs, mean_neigh = morans_i_binary_distance(dist_square, values, d, rng)
            rows.append({
                "variable": label,
                "threshold_m": d,
                "moran_I": I,
                "permutation_p_two_sided": p,
                "unique_neighbor_pairs": pairs,
                "mean_neighbors_per_footprint": mean_neigh,
                "permutations": N_PERM,
            })

    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "MoranI_distance_band_diagnostics.csv", index=False)

    for label in out["variable"].unique():
        sub = out[out["variable"] == label]
        fig, ax = plt.subplots(figsize=(8.2, 5.4))
        ax.plot(sub["threshold_m"] / 1000.0, sub["moran_I"], marker="o")
        ax.axhline(0.0, linewidth=1.0)
        ax.set_xlabel("Cumulative distance threshold (km)")
        ax.set_ylabel("Moran's I")
        ax.set_title(f"Distance-band Moran's I — {label}", pad=12)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"{label}_MoranI_distance_band.png", dpi=450, bbox_inches="tight")
        fig.savefig(FIG_DIR / f"{label}_MoranI_distance_band.pdf", bbox_inches="tight")
        plt.close(fig)

    return out

def nearest_neighbor(x, y):
    coords = np.column_stack([x, y])
    D = squareform(pdist(coords))
    np.fill_diagonal(D, np.inf)
    nn = np.min(D, axis=1)
    return D, nn

def block_diagnostics(df, x, y):
    xmin = float(np.min(x))
    ymin = float(np.min(y))
    rows = []
    assignment_cols = pd.DataFrame({
        "shot_number": df["shot_number"] if "shot_number" in df.columns else np.arange(len(df)),
        "nisar_x": x,
        "nisar_y": y,
        "agbd": df[RESPONSE].to_numpy(),
    })

    for bs in CANDIDATE_BLOCK_SIZES_M:
        bx = np.floor((x - xmin) / bs).astype(int)
        by = np.floor((y - ymin) / bs).astype(int)
        bid = np.array([f"{a}_{b}" for a, b in zip(bx, by)])

        temp = pd.DataFrame({"block_id": bid, "agbd": df[RESPONSE].to_numpy()})
        g = temp.groupby("block_id").agg(
            n=("agbd", "size"),
            mean_agbd=("agbd", "mean"),
            sd_agbd=("agbd", "std"),
        ).reset_index()

        counts = g["n"]
        rows.append({
            "block_size_m": bs,
            "occupied_blocks": int(len(g)),
            "median_footprints_per_block": float(counts.median()),
            "mean_footprints_per_block": float(counts.mean()),
            "max_footprints_per_block": int(counts.max()),
            "min_footprints_per_block": int(counts.min()),
            "blocks_with_ge_5": int((counts >= 5).sum()),
            "blocks_with_ge_10": int((counts >= 10).sum()),
            "theoretical_max_folds_if_one_block_each": int(len(g)),
            "five_fold_feasible_by_block_count": bool(len(g) >= 10),
        })

        assignment_cols[f"block_{bs}m"] = bid
        g.to_csv(TABLE_DIR / f"block_summary_{bs}m.csv", index=False)

    diag = pd.DataFrame(rows)
    diag.to_csv(TABLE_DIR / "candidate_block_size_diagnostics.csv", index=False)
    assignment_cols.to_csv(TABLE_DIR / "candidate_block_ids_by_footprint.csv", index=False)
    return diag

def main():
    print("=" * 72)
    print("PHASE 5B — CORRECTED SPATIAL DEPENDENCE & BLOCK-SIZE DIAGNOSTICS")
    print("=" * 72)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Final modelling table not found:\n{INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    required = {RESPONSE, "longitude", "latitude"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {sorted(missing)}")

    print(f"Input rows: {len(df):,}")
    print(f"Input SHA256: {sha256_file(INPUT_CSV)}")

    tr = Transformer.from_crs("EPSG:4326", "EPSG:32645", always_xy=True)
    x, y = tr.transform(
        df["longitude"].to_numpy(dtype=float),
        df["latitude"].to_numpy(dtype=float)
    )
    x = np.asarray(x)
    y = np.asarray(y)
    raw = pd.to_numeric(df[RESPONSE], errors="coerce").to_numpy(dtype=float)

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(raw)
    if not np.all(valid):
        df = df.loc[valid].reset_index(drop=True)
        x, y, raw = x[valid], y[valid], raw[valid]

    detrended = detrend_first_order(x, y, raw)
    coords = np.column_stack([x, y])

    dist_condensed = pdist(coords)
    dist_square, nn = nearest_neighbor(x, y)

    print(f"Nearest-neighbor median: {np.median(nn):.1f} m")
    print(f"Nearest-neighbor 95th percentile: {np.quantile(nn, 0.95):.1f} m")

    raw_sv = binned_semivariogram(dist_condensed, raw, "AGBD_raw")
    det_sv = binned_semivariogram(dist_condensed, detrended, "AGBD_detrended")
    moran = moran_table(dist_square, raw, detrended)
    block_diag = block_diagnostics(df, x, y)

    # A simple, transparent evidence table for review — NOT an automatic final decision.
    combined = pd.concat([raw_sv, det_sv], ignore_index=True)
    combined.to_csv(TABLE_DIR / "short_range_dependence_combined.csv", index=False)

    # Identify first bin midpoint where |proxy| <= 0.05, only as a descriptive guide.
    guides = []
    for label, sub in combined.groupby("variable"):
        sub = sub.sort_values("distance_mid_m")
        ok = sub[np.abs(sub["correlation_proxy"]) <= 0.05]
        first_mid = float(ok.iloc[0]["distance_mid_m"]) if len(ok) else np.nan
        guides.append({
            "variable": label,
            "first_bin_midpoint_with_abs_correlation_proxy_le_0p05_m": first_mid,
            "note": "Descriptive guide only; not a fitted variogram range."
        })
    guide_df = pd.DataFrame(guides)
    guide_df.to_csv(TABLE_DIR / "decorrelation_distance_guides.csv", index=False)

    print("\nShort-range dependence guide:")
    for _, r in guide_df.iterrows():
        print(f"  {r['variable']:<16s}: {r.iloc[1]:.0f} m")

    print("\nCandidate block sizes:")
    print(block_diag.to_string(index=False))

    append_log([
        "Script: phase5b_spatial_diagnostics.py",
        f"Input rows: {len(df):,}",
        f"Input SHA256: {sha256_file(INPUT_CSV)}",
        f"Nearest-neighbor median: {np.median(nn):.3f} m",
        f"Nearest-neighbor p95: {np.quantile(nn, 0.95):.3f} m",
        "The previous 137.75 km exponential variogram range was rejected as a degenerate extrapolation "
        "because the fitted partial sill was effectively zero and observed pair distances ended near 15.3 km.",
        "This phase uses fine short-range semivariogram bins, distance-band Moran's I, and candidate block-count diagnostics.",
        "No final CV folds are assigned here.",
        f"Python: {sys.version.split()[0]}; numpy={package_version('numpy')}; "
        f"pandas={package_version('pandas')}; scipy={package_version('scipy')}; "
        f"pyproj={package_version('pyproj')}; matplotlib={package_version('matplotlib')}",
    ])

    print("\n" + "=" * 72)
    print("PHASE 5B COMPLETE — REVIEW REQUIRED")
    print("=" * 72)
    print(f"Tables:  {TABLE_DIR}")
    print(f"Figures: {FIG_DIR}")
    print("\nUpload these key outputs:")
    print(f" - {TABLE_DIR / 'short_range_dependence_combined.csv'}")
    print(f" - {TABLE_DIR / 'MoranI_distance_band_diagnostics.csv'}")
    print(f" - {TABLE_DIR / 'candidate_block_size_diagnostics.csv'}")
    print(f" - {TABLE_DIR / 'decorrelation_distance_guides.csv'}")
    print(f" - {FIG_DIR / 'AGBD_raw_correlation_proxy.png'}")
    print(f" - {FIG_DIR / 'AGBD_detrended_correlation_proxy.png'}")
    print(f" - {FIG_DIR / 'AGBD_raw_MoranI_distance_band.png'}")
    print(f" - {FIG_DIR / 'AGBD_detrended_MoranI_distance_band.png'}")
    print("\nDo NOT begin Phase 6 until these are reviewed.")
    input("\nPress Enter to close...")

if __name__ == "__main__":
    main()
