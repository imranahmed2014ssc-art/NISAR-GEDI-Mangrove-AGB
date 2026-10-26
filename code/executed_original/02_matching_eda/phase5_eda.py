from pathlib import Path
from datetime import datetime
import sys
import hashlib
import warnings

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy import stats
from scipy.spatial.distance import pdist
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
except Exception as e:
    raise SystemExit(
        "statsmodels is required for LOWESS plots.\n"
        "Install with:\n"
        "python -m pip install statsmodels\n"
        f"Original error: {e}"
    )

# ============================================================
# PROJECT SETTINGS
# ============================================================
PROJECT_ROOT = Path(r"D:\NISAR_Mangrove_Paper")
INPUT_CSV = PROJECT_ROOT / "04_Matched_Data" / "GEDI_NISAR_modeling_table_FINAL.csv"

OUT_DIR = PROJECT_ROOT / "06_Results" / "EDA"
TABLE_DIR = OUT_DIR / "Tables"
FIG_DIR = OUT_DIR / "Figures"
LOG_PATH = PROJECT_ROOT / "00_Documentation" / "analysis_log.txt"

PREDICTORS = ["HH_dB", "HV_dB", "HH_minus_HV", "HV_SD"]
RESPONSE = "agbd"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# HELPERS
# ============================================================
def package_version(name):
    try:
        import importlib.metadata as md
        return md.version(name)
    except Exception:
        return "unknown"

def append_log(lines):
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    block = ["", f"[{stamp}] PHASE 5 — EXPLORATORY DATA ANALYSIS"] + lines + ["-" * 72]
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write("\n".join(block) + "\n")

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def descriptive_row(name, s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    return {
        "variable": name,
        "n": int(s.size),
        "mean": float(s.mean()),
        "sd": float(s.std(ddof=1)),
        "median": float(s.median()),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "min": float(s.min()),
        "max": float(s.max()),
    }

def save_distribution(df):
    vals = df[RESPONSE].dropna()
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.hist(vals, bins=35)
    ax.set_xlabel("GEDI AGBD (Mg/ha)")
    ax.set_ylabel("Footprint count")
    ax.set_title("GEDI AGBD distribution — final modelling table", pad=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "AGBD_distribution.png", dpi=450, bbox_inches="tight")
    fig.savefig(FIG_DIR / "AGBD_distribution.pdf", bbox_inches="tight")
    plt.close(fig)

    # Descriptive 25 Mg/ha bins (not ecological classes).
    max_edge = max(225, int(np.ceil(vals.max() / 25.0) * 25))
    edges = np.arange(0, max_edge + 25, 25)
    cats = pd.cut(vals, bins=edges, right=False, include_lowest=True)
    counts = cats.value_counts(sort=False).reset_index()
    counts.columns = ["AGBD_bin_Mg_ha", "count"]
    counts.to_csv(TABLE_DIR / "AGBD_25Mg_bin_counts.csv", index=False)

    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    x = np.arange(len(counts))
    ax.bar(x, counts["count"])
    ax.set_xticks(x)
    ax.set_xticklabels(counts["AGBD_bin_Mg_ha"].astype(str), rotation=45, ha="right")
    ax.set_xlabel("AGBD descriptive bin (Mg/ha)")
    ax.set_ylabel("Footprint count")
    ax.set_title("AGBD counts in descriptive 25 Mg/ha bins", pad=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "AGBD_bin_counts.png", dpi=450, bbox_inches="tight")
    fig.savefig(FIG_DIR / "AGBD_bin_counts.pdf", bbox_inches="tight")
    plt.close(fig)

def correlation_outputs(df):
    rows = []
    for p in PREDICTORS:
        sub = df[[RESPONSE, p]].dropna()
        pearson_r, pearson_p = stats.pearsonr(sub[p], sub[RESPONSE])
        spearman_rho, spearman_p = stats.spearmanr(sub[p], sub[RESPONSE])
        rows.append({
            "predictor": p,
            "n": len(sub),
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_rho": spearman_rho,
            "spearman_p": spearman_p,
        })
    corr_response = pd.DataFrame(rows)
    corr_response.to_csv(TABLE_DIR / "predictor_vs_AGBD_correlations.csv", index=False)

    predictor_corr = df[PREDICTORS].corr(method="pearson")
    predictor_corr.to_csv(TABLE_DIR / "predictor_Pearson_correlation_matrix.csv")

    fig, ax = plt.subplots(figsize=(7.0, 6.2))
    im = ax.imshow(predictor_corr.to_numpy(), vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(PREDICTORS)))
    ax.set_yticks(np.arange(len(PREDICTORS)))
    ax.set_xticklabels(PREDICTORS, rotation=35, ha="right")
    ax.set_yticklabels(PREDICTORS)
    for i in range(len(PREDICTORS)):
        for j in range(len(PREDICTORS)):
            ax.text(j, i, f"{predictor_corr.iloc[i, j]:.2f}",
                    ha="center", va="center")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Pearson r")
    ax.set_title("Predictor correlation matrix", pad=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "predictor_correlation_matrix.png", dpi=450, bbox_inches="tight")
    fig.savefig(FIG_DIR / "predictor_correlation_matrix.pdf", bbox_inches="tight")
    plt.close(fig)

    return corr_response, predictor_corr

def collinearity_diagnostics(df):
    X = df[PREDICTORS].dropna().to_numpy(dtype=float)
    rank = int(np.linalg.matrix_rank(X))
    condition = float(np.linalg.cond(X))

    rows = [{
        "diagnostic": "predictor_matrix_rank",
        "value": rank,
        "note": f"{len(PREDICTORS)} columns total"
    },{
        "diagnostic": "condition_number_unscaled",
        "value": condition,
        "note": "Very large values indicate severe multicollinearity"
    }]

    # VIF-like calculation using R² from regressing each predictor on the others.
    for j, name in enumerate(PREDICTORS):
        y = X[:, j]
        Z = np.delete(X, j, axis=1)
        Z = np.column_stack([np.ones(len(Z)), Z])
        beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
        pred = Z @ beta
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        vif = np.inf if (np.isfinite(r2) and r2 >= 1 - 1e-12) else 1.0 / (1.0 - r2)
        rows.append({
            "diagnostic": f"VIF_{name}",
            "value": vif,
            "note": f"R2 of {name} explained by other predictors = {r2:.12f}"
        })

    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "collinearity_diagnostics.csv", index=False)
    return rank, condition, out

def scatter_lowess(df):
    for p in PREDICTORS:
        sub = df[[p, RESPONSE]].dropna().sort_values(p)
        x = sub[p].to_numpy()
        y = sub[RESPONSE].to_numpy()
        smooth = lowess(y, x, frac=0.25, it=1, return_sorted=True)

        fig, ax = plt.subplots(figsize=(7.4, 5.6))
        ax.scatter(x, y, s=11, alpha=0.28)
        ax.plot(smooth[:, 0], smooth[:, 1], linewidth=2.0)
        ax.set_xlabel(p.replace("_", " "))
        ax.set_ylabel("GEDI AGBD (Mg/ha)")
        ax.set_title(f"{p} vs GEDI AGBD with LOWESS", pad=12)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"{p}_vs_AGBD_LOWESS.png", dpi=450, bbox_inches="tight")
        fig.savefig(FIG_DIR / f"{p}_vs_AGBD_LOWESS.pdf", bbox_inches="tight")
        plt.close(fig)

def temporal_outputs(df):
    if "delta_days_to_NISAR" not in df.columns:
        return None

    delta = pd.to_numeric(df["delta_days_to_NISAR"], errors="coerce")
    rows = [{
        "n": int(delta.notna().sum()),
        "min_days": float(delta.min()),
        "median_days": float(delta.median()),
        "mean_days": float(delta.mean()),
        "max_days": float(delta.max()),
    }]
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "temporal_mismatch_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    ax.hist(delta.dropna(), bins=30)
    ax.set_xlabel("Days from GEDI acquisition to NISAR acquisition")
    ax.set_ylabel("Footprint count")
    ax.set_title("GEDI–NISAR temporal mismatch", pad=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "temporal_mismatch_histogram.png", dpi=450, bbox_inches="tight")
    fig.savefig(FIG_DIR / "temporal_mismatch_histogram.pdf", bbox_inches="tight")
    plt.close(fig)

    if "acquisition_date" in df.columns:
        dt = pd.to_datetime(df["acquisition_date"], errors="coerce", utc=True)
        by_year = dt.dt.year.value_counts().sort_index().rename_axis("GEDI_year").reset_index(name="count")
        by_year.to_csv(TABLE_DIR / "GEDI_counts_by_year_final_table.csv", index=False)

    return out

def spatial_map(df):
    fig, ax = plt.subplots(figsize=(8.3, 7.0))
    sc = ax.scatter(
        df["longitude"], df["latitude"],
        c=df[RESPONSE], s=14, alpha=0.75
    )
    cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.025)
    cb.set_label("GEDI AGBD (Mg/ha)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Spatial distribution of final GEDI modelling footprints", pad=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "GEDI_footprints_AGBD_map.png", dpi=450, bbox_inches="tight")
    fig.savefig(FIG_DIR / "GEDI_footprints_AGBD_map.pdf", bbox_inches="tight")
    plt.close(fig)

def nearest_neighbor_summary(x, y):
    coords = np.column_stack([x, y])
    n = len(coords)
    # Efficient enough for n~1641 without an extra dependency.
    nn = np.full(n, np.inf)
    for i in range(n):
        d2 = np.sum((coords - coords[i])**2, axis=1)
        d2[i] = np.inf
        nn[i] = np.sqrt(np.min(d2))
    s = pd.Series(nn)
    out = pd.DataFrame([{
        "n": n,
        "min_m": float(s.min()),
        "p25_m": float(s.quantile(0.25)),
        "median_m": float(s.median()),
        "mean_m": float(s.mean()),
        "p75_m": float(s.quantile(0.75)),
        "p95_m": float(s.quantile(0.95)),
        "max_m": float(s.max()),
    }])
    out.to_csv(TABLE_DIR / "nearest_neighbor_distance_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    ax.hist(nn, bins=35)
    ax.set_xlabel("Nearest-neighbor distance (m)")
    ax.set_ylabel("Footprint count")
    ax.set_title("GEDI footprint nearest-neighbor distances", pad=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "nearest_neighbor_distance_histogram.png", dpi=450, bbox_inches="tight")
    fig.savefig(FIG_DIR / "nearest_neighbor_distance_histogram.pdf", bbox_inches="tight")
    plt.close(fig)
    return out

def exponential_variogram(h, nugget, partial_sill, a):
    return nugget + partial_sill * (1.0 - np.exp(-h / a))

def empirical_variogram(coords, values, label):
    # Pairwise distances and semivariances.
    dist = pdist(coords, metric="euclidean")
    semivar = 0.5 * pdist(values.reshape(-1, 1), metric="sqeuclidean")

    # Limit to 90th percentile distance to avoid sparse extreme-distance bins.
    max_d = float(np.quantile(dist, 0.90))
    bin_width = 500.0
    edges = np.arange(0.0, max_d + bin_width, bin_width)
    if len(edges) < 8:
        bin_width = max_d / 10.0
        edges = np.linspace(0.0, max_d, 11)

    centers = []
    gamma = []
    counts = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (dist >= lo) & (dist < hi)
        if np.sum(m) >= 30:
            centers.append((lo + hi) / 2.0)
            gamma.append(float(np.mean(semivar[m])))
            counts.append(int(np.sum(m)))

    centers = np.asarray(centers, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    counts = np.asarray(counts, dtype=int)

    emp = pd.DataFrame({
        "distance_m": centers,
        "semivariance": gamma,
        "pair_count": counts,
    })
    emp.to_csv(TABLE_DIR / f"{label}_empirical_variogram.csv", index=False)

    sample_var = float(np.var(values, ddof=1))
    fit_status = "not_fitted"
    nugget = partial = a = practical_range = np.nan

    if len(centers) >= 5 and np.all(np.isfinite(gamma)):
        p0 = [max(0.0, float(np.min(gamma))), max(sample_var * 0.8, 1e-6), max(centers[len(centers)//3], 500.0)]
        try:
            popt, _ = curve_fit(
                exponential_variogram,
                centers,
                gamma,
                p0=p0,
                bounds=([0.0, 0.0, 50.0], [sample_var*5 + 1e-6, sample_var*5 + 1e-6, max_d*3]),
                maxfev=50000,
            )
            nugget, partial, a = map(float, popt)
            practical_range = 3.0 * a
            fit_status = "exponential_fit"
        except Exception:
            pass

    if not np.isfinite(practical_range):
        target = 0.95 * sample_var
        idx = np.where(gamma >= target)[0]
        if len(idx):
            practical_range = float(centers[idx[0]])
            fit_status = "empirical_95pct_variance"
        else:
            practical_range = float(max_d)
            fit_status = "range_not_reached_within_90pct_pair_distance"

    summary = pd.DataFrame([{
        "variable": label,
        "sample_variance": sample_var,
        "fit_status": fit_status,
        "nugget": nugget,
        "partial_sill": partial,
        "scale_a_m": a,
        "estimated_practical_range_m": practical_range,
        "estimated_practical_range_km": practical_range / 1000.0,
        "analysis_max_distance_m": max_d,
        "bin_width_m": bin_width,
    }])
    summary.to_csv(TABLE_DIR / f"{label}_variogram_range_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.scatter(centers / 1000.0, gamma, s=np.clip(counts / 1000.0, 12, 70), alpha=0.75)
    if fit_status == "exponential_fit":
        xx = np.linspace(0, max(centers.max(), practical_range), 300)
        yy = exponential_variogram(xx, nugget, partial, a)
        ax.plot(xx / 1000.0, yy, linewidth=2.0)
        ax.axvline(practical_range / 1000.0, linestyle="--", linewidth=1.2)
    ax.set_xlabel("Separation distance (km)")
    ax.set_ylabel("Semivariance")
    ax.set_title(f"Empirical semivariogram — {label}", pad=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{label}_semivariogram.png", dpi=450, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{label}_semivariogram.pdf", bbox_inches="tight")
    plt.close(fig)

    return summary

def main():
    print("=" * 72)
    print("PHASE 5 — EXPLORATORY DATA ANALYSIS")
    print("=" * 72)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Final modelling table not found:\n{INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    required = {RESPONSE, "longitude", "latitude", *PREDICTORS}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {sorted(missing)}")

    print(f"Input: {INPUT_CSV}")
    print(f"Rows: {len(df):,}")
    print(f"SHA256: {sha256_file(INPUT_CSV)}")

    # Freeze provenance of the final table used for EDA.
    provenance = pd.DataFrame([{
        "input_file": str(INPUT_CSV),
        "rows": len(df),
        "columns": len(df.columns),
        "sha256": sha256_file(INPUT_CSV),
        "analysis_timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
    }])
    provenance.to_csv(TABLE_DIR / "EDA_input_provenance.csv", index=False)

    # Core descriptive statistics.
    stats_rows = [descriptive_row(RESPONSE, df[RESPONSE])]
    for p in PREDICTORS:
        stats_rows.append(descriptive_row(p, df[p]))
    if "agbd_se" in df.columns:
        stats_rows.append(descriptive_row("agbd_se", df["agbd_se"]))
    if "delta_days_to_NISAR" in df.columns:
        stats_rows.append(descriptive_row("delta_days_to_NISAR", df["delta_days_to_NISAR"]))
    desc = pd.DataFrame(stats_rows)
    desc.to_csv(TABLE_DIR / "EDA_summary_statistics.csv", index=False)

    save_distribution(df)
    corr_response, predictor_corr = correlation_outputs(df)
    rank, condition, coll = collinearity_diagnostics(df)
    scatter_lowess(df)
    temporal_outputs(df)
    spatial_map(df)

    # Project coordinates.
    tr = Transformer.from_crs("EPSG:4326", "EPSG:32645", always_xy=True)
    x, y = tr.transform(
        df["longitude"].to_numpy(dtype=float),
        df["latitude"].to_numpy(dtype=float)
    )
    x = np.asarray(x)
    y = np.asarray(y)
    coords = np.column_stack([x, y])

    nearest_neighbor_summary(x, y)

    agbd = pd.to_numeric(df[RESPONSE], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(agbd) & np.isfinite(x) & np.isfinite(y)
    coords_v = coords[valid]
    agbd_v = agbd[valid]

    # Raw AGBD semivariogram.
    raw_range = empirical_variogram(coords_v, agbd_v, "AGBD_raw")

    # Detrend AGBD by first-order spatial trend and calculate residual semivariogram.
    Z = np.column_stack([
        np.ones(len(coords_v)),
        (coords_v[:, 0] - coords_v[:, 0].mean()) / 1000.0,
        (coords_v[:, 1] - coords_v[:, 1].mean()) / 1000.0,
    ])
    beta, *_ = np.linalg.lstsq(Z, agbd_v, rcond=None)
    residual = agbd_v - Z @ beta
    detrended_range = empirical_variogram(coords_v, residual, "AGBD_detrended")

    spatial_ranges = pd.concat([raw_range, detrended_range], ignore_index=True)
    spatial_ranges.to_csv(TABLE_DIR / "spatial_autocorrelation_range_summary.csv", index=False)

    # Correlation significance summary to console.
    print("\nPredictor vs AGBD correlations:")
    for _, r in corr_response.iterrows():
        print(
            f"  {r['predictor']:<12s} "
            f"Pearson r={r['pearson_r']:+.3f}, "
            f"Spearman rho={r['spearman_rho']:+.3f}"
        )

    print("\nCollinearity:")
    print(f"  Predictor matrix rank: {rank}/{len(PREDICTORS)}")
    print(f"  Condition number: {condition:.3e}")
    if rank < len(PREDICTORS):
        print("  WARNING: predictor matrix is rank-deficient.")
        print("  HH_minus_HV is mathematically derived from HH_dB and HV_dB.")
        print("  This matters especially for ordinary multiple linear regression.")

    raw_km = float(raw_range["estimated_practical_range_km"].iloc[0])
    det_km = float(detrended_range["estimated_practical_range_km"].iloc[0])

    print("\nSpatial dependence:")
    print(f"  Raw AGBD estimated range:        {raw_km:.2f} km")
    print(f"  Detrended AGBD estimated range:  {det_km:.2f} km")
    print("  Do NOT assign 5 CV folds yet.")
    print("  We will review these ranges before selecting a defensible block size.")

    append_log([
        "Script: phase5_eda.py",
        f"Frozen modelling table: {INPUT_CSV}",
        f"Rows analyzed: {len(df):,}",
        f"Input SHA256: {sha256_file(INPUT_CSV)}",
        "EDA outputs: descriptive statistics, AGBD distribution/bin counts, Pearson/Spearman correlations, "
        "LOWESS scatterplots, predictor collinearity, temporal mismatch, spatial footprint map, nearest-neighbor distances.",
        f"Predictor matrix rank: {rank}/{len(PREDICTORS)}",
        f"Raw AGBD variogram estimated practical range: {raw_km:.3f} km",
        f"Detrended AGBD variogram estimated practical range: {det_km:.3f} km",
        "Decision: Phase 5 EDA complete; spatial block size/fold assignment deferred until review.",
        f"Python: {sys.version.split()[0]}; pandas={package_version('pandas')}; "
        f"numpy={package_version('numpy')}; scipy={package_version('scipy')}; "
        f"statsmodels={package_version('statsmodels')}; pyproj={package_version('pyproj')}; "
        f"matplotlib={package_version('matplotlib')}",
    ])

    print("\n" + "=" * 72)
    print("PHASE 5 EDA COMPLETE — REVIEW REQUIRED BEFORE SPATIAL CV")
    print("=" * 72)
    print(f"Tables:  {TABLE_DIR}")
    print(f"Figures: {FIG_DIR}")
    print("\nUpload these key outputs:")
    print(f" - {TABLE_DIR / 'EDA_summary_statistics.csv'}")
    print(f" - {TABLE_DIR / 'predictor_vs_AGBD_correlations.csv'}")
    print(f" - {TABLE_DIR / 'collinearity_diagnostics.csv'}")
    print(f" - {TABLE_DIR / 'spatial_autocorrelation_range_summary.csv'}")
    print(f" - {FIG_DIR / 'predictor_correlation_matrix.png'}")
    print(f" - {FIG_DIR / 'AGBD_raw_semivariogram.png'}")
    print(f" - {FIG_DIR / 'AGBD_detrended_semivariogram.png'}")
    print(f" - the four *_vs_AGBD_LOWESS.png figures")
    print("\nDo not begin Phase 6 spatial CV until these are reviewed.")
    input("\nPress Enter to close...")

if __name__ == "__main__":
    main()
