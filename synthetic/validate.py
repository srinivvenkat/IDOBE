"""Validate a synthetic dataset against the real one using IDOBE's own metrics.

"Statistically similar" is defined here exactly as in stats/stats_compute.ipynb:
Shannon entropy and permutation entropy of the normalized curve, plus the shape
moments (mean, median, variance, skewness, excess kurtosis) of the curve treated
as a probability distribution over weeks.

This script recomputes those metrics on both the real and the synthetic tables
and reports, per metric, a two-sample Kolmogorov-Smirnov test. A small KS
statistic (and large p-value) means the two distributions overlap -- i.e. the
synthetic data is statistically similar by the repo's own definition.

Example:
    python synthetic/validate.py \
        --real raw_data/outbreaks_disease_location.csv \
        --synthetic synthetic/output/synthetic_outbreaks.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

VALUE_COLS = [str(i) for i in range(60)]


def _series(row: np.ndarray) -> np.ndarray:
    y = row[~np.isnan(row)]
    return np.clip(y, 0.0, None)


def shannon_entropy(row: np.ndarray) -> float:
    y = _series(row)
    if y.sum() <= 0:
        return np.nan
    p = y / y.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def permutation_entropy(row: np.ndarray) -> float:
    import ordpy
    y = _series(row)
    if y.size < 4:
        return np.nan
    return float(ordpy.permutation_entropy(y, dx=3, taux=1, normalized=True))


def shape_moments(row: np.ndarray) -> dict[str, float]:
    """Moments of the curve as a discrete PDF over week index (see notebook)."""
    y = _series(row)
    if y.sum() <= 0:
        return {k: np.nan for k in
                ("mean", "median", "variance", "skewness", "excess_kurtosis")}
    x = np.arange(y.size, dtype=float)
    p = y / y.sum()
    mu = np.sum(x * p)
    order = np.argsort(x)
    cdf = np.cumsum(p[order])
    median = x[order][np.searchsorted(cdf, 0.5)]
    var = np.sum((x - mu) ** 2 * p)
    sigma = np.sqrt(var)
    if sigma < 1e-12:
        skew = kurt = np.nan
    else:
        skew = np.sum((x - mu) ** 3 * p) / sigma ** 3
        kurt = np.sum((x - mu) ** 4 * p) / sigma ** 4 - 3
    return {"mean": mu, "median": float(median), "variance": var,
            "skewness": skew, "excess_kurtosis": kurt}


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    values = df[VALUE_COLS].to_numpy(dtype=float)
    out = {"entropy": [], "PE": [], "mean": [], "median": [],
           "variance": [], "skewness": [], "excess_kurtosis": []}
    for row in values:
        out["entropy"].append(shannon_entropy(row))
        out["PE"].append(permutation_entropy(row))
        for k, v in shape_moments(row).items():
            out[k].append(v)
    return pd.DataFrame(out)


def main() -> None:
    from scipy import stats

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--real", default="raw_data/outbreaks_disease_location.csv")
    p.add_argument("--synthetic", default="synthetic/output/synthetic_outbreaks.csv")
    p.add_argument("--report", default=None,
                   help="Optional path to write the comparison table as CSV.")
    args = p.parse_args()

    real = compute_metrics(pd.read_csv(args.real))
    synth = compute_metrics(pd.read_csv(args.synthetic))

    rows = []
    for metric in real.columns:
        r = real[metric].dropna().to_numpy()
        s = synth[metric].dropna().to_numpy()
        ks, pval = stats.ks_2samp(r, s)
        rows.append({
            "metric": metric,
            "real_mean": r.mean(), "synth_mean": s.mean(),
            "real_std": r.std(), "synth_std": s.std(),
            "ks_stat": ks, "ks_pvalue": pval,
        })
    report = pd.DataFrame(rows)
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print(report.to_string(index=False))
    print("\nLower ks_stat = more similar. ks_stat < ~0.1 indicates close overlap.")

    if args.report:
        report.to_csv(args.report, index=False)
        print(f"\nWrote comparison to {args.report}")


if __name__ == "__main__":
    main()
