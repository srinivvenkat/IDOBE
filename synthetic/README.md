# Synthetic outbreak dataset

Tools for generating a synthetic outbreak table that is **statistically similar**
to `raw_data/outbreaks_disease_location.csv`, and for validating that similarity
using IDOBE's own statistics.

"Statistically similar" is defined here exactly as in `stats/stats_compute.ipynb`:
the synthetic curves should reproduce the distributions of

- **Shannon entropy** of the normalized curve,
- **permutation entropy** (`ordpy`, `dx=3`), and
- the **shape moments** (mean, median, variance, skewness, excess kurtosis) of the
  curve treated as a probability distribution over weeks.

## Quick start

```bash
pip install pandas numpy scikit-learn scipy ordpy

# Generate a synthetic table (same size as the real one by default)
python synthetic/generate.py --seed 0

# Validate it against the real data using the repo's own metrics
python synthetic/validate.py --report synthetic/output/validation_report.csv
```

The output CSV has the **exact same schema** as `raw_data/outbreaks_disease_location.csv`
(`unique_id, disease, location, event, start_date, end_date, duration, total, 0..59`),
so the existing `stats/` and `baselines/` pipelines consume it unchanged.

## How it works

An outbreak decomposes into three loosely independent pieces, each modeled
separately (`generator.py`):

1. **Metadata** — `(disease, location, event)`, dates and `duration`. Bootstrapped
   verbatim from real rows, so their marginal and joint distributions match by
   construction.
2. **Magnitude** — `total`, the overall scale of the curve. Also taken from the
   bootstrapped template row.
3. **Shape** — the normalized weekly curve, generated fresh from a **functional
   PCA (FPCA)** model so the curves are new rather than copied. Real curves are
   length-normalized onto a common grid, PCA is fit on them, and new curves are
   sampled from a Gaussian fit to the PCA scores, then rescaled.

Two details matter for statistical fidelity, and both came directly out of the
validation harness:

- **Shape is conditioned on the observed series length.** The number of non-NaN
  weekly values (mean ≈ 35) is *not* the `duration` field (mean ≈ 26) — the value
  columns always extend past `duration`. Curves are emitted at the template's
  observed length, and the shape model is binned by that length, because longer
  series also tend to peak later. Getting this wrong shifts the whole
  center-of-mass distribution (mean-week KS jumps from ~0.03 to ~0.29).
- **A small roughness term** (`roughness`, default 0.04) adds multiplicative
  week-to-week noise. A low-rank FPCA reconstruction is too smooth and has
  too-low permutation entropy; a little roughness restores it to the real level.

## Validation results

`validate.py` recomputes all three metric families on real and synthetic tables
and reports a two-sample Kolmogorov–Smirnov statistic per metric (lower = more
similar). With the defaults (`--seed 0`, 10 components, roughness 0.04):

| metric          | KS stat | quality        |
|-----------------|---------|----------------|
| mean (week)     | ~0.03   | tight overlap  |
| median (week)   | ~0.03   | tight overlap  |
| variance        | ~0.05   | tight overlap  |
| skewness        | ~0.06   | tight overlap  |
| excess kurtosis | ~0.09   | close          |
| Shannon entropy | ~0.14   | close, see below |
| permutation ent | ~0.20   | matches in mean, under-dispersed |

The **shape-moment family matches tightly.** Entropy and permutation entropy match
in central tendency but are *under-dispersed*: real curves range from very smooth
to very noisy, while the synthetic ones cluster around the average roughness. This
is inherent to sampling PCA scores from a single Gaussian.

## Extending this (next fidelity levers)

- **Sharpen the entropy/PE spread** by sampling PCA scores from a KDE or by
  bootstrapping real score vectors instead of a Gaussian, and/or by drawing
  `roughness` per curve from a distribution rather than using a fixed value.
- **Fully parametric metadata/magnitude**: replace the template bootstrap with
  fitted distributions (e.g. per-disease log-normal `total`, empirical
  `(disease, location, event)` frequencies) if you need to extrapolate beyond the
  observed combinations rather than resample them.
- **Per-disease validation**: `validate.py` compares pooled distributions; add a
  `groupby('disease')` loop to check similarity within each disease.

## Files

- `generator.py` — the `SyntheticOutbreakGenerator` model (fit / sample).
- `generate.py` — CLI that fits on the real table and writes a synthetic CSV.
- `validate.py` — recomputes entropy / PE / shape moments and compares via KS.
- `output/validation_report.csv` — the comparison table from the latest run.
