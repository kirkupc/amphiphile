# Design notes

Detailed technical discussion that supplements the README. Intended for reviewers who want to understand the *why* behind each choice.

## Data quality (ceiling reference)

1,852 compounds in the training set have ≥2 assay observations. The distribution of intra-compound std is:

- **Median: 0.21 log units** — the typical within-compound disagreement between labs.
- Mean: 0.58, P90: 1.55, max: 44 (ChEMBL has a few wildly inconsistent entries; median is the robust summary).

An RMSE near ~0.5 log units on scaffold test is near the noise floor of the reference data. A much higher RMSE indicates modelling slack; much lower would be suspicious (possibly memorisation via scaffold leakage). See `reports/noise_floor.png`.

## Conformal calibration

We use absolute residuals (`|y - ŷ|`) as the nonconformity score rather than the Mondrian approach of standardising by ensemble std (`|y - ŷ| / σ`). The Mondrian quantile inflated to ~10.6 because ensemble std is poorly correlated with actual error on OOD data (Spearman ~0.05) — compounds with small σ but large error produce extreme standardised residuals. With absolute residuals, the baseline's calibrated quantile is ~1.17 log units, giving constant-width 90% intervals of ŷ ± 1.17 — usable for decision-making.

The Chemprop model's saved artifacts still use the Mondrian approach (quantile ~5.16) because it was trained on Colab before the conformal fix. A recalibration script (`scripts/recalibrate_chemprop.py`) exists to update conformal/reliability artifacts without retraining, but requires a GPU/Colab environment (Chemprop's PyTorch inference hits SIGSEGV on macOS ARM due to NumPy copy-on-write interactions).

A further improvement would be to bin conformal calibration by Tanimoto similarity so that in-domain compounds get tighter bands than OOD ones.

## Reliability threshold calibration

The reliability flag is AND(ensemble_std ≤ threshold, nearest_tanimoto ≥ threshold). Thresholds are calibrated on val via grid search targeting ≥88% precision (fraction of flagged-reliable with |error| ≤ 1 log unit).

The Tanimoto grid uses absolute values (0.20, 0.25, ..., 0.60) rather than val-set quantiles. This is critical: val compounds are in-domain with high Tanimoto, so quantile-based thresholds produce strict thresholds (~0.55) that reject 99.5% of OOD data. Absolute values let the search find thresholds that transfer meaningfully to OOD.

Relaxing the precision target from 90% to 88%, combined with hyperparameter tuning (which improved ensemble calibration), expanded ExpansionRx coverage to 2,074 compounds (41.2%) with 86.1% precision and RMSE 0.70 on the reliable subset vs 0.82 overall.

## Feature engineering

Features: 31 RDKit descriptors + 6 ionisable-group counts + 2 Henderson-Hasselbalch pKa corrections + 2048-bit Morgan fingerprints = 2,087 dimensions.

The ionisable-group counts (basic amines, amidines, guanidines, quaternary N, acidic OH, aromatic basic N) tell the model *which* groups are present. The Henderson-Hasselbalch corrections (estimated logD shift and net charge at pH 7.4) tell it *how much* each shifts logD, using group-average literature pKa values. This is a partial mitigation — group-average pKa (e.g. basic amine ~9.5, guanidine ~12.5) ignores substituent effects that can shift pKa by 2+ units. Per-molecule pKa predictions via Dimorphite-DL or pkasolver would capture these shifts.

## Inference profiling

Across batch sizes on real drug-like compounds (ExpansionRx pool). Single-process, no multi-threading. M-series Mac laptop CPU.

| Batch | Total (s) | Throughput (mol/s) | Parse (s) | Featurise (s) | Model (s) | Uncertainty (s) | Peak RSS (MB) |
|------:|----------:|-------------------:|----------:|--------------:|----------:|----------------:|--------------:|
| 1     | 0.025     | 39.4               | 0.000     | 0.002         | 0.004     | 0.020           | 508           |
| 100   | 1.339     | 74.7               | 0.007     | 0.136         | 0.022     | 1.174           | 524           |
| 1,000 | 13.386    | 74.7               | 0.075     | 1.361         | 0.200     | 11.750          | 1,060         |
| 10,000| 133.799   | 74.7               | 0.737     | 13.883        | 1.566     | **117.613**     | 1,897         |

**The bottleneck is the applicability-domain Tanimoto scan.** At batch=10k, 88% of wall time is the AD step — a dense `(batch × n_train)` Tanimoto matrix (10,000 × 18,799 = 188M pairwise computations). Featurisation is 10%. Model forward is 1.2%. Throughput plateaus around **74 mol/s**.

### Optimisations for 100k+ molecules

1. **Packed-bit Tanimoto with popcount.** Pack fingerprints to `(n_train, 256)` uint64 and use hardware popcount — 8× memory reduction, 4–8× speedup on the dominant 88% stage.
2. **Approximate nearest-neighbour index.** FAISS IVF-PQ or HNSW over packed fingerprints: O(batch × log n_train) instead of O(batch × n_train). Additional 50–100× on top of packed-bit.
3. **Parallel featurisation.** `multiprocessing.Pool(cpu_count())` for the second-largest stage (10%).
4. **Other.** LRU cache keyed by InChIKey (10-30% hit rate on production workloads), ensemble distillation, ONNX export for Chemprop.

## Hyperparameter tuning

LightGBM hyperparameters are tuned via grid search over 27 combinations of `num_leaves` (63, 127, 255), `learning_rate` (0.02, 0.05, 0.1), and `min_child_samples` (5, 10, 20) — the three parameters with the largest impact on bias-variance tradeoff for gradient boosting. Best config is selected by val RMSE, then used for all k=5 ensemble members. Ensemble diversity comes from per-member bagging/feature subsampling (0.7/0.7), not from different hyperparameters.

The tuning grid is deliberately small (27 trials, ~5 min) because the brief values *understanding* over marginal performance. A larger Bayesian search (Optuna) could squeeze another 0.01–0.02 RMSE but would not change the modelling story.

## Design decisions

**Why this data pipeline.** OpenADMET publishes a curated ChEMBL 35 LogD aggregated parquet — smaller and more consistent than rolling our own `chembl_downloader` pull. We still honour the brief's "assemble from ChEMBL" by using ChEMBL-sourced data and doing our own canonicalisation/desalting/dedup on top.

**Why ExpansionRx as the benchmark.** The brief asks for "the OpenADMET logD benchmark." The public OpenADMET logD benchmark is the ExpansionRx challenge training set — 5,039 compounds measured in real drug discovery at Expansion Therapeutics. Held out from ChEMBL entirely; InChIKey-deduplicated against our training set at load time.

**Why scaffold split.** Random splits let tree models memorise chemical series via scaffold proximity. Scaffold splits (Bemis-Murcko) measure generalisation to new chemical matter. We train a separate random-split baseline and report both sets of numbers so reviewers can see the gap directly.

**Why deep ensemble + conformal + AD.** They fail independently. An ensemble can be confidently wrong when all members extrapolate the same way — the AD Tanimoto check catches that. Conformal gives calibrated prediction intervals. The data-quality audit documents the irreducible aleatoric floor separately.

**Why LightGBM baseline + Chemprop.** LightGBM on descriptors + Morgan is the pre-2020 standard; fast, interpretable. Chemprop D-MPNN is the reference graph model. The gap between them on ExpansionRx is the modelling story.

**Why salt stripping and explicit canonicalisation.** Half of ChEMBL's entries have counter-ions not part of the logD measurement. Stripping + canonicalising ensures identical molecules produce identical predictions — test coverage enforces this.

**Why a FastAPI service layer.** The brief calls it a service. An importable library is useful for Python callers; an HTTP endpoint is what "service" means to the rest of a stack. Both surfaces share one implementation.
