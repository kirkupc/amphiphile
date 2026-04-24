# Design notes

Detailed technical discussion that supplements the README. Intended for reviewers who want to understand the *why* behind each choice.

## Data quality (ceiling reference)

1,852 compounds in the training set have ≥2 assay observations. The distribution of intra-compound std is:

- **Median: 0.21 log units** — the typical within-compound disagreement between labs.
- Mean: 0.58, P90: 1.55, max: 44 (ChEMBL has a few wildly inconsistent entries; median is the robust summary).

An RMSE near ~0.5 log units on scaffold test is near the noise floor of the reference data. A much higher RMSE indicates modelling slack; much lower would be suspicious (possibly memorisation via scaffold leakage). See `reports/noise_floor.png`.

## Conformal calibration

We use absolute residuals (`|y - ŷ|`) as the nonconformity score rather than the Mondrian approach of standardising by ensemble std (`|y - ŷ| / σ`). The Mondrian quantile inflated to ~10.6 on the baseline (and ~5.16 on Chemprop) because ensemble std is poorly correlated with actual error on OOD data (Spearman ~0.05) — compounds with small σ but large error produce extreme standardised residuals. With absolute residuals, the baseline's calibrated quantile is ~1.20 log units, giving constant-width 90% intervals of ŷ ± 1.20 — usable for decision-making.

**Known issue: Chemprop conformal quantile is broken.** The Chemprop model's saved artifacts still use the Mondrian approach (quantile ~5.16), producing intervals of ŷ ± 5.16 — effectively useless. This happened because the model was trained on Colab before the conformal fix was applied. A recalibration script (`scripts/recalibrate_chemprop.py`) exists to update conformal/reliability artifacts without retraining; the Colab notebook includes a recalibration cell. This requires a GPU environment — Chemprop's PyTorch inference hits SIGSEGV on macOS ARM due to NumPy copy-on-write interactions despite the safe-collation workaround. **Users should run the Colab recalibration before using Chemprop's uncertainty outputs.** The baseline model's conformal calibration (quantile 1.20, 86.1% OOD coverage) is correct and production-ready.

**OOD coverage check:** On ExpansionRx (genuinely OOD), the baseline's constant-width intervals achieve 86.1% empirical coverage vs the 90% target. The 3.9% gap is expected — conformal prediction guarantees marginal coverage on exchangeable data, which OOD compounds are not. The gap quantifies the distribution shift between ChEMBL training chemistry and ExpansionRx's RNA-targeting drug-discovery compounds, and is small enough that the intervals remain practically useful for ranking and triage.

A further improvement would be to bin conformal calibration by Tanimoto similarity so that in-domain compounds get tighter bands than OOD ones.

## Reliability threshold calibration

The reliability flag is AND(ensemble_std ≤ threshold, nearest_tanimoto ≥ threshold). Thresholds are calibrated on a held-out calibration portion (70%) of the val set (val_cal) that was not used for early stopping or hyperparameter tuning (val_select, 30%). The 30/70 split gives tuning enough data while maximising calibration set size. Grid search targets ≥88% precision (fraction of flagged-reliable with |error| ≤ 1 log unit).

The Tanimoto grid uses absolute values (0.25, 0.30, ..., 0.60) rather than val-set quantiles. This is critical: val compounds are in-domain with high Tanimoto, so quantile-based thresholds produce strict thresholds (~0.55) that reject 99.5% of OOD data. Absolute values let the search find thresholds that transfer meaningfully to OOD. The grid search prefers the **lowest** Tanimoto threshold meeting the precision target, then maximises coverage within that tier — this prevents the search from picking strict thresholds that happen to have high precision on in-domain val data but reject most OOD compounds.

With these design choices, ExpansionRx coverage is 3,706 compounds (73.5%) with 81.0% precision and RMSE 0.79 on the reliable subset vs 0.82 overall. The precision is below the 88% val_cal target, which is expected — conformal and reliability guarantees are calibrated on in-domain data and degrade on OOD compounds.

## Prediction bias

The baseline systematically overpredicts logD on ExpansionRx (mean signed error −0.21 log units, median −0.15). The bias is strongly range-dependent:

| logD range | n | Mean signed error | RMSE | % overpredicted |
|--:|--:|--:|--:|--:|
| [−5, 0) | 238 | −1.88 | 1.95 | 100% |
| [0, 1) | 553 | −1.12 | 1.22 | 99% |
| [1, 2) | 1,404 | −0.53 | 0.71 | 86% |
| [2, 3) | 1,600 | +0.01 | 0.43 | 49% |
| [3, 5) | 1,233 | +0.57 | 0.74 | 10% |

This is regression-to-the-mean: the ChEMBL training distribution centres around logD ~2, so extreme values get pulled toward the centre. The model performs best on the most populated range (2–3, RMSE 0.43). The strong bias at the tails suggests a systematic limitation of tree-based regression on this distribution — monotonic extrapolation would help, but LightGBM doesn't natively support it without constraints.

## Confidence curves

**Ensemble std vs actual error.** RMSE increases monotonically across ensemble std quintiles on ExpansionRx: Q1 (lowest std) 0.76 → Q5 (highest std) 0.91. The signal is directionally correct but weak — the ratio of worst-to-best quintile RMSE is only 1.2×. This confirms that ensemble std is a useful but weak discriminator on OOD data (consistent with Spearman ~0.05).

**Tanimoto similarity vs actual error.** RMSE decreases monotonically from 1.08 (Tanimoto [0.2, 0.3)) to 0.67 (Tanimoto [0.5, 0.6)). The ratio is 1.6× — substantially stronger than ensemble std. This validates the reliability flag design: the Tanimoto channel carries more information about prediction quality on OOD data than the ensemble channel.

Together, these curves demonstrate that the uncertainty signals, while individually imperfect, are directionally correct and complementary. See `reports/confidence_curves.png`.

## Feature importance

LightGBM gain-based feature importance (averaged across 5 ensemble members) reveals:

1. **MolLogP** dominates (78k gain) — unsurprising, as logD ≈ logP for neutral compounds.
2. **estimated_logd_shift** ranks #2 (15k) — the Henderson-Hasselbalch correction is the most important custom feature.
3. **n_acidic_oh** ranks #3 (13k) — carboxylic acids have large logD shifts from ionisation.
4. **TPSA** and **net_charge_pH7_4** rank #5 and #6 (10k each) — polarity and charge directly affect partitioning.
5. Several Morgan fingerprint bits appear in the top 30 (bits 1003, 463, 1838, 1308) — specific substructures carry logD-relevant information beyond global descriptors.

The fact that 3 of the top 6 features are our custom pH-correction features validates the feature engineering decision. Without them, the model would rely more heavily on MolLogP alone, which ignores ionisation effects.

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
