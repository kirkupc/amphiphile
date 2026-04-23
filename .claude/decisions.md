# Technical decision log

Engineering decisions made during development, what alternatives were considered, and why.

## Data

**OpenADMET curated parquet vs raw ChEMBL download.**
We use OpenADMET's pre-curated ChEMBL 35 LogD aggregated parquet (1.1 MB, pinned SHA) instead of pulling directly from ChEMBL via `chembl_downloader` (~4 GB). The curated version handles assay heterogeneity and unit normalization upstream. We still do our own canonicalization, desalting, and deduplication on top. We use `standard_value_median` as the target (robust to assay outliers) and filter to the chemically plausible range [-5.0, 10.0].

**ExpansionRx as the external benchmark.**
The public OpenADMET logD benchmark is the ExpansionRx challenge training set — 5,039 compounds measured at Expansion Therapeutics, held out from ChEMBL entirely. We pin the Hugging Face revision SHA so results are reproducible.

**InChIKey deduplication across datasets.**
We drop any training compound whose InChIKey matches an ExpansionRx compound. InChIKeys are recomputed after desalting (not trusted from the source) because salt stripping can change molecular identity.

**Canonicalization + salt stripping on both datasets.**
RDKit `MolFromSmiles` + `SaltRemover.StripMol(dontRemoveEverything=True)` applied uniformly. This ensures identical molecules in different input forms produce identical predictions.

## Splitting

**Scaffold split (Bemis-Murcko) over random split.**
Random splits let tree models memorize chemical series by scaffold proximity and report inflated numbers. Scaffold splits measure generalization to new chemical matter. We use greedy bucket packing: large scaffolds go to train, singletons are distributed to val/test. Random-split comparison is included to quantify the gap (0.103 RMSE difference confirms scaffold is harder).

## Featurization

**31 explicit RDKit descriptors + 6 ionisable group counts + 2 pKa corrections + 2048-bit Morgan fingerprints.**
Descriptors chosen for logD relevance. Listed explicitly in code (not `Descriptors._descList`) so the feature set is stable across RDKit versions. Morgan radius=2, 2048 bits gives ECFP4-equivalent coverage. Total dimension: 2,087.

The ionisable-group counts tell the model *which* pH-sensitive groups are present. The Henderson-Hasselbalch corrections tell it *how much* each shifts logD, using group-average literature pKa values. Known limitation: group-average pKa ignores substituent effects (can shift pKa by 2+ units). This is the primary blind spot identified in error analysis.

**Morgan stored as uint8 binary matrix.**
Dense `[0,1]` uint8 rather than sparse bit indices. Enables fast vectorized Tanimoto via popcount for the applicability-domain scan. The AD step is 88% of inference time at batch=10k — representation format matters here.

## Modelling

**LightGBM k=5 ensemble with grid-tuned hyperparameters.**
LightGBM on descriptors + Morgan is the pre-2020 standard for tabular QSAR. Hyperparameters tuned via 27-combo grid search (num_leaves, learning_rate, min_child_samples) using val RMSE. Best config: `num_leaves=63, lr=0.05, min_child_samples=5` — smaller trees than the default, suggesting the data benefits from more regularization.

Ensemble diversity: `feature_fraction=0.7`, `bagging_fraction=0.7`, `bagging_freq=1`. These were lowered from 0.9/0.9 after observing ensemble collapse — Spearman(std, |error|) was 0.006 on OOD at the higher values. At 0.7/0.7, scaffold Spearman improved to 0.24.

**Chemprop v2 D-MPNN as the stronger model.**
Chemprop is the reference graph model for molecular property prediction. It learns directly from SMILES (molecular graph) without explicit featurization. k=5 ensemble, 50 epochs, batch size 32. Results validated the choice: Chemprop RMSE 0.749 vs baseline 0.820 on ExpansionRx, with near-zero scaffold-to-OOD gap (0.747 vs 0.749).

**Safe collation for Chemprop dataloaders.**
Chemprop's `BatchMolGraph.__post_init__` uses `torch.from_numpy` (zero-copy). With NumPy >= 2.0 copy-on-write semantics on macOS ARM, this causes SIGSEGV. We provide `_safe_collate_batch` that deep-copies arrays before conversion.

## Uncertainty

**Three complementary signals, not one.**
They fail independently. An ensemble can be confidently wrong when all members extrapolate the same way — the Tanimoto check catches that. Conformal gives calibrated intervals.

1. **Ensemble std** — epistemic uncertainty. Fires when the ensemble internally disagrees.
2. **Conformal intervals** — absolute residuals (not Mondrian). Mondrian standardisation inflated the quantile to ~10.6 because ensemble std is poorly correlated with actual error on OOD data. Absolute residuals give constant-width ŷ ± 1.17 intervals.
3. **Applicability-domain Tanimoto** — max Morgan similarity of the query vs the training set. Chunked at 1024 molecules to bound memory.

**Reliability flag: AND of two thresholds.**
`reliable := (ensemble_std <= threshold) AND (nearest_tanimoto >= threshold)`. Thresholds calibrated on val via grid search targeting ≥88% precision (flagged-reliable within 1 log unit). Tanimoto grid uses absolute values (0.20–0.60), not val-set quantiles — critical because val compounds are in-domain with high Tanimoto, so quantiles produce thresholds that reject 99.5% of OOD data.

## Infrastructure

**FastAPI with lifespan-loaded model singleton.**
Model loaded once at startup, shared across requests. Max 10k SMILES per request. `LOGD_MODEL` env var selects baseline or chemprop.

**Docker: single-stage python:3.12-slim with uv.**
Separate layer for dependency caching. `uv sync --frozen` for reproducible installs. Models mounted as volume rather than baked into the image.

**CLI via Typer with subcommands.**
`prepare-data`, `train`, `predict`, `profile`, `error-analysis`, `data-quality`. Each does one thing. `train` includes `--tune` flag for hyperparameter search (enabled by default).

## Key hyperparameters

| Component | Parameter | Value | Rationale |
|---|---|---|---|
| Data | LogD range filter | [-5.0, 10.0] | Chemical domain bounds |
| Data | Target | standard_value_median | Robust to assay outliers |
| Split | Strategy | Scaffold (Bemis-Murcko) | Avoids series memorization |
| Features | Descriptors | 31 explicit | Stable, logD-relevant |
| Features | Morgan | r=2, 2048 bits | ECFP4-equivalent |
| Features | pKa corrections | 2 (logD shift, net charge) | Henderson-Hasselbalch at pH 7.4 |
| Baseline | Tuned: num_leaves | 63 | Grid search winner |
| Baseline | Tuned: learning_rate | 0.05 | Grid search winner |
| Baseline | Tuned: min_child_samples | 5 | Grid search winner |
| Baseline | feature/bagging fraction | 0.7 / 0.7 | Prevents ensemble collapse |
| Chemprop | Epochs | 50 | Reasonable for T4 GPU |
| Conformal | Alpha | 0.1 | 90% coverage target |
| Reliability | Target precision | 0.88 | 88% within 1 log unit |
| API | Max batch | 10,000 | Reasonable endpoint limit |
