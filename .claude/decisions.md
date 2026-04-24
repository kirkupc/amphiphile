# Technical decision log

Engineering decisions made during development, what alternatives were considered, what was chosen, and why.

## Data

**OpenADMET curated parquet vs raw ChEMBL download.**
We use OpenADMET's pre-curated ChEMBL 35 LogD aggregated parquet (1.1 MB, pinned SHA) instead of pulling directly from ChEMBL via `chembl_downloader` (~4 GB). The curated version handles assay heterogeneity and unit normalization upstream. We still do our own canonicalization, desalting, and deduplication on top — OpenADMET is a quality-control layer, not a replacement. We use `standard_value_median` as the target (robust to assay outliers) and filter to the chemically plausible range [-5.0, 10.0].

**ExpansionRx as the external benchmark.**
The public OpenADMET logD benchmark is the ExpansionRx challenge training set — 5,039 compounds measured at Expansion Therapeutics, held out from ChEMBL entirely. We pin the Hugging Face revision SHA so results are reproducible. The alternative was evaluating only on ChEMBL scaffold-test, which wouldn't test true out-of-distribution generalization.

**InChIKey deduplication across datasets.**
We drop any training compound whose InChIKey matches an ExpansionRx compound. InChIKeys are recomputed after desalting (not trusted from the source) because salt stripping can change molecular identity. The alternative — allowing overlap — would leak test information into training.

**Canonicalization + salt stripping on both datasets.**
RDKit `MolFromSmiles` + `SaltRemover.StripMol(dontRemoveEverything=True)` applied uniformly. This ensures identical molecules in different input forms produce identical predictions. The alternative — trusting pre-canonicalized SMILES from OpenADMET — risks inconsistency when user inputs differ from the training format.

## Splitting

**Scaffold split (Bemis-Murcko) over random split.**
Random splits let tree models memorize chemical series by scaffold proximity and report inflated numbers. Scaffold splits measure generalization to new chemical matter. We use greedy bucket packing: large scaffolds go to train, singletons are distributed to val/test. This minimizes the risk of a huge scaffold landing entirely in test and skewing metrics. Split fractions: 80/10/10.

Random-split comparison included to quantify the gap directly — the 0.103 RMSE difference (0.687 random vs 0.790 scaffold) confirms scaffold is the harder evaluation.

## Featurization

**31 explicit RDKit descriptors + 6 ionisable group counts + 2 pKa corrections + 2048-bit Morgan fingerprints.**
Descriptors chosen for logD relevance: MolWt, TPSA, LogP, HBD/HBA, aromaticity, rotatable bonds, Balaban J, Chi/Kappa topological indices, QED. Listed explicitly in code (not `Descriptors._descList`) so the feature set is stable across RDKit versions. Morgan radius=2, 2048 bits gives ECFP4-equivalent coverage. Total dimension: 2,087.

Ionisable-group counts (basic amines, amidines, guanidines, quaternary N, acidic OH, pyridine N) tell the model *which* groups are present. Henderson-Hasselbalch corrections (estimated logD shift and net charge at pH 7.4) tell it *how much* each shifts logD, using group-average literature pKa values. Known limitation: group-average pKa ignores substituent effects (can shift pKa by 2+ units). Per-molecule pKa predictions via Dimorphite-DL or pkasolver would capture these shifts.

Alternatives considered: all available RDKit descriptors (~200, many NaN-prone or redundant), ECFP indices only (loses physicochemical signal), learned representations (reserved for Chemprop). The dual scheme gives tree models both global molecular properties and local substructure information.

**Morgan stored as uint8 binary matrix.**
Dense `[0,1]` uint8 rather than sparse bit indices. Enables fast vectorized Tanimoto via popcount for the applicability-domain scan. The AD step is 88% of inference time at batch=10k — representation format matters here.

## Modelling

**LightGBM k=5 seed ensemble as baseline.**
LightGBM on descriptors + Morgan is the pre-2020 standard for tabular QSAR; fast, interpretable, hard to beat on tabular logD. k=5 with different seeds gives an epistemic uncertainty signal via ensemble disagreement. 2,000 boost rounds with early stopping (patience=100) on validation RMSE.

Hyperparameters tuned via grid search over 27 combinations (num_leaves × learning_rate × min_child_samples). Best config: `num_leaves=63, lr=0.05, min_child_samples=5`. Smaller trees than the default (127), suggesting the data benefits from more regularization.

Key diversity parameters: `feature_fraction=0.7`, `bagging_fraction=0.7`, `bagging_freq=1`. These were lowered from 0.9/0.9 after observing ensemble collapse — Spearman(std, |error|) was 0.006 on OOD at the higher values. At 0.7/0.7, scaffold Spearman improved to 0.24. Still modest, but the Tanimoto AD channel compensates.

**Chemprop v2 D-MPNN as the stronger model.**
Chemprop is the reference published graph model for molecular property prediction, well-documented and understood by reviewers. It learns directly from SMILES (molecular graph) without explicit featurization. k=5 ensemble, 50 epochs, batch size 32, gradient clipping 1.0, BondMessagePassing + MeanAggregation + RegressionFFN with batch norm. Training designed for Colab GPU (T4); ~15 min per member.

Results validated the choice: Chemprop RMSE 0.749 vs baseline 0.824 on ExpansionRx, with near-zero scaffold-to-OOD gap (0.747 vs 0.749), suggesting the GNN is less prone to scaffold memorization than the tree ensemble.

**Safe collation for Chemprop dataloaders.**
Chemprop's `BatchMolGraph.__post_init__` uses `torch.from_numpy` (zero-copy). With NumPy >= 2.0 copy-on-write semantics on macOS ARM, this causes use-after-free (SIGSEGV). We provide `_safe_collate_batch` that deep-copies every MolGraph array with `np.array(..., copy=True)` then converts via `torch.tensor` (which always copies). Used for both training (Colab) and inference (local).

## Uncertainty

**Three complementary signals, not one.**
They fail independently. An ensemble can be confidently wrong when all members extrapolate the same way — the Tanimoto check catches that. Conformal gives calibrated intervals. Each serves a different purpose:

1. **Ensemble std** — epistemic uncertainty. Fires when the ensemble internally disagrees.
2. **Conformal intervals** — absolute residuals, calibrated on val_cal. Initially used Mondrian approach (standardised by ensemble std), but the quantile inflated to ~10.6 because ensemble std is poorly correlated with actual error on OOD data (Spearman ~0.05). Switched to absolute residuals: constant-width intervals ŷ ± 1.20.
3. **Applicability-domain Tanimoto** — max Morgan similarity of the query vs the training set. Independent of the ensemble's agreement. Chunked at 1024 molecules to bound memory.

**Val split: val_select (30%) + val_cal (70%).**
Val set split 30/70 into val_select (early stopping + hyperparameter tuning) and val_cal (conformal calibration + threshold calibration). The asymmetric split gives calibration more data since tuning needs fewer points. This separation ensures conformal coverage guarantees are not violated by model selection.

**Reliability flag: AND of two thresholds.**
`reliable := (ensemble_std <= threshold) AND (nearest_tanimoto >= threshold)`. Thresholds calibrated on val_cal via grid search (7 std quantiles × 8 Tanimoto thresholds) targeting ≥88% precision (flagged-reliable predictions within 1 log unit of truth).

Critical design choice: the Tanimoto grid uses absolute values (0.25, 0.30, ..., 0.60), not val-set quantiles. Val compounds are in-domain with high Tanimoto, so quantile-based thresholds produced strict values (~0.55) that rejected 99.5% of OOD data. Absolute values let the search find thresholds that transfer meaningfully to OOD.

The grid search prefers the **lowest** Tanimoto threshold meeting the precision target, then maximises coverage within that tier. Without this preference, the search picks strict thresholds (e.g. 0.45) that have high precision on in-domain val but reject 90%+ of OOD compounds. With it, the search finds Tanimoto 0.25 — covering 73.5% of ExpansionRx with 81% precision (below the 88% val target, as expected for OOD).

The AND logic is the key — either channel can veto. This is why the reliability flag catches the worst predictions on ExpansionRx even though ensemble std alone is nearly uncorrelated with error on OOD.

**Conformal OOD coverage check.**
Computed empirical conformal coverage on ExpansionRx (genuinely OOD): 86.1% vs 90% target. The 3.9% gap quantifies distribution shift — conformal guarantees marginal coverage on exchangeable data, which OOD compounds are not. Small enough that constant-width intervals remain practically useful.

## Infrastructure

**FastAPI with lifespan-loaded model singleton.**
Model loaded once at startup (~1s), shared across requests. Max 10k SMILES per request. `LOGD_MODEL` env var selects baseline or chemprop at startup. Alternative was per-request loading (kills latency) or async model loading (adds complexity for no gain at this scale).

**Docker: single-stage python:3.12-slim with uv.**
Separate layer for dependency caching (pyproject.toml + uv.lock copied first, then source). `uv sync --frozen` for reproducible installs. Models mounted as volume rather than baked into the image — keeps the image small, artifacts can be swapped without rebuild. Healthcheck wired to `/health`.

**CLI via Typer with subcommands.**
`prepare-data`, `train`, `predict`, `profile`, `error-analysis`, `data-quality`. Each subcommand does one thing. `predict` accepts `--smiles` (repeatable) or `--input-file`, outputs JSON lines. `train` accepts `--model baseline|chemprop` and `--tune` for hyperparameter search. Profiler and error analysis are separate scripts imported as subcommands rather than inline — keeps cli.py thin.

**Code quality: ruff + mypy strict + pytest with markers.**
`make check` runs lint + typecheck + fast tests in under 5 seconds. Slow tests (Chemprop training) marked `@pytest.mark.slow` and excluded by default. 49 fast tests, 8 slow tests.

## Model diagnostics

**Prediction bias analysis.**
The baseline systematically overpredicts logD on ExpansionRx (mean signed error −0.21). Bias is range-dependent: 100% overprediction for logD < 0, near-zero at logD 2–3, underprediction above 3. This is regression-to-the-mean — training distribution centres around logD ~2.

**Confidence curves.**
RMSE increases monotonically with ensemble std quintile (0.76 → 0.91, ratio 1.2×) and decreases with Tanimoto similarity (1.08 → 0.67, ratio 1.6×). Tanimoto is the stronger discriminator on OOD data, validating its role as the binding constraint in the reliability flag.

**Feature importance.**
MolLogP dominates (78k gain). Henderson-Hasselbalch estimated_logd_shift is #2 (15k), n_acidic_oh #3 (13k), net_charge_pH7_4 #6 (10k). Three of the top 6 features are custom pH-correction features, validating the feature engineering decision.

## Profiling

**Per-stage breakdown on realistic compounds.**
Profiling uses ExpansionRx compounds (real drug-like, OOD) across batch sizes [1, 100, 1k, 10k]. Four stages timed independently: parse/canonicalize, featurize, model forward, uncertainty/flag. This revealed the bottleneck is the AD Tanimoto scan (88% of time at 10k), not featurization (10%) or model forward (1.2%).

**Two concrete optimizations for 100k+.**
1. Packed-bit Tanimoto with popcount — pack fingerprints to uint64, use hardware popcount. Expected 4–8× on the dominant 88% stage.
2. FAISS IVF-PQ or HNSW index for approximate nearest-neighbor — drops O(batch × n_train) to O(batch × log n_train). Expected 50–100× additional.

## Key hyperparameters

| Component | Parameter | Value | Rationale |
|---|---|---|---|
| Data | LogD range filter | [-5.0, 10.0] | Chemical domain bounds |
| Data | Target | standard_value_median | Robust to assay outliers |
| Split | Strategy | Scaffold (Bemis-Murcko) | Avoids series memorization |
| Split | Fractions | 80/10/10 | Standard |
| Features | Descriptors | 31 explicit | Stable, logD-relevant |
| Features | Morgan | r=2, 2048 bits | ECFP4-equivalent |
| Features | pKa corrections | 2 (logD shift, net charge) | Henderson-Hasselbalch at pH 7.4 |
| Baseline | Tuned: num_leaves | 63 | Grid search winner |
| Baseline | Tuned: learning_rate | 0.05 | Grid search winner |
| Baseline | Tuned: min_child_samples | 5 | Grid search winner |
| Baseline | feature/bagging fraction | 0.7 / 0.7 | Prevents ensemble collapse |
| Chemprop | k | 5 | Matches baseline |
| Chemprop | Epochs | 50 | Reasonable for T4 GPU |
| Chemprop | Batch size | 32 | Stability + safe collation |
| Conformal | Alpha | 0.1 | 90% coverage target |
| AD | Chunk size | 1024 | Memory bounded |
| Reliability | Target precision | 0.88 | 88% within 1 log unit |
| API | Max batch | 10,000 | Reasonable endpoint limit |
