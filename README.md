# amphiphile

A logD prediction service. Given a list of SMILES, returns predicted logD7.4 with an uncertainty estimate and a reliability flag per molecule. Invalid SMILES are flagged, not crashed.

- **Training data:** OpenADMET-curated ChEMBL 35 LogD (pinned commit SHA)
- **External benchmark:** OpenADMET ExpansionRx challenge data on Hugging Face (5,039 real drug-discovery compounds, held out from ChEMBL)
- **Baseline model:** LightGBM ensemble (k=5) on RDKit 2D descriptors + Morgan fingerprints
- **Stronger model:** Chemprop v2 D-MPNN ensemble (optional, heavier)
- **Uncertainty:** deep ensemble std + Mondrian conformal + applicability-domain Tanimoto
- **Interfaces:** Python library, CLI (`logd`), and FastAPI service (`POST /predict`)

## Quickstart — service

```bash
docker build -t amphiphile .
docker run -p 8000:8000 -v $(pwd)/models:/app/models amphiphile
curl -X POST http://localhost:8000/predict \
     -H 'Content-Type: application/json' \
     -d '{"smiles": ["CCO", "c1ccccc1", "not_a_smiles"]}'
```

## Quickstart — library

```python
from logd import load_model, predict

model = load_model()                              # cold-start from models/
results = predict(["CCO", "c1ccccc1", "junk"], model=model)
for r in results:
    print(r)
# Prediction(smiles='CCO',          predicted_logd=-0.31, uncertainty=0.18, reliable=True,  error=None)
# Prediction(smiles='c1ccccc1',     predicted_logd=2.13,  uncertainty=0.22, reliable=True,  error=None)
# Prediction(smiles='junk',         predicted_logd=None,  uncertainty=None, reliable=False, error='invalid_smiles')
```

Or via CLI:

```bash
uv run logd predict --smiles CCO --smiles c1ccccc1
```

## Install from source

Requires Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
```

## Pre-trained model (skip retraining)

Reviewers who want to run inference without training first can download the baseline artifacts from the GitHub release:

```bash
mkdir -p models && cd models
gh release download v0.1.0-baseline -R kirkupc/amphiphile
cd ..
uv run logd predict --smiles CCO --smiles c1ccccc1
```

Artifacts: `baseline.joblib` (94 MB, k=5 LightGBM ensemble) and `reliability.joblib` (37 MB, conformal + AD). Trained on pinned OpenADMET-curated ChEMBL 35 LogD.

## Reproduce the results from scratch

```bash
uv run logd prepare-data     # fetch + cache ChEMBL LogD + ExpansionRx (pinned SHAs)
uv run logd train            # baseline ensemble, writes models/ + reports/metrics.json
uv run logd data-quality     # intra-compound std → reports/noise_floor.{json,png}
uv run logd error-analysis   # worst-10 compounds → reports/error_analysis/
uv run logd profile          # batch 1/100/1k/10k → reports/profiling.json
```

ChEMBL version, OpenADMET catalog SHA, Hugging Face dataset SHA, and all seeds are pinned in code. The whole pipeline finishes in about 10 minutes on a laptop (most of it featurisation + the 5-member ensemble training; the external-eval step is a few seconds).

## Run tests

```bash
make check        # ruff + mypy + pytest -m 'not slow'   (under 5 s)
make test         # adds slow tests (small Chemprop training)
```

## Results

### Scaffold test (ChEMBL 35 held-out, 2,351 compounds)

| Model | RMSE | MAE | Pearson r | Spearman(std, \|err\|) |
|---|--:|--:|--:|--:|
| Baseline (LightGBM ensemble, k=5) | **0.823** | 0.564 | 0.859 | 0.236 |
| Chemprop D-MPNN (k=5) | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

### ExpansionRx external test (real drug-discovery data, 5,039 compounds, zero overlap with training)

| Model | RMSE | MAE | Pearson r | Spearman(std, \|err\|) |
|---|--:|--:|--:|--:|
| Baseline | **0.855** | 0.655 | 0.744 | 0.006 |
| Chemprop | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

**Reading the numbers.** Baseline generalises well — RMSE only bumps 0.03 log units from scaffold to OOD, Pearson r drops from 0.86 to 0.74 (still strong). That's 23,499 ChEMBL compounds predicting 5,039 unseen real-world drug-discovery compounds. Solid baseline.

**The uncertainty story is mixed.** Spearman between ensemble-std and absolute error is 0.24 on scaffold (modest) and effectively zero on OOD. This is a known failure mode of seed-ensembled tree boosters: the different seeds converge to very similar predictors, the ensemble std collapses, and the signal stops tracking error on distribution shift. The conformal quantile blows up to 11.6× the ensemble std as a consequence (a symptom of underestimated σ, not of the conformal method itself). The Tanimoto applicability-domain channel is the more reliable OOD signal for this model family — see the error analysis below.

### Data quality (ceiling reference)

1,852 compounds in the training set have ≥2 assay observations. The distribution of intra-compound std is:

- **Median: 0.21 log units** — the typical within-compound disagreement between labs.
- Mean: 0.58, P90: 1.55, max: 44 (ChEMBL has a few wildly inconsistent entries; median is the robust summary).

Interpretation: an RMSE near ~0.5 log units on scaffold test is near the noise floor of the reference data. A much higher RMSE indicates modelling slack; much lower would be suspicious (possibly memorisation via scaffold leakage).

See `reports/noise_floor.png`.

## Uncertainty evaluation

Three complementary signals, each serving a different purpose:

1. **Ensemble std** — epistemic uncertainty from 5 differently-seeded models. Fires when the ensemble internally disagrees.
2. **Mondrian conformal intervals** — distribution-free 90% intervals, calibrated on val. Scales with ensemble std so uncertain predictions get wider bands.
3. **Applicability-domain Tanimoto** — max Morgan similarity of the query vs the training set. Low similarity → extrapolation beyond training distribution, independent of the ensemble's agreement.

**Reliability flag** is the AND of "ensemble_std ≤ threshold" and "nearest_tanimoto ≥ threshold", with thresholds calibrated on val for target precision (≥90% of flagged-reliable predictions within 1 log unit).

**How we show it's meaningful:** Spearman rank correlation between ensemble_std and |prediction error| on the external test — reported in the table above. Plus the confidence curve (RMSE vs coverage as we drop the most-uncertain predictions) in `reports/`.

## Error analysis — worst 10 predictions on ExpansionRx

See `reports/error_analysis/report.md` for structures + per-compound rationale.

**Summary: 10 loud failures, 0 silent failures.** Every one of the worst-10 predictions would have been flagged unreliable by the system — either the ensemble std exceeded its threshold, or the Tanimoto AD said the compound was too far from training chemistry, or both. The reliability flag caught 100% of the worst cases in this sample.

That result answers the narrative question "does the reliability flag work on real data?": yes, on this external benchmark. More importantly, it means the OOD uncertainty signal that *is* working (Tanimoto AD) compensates for the one that isn't (ensemble std). The reliability-AND is more robust than either channel alone.

## Inference profiling

Across batch sizes on real drug-like compounds (ExpansionRx pool, held-out from training). Single-process, no multi-threading. M-series Mac laptop CPU.

| Batch | Total (s) | Throughput (mol/s) | Parse (s) | Featurise (s) | Model (s) | Uncertainty (s) | Peak RSS (MB) |
|------:|----------:|-------------------:|----------:|--------------:|----------:|----------------:|--------------:|
| 1     | 0.025     | 40.1               | 0.000     | 0.002         | 0.003     | 0.020           | 522           |
| 100   | 1.355     | 73.8               | 0.008     | 0.141         | 0.018     | 1.187           | 539           |
| 1,000 | 13.581    | 73.6               | 0.078     | 1.394         | 0.140     | 11.969          | 1,078         |
| 10,000| 133.852   | 74.7               | 0.797     | 14.459        | 1.116     | **117.481**     | 1,912         |

**The bottleneck is the applicability-domain Tanimoto scan, not featurisation.** At batch=10k, 88% of wall time is the AD step — a dense `(batch × n_train)` Tanimoto matrix (10,000 × 18,799 = 188M pairwise computations with int32 upcast from uint8 fingerprints). Featurisation is 11%. Model forward is 0.8%.

Throughput plateaus around **74 mol/s** once batching overhead is amortised; the scan is essentially linear in batch size.

### Two optimisations for 100k+ molecules per request

**1. Index-based applicability-domain search (#1 optimisation).** The current AD does an O(batch × n_train) dense Tanimoto matmul every call. At 100k queries × 19k training fps that's ~1.9B pairwise comparisons. A FAISS IVF-PQ index over the packed fingerprints (or an HNSW approximate-NN index over Morgan-count-vectors) would drop this to O(batch × log n_train). Expect 50–100× speedup on the AD step, which is 88% of wall time. Additional win: the dense matmul also upcasts uint8 to int64 internally in numpy; a packed-bit popcount implementation (or a C extension) would give another 4–8× on its own, independent of index use. Chosen first because it attacks the dominant stage.

**2. Parallel featurisation via `multiprocessing.Pool`.** The second-largest stage at 11% of time. RDKit descriptor computation is CPU-bound Python and embarrassingly parallel. A `Pool(cpu_count())` wrapper in `featurise_batch` should cut this ~`ncores`× with no accuracy risk. On an 8-core machine this roughly halves the remaining 12% after optimisation #1, so together they'd move the overall latency floor an order of magnitude.

Other optimisations we'd document but not ship first: LRU cache keyed by canonical InChIKey (production workloads repeat — typically 10-30% hit rate), ensemble distillation into a single booster (model forward is only ~1% of time today, so ordering matters less than it would in a compute-bound setting), ONNX export for Chemprop if we deploy that variant.

## Design decisions

**Why this data pipeline.** OpenADMET publishes a curated ChEMBL 35 LogD aggregated parquet (median + std per compound across assays) — smaller and more consistent than rolling our own `chembl_downloader` pull. We still honour the brief's "assemble from ChEMBL" by using ChEMBL-sourced data and doing our own canonicalisation/desalting/dedup on top; OpenADMET's curation is a quality-control layer, not a replacement.

**Why ExpansionRx as the benchmark.** The brief asks for "the OpenADMET logD benchmark." The public OpenADMET logD benchmark is the ExpansionRx challenge training set — 5,039 compounds measured in real drug discovery at Expansion Therapeutics. Held out from ChEMBL entirely; InChIKey-deduplicated against our training set at load time.

**Why scaffold split.** Random splits let tree models memorise chemical series via scaffold proximity and report inflated numbers. Scaffold splits (Bemis-Murcko) measure generalisation to new chemical matter, which is what a deployed model is held to. We report both random and scaffold numbers so the gap is visible.

**Why deep ensemble + conformal + AD.** They fail independently. An ensemble can be confidently wrong when all members extrapolate the same way — the AD Tanimoto check catches that. Conformal gives calibrated intervals that compose with the ensemble-std scale. Tabular trees don't naturally produce aleatoric uncertainty, so we stick to epistemic via ensembling + the conformal shell; the data-quality audit documents the irreducible aleatoric floor separately.

**Why LightGBM baseline + Chemprop as stronger.** LightGBM on descriptors + Morgan is the pre-2020 standard; fast, interpretable, hard to beat on tabular logD. Chemprop D-MPNN is the reference published graph model, well-documented, understood by reviewers. The gap between them on ExpansionRx is the modelling story.

**Why salt stripping and explicit canonicalisation.** Half of ChEMBL's entries have counter-ions that are not part of the logD measurement. We strip salts via RDKit's `SaltRemover` and canonicalise explicitly, so identical molecules in different input forms produce identical predictions — test coverage enforces this.

**Why a FastAPI service layer.** The brief calls it a service. An importable library is useful for internal Python callers; an HTTP endpoint is what "service" means to the rest of a stack. Both surfaces share one implementation.

## What I'd do with more time

- **Aleatoric uncertainty** — an MVE (mean + variance) head in Chemprop to decompose total uncertainty into epistemic (ensemble disagreement) and aleatoric (data noise). Would let us tell "this molecule is hard" from "this measurement is noisy."
- **Harder splits** — Butina / Taylor-Butina cluster split at a 0.4 Tanimoto threshold for a tighter generalisation test. Report the gap vs scaffold.
- **Ensemble distillation** — actually implement it (listed above as hypothetical).
- **Pretrained molecular representations** — MolFormer / ChemBERTa as a comparison point. Not obvious they'd win at this scale, but worth benchmarking.
- **A retraining / monitoring loop** — production logD services drift as chemistry distributions change. A weekly revalidation against a held-out slice, with an alert if RMSE shifts > 0.1 log units.
- **Data cleaning beyond OpenADMET** — filter out ChEMBL entries where `standard_value_std > 2.0` (bad assay noise). Would cost a small amount of training data for more reliable labels.
- **ONNX export + hosted inference** — drops cold start and enables batched GPU inference when the service is called heavily.

## A note on AI tools

This implementation was built with heavy use of Claude (Anthropic's model, via the Claude Code CLI). I used it for:

- Scaffolding the package structure and typed interfaces from a hand-written plan (`PLAN.md`, kept local).
- Writing the unit tests against the contracts I specified.
- Drafting the FastAPI layer, Dockerfile, Makefile, and error-analysis/profiler scripts after I decided what they should do.
- Catching at least one bug I would have shipped silently (a `pytorch_lightning` vs `lightning.pytorch` namespace clash in the Chemprop wrapper that broke trainer/model compatibility).

I did not use it to:

- Decide the modelling approach (featurisation, split strategy, uncertainty method) — those came from reading the problem and the brief. The model picked Chemprop only after I named it as the target.
- Audit the data sources — I verified the OpenADMET catalog and Hugging Face dataset URLs myself (an earlier AI-guessed URL was wrong; REVIEW.md has the trace).
- Write the design-decisions section above — the *why* is mine.

Every decision in this repo is one I can defend.
