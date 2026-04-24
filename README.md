# amphiphile

A logD prediction service. Given a list of SMILES strings, it returns a predicted logD7.4 value, an uncertainty estimate, and a reliability flag for each molecule. Invalid SMILES are handled gracefully.

The model is a k=5 LightGBM ensemble trained on 18,799 compounds (scaffold-split from OpenADMET-curated ChEMBL 35), evaluated against 5,039 held-out drug-discovery compounds from the OpenADMET ExpansionRx challenge (zero training overlap, verified by InChIKey deduplication). Features: 31 RDKit descriptors, 6 ionisable-group counts, 2 Henderson-Hasselbalch pKa corrections, and 2048-bit Morgan fingerprints. A second model (Chemprop v2 D-MPNN ensemble, GPU-trained) is included for comparison. See [DESIGN.md](DESIGN.md) for detailed design rationale — data quality, conformal calibration, threshold tuning, feature engineering, profiling analysis, and architectural decisions.

The service is exposed as a Python library, a CLI (`logd`), and a FastAPI HTTP endpoint (`POST /predict`).

## Getting started

Requires Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/kirkupc/amphiphile.git && cd amphiphile
uv sync --extra dev
```

### Try it without retraining

Pre-trained artifacts are attached to the GitHub release. Download them and run inference immediately:

```bash
mkdir -p models && cd models
gh release download v0.1.0-baseline -R kirkupc/amphiphile
cd ..
uv run logd predict --smiles CCO --smiles c1ccccc1
```

Or as a Python library:

```python
from logd import load_model, predict

model = load_model()
results = predict(["CCO", "c1ccccc1", "not_a_smiles"], model=model)
for r in results:
    print(r)
# Prediction(smiles='CCO',          predicted_logd=-0.31, uncertainty=0.18, reliable=True,  error=None)
# Prediction(smiles='c1ccccc1',     predicted_logd=2.13,  uncertainty=0.22, reliable=True,  error=None)
# Prediction(smiles='not_a_smiles', predicted_logd=None,  uncertainty=None, reliable=False, error='invalid_smiles')
```

Or as a Docker service:

```bash
docker build -t amphiphile .
docker run -p 8000:8000 -v $(pwd)/models:/app/models amphiphile
curl -X POST http://localhost:8000/predict \
     -H 'Content-Type: application/json' \
     -d '{"smiles": ["CCO", "c1ccccc1", "not_a_smiles"]}'
```

### Reproduce everything from scratch

All data sources, splits, and seeds are pinned. The full pipeline takes about 10 minutes on an M-series Mac.

```bash
uv run logd prepare-data     # fetch + cache ChEMBL LogD + ExpansionRx (pinned SHAs)
uv run logd train            # baseline ensemble → models/ + reports/metrics.json
uv run logd data-quality     # intra-compound std → reports/noise_floor.{json,png}
uv run logd error-analysis   # worst-10 compounds → reports/error_analysis/
uv run logd diagnostics      # bias, confidence curves, feature importance → reports/
uv run logd profile          # batch 1/100/1k/10k → reports/profiling.json
```

### Train the Chemprop D-MPNN ensemble (GPU)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kirkupc/amphiphile/blob/main/notebooks/train_chemprop_colab.ipynb) ~20 min on a free T4. **Important:** the notebook includes a recalibration cell that fixes Chemprop's conformal intervals (the shipped artifacts use the pre-fix Mondrian approach with quantile ~5.16, which produces unusable intervals). Run recalibration before using Chemprop's uncertainty outputs.

### Run tests

```bash
make check        # ruff + mypy + pytest -m 'not slow'   (under 5 s)
make test         # adds slow + integration tests
```

## Results

### Scaffold test (ChEMBL 35 held-out, 2,351 compounds)

| Model | RMSE | MAE | Pearson r | Spearman(std, \|err\|) |
|---|--:|--:|--:|--:|
| Baseline (LightGBM ensemble, k=5) | 0.791 | 0.541 | 0.871 | 0.238 |
| Chemprop D-MPNN (k=5) | **0.747** | **0.499** | **0.885** | 0.220 |

### Random-split test (same data, random 80/10/10 split)

| Model | RMSE | MAE | Pearson r | Spearman(std, \|err\|) |
|---|--:|--:|--:|--:|
| Baseline | 0.687 | 0.445 | 0.900 | 0.275 |

The 0.104 RMSE gap (0.687 → 0.791) between random and scaffold split confirms that scaffold evaluation is substantially harder — and more honest about real-world generalisation.

### ExpansionRx external test (5,039 drug-discovery compounds, zero training overlap)

| Model | RMSE | MAE | Pearson r | Spearman(std, \|err\|) |
|---|--:|--:|--:|--:|
| Baseline | 0.824 | 0.640 | 0.771 | 0.048 |
| Chemprop | **0.749** | **0.590** | **0.784** | 0.012 |

Chemprop drops OOD RMSE from 0.824 to 0.749 with a negligible scaffold-to-OOD gap (0.747 → 0.749), suggesting the D-MPNN generalises better than hand-crafted features. The data noise floor is ~0.21 log units (median intra-compound std across ChEMBL assays). Baseline hyperparameters were tuned via grid search (27 combinations of num_leaves, learning_rate, min_child_samples); see [DESIGN.md](DESIGN.md).

### Prediction bias

The baseline systematically overpredicts logD (mean signed error **−0.21 log units**). The bias is range-dependent: 100% overprediction for logD < 0 (mean error −1.87), near-zero bias at logD 2–3, and underprediction above logD 3 (mean error +0.57). This is consistent with regression-to-the-mean behaviour: the training distribution is centred around logD ~2, so the model pulls extreme values toward the centre. See `reports/diagnostics.json` for full breakdown.

### Confidence curves

RMSE increases monotonically with ensemble std quintile (0.76 → 0.91) and decreases with Tanimoto similarity (1.08 at <0.3 → 0.67 at 0.5–0.6). The Tanimoto signal is stronger than ensemble std for discriminating prediction quality on OOD data, which explains why Tanimoto is the binding constraint in the reliability flag. See `reports/confidence_curves.png`.

### Feature importance

Top LightGBM features by gain: **MolLogP** (78k, dominant), **estimated_logd_shift** (15k, Henderson-Hasselbalch correction), **n_acidic_oh** (13k), **BertzCT** (12k), **TPSA** (10k), **net_charge_pH7_4** (10k). The pH-correction features rank #2 and #6, validating the decision to include them. Full top-30 in `reports/diagnostics.json`.

## Uncertainty and reliability

Three signals: **ensemble std** (epistemic), **conformal intervals** (constant-width, 90% coverage target), and **Tanimoto AD** (nearest-neighbour similarity to training set). The `uncertainty` field in predictions is ensemble std in log units.

Conformal and reliability thresholds are calibrated on a held-out portion of the val set (val_cal, 70%) that was not used for early stopping or hyperparameter tuning (val_select, 30%). This ensures the conformal coverage guarantee is not violated by model selection.

**Reliability flag** = AND(ensemble_std ≤ threshold, Tanimoto ≥ threshold), calibrated on val_cal for ≥88% precision within 1 log unit. The Tanimoto channel is the binding constraint; ensemble std alone is weak on OOD data (Spearman ~0.05). Coverage on ExpansionRx reflects the genuine structural distance between ChEMBL training data and ExpansionRx's RNA-targeting drug-discovery compounds; on in-domain queries, coverage would be substantially higher.

## Error analysis — worst 10 on ExpansionRx

See `reports/error_analysis/report.md` for structures + per-compound rationale.

**3 loud failures, 7 silent failures.** The 3 loud failures are flagged unreliable (low Tanimoto or high ensemble std). The 7 silent failures are genuine blind spots — compounds that pass the reliability filter but have large errors. On ExpansionRx: **3,706 of 5,039 (73.5%) flagged reliable**, of which 81.0% have |error| < 1 log unit. RMSE on the reliable subset is **0.79 vs 0.82 overall**.

**Common pattern:** 8 of 10 share an aminoquinoline/aminonaphthaline core with amidine or guanidine substituents (pKa ~10–13, fully protonated at pH 7.4). All 10 over-predict logD — the model approximates logP but these compounds are substantially more hydrophilic due to ionisation. The Henderson-Hasselbalch features help but group-average pKa can't capture substituent effects.

## What I'd do with more time

- **Per-molecule pKa predictions** via Dimorphite-DL or pkasolver — group-average pKa ignores substituent effects that shift pKa by 2+ units.
- **Aleatoric uncertainty** — MVE head in Chemprop to separate "hard molecule" from "noisy measurement."
- **Harder splits** — Butina cluster split at 0.4 Tanimoto for a tighter generalisation test.
- **Packed-bit Tanimoto** — the AD scan is 88% of inference time; packing fingerprints to uint64 + popcount gives 4–8× speedup. See [DESIGN.md](DESIGN.md) for details.
- **Retraining/monitoring loop** — weekly revalidation with drift alerts.
- **ONNX export** — drops cold start for production deployment.

## A note on AI tools

This implementation was built with heavy use of Claude (Anthropic's model, via the Claude Code CLI). I used it for scaffolding the package structure, writing tests against contracts I specified, drafting the FastAPI/Docker/profiler scripts, and catching bugs (e.g. a `pytorch_lightning` vs `lightning.pytorch` namespace clash in the Chemprop wrapper).

I did not use it to decide the modelling approach, audit data sources (I verified URLs myself — an AI-guessed one was wrong), or write the design decisions. Every decision in this repo is one I can defend.
