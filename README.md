# logd

A logD prediction service. Given SMILES strings, returns predicted logD7.4 with an uncertainty estimate and a reliability flag per molecule.

**Status:** work in progress. See [PLAN.md](PLAN.md) for the full design.

## Install

Requires Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Run inference

```python
from logd import load_model, predict

model = load_model()  # loads serialized artifacts — cold start, no training
results = predict(["CCO", "c1ccccc1", "not a smiles"], model=model)
for r in results:
    print(r)
```

Or via the CLI:

```bash
uv run logd predict --smiles "CCO" --smiles "c1ccccc1"
```

## Run training from scratch

```bash
uv run logd prepare-data        # fetch + clean ChEMBL + OpenADMET
uv run logd train --model baseline    # LightGBM on RDKit descriptors + Morgan fps
uv run logd train --model chemprop    # Chemprop D-MPNN (uses GPU if available)
uv run logd evaluate            # runs on scaffold test + OpenADMET, writes reports/metrics.json
```

All scripts are seed-controlled and ChEMBL version is pinned.

## Run tests

```bash
uv run pytest
```

## Profile inference

```bash
uv run logd profile   # batch sizes 1, 100, 1000, 10000 on drug-like compounds
```

Writes `reports/profiling.json` and a per-stage breakdown.

## Results

_To be filled in after training runs. See `reports/metrics.json` and `reports/profiling.json`._

## Design decisions

See [PLAN.md](PLAN.md) for the full reasoning. Short version:

- **Featurisation:** RDKit 2D descriptors + Morgan fingerprints (baseline); Chemprop D-MPNN (stronger).
- **Split:** scaffold (Bemis-Murcko) on ChEMBL, plus external test on OpenADMET.
- **Uncertainty:** deep ensemble (k=5) + Mondrian conformal + applicability-domain Tanimoto check.
- **Reliability flag:** true iff ensemble_std below threshold AND nearest-neighbour Tanimoto above threshold.

## AI tools

_To be filled in._
