# Development plan: logD prediction service

**Scope:** Build a logD7.4 prediction service with uncertainty quantification, evaluated on external drug-discovery compounds.

**Constraints:** Laptop CPU + occasional Colab GPU. Python package, not notebook.

---

## Domain primer

**logD** = octanol-water partition coefficient at pH 7.4. Differs from logP (neutral form) because ionization at physiological pH matters. Core ADME property — drives absorption, BBB penetration, volume of distribution. Range typically −3 to +7.

Two things that shape modelling choices:
- **Ionization matters.** Many drugs are zwitterions or ionizable at pH 7.4 → models that ignore this struggle on those cases.
- **Assay noise floor.** Literature ~0.3–0.5 log units inter-lab. Our RMSE won't go meaningfully below that.

---

## Data sources (pinned for reproducibility)

### Training — OpenADMET-curated ChEMBL 35 LogD
- Aggregated parquet pinned to specific commit SHA of `OpenADMET/data-catalogs`.
- ~23k compounds after curation. We do our own canonicalization, desalting, dedup on top.
- Target: `standard_value_median` (robust to assay outliers). Range filter: [-5.0, 10.0].

### External benchmark — OpenADMET ExpansionRx challenge
- 5,039 LogD-labeled compounds from Expansion Therapeutics drug discovery.
- Pinned HF dataset revision. Held out from ChEMBL entirely.
- InChIKey deduplication against training set prevents leakage.

---

## Modelling

### Features (2,087 dimensions)
| Block | Count | Purpose |
|---|---|---|
| RDKit descriptors | 31 | Global molecular properties (MolWt, TPSA, LogP, etc.) |
| Ionisable group counts | 6 | Which pH-sensitive groups are present |
| Henderson-Hasselbalch pKa corrections | 2 | Estimated logD shift + net charge at pH 7.4 |
| Morgan fingerprints (r=2) | 2048 | Local substructure (ECFP4-equivalent) |

### Splits
- **Scaffold split** (Bemis-Murcko) 80/10/10 — primary evaluation.
- **Random split** — comparison to quantify generalisation gap.
- **ExpansionRx** — true OOD external test.

### Models
1. **LightGBM k=5 ensemble** — baseline. Hyperparameters tuned via grid search (27 combos: num_leaves, lr, min_child_samples). Ensemble diversity from bagging/feature subsampling (0.7/0.7).
2. **Chemprop v2 D-MPNN k=5 ensemble** — learns from molecular graph. Trained on Colab GPU.

### Uncertainty (three independent signals)
1. **Ensemble std** — epistemic. Fires when members disagree.
2. **Conformal intervals** — absolute residuals, 90% coverage, constant-width.
3. **Tanimoto AD** — max nearest-neighbour similarity to training set.
4. **Reliability flag** = AND(std ≤ threshold, Tanimoto ≥ threshold). Calibrated on val for ≥88% precision within 1 log unit.

---

## Package structure

```
src/logd/
├── data/           — openadmet_chembl.py, expansionrx.py, splits.py
├── features.py     — featurisation pipeline
├── models/         — baseline.py, chemprop_wrap.py
├── uncertainty.py  — conformal + AD + reliability
├── inference.py    — load_model() + predict()
├── training.py     — train_baseline / train_chemprop
├── api.py          — FastAPI /predict endpoint
├── cli.py          — Typer CLI
└── utils.py
```

---

## Profiling strategy

- Batch sizes: 1, 100, 1k, 10k on real ExpansionRx compounds.
- Per-stage breakdown: parse, featurise, model forward, uncertainty.
- Two optimisations for 100k+: packed-bit Tanimoto (4–8×), FAISS ANN index (50–100×).

---

## Error analysis (centerpiece)

- Worst 10 on ExpansionRx with RDKit structure images.
- Per-compound: true vs predicted, ensemble std, Tanimoto, conformal interval, reliability flag.
- Chemistry rationale: connect failures to ionisation, pKa, structural novelty.
- Classify as loud (caught by flag) vs silent (missed) failures.

---

## What's deliberately excluded (future work)

- Aleatoric MVE head (epistemic-only is defensible with ensemble + conformal)
- Butina cluster split (scaffold + random covers the ground)
- Per-molecule pKa predictions (group-average is the known limitation)
- ONNX export, packed-bit Tanimoto (documented as optimisations, not implemented)
- Pretrained molecular foundation models (MolFormer, ChemBERTa)

Restraint rationale: five solid things > eight half-done things.
