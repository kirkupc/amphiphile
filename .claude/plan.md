# logD prediction service — implementation plan

**Constraints:** Laptop + occasional Colab GPU. **Tooling:** `uv` + `pyproject.toml`. **Ambition:** baseline + one stronger model.

---

## 0. Status

Day 1 scaffold landed (27 tests passing). This plan was revised after REVIEW.md adversarial pass. Key changes from v1:

1. **Data sources verified.** Original plan guessed OpenADMET URLs; real sources identified (see §2).
2. **Training data simplified.** Use OpenADMET's *curated* ChEMBL35 LogD parquet (1.1 MB, pinned URL) instead of a 4 GB `chembl_downloader` pull. Still ChEMBL-sourced, reproducible in minutes.
3. **External benchmark corrected.** OpenADMET's logD benchmark = ExpansionRx challenge training data (5,039 compounds, real drug discovery, held out from ChEMBL). Not a guessed path to a curated ChEMBL mirror.
4. **"Service" framing taken literally.** FastAPI endpoint + Dockerfile added. Brief says "service" three times.
5. **Error analysis promoted to centerpiece.** Most submissions are weak here; highest leverage for the communication rubric.
6. **Complexity restrained.** Cut MVE head, Butina cluster split, and multi-extras dep split. Deep ensemble + Mondrian conformal + AD Tanimoto is a defensible uncertainty story on its own. Cuts moved to "what I'd do with more time."

---

## 1. Domain primer (for defendable choices)

**logD** = octanol-water partition coefficient at pH 7.4. Differs from logP (neutral form) because ionization at physiological pH matters. Core ADME property — drives absorption, BBB penetration, volume of distribution. Range typically −3 to +7.

Two things that shape modelling choices:
- **Ionization matters.** Many drugs are zwitterions or ionizable at pH 7.4 → models that ignore this struggle on those cases. Shows up in error analysis.
- **Assay noise floor.** Literature ~0.3–0.5 log units inter-lab. Our RMSE won't go meaningfully below that; if we approach it, we're near the ceiling.

---

## 2. Data sources (verified, pinned)

### Training data — OpenADMET-curated ChEMBL 35 LogD
- **URL:** `https://raw.githubusercontent.com/OpenADMET/data-catalogs/{COMMIT_SHA}/catalogs/activities/ChEMBL_LogD/ChEMBL35_LogD/ChEMBL_LogD_LogD_aggregated.parquet`
- Pin a specific commit SHA from `OpenADMET/data-catalogs` at first fetch; store in code.
- 1.1 MB parquet; columns likely include SMILES + aggregated logD value.
- Also available: raw (4.6 MB) — fall back if aggregation differs from what we want.

### External benchmark — OpenADMET ExpansionRx challenge
- **URL:** `https://huggingface.co/datasets/openadmet/openadmet-expansionrx-challenge-train-data/resolve/{COMMIT_SHA}/expansion_data_train.csv`
- Pin the dataset revision SHA.
- **5,039 LogD-labeled compounds.** Range −2.0 to +5.2, mean 2.1, std 1.19.
- Real drug discovery data from Expansion Therapeutics, held out from ChEMBL. Different distribution — legitimate generalization test.
- We use this for external eval only; do NOT mix into training.

### De-duplication
- Compute InChIKey for both sets; drop any training compound that appears in the ExpansionRx test set. Report overlap count.

---

## 3. Modelling

### Featurisation

| Track | Features | Why |
|---|---|---|
| Baseline | RDKit 2D descriptors (~31 named) + ionisable group counts + pKa corrections + Morgan fps (r=2, 2048) | Fast, interpretable, strong for logD — partitioning well-captured by count descriptors + pH corrections. |
| Stronger | Chemprop v2 D-MPNN | Learns features end-to-end. Reference implementation in the logD literature. |

### Splits
- Primary: **scaffold split (Bemis-Murcko)** on ChEMBL, 80/10/10 train/val/test.
- Also report: **random split** on the same data — gap between random and scaffold RMSE is the generalization story.
- External test: **ExpansionRx** (held out entirely from training).

### Uncertainty (kept simple)
- **Deep ensemble (k=5)** — epistemic uncertainty from 5 differently-seeded models.
- **Conformal prediction** — calibrate on val; produces 90% intervals with guaranteed marginal coverage.
- **Applicability-domain Tanimoto** — max Morgan similarity of query vs training set; catches "ensemble happens to agree while extrapolating."
- **Reliability flag:** true iff `(ensemble_std ≤ std_thr) AND (nearest_tanimoto ≥ tani_thr)`. Thresholds calibrated on val for target precision.

Evidence it works: Spearman(std, |error|), confidence curve, **empirical coverage on ExpansionRx** (the conformal coverage reality-check).

**Not doing** (future work): aleatoric MVE head (cuts scope; ensemble+conformal is defensible), Butina cluster split (scaffold + random covers the ground), temperature scaling.

### Error analysis (the centerpiece)
- 5–10 worst-predicted compounds on the ExpansionRx set.
- For each: RDKit structure image + 2–3 sentences tying the failure to the uncertainty channel(s) that fired (or didn't) — ensemble std, Tanimoto, conformal interval width.
- The narrative question: **does high uncertainty predict high error, or do we have silent failures?**

### Small data quality audit
- For compounds with multiple ChEMBL assay measurements: report within-compound variance distribution. Gives us the "noise floor" line on the RMSE plot.
- One figure, one paragraph in the README. Not a deep analysis.

---

## 4. Package structure

```
amphiphile/
├── pyproject.toml, uv.lock, README.md
├── Dockerfile
├── Makefile                        # check / test / lint / train / serve
├── src/logd/
│   ├── data/                       # openadmet_chembl.py, expansionrx.py, splits.py
│   ├── features.py
│   ├── models/                     # baseline.py, chemprop_wrap.py
│   ├── uncertainty.py
│   ├── inference.py                # load_model() + predict()
│   ├── training.py                 # train_baseline / train_chemprop
│   ├── api.py                      # FastAPI /predict endpoint
│   ├── cli.py
│   └── utils.py
├── scripts/                        # prepare_data, profile_inference, error_analysis
├── tests/                          # features, splits, uncertainty, inference, robustness, api
├── models/                         # serialized artifacts
└── reports/                        # metrics.json, profiling.json, error_analysis/*.png
```

### Public API (Python)
```python
from logd import load_model, predict
model = load_model()
results = predict(["CCO", "invalid", "c1ccccc1"], model=model)
# [Prediction(smiles="CCO", predicted_logd=-0.31, uncertainty=0.18, reliable=True, error=None), ...]
```

### Service API (HTTP)
```
POST /predict
{ "smiles": ["CCO", "c1ccccc1", ...] }
→ { "predictions": [ { smiles, predicted_logd, uncertainty, reliable, error }, ... ] }

GET /health → { "status": "ok", "model_version": "..." }
```

### Dockerfile
Single-stage, uv-based. `docker build -t logd . && docker run -p 8000:8000 logd`. Health check wired.

---

## 5. Reproducibility

- Pin data via commit SHAs for both OpenADMET catalogs repo and the ExpansionRx HF dataset.
- Commit `uv.lock`.
- Seeds controlled via `logd.utils.set_seed` (numpy, random, torch, lightning).
- Publish pre-trained artifacts (GitHub release asset) so reviewer doesn't have to retrain. `load_model()` downloads on first use if `models/` is empty.
- README: one command to reproduce metrics (`make train && cat reports/metrics.json`).

---

## 6. Inference performance

### Profiling harness
- Batch sizes: **1, 100, 1,000, 10,000**
- Compound set: ExpansionRx pool (realistic drug-like distribution, OOD, no overlap with training).
- Metrics: wall time, throughput (mol/s), peak RSS, per-stage breakdown (parse+canonicalize, featurize, model forward, uncertainty+postprocess).
- Tooling: `time.perf_counter` + `psutil`.

### Two+ concrete optimisations for 100k+ req
1. **Packed-bit Tanimoto with popcount.** Pack fingerprints to uint64 + hardware popcount. 4–8× speedup on the 88% dominant stage.
2. **Approximate nearest-neighbour index (FAISS).** Drops O(batch × n_train) to O(batch × log n_train). Additional 50–100× on top of packed-bit.
3. **Parallelize featurisation with `multiprocessing.Pool`.** RDKit descriptor calc is CPU-bound Python → ~N× speedup on N cores.
4. **LRU cache keyed by canonical InChIKey.** Production workloads have duplicates; 10–30% hit rate typical.

---

## 7. Tests

- `test_features.py` — invalid SMILES, canonicalisation, Morgan shape/determinism, pKa corrections.
- `test_splits.py` — scaffold no-leakage, random-split proportions.
- `test_uncertainty.py` — conformal shape, coverage near target, AD self-similarity, threshold calibration.
- `test_inference.py` — invalid SMILES, canonical equivalence, batch vs single, order preservation, save/load round-trip.
- `test_robustness.py` — Empty, None, huge molecule, unicode, whitespace, known-pathological SMILES. No crashes.
- `test_api.py` — FastAPI endpoint round-trip for valid + invalid inputs.
- `test_chemprop.py` — slow marker; train tiny ensemble, predict, round-trip.
- `test_integration.py` — end-to-end with real trained model artifacts.

`make check` = `ruff check && mypy src && pytest -m 'not slow'`. Must be clean before any "done" claim.

---

## 8. Day-by-day schedule

| Day | Deliverable |
|---|---|
| **1** | Fix Chemprop namespace bug. Rewrite data loading with pinned parquet. First real baseline training run; `reports/metrics.json` with scaffold test + ExpansionRx RMSE/MAE/R. Commit `uv.lock`. |
| **2** | Chemprop ensemble training on Colab GPU. Metrics compared to baseline. If Chemprop install flakes badly, fall back to descriptor-MLP ensemble with the same calibration pipeline. |
| **3** | Error analysis (centerpiece — structures + chemistry narrative tied to uncertainty channels). Small data-quality audit. Conformal coverage on external test. |
| **4** | FastAPI `/predict` endpoint + Dockerfile. Robustness test for pathological inputs. Profiling harness with drug-like compound set. Makefile with `check` target. |
| **5** | Profiling run + tables. README writeup. Publish model artifacts to GitHub release. Polish buffer. |

**Highest-risk items** (address early): OpenADMET data access (✓ verified), Chemprop install across environments, ExpansionRx column naming stability. Mitigation: Day 1 must end with working end-to-end baseline; if Chemprop breaks Day 2, descriptor-MLP fallback ships.

---

## 9. What's been cut vs original

Moved to "more time":
- Aleatoric MVE head (epistemic-only is fine; document the gap).
- Butina cluster split (scaffold + random covers it).
- Dependency extras split (single `uv sync` is simpler for reviewers).
- Pretrained molecular foundation models (MolFormer etc).

Restraint rationale: every added axis is an extra thing to defend. Five solid things > eight half-done things.
