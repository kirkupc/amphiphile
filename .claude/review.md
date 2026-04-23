# Plan review: line-by-line audit + adversarial pass

Written after Day 1 scaffold landed, before the first real training run. Goal: make sure every single requirement is addressed, and pressure-test the weak assumptions before we burn more time.

---

## Part 1: Requirements → plan, requirement by requirement

| Requirement | Plan/code status | Gap | Severity |
|---|---|---|---|
| "Build a logD prediction **service**" | Python library + CLI | **"Service" implies HTTP endpoint + Dockerfile. A biotech reviewer will read "service" literally.** | **HIGH** |
| List of SMILES → predicted logD per molecule | `predict(smiles_list)` returns `Prediction` dataclass per input | — | OK |
| Uncertainty estimate per molecule | `Prediction.uncertainty` (ensemble std) | — | OK |
| Reliability flag per molecule | `Prediction.reliable` (bool: std < thr AND Tanimoto > thr) | — | OK |
| Invalid SMILES handled gracefully | `error='invalid_smiles'`, never crashes | — | OK |
| **"We will re-run your code, so results must be reproducible"** | Seeds pinned in code | **ChEMBL version NOT pinned. OpenADMET URL tracks `main` branch (drifts). uv.lock not committed. Pre-trained artifacts not published anywhere for re-use.** | **HIGH** |
| Production-quality code (not notebook) | package layout ✓ | — | OK |
| Python package we can install and run | `uv sync` works ✓ | — | OK |
| Train on ChEMBL logD | `chembl.py` pulls `LogD` + `LogD7.4` | — | OK |
| "Care less *which* methods, more *why*" | PLAN.md rationale per decision | Needs to carry over to README; PLAN will eventually be superseded | MED |
| Featurisation strategy | RDKit desc + Morgan (baseline); D-MPNN (Chemprop) | Fine. Could also mention: pretrained molecular LMs (MolFormer, ChemBERTa) as "what I'd do with more time" | LOW |
| Splitting approach | Scaffold (Bemis-Murcko) + random for comparison | Could add cluster-split (Butina) for a harder test. Cheap to do. | MED |
| Uncertainty quantification | Ensemble + conformal + AD Tanimoto | Missing: **aleatoric vs epistemic decomposition** (MVE head), reliability diagram, coverage on external test. | MED |
| Tradeoffs considered | Scattered in PLAN | Need explicit "alternatives considered" section in README | MED |
| **Evaluate on OpenADMET logD benchmark** | `openadmet.py` fetches from a hardcoded URL | **URL is unverified — I have not confirmed this file actually exists at that path.** If it's wrong, external eval is broken. | **HIGH** |
| Uncertainty correlates with error (Spearman) | `spearman(std, abs_err)` in metrics | OK. Strengthen with confidence curve + coverage. | LOW |
| "Handful of poorly predicted compounds" with structures + chemistry | `error_analysis.py` planned, Day 3 | Not yet implemented. **This is the narrative centerpiece — most submissions will be weak here. Invest.** | MED (scheduling) |
| Clear separation: training / inference / data | features, data/, models/, inference, training are separate modules | `training.py` does data prep + train + eval — arguable. Fine for now. | LOW |
| Type hints | Present throughout | mypy --strict not run. Might have holes. | MED |
| Sensible error handling | Invalid SMILES path is solid | Bare `except Exception` in features.py and splits.py around descriptor calc / scaffold calc. Fail-soft is defensible but should log. | MED |
| Small set of unit tests | 27 tests, 4 files | Solid coverage. Missing: fuzz for pathological SMILES (huge mols, stereo, charges). | LOW |
| Model serialisable, cold-start inference | `BaselineModel.save/load` + `load_model()` | Baseline ✓. Chemprop serialisation works, but **we haven't validated round-trip** (test file failed due to unrelated bug). | MED |
| Profile pipeline, batch sizes {1, 100, 1k, 10k} | Planned | Not implemented. Need to pick **realistic drug-like compound set**. | MED (scheduling) |
| Identify where time is spent | Per-stage breakdown planned | Not implemented | MED (scheduling) |
| ≥2 concrete optimisations for 100k+ req | 4 listed in PLAN section 4 | Good. Make them **more specific** in README (which lines of code, which libraries). | LOW |
| README: setup, training, inference, tests, profiling results, design decisions, more-time notes | Stub written | Needs results after training runs | OK (scheduling) |
| Note on AI tools usage | Stub | Needs to be written with specifics — which tools for which tasks | LOW |

---

## Part 2: Adversarial challenges

### Challenge 1: "Service" vs "library" — biggest framing risk

The brief says **service** three times. Our current delivery is an importable Python library + CLI. A senior biotech engineer reads "logD prediction service" as: "I can POST SMILES to an HTTP endpoint and get JSON back, with a Dockerfile I can run."

**Cost to address:** ~100 lines of FastAPI (`src/logd/api.py` — `POST /predict` with pydantic request/response), a 15-line Dockerfile, and a README section showing `curl` against the running container.

**Cost to ignore:** The reviewer notices the gap within 30 seconds of reading the repo. Easy signal to read as "didn't read the brief carefully."

**Recommendation:** Add FastAPI + Dockerfile to Day 4 scope.

### Challenge 2: Reproducibility is soft

- `chembl.load(version=None)` defaults to "latest." In 3 months, reviewer pulls different data than we trained on.
- `openadmet.py` hits `https://raw.githubusercontent.com/OpenADMET/openadmet-benchmarks/main/...` — `main` drifts.
- No `uv.lock` committed.
- No pre-trained artifacts published. Reviewer has to re-train (hours) before they can eval.

**Fix:**
- Pin OpenADMET to a specific commit SHA.
- Commit `uv.lock`.
- Publish model artifacts to a GitHub release and have `load_model()` download on first use from a URL.

### Challenge 3: OpenADMET URL is unverified

I hardcoded an OpenADMET URL based on a plausible path structure. I do not know if that file exists at that URL. If it doesn't, the whole external eval is broken.

**Fix:** First action of next work session — curl the URL. If it's wrong, find the real benchmark (probably `OpenADMET/openadmet` or Polaris or a paper supplement) and pin to a commit SHA.

### Challenge 4: "Strong ML fundamentals" — are we just competent?

Current plan: RDKit desc + Morgan → LightGBM ensemble (baseline); Chemprop D-MPNN ensemble (stronger); conformal + ensemble std + Tanimoto AD.

This is solid. It is also what a decent 2021 submission would look like.

Things that read as **sophisticated** and are cheap to add:

1. **Data quality audit** — group ChEMBL assays by within-assay variance on compounds that appear in multiple assays. Report the intrinsic noise floor (~0.3–0.5 log units in the literature). Argue our RMSE is close to that floor or explain why not. Shows we understand the data, not just the model.
2. **Aleatoric uncertainty** — add an MVE head so we can decompose total uncertainty into epistemic (ensemble disagreement) and aleatoric (data noise). Chemprop has this built in.
3. **Multiple split strategies** — scaffold, random, AND Butina cluster split. Report the gap.
4. **Conformal coverage on external test** — not just "the calibration holds on val," but "our 90% intervals cover X% of OOD compounds, gap explained by distribution shift."

**Pick 2–3.** Minimum: data quality audit (cheap, huge narrative payoff) + coverage on external test (literally 2 lines of code).

### Challenge 5: Error analysis is the differentiator

The brief asks for "structures + chemical explanation" of poorly predicted compounds. This is the one place where:
- Most submissions will show 3 structures + 1-line "model overfits" captions.
- A sophisticated submission will connect the worst cases to the uncertainty signal (were they flagged as unreliable? yes/no, by which channel — high ensemble std vs low Tanimoto vs both?).

This is where reviewers see whether we understand our own model.

**Recommendation:** Treat error analysis as the writeup centerpiece, not a footer. Plan ~half a day on it Day 3.

### Challenge 6: Dependency weight

Full install pulls torch + chemprop + lightning (~1.5 GB). A reviewer installing fresh on their laptop may push back. We can split into core vs optional extras. But: single `uv sync` is simpler for reviewers. Decided to keep it simple.

### Challenge 7: Chemprop namespace bug

`tests/test_chemprop.py` failed with:
```
TypeError: `model` must be a `LightningModule`... got `MPNN`
```

Root cause: `chemprop_wrap.py` imports `pytorch_lightning as pl` but Chemprop v2 builds models against `lightning.pytorch`. The trainer from `pytorch_lightning` refuses a model registered against `lightning.pytorch`.

Symptom is minor, cause is cultural: **we shipped this without running the test.** Need a pre-commit / CI that runs `pytest -m 'not slow'` + `ruff check` + `mypy`. A 5-line `Makefile` is enough.

### Challenge 8: Production-quality = survives pathological inputs

A "service" should not crash on:
- Empty string, `None`, non-string
- Enormous molecule (peptide with 500 atoms)
- Repeated submission of the same SMILES (cache behaviour)
- SMILES with exotic stereo or charges that RDKit parses but our features choke on
- Unicode / whitespace / trailing newlines

**Recommendation:** A `test_robustness.py` with ~10 pathological inputs that must round-trip through `predict()` without raising. Also: add a max molecule size check with a clear error.

---

## Part 3: Revised plan (minimal-diff from current)

### Immediate before any more coding

1. **Verify OpenADMET URL.** One curl. If wrong, find the real source. Blocker.
2. **Fix Chemprop namespace bug.** Swap `pytorch_lightning` → `lightning.pytorch` in `chemprop_wrap.py`. 1-line fix.
3. **Commit uv.lock.** Remove `.uv-cache/` pattern from gitignore, add `!uv.lock`.

### Day 1 (remainder)

4. First real baseline training run → `reports/metrics.json` with scaffold test + ExpansionRx RMSE/MAE/R.
5. Sanity-check: are the numbers in a sensible range? (Literature says ~0.6–0.8 log units RMSE for logD with decent models.)

### Day 2

6. Fix Chemprop wrapper, train ensemble on Colab GPU.
7. Compare: baseline vs Chemprop on both splits.

### Day 3

8. **Error analysis is the centerpiece.** Worst-10 compounds on ExpansionRx, structures in `reports/error_analysis/`, narrative tying each to the uncertainty channels.
9. Data quality audit: intra-assay vs inter-assay variance from ChEMBL, report noise floor.

### Day 4

10. **FastAPI + Dockerfile** (new scope, ~1 hour total).
11. Profiling harness with drug-like compound set.
12. Robustness test for pathological inputs.
13. `Makefile` with `lint / type-check / test / profile` targets.

### Day 5

14. Run profiler, write results tables.
15. README + design-decisions doc + AI-tools note + model-artifact download instructions.
16. Publish model artifacts (GitHub release).
17. Polish.

---

## Part 4: Rubric self-scoring (current plan vs revised)

| Axis | Current plan | Revised plan |
|---|---|---|
| ML fundamentals | 7/10 | 8.5/10 (+data audit, +cluster split) |
| Code quality | 7/10 | 8.5/10 (+ lint/type CI, +robustness tests, +Makefile) |
| Performance awareness | 6/10 | 8/10 (+ real profile, + FastAPI service framing) |
| Communication | 6/10 | 9/10 (+ error analysis centerpiece, + data quality narrative) |

**Delta: ~+1.5 per axis on average for ~1 extra day of total effort.**

---

## Top-3 changes to commit to

If we only accept 3 changes from this review:

1. **Verify OpenADMET URL + pin all data sources.** Non-negotiable for reproducibility.
2. **FastAPI + Dockerfile.** Addresses the "service" framing with minimal effort.
3. **Error analysis as centerpiece.** Highest leverage for the "communication" axis.

Everything else is optional. But each has real value.
