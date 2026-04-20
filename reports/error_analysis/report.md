# Error analysis — worst 10 predictions on ExpansionRx

- Median ensemble std across full test: 0.128 log units
- Reliability thresholds: std ≤ 0.167, Tanimoto ≥ 0.552
- Of the worst 10: **10 would have been flagged unreliable** (loud), **0 silent failures**

## Per-compound breakdown

### 1. abs_error = 4.48 log units

![structure](worst_01.png)

- SMILES: `Cn1cc(-c2ccc(OCC[N+]3(C)CCCCC3)cc2)c2cc[nH]c2c1=O`
- True: -1.50, Predicted: 2.98
- Ensemble std: 0.165, Nearest-training Tanimoto: 0.313
- Conformal ±1.91 (covers truth: NO)
- Would reliability flag fire: YES (loud failure — system knew)

**Rationale:** No obvious structural outlier; this is a silent failure — high error but all uncertainty channels reported confidence. worth investigating the train distribution..

### 2. abs_error = 4.27 log units

![structure](worst_02.png)

- SMILES: `Cn1cc(C[N+]2(CCCn3ccc4c5cc(O)ccc5n(C)c4c3=O)CCCCC2)cn1`
- True: -1.20, Predicted: 3.07
- Ensemble std: 0.097, Nearest-training Tanimoto: 0.256
- Conformal ±1.13 (covers truth: NO)
- Would reliability flag fire: YES (loud failure — system knew)

**Rationale:** Far from training distribution (max tanimoto to train = 0.26) — applicability-domain check correctly flags this as extrapolation.

### 3. abs_error = 4.09 log units

![structure](worst_03.png)

- SMILES: `CNC(=N)c1ccc2nc(C)cc(Nc3cccc(OC)c3)c2c1`
- True: -0.50, Predicted: 3.59
- Ensemble std: 0.285, Nearest-training Tanimoto: 0.446
- Conformal ±3.31 (covers truth: NO)
- Would reliability flag fire: YES (loud failure — system knew)

**Rationale:** High ensemble disagreement (std=0.29 vs median 0.13) — model internals knew this was uncertain.

### 4. abs_error = 4.01 log units

![structure](worst_04.png)

- SMILES: `COc1cccc(Nc2cc(C)nc3ccc(N(C)C(=N)N)cc23)c1`
- True: -0.70, Predicted: 3.31
- Ensemble std: 0.221, Nearest-training Tanimoto: 0.438
- Conformal ±2.57 (covers truth: NO)
- Would reliability flag fire: YES (loud failure — system knew)

**Rationale:** High ensemble disagreement (std=0.22 vs median 0.13) — model internals knew this was uncertain.

### 5. abs_error = 3.84 log units

![structure](worst_05.png)

- SMILES: `CNC(=N)Nc1ccc2nc(C)cc(Nc3cccc(OC)c3)c2c1`
- True: -0.30, Predicted: 3.54
- Ensemble std: 0.218, Nearest-training Tanimoto: 0.484
- Conformal ±2.53 (covers truth: NO)
- Would reliability flag fire: YES (loud failure — system knew)

**Rationale:** High ensemble disagreement (std=0.22 vs median 0.13) — model internals knew this was uncertain.

### 6. abs_error = 3.64 log units

![structure](worst_06.png)

- SMILES: `COc1cccc(Nc2cc(C)nc3ccc(NC(=N)N)cc23)c1`
- True: -0.50, Predicted: 3.14
- Ensemble std: 0.205, Nearest-training Tanimoto: 0.492
- Conformal ±2.38 (covers truth: NO)
- Would reliability flag fire: YES (loud failure — system knew)

**Rationale:** High ensemble disagreement (std=0.21 vs median 0.13) — model internals knew this was uncertain.

### 7. abs_error = 3.59 log units

![structure](worst_07.png)

- SMILES: `CNC(=N)c1ccc2nc(N(C)C)cc(Nc3cccc(OC)c3)c2c1`
- True: -0.30, Predicted: 3.29
- Ensemble std: 0.243, Nearest-training Tanimoto: 0.347
- Conformal ±2.82 (covers truth: NO)
- Would reliability flag fire: YES (loud failure — system knew)

**Rationale:** High ensemble disagreement (std=0.24 vs median 0.13) — model internals knew this was uncertain.

### 8. abs_error = 3.39 log units

![structure](worst_08.png)

- SMILES: `CNc1ccnc(N(CCc2ccncc2)c2ccnc(Nc3ccc(OC)cc3)n2)n1`
- True: -0.60, Predicted: 2.79
- Ensemble std: 0.145, Nearest-training Tanimoto: 0.368
- Conformal ±1.68 (covers truth: NO)
- Would reliability flag fire: YES (loud failure — system knew)

**Rationale:** No obvious structural outlier; this is a silent failure — high error but all uncertainty channels reported confidence. worth investigating the train distribution..

### 9. abs_error = 3.36 log units

![structure](worst_09.png)

- SMILES: `CNC(=N)Nc1ccc2c(Nc3cccc(OC)c3)cc(C)nc2c1`
- True: 0.00, Predicted: 3.36
- Ensemble std: 0.121, Nearest-training Tanimoto: 0.439
- Conformal ±1.41 (covers truth: NO)
- Would reliability flag fire: YES (loud failure — system knew)

**Rationale:** No obvious structural outlier; this is a silent failure — high error but all uncertainty channels reported confidence. worth investigating the train distribution..

### 10. abs_error = 3.31 log units

![structure](worst_10.png)

- SMILES: `Cc1nnc(-c2ccn3nc(C(=O)NCCCC4=NCCN4)cc3c2)s1`
- True: -1.90, Predicted: 1.41
- Ensemble std: 0.186, Nearest-training Tanimoto: 0.256
- Conformal ±2.15 (covers truth: NO)
- Would reliability flag fire: YES (loud failure — system knew)

**Rationale:** Far from training distribution (max tanimoto to train = 0.26) — applicability-domain check correctly flags this as extrapolation.
