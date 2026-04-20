# Error analysis — worst 10 predictions on ExpansionRx

- Median ensemble std across full test: 0.136 log units
- Reliability thresholds: std ≤ 0.213, Tanimoto ≥ 0.552
- Of the worst 10: **10 would have been flagged unreliable** (loud), **0 silent failures**

## Per-compound breakdown

### 1. abs_error = 4.23 log units

![structure](worst_01.png)

- SMILES: `Cn1cc(-c2ccc(OCC[N+]3(C)CCCCC3)cc2)c2cc[nH]c2c1=O`
- True: -1.50, Predicted: 2.73
- Ensemble std: 0.179, Nearest-training Tanimoto: 0.313
- Conformal ±1.89 (covers truth: NO)
- Would reliability flag fire: YES (loud failure — system knew)

**Rationale:** No obvious structural outlier; this is a silent failure — high error but all uncertainty channels reported confidence. worth investigating the train distribution..

### 2. abs_error = 4.19 log units

![structure](worst_02.png)

- SMILES: `Cn1cc(C[N+]2(CCCn3ccc4c5cc(O)ccc5n(C)c4c3=O)CCCCC2)cn1`
- True: -1.20, Predicted: 2.99
- Ensemble std: 0.181, Nearest-training Tanimoto: 0.256
- Conformal ±1.91 (covers truth: NO)
- Would reliability flag fire: YES (loud failure — system knew)

**Rationale:** Far from training distribution (max tanimoto to train = 0.26) — applicability-domain check correctly flags this as extrapolation.

### 3. abs_error = 4.13 log units

![structure](worst_03.png)

- SMILES: `CNC(=N)c1ccc2nc(C)cc(Nc3cccc(OC)c3)c2c1`
- True: -0.50, Predicted: 3.63
- Ensemble std: 0.156, Nearest-training Tanimoto: 0.446
- Conformal ±1.65 (covers truth: NO)
- Would reliability flag fire: YES (loud failure — system knew)

**Rationale:** No obvious structural outlier; this is a silent failure — high error but all uncertainty channels reported confidence. worth investigating the train distribution..

### 4. abs_error = 3.95 log units

![structure](worst_04.png)

- SMILES: `COc1cccc(Nc2cc(C)nc3ccc(N(C)C(=N)N)cc23)c1`
- True: -0.70, Predicted: 3.25
- Ensemble std: 0.167, Nearest-training Tanimoto: 0.438
- Conformal ±1.77 (covers truth: NO)
- Would reliability flag fire: YES (loud failure — system knew)

**Rationale:** No obvious structural outlier; this is a silent failure — high error but all uncertainty channels reported confidence. worth investigating the train distribution..

### 5. abs_error = 3.67 log units

![structure](worst_05.png)

- SMILES: `CNC(=N)Nc1ccc2nc(C)cc(Nc3cccc(OC)c3)c2c1`
- True: -0.30, Predicted: 3.37
- Ensemble std: 0.258, Nearest-training Tanimoto: 0.484
- Conformal ±2.73 (covers truth: NO)
- Would reliability flag fire: YES (loud failure — system knew)

**Rationale:** High ensemble disagreement (std=0.26 vs median 0.14) — model internals knew this was uncertain.

### 6. abs_error = 3.52 log units

![structure](worst_06.png)

- SMILES: `CNC(=N)c1ccc2nc(N(C)C)cc(Nc3cccc(OC)c3)c2c1`
- True: -0.30, Predicted: 3.22
- Ensemble std: 0.180, Nearest-training Tanimoto: 0.347
- Conformal ±1.90 (covers truth: NO)
- Would reliability flag fire: YES (loud failure — system knew)

**Rationale:** No obvious structural outlier; this is a silent failure — high error but all uncertainty channels reported confidence. worth investigating the train distribution..

### 7. abs_error = 3.48 log units

![structure](worst_07.png)

- SMILES: `COc1cccc(Nc2cc(C)nc3ccc(NC(=N)N)cc23)c1`
- True: -0.50, Predicted: 2.98
- Ensemble std: 0.280, Nearest-training Tanimoto: 0.492
- Conformal ±2.95 (covers truth: NO)
- Would reliability flag fire: YES (loud failure — system knew)

**Rationale:** High ensemble disagreement (std=0.28 vs median 0.14) — model internals knew this was uncertain.

### 8. abs_error = 3.35 log units

![structure](worst_08.png)

- SMILES: `CNc1ccnc(N(CCc2ccncc2)c2ccnc(Nc3ccc(OC)cc3)n2)n1`
- True: -0.60, Predicted: 2.75
- Ensemble std: 0.167, Nearest-training Tanimoto: 0.368
- Conformal ±1.76 (covers truth: NO)
- Would reliability flag fire: YES (loud failure — system knew)

**Rationale:** No obvious structural outlier; this is a silent failure — high error but all uncertainty channels reported confidence. worth investigating the train distribution..

### 9. abs_error = 3.30 log units

![structure](worst_09.png)

- SMILES: `CNC(=N)c1ccc2nc(C)cc(Oc3cccc(OC)c3)c2c1`
- True: 0.10, Predicted: 3.40
- Ensemble std: 0.099, Nearest-training Tanimoto: 0.460
- Conformal ±1.05 (covers truth: NO)
- Would reliability flag fire: YES (loud failure — system knew)

**Rationale:** No obvious structural outlier; this is a silent failure — high error but all uncertainty channels reported confidence. worth investigating the train distribution..

### 10. abs_error = 3.23 log units

![structure](worst_10.png)

- SMILES: `CNC(=N)Nc1ccc2c(Nc3cccc(OC)c3)cc(C)nc2c1`
- True: 0.00, Predicted: 3.23
- Ensemble std: 0.155, Nearest-training Tanimoto: 0.439
- Conformal ±1.64 (covers truth: NO)
- Would reliability flag fire: YES (loud failure — system knew)

**Rationale:** No obvious structural outlier; this is a silent failure — high error but all uncertainty channels reported confidence. worth investigating the train distribution..
