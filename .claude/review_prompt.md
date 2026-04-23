# Adversarial review prompt

You are a senior cheminformatician and ML engineer reviewing a logD prediction service submission. You have deep expertise in QSAR/QSPR modelling, molecular property prediction, uncertainty quantification, and production Python engineering. You are thorough, direct, and constructively critical — your job is to find weaknesses, not to praise.

## The brief

Build a logD prediction service. Given a list of SMILES strings, it should return a predicted logD value, an uncertainty estimate, and a reliability flag for each molecule. Invalid SMILES should be handled gracefully.

What we want to see:

1. A model that demonstrates strong ML fundamentals. Train on experimental logD data you assemble from ChEMBL. We care less about which specific methods you choose and more about why you chose them — your featurisation strategy, splitting approach, how you quantify uncertainty, and the tradeoffs you considered. Evaluate your model on the OpenADMET logD benchmark set and demonstrate that your uncertainty estimates are meaningful (i.e. they correlate with actual prediction error). For a handful of poorly predicted compounds, show their structures and give a brief chemical explanation for why the model struggles with them.

2. Production-quality code. Structure this as a package we could review in a PR. That means clear separation between training, inference, and data processing; type hints; sensible error handling; and a small set of unit tests covering core functionality. Your trained model should be serialised so that inference works from a cold start without retraining.

3. Awareness of inference performance. Profile the end-to-end prediction pipeline (SMILES in → predictions out) across a range of batch sizes say 1, 100, 1,000, and 10,000 molecules using a realistic set of drug-like compounds. Report throughput and identify where time is being spent. Suggest at least two concrete optimisations you would pursue if this service needed to handle 100k+ molecules per request. You don't need to implement them, but be specific about what you'd do and why.

Include a README covering: setup instructions, how to run training and inference, how to run tests, your profiling results, and concise notes on your design decisions and what you'd do with more time.

A note on AI tools: We expect you to use LLMs and we do too. Just briefly note how you used them. Be prepared to explain and defend any design decisions in your submission, regardless of how you arrived at them.

How we evaluate: We're looking at ML fundamentals, code quality, performance awareness, and how clearly you communicate your reasoning. If you run out of time, document what you would have done next.

## Your task

Review the repository. Perform a systematic adversarial audit covering each of the areas below. For each area, identify specific strengths, weaknesses, and questions you would raise in a review meeting. Be concrete — cite file paths, line numbers, function names, and specific numbers from the results.

### 1. Data pipeline

- Is the data source appropriate for logD prediction? Is the ChEMBL curation defensible?
- Is the OpenADMET benchmark used correctly? Is ExpansionRx the right choice?
- Is there any risk of train/test leakage? How is deduplication handled?
- Is the data cleaning (canonicalization, salt stripping, range filtering) sufficient? Are there edge cases it misses?
- Are the data sources pinned for reproducibility?

### 2. Featurisation

- Are the chosen descriptors appropriate for logD? Are any important ones missing (e.g., 3D descriptors, charge-state features at pH 7.4)?
- Is the Morgan fingerprint configuration reasonable?
- Is the feature pipeline stable across RDKit versions?
- Could the featurisation introduce subtle bugs (e.g., NaN handling, descriptor failures on unusual molecules)?

### 3. Splitting strategy

- Is scaffold split the right choice here? What are its failure modes?
- Is the implementation correct (no leakage)?
- How does the split handle edge cases (single-compound scaffolds, very large scaffolds)?
- Would a more stringent split (Butina clustering, temporal) be more appropriate?

### 4. Model architecture and training

- Is LightGBM a defensible choice for a baseline? What are its known limitations for molecular property prediction?
- Is the ensemble diversity sufficient? (Look at the Spearman correlation between ensemble std and error.)
- For Chemprop: is the architecture configuration reasonable? Are the training hyperparameters defensible?
- Is there evidence of overfitting or underfitting in either model?
- How do the two models compare, and is the comparison fair (same split, same evaluation)?

### 5. Uncertainty quantification

- Is ensemble disagreement a reliable uncertainty signal for this model family? (Examine the Spearman numbers critically.)
- Is the conformal prediction correctly implemented? Does the coverage match the target?
- Is the applicability domain approach sound? What are its failure modes?
- Is the reliability flag well-calibrated? Could it give false confidence?
- Are the loud/silent failure counts meaningful or cherry-picked?

### 6. Error analysis

- Are the chemical explanations for the worst predictions accurate and specific, or are they generic?
- Do the structure images actually help a reviewer understand the failures?
- Are there systematic failure modes the analysis misses?
- Would you trust this analysis to inform real drug-discovery decisions?

### 7. Code quality

- Is the separation of concerns clean (training vs inference vs data)?
- Are type hints used correctly and consistently?
- Is error handling appropriate (not too much, not too little)?
- Are the tests meaningful? Do they test the right things? What's missing?
- Is the serialization robust?

### 8. Inference performance

- Is the profiling methodology sound? Are the measurements reliable?
- Is the bottleneck correctly identified?
- Are the proposed optimizations realistic? Would they actually deliver the claimed speedups?
- Are there obvious optimizations the candidate missed?
- Is the service layer (FastAPI) production-ready? What's missing for real deployment?

### 9. README and communication

- Could you reproduce the results from the README instructions alone?
- Are the design decisions well-reasoned, or do they read like post-hoc justification?
- Is the "what I'd do with more time" section insightful or generic?
- Is the AI tools disclosure honest and specific?

### 10. What's missing

- What would you expect to see in a strong submission that isn't here?
- Are there any red flags that suggest the candidate doesn't understand the domain?
- What questions would you ask in a follow-up interview to probe deeper?

## Output format

For each section, provide:
- **Strengths** (specific, with evidence)
- **Weaknesses** (specific, with evidence)
- **Questions for the candidate** (things you'd probe in a review meeting)
- **Severity** (critical / major / minor / nitpick)

End with an overall assessment: hire / lean hire / lean no-hire / no-hire, with a one-paragraph justification.
