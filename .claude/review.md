# Review methodology

This document describes the adversarial review process used throughout development to identify gaps, inconsistencies, and weaknesses before shipping.

## Approach

Each review round follows a systematic audit against the project requirements:

1. **Requirements coverage** — every stated requirement mapped to code, tests, and documentation.
2. **Number consistency** — all metrics in README, DESIGN.md, and reports/*.json cross-checked for agreement.
3. **Code-documentation alignment** — claims in docs verified against actual implementation.
4. **ML methodology** — splitting, calibration, evaluation methodology reviewed for correctness.
5. **Production readiness** — error handling, edge cases, CLI behavior, Docker, serialization.

## Review prompt

Reviews are conducted by prompting Claude Code to act as a senior cheminformatician and ML engineer. The prompt covers 10 areas:

1. Data pipeline (sources, leakage, reproducibility)
2. Featurisation (descriptor choice, stability, edge cases)
3. Splitting strategy (scaffold correctness, failure modes)
4. Model architecture and training (hyperparameters, overfitting, comparison fairness)
5. Uncertainty quantification (calibration, coverage, silent failures)
6. Error analysis (chemical accuracy, systematic patterns)
7. Code quality (separation of concerns, types, tests, serialization)
8. Inference performance (profiling methodology, bottleneck identification, optimisation realism)
9. README and communication (reproducibility, design decisions, honesty)
10. What's missing (red flags, domain understanding gaps)

Each finding is classified by severity (critical / major / minor / nitpick) with specific file:line references.

## Review rounds

Multiple rounds were conducted during development. Each round:
- Identifies discrepancies
- Fixes are implemented
- Next round verifies fixes and looks for new issues
- Process continues until 0 critical/major discrepancies remain

Key issues caught and fixed through this process:
- Stale numbers in documentation after model retraining
- Conformal calibration using Mondrian approach (inflated quantile on OOD data) → switched to absolute residuals
- Reliability threshold calibration using val-set quantiles (rejected 99.5% of OOD) → switched to absolute Tanimoto grid
- `assert` in production code → replaced with `ValueError`
- Missing `--input-file` error handling in CLI
- Feature ordering fragility (dict key ordering assumption) → explicit name-based lookup
- pKa test assertions only checking sign → added magnitude bounds

## Outcome

The review process is designed to catch the kinds of issues a thorough code reviewer would find — number mismatches, methodology gaps, code quality issues — before submission rather than after.
