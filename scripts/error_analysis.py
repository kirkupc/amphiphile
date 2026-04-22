"""Error analysis — the writeup centerpiece.

Identifies the N compounds with the worst |prediction - truth| on the
ExpansionRx external test set, renders their 2D structures, and writes a
markdown report tying each failure to the uncertainty channels that did (or
did not) fire.

The narrative question: does high uncertainty predict high error? Split the
worst cases into:
  - loud failures (high ensemble_std OR low Tanimoto OR wide conformal interval)
    → model knew it was uncertain, reliability flag would have caught these
  - silent failures (all three uncertainty channels said "confident")
    → these are the real problem; they reveal model blind spots

Usage:
    uv run python scripts/error_analysis.py --top 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

from logd.data import expansionrx
from logd.features import featurise_batch, mol_from_smiles, morgan_fp
from logd.inference import load_model
from logd.utils import get_logger, reports_dir

LOG = get_logger(__name__)


def _rationalise(row: dict) -> str:
    """Heuristic chemistry rationale for a failure case.

    Pattern-matches on structural features known to cause logD prediction
    failures: ionisable groups (amidines, guanidines, quaternary N),
    size extremes, unusual atoms, flexibility, and AD/uncertainty signals.
    """
    mol = Chem.MolFromSmiles(row["smiles"])
    if mol is None:
        return "Unparseable at analysis time."

    reasons: list[str] = []

    # Size
    n_heavy = mol.GetNumHeavyAtoms()
    if n_heavy > 50:
        reasons.append(
            f"large molecule ({n_heavy} heavy atoms) — outside typical training mass range"
        )
    elif n_heavy < 6:
        reasons.append(f"very small molecule ({n_heavy} heavy atoms) — sparse graph features")

    # Charge / ionizability proxy
    has_cooh = mol.HasSubstructMatch(Chem.MolFromSmarts("C(=O)[OH]"))
    has_amine = mol.HasSubstructMatch(Chem.MolFromSmarts("[NX3;H2,H1;!$(NC=O)]"))
    has_quat_n = mol.HasSubstructMatch(Chem.MolFromSmarts("[N+;!$([N+]-[O-])]"))
    has_amidine = mol.HasSubstructMatch(Chem.MolFromSmarts("[NX3][CX3]=[NX2]"))
    has_guanidine = mol.HasSubstructMatch(Chem.MolFromSmarts("[NX3][CX3](=[NX2])[NX3]"))
    pyridine_n_pat = Chem.MolFromSmarts("[nR1;H0]")
    aminopyrimidine_pat = Chem.MolFromSmarts("[NX3][cR1]1[nR1][cR1][nR1][cR1][cR1]1")
    n_pyridine_n = len(mol.GetSubstructMatches(pyridine_n_pat)) if pyridine_n_pat else 0
    has_aminopyrimidine = (
        bool(mol.HasSubstructMatch(aminopyrimidine_pat)) if aminopyrimidine_pat else False
    )

    if has_quat_n:
        reasons.append(
            "permanent positive charge (quaternary nitrogen) — logD is dominated by the ionic species "
            "at pH 7.4; the model counts this group but lacks the pKa magnitude to quantify the shift"
        )
    if has_guanidine:
        reasons.append(
            "guanidine group (pKa ~12-13, fully protonated at pH 7.4) — the cationic form dominates "
            "partitioning; the model counts this group but lacks pKa-derived features to quantify the logD shift"
        )
    elif has_amidine:
        reasons.append(
            "amidine group (pKa ~10-11, protonated at pH 7.4) — the cationic form dominates "
            "partitioning; the model counts this group but lacks pKa-derived features to quantify the logD shift"
        )
    if has_aminopyrimidine:
        reasons.append(
            "aminopyrimidine group (pKa ~4-6) — weakly basic heterocyclic nitrogen, "
            "individually not strongly ionised at pH 7.4 but multiple instances shift the protonation equilibrium"
        )
    elif n_pyridine_n > 0:
        qualifier = f"multiple ({n_pyridine_n}) " if n_pyridine_n > 1 else ""
        reasons.append(
            f"{qualifier}aromatic basic nitrogen(s) (pKa ~3-5) — weakly basic heterocyclic nitrogens, "
            "individually not strongly ionised at pH 7.4 but multiple instances shift the protonation equilibrium"
        )
    if has_cooh and has_amine:
        reasons.append(
            "zwitterionic potential (both carboxylic acid and basic amine) — pH-dependent partitioning"
        )
    elif has_cooh:
        reasons.append(
            "carboxylic acid — ionises at pH 7.4, models assuming neutral form under-predict logD"
        )
    elif has_amine and not (has_amidine or has_guanidine or has_quat_n):
        reasons.append(
            "basic amine (pKa ~9-10, protonated at pH 7.4) — the model counts this group "
            "but lacks pKa-derived features to quantify the logD shift from ionisation"
        )

    # Unusual atoms
    atoms = {a.GetSymbol() for a in mol.GetAtoms()}
    unusual = atoms - {"C", "H", "N", "O", "F", "Cl", "Br", "I", "S", "P"}
    if unusual:
        reasons.append(f"contains atypical atoms: {sorted(unusual)}")

    # Rotatable bonds as flexibility proxy
    from rdkit.Chem import Descriptors

    n_rot = int(Descriptors.NumRotatableBonds(mol))  # type: ignore[attr-defined]
    if n_rot > 12:
        reasons.append(f"highly flexible ({n_rot} rotatable bonds) — conformationally demanding")

    # AD / uncertainty flags
    if row["nn_tanimoto"] < 0.3:
        reasons.append(
            f"far from training distribution (max Tanimoto to train = {row['nn_tanimoto']:.2f}) — "
            "applicability-domain check correctly flags this as extrapolation"
        )
    if row["ensemble_std"] > row["median_std"] * 1.5:
        reasons.append(
            f"high ensemble disagreement (std={row['ensemble_std']:.2f} vs median {row['median_std']:.2f}) — "
            "model internals knew this was uncertain"
        )

    if not reasons:
        reasons.append(
            "no obvious structural explanation from heuristic analysis — "
            "manual inspection of the training distribution around this scaffold is recommended"
        )

    return ". ".join(r.capitalize() if i == 0 else r for i, r in enumerate(reasons)) + "."


def run(top: int = 10, out_dir: Path | None = None) -> None:
    out_dir = out_dir or (reports_dir() / "error_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("Loading model + ExpansionRx test set")
    model = load_model()
    eval_df = expansionrx.load()
    smiles = eval_df["smiles"].tolist()
    y_true = eval_df["logd"].to_numpy()

    LOG.info("Featurising %d test compounds", len(smiles))
    assert model.feature_spec is not None and model.baseline is not None
    X, mask = featurise_batch(smiles, model.feature_spec)
    y_pred, y_std = model.baseline.predict(X)
    valid_smiles = [s for s, m in zip(smiles, mask, strict=True) if m]
    fps = np.stack([morgan_fp(mol_from_smiles(s)) for s in valid_smiles], axis=0)  # type: ignore[arg-type]
    nn = model.reliability.ad.nearest_similarity(fps)

    # Conformal interval half-width (constant after switch to absolute residuals)
    conformal = model.reliability.conformal
    half_width = np.full_like(y_std, conformal.quantile)

    err = np.abs(y_true[mask] - y_pred)
    median_std = float(np.median(y_std))

    df = (
        pd.DataFrame(
            {
                "smiles": valid_smiles,
                "true_logd": y_true[mask],
                "pred_logd": y_pred,
                "abs_error": err,
                "ensemble_std": y_std,
                "nn_tanimoto": nn,
                "conformal_halfwidth": half_width,
                "median_std": median_std,
            }
        )
        .sort_values("abs_error", ascending=False)
        .head(top)
        .reset_index(drop=True)
    )

    # Render structures
    LOG.info("Rendering %d structures", len(df))
    for i, row in df.iterrows():
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is None:
            continue
        img = Draw.MolToImage(mol, size=(500, 400))
        img.save(out_dir / f"worst_{i + 1:02d}.png")

    # Uncertainty-channel classification
    std_thr = model.reliability.std_threshold
    tani_thr = model.reliability.tanimoto_threshold
    df["would_be_flagged_unreliable"] = (df["ensemble_std"] > std_thr) | (
        df["nn_tanimoto"] < tani_thr
    )

    loud = int(df["would_be_flagged_unreliable"].sum())
    silent = len(df) - loud

    # Full-dataset reliability coverage
    all_reliable = (y_std <= std_thr) & (nn >= tani_thr)
    n_reliable = int(all_reliable.sum())
    n_total = len(y_std)
    reliable_err = err[all_reliable]
    reliable_precision = float((reliable_err <= 1.0).mean()) if n_reliable > 0 else float("nan")
    full_rmse = float(np.sqrt(np.mean(err**2)))
    reliable_rmse = float(np.sqrt(np.mean(reliable_err**2))) if n_reliable > 0 else float("nan")

    # Markdown report
    report = [
        f"# Error analysis — worst {top} predictions on ExpansionRx\n",
        "## Reliability coverage (full test set)\n",
        f"- Total valid compounds: {n_total}",
        f"- Flagged reliable: {n_reliable} ({100 * n_reliable / n_total:.1f}%)",
        f"- Precision on reliable subset (|error| <= 1.0 log unit): {reliable_precision:.1%}",
        f"- RMSE overall: {full_rmse:.3f}, RMSE on reliable subset: {reliable_rmse:.3f}\n",
        f"## Worst-{top} analysis\n",
        f"- Median ensemble std across full test: {median_std:.3f} log units",
        f"- Reliability thresholds: std <= {std_thr:.3f}, Tanimoto >= {tani_thr:.3f}",
        f"- Of the worst {top}: **{loud}/{top} would have been flagged unreliable** (loud), **{silent} silent failures**\n",
        "## Per-compound breakdown\n",
    ]
    for i, row in df.iterrows():
        report.append(f"### {i + 1}. abs_error = {row['abs_error']:.2f} log units\n")
        report.append(f"![structure](worst_{i + 1:02d}.png)\n")
        report.append(
            f"- SMILES: `{row['smiles']}`\n"
            f"- True: {row['true_logd']:.2f}, Predicted: {row['pred_logd']:.2f}\n"
            f"- Ensemble std: {row['ensemble_std']:.3f}, "
            f"Nearest-training Tanimoto: {row['nn_tanimoto']:.3f}\n"
            f"- Conformal ±{row['conformal_halfwidth']:.2f} (covers truth: "
            f"{'yes' if row['abs_error'] <= row['conformal_halfwidth'] else 'NO'})\n"
            f"- Would reliability flag fire: "
            f"{'YES (loud failure — system knew)' if row['would_be_flagged_unreliable'] else 'NO (silent failure — blind spot)'}\n"
        )
        report.append(f"**Rationale:** {_rationalise(row.to_dict())}\n")

    (out_dir / "report.md").write_text("\n".join(report))
    df.to_csv(out_dir / "worst_compounds.csv", index=False)

    LOG.info("Wrote %d structures + report.md + worst_compounds.csv to %s", len(df), out_dir)
    LOG.info("Summary: %d loud failures, %d silent failures", loud, silent)

    summary = {
        "top": top,
        "loud_failures": loud,
        "silent_failures": silent,
        "median_ensemble_std": median_std,
        "std_threshold": std_thr,
        "tanimoto_threshold": tani_thr,
        "reliability_coverage": {
            "n_total": n_total,
            "n_reliable": n_reliable,
            "fraction_reliable": n_reliable / n_total,
            "precision_within_1_log": reliable_precision,
            "rmse_overall": full_rmse,
            "rmse_reliable_subset": reliable_rmse,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()
    run(top=args.top)
