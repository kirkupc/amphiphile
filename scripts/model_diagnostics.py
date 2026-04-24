"""Model diagnostics: bias analysis, confidence curves, and feature importance.

Loads existing baseline artifacts and ExpansionRx data, computes:
1. Prediction bias — mean signed error overall and by logD range
2. Confidence curves — RMSE binned by ensemble std quintile and Tanimoto bins
3. Feature importance — top features from LightGBM (gain-based)

Results saved to reports/diagnostics.json and reports/confidence_curve.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from logd.data import expansionrx
from logd.features import FeatureSpec, featurise_batch, mol_from_smiles, morgan_fp
from logd.models.baseline import BaselineModel
from logd.uncertainty import Reliability
from logd.utils import get_logger, models_dir, reports_dir

LOG = get_logger(__name__)


def _safe_morgan(smiles: str) -> np.ndarray:
    mol = mol_from_smiles(smiles)
    if mol is None:
        raise ValueError(f"Expected valid SMILES: {smiles}")
    return morgan_fp(mol)


def _feature_names(spec: FeatureSpec) -> list[str]:
    from logd.features import _IONISABLE_SMARTS, DESCRIPTOR_NAMES, PKA_FEATURE_NAMES

    names: list[str] = []
    if spec.use_descriptors:
        names.extend(DESCRIPTOR_NAMES)
    if spec.use_ionisable:
        names.extend(name for name, _ in _IONISABLE_SMARTS)
    if spec.use_pka:
        names.extend(PKA_FEATURE_NAMES)
    if spec.use_morgan:
        names.extend(f"morgan_{i}" for i in range(spec.morgan_bits))
    return names


def run(output_dir: Path | None = None) -> dict:
    if output_dir is None:
        output_dir = reports_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir() / "baseline.joblib"
    rel_path = models_dir() / "reliability.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Baseline model not found: {model_path}")
    if not rel_path.exists():
        raise FileNotFoundError(f"Reliability artifacts not found: {rel_path}")

    LOG.info("Loading baseline model and reliability artifacts")
    model = BaselineModel.load(model_path)
    reliability = Reliability.load(rel_path)

    LOG.info("Loading ExpansionRx data")
    eval_df = expansionrx.load()
    smiles = eval_df["smiles"].tolist()
    y_true = eval_df["logd"].to_numpy()

    LOG.info("Featurising %d compounds", len(smiles))
    X, mask = featurise_batch(smiles, model.feature_spec)
    y_true = y_true[mask]
    valid_smiles = [s for s, m in zip(smiles, mask, strict=True) if m]

    LOG.info("Predicting")
    pred, std = model.predict(X)
    errors = y_true - pred
    abs_errors = np.abs(errors)

    fps = np.stack([_safe_morgan(s) for s in valid_smiles], axis=0)
    nn_sim = reliability.ad.nearest_similarity(fps)

    results: dict = {}

    # --- 1. Prediction bias analysis ---
    LOG.info("Computing prediction bias analysis")
    mse_signed = float(np.mean(errors))
    median_signed = float(np.median(errors))
    frac_over = float(np.mean(errors < 0))  # pred > true = overprediction = negative error

    bins: list[tuple[float, float]] = [(-5, 0), (0, 1), (1, 2), (2, 3), (3, 5), (5, 10)]
    bias_by_range: list[dict] = []
    for lo, hi in bins:
        in_bin = (y_true >= lo) & (y_true < hi)
        n = int(in_bin.sum())
        if n == 0:
            continue
        bias_by_range.append(
            {
                "range": f"[{lo}, {hi})",
                "n": n,
                "mean_signed_error": round(float(np.mean(errors[in_bin])), 4),
                "rmse": round(float(np.sqrt(np.mean(errors[in_bin] ** 2))), 4),
                "frac_overpredicted": round(float(np.mean(errors[in_bin] < 0)), 4),
            }
        )

    results["bias"] = {
        "mean_signed_error": round(mse_signed, 4),
        "median_signed_error": round(median_signed, 4),
        "frac_overpredicted": round(frac_over, 4),
        "by_logd_range": bias_by_range,
    }
    LOG.info("  Mean signed error: %.4f, median: %.4f", mse_signed, median_signed)

    # --- 2. Confidence curves ---
    LOG.info("Computing confidence curves")

    # RMSE by ensemble std quintile
    std_quantiles = np.quantile(std, [0.2, 0.4, 0.6, 0.8])
    std_bins = [
        ("Q1 (lowest std)", std <= std_quantiles[0]),
        ("Q2", (std > std_quantiles[0]) & (std <= std_quantiles[1])),
        ("Q3", (std > std_quantiles[1]) & (std <= std_quantiles[2])),
        ("Q4", (std > std_quantiles[2]) & (std <= std_quantiles[3])),
        ("Q5 (highest std)", std > std_quantiles[3]),
    ]
    std_curve: list[dict] = []
    for label, sel in std_bins:
        n = int(sel.sum())
        if n == 0:
            continue
        std_curve.append(
            {
                "bin": label,
                "n": n,
                "mean_std": round(float(np.mean(std[sel])), 4),
                "rmse": round(float(np.sqrt(np.mean(abs_errors[sel] ** 2))), 4),
                "mae": round(float(np.mean(abs_errors[sel])), 4),
            }
        )

    # RMSE by Tanimoto bins
    tani_edges: list[float] = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 1.01]
    tani_curve: list[dict] = []
    for i in range(len(tani_edges) - 1):
        lo, hi = tani_edges[i], tani_edges[i + 1]
        sel = (nn_sim >= lo) & (nn_sim < hi)
        n = int(sel.sum())
        if n == 0:
            continue
        tani_curve.append(
            {
                "bin": f"[{lo:.1f}, {hi:.1f})" if hi <= 1.0 else f"[{lo:.1f}, 1.0]",
                "n": n,
                "mean_tanimoto": round(float(np.mean(nn_sim[sel])), 4),
                "rmse": round(float(np.sqrt(np.mean(abs_errors[sel] ** 2))), 4),
                "mae": round(float(np.mean(abs_errors[sel])), 4),
            }
        )

    # Conformal coverage per Tanimoto bin
    conformal_quantile = reliability.conformal.quantile
    conformal_by_tani: list[dict] = []
    for i in range(len(tani_edges) - 1):
        lo, hi = tani_edges[i], tani_edges[i + 1]
        sel = (nn_sim >= lo) & (nn_sim < hi)
        n = int(sel.sum())
        if n == 0:
            continue
        covered = float(np.mean(abs_errors[sel] <= conformal_quantile))
        conformal_by_tani.append(
            {
                "bin": f"[{lo:.1f}, {hi:.1f})" if hi <= 1.0 else f"[{lo:.1f}, 1.0]",
                "n": n,
                "conformal_coverage": round(covered, 4),
            }
        )

    results["confidence_curves"] = {
        "by_ensemble_std_quintile": std_curve,
        "by_tanimoto_bin": tani_curve,
        "conformal_coverage_by_tanimoto": conformal_by_tani,
    }

    # --- 3. Feature importance ---
    LOG.info("Computing feature importance (gain-based)")
    feat_names = _feature_names(model.feature_spec)
    total_importance = np.zeros(len(feat_names), dtype=np.float64)
    for booster in model.boosters:
        imp = booster.feature_importance(importance_type="gain")
        total_importance += imp
    total_importance /= len(model.boosters)

    top_idx = np.argsort(total_importance)[::-1][:30]
    feature_importance: list[dict] = []
    for idx in top_idx:
        if total_importance[idx] <= 0:
            break
        feature_importance.append(
            {
                "feature": feat_names[idx],
                "importance_gain": round(float(total_importance[idx]), 2),
            }
        )
    results["feature_importance_top30"] = feature_importance

    # --- Save JSON ---
    json_path = output_dir / "diagnostics.json"
    json_path.write_text(json.dumps(results, indent=2))
    LOG.info("Wrote diagnostics to %s", json_path)

    # --- Plot confidence curves ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Std quintile curve
    ax = axes[0]
    labels = [d["bin"] for d in std_curve]
    rmses = [d["rmse"] for d in std_curve]
    counts = [d["n"] for d in std_curve]
    x = range(len(labels))
    bars = ax.bar(x, rmses, color="#4C72B0", alpha=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("RMSE (log units)")
    ax.set_title("RMSE by ensemble std quintile (ExpansionRx)")
    for bar, n in zip(bars, counts, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"n={n}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    # Tanimoto curve
    ax = axes[1]
    labels = [d["bin"] for d in tani_curve]
    rmses = [d["rmse"] for d in tani_curve]
    counts = [d["n"] for d in tani_curve]
    x = range(len(labels))
    bars = ax.bar(x, rmses, color="#DD8452", alpha=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("RMSE (log units)")
    ax.set_title("RMSE by Tanimoto similarity (ExpansionRx)")
    for bar, n in zip(bars, counts, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"n={n}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    fig.tight_layout()
    plot_path = output_dir / "confidence_curves.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    LOG.info("Saved confidence curve plot to %s", plot_path)

    return results


if __name__ == "__main__":
    run()
