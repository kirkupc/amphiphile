"""Recalibrate Chemprop conformal/reliability artifacts without retraining.

Loads the existing Chemprop model checkpoints, predicts on the same val set
used during training (same scaffold split, same seed), and recalibrates the
conformal intervals and reliability thresholds using the current code
(absolute residuals, absolute Tanimoto grid, 88% precision target).

Usage:
    uv run python scripts/recalibrate_chemprop.py
"""

from __future__ import annotations

import json

import numpy as np

from logd.data import expansionrx, openadmet_chembl, splits
from logd.features import mol_from_smiles, morgan_fp
from logd.models.chemprop_wrap import ChempropModel
from logd.uncertainty import (
    ApplicabilityDomain,
    ConformalCalibrator,
    Reliability,
    calibrate_thresholds,
)
from logd.utils import get_logger, models_dir, reports_dir, set_seed

LOG = get_logger(__name__)


def _safe_morgan(smiles: str) -> np.ndarray:
    mol = mol_from_smiles(smiles)
    assert mol is not None, f"Expected valid SMILES: {smiles}"
    return morgan_fp(mol)


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    rho, _ = spearmanr(a, b)
    return float(rho)


def run(seed: int = 0, alpha: float = 0.1) -> dict:
    set_seed(seed)

    ckpt_dir = models_dir() / "chemprop"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Chemprop checkpoints not found at {ckpt_dir}")

    LOG.info("Loading Chemprop model from %s", ckpt_dir)
    model = ChempropModel.load(ckpt_dir)

    LOG.info("Loading data + scaffold split")
    df = openadmet_chembl.load()
    eval_df = expansionrx.load()
    eval_keys = set(eval_df["inchikey"])
    overlap = df["inchikey"].isin(eval_keys).sum()
    if overlap > 0:
        df = df[~df["inchikey"].isin(eval_keys)].reset_index(drop=True)
    split = splits.scaffold_split(df["smiles"], seed=seed)

    smiles_all = df["smiles"].tolist()
    y_all = df["logd"].to_numpy()

    train_smiles = [smiles_all[i] for i in split.train]
    val_smiles = [smiles_all[i] for i in split.val]
    test_smiles = [smiles_all[i] for i in split.test]
    y_val = y_all[split.val]
    y_test = y_all[split.test]

    LOG.info("Predicting on val (%d) and test (%d)", len(val_smiles), len(test_smiles))
    val_pred, val_std, val_mask = model.predict_smiles(val_smiles)
    test_pred, test_std, test_mask = model.predict_smiles(test_smiles)

    LOG.info("Building applicability domain from %d training compounds", len(train_smiles))
    train_fps = np.stack([_safe_morgan(s) for s in train_smiles], axis=0)
    ad = ApplicabilityDomain(train_fps=train_fps)

    val_fps = np.stack(
        [_safe_morgan(s) for s, m in zip(val_smiles, val_mask, strict=True) if m], axis=0
    )
    val_nn = ad.nearest_similarity(val_fps)

    LOG.info("Recalibrating conformal + thresholds")
    conformal = ConformalCalibrator.fit(y_val[val_mask], val_pred, val_std, alpha=alpha)
    std_thr, tani_thr = calibrate_thresholds(y_val[val_mask], val_pred, val_std, val_nn)
    reliability = Reliability(
        conformal=conformal, ad=ad, std_threshold=std_thr, tanimoto_threshold=tani_thr
    )

    LOG.info("Evaluating on ExpansionRx (%d compounds)", len(eval_df))
    oa_smiles = eval_df["smiles"].tolist()
    oa_y = eval_df["logd"].to_numpy()
    oa_pred, oa_std, oa_mask = model.predict_smiles(oa_smiles)

    metrics = {
        "model": "chemprop_v2_dmpnn_ensemble",
        "ensemble_size": model.k,
        "n_train": len(train_smiles),
        "n_val": len(val_smiles),
        "n_test": len(test_smiles),
        "n_expansionrx": int(oa_mask.sum()),
        "scaffold_test": {
            "rmse": _rmse(y_test[test_mask], test_pred),
            "mae": _mae(y_test[test_mask], test_pred),
            "pearson_r": _pearson(y_test[test_mask], test_pred),
            "spearman_std_vs_abs_err": _spearman(test_std, np.abs(y_test[test_mask] - test_pred)),
        },
        "expansionrx": {
            "rmse": _rmse(oa_y[oa_mask], oa_pred),
            "mae": _mae(oa_y[oa_mask], oa_pred),
            "pearson_r": _pearson(oa_y[oa_mask], oa_pred),
            "spearman_std_vs_abs_err": _spearman(oa_std, np.abs(oa_y[oa_mask] - oa_pred)),
        },
        "conformal": {"alpha": alpha, "quantile": conformal.quantile},
        "reliability_thresholds": {"std": std_thr, "tanimoto": tani_thr},
    }

    reliability_path = models_dir() / "chemprop_reliability.joblib"
    reliability.save(reliability_path)
    LOG.info("Saved recalibrated reliability to %s", reliability_path)

    metrics_path = reports_dir() / "chemprop_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    LOG.info("Wrote Chemprop metrics to %s", metrics_path)
    LOG.info(
        "Conformal quantile: %.3f (was ~5.16 Mondrian), thresholds: std=%.3f tani=%.3f",
        conformal.quantile,
        std_thr,
        tani_thr,
    )
    return metrics


if __name__ == "__main__":
    run()
