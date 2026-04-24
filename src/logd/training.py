"""End-to-end training: OpenADMET ChEMBL35 → features → ensemble → calibration → artifact.

Training data: OpenADMET's curated ChEMBL35 LogD aggregated parquet (pinned SHA).
External eval: ExpansionRx challenge training data (pinned HF revision).
Cross-dataset deduplication by InChIKey prevents leakage.
"""

from __future__ import annotations

import json

import numpy as np
from rdkit.Chem import Mol

from logd.data import expansionrx, openadmet_chembl, splits
from logd.features import FeatureSpec, featurise_batch, mol_from_smiles, morgan_fp
from logd.models.baseline import train_ensemble
from logd.uncertainty import (
    ApplicabilityDomain,
    ConformalCalibrator,
    Reliability,
    calibrate_thresholds,
)
from logd.utils import get_logger, models_dir, reports_dir, set_seed

LOG = get_logger(__name__)


def _safe_morgan(smiles: str) -> np.ndarray:
    """morgan_fp with a None guard — training SMILES already passed validity checks."""
    mol: Mol | None = mol_from_smiles(smiles)
    if mol is None:
        raise ValueError(f"Expected valid SMILES in training set: {smiles}")
    return morgan_fp(mol)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    rho, _ = spearmanr(a, b)
    return float(rho)


def train_baseline(seed: int = 0, k: int = 5, alpha: float = 0.1, tune: bool = True) -> dict:
    """Full Day-1 pipeline. Returns metrics dict, writes artifacts to models/."""
    set_seed(seed)

    LOG.info("Loading OpenADMET ChEMBL35 logD data")
    df = openadmet_chembl.load()
    LOG.info("Loaded %d compounds", len(df))

    # Drop any training compound that also appears in the ExpansionRx external
    # test — prevents leakage if OpenADMET's curation happened to pull in any
    # overlapping InChIKeys.
    LOG.info("Loading ExpansionRx external test to deduplicate training")
    eval_df = expansionrx.load()
    eval_keys = set(eval_df["inchikey"])
    overlap = df["inchikey"].isin(eval_keys).sum()
    if overlap > 0:
        LOG.warning("Dropping %d training compounds that overlap ExpansionRx test", overlap)
        df = df[~df["inchikey"].isin(eval_keys)].reset_index(drop=True)
    else:
        LOG.info("No overlap between training and ExpansionRx test (clean)")

    LOG.info("Computing scaffold split")
    split = splits.scaffold_split(df["smiles"], seed=seed)
    LOG.info(
        "Split sizes: train=%d val=%d test=%d", len(split.train), len(split.val), len(split.test)
    )

    feature_spec = FeatureSpec()
    LOG.info("Featurising (dim=%d)", feature_spec.dim)
    X, mask = featurise_batch(df["smiles"].tolist(), feature_spec)
    if not mask.all():
        LOG.warning("%d compounds failed featurisation; dropping", (~mask).sum())
    y = df["logd"].to_numpy()[mask]

    # Re-index split masks to the kept rows.
    valid_indices = np.where(mask)[0]
    old_to_new = {old: new for new, old in enumerate(valid_indices)}

    def _remap(idx: np.ndarray) -> np.ndarray:
        return np.array([old_to_new[i] for i in idx if i in old_to_new], dtype=np.int64)

    train_idx = _remap(split.train)
    val_idx = _remap(split.val)
    test_idx = _remap(split.test)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Split val into two disjoint halves: val_select (early stopping + tuning)
    # and val_cal (conformal + threshold calibration). This ensures the
    # conformal coverage guarantee is not violated by model selection.
    rng = np.random.default_rng(seed)
    n_val = len(val_idx)
    perm = rng.permutation(n_val)
    n_select = int(n_val * 0.3)
    val_select_mask = np.zeros(n_val, dtype=bool)
    val_select_mask[perm[:n_select]] = True
    val_cal_mask = ~val_select_mask
    X_val_select, y_val_select = X_val[val_select_mask], y_val[val_select_mask]
    X_val_cal, y_val_cal = X_val[val_cal_mask], y_val[val_cal_mask]
    LOG.info("Val split: %d for model selection, %d for calibration", n_select, n_val - n_select)

    LOG.info("Training baseline ensemble (k=%d, tune=%s)", k, tune)
    model = train_ensemble(
        X_train,
        y_train,
        X_val_select,
        y_val_select,
        feature_spec=feature_spec,
        k=k,
        base_seed=seed,
        tune=tune,
    )

    y_cal_pred, y_cal_std = model.predict(X_val_cal)
    y_test_pred, y_test_std = model.predict(X_test)

    # Build applicability-domain from training fingerprints.
    train_fps = np.stack(
        [_safe_morgan(df["smiles"].iloc[i]) for i in valid_indices[train_idx]],
        axis=0,
    )
    ad = ApplicabilityDomain(train_fps=train_fps)

    # Nearest-neighbour similarity for cal + test.
    cal_fps = np.stack(
        [_safe_morgan(df["smiles"].iloc[i]) for i in valid_indices[val_idx[val_cal_mask]]],
        axis=0,
    )
    cal_nn = ad.nearest_similarity(cal_fps)

    # Conformal + thresholds calibrated on val_cal only (not used for model selection).
    conformal = ConformalCalibrator.fit(y_val_cal, y_cal_pred, y_cal_std, alpha=alpha)
    std_thr, tani_thr = calibrate_thresholds(y_val_cal, y_cal_pred, y_cal_std, cal_nn)
    reliability = Reliability(
        conformal=conformal, ad=ad, std_threshold=std_thr, tanimoto_threshold=tani_thr
    )

    # ExpansionRx external eval (already loaded above for dedup).
    LOG.info("Evaluating on ExpansionRx logD benchmark (%d compounds)", len(eval_df))
    X_oa, oa_mask = featurise_batch(eval_df["smiles"].tolist(), feature_spec)
    y_oa = eval_df["logd"].to_numpy()[oa_mask]
    oa_pred, oa_std = model.predict(X_oa)
    oa_fps = np.stack(
        [_safe_morgan(s) for s, m in zip(eval_df["smiles"].tolist(), oa_mask, strict=True) if m],
        axis=0,
    )
    oa_nn = ad.nearest_similarity(oa_fps)

    # Conformal coverage on ExpansionRx (OOD reality check).
    oa_conformal_coverage = float(np.mean(np.abs(y_oa - oa_pred) <= conformal.quantile))
    oa_reliable = reliability.flag(oa_std, oa_nn)
    LOG.info(
        "ExpansionRx conformal coverage: %.1f%% (target %d%%)",
        oa_conformal_coverage * 100,
        int((1 - alpha) * 100),
    )
    LOG.info(
        "ExpansionRx reliability: %d / %d (%.1f%%) flagged reliable",
        int(oa_reliable.sum()),
        len(oa_reliable),
        float(oa_reliable.mean()) * 100,
    )

    # Random-split comparison: train a separate ensemble on random split to
    # quantify the scaffold-vs-random gap. This demonstrates that scaffold
    # split is the harder (and more honest) evaluation.
    LOG.info("Training random-split baseline for comparison")
    rand_split = splits.random_split(len(y), seed=seed)
    rand_train_idx, rand_test_idx = rand_split.train, rand_split.test
    rand_model = train_ensemble(
        X[rand_train_idx],
        y[rand_train_idx],
        X[rand_split.val],
        y[rand_split.val],
        feature_spec=feature_spec,
        k=k,
        base_seed=seed + 100,
    )
    rand_test_pred, rand_test_std = rand_model.predict(X[rand_test_idx])
    random_test_metrics = {
        "rmse": _rmse(y[rand_test_idx], rand_test_pred),
        "mae": _mae(y[rand_test_idx], rand_test_pred),
        "pearson_r": _pearson(y[rand_test_idx], rand_test_pred),
        "spearman_std_vs_abs_err": _spearman(
            rand_test_std, np.abs(y[rand_test_idx] - rand_test_pred)
        ),
    }
    LOG.info(
        "Random-split test RMSE=%.3f vs scaffold test RMSE=%.3f",
        random_test_metrics["rmse"],
        _rmse(y_test, y_test_pred),
    )

    metrics = {
        "model": "baseline_lightgbm_ensemble",
        "ensemble_size": k,
        "n_train": len(y_train),
        "n_val": len(y_val),
        "n_val_select": int(val_select_mask.sum()),
        "n_val_cal": int(val_cal_mask.sum()),
        "n_test": len(y_test),
        "n_expansionrx": len(y_oa),
        "scaffold_test": {
            "rmse": _rmse(y_test, y_test_pred),
            "mae": _mae(y_test, y_test_pred),
            "pearson_r": _pearson(y_test, y_test_pred),
            "spearman_std_vs_abs_err": _spearman(y_test_std, np.abs(y_test - y_test_pred)),
        },
        "random_test": random_test_metrics,
        "expansionrx": {
            "rmse": _rmse(y_oa, oa_pred),
            "mae": _mae(y_oa, oa_pred),
            "pearson_r": _pearson(y_oa, oa_pred),
            "spearman_std_vs_abs_err": _spearman(oa_std, np.abs(y_oa - oa_pred)),
            "conformal_coverage": oa_conformal_coverage,
            "n_reliable": int(oa_reliable.sum()),
            "frac_reliable": float(oa_reliable.mean()),
            "rmse_reliable": (
                _rmse(y_oa[oa_reliable], oa_pred[oa_reliable])
                if oa_reliable.sum() > 0
                else float("nan")
            ),
        },
        "conformal": {"alpha": alpha, "quantile": conformal.quantile},
        "reliability_thresholds": {"std": std_thr, "tanimoto": tani_thr},
    }

    model_path = models_dir() / "baseline.joblib"
    reliability_path = models_dir() / "reliability.joblib"
    model.save(model_path)
    reliability.save(reliability_path)
    LOG.info("Saved model to %s", model_path)
    LOG.info("Saved reliability to %s", reliability_path)

    metrics_path = reports_dir() / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    LOG.info("Wrote metrics to %s", metrics_path)
    return metrics


def train_chemprop(
    seed: int = 0,
    k: int = 5,
    alpha: float = 0.1,
    max_epochs: int = 50,
    batch_size: int = 32,
) -> dict:
    """Full Chemprop D-MPNN training pipeline.

    Shares data prep with train_baseline() (same ChEMBL, same scaffold split,
    same OpenADMET external test) so results are directly comparable. The only
    difference is the model family.
    """
    set_seed(seed)

    LOG.info("Loading OpenADMET ChEMBL35 logD data")
    df = openadmet_chembl.load()
    LOG.info("Loaded %d compounds", len(df))

    # Drop any training compound that also appears in the ExpansionRx external
    # test — prevents leakage if OpenADMET's curation happened to pull in any
    # overlapping InChIKeys.
    LOG.info("Loading ExpansionRx external test to deduplicate training")
    eval_df = expansionrx.load()
    eval_keys = set(eval_df["inchikey"])
    overlap = df["inchikey"].isin(eval_keys).sum()
    if overlap > 0:
        LOG.warning("Dropping %d training compounds that overlap ExpansionRx test", overlap)
        df = df[~df["inchikey"].isin(eval_keys)].reset_index(drop=True)
    else:
        LOG.info("No overlap between training and ExpansionRx test (clean)")

    LOG.info("Computing scaffold split")
    split = splits.scaffold_split(df["smiles"], seed=seed)
    LOG.info(
        "Split sizes: train=%d val=%d test=%d", len(split.train), len(split.val), len(split.test)
    )

    smiles_all = df["smiles"].tolist()
    y_all = df["logd"].to_numpy()

    train_smiles = [smiles_all[i] for i in split.train]
    val_smiles = [smiles_all[i] for i in split.val]
    test_smiles = [smiles_all[i] for i in split.test]
    y_train = y_all[split.train]
    y_val = y_all[split.val]
    y_test = y_all[split.test]

    # Split val into val_select (early stopping) and val_cal (conformal + thresholds).
    rng = np.random.default_rng(seed)
    n_val = len(val_smiles)
    perm = rng.permutation(n_val)
    n_select = int(n_val * 0.3)
    val_select_idx = np.sort(perm[:n_select])
    val_cal_idx = np.sort(perm[n_select:])
    val_select_smiles = [val_smiles[i] for i in val_select_idx]
    val_cal_smiles = [val_smiles[i] for i in val_cal_idx]
    y_val_select = y_val[val_select_idx]
    y_val_cal = y_val[val_cal_idx]
    LOG.info("Val split: %d for model selection, %d for calibration", n_select, n_val - n_select)

    from logd.models.chemprop_wrap import ChempropModel

    LOG.info("Training Chemprop ensemble (k=%d, max_epochs=%d)", k, max_epochs)
    ckpt_dir = models_dir() / "chemprop"
    model = ChempropModel(checkpoint_dir=ckpt_dir, k=k)
    model.train(
        train_smiles=train_smiles,
        train_y=y_train,
        val_smiles=val_select_smiles,
        val_y=y_val_select,
        k=k,
        max_epochs=max_epochs,
        batch_size=batch_size,
        base_seed=seed,
    )

    cal_pred, cal_std, cal_mask = model.predict_smiles(val_cal_smiles)
    test_pred, test_std, test_mask = model.predict_smiles(test_smiles)

    # Applicability-domain from training fingerprints (reuses baseline approach).
    train_fps = np.stack([_safe_morgan(s) for s in train_smiles], axis=0)
    ad = ApplicabilityDomain(train_fps=train_fps)

    cal_fps = np.stack(
        [_safe_morgan(s) for s, m in zip(val_cal_smiles, cal_mask, strict=True) if m], axis=0
    )
    cal_nn = ad.nearest_similarity(cal_fps)

    # Conformal + thresholds calibrated on val_cal only (not used for model selection).
    conformal = ConformalCalibrator.fit(y_val_cal[cal_mask], cal_pred, cal_std, alpha=alpha)
    std_thr, tani_thr = calibrate_thresholds(y_val_cal[cal_mask], cal_pred, cal_std, cal_nn)
    reliability = Reliability(
        conformal=conformal, ad=ad, std_threshold=std_thr, tanimoto_threshold=tani_thr
    )

    # ExpansionRx external eval (eval_df already loaded + deduped above).
    LOG.info("Evaluating Chemprop on ExpansionRx logD benchmark (%d compounds)", len(eval_df))
    oa_smiles = eval_df["smiles"].tolist()
    oa_y = eval_df["logd"].to_numpy()
    oa_pred, oa_std, oa_mask = model.predict_smiles(oa_smiles)

    # Conformal coverage on ExpansionRx (OOD reality check).
    oa_fps = np.stack(
        [_safe_morgan(s) for s, m in zip(oa_smiles, oa_mask, strict=True) if m], axis=0
    )
    oa_nn = ad.nearest_similarity(oa_fps)
    oa_conformal_coverage = float(np.mean(np.abs(oa_y[oa_mask] - oa_pred) <= conformal.quantile))
    oa_reliable = reliability.flag(oa_std, oa_nn)
    LOG.info(
        "ExpansionRx conformal coverage: %.1f%% (target %d%%)",
        oa_conformal_coverage * 100,
        int((1 - alpha) * 100),
    )

    metrics = {
        "model": "chemprop_v2_dmpnn_ensemble",
        "ensemble_size": k,
        "max_epochs": max_epochs,
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
            "conformal_coverage": oa_conformal_coverage,
            "n_reliable": int(oa_reliable.sum()),
            "frac_reliable": float(oa_reliable.mean()),
            "rmse_reliable": (
                _rmse(oa_y[oa_mask][oa_reliable], oa_pred[oa_reliable])
                if oa_reliable.sum() > 0
                else float("nan")
            ),
        },
        "conformal": {"alpha": alpha, "quantile": conformal.quantile},
        "reliability_thresholds": {"std": std_thr, "tanimoto": tani_thr},
    }

    reliability_path = models_dir() / "chemprop_reliability.joblib"
    reliability.save(reliability_path)
    LOG.info("Saved Chemprop reliability to %s", reliability_path)

    metrics_path = reports_dir() / "chemprop_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    LOG.info("Wrote Chemprop metrics to %s", metrics_path)
    return metrics


if __name__ == "__main__":
    train_baseline()
