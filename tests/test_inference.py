"""Tests for the public inference API using a tiny in-memory trained model.

We don't want tests to require ChEMBL downloads, so these build a mini
BaselineModel + Reliability from a small SMILES set and validate the API
contract (invalid-SMILES handling, order preservation, round-trip).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from logd.features import FeatureSpec, featurise_batch, mol_from_smiles, morgan_fp
from logd.inference import LoadedModel, predict
from logd.models.baseline import BaselineModel, train_ensemble
from logd.uncertainty import ApplicabilityDomain, ConformalCalibrator, Reliability


TRAIN_SMILES = [
    "CCO", "CCC", "CCCC", "CCCCC", "CCCCCC",
    "c1ccccc1", "c1ccc(C)cc1", "c1ccc(CC)cc1", "c1ccc(CCC)cc1",
    "CC(=O)O", "CC(=O)OC", "CC(=O)N", "C(=O)O",
    "c1ccncc1", "c1ccncc1C", "C1CCCCC1", "C1CCCCC1C", "C1CCCCC1CC",
    "CCN", "CCCN", "CCCCN",
    "OCC(O)CO", "OCC(O)C", "OCCO",
]
TRAIN_Y = np.linspace(-1.5, 3.5, len(TRAIN_SMILES)).astype(np.float32)


@pytest.fixture(scope="module")
def tiny_model() -> LoadedModel:
    spec = FeatureSpec()
    X, mask = featurise_batch(TRAIN_SMILES, spec)
    assert mask.all()
    y = TRAIN_Y
    # train/val split for early stopping
    n_val = 5
    model = train_ensemble(
        X_train=X[:-n_val],
        y_train=y[:-n_val],
        X_val=X[-n_val:],
        y_val=y[-n_val:],
        feature_spec=spec,
        k=2,
        base_seed=0,
    )
    train_fps = np.stack([morgan_fp(mol_from_smiles(s)) for s in TRAIN_SMILES[:-n_val]], axis=0)
    ad = ApplicabilityDomain(train_fps=train_fps)
    y_val_pred, y_val_std = model.predict(X[-n_val:])
    conformal = ConformalCalibrator.fit(y[-n_val:], y_val_pred, y_val_std, alpha=0.1)
    reliability = Reliability(
        conformal=conformal,
        ad=ad,
        std_threshold=float(np.quantile(y_val_std, 0.8)),
        tanimoto_threshold=0.2,
    )
    return LoadedModel(baseline=model, reliability=reliability, feature_spec=spec)


def test_predict_returns_one_per_input(tiny_model: LoadedModel) -> None:
    out = predict(["CCO", "CCC", "invalid"], model=tiny_model)
    assert len(out) == 3
    assert [r.smiles for r in out] == ["CCO", "CCC", "invalid"]


def test_predict_handles_invalid_smiles(tiny_model: LoadedModel) -> None:
    out = predict(["CCO", "not_a_smiles"], model=tiny_model)
    assert out[0].error is None
    assert out[0].predicted_logd is not None
    assert out[1].error == "invalid_smiles"
    assert out[1].predicted_logd is None
    assert out[1].uncertainty is None
    assert out[1].reliable is False


def test_predict_canonical_equivalence(tiny_model: LoadedModel) -> None:
    a = predict(["CCO"], model=tiny_model)[0]
    b = predict(["OCC"], model=tiny_model)[0]
    assert a.predicted_logd == pytest.approx(b.predicted_logd)
    assert a.uncertainty == pytest.approx(b.uncertainty)


def test_predict_batch_matches_single(tiny_model: LoadedModel) -> None:
    single = [predict([s], model=tiny_model)[0] for s in ["CCO", "c1ccccc1", "CC(=O)O"]]
    batch = predict(["CCO", "c1ccccc1", "CC(=O)O"], model=tiny_model)
    for a, b in zip(single, batch):
        assert a.predicted_logd == pytest.approx(b.predicted_logd)


def test_baseline_save_load_roundtrip(tiny_model: LoadedModel) -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "m.joblib"
        tiny_model.baseline.save(p)
        loaded = BaselineModel.load(p)
        X, _ = featurise_batch(["CCO", "c1ccccc1"], tiny_model.feature_spec)
        m1, s1 = tiny_model.baseline.predict(X)
        m2, s2 = loaded.predict(X)
        np.testing.assert_allclose(m1, m2)
        np.testing.assert_allclose(s1, s2)


def test_predict_empty_list(tiny_model: LoadedModel) -> None:
    assert predict([], model=tiny_model) == []


def test_predict_preserves_order_with_mixed_validity(tiny_model: LoadedModel) -> None:
    inputs = ["CCO", "bad1", "c1ccccc1", "bad2", "CC(=O)O"]
    out = predict(inputs, model=tiny_model)
    assert [r.smiles for r in out] == inputs
    assert [r.error is None for r in out] == [True, False, True, False, True]
