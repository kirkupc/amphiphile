"""Smoke tests for the FastAPI endpoint.

Uses FastAPI's TestClient with the lifespan manager patched to inject a tiny
in-memory model — no disk-loaded artifacts needed, no network, fast.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from logd import api as api_module
from logd.features import FeatureSpec, featurise_batch, mol_from_smiles, morgan_fp
from logd.inference import LoadedModel
from logd.models.baseline import train_ensemble
from logd.uncertainty import ApplicabilityDomain, ConformalCalibrator, Reliability


TRAIN_SMILES = [
    "CCO", "CCC", "CCCC", "CCCCC", "CCCCCC",
    "c1ccccc1", "c1ccc(C)cc1", "c1ccc(CC)cc1",
    "CC(=O)O", "CC(=O)OC", "CC(=O)N",
    "c1ccncc1", "C1CCCCC1", "C1CCCCC1C",
    "CCN", "CCCN", "OCC(O)CO", "OCCO", "CCS", "CCSC",
]
TRAIN_Y = np.linspace(-1.5, 3.5, len(TRAIN_SMILES)).astype(np.float32)


@pytest.fixture(scope="module")
def _tiny_model() -> LoadedModel:
    spec = FeatureSpec()
    X, mask = featurise_batch(TRAIN_SMILES, spec)
    n_val = 4
    model = train_ensemble(
        X_train=X[:-n_val],
        y_train=TRAIN_Y[:-n_val],
        X_val=X[-n_val:],
        y_val=TRAIN_Y[-n_val:],
        feature_spec=spec,
        k=2,
        base_seed=0,
    )
    train_fps = np.stack([morgan_fp(mol_from_smiles(s)) for s in TRAIN_SMILES[:-n_val]], axis=0)
    ad = ApplicabilityDomain(train_fps=train_fps)
    y_val_pred, y_val_std = model.predict(X[-n_val:])
    conformal = ConformalCalibrator.fit(TRAIN_Y[-n_val:], y_val_pred, y_val_std, alpha=0.1)
    reliability = Reliability(
        conformal=conformal, ad=ad, std_threshold=1.0, tanimoto_threshold=0.1
    )
    return LoadedModel(baseline=model, reliability=reliability, feature_spec=spec)


@pytest.fixture()
def client(_tiny_model: LoadedModel) -> TestClient:
    """TestClient with _model patched in directly; bypasses lifespan load."""
    api_module._model = _tiny_model
    with TestClient(api_module.app) as c:
        # Re-patch after lifespan since startup may have nulled it if artifacts missing.
        api_module._model = _tiny_model
        yield c
    api_module._model = None


def test_health_returns_ok_when_model_loaded(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_happy_path(client: TestClient) -> None:
    r = client.post("/predict", json={"smiles": ["CCO", "c1ccccc1"]})
    assert r.status_code == 200
    body = r.json()
    preds = body["predictions"]
    assert len(preds) == 2
    assert all(p["predicted_logd"] is not None for p in preds)
    assert all(p["error"] is None for p in preds)


def test_predict_invalid_smiles_flagged(client: TestClient) -> None:
    r = client.post("/predict", json={"smiles": ["CCO", "not_a_smiles", ""]})
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert preds[0]["error"] is None
    assert preds[1]["error"] == "invalid_smiles"
    assert preds[1]["predicted_logd"] is None
    assert preds[2]["error"] == "invalid_smiles"


def test_predict_empty_list_rejected(client: TestClient) -> None:
    r = client.post("/predict", json={"smiles": []})
    assert r.status_code == 422


def test_predict_too_many_smiles_rejected(client: TestClient) -> None:
    many = ["CCO"] * (api_module.MAX_SMILES_PER_REQUEST + 1)
    r = client.post("/predict", json={"smiles": many})
    assert r.status_code == 413


def test_predict_order_preserved(client: TestClient) -> None:
    inputs = ["CCO", "invalid", "c1ccccc1", "nope"]
    r = client.post("/predict", json={"smiles": inputs})
    preds = r.json()["predictions"]
    assert [p["smiles"] for p in preds] == inputs


def test_predict_503_when_model_missing() -> None:
    api_module._model = None
    with TestClient(api_module.app) as c:
        api_module._model = None  # override lifespan
        r = c.post("/predict", json={"smiles": ["CCO"]})
        assert r.status_code == 503
