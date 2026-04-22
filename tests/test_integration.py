"""End-to-end integration tests that load real trained model artifacts.

These tests verify the full stack: FastAPI endpoint → inference pipeline →
trained model → uncertainty → response schema. Skipped if model artifacts
are not present (e.g. fresh clone without `logd train`).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from logd import api as api_module
from logd.inference import load_model

_BASELINE_PATH = Path(__file__).resolve().parent.parent / "models" / "baseline.joblib"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not _BASELINE_PATH.exists(), reason="trained model artifacts not present"),
]


@pytest.fixture(scope="module")
def real_model() -> api_module.LoadedModel:
    return load_model(model_type="baseline")


@pytest.fixture()
def client(real_model: api_module.LoadedModel) -> TestClient:
    api_module._model = real_model
    with TestClient(api_module.app) as c:
        api_module._model = real_model
        yield c
    api_module._model = None


def test_health_ok(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert "model_version" in body


def test_predict_full_schema(client: TestClient) -> None:
    """Verify every response field is populated correctly for valid SMILES."""
    r = client.post("/predict", json={"smiles": ["CCO", "c1ccccc1"]})
    assert r.status_code == 200
    body = r.json()
    assert "model_version" in body
    assert len(body["predictions"]) == 2
    for p in body["predictions"]:
        assert isinstance(p["smiles"], str)
        assert isinstance(p["predicted_logd"], float)
        assert isinstance(p["uncertainty"], float)
        assert isinstance(p["reliable"], bool)
        assert p["error"] is None


def test_predict_mixed_valid_invalid(client: TestClient) -> None:
    """Valid + invalid + empty SMILES in one batch."""
    smiles = ["CCO", "not_valid", "", "c1ccccc1"]
    r = client.post("/predict", json={"smiles": smiles})
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert len(preds) == 4
    assert [p["smiles"] for p in preds] == smiles
    assert preds[0]["predicted_logd"] is not None
    assert preds[0]["error"] is None
    assert preds[1]["error"] == "invalid_smiles"
    assert preds[1]["predicted_logd"] is None
    assert preds[2]["error"] == "invalid_smiles"
    assert preds[3]["predicted_logd"] is not None


def test_predict_all_invalid(client: TestClient) -> None:
    r = client.post("/predict", json={"smiles": ["xxx", "yyy"]})
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert all(p["error"] == "invalid_smiles" for p in preds)
    assert all(p["predicted_logd"] is None for p in preds)


def test_predictions_are_reasonable(client: TestClient) -> None:
    """Sanity check: ethanol and benzene should have logD in a plausible range."""
    r = client.post("/predict", json={"smiles": ["CCO", "c1ccccc1"]})
    preds = r.json()["predictions"]
    ethanol_logd = preds[0]["predicted_logd"]
    benzene_logd = preds[1]["predicted_logd"]
    assert -3.0 < ethanol_logd < 1.0, f"ethanol logD {ethanol_logd} out of plausible range"
    assert 0.5 < benzene_logd < 4.0, f"benzene logD {benzene_logd} out of plausible range"
    assert benzene_logd > ethanol_logd, "benzene should be more lipophilic than ethanol"


def test_health_degraded_when_no_model() -> None:
    api_module._model = None
    with TestClient(api_module.app) as c:
        api_module._model = None
        r = c.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "degraded"
        assert body["model_loaded"] is False
