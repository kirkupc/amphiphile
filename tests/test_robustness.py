"""Robustness tests: pathological inputs must not crash the prediction pipeline.

A "service" has to survive weird inputs. These tests instantiate a tiny in-memory
model (same pattern as test_inference.py) and throw pathological SMILES at it;
every call must return a Prediction per input with an appropriate error flag
when applicable, and never raise.
"""

from __future__ import annotations

import numpy as np
import pytest

from logd.features import FeatureSpec, featurise_batch, mol_from_smiles, morgan_fp
from logd.inference import LoadedModel, predict
from logd.models.baseline import train_ensemble
from logd.uncertainty import ApplicabilityDomain, ConformalCalibrator, Reliability

TRAIN_SMILES = [
    "CCO",
    "CCC",
    "CCCC",
    "CCCCC",
    "CCCCCC",
    "c1ccccc1",
    "c1ccc(C)cc1",
    "c1ccc(CC)cc1",
    "CC(=O)O",
    "CC(=O)OC",
    "CC(=O)N",
    "c1ccncc1",
    "C1CCCCC1",
    "C1CCCCC1C",
    "CCN",
    "CCCN",
    "OCC(O)CO",
    "OCCO",
    "CCS",
    "CCSC",
]
TRAIN_Y = np.linspace(-1.5, 3.5, len(TRAIN_SMILES)).astype(np.float32)


@pytest.fixture(scope="module")
def tiny_model() -> LoadedModel:
    spec = FeatureSpec()
    X, _mask = featurise_batch(TRAIN_SMILES, spec)
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
    reliability = Reliability(conformal=conformal, ad=ad, std_threshold=1.0, tanimoto_threshold=0.1)
    return LoadedModel(
        baseline=model,
        chemprop=None,
        reliability=reliability,
        feature_spec=spec,
        model_type="baseline",
    )


def test_empty_string(tiny_model: LoadedModel) -> None:
    out = predict([""], model=tiny_model)
    assert len(out) == 1
    assert out[0].error == "invalid_smiles"


def test_whitespace_only(tiny_model: LoadedModel) -> None:
    out = predict(["   ", "\t\n"], model=tiny_model)
    assert all(r.error == "invalid_smiles" for r in out)


def test_unicode_garbage(tiny_model: LoadedModel) -> None:
    out = predict(["🧪", "日本語", "\x00nul"], model=tiny_model)
    assert all(r.error == "invalid_smiles" for r in out)


def test_mixed_valid_and_pathological(tiny_model: LoadedModel) -> None:
    inputs = ["CCO", "", "🧪", "c1ccccc1", "nope", "   "]
    out = predict(inputs, model=tiny_model)
    assert len(out) == len(inputs)
    assert out[0].error is None and out[0].predicted_logd is not None
    assert out[1].error == "invalid_smiles"
    assert out[2].error == "invalid_smiles"
    assert out[3].error is None and out[3].predicted_logd is not None
    assert out[4].error == "invalid_smiles"
    assert out[5].error == "invalid_smiles"


def test_very_long_but_valid_smiles(tiny_model: LoadedModel) -> None:
    """A 100-carbon chain is valid SMILES but unusual. Should not crash."""
    long_smiles = "C" * 100
    out = predict([long_smiles], model=tiny_model)
    assert len(out) == 1
    # May or may not be invalid depending on RDKit's hydrogen-budget; either way no crash.
    assert out[0].error in (None, "invalid_smiles")


def test_smiles_with_trailing_whitespace(tiny_model: LoadedModel) -> None:
    out = predict(["CCO ", "CCO\n", " CCO"], model=tiny_model)
    # RDKit accepts leading/trailing whitespace; canonicalisation handles it.
    assert all(r.error is None for r in out)


def test_duplicate_inputs(tiny_model: LoadedModel) -> None:
    """Duplicates must return identical predictions in the same positions."""
    out = predict(["CCO", "CCO", "CCO"], model=tiny_model)
    assert out[0].predicted_logd == out[1].predicted_logd == out[2].predicted_logd


def test_single_atom(tiny_model: LoadedModel) -> None:
    out = predict(["C", "N", "O"], model=tiny_model)
    # These are methane, ammonia, water — valid but trivial.
    assert all(r.error is None for r in out)


def test_charged_species(tiny_model: LoadedModel) -> None:
    out = predict(["[NH4+]", "CC(=O)[O-]"], model=tiny_model)
    assert all(r.error is None for r in out)


def test_huge_batch_does_not_crash(tiny_model: LoadedModel) -> None:
    """1000 duplicated SMILES — just a smoke test that batching scales."""
    out = predict(["CCO"] * 1000, model=tiny_model)
    assert len(out) == 1000
    assert all(r.error is None for r in out)
