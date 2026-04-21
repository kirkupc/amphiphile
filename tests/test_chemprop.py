"""Smoke test for the Chemprop wrapper.

Validates the train → save → load → predict pipeline runs end-to-end on tiny
data with minimal epochs. We don't check prediction quality here (2 epochs on
24 compounds won't produce good numbers) — we check that the contract is
honoured: shape, mask alignment, round-trip determinism.

Marked `slow` because even 2 epochs pull in pytorch-lightning and actually
train. Skip with `pytest -m 'not slow'` for fast inner-loop.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

# torch must be imported before any logd module that transitively imports
# numpy/lightgbm.  On macOS ARM + NumPy >= 2.0, late torch initialization
# causes SIGSEGV during Chemprop's molecular-graph tensor operations.
import torch  # noqa: F401  — must precede logd imports

import numpy as np
import pytest

from logd.models.chemprop_wrap import ChempropModel


TRAIN_SMILES = [
    "CCO", "CCC", "CCCC", "CCCCC", "CCCCCC", "CCCCCCC",
    "c1ccccc1", "c1ccc(C)cc1", "c1ccc(CC)cc1", "c1ccc(CCC)cc1",
    "CC(=O)O", "CC(=O)OC", "CC(=O)N", "C(=O)O",
    "c1ccncc1", "c1ccncc1C", "C1CCCCC1", "C1CCCCC1C",
    "CCN", "CCCN", "OCC(O)CO", "OCCO", "CCS", "CCSC",
]
TRAIN_Y = np.linspace(-1.5, 3.5, len(TRAIN_SMILES)).astype(np.float32)
VAL_SMILES = TRAIN_SMILES[:4]
VAL_Y = TRAIN_Y[:4]


@pytest.mark.slow
def test_chemprop_train_predict_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as td:
        ckpt = Path(td) / "chemprop"
        model = ChempropModel(checkpoint_dir=ckpt, k=2)
        model.train(
            train_smiles=TRAIN_SMILES,
            train_y=TRAIN_Y,
            val_smiles=VAL_SMILES,
            val_y=VAL_Y,
            k=2,
            max_epochs=2,
            batch_size=8,
            base_seed=0,
        )
        mean, std, mask = model.predict_smiles(["CCO", "not_a_smiles", "c1ccccc1"])
        # Shape + mask contract
        assert mask.tolist() == [True, False, True]
        assert mean.shape == (2,)
        assert std.shape == (2,)
        assert np.all(std >= 0)

        # Load from disk and confirm deterministic predictions
        loaded = ChempropModel.load(ckpt)
        mean2, std2, mask2 = loaded.predict_smiles(["CCO", "c1ccccc1"])
        np.testing.assert_allclose(mean, mean2, atol=1e-5)
        np.testing.assert_allclose(std, std2, atol=1e-5)


@pytest.mark.slow
def test_chemprop_all_invalid_smiles() -> None:
    """When nothing is parseable, return zero arrays with a false mask."""
    with tempfile.TemporaryDirectory() as td:
        ckpt = Path(td) / "chemprop"
        model = ChempropModel(checkpoint_dir=ckpt, k=2)
        # Write minimal config so ChempropModel.load would succeed later,
        # but for this test we skip training entirely — just validate the
        # all-invalid fast path doesn't try to load models.
        mean, std, mask = ChempropModel(checkpoint_dir=ckpt, k=0).predict_smiles(
            ["not", "smiles", "xyz123"]
        )
        assert mask.sum() == 0
        assert mean.shape == (3,)
        assert std.shape == (3,)
