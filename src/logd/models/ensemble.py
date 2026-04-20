"""Unified prediction interface across model families.

Both BaselineModel and (eventually) ChempropModel expose the same surface:

    predict(smiles_list) -> list[Prediction]

This module owns the SMILES → features → ensemble → postprocess pipeline, so
inference.py can stay model-agnostic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class EnsemblePredictor(Protocol):
    """What inference.py expects from any trained model artifact."""

    def predict_smiles(
        self, smiles: list[str]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (mean_logd, std_logd, valid_mask). All length n = len(smiles)."""
        ...

    def save(self, path: Path) -> None: ...

    @classmethod
    def load(cls, path: Path) -> "EnsemblePredictor": ...
