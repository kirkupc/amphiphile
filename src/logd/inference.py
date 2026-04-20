"""Public inference API.

The two functions callers should use:

    model = load_model()                # cold-start from serialized artifacts
    results = predict(smiles_list, model=model)

Each result is a Prediction dataclass (see below). Invalid SMILES get
predicted_logd=None and error='invalid_smiles'; callers never need to
pre-filter.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from logd.features import FeatureSpec, featurise_batch, morgan_fp, mol_from_smiles
from logd.models.baseline import BaselineModel
from logd.uncertainty import Reliability
from logd.utils import get_logger, models_dir

LOG = get_logger(__name__)


@dataclass
class Prediction:
    smiles: str
    predicted_logd: float | None
    uncertainty: float | None
    reliable: bool
    error: str | None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class LoadedModel:
    """Opaque bundle carrying everything needed for cold-start inference."""

    baseline: BaselineModel
    reliability: Reliability
    feature_spec: FeatureSpec


def load_model(model_path: Path | None = None, reliability_path: Path | None = None) -> LoadedModel:
    """Load a trained model bundle. Defaults resolve to models/ in the repo."""
    model_path = model_path or (models_dir() / "baseline.joblib")
    reliability_path = reliability_path or (models_dir() / "reliability.joblib")
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {model_path}. Run `uv run logd train` first."
        )
    if not reliability_path.exists():
        raise FileNotFoundError(
            f"Reliability artifact not found at {reliability_path}. "
            "Run `uv run logd train` (training writes it)."
        )
    baseline = BaselineModel.load(model_path)
    reliability = Reliability.load(reliability_path)
    return LoadedModel(
        baseline=baseline,
        reliability=reliability,
        feature_spec=baseline.feature_spec,
    )


def _nn_similarity(smiles_valid: list[str], reliability: Reliability) -> np.ndarray:
    """Morgan fp of each valid SMILES → max Tanimoto against stored training bank."""
    if not smiles_valid:
        return np.zeros(0, dtype=np.float32)
    fps = np.stack([morgan_fp(mol_from_smiles(s)) for s in smiles_valid], axis=0)
    return reliability.ad.nearest_similarity(fps)


def predict(
    smiles: Iterable[str], model: LoadedModel | None = None
) -> list[Prediction]:
    """Predict logD + uncertainty + reliability for a batch of SMILES.

    Order-preserving: returns one Prediction per input SMILES.
    """
    if model is None:
        model = load_model()

    smiles_list = list(smiles)
    if not smiles_list:
        return []

    X, mask = featurise_batch(smiles_list, model.feature_spec)

    results: list[Prediction] = [
        Prediction(smiles=s, predicted_logd=None, uncertainty=None, reliable=False, error="invalid_smiles")
        for s in smiles_list
    ]

    if mask.sum() == 0:
        return results

    y_pred, y_std = model.baseline.predict(X)
    valid_smiles = [s for s, m in zip(smiles_list, mask) if m]
    nn_sim = _nn_similarity(valid_smiles, model.reliability)
    flags = model.reliability.flag(y_std, nn_sim)

    j = 0
    for i, is_valid in enumerate(mask):
        if not is_valid:
            continue
        results[i] = Prediction(
            smiles=smiles_list[i],
            predicted_logd=float(y_pred[j]),
            uncertainty=float(y_std[j]),
            reliable=bool(flags[j]),
            error=None,
        )
        j += 1

    return results
