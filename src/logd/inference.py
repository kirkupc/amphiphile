"""Public inference API.

The two functions callers should use:

    model = load_model()                # baseline (default)
    model = load_model(model_type="chemprop")  # Chemprop D-MPNN
    results = predict(smiles_list, model=model)

Each result is a Prediction dataclass (see below). Invalid SMILES get
predicted_logd=None and error='invalid_smiles'; callers never need to
pre-filter.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np

from logd.features import FeatureSpec, featurise_batch, morgan_fp, mol_from_smiles
from logd.models.baseline import BaselineModel
from logd.uncertainty import Reliability
from logd.utils import get_logger, models_dir

if TYPE_CHECKING:
    from logd.models.chemprop_wrap import ChempropModel

LOG = get_logger(__name__)


@dataclass
class Prediction:
    smiles: str
    predicted_logd: float | None
    uncertainty: float | None
    reliable: bool
    error: str | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class LoadedModel:
    """Opaque bundle carrying everything needed for cold-start inference."""

    baseline: BaselineModel | None
    chemprop: "ChempropModel | None"
    reliability: Reliability
    feature_spec: FeatureSpec | None
    model_type: str

    @property
    def is_chemprop(self) -> bool:
        return self.model_type == "chemprop"


def load_model(
    model_type: str = "baseline",
    model_path: Path | None = None,
    reliability_path: Path | None = None,
) -> LoadedModel:
    """Load a trained model bundle.

    model_type: "baseline" (LightGBM, default) or "chemprop" (D-MPNN).
    """
    if model_type == "chemprop":
        chemprop_dir = model_path or (models_dir() / "chemprop")
        reliability_path = reliability_path or (models_dir() / "chemprop_reliability.joblib")
        if not (chemprop_dir / "config.json").exists():
            raise FileNotFoundError(
                f"Chemprop artifacts not found at {chemprop_dir}. "
                "Download from the v0.1.0-chemprop release or train on Colab."
            )
        if not reliability_path.exists():
            raise FileNotFoundError(
                f"Chemprop reliability artifact not found at {reliability_path}."
            )
        from logd.models.chemprop_wrap import ChempropModel

        chemprop = ChempropModel.load(chemprop_dir)
        reliability = Reliability.load(reliability_path)
        return LoadedModel(
            baseline=None,
            chemprop=chemprop,
            reliability=reliability,
            feature_spec=None,
            model_type="chemprop",
        )

    # Baseline (default)
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
        chemprop=None,
        reliability=reliability,
        feature_spec=baseline.feature_spec,
        model_type="baseline",
    )


def _nn_similarity(smiles_valid: list[str], reliability: Reliability) -> "np.ndarray[Any, np.dtype[np.float32]]":
    """Morgan fp of each valid SMILES -> max Tanimoto against stored training bank.

    If a SMILES passes the upstream validity mask but fails salt-stripping here
    (e.g. counter-ion-only entries), it gets similarity 0.0 (maximally out of
    domain) rather than crashing.
    """
    if not smiles_valid:
        return np.zeros(0, dtype=np.float32)
    fps_list = []
    for s in smiles_valid:
        mol = mol_from_smiles(s)
        if mol is None:
            fps_list.append(np.zeros(2048, dtype=np.uint8))
        else:
            fps_list.append(morgan_fp(mol))
    fps = np.stack(fps_list, axis=0)
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

    results: list[Prediction] = [
        Prediction(smiles=s, predicted_logd=None, uncertainty=None, reliable=False, error="invalid_smiles")
        for s in smiles_list
    ]

    if model.is_chemprop:
        assert model.chemprop is not None
        y_pred, y_std, mask = model.chemprop.predict_smiles(smiles_list)
        valid_smiles = [s for s, m in zip(smiles_list, mask) if m]
    else:
        assert model.baseline is not None and model.feature_spec is not None
        X, mask = featurise_batch(smiles_list, model.feature_spec)
        if mask.sum() == 0:
            return results
        y_pred, y_std = model.baseline.predict(X)
        valid_smiles = [s for s, m in zip(smiles_list, mask) if m]

    if not valid_smiles:
        return results

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
