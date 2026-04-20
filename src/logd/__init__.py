"""logD prediction service — deepmirror take-home."""

from logd.inference import Prediction, load_model, predict

__all__ = ["Prediction", "load_model", "predict"]
__version__ = "0.1.0"
