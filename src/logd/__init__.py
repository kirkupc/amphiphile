"""logD prediction service — deepmirror take-home."""

# Preempt a Chemprop/astartes/matplotlib/Colab trifecta: Colab sets MPLBACKEND
# to 'module://matplotlib_inline.backend_inline' at kernel start, but a fresh
# matplotlib install may not register that backend yet, so `import chemprop`
# blows up inside matplotlib's rcParams validation. Force 'Agg' before any
# transitive chemprop import happens. No-op outside Colab.
import os as _os

if _os.environ.get("MPLBACKEND", "").startswith("module://matplotlib_inline"):
    _os.environ["MPLBACKEND"] = "Agg"

from logd.inference import Prediction, load_model, predict

__all__ = ["Prediction", "load_model", "predict"]
__version__ = "0.1.0"
