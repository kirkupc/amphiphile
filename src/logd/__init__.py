"""logD prediction service — deepmirror take-home."""

# Preempt a Chemprop/astartes/matplotlib/Colab trifecta: Colab sets MPLBACKEND
# to 'module://matplotlib_inline.backend_inline' at kernel start, but a fresh
# matplotlib install may not register that backend yet, so `import chemprop`
# blows up inside matplotlib's rcParams validation. Force 'Agg' before any
# transitive chemprop import happens. No-op outside Colab.
import os as _os

if _os.environ.get("MPLBACKEND", "").startswith("module://matplotlib_inline"):
    _os.environ["MPLBACKEND"] = "Agg"


# Lazy imports — defer ``logd.inference`` (and its transitive dependency on
# lightgbm / numpy) until actually accessed.  This lets ``import torch`` happen
# first when the Chemprop training path is taken, avoiding a SIGSEGV on macOS
# ARM with NumPy >= 2.0.  See ``chemprop_wrap.py`` module docstring for details.
def __getattr__(name: str):
    if name in ("Prediction", "load_model", "predict"):
        from logd.inference import Prediction, load_model, predict

        # Cache in module namespace so __getattr__ is not called again.
        globals().update(Prediction=Prediction, load_model=load_model, predict=predict)
        return globals()[name]
    raise AttributeError(f"module 'logd' has no attribute {name!r}")


__all__ = ["Prediction", "load_model", "predict"]
__version__ = "0.1.0"
