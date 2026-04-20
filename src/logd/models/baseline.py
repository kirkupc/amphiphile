"""LightGBM baseline on RDKit descriptors + Morgan fingerprints.

Trained as a deep ensemble (k=5 seeds) to give epistemic uncertainty. LightGBM
handles sparse/dense mixed features well and trains in seconds on 20-40k rows
without hyperparameter tuning.

Hyperparameters: conservative defaults. No tuning sweep; the ML story is in
modelling + uncertainty, not gradient-boosting hyperparameter search.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np

from logd.features import FeatureSpec
from logd.utils import get_logger, set_seed

LOG = get_logger(__name__)

DEFAULT_LGBM_PARAMS: dict[str, Any] = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_child_samples": 10,
    # Lower fractions produce genuinely different ensemble members: each
    # booster sees a different 70% of rows and 70% of features with a
    # per-member bagging_seed. Without this, seed-ensembled LightGBM members
    # converge to near-identical predictors and the ensemble std collapses
    # (we observed Spearman(std, |err|) = 0.006 on OOD at 0.9/0.9). Diversity
    # injected here is the standard fix.
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 1,
    "verbose": -1,
}

DEFAULT_NUM_BOOST_ROUND = 2000
DEFAULT_EARLY_STOPPING = 100


@dataclass
class BaselineModel:
    """A k-member LightGBM ensemble over the same features, different seeds."""

    feature_spec: FeatureSpec
    boosters: list[lgb.Booster] = field(default_factory=list)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (mean, std) across ensemble members, shape (n,) each."""
        if not self.boosters:
            raise RuntimeError("Model is not trained.")
        preds = np.stack([b.predict(X) for b in self.boosters], axis=0)
        return preds.mean(axis=0), preds.std(axis=0)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "feature_spec": self.feature_spec,
                "booster_strings": [b.model_to_string() for b in self.boosters],
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> BaselineModel:
        blob = joblib.load(path)
        boosters = [lgb.Booster(model_str=s) for s in blob["booster_strings"]]
        return cls(feature_spec=blob["feature_spec"], boosters=boosters)


def train_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_spec: FeatureSpec,
    k: int = 5,
    base_seed: int = 0,
    params: dict[str, Any] | None = None,
) -> BaselineModel:
    params = {**DEFAULT_LGBM_PARAMS, **(params or {})}
    boosters: list[lgb.Booster] = []
    for i in range(k):
        seed = base_seed + i
        set_seed(seed)
        ds_train = lgb.Dataset(X_train, label=y_train)
        ds_val = lgb.Dataset(X_val, label=y_val, reference=ds_train)
        params_i = {**params, "seed": seed, "bagging_seed": seed, "feature_fraction_seed": seed}
        LOG.info("Training baseline booster %d/%d (seed=%d)", i + 1, k, seed)
        booster = lgb.train(
            params_i,
            ds_train,
            num_boost_round=DEFAULT_NUM_BOOST_ROUND,
            valid_sets=[ds_val],
            valid_names=["val"],
            callbacks=[lgb.early_stopping(DEFAULT_EARLY_STOPPING), lgb.log_evaluation(0)],
        )
        boosters.append(booster)
    return BaselineModel(feature_spec=feature_spec, boosters=boosters)
