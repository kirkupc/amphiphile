"""LightGBM baseline on RDKit descriptors + Morgan fingerprints.

Trained as a deep ensemble (k=5 seeds) to give epistemic uncertainty. LightGBM
handles sparse/dense mixed features well and trains in seconds on 20-40k rows.

Hyperparameters are tuned via grid search over num_leaves, learning_rate, and
min_child_samples (27 combinations) using val RMSE, then the best config is
used for all k ensemble members. Ensemble diversity comes from per-member
bagging and feature subsampling (0.7/0.7), not from different hyperparameters.
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


def _train_single_booster(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: dict[str, Any],
    seed: int,
) -> lgb.Booster:
    set_seed(seed)
    ds_train = lgb.Dataset(X_train, label=y_train)
    ds_val = lgb.Dataset(X_val, label=y_val, reference=ds_train)
    params_i = {**params, "seed": seed, "bagging_seed": seed, "feature_fraction_seed": seed}
    return lgb.train(
        params_i,
        ds_train,
        num_boost_round=DEFAULT_NUM_BOOST_ROUND,
        valid_sets=[ds_val],
        valid_names=["val"],
        callbacks=[lgb.early_stopping(DEFAULT_EARLY_STOPPING), lgb.log_evaluation(0)],
    )


def tune_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int = 0,
) -> dict[str, Any]:
    """Grid search over key LightGBM hyperparameters using val RMSE.

    Searches num_leaves, learning_rate, and min_child_samples — the parameters
    with the largest impact on bias-variance tradeoff for gradient boosting.
    Returns the best parameter dict (merged with defaults).
    """
    grid = [
        {"num_leaves": nl, "learning_rate": lr, "min_child_samples": mc}
        for nl in [63, 127, 255]
        for lr in [0.02, 0.05, 0.1]
        for mc in [5, 10, 20]
    ]
    best_rmse = float("inf")
    best_override: dict[str, Any] = {}
    for i, override in enumerate(grid):
        params = {**DEFAULT_LGBM_PARAMS, **override}
        booster = _train_single_booster(X_train, y_train, X_val, y_val, params, seed)
        preds = booster.predict(X_val)
        rmse = float(np.sqrt(np.mean((y_val - preds) ** 2)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_override = override
            LOG.info("Tuning [%d/%d]: new best RMSE=%.4f with %s", i + 1, len(grid), rmse, override)
    LOG.info("Best hyperparameters: %s (val RMSE=%.4f)", best_override, best_rmse)
    return {**DEFAULT_LGBM_PARAMS, **best_override}


def train_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_spec: FeatureSpec,
    k: int = 5,
    base_seed: int = 0,
    params: dict[str, Any] | None = None,
    tune: bool = False,
) -> BaselineModel:
    if tune:
        LOG.info("Running hyperparameter search before ensemble training")
        tuned = tune_hyperparameters(X_train, y_train, X_val, y_val, seed=base_seed)
        params = {**tuned, **(params or {})}
    else:
        params = {**DEFAULT_LGBM_PARAMS, **(params or {})}
    boosters: list[lgb.Booster] = []
    for i in range(k):
        seed = base_seed + i
        LOG.info("Training baseline booster %d/%d (seed=%d)", i + 1, k, seed)
        booster = _train_single_booster(X_train, y_train, X_val, y_val, params, seed)
        boosters.append(booster)
    return BaselineModel(feature_spec=feature_spec, boosters=boosters)
