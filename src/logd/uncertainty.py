"""Uncertainty quantification + applicability domain.

Three complementary signals:

1. **Ensemble standard deviation** — epistemic uncertainty from the deep ensemble.
   Already produced by BaselineModel.predict / ChempropModel.predict.

2. **Mondrian conformal intervals** — distribution-free prediction intervals with
   a target marginal coverage (e.g. 90%). Calibrated on a held-out val set using
   the standardised residual |y - ŷ| / σ. Gives a scale-corrected interval so
   high-uncertainty predictions get wider bands, not the same band.

3. **Applicability-domain Tanimoto score** — nearest-neighbour Morgan Tanimoto
   similarity of the query to the training set. Low similarity means the model
   is extrapolating into chemistry it didn't see. This catches the case where
   the ensemble happens to be confidently wrong because all members extrapolate
   the same way.

**Reliability flag**: reliable iff
    (ensemble_std <= threshold_std) AND (nearest_tanimoto >= threshold_tanimoto)

Thresholds are calibrated on val so that among reliable predictions, a target
precision (e.g. 90% within 1 log unit) is achieved.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np


@dataclass
class ConformalCalibrator:
    """Mondrian conformal on standardised residuals.

    Stored state: the empirical quantile q of |y - ŷ| / σ on the calibration set.
    Prediction interval for test point: ŷ ± q·σ.
    """

    quantile: float
    alpha: float  # target miscoverage (e.g. 0.1 for 90% intervals)

    @classmethod
    def fit(
        cls, y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray, alpha: float = 0.1
    ) -> ConformalCalibrator:
        eps = 1e-6
        residuals = np.abs(y_true - y_pred) / (y_std + eps)
        n = len(residuals)
        # Finite-sample adjusted quantile (per Vovk/Romano).
        k = int(np.ceil((n + 1) * (1 - alpha))) - 1
        k = max(0, min(n - 1, k))
        quantile = float(np.sort(residuals)[k])
        return cls(quantile=quantile, alpha=alpha)

    def interval(self, y_pred: np.ndarray, y_std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        half = self.quantile * y_std
        return y_pred - half, y_pred + half


@dataclass
class ApplicabilityDomain:
    """Training-set fingerprint bank for nearest-neighbour Tanimoto scoring.

    Stores Morgan fps as a packed uint8 matrix. Similarity is computed with
    BitVect-style popcount — fast, vectorisable.
    """

    train_fps: np.ndarray  # shape (n_train, n_bits), uint8 in {0,1}

    def nearest_similarity(self, query_fps: np.ndarray) -> np.ndarray:
        """Max Tanimoto similarity of each query row vs training set."""
        if query_fps.size == 0:
            return np.zeros(0, dtype=np.float32)
        # Tanimoto = |A∩B| / |A∪B| = popcount(A & B) / (|A| + |B| - popcount(A & B))
        q = query_fps.astype(np.uint8)
        t = self.train_fps.astype(np.uint8)
        q_count = q.sum(axis=1)  # (Q,)
        t_count = t.sum(axis=1)  # (T,)
        # Batched in chunks to keep memory reasonable
        sims = np.empty(len(q), dtype=np.float32)
        chunk = 1024
        for i in range(0, len(q), chunk):
            qi = q[i : i + chunk]
            inter = (qi @ t.T).astype(np.int32)  # (chunk, T)
            union = q_count[i : i + chunk, None] + t_count[None, :] - inter
            tani = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
            sims[i : i + chunk] = tani.max(axis=1)
        return sims


@dataclass
class Reliability:
    conformal: ConformalCalibrator
    ad: ApplicabilityDomain
    std_threshold: float
    tanimoto_threshold: float

    def flag(self, y_std: np.ndarray, nn_sim: np.ndarray) -> np.ndarray:
        return (y_std <= self.std_threshold) & (nn_sim >= self.tanimoto_threshold)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> Reliability:
        return joblib.load(path)


def calibrate_thresholds(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    nn_sim: np.ndarray,
    target_within: float = 1.0,
    target_precision: float = 0.9,
) -> tuple[float, float]:
    """Pick (std_threshold, tanimoto_threshold) so that among compounds flagged
    reliable on val, >= target_precision fraction have |error| <= target_within.

    Grid search over quantiles; picks the combination that keeps the most
    compounds reliable while hitting the precision target. Returns (std_thr,
    tani_thr); if no grid cell hits the target, falls back to medians.
    """
    err = np.abs(y_true - y_pred)
    best: tuple[float, float] | None = None
    best_n = -1
    for std_q in np.linspace(0.3, 0.9, 7):
        for tani_q in np.linspace(0.1, 0.5, 5):
            std_thr = float(np.quantile(y_std, std_q))
            tani_thr = float(np.quantile(nn_sim, tani_q))
            mask = (y_std <= std_thr) & (nn_sim >= tani_thr)
            if mask.sum() == 0:
                continue
            prec = float((err[mask] <= target_within).mean())
            if prec >= target_precision and int(mask.sum()) > best_n:
                best_n = int(mask.sum())
                best = (std_thr, tani_thr)
    if best is None:
        return float(np.median(y_std)), float(np.median(nn_sim))
    return best
