"""Uncertainty quantification + applicability domain.

Three complementary signals:

1. **Ensemble standard deviation** — epistemic uncertainty from the deep ensemble.
   Already produced by BaselineModel.predict / ChempropModel.predict.

2. **Conformal prediction intervals** — distribution-free prediction intervals
   with a target marginal coverage (e.g. 90%). Calibrated on a held-out val set
   using absolute residuals |y - ŷ|. Produces constant-width intervals (ŷ ± q)
   rather than Mondrian-style scale-corrected ones, because ensemble std showed
   poor correlation with actual error, inflating the Mondrian quantile.

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
    """Absolute-residual conformal calibration.

    Stored state: the empirical quantile q of |y - ŷ| on the calibration set.
    Prediction interval for test point: ŷ ± q (constant width).

    Earlier versions divided by ensemble std to produce Mondrian-style
    scale-corrected intervals, but when ensemble std is poorly correlated
    with actual error the quantile inflates (we saw q ~10). Constant-width
    intervals are less elegant but give honest, usable coverage.
    """

    quantile: float
    alpha: float  # target miscoverage (e.g. 0.1 for 90% intervals)

    @classmethod
    def fit(
        cls, y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray, alpha: float = 0.1
    ) -> ConformalCalibrator:
        # y_std kept in signature for API compatibility; unused after Mondrian→absolute switch.
        del y_std
        residuals = np.abs(y_true - y_pred)
        n = len(residuals)
        # Finite-sample adjusted quantile (per Vovk/Romano).
        k = int(np.ceil((n + 1) * (1 - alpha))) - 1
        k = max(0, min(n - 1, k))
        quantile = float(np.sort(residuals)[k])
        return cls(quantile=quantile, alpha=alpha)

    def interval(self, y_pred: np.ndarray, y_std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        half = self.quantile
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
        return joblib.load(path)  # type: ignore[no-any-return]


def calibrate_thresholds(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    nn_sim: np.ndarray,
    target_within: float = 1.0,
    target_precision: float = 0.88,
) -> tuple[float, float]:
    """Pick (std_threshold, tanimoto_threshold) so that among compounds flagged
    reliable on val, >= target_precision fraction have |error| <= target_within.

    Grid search: std uses val quantiles (model-specific scale), Tanimoto uses
    absolute values (fixed 0–1 scale) so thresholds transfer to OOD data.
    Picks the combination that keeps the most compounds reliable while hitting
    the precision target. Falls back to medians if no cell hits the target.
    """
    err = np.abs(y_true - y_pred)
    best: tuple[float, float] | None = None
    best_tani = 1.0
    best_n = -1
    for std_q in np.linspace(0.3, 0.9, 7):
        for tani_thr in [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
            std_thr = float(np.quantile(y_std, std_q))
            mask = (y_std <= std_thr) & (nn_sim >= tani_thr)
            if mask.sum() == 0:
                continue
            prec = float((err[mask] <= target_within).mean())
            if prec >= target_precision and (
                tani_thr < best_tani or (tani_thr == best_tani and int(mask.sum()) > best_n)
            ):
                best_tani = tani_thr
                best_n = int(mask.sum())
                best = (std_thr, tani_thr)
    if best is None:
        return float(np.median(y_std)), float(np.median(nn_sim))
    return best
