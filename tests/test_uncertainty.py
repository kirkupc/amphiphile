"""Tests for uncertainty calibration — shape, monotonicity, and basic coverage."""

from __future__ import annotations

import numpy as np

from logd.uncertainty import ApplicabilityDomain, ConformalCalibrator, calibrate_thresholds


def test_conformal_interval_shape_and_sign() -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(size=500)
    y_pred = y + rng.normal(scale=0.3, size=500)
    y_std = np.abs(rng.normal(scale=0.3, size=500)) + 0.05

    cal = ConformalCalibrator.fit(y, y_pred, y_std, alpha=0.1)
    lo, hi = cal.interval(y_pred, y_std)
    assert lo.shape == y_pred.shape
    assert np.all(hi >= lo)
    assert cal.quantile > 0


def test_conformal_coverage_close_to_target() -> None:
    """Empirical coverage on the calibration set should be near (1 - alpha)."""
    rng = np.random.default_rng(0)
    n = 2000
    y_pred = rng.normal(size=n)
    y_std = np.full(n, 0.5)
    y = y_pred + rng.normal(scale=0.5, size=n)

    cal = ConformalCalibrator.fit(y, y_pred, y_std, alpha=0.1)
    lo, hi = cal.interval(y_pred, y_std)
    coverage = float(((y >= lo) & (y <= hi)).mean())
    assert 0.85 <= coverage <= 0.97


def test_conformal_tighter_intervals_for_lower_alpha() -> None:
    rng = np.random.default_rng(0)
    n = 500
    y_pred = rng.normal(size=n)
    y_std = np.full(n, 0.5)
    y = y_pred + rng.normal(scale=0.5, size=n)

    cal_strict = ConformalCalibrator.fit(y, y_pred, y_std, alpha=0.2)  # 80% coverage
    cal_loose = ConformalCalibrator.fit(y, y_pred, y_std, alpha=0.05)  # 95% coverage
    assert cal_loose.quantile > cal_strict.quantile


def test_applicability_domain_self_similarity_is_one() -> None:
    rng = np.random.default_rng(0)
    fps = (rng.random((10, 128)) > 0.7).astype(np.uint8)
    ad = ApplicabilityDomain(train_fps=fps)
    sims = ad.nearest_similarity(fps)
    np.testing.assert_allclose(sims, 1.0)


def test_applicability_domain_unrelated_compound_has_low_sim() -> None:
    rng = np.random.default_rng(0)
    train = (rng.random((50, 128)) > 0.9).astype(np.uint8)  # very sparse
    query = (rng.random((5, 128)) > 0.1).astype(np.uint8)  # very dense
    ad = ApplicabilityDomain(train_fps=train)
    sims = ad.nearest_similarity(query)
    assert (sims < 0.5).all()


def test_calibrate_thresholds_returns_valid_values() -> None:
    rng = np.random.default_rng(0)
    n = 200
    y = rng.normal(size=n)
    y_pred = y + rng.normal(scale=0.3, size=n)
    y_std = np.abs(rng.normal(scale=0.3, size=n)) + 0.05
    nn_sim = rng.uniform(0.1, 0.9, size=n)

    std_thr, tani_thr = calibrate_thresholds(y, y_pred, y_std, nn_sim)
    assert np.isfinite(std_thr)
    assert np.isfinite(tani_thr)
    assert 0.0 <= tani_thr <= 1.0
