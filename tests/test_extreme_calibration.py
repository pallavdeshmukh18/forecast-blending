"""
Unit tests for Advanced Extreme Weather Calibration:
Gamma EMOS, Zero-Inflated / Censored Gamma, GEV Tail Fitting, and Extreme Probability Engine.
"""

import pytest
import numpy as np
import pandas as pd

from models.extreme_calibration import (
    GammaEMOS,
    ZeroInflatedGammaEMOS,
    GEVTailCalibrator,
    ExtremeProbabilityEngine,
)


def test_gamma_emos_fitting_and_probabilities():
    np.random.seed(42)
    n = 100
    ens_mean = np.random.uniform(5.0, 30.0, size=n)
    ens_var = np.random.uniform(2.0, 10.0, size=n)
    # Observations generated from positive skewed distribution
    obs = ens_mean + np.random.gamma(shape=2.0, scale=2.0, size=n)

    gamma_emos = GammaEMOS()
    gamma_emos.fit(ens_mean, ens_var, obs)

    assert gamma_emos.fitted
    alpha, scale = gamma_emos.predict_distribution(ens_mean[:5], ens_var[:5])
    assert len(alpha) == 5
    assert np.all(alpha > 0.0)
    assert np.all(scale > 0.0)

    # Exceedance probability for 25 mm threshold
    probs = gamma_emos.exceedance_probability(ens_mean[:5], ens_var[:5], threshold=25.0)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


def test_zero_inflated_gamma_emos():
    np.random.seed(42)
    n = 150
    ens_mean = np.random.uniform(0.0, 25.0, size=n)
    ens_var = np.random.uniform(1.0, 5.0, size=n)
    # 40% zero rainfall
    obs = np.where(ens_mean < 8.0, 0.0, ens_mean + np.random.exponential(scale=5.0, size=n))

    zig = ZeroInflatedGammaEMOS(censoring_threshold=0.1)
    zig.fit(ens_mean, ens_var, obs)

    assert zig.fitted
    p_zero = zig.predict_zero_probability(np.array([1.0, 15.0]))
    # Lower ensemble mean should have higher zero rainfall probability
    assert p_zero[0] > p_zero[1]

    # Exceedance probability
    p_exceed = zig.exceedance_probability(np.array([2.0, 20.0]), np.array([1.0, 4.0]), threshold=10.0)
    assert p_exceed[1] > p_exceed[0]
    assert np.all(p_exceed >= 0.0) and np.all(p_exceed <= 1.0)


def test_gev_tail_calibrator_sample_size_check():
    # Insufficient samples (< 50) should gracefully return False and not fit GEV
    small_obs = np.array([10.0, 15.0, 20.0])
    gev = GEVTailCalibrator(min_samples=50)
    success = gev.fit(small_obs)
    assert not success
    assert not gev.fitted
    assert gev.exceedance_probability(30.0) is None

    # Adequate sample size
    np.random.seed(42)
    large_obs = np.random.gumbel(loc=20.0, scale=8.0, size=80)
    success_large = gev.fit(large_obs)
    assert success_large
    assert gev.fitted
    p_gev = gev.exceedance_probability(40.0)
    assert p_gev is not None
    assert 0.0 <= p_gev <= 1.0


def test_extreme_probability_engine_multi_variable():
    engine = ExtremeProbabilityEngine()
    df = pd.DataFrame({
        "timestamp": ["2026-01-01T12:00:00Z", "2026-01-01T12:00:00Z", "2026-01-01T12:00:00Z"],
        "latitude": [20.0, 20.0, 20.0],
        "longitude": [75.0, 75.0, 75.0],
        "lead_time": [24, 24, 24],
        "variable": ["rainfall", "wind_speed", "temperature"],
        "blend_value": [30.0, 18.0, 42.0],
        "spread": [5.0, 2.0, 1.5],
    })

    probs_df = engine.compute_extreme_probabilities(df)
    assert not probs_df.empty
    assert "probability" in probs_df.columns
    assert "threshold" in probs_df.columns
    assert "calibration_method" in probs_df.columns
    assert np.all((probs_df["probability"] >= 0.0) & (probs_df["probability"] <= 1.0))
