"""
Tests for Probabilistic & Spatial Verification Metrics (Phase 5).
"""

import numpy as np
import pytest

from evaluation.probabilistic_metrics import (
    crps_gaussian,
    crps_ensemble,
    crps_skill_score,
    prediction_interval_coverage_probability,
    mean_prediction_interval_width,
    sharpness,
    spatial_mae,
    spatial_rmse,
    spatial_weight_entropy,
)


def test_crps_gaussian_non_negative():
    obs = np.array([12.0, 15.0])
    mu = np.array([10.0, 15.0])
    sigma = np.array([2.0, 1.0])
    c = crps_gaussian(obs, mu, sigma)
    assert len(c) == 2
    assert np.all(c >= 0.0)


def test_crps_ensemble():
    obs = np.array([10.0, 20.0])
    members = np.array([
        [9.0, 10.0, 11.0],
        [18.0, 20.0, 22.0],
    ])
    c = crps_ensemble(obs, members)
    assert len(c) == 2
    assert np.all(c >= 0.0)


def test_crps_skill_score():
    score_better = crps_skill_score(0.5, 1.0)
    score_same = crps_skill_score(1.0, 1.0)
    score_worse = crps_skill_score(1.5, 1.0)
    assert score_better == 0.5
    assert score_same == 0.0
    assert score_worse == -0.5


def test_picp_and_mpiw():
    obs = np.array([10.0, 20.0, 30.0])
    low = np.array([8.0, 15.0, 35.0])   # obs 1, 2 inside, obs 3 outside
    high = np.array([12.0, 25.0, 40.0])
    picp = prediction_interval_coverage_probability(obs, low, high)
    mpiw = mean_prediction_interval_width(low, high)
    assert np.isclose(picp, 2.0 / 3.0)
    assert np.isclose(mpiw, (4.0 + 10.0 + 5.0) / 3.0)


def test_spatial_metrics():
    p = np.zeros((4, 4))
    o = np.ones((4, 4)) * 2.0
    assert spatial_mae(p, o) == 2.0
    assert spatial_rmse(p, o) == 2.0

    # Weight entropy of equal 3-way weights should be log(3)
    W_eq = np.full((3, 4, 4), 1.0 / 3.0)
    ent = spatial_weight_entropy(W_eq)
    assert np.isclose(ent, np.log(3.0), atol=1e-3)
