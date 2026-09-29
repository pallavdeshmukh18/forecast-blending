"""
Tests for Gaussian Ensemble Model Output Statistics (EMOS) (Phase 5).
"""

import numpy as np
import pytest

from models.emos import EnsembleModelOutputStatistics, gaussian_crps_loss


def test_gaussian_crps_loss():
    obs = np.array([10.0, 20.0])
    mu = np.array([10.0, 20.0])
    sigma = np.array([1.0, 1.0])
    loss = gaussian_crps_loss(obs, mu, sigma)
    # Perfect point match should yield positive CRPS proportional to spread
    assert loss > 0.0


def test_emos_fit_and_predict():
    emos = EnsembleModelOutputStatistics(fit_method="crps")

    rng = np.random.default_rng(42)
    # 20 samples, 3 ensemble members, 4x4 grid
    E = rng.normal(15.0, 3.0, (20, 3, 4, 4))
    Y = rng.normal(15.0, 2.0, (20, 4, 4))

    emos.fit(E, Y)
    assert emos.is_fitted is True
    assert emos.c > 0.0
    assert emos.d >= 0.0

    pf = emos.predict(E[:2])
    assert pf.shape == (2, 4, 4)
    assert np.all(pf.std > 0)
    assert pf.distribution_type == "emos_gaussian"
