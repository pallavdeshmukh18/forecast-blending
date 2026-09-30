"""
Tests for Probabilistic Forecast Representations (Phase 5).
"""

import numpy as np
import pytest

from models.probabilistic import (
    ProbabilisticForecast,
    EnsembleProbabilisticModel,
    ResidualUncertaintyProbabilisticModel,
)


def test_probabilistic_forecast_container():
    mean = np.array([20.0, 25.0], dtype=np.float32)
    std = np.array([2.0, 3.0], dtype=np.float32)
    pf = ProbabilisticForecast(mean=mean, std=std)

    assert pf.shape == (2,)
    # At mean, CDF is 0.50
    assert np.allclose(pf.cdf(20.0)[0], 0.50)
    # Exceedance prob at mean is 0.50
    assert np.allclose(pf.prob_exceeds(20.0)[0], 0.50)
    # Quantile at 0.50 is mean
    assert np.allclose(pf.quantile(0.50), mean)

    low_80, high_80 = pf.prediction_interval(0.80)
    assert np.all(high_80 > low_80)


def test_ensemble_probabilistic_model():
    ens_model = EnsembleProbabilisticModel()
    # 5 ensemble members over 2 spatial points
    E = np.array([
        [10.0, 20.0],
        [12.0, 22.0],
        [14.0, 24.0],
        [16.0, 26.0],
        [18.0, 28.0],
    ], dtype=np.float32)

    pf = ens_model.predict(E)
    assert pf.shape == (2,)
    assert np.allclose(pf.mean, [14.0, 24.0])
    assert np.all(pf.std > 0)
    assert pf.distribution_type == "ensemble"


def test_residual_uncertainty_probabilistic_model():
    res_model = ResidualUncertaintyProbabilisticModel()
    preds = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    obs = np.array([11.0, 19.0, 31.0], dtype=np.float32)

    res_model.fit_residuals(preds, obs)
    assert res_model.residual_std > 0

    pf = res_model.predict(preds)
    assert pf.shape == (3,)
    assert np.allclose(pf.mean, preds)
