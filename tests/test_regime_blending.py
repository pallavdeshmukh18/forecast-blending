"""
Automated unit tests for Regime-Conditioned Skill-Weighted Multi-Model Blenders.
"""

import pytest
import numpy as np
import pandas as pd

from models.regime_blending import (
    RegimeSkillBlender,
    RegimeInverseVarianceBlender,
    RegimeInverseMAEBlender,
)


@pytest.fixture
def regime_verified_dataset():
    """
    Construct a verified dataset where:
    - In 'stable' regime, Model A has low error (MSE=1.0) and Model B has high error (MSE=9.0).
    - In 'convective' regime, Model B has low error (MSE=1.0) and Model A has high error (MSE=9.0).
    """
    rows = []
    # 50 stable records
    for i in range(50):
        # Model A
        rows.append({
            "model_name": "model_a",
            "timestamp": "2026-06-01T00:00:00Z",
            "variable": "rainfall",
            "lead_time": 24,
            "region": "inland",
            "weather_regime": "stable",
            "forecast_value": 0.0,
            "observed_value": 0.0,
            "error": 1.0,  # low error
        })
        # Model B
        rows.append({
            "model_name": "model_b",
            "timestamp": "2026-06-01T00:00:00Z",
            "variable": "rainfall",
            "lead_time": 24,
            "region": "inland",
            "weather_regime": "stable",
            "forecast_value": 3.0,
            "observed_value": 0.0,
            "error": 3.0,  # high error
        })

    # 50 convective records
    for i in range(50):
        # Model A
        rows.append({
            "model_name": "model_a",
            "timestamp": "2026-06-02T00:00:00Z",
            "variable": "rainfall",
            "lead_time": 24,
            "region": "inland",
            "weather_regime": "convective",
            "forecast_value": 10.0,
            "observed_value": 40.0,
            "error": 30.0,  # high error
        })
        # Model B
        rows.append({
            "model_name": "model_b",
            "timestamp": "2026-06-02T00:00:00Z",
            "variable": "rainfall",
            "lead_time": 24,
            "region": "inland",
            "weather_regime": "convective",
            "forecast_value": 39.0,
            "observed_value": 40.0,
            "error": 1.0,  # low error
        })

    return pd.DataFrame(rows)


def test_regime_blender_weights_sum_to_one(regime_verified_dataset):
    """Verify weights for each regime sum strictly to 1.0."""
    blender = RegimeInverseVarianceBlender(minimum_samples=10, weight_smoothing=0.0)
    blender.fit_verification_data(regime_verified_dataset)

    w_stable = blender.get_weights(variable="rainfall", lead_time=24, region="inland", weather_regime="stable")
    w_conv = blender.get_weights(variable="rainfall", lead_time=24, region="inland", weather_regime="convective")

    assert np.isclose(sum(w_stable.values()), 1.0, atol=1e-5)
    assert np.isclose(sum(w_conv.values()), 1.0, atol=1e-5)


def test_regime_blender_learns_regime_specific_weights(regime_verified_dataset):
    """Verify Model A gets higher weight under 'stable' and Model B under 'convective'."""
    blender = RegimeInverseVarianceBlender(minimum_samples=10, weight_smoothing=0.0)
    blender.fit_verification_data(regime_verified_dataset)

    w_stable = blender.get_weights(variable="rainfall", lead_time=24, region="inland", weather_regime="stable")
    w_conv = blender.get_weights(variable="rainfall", lead_time=24, region="inland", weather_regime="convective")

    # In stable, Model A is better -> w_stable['model_a'] > w_stable['model_b']
    assert w_stable["model_a"] > w_stable["model_b"]

    # In convective, Model B is better -> w_conv['model_b'] > w_conv['model_a']
    assert w_conv["model_b"] > w_conv["model_a"]


def test_regime_blender_hierarchical_fallback_for_sparse_regime(regime_verified_dataset):
    """
    When an unseen or rare regime (e.g. 'heavy_rain' with < minimum_samples) is queried,
    blender must fall back hierarchically to lead_time+region or lead_time without crashing.
    """
    blender = RegimeInverseVarianceBlender(minimum_samples=20)
    blender.fit_verification_data(regime_verified_dataset)

    # 'heavy_rain' was not in training dataset
    w_sparse = blender.get_weights(variable="rainfall", lead_time=24, region="inland", weather_regime="heavy_rain")

    assert np.isclose(sum(w_sparse.values()), 1.0, atol=1e-5)
    assert "model_a" in w_sparse and "model_b" in w_sparse


def test_regime_blender_predict_pipeline(regime_verified_dataset):
    """Verify full predict pipeline generates blended forecasts and weights."""
    blender = RegimeInverseMAEBlender(minimum_samples=10)
    blender.fit_verification_data(regime_verified_dataset)

    test_forecasts = pd.DataFrame([
        {
            "model_name": "model_a",
            "timestamp": "2026-06-03T00:00:00Z",
            "variable": "rainfall",
            "lead_time": 24,
            "region": "inland",
            "latitude": 28.0,
            "longitude": 77.0,
            "weather_regime": "stable",
            "forecast_value": 5.0,
        },
        {
            "model_name": "model_b",
            "timestamp": "2026-06-03T00:00:00Z",
            "variable": "rainfall",
            "lead_time": 24,
            "region": "inland",
            "latitude": 28.0,
            "longitude": 77.0,
            "weather_regime": "stable",
            "forecast_value": 7.0,
        },
    ])

    blended = blender.predict(test_forecasts)
    assert len(blended) == 1
    assert blended.iloc[0]["model_name"] == "regime_inverse_mae_blend"
    assert "forecast_value" in blended.columns
    assert 5.0 <= blended.iloc[0]["forecast_value"] <= 7.0
