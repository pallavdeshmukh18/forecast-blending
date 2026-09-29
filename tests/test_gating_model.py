"""
Automated unit tests for PyTorch Neural Weather Regime Gating Network.
"""

import pytest
import numpy as np
import pandas as pd
import torch

from models.gating_model import (
    GatingPyTorchModule,
    WeatherRegimeGatingNetwork,
    RegimeGatingBlender,
)


@pytest.fixture
def synthetic_gating_training_data():
    """Construct multi-model training dataset for neural gating."""
    rng = np.random.default_rng(42)
    n = 120
    rows = []

    for i in range(n):
        ts = f"2026-06-{(i % 28) + 1:02d}T00:00:00Z"
        # Alternate between stable and convective
        regime = "convective" if i % 2 == 0 else "stable"
        true_val = 30.0 if regime == "convective" else 5.0

        # Model A: good at stable (err ~ 0.5), bad at convective (err ~ 15.0)
        # Model B: good at convective (err ~ 0.5), bad at stable (err ~ 10.0)
        fa = true_val + (rng.normal(0, 0.5) if regime == "stable" else rng.normal(12, 1))
        fb = true_val + (rng.normal(0, 0.5) if regime == "convective" else rng.normal(-8, 1))

        rows.append({
            "model_name": "model_a",
            "timestamp": ts,
            "variable": "temperature",
            "lead_time": 24,
            "region": "coastal",
            "latitude": 20.0,
            "longitude": 75.0,
            "weather_regime": regime,
            "forecast_value": float(np.clip(fa, 0, 50)),
            "observed_value": float(true_val),
            "error": float(fa - true_val),
        })
        rows.append({
            "model_name": "model_b",
            "timestamp": ts,
            "variable": "temperature",
            "lead_time": 24,
            "region": "coastal",
            "latitude": 20.0,
            "longitude": 75.0,
            "weather_regime": regime,
            "forecast_value": float(np.clip(fb, 0, 50)),
            "observed_value": float(true_val),
            "error": float(fb - true_val),
        })

    return pd.DataFrame(rows)


def test_gating_pytorch_module_forward_pass():
    """Verify raw PyTorch module outputs valid softmax simplex weights."""
    batch_size = 8
    input_dim = 10
    num_models = 3

    module = GatingPyTorchModule(input_dim=input_dim, num_models=num_models, hidden_sizes=[32, 16], dropout=0.0)
    x = torch.randn(batch_size, input_dim)

    weights, residual = module(x)
    assert weights.shape == (batch_size, num_models)
    assert residual is None  # default disabled

    # Check softmax properties: non-negative and sum to 1.0
    w_np = weights.detach().cpu().numpy()
    assert (w_np >= 0.0).all()
    assert np.allclose(w_np.sum(axis=1), 1.0, atol=1e-5)


def test_gating_pytorch_module_residual_head():
    """Verify residual head returns additive correction tensor when enabled."""
    module = GatingPyTorchModule(input_dim=5, num_models=2, use_residual_head=True)
    x = torch.randn(4, 5)
    weights, residual = module(x)

    assert residual is not None
    assert residual.shape == (4, 1)


def test_weather_regime_gating_network_fit_and_predict(synthetic_gating_training_data):
    """Verify full end-to-end training and prediction of WeatherRegimeGatingNetwork."""
    gating_net = WeatherRegimeGatingNetwork(
        hidden_sizes=[32, 16],
        epochs=15,
        batch_size=16,
        learning_rate=0.01,
        random_seed=42,
    )

    gating_net.fit_verification_data(synthetic_gating_training_data)
    assert gating_net.is_fitted
    assert gating_net.module is not None
    assert len(gating_net.known_models) == 2

    test_forecasts = pd.DataFrame([
        {
            "model_name": "model_a",
            "timestamp": "2026-07-01T00:00:00Z",
            "variable": "temperature",
            "lead_time": 24,
            "region": "coastal",
            "latitude": 20.0,
            "longitude": 75.0,
            "weather_regime": "stable",
            "forecast_value": 5.2,
        },
        {
            "model_name": "model_b",
            "timestamp": "2026-07-01T00:00:00Z",
            "variable": "temperature",
            "lead_time": 24,
            "region": "coastal",
            "latitude": 20.0,
            "longitude": 75.0,
            "weather_regime": "stable",
            "forecast_value": 14.8,
        },
    ])

    blended = gating_net.predict(test_forecasts)
    assert len(blended) == 1
    assert blended.iloc[0]["model_name"] == "neural_gating_blend"
    assert "forecast_value" in blended.columns

    # In stable condition, Model A is accurate (~5.0) and Model B is inaccurate (~15.0)
    # Blended forecast should be heavily weighted towards Model A (closer to 5.2 than 14.8)
    assert blended.iloc[0]["forecast_value"] < 10.0


def test_gating_get_weights_simplex():
    """Verify get_weights method returns non-negative weights summing to 1.0."""
    gating_net = WeatherRegimeGatingNetwork()
    gating_net.known_models = ["model_a", "model_b", "model_c"]
    weights = gating_net.get_weights(variable="temperature", lead_time=24, region="coastal")

    assert len(weights) == 3
    assert np.isclose(sum(weights.values()), 1.0, atol=1e-5)
    for w in weights.values():
        assert w >= 0.0
