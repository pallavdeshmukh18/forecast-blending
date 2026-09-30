"""
Tests for Spatial Mixture-of-Experts Gating Network (Phase 5).
"""

import numpy as np
import torch
import pytest

from models.spatial_gating import SpatialGatingNetwork, SpatialCNNBlender


def test_spatial_gating_network_forward():
    net = SpatialGatingNetwork(in_channels=8, num_models=3, channels=[16, 32], kernel_size=3)
    x = torch.randn(2, 8, 12, 12)
    f = torch.randn(2, 3, 12, 12)

    blended, weights = net(x, f)
    assert blended.shape == (2, 1, 12, 12)
    assert weights.shape == (2, 3, 12, 12)

    # Check softmax simplex constraints: w >= 0 and sum(w) = 1
    assert torch.all(weights >= 0.0)
    w_sum = torch.sum(weights, dim=1)
    assert torch.allclose(w_sum, torch.ones_like(w_sum), atol=1e-5)


def test_spatial_cnn_blender_fit_predict():
    blender = SpatialCNNBlender(
        num_models=3,
        in_channels=6,
        channels=[16, 32],
        epochs=3,
        batch_size=4,
        random_seed=42,
    )

    X_train = np.random.randn(8, 6, 8, 8).astype(np.float32)
    F_train = np.random.randn(8, 3, 8, 8).astype(np.float32)
    Y_train = np.random.randn(8, 8, 8).astype(np.float32)

    blender.fit(X_train, F_train, Y_train)
    blended, weights = blender.predict(X_train[:2], F_train[:2])

    assert blended.shape == (2, 8, 8)
    assert weights.shape == (2, 3, 8, 8)
    assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-5)
