"""
Unit tests for SpatioTemporalGatingNetwork and SpatioTemporalCNNBlender.
"""

import numpy as np
import torch
import pytest

from models.spatiotemporal_gating import (
    SpatioTemporalGatingNetwork,
    SpatioTemporalCNNBlender,
)


def test_spatiotemporal_gating_convex_weights():
    B, T_seq, C_in, H, W = 2, 4, 8, 10, 10
    M = 3  # num models

    net = SpatioTemporalGatingNetwork(
        in_channels=C_in,
        num_models=M,
        spatial_channels=[16, 16],
        hidden_dim=16,
        kernel_size=3,
    )

    x_seq = torch.randn(B, T_seq, C_in, H, W)
    f_target = torch.randn(B, M, H, W)

    blended, weights = net(x_seq, f_target)

    assert blended.shape == (B, 1, H, W)
    assert weights.shape == (B, M, H, W)

    # Verify W >= 0
    assert torch.all(weights >= 0.0)

    # Verify sum(W) == 1 across models
    weight_sums = torch.sum(weights, dim=1)
    assert torch.allclose(weight_sums, torch.ones(B, H, W), atol=1e-5)


def test_spatiotemporal_cnn_blender_fit_predict():
    np.random.seed(42)
    N, T_seq, C_in, H, W = 10, 4, 6, 8, 8
    M = 3

    X_seq_train = np.random.randn(N, T_seq, C_in, H, W).astype(np.float32)
    F_train = np.random.randn(N, M, H, W).astype(np.float32)
    # Ground truth is linear combination of models
    Y_train = 0.5 * F_train[:, 0] + 0.3 * F_train[:, 1] + 0.2 * F_train[:, 2]

    blender = SpatioTemporalCNNBlender(
        num_models=M,
        in_channels=C_in,
        sequence_length=T_seq,
        spatial_channels=[16, 16],
        hidden_dim=16,
        epochs=3,
        batch_size=4,
        random_seed=42,
    )

    blender.fit(X_seq_train, F_train, Y_train)
    blended_pred, weights = blender.predict(X_seq_train[:2], F_train[:2])

    assert blended_pred.shape == (2, H, W)
    assert weights.shape == (2, M, H, W)
    assert np.all(weights >= 0.0)
    assert np.allclose(np.sum(weights, axis=1), np.ones((2, H, W)), atol=1e-5)
