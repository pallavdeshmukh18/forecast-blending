"""
Tests for Spatial Blending Baselines (Phase 5).
"""

import numpy as np
import pytest

from models.spatial_blending import (
    SpatialWeightedBlender,
    SpatialEqualWeightBlender,
    SpatialInverseVarianceBlender,
)


def test_spatial_weighted_blender_normalization():
    blender = SpatialWeightedBlender()
    F = np.array([
        [[10.0, 10.0], [10.0, 10.0]],
        [[20.0, 20.0], [20.0, 20.0]],
    ], dtype=np.float32)  # (2, 2, 2)

    # Unnormalized weights
    W = np.array([
        [[2.0, 1.0], [3.0, 0.0]],
        [[2.0, 3.0], [1.0, 4.0]],
    ], dtype=np.float32)

    blended, W_norm = blender.blend_fields(F, W)
    assert blended.shape == (2, 2)
    assert np.allclose(W_norm.sum(axis=0), 1.0)
    assert np.all(W_norm >= 0.0)


def test_spatial_equal_weight_blender():
    blender = SpatialEqualWeightBlender(model_names=["m1", "m2", "m3"])
    F = np.array([
        [[10.0, 10.0], [10.0, 10.0]],
        [[20.0, 20.0], [20.0, 20.0]],
        [[30.0, 30.0], [30.0, 30.0]],
    ], dtype=np.float32)

    blended, W = blender.predict(F)
    assert np.allclose(blended, 20.0)
    assert np.allclose(W, 1.0 / 3.0)


def test_spatial_inverse_variance_blender_fitting():
    blender = SpatialInverseVarianceBlender(model_names=["m1", "m2"], weight_smoothing=0.0)

    # Create synthetic history where m1 is twice as accurate as m2
    hist_f = np.zeros((20, 2, 4, 4), dtype=np.float32)
    hist_y = np.zeros((20, 4, 4), dtype=np.float32)

    # m1 has error variance 1.0, m2 has error variance 4.0
    rng = np.random.default_rng(42)
    hist_f[:, 0] = rng.normal(0, 1.0, (20, 4, 4))
    hist_f[:, 1] = rng.normal(0, 2.0, (20, 4, 4))

    blender.fit(hist_f, hist_y)
    weights = blender.get_weights()

    assert weights.shape == (2, 4, 4)
    assert np.allclose(weights.sum(axis=0), 1.0)
    # Model 1 should have higher weight than Model 2
    assert np.mean(weights[0]) > np.mean(weights[1])
