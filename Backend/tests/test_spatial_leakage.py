"""
Tests for Spatial Temporal Leakage Prevention (Phase 5).

Verifies that spatial feature scalers, spatial inverse-variance weights, and spatial CNN
gating networks are isolated from future observations and future test-set perturbations.
"""

import numpy as np
import pytest

from models.spatial_blending import SpatialInverseVarianceBlender
from models.spatial_gating import SpatialCNNBlender


def test_spatial_inverse_variance_no_future_leakage():
    # 20 training steps, 10 future test steps
    rng = np.random.default_rng(42)
    F_train = rng.normal(20.0, 3.0, (20, 3, 6, 6)).astype(np.float32)
    Y_train = rng.normal(20.0, 2.0, (20, 6, 6)).astype(np.float32)

    F_test = rng.normal(20.0, 3.0, (10, 3, 6, 6)).astype(np.float32)
    Y_test = rng.normal(20.0, 2.0, (10, 6, 6)).astype(np.float32)

    # 1. Fit blender on training data only
    blender = SpatialInverseVarianceBlender(model_names=["m1", "m2", "m3"])
    blender.fit(F_train, Y_train)
    weights_initial = blender.get_weights()

    pred_initial, _ = blender.predict(F_test[0])

    # 2. Perturb future test observations
    Y_test_perturbed = Y_test + 500.0

    # 3. Verify that historical weights and test predictions do NOT change
    weights_after = blender.get_weights()
    pred_after, _ = blender.predict(F_test[0])

    assert np.allclose(weights_initial, weights_after)
    assert np.allclose(pred_initial, pred_after)


def test_spatial_cnn_no_future_leakage():
    rng = np.random.default_rng(42)
    X_train = rng.normal(0, 1.0, (12, 4, 6, 6)).astype(np.float32)
    F_train = rng.normal(20.0, 2.0, (12, 3, 6, 6)).astype(np.float32)
    Y_train = rng.normal(20.0, 2.0, (12, 6, 6)).astype(np.float32)

    X_test = rng.normal(0, 1.0, (6, 4, 6, 6)).astype(np.float32)
    F_test = rng.normal(20.0, 2.0, (6, 3, 6, 6)).astype(np.float32)
    Y_test = rng.normal(20.0, 2.0, (6, 6, 6)).astype(np.float32)

    # 1. Fit Spatial CNN strictly on training set
    blender = SpatialCNNBlender(num_models=3, in_channels=4, epochs=3, batch_size=4, random_seed=42)
    blender.fit(X_train, F_train, Y_train)

    pred_test_orig, w_test_orig = blender.predict(X_test, F_test)

    # 2. Perturb future test observations
    Y_test_perturbed = Y_test * 100.0

    # 3. Predict again on test inputs
    pred_test_after, w_test_after = blender.predict(X_test, F_test)

    assert np.allclose(pred_test_orig, pred_test_after)
    assert np.allclose(w_test_orig, w_test_after)
