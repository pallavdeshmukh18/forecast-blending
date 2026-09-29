"""
Tests for Spatial Feature Engineering & Model Disagreement (Phase 5).
"""

import numpy as np
import pytest

from preprocessing.spatial_features import (
    compute_spatial_gradients,
    compute_local_spatial_statistics,
    compute_spatial_anomaly,
    compute_model_disagreement_2d,
    extract_spatial_feature_stack,
)
from data.synthetic_spatial import generate_synthetic_spatial_dataset


def test_spatial_gradients():
    field = np.arange(16, dtype=np.float32).reshape(4, 4)
    gy, gx, gmag = compute_spatial_gradients(field)
    assert gy.shape == (4, 4)
    assert gx.shape == (4, 4)
    assert gmag.shape == (4, 4)
    assert np.all(gmag >= 0)


def test_local_spatial_statistics():
    field = np.ones((8, 8), dtype=np.float32) * 10.0
    field[2:5, 2:5] = 20.0
    l_mean, l_std = compute_local_spatial_statistics(field, window_size=3)
    assert l_mean.shape == (8, 8)
    assert l_std.shape == (8, 8)
    assert np.all(l_std >= 0)


def test_spatial_anomaly():
    field = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    anom = compute_spatial_anomaly(field)
    assert anom.shape == (2, 2)
    assert np.isclose(np.mean(anom), 0.0)


def test_model_disagreement_2d():
    m1 = np.ones((4, 4), dtype=np.float32) * 10.0
    m2 = np.ones((4, 4), dtype=np.float32) * 12.0
    m3 = np.ones((4, 4), dtype=np.float32) * 14.0
    fields = {"model_a": m1, "model_b": m2, "model_c": m3}

    dis = compute_model_disagreement_2d(fields)
    assert np.allclose(dis["mean"], 12.0)
    assert np.allclose(dis["range"], 4.0)
    assert "model_a_vs_model_b" in dis["pairwise_diffs"]
    assert np.allclose(dis["pairwise_diffs"]["model_a_vs_model_b"], 2.0)


def test_extract_spatial_feature_stack():
    fc, obs = generate_synthetic_spatial_dataset(
        start_date="2026-01-01",
        end_date="2026-01-02",
        grid_shape=(8, 8),
    )
    stack, names = extract_spatial_feature_stack(fc, variable="rainfall", lead_time=6, time_idx=0)
    assert stack.ndim == 3
    assert stack.shape[1:] == (8, 8)
    assert len(names) == stack.shape[0]
    assert "nwp_model_a_rainfall" in names
    assert "rainfall_ensemble_mean" in names
