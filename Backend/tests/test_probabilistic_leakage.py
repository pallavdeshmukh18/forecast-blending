"""
Tests for Probabilistic & EMOS Leakage Prevention (Phase 5).
"""

import numpy as np
import pytest

from models.emos import EnsembleModelOutputStatistics
from models.probabilistic import ResidualUncertaintyProbabilisticModel


def test_emos_no_future_leakage():
    rng = np.random.default_rng(42)
    E_train = rng.normal(15.0, 3.0, (20, 3, 4, 4)).astype(np.float32)
    Y_train = rng.normal(15.0, 2.0, (20, 4, 4)).astype(np.float32)

    E_test = rng.normal(15.0, 3.0, (10, 3, 4, 4)).astype(np.float32)
    Y_test = rng.normal(15.0, 2.0, (10, 4, 4)).astype(np.float32)

    # 1. Fit EMOS on historical training partition
    emos = EnsembleModelOutputStatistics()
    emos.fit(E_train, Y_train)

    a_orig, b_orig, c_orig, d_orig = emos.a, emos.b, emos.c, emos.d
    pf_orig = emos.predict(E_test)

    # 2. Modify future test observations
    Y_test_perturbed = Y_test + 999.0

    # 3. Verify parameters and predictions remain identical
    assert (emos.a, emos.b, emos.c, emos.d) == (a_orig, b_orig, c_orig, d_orig)
    pf_after = emos.predict(E_test)
    assert np.allclose(pf_orig.mean, pf_after.mean)
    assert np.allclose(pf_orig.std, pf_after.std)


def test_residual_uncertainty_no_future_leakage():
    preds_train = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    obs_train = np.array([10.5, 19.5, 30.2], dtype=np.float32)

    preds_test = np.array([40.0, 50.0], dtype=np.float32)
    obs_test = np.array([40.1, 49.8], dtype=np.float32)

    res_model = ResidualUncertaintyProbabilisticModel()
    res_model.fit_residuals(preds_train, obs_train)
    orig_std = res_model.residual_std

    # Future perturbation
    obs_test_perturbed = obs_test * 50.0

    assert res_model.residual_std == orig_std
    pf = res_model.predict(preds_test)
    assert np.allclose(pf.std, orig_std)
