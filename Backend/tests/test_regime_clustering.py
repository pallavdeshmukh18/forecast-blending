"""
Automated unit tests for unsupervised KMeans weather regime clustering and cluster diagnostics.
"""

import pytest
import numpy as np
import pandas as pd

from preprocessing.regime_clustering import KMeansRegimeClassifier, evaluate_cluster_range


@pytest.fixture
def sample_weather_training_data():
    """Generate realistic synthetic weather dataset for clustering tests."""
    rng = np.random.default_rng(42)
    n = 200
    dates = pd.date_range("2026-01-01", periods=n, freq="6h", tz="UTC")

    # Three physical weather modes
    t = np.concatenate([
        rng.normal(28, 2, 70),   # Warm/convective
        rng.normal(15, 3, 70),   # Cool/stable
        rng.normal(22, 2, 60),   # Moderate/windy
    ])
    r = np.concatenate([
        rng.exponential(25, 70),  # Heavy rain
        np.zeros(70),             # Dry
        rng.exponential(2, 60),   # Light rain
    ])
    w = np.concatenate([
        rng.normal(4, 1, 70),    # Low wind
        rng.normal(3, 1, 70),    # Calm
        rng.normal(16, 3, 60),   # Gale / high wind
    ])

    return pd.DataFrame({
        "timestamp": dates,
        "temperature": t,
        "rainfall": np.clip(r, 0, None),
        "wind_speed": np.clip(w, 0, None),
        "latitude": 28.5,
        "longitude": 77.2,
    })


def test_kmeans_regime_clustering_fit_and_predict(sample_weather_training_data):
    """Verify KMeans classifier fits and labels clusters with valid IDs."""
    clf = KMeansRegimeClassifier(n_clusters=3, random_state=42)
    clf.fit(sample_weather_training_data)

    assert clf.is_fitted
    assert clf.kmeans is not None
    assert len(clf.active_features) > 0

    pred_df = clf.predict(sample_weather_training_data)
    assert "weather_regime" in pred_df.columns
    assert "regime_id" in pred_df.columns
    assert "regime_confidence" in pred_df.columns

    unique_regimes = pred_df["weather_regime"].unique()
    assert len(unique_regimes) == 3


def test_kmeans_cluster_diagnostics_and_profiles(sample_weather_training_data):
    """Verify silhouette score, inertia, and interpretable centroid profiles are computed."""
    clf = KMeansRegimeClassifier(n_clusters=3, random_state=42)
    clf.fit(sample_weather_training_data)

    diag = clf.get_diagnostics()
    assert "n_clusters" in diag
    assert diag["n_clusters"] == 3
    assert "inertia" in diag
    assert "silhouette_score" in diag
    assert -1.0 <= diag["silhouette_score"] <= 1.0

    profiles = clf.get_cluster_profiles()
    assert len(profiles) == 3
    for c_id, prof in profiles.items():
        assert "mean_temperature" in prof
        assert "mean_rainfall" in prof
        assert "mean_wind_speed" in prof
        assert "sample_count" in prof


def test_evaluate_cluster_range_k3_to_k6(sample_weather_training_data):
    """Verify multi-K diagnostic evaluator computes inertia and silhouette for K in [3, 4, 5, 6]."""
    results = evaluate_cluster_range(
        sample_weather_training_data,
        k_values=[3, 4, 5, 6],
        random_state=42,
    )
    assert len(results) == 4
    k_list = [r["k"] for r in results]
    assert k_list == [3, 4, 5, 6]

    for r in results:
        assert "silhouette_score" in r
        assert "inertia" in r
        assert "cluster_sizes" in r
