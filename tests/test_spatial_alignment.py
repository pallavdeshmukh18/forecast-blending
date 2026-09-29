"""
Unit tests for spatial alignment, Haversine geodesic distance, and nearest-neighbor matching.
"""

import pytest
import pandas as pd
import numpy as np

from preprocessing.spatial import haversine_distance_km, match_spatial_nearest


def test_haversine_distance_zero_identical_points():
    d = haversine_distance_km(19.0, 72.8, 19.0, 72.8)
    assert d == pytest.approx(0.0, abs=1e-5)


def test_haversine_distance_known_cities():
    # Mumbai (18.95, 72.82) to Delhi (28.61, 77.20) is approximately 1148 km
    d = haversine_distance_km(18.95, 72.82, 28.61, 77.20)
    assert d == pytest.approx(1148.0, abs=20.0)


def test_match_spatial_nearest_valid_point():
    obs = pd.DataFrame([
        {"latitude": 19.05, "longitude": 72.85, "station_id": "STN_01"},
    ])
    fcs = pd.DataFrame([
        {"latitude": 19.00, "longitude": 72.80},
        {"latitude": 28.00, "longitude": 77.00},
    ])

    mapping = match_spatial_nearest(obs, fcs, max_distance_km=50.0)
    assert len(mapping) == 1
    assert mapping["is_spatial_matched"].iloc[0] is True or mapping["is_spatial_matched"].iloc[0] == 1
    assert mapping["forecast_latitude"].iloc[0] == 19.00
    assert mapping["forecast_longitude"].iloc[0] == 72.80
    assert mapping["spatial_distance_km"].iloc[0] < 20.0


def test_match_spatial_nearest_max_distance_rejection():
    obs = pd.DataFrame([
        {"latitude": 19.00, "longitude": 72.80, "station_id": "STN_01"},
    ])
    # Distant forecast point > 500 km away
    fcs = pd.DataFrame([
        {"latitude": 25.00, "longitude": 80.00},
    ])

    mapping = match_spatial_nearest(obs, fcs, max_distance_km=50.0)
    assert len(mapping) == 1
    # Must be flagged as unmatched
    assert bool(mapping["is_spatial_matched"].iloc[0]) is False
    assert mapping["spatial_distance_km"].iloc[0] > 500.0
