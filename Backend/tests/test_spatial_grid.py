"""
Tests for Spatial Grid Representation & Alignment (Phase 5).
"""

import numpy as np
import pytest
import xarray as xr

from preprocessing.spatial_grid import GridDefinition, SpatialGridAligner


def test_grid_definition_initialization():
    gd = GridDefinition(lat_min=10.0, lat_max=30.0, lon_min=70.0, lon_max=85.0, n_lat=10, n_lon=15)
    assert gd.shape == (10, 15)
    assert len(gd.latitudes) == 10
    assert len(gd.longitudes) == 15
    assert gd.lat_min == 10.0
    assert gd.lat_max == 30.0
    assert gd.contains_point(20.0, 75.0)
    assert not gd.contains_point(5.0, 75.0)

    mesh_lat, mesh_lon = gd.get_mesh()
    assert mesh_lat.shape == (10, 15)
    assert mesh_lon.shape == (10, 15)


def test_grid_definition_invalid_bounds():
    with pytest.raises(ValueError):
        GridDefinition(lat_min=30.0, lat_max=10.0)


def test_spatial_grid_aligner_regular_grid():
    target_gd = GridDefinition(lat_min=10.0, lat_max=20.0, lon_min=70.0, lon_max=80.0, n_lat=10, n_lon=10)
    aligner = SpatialGridAligner(target_grid=target_gd, interpolation="bilinear")

    # Source regular grid (5x5)
    src_lats = np.linspace(10.0, 20.0, 5)
    src_lons = np.linspace(70.0, 80.0, 5)
    src_values = np.arange(25, dtype=np.float32).reshape(5, 5)

    aligned, diag = aligner.align_array_2d(src_values, src_lats, src_lons)
    assert aligned.shape == (10, 10)
    assert diag["is_valid"] is True
    assert diag["missing_fraction"] == 0.0
    assert diag["coverage_fraction"] == 1.0


def test_spatial_grid_aligner_nearest():
    target_gd = GridDefinition(lat_min=10.0, lat_max=20.0, lon_min=70.0, lon_max=80.0, n_lat=8, n_lon=8)
    aligner = SpatialGridAligner(target_grid=target_gd, interpolation="nearest")

    src_lats = np.linspace(10.0, 20.0, 4)
    src_lons = np.linspace(70.0, 80.0, 4)
    src_values = np.ones((4, 4), dtype=np.float32) * 5.0

    aligned, diag = aligner.align_array_2d(src_values, src_lats, src_lons)
    assert aligned.shape == (8, 8)
    assert np.allclose(aligned, 5.0)


def test_spatial_grid_aligner_xarray_dataset():
    target_gd = GridDefinition(lat_min=10.0, lat_max=20.0, lon_min=70.0, lon_max=80.0, n_lat=8, n_lon=8)
    aligner = SpatialGridAligner(target_grid=target_gd, interpolation="bilinear")

    # Build small xarray Dataset
    lats = np.linspace(10.0, 20.0, 4)
    lons = np.linspace(70.0, 80.0, 4)
    ds = xr.Dataset(
        data_vars={"temperature": (["latitude", "longitude"], np.ones((4, 4), dtype=np.float32) * 22.0)},
        coords={"latitude": lats, "longitude": lons},
    )

    aligned_ds = aligner.align_dataset(ds)
    assert aligned_ds["temperature"].shape == (8, 8)
    assert np.allclose(aligned_ds["temperature"].values, 22.0)
