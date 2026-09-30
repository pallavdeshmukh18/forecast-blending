"""
Spatial Grid Representation & Alignment Module for Phase 5.

Provides GridDefinition and SpatialGridAligner to align heterogeneous NWP and AI forecast
grids to a unified target meteorological spatial grid using bilinear or nearest-neighbor interpolation.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import scipy.interpolate as interp
import xarray as xr

from configs import AppConfig, get_config


class GridDefinition:
    """
    Defines a regular latitude-longitude spatial grid.
    """

    def __init__(
        self,
        lat_min: float = 10.0,
        lat_max: float = 36.0,
        lon_min: float = 70.0,
        lon_max: float = 86.0,
        resolution_lat: Optional[float] = None,
        resolution_lon: Optional[float] = None,
        n_lat: Optional[int] = None,
        n_lon: Optional[int] = None,
    ):
        if lat_min >= lat_max:
            raise ValueError(f"lat_min ({lat_min}) must be less than lat_max ({lat_max})")
        if lon_min >= lon_max:
            raise ValueError(f"lon_min ({lon_min}) must be less than lon_max ({lon_max})")

        self.lat_min = float(lat_min)
        self.lat_max = float(lat_max)
        self.lon_min = float(lon_min)
        self.lon_max = float(lon_max)

        if n_lat is not None and n_lon is not None:
            self.latitudes = np.linspace(self.lat_min, self.lat_max, n_lat, dtype=np.float32)
            self.longitudes = np.linspace(self.lon_min, self.lon_max, n_lon, dtype=np.float32)
            self.resolution_lat = float(self.latitudes[1] - self.latitudes[0]) if n_lat > 1 else 0.0
            self.resolution_lon = float(self.longitudes[1] - self.longitudes[0]) if n_lon > 1 else 0.0
        elif resolution_lat is not None:
            res_lon = resolution_lon if resolution_lon is not None else resolution_lat
            self.resolution_lat = float(resolution_lat)
            self.resolution_lon = float(res_lon)
            self.latitudes = np.arange(self.lat_min, self.lat_max + self.resolution_lat * 0.5, self.resolution_lat, dtype=np.float32)
            self.longitudes = np.arange(self.lon_min, self.lon_max + self.resolution_lon * 0.5, self.resolution_lon, dtype=np.float32)
        else:
            # Default to 16x16 grid
            self.latitudes = np.linspace(self.lat_min, self.lat_max, 16, dtype=np.float32)
            self.longitudes = np.linspace(self.lon_min, self.lon_max, 16, dtype=np.float32)
            self.resolution_lat = float(self.latitudes[1] - self.latitudes[0])
            self.resolution_lon = float(self.longitudes[1] - self.longitudes[0])

        self.n_lat = len(self.latitudes)
        self.n_lon = len(self.longitudes)
        self.shape = (self.n_lat, self.n_lon)

    @property
    def bounds(self) -> Dict[str, float]:
        return {
            "lat_min": self.lat_min,
            "lat_max": self.lat_max,
            "lon_min": self.lon_min,
            "lon_max": self.lon_max,
        }

    def get_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (lat_grid, lon_grid) 2D arrays."""
        lon_grid, lat_grid = np.meshgrid(self.longitudes, self.latitudes)
        return lat_grid, lon_grid

    def contains_point(self, lat: float, lon: float) -> bool:
        return (self.lat_min <= lat <= self.lat_max) and (self.lon_min <= lon <= self.lon_max)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lat_min": self.lat_min,
            "lat_max": self.lat_max,
            "lon_min": self.lon_min,
            "lon_max": self.lon_max,
            "resolution_lat": self.resolution_lat,
            "resolution_lon": self.resolution_lon,
            "shape": list(self.shape),
            "n_lat": self.n_lat,
            "n_lon": self.n_lon,
        }


class SpatialGridAligner:
    """
    Interpolates and aligns 2D spatial meteorological fields to a target GridDefinition.
    """

    def __init__(
        self,
        target_grid: Optional[GridDefinition] = None,
        interpolation: str = "bilinear",
        max_missing_fraction: float = 0.20,
    ):
        self.target_grid = target_grid or GridDefinition()
        self.interpolation = interpolation.lower()
        if self.interpolation not in ["nearest", "bilinear", "linear"]:
            raise ValueError(f"Unsupported interpolation method '{interpolation}'. Use 'nearest' or 'bilinear'.")
        self.max_missing_fraction = float(max_missing_fraction)

    def align_array_2d(
        self,
        src_values: np.ndarray,
        src_lats: np.ndarray,
        src_lons: np.ndarray,
        fill_value: float = np.nan,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Interpolate a 2D field (n_src_lat, n_src_lon) to the target grid.

        Returns:
            aligned_array: 2D array of shape target_grid.shape
            diagnostics: dict with coverage, missing fraction, validity
        """
        src_lats = np.asarray(src_lats, dtype=np.float64)
        src_lons = np.asarray(src_lons, dtype=np.float64)
        src_values = np.asarray(src_values, dtype=np.float64)

        target_lat_grid, target_lon_grid = self.target_grid.get_mesh()

        # Check if source is strictly regular 1D coordinates
        is_regular = (
            src_lats.ndim == 1
            and src_lons.ndim == 1
            and src_values.shape == (len(src_lats), len(src_lons))
            and np.all(np.diff(src_lats) > 0)
            and np.all(np.diff(src_lons) > 0)
        )

        method = "linear" if self.interpolation in ["bilinear", "linear"] else "nearest"

        if is_regular:
            rgi = interp.RegularGridInterpolator(
                (src_lats, src_lons),
                src_values,
                method=method,
                bounds_error=False,
                fill_value=fill_value,
            )
            # Evaluate on target mesh points
            target_pts = np.stack([target_lat_grid.ravel(), target_lon_grid.ravel()], axis=-1)
            aligned_flat = rgi(target_pts)
            aligned_2d = aligned_flat.reshape(self.target_grid.shape)
        else:
            # Irregular / scattered or flipped coordinates
            if src_lats.ndim == 1:
                src_lon_mesh, src_lat_mesh = np.meshgrid(src_lons, src_lats)
            else:
                src_lat_mesh, src_lon_mesh = src_lats, src_lons

            points = np.stack([src_lat_mesh.ravel(), src_lon_mesh.ravel()], axis=-1)
            values = src_values.ravel()

            valid_mask = ~np.isnan(values)
            if np.sum(valid_mask) < 4:
                aligned_2d = np.full(self.target_grid.shape, fill_value, dtype=np.float32)
            else:
                aligned_2d = interp.griddata(
                    points[valid_mask],
                    values[valid_mask],
                    (target_lat_grid, target_lon_grid),
                    method=method,
                    fill_value=fill_value,
                )

        # Compute interpolation diagnostics
        total_cells = aligned_2d.size
        nan_count = int(np.isnan(aligned_2d).sum())
        missing_fraction = float(nan_count / total_cells) if total_cells > 0 else 1.0
        coverage_fraction = float(1.0 - missing_fraction)
        is_valid = missing_fraction <= self.max_missing_fraction

        diagnostics = {
            "total_cells": total_cells,
            "nan_count": nan_count,
            "missing_fraction": round(missing_fraction, 4),
            "coverage_fraction": round(coverage_fraction, 4),
            "is_valid": is_valid,
            "interpolation_method": self.interpolation,
            "target_shape": list(self.target_grid.shape),
        }

        return aligned_2d.astype(np.float32), diagnostics

    def align_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Interpolate all spatial variables in an xarray Dataset to target_grid.
        Preserves non-spatial coordinates (time, lead_time, model).
        """
        # Build target coords
        target_lats = self.target_grid.latitudes
        target_lons = self.target_grid.longitudes

        method = "linear" if self.interpolation in ["bilinear", "linear"] else "nearest"

        # Use xarray's interp method
        aligned_ds = ds.interp(
            latitude=target_lats,
            longitude=target_lons,
            method=method,
            kwargs={"fill_value": np.nan},
        )
        aligned_ds.attrs["aligned_by"] = "SpatialGridAligner"
        aligned_ds.attrs["interpolation"] = self.interpolation
        return aligned_ds
