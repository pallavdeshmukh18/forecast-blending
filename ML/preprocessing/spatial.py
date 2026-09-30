"""
Spatial alignment and nearest-neighbour geodesic matching routines for meteorological grids and stations.
"""

from typing import Optional, Tuple
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from configs import AppConfig, get_config

# Earth mean radius in kilometers (IUGG standard)
EARTH_RADIUS_KM = 6371.0088


def haversine_distance_km(
    lat1: np.ndarray | float,
    lon1: np.ndarray | float,
    lat2: np.ndarray | float,
    lon2: np.ndarray | float,
) -> np.ndarray | float:
    """
    Compute great-circle distance between pairs of coordinates using the Haversine formula.

    Parameters
    ----------
    lat1, lon1 : float or array-like
        First latitude/longitude in decimal degrees.
    lat2, lon2 : float or array-like
        Second latitude/longitude in decimal degrees.

    Returns
    -------
    float or np.ndarray
        Distance in kilometers.
    """
    phi1, lambda1 = np.radians(lat1), np.radians(lon1)
    phi2, lambda2 = np.radians(lat2), np.radians(lon2)

    dphi = phi2 - phi1
    dlambda = lambda2 - lambda1

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    c = 2.0 * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))

    return EARTH_RADIUS_KM * c


def _lat_lon_to_cartesian(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """Convert spherical latitude/longitude in degrees to 3D Cartesian coordinates on unit sphere."""
    phi = np.radians(lat_deg)
    lam = np.radians(lon_deg)
    x = np.cos(phi) * np.cos(lam)
    y = np.cos(phi) * np.sin(lam)
    z = np.sin(phi)
    return np.column_stack((x, y, z))


def match_spatial_nearest(
    obs_coords: pd.DataFrame,
    forecast_coords: pd.DataFrame,
    max_distance_km: float = 50.0,
) -> pd.DataFrame:
    """
    Find the nearest forecast grid coordinate for each observation point.

    Parameters
    ----------
    obs_coords : pd.DataFrame
        DataFrame containing ['latitude', 'longitude'] (and optionally 'station_id').
    forecast_coords : pd.DataFrame
        DataFrame containing unique forecast grid ['latitude', 'longitude'].
    max_distance_km : float, default 50.0
        Maximum allowable spatial distance in kilometers. Matches exceeding this threshold are marked as unmatched.

    Returns
    -------
    pd.DataFrame
        Mapping DataFrame with columns:
        - 'obs_latitude', 'obs_longitude'
        - 'forecast_latitude', 'forecast_longitude'
        - 'spatial_distance_km'
        - 'is_spatial_matched' (bool)
    """
    unique_obs = obs_coords[["latitude", "longitude"]].drop_duplicates().reset_index(drop=True)
    unique_fcs = forecast_coords[["latitude", "longitude"]].drop_duplicates().reset_index(drop=True)

    if unique_obs.empty or unique_fcs.empty:
        return pd.DataFrame(
            columns=[
                "obs_latitude",
                "obs_longitude",
                "forecast_latitude",
                "forecast_longitude",
                "spatial_distance_km",
                "is_spatial_matched",
            ]
        )

    # Build 3D spatial KD-Tree on unit sphere
    fcs_xyz = _lat_lon_to_cartesian(unique_fcs["latitude"].values, unique_fcs["longitude"].values)
    obs_xyz = _lat_lon_to_cartesian(unique_obs["latitude"].values, unique_obs["longitude"].values)

    tree = cKDTree(fcs_xyz)
    _, nearest_indices = tree.query(obs_xyz, k=1)

    matched_fcs = unique_fcs.iloc[nearest_indices].reset_index(drop=True)

    # Compute exact Haversine great-circle distances
    distances = haversine_distance_km(
        lat1=unique_obs["latitude"].values,
        lon1=unique_obs["longitude"].values,
        lat2=matched_fcs["latitude"].values,
        lon2=matched_fcs["longitude"].values,
    )

    mapping_df = pd.DataFrame({
        "obs_latitude": unique_obs["latitude"].values,
        "obs_longitude": unique_obs["longitude"].values,
        "forecast_latitude": matched_fcs["latitude"].values,
        "forecast_longitude": matched_fcs["longitude"].values,
        "spatial_distance_km": np.round(distances, 3),
        "is_spatial_matched": distances <= max_distance_km,
    })

    return mapping_df
