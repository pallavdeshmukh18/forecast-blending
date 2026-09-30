"""
Synthetic 2D Gridded Weather & Multi-Model Forecast Generator for Phase 5.

Generates realistic 2D meteorological fields with spatial gradients, moving convective
weather systems, orographic features, and regionally heterogeneous model error profiles.
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import xarray as xr

from configs import AppConfig, get_config


def generate_synthetic_spatial_dataset(
    config: Optional[AppConfig] = None,
    start_date: str = "2026-01-01",
    end_date: str = "2026-01-20",
    time_step_hours: int = 6,
    lead_times: Optional[List[int]] = None,
    grid_shape: Tuple[int, int] = (16, 16),
    lat_bounds: Tuple[float, float] = (10.0, 34.0),
    lon_bounds: Tuple[float, float] = (70.0, 86.0),
    seed: int = 42,
) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Generate synthetic 2D gridded forecast and observation datasets across time, lead times,
    spatial grid (lat, lon), models, and variables.

    Parameters
    ----------
    config : AppConfig, optional
    start_date : str
    end_date : str
    time_step_hours : int
    lead_times : list of int, optional
    grid_shape : tuple of (n_lat, n_lon)
    lat_bounds : tuple of (lat_min, lat_max)
    lon_bounds : tuple of (lon_min, lon_max)
    seed : int

    Returns
    -------
    forecast_ds : xr.Dataset
        Coordinates: (time, lead_time, model, latitude, longitude)
        Variables: temperature, rainfall, wind_speed, surface_pressure, relative_humidity
    obs_ds : xr.Dataset
        Coordinates: (time, latitude, longitude)
        Variables: temperature, rainfall, wind_speed, surface_pressure, relative_humidity, elevation
    """
    if config is None:
        config = get_config()

    if lead_times is None:
        lead_times = [6, 12, 24, 48]

    models = list(config.models.keys())
    rng = np.random.default_rng(seed)

    # 1. Spatial coordinates
    n_lat, n_lon = grid_shape
    lats = np.linspace(lat_bounds[0], lat_bounds[1], n_lat, dtype=np.float32)
    lons = np.linspace(lon_bounds[0], lon_bounds[1], n_lon, dtype=np.float32)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # 2. Temporal coordinates
    timestamps = pd.date_range(start=start_date, end=end_date, freq=f"{time_step_hours}h")
    n_times = len(timestamps)
    n_leads = len(lead_times)
    n_models = len(models)

    # 3. Static Orography / Terrain elevation proxy
    # High elevation in North-West (mountain domain), low in South / East
    elevation_grid = (
        1500.0 * np.exp(-((lat_grid - 32.0) ** 2 / 20.0 + (lon_grid - 75.0) ** 2 / 20.0))
        + 800.0 * np.exp(-((lat_grid - 28.0) ** 2 / 30.0 + (lon_grid - 73.0) ** 2 / 25.0))
        + 100.0
    ).astype(np.float32)

    # Initialize observation arrays (n_times, n_lat, n_lon)
    obs_temp = np.zeros((n_times, n_lat, n_lon), dtype=np.float32)
    obs_rain = np.zeros((n_times, n_lat, n_lon), dtype=np.float32)
    obs_wind = np.zeros((n_times, n_lat, n_lon), dtype=np.float32)
    obs_press = np.zeros((n_times, n_lat, n_lon), dtype=np.float32)
    obs_rh = np.zeros((n_times, n_lat, n_lon), dtype=np.float32)

    # Initialize forecast arrays (n_times, n_leads, n_models, n_lat, n_lon)
    fc_temp = np.zeros((n_times, n_leads, n_models, n_lat, n_lon), dtype=np.float32)
    fc_rain = np.zeros((n_times, n_leads, n_models, n_lat, n_lon), dtype=np.float32)
    fc_wind = np.zeros((n_times, n_leads, n_models, n_lat, n_lon), dtype=np.float32)
    fc_press = np.zeros((n_times, n_leads, n_models, n_lat, n_lon), dtype=np.float32)
    fc_rh = np.zeros((n_times, n_leads, n_models, n_lat, n_lon), dtype=np.float32)

    # Generate Ground Truth Atmosphere
    for t_idx, ts in enumerate(timestamps):
        hour = ts.hour
        diurnal = np.sin(2 * np.pi * (hour - 6) / 24.0)
        synoptic_wave = np.sin(2 * np.pi * (t_idx / 28.0 + lon_grid / 15.0))

        # Temperature: Latitude gradient (warmer south) - lapse rate + diurnal + synoptic wave
        base_t = (
            32.0
            - 0.55 * (lat_grid - 10.0)
            - 0.0065 * elevation_grid
            + 4.0 * diurnal
            + 2.5 * synoptic_wave
            + rng.normal(0, 0.4, size=(n_lat, n_lon))
        )
        obs_temp[t_idx] = np.clip(base_t, -20.0, 50.0).astype(np.float32)

        # Moving convective storm centers (moving from West to East-North-East)
        storm_center_lat = 18.0 + 8.0 * np.sin(2 * np.pi * t_idx / 16.0)
        storm_center_lon = 72.0 + ((t_idx * 1.2) % 14.0)
        dist_sq = (lat_grid - storm_center_lat) ** 2 + (lon_grid - storm_center_lon) ** 2
        storm_rain = 45.0 * np.exp(-dist_sq / 12.0)

        # Orographic precipitation enhancement
        orographic_rain = 8.0 * (elevation_grid / 1500.0) * np.maximum(0.0, np.sin(2 * np.pi * t_idx / 8.0))
        total_rain = storm_rain + orographic_rain + rng.exponential(0.5, size=(n_lat, n_lon))
        total_rain[total_rain < 1.0] = 0.0
        obs_rain[t_idx] = np.clip(total_rain, 0.0, 300.0).astype(np.float32)

        # Wind speed: Pressure gradient + convective gust
        base_wind = (
            5.0
            + 3.5 * np.abs(synoptic_wave)
            + 0.3 * storm_rain
            + 0.002 * elevation_grid
            + rng.normal(0, 0.5, size=(n_lat, n_lon))
        )
        obs_wind[t_idx] = np.clip(base_wind, 0.0, 60.0).astype(np.float32)

        # Surface pressure: barometric formula proxy
        base_press = 1013.25 * np.exp(-elevation_grid / 8400.0) - 12.0 * synoptic_wave - 0.15 * storm_rain
        obs_press[t_idx] = base_press.astype(np.float32)

        # Relative humidity: correlated with rain and negative temp
        base_rh = 60.0 + 0.8 * storm_rain - 0.7 * (base_t - 20.0) + rng.normal(0, 3.0, size=(n_lat, n_lon))
        obs_rh[t_idx] = np.clip(base_rh, 15.0, 100.0).astype(np.float32)

    # Generate Multi-Model Forecasts with Region-Specific & Regime-Specific Error Structures
    for l_idx, lead in enumerate(lead_times):
        lead_factor = 1.0 + (lead / 48.0) * 0.4  # Error growth with lead time

        for m_idx, model_name in enumerate(models):
            for t_idx in range(n_times):
                y_t = obs_temp[t_idx]
                y_r = obs_rain[t_idx]
                y_w = obs_wind[t_idx]
                y_p = obs_press[t_idx]
                y_rh = obs_rh[t_idx]

                if model_name == "nwp_model_a":
                    # Model A: Strong in West (lon < 78), weaker in East (lon >= 78)
                    west_mask = lon_grid < 78.0
                    err_t = np.where(west_mask, rng.normal(0.1, 0.6 * lead_factor, (n_lat, n_lon)), rng.normal(0.5, 1.4 * lead_factor, (n_lat, n_lon)))
                    err_r = np.where(west_mask, rng.normal(0.0, 1.2 * lead_factor, (n_lat, n_lon)), rng.normal(-1.5, 3.5 * lead_factor, (n_lat, n_lon)))
                    err_w = np.where(west_mask, rng.normal(0.0, 0.7 * lead_factor, (n_lat, n_lon)), rng.normal(0.8, 1.8 * lead_factor, (n_lat, n_lon)))

                elif model_name == "nwp_model_b":
                    # Model B: Strong in East (lon >= 78) / plains, weaker in West / mountain elevation
                    east_mask = lon_grid >= 78.0
                    err_t = np.where(east_mask, rng.normal(-0.1, 0.5 * lead_factor, (n_lat, n_lon)), rng.normal(-0.8, 1.5 * lead_factor, (n_lat, n_lon)))
                    err_r = np.where(east_mask, rng.normal(0.0, 1.5 * lead_factor, (n_lat, n_lon)), rng.normal(2.0, 4.0 * lead_factor, (n_lat, n_lon)))
                    err_w = np.where(east_mask, rng.normal(0.0, 0.6 * lead_factor, (n_lat, n_lon)), rng.normal(-1.0, 1.9 * lead_factor, (n_lat, n_lon)))

                else:  # ai_model_c
                    # AI Model C: Superior on non-linear convective rain cells (y_r > 15), slightly noisier in calm background
                    convective_mask = y_r > 15.0
                    err_t = rng.normal(0.0, 0.8 * lead_factor, (n_lat, n_lon))
                    err_r = np.where(convective_mask, rng.normal(0.0, 0.9 * lead_factor, (n_lat, n_lon)), rng.normal(0.2, 2.0 * lead_factor, (n_lat, n_lon)))
                    err_w = np.where(y_w > 12.0, rng.normal(0.0, 0.8 * lead_factor, (n_lat, n_lon)), rng.normal(0.3, 1.2 * lead_factor, (n_lat, n_lon)))

                fc_temp[t_idx, l_idx, m_idx] = np.clip(y_t + err_t, -60.0, 65.0).astype(np.float32)
                fc_rain[t_idx, l_idx, m_idx] = np.clip(y_r + err_r, 0.0, 600.0).astype(np.float32)
                fc_wind[t_idx, l_idx, m_idx] = np.clip(y_w + err_w, 0.0, 120.0).astype(np.float32)
                fc_press[t_idx, l_idx, m_idx] = (y_p + rng.normal(0, 1.0 * lead_factor, (n_lat, n_lon))).astype(np.float32)
                fc_rh[t_idx, l_idx, m_idx] = np.clip(y_rh + rng.normal(0, 4.0 * lead_factor, (n_lat, n_lon)), 0.0, 100.0).astype(np.float32)

    # Build xarray Datasets with standardized coordinates and attributes
    coords_fc = {
        "time": timestamps,
        "lead_time": np.array(lead_times, dtype=np.int32),
        "model": models,
        "latitude": lats,
        "longitude": lons,
    }

    forecast_ds = xr.Dataset(
        data_vars={
            "temperature": (["time", "lead_time", "model", "latitude", "longitude"], fc_temp, {"unit": "celsius", "standard_name": "2m_air_temperature"}),
            "rainfall": (["time", "lead_time", "model", "latitude", "longitude"], fc_rain, {"unit": "mm", "standard_name": "total_precipitation_accumulation"}),
            "wind_speed": (["time", "lead_time", "model", "latitude", "longitude"], fc_wind, {"unit": "m/s", "standard_name": "10m_wind_speed"}),
            "surface_pressure": (["time", "lead_time", "model", "latitude", "longitude"], fc_press, {"unit": "hPa", "standard_name": "surface_air_pressure"}),
            "relative_humidity": (["time", "lead_time", "model", "latitude", "longitude"], fc_rh, {"unit": "%", "standard_name": "relative_humidity"}),
        },
        coords=coords_fc,
        attrs={
            "description": "Synthetic 2D Gridded Multi-Model Weather Forecasts",
            "phase": "Phase 5",
            "seed": seed,
        },
    )

    coords_obs = {
        "time": timestamps,
        "latitude": lats,
        "longitude": lons,
    }

    obs_ds = xr.Dataset(
        data_vars={
            "temperature": (["time", "latitude", "longitude"], obs_temp, {"unit": "celsius", "standard_name": "2m_air_temperature"}),
            "rainfall": (["time", "latitude", "longitude"], obs_rain, {"unit": "mm", "standard_name": "total_precipitation_accumulation"}),
            "wind_speed": (["time", "latitude", "longitude"], obs_wind, {"unit": "m/s", "standard_name": "10m_wind_speed"}),
            "surface_pressure": (["time", "latitude", "longitude"], obs_press, {"unit": "hPa", "standard_name": "surface_air_pressure"}),
            "relative_humidity": (["time", "latitude", "longitude"], obs_rh, {"unit": "%", "standard_name": "relative_humidity"}),
            "elevation": (["latitude", "longitude"], elevation_grid, {"unit": "m", "standard_name": "surface_altitude"}),
        },
        coords=coords_obs,
        attrs={
            "description": "Synthetic 2D Gridded Weather Observations",
            "phase": "Phase 5",
            "seed": seed,
        },
    )

    return forecast_ds, obs_ds


def spatial_dataset_to_tabular(
    forecast_ds: xr.Dataset,
    obs_ds: Optional[xr.Dataset] = None,
    variables: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Convert 2D gridded xarray forecast/observation datasets to tabular DataFrame
    compatible with Phase 1-4 validation schemas.
    """
    if variables is None:
        variables = ["temperature", "rainfall", "wind_speed"]

    records = []
    lats = forecast_ds.latitude.values
    lons = forecast_ds.longitude.values
    times = forecast_ds.time.values
    leads = forecast_ds.lead_time.values
    models = forecast_ds.model.values

    for t in times:
        t_dt = pd.to_datetime(t)
        for lead in leads:
            valid_time = t_dt + pd.Timedelta(hours=int(lead))
            for lat in lats:
                for lon in lons:
                    # Determine region based on coordinates
                    if lat < 20.0:
                        region = "coastal"
                    elif lat < 30.0:
                        region = "plains"
                    else:
                        region = "mountain"

                    for var in variables:
                        if var not in forecast_ds:
                            continue

                        # Extract observation if available
                        obs_val = None
                        if obs_ds is not None and var in obs_ds:
                            if valid_time in obs_ds.time.values:
                                obs_val = float(obs_ds[var].sel(time=valid_time, latitude=lat, longitude=lon).values)
                            elif t in obs_ds.time.values:
                                obs_val = float(obs_ds[var].sel(time=t, latitude=lat, longitude=lon).values)

                        for m in models:
                            fc_val = float(forecast_ds[var].sel(time=t, lead_time=lead, model=m, latitude=lat, longitude=lon).values)
                            row = {
                                "timestamp": t_dt,
                                "valid_time": valid_time,
                                "lead_time_hours": int(lead),
                                "model_name": str(m),
                                "variable": str(var),
                                "latitude": float(lat),
                                "longitude": float(lon),
                                "region": region,
                                "forecast_value": fc_val,
                            }
                            if obs_val is not None:
                                row["observed_value"] = obs_val
                            records.append(row)

    return pd.DataFrame(records)
