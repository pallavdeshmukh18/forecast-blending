"""
Grid-based meteorological file readers and standardization module for Phase 5.

Provides direct xarray Dataset ingestion for NetCDF and GRIB2 gridded forecast and
observation files, along with variable and unit standardization.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import xarray as xr

from configs import AppConfig, get_config
from ingestion.variable_mapping import standardize_variable_name
from ingestion.units import convert_units


def load_forecast_grid(
    filepath: Union[str, Path],
    variables: Optional[List[str]] = None,
    engine: Optional[str] = None,
    **kwargs,
) -> xr.Dataset:
    """
    Load a gridded forecast file (NetCDF / GRIB2) directly into an xarray.Dataset.

    Parameters
    ----------
    filepath : Union[str, Path]
    variables : Optional[List[str]]
    engine : Optional[str]

    Returns
    -------
    xr.Dataset
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Gridded forecast file not found: {path}")

    ext = path.suffix.lower()
    if engine is None:
        if ext in [".grib", ".grib2", ".grb2"]:
            engine = "cfgrib"
        else:
            engine = "netcdf4" if "netCDF4" in xr.backends.list_engines() else "scipy"

    try:
        ds = xr.open_dataset(path, engine=engine, **kwargs)
    except Exception:
        # Fallback to default engine
        ds = xr.open_dataset(path, **kwargs)

    if variables:
        present_vars = [v for v in variables if v in ds.data_vars]
        if present_vars:
            ds = ds[present_vars]

    return ds


def load_observation_grid(
    filepath: Union[str, Path],
    variables: Optional[List[str]] = None,
    engine: Optional[str] = None,
    **kwargs,
) -> xr.Dataset:
    """
    Load a gridded observation/reanalysis file directly into an xarray.Dataset.
    """
    return load_forecast_grid(filepath, variables=variables, engine=engine, **kwargs)


# Alias for general grid file loading
load_grid_weather_file = load_forecast_grid


def standardize_grid_variables(
    ds: xr.Dataset,
    config: Optional[AppConfig] = None,
) -> xr.Dataset:
    """
    Standardize coordinates and variable names in an xarray Dataset according to project conventions.
    Converts coordinate names to standard ('latitude', 'longitude', 'time', 'lead_time', 'model'),
    maps variable aliases to standard names ('temperature', 'rainfall', 'wind_speed', 'surface_pressure', 'relative_humidity'),
    and converts units (e.g. Kelvin -> Celsius, Pa -> hPa).
    """
    if config is None:
        config = get_config()

    renamed_coords = {}
    for c in ds.coords:
        c_lower = str(c).lower()
        if c_lower in ["lat", "lats", "nav_lat"]:
            renamed_coords[c] = "latitude"
        elif c_lower in ["lon", "lons", "nav_lon", "long"]:
            renamed_coords[c] = "longitude"
        elif c_lower in ["valid_time", "datetime", "date"]:
            renamed_coords[c] = "time"
        elif c_lower in ["step", "forecast_period", "lead"]:
            renamed_coords[c] = "lead_time"

    if renamed_coords:
        ds = ds.rename(renamed_coords)

    # Standardize data variable names
    renamed_vars = {}
    for var in list(ds.data_vars):
        std_name = standardize_variable_name(str(var), config)
        if std_name is not None and std_name != var and std_name not in ds.data_vars:
            renamed_vars[var] = std_name

    if renamed_vars:
        ds = ds.rename(renamed_vars)

    # Standardize Units
    if "temperature" in ds:
        # Check if Kelvin (e.g. mean > 200)
        temp_vals = ds["temperature"].values
        valid_t = temp_vals[~np.isnan(temp_vals)]
        if len(valid_t) > 0 and np.mean(valid_t) > 150.0:
            ds["temperature"] = ds["temperature"] - 273.15
            ds["temperature"].attrs["unit"] = "celsius"

    if "surface_pressure" in ds:
        # Check if Pa (e.g. mean > 50000)
        p_vals = ds["surface_pressure"].values
        valid_p = p_vals[~np.isnan(p_vals)]
        if len(valid_p) > 0 and np.mean(valid_p) > 20000.0:
            ds["surface_pressure"] = ds["surface_pressure"] / 100.0
            ds["surface_pressure"].attrs["unit"] = "hPa"

    return ds
