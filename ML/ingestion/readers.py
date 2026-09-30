"""
File format decoders and ingestion loaders for real meteorological datasets.
Supports CSV, Parquet, NetCDF (.nc), and GRIB2 (.grib/.grib2) with optional dependency handling.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np


def read_weather_csv(
    filepath: Union[str, Path],
    timestamp_col: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Load meteorological records from a CSV file.

    Parameters
    ----------
    filepath : Union[str, Path]
        Path to CSV file.
    timestamp_col : Optional[str]
        Optional column to parse as datetime.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Meteorological CSV file not found: {path}")

    df = pd.read_csv(path, **kwargs)
    if timestamp_col and timestamp_col in df.columns:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)
    elif "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    return df


def read_weather_parquet(
    filepath: Union[str, Path],
    **kwargs,
) -> pd.DataFrame:
    """
    Load meteorological records from a Parquet file.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Meteorological Parquet file not found: {path}")

    return pd.read_parquet(path, **kwargs)


def read_weather_netcdf(
    filepath: Union[str, Path],
    variables: Optional[List[str]] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Load meteorological grid/station data from a NetCDF (.nc / .nc4) file and convert to DataFrame.
    Requires 'xarray' and an engine such as 'netCDF4', 'scipy', or 'h5netcdf'.

    Parameters
    ----------
    filepath : Union[str, Path]
        Path to NetCDF file.
    variables : Optional[List[str]]
        Specific data variables to extract.

    Returns
    -------
    pd.DataFrame
        Extracted tabular DataFrame.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"NetCDF file not found: {path}")

    try:
        import xarray as xr
    except ImportError as e:
        raise ImportError(
            "NetCDF reading requires 'xarray'. Please install xarray with: pip install xarray"
        ) from e

    try:
        ds = xr.open_dataset(path, **kwargs)
    except Exception as e:
        raise RuntimeError(
            f"Failed to open NetCDF file '{path}'. NetCDF reading may require 'netCDF4', 'scipy', or 'h5netcdf'. "
            f"Original error: {e}"
        ) from e

    try:
        if variables:
            ds = ds[variables]
        df = ds.to_dataframe().reset_index()
    finally:
        ds.close()

    return df


def read_weather_grib2(
    filepath: Union[str, Path],
    filter_by_keys: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Load meteorological grid data from a GRIB / GRIB2 file (.grb2, .grib, .grib2).
    Requires 'cfgrib' and 'ecCodes' libraries.

    Parameters
    ----------
    filepath : Union[str, Path]
        Path to GRIB2 file.
    filter_by_keys : Optional[Dict[str, Any]]
        Optional cfgrib filtering keys (e.g. {'typeOfLevel': 'surface'}).

    Returns
    -------
    pd.DataFrame
        Extracted tabular DataFrame.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"GRIB2 file not found: {path}")

    try:
        import xarray as xr
    except ImportError as e:
        raise ImportError(
            "GRIB2 reading requires 'xarray'. Please install xarray with: pip install xarray"
        ) from e

    try:
        import cfgrib  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "GRIB2 decoding requires 'cfgrib' and 'eccodes'. "
            "Please install cfgrib and ecCodes system libraries (e.g. conda install -c conda-forge cfgrib eccodes)."
        ) from e

    backend_kwargs = {"filter_by_keys": filter_by_keys} if filter_by_keys else {}
    backend_kwargs.update(kwargs.get("backend_kwargs", {}))

    try:
        ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs=backend_kwargs)
        df = ds.to_dataframe().reset_index()
        ds.close()
    except Exception as e:
        raise RuntimeError(
            f"Failed to decode GRIB2 file '{path}'. Ensure ecCodes and cfgrib are properly configured. Error: {e}"
        ) from e

    return df


def load_weather_file(
    filepath: Union[str, Path],
    file_format: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Universal weather file loader with automatic format detection.

    Parameters
    ----------
    filepath : Union[str, Path]
        Path to dataset file.
    file_format : Optional[str]
        Explicit format specifier: 'csv', 'parquet', 'netcdf', 'nc', 'grib', 'grib2'.
        If None, inferred from file extension.

    Returns
    -------
    pd.DataFrame
        Standardized loaded DataFrame.
    """
    path = Path(filepath)
    suffix = path.suffix.lower().lstrip(".")
    fmt = (file_format or suffix).lower()

    if fmt in ["csv", "txt", "tsv"]:
        sep = "\t" if fmt == "tsv" else ","
        return read_weather_csv(path, sep=kwargs.pop("sep", sep), **kwargs)
    elif fmt in ["parquet", "pq"]:
        return read_weather_parquet(path, **kwargs)
    elif fmt in ["nc", "nc4", "netcdf", "hdf"]:
        return read_weather_netcdf(path, **kwargs)
    elif fmt in ["grib", "grib2", "grb", "grb2"]:
        return read_weather_grib2(path, **kwargs)
    else:
        raise ValueError(
            f"Unsupported weather file format '{fmt}' for file '{path}'. "
            "Supported formats: CSV (.csv), Parquet (.parquet), NetCDF (.nc), GRIB2 (.grib/.grib2)."
        )
