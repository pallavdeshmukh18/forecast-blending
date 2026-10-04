"""
Data Provider Adapter hierarchy for Numerical Weather Prediction (NWP),
AI meteorological models, and ground-truth observations.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from configs import AppConfig, get_config
from ingestion.schema import (
    validate_forecast_dataframe,
    validate_observation_dataframe,
)
from ingestion.variable_mapping import standardize_dataframe_variables, standardize_variable_name
from ingestion.units import normalize_dataframe_units
from ingestion.quality_control import apply_quality_control, DataQualityReport
from ingestion.readers import load_weather_file


# =====================================================================
# Forecast Provider Hierarchy
# =====================================================================

class BaseForecastProvider(ABC):
    """
    Abstract Base Class for all external forecast provider adapters (NWP, AI, Ensembles, File archives).
    """

    def __init__(
        self,
        provider_name: str,
        model_type: str = "nwp",
        default_model_name: Optional[str] = None,
        source_units: Optional[Dict[str, str]] = None,
        config: Optional[AppConfig] = None,
    ):
        self.provider_name = provider_name
        self.model_type = model_type  # 'nwp', 'ai', 'ensemble', 'statistical'
        self.default_model_name = default_model_name or provider_name
        self.source_units = source_units or {}
        self.config = config or get_config()
        self.last_quality_report: Optional[DataQualityReport] = None

    @abstractmethod
    def fetch_forecast(
        self,
        run_times: Optional[List[datetime]] = None,
        lead_times: Optional[List[int]] = None,
        variables: Optional[List[str]] = None,
        region: Optional[str] = None,
        file_path: Optional[Union[str, Path]] = None,
    ) -> pd.DataFrame:
        """
        Fetch forecast records and return DataFrame conforming to standard forecast schema.
        """
        pass

    def process_raw_dataframe(
        self,
        df: pd.DataFrame,
        model_name: Optional[str] = None,
        variable_col: str = "variable",
        value_col: str = "forecast_value",
        source_units: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """
        Standardize variables, normalize units, apply quality control, and validate schema.
        """
        raw = df.copy()

        # Rename common external column aliases to standard schema columns
        rename_map = {
            "valid_time": "timestamp",
            "time": "timestamp",
            "forecast_time": "timestamp",
            "lat": "latitude",
            "lon": "longitude",
            "lead": "lead_time",
            "leadtime": "lead_time",
            "lead_hours": "lead_time",
            "step": "lead_time",
            "val": "forecast_value",
            "value": "forecast_value",
            "model": "model_name",
            "var": "variable",
            "param": "variable",
            "init_time": "run_time",
            "runtime": "run_time",
            "cycle": "run_time",
        }
        raw = raw.rename(columns={k: v for k, v in rename_map.items() if k in raw.columns and v not in raw.columns})

        # Ensure model_name is set
        active_model = model_name or self.default_model_name
        if "model_name" not in raw.columns:
            raw["model_name"] = active_model
        else:
            raw["model_name"] = raw["model_name"].fillna(active_model)

        # Standardize variable names
        if variable_col in raw.columns:
            raw = standardize_dataframe_variables(raw, variable_col=variable_col, config=self.config)

        # Normalize units
        units_to_use = source_units or self.source_units
        if units_to_use:
            raw = normalize_dataframe_units(
                raw,
                value_col=value_col,
                variable_col=variable_col,
                source_units=units_to_use,
                config=self.config,
            )

        # Apply QC physical bounds & checks
        valid_df, _, report = apply_quality_control(
            raw,
            value_col=value_col,
            variable_col=variable_col,
            config=self.config,
        )
        self.last_quality_report = report

        # Final schema validation
        return validate_forecast_dataframe(valid_df, enforce_utc=True)

    def load_from_file(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """
        Load forecasts from a local file (CSV, Parquet, NetCDF, GRIB2) and standardize.
        """
        df = load_weather_file(file_path, **kwargs)
        return self.process_raw_dataframe(df)


class GFSProvider(BaseForecastProvider):
    """
    Adapter for NOAA Global Forecast System (GFS) 0.25° / GRIB2 / Open Data products.
    Default raw units: Temperature in Kelvin, Rainfall in meters (or kg/m^2), Wind in m/s.
    """

    def __init__(
        self,
        provider_name: str = "gfs",
        default_model_name: str = "nwp_model_a",
        config: Optional[AppConfig] = None,
    ):
        source_units = {
            "temperature": "kelvin",
            "rainfall": "m",
            "wind_speed": "m/s",
        }
        super().__init__(
            provider_name=provider_name,
            model_type="nwp",
            default_model_name=default_model_name,
            source_units=source_units,
            config=config,
        )

    def fetch_forecast(
        self,
        run_times: Optional[List[datetime]] = None,
        lead_times: Optional[List[int]] = None,
        variables: Optional[List[str]] = None,
        region: Optional[str] = None,
        file_path: Optional[Union[str, Path]] = None,
    ) -> pd.DataFrame:
        if file_path:
            return self.load_from_file(file_path)

        raise NotImplementedError(
            f"Direct live API / NOMADS downloading for '{self.provider_name}' is not enabled in offline mode. "
            "Please provide a local file_path (CSV, Parquet, NetCDF, or GRIB2)."
        )


class ECMWFProvider(BaseForecastProvider):
    """
    Adapter for ECMWF IFS (Integrated Forecasting System) open forecast data.
    Default raw units: Temperature in Kelvin, Total Precipitation in meters, Wind Speed in m/s.
    """

    def __init__(
        self,
        provider_name: str = "ecmwf_ifs",
        default_model_name: str = "nwp_model_b",
        config: Optional[AppConfig] = None,
    ):
        source_units = {
            "temperature": "kelvin",
            "rainfall": "m",
            "wind_speed": "m/s",
        }
        super().__init__(
            provider_name=provider_name,
            model_type="nwp",
            default_model_name=default_model_name,
            source_units=source_units,
            config=config,
        )

    def fetch_forecast(
        self,
        run_times: Optional[List[datetime]] = None,
        lead_times: Optional[List[int]] = None,
        variables: Optional[List[str]] = None,
        region: Optional[str] = None,
        file_path: Optional[Union[str, Path]] = None,
    ) -> pd.DataFrame:
        if file_path:
            return self.load_from_file(file_path)

        raise NotImplementedError(
            f"Direct live ECMWF Open Data downloading for '{self.provider_name}' is not enabled in offline mode. "
            "Please provide a local file_path (CSV, Parquet, NetCDF, or GRIB2)."
        )


class WRFProvider(BaseForecastProvider):
    """
    Adapter for Weather Research and Forecasting (WRF) mesoscale model outputs.
    Default raw units: Temperature in Celsius, Rain accumulation in mm, Wind in m/s.
    """

    def __init__(
        self,
        provider_name: str = "wrf_mesoscale",
        default_model_name: str = "wrf_model",
        config: Optional[AppConfig] = None,
    ):
        source_units = {
            "temperature": "celsius",
            "rainfall": "mm",
            "wind_speed": "m/s",
        }
        super().__init__(
            provider_name=provider_name,
            model_type="nwp",
            default_model_name=default_model_name,
            source_units=source_units,
            config=config,
        )

    def fetch_forecast(
        self,
        run_times: Optional[List[datetime]] = None,
        lead_times: Optional[List[int]] = None,
        variables: Optional[List[str]] = None,
        region: Optional[str] = None,
        file_path: Optional[Union[str, Path]] = None,
    ) -> pd.DataFrame:
        if file_path:
            return self.load_from_file(file_path)

        raise NotImplementedError(
            f"Direct WRF netCDF runtime generation for '{self.provider_name}' requires local WRF output file. "
            "Please provide a local file_path."
        )


class AIProvider(BaseForecastProvider):
    """
    Adapter for Data-Driven Neural Weather Prediction models (e.g., GraphCast, Pangu-Weather, FourCastNet, ClimaX).
    Default raw units: Temperature in Celsius, Rainfall in mm, Wind Speed in m/s.
    """

    def __init__(
        self,
        provider_name: str = "neural_weather_ai",
        default_model_name: str = "ai_model_c",
        architecture: str = "graph_neural_network",
        config: Optional[AppConfig] = None,
    ):
        self.architecture = architecture
        source_units = {
            "temperature": "celsius",
            "rainfall": "mm",
            "wind_speed": "m/s",
        }
        super().__init__(
            provider_name=provider_name,
            model_type="ai",
            default_model_name=default_model_name,
            source_units=source_units,
            config=config,
        )

    def fetch_forecast(
        self,
        run_times: Optional[List[datetime]] = None,
        lead_times: Optional[List[int]] = None,
        variables: Optional[List[str]] = None,
        region: Optional[str] = None,
        file_path: Optional[Union[str, Path]] = None,
    ) -> pd.DataFrame:
        if file_path:
            return self.load_from_file(file_path)

        raise NotImplementedError(
            f"Live AI model inference execution for '{self.provider_name}' ({self.architecture}) is not active in this phase. "
            "Please provide a pre-computed forecast file_path (CSV, Parquet, or NetCDF)."
        )


class GenericForecastProvider(BaseForecastProvider):
    """
    Universal forecast file adapter with custom unit and model configuration.
    """

    def __init__(
        self,
        provider_name: str = "generic_forecast",
        default_model_name: str = "nwp_model",
        source_units: Optional[Dict[str, str]] = None,
        config: Optional[AppConfig] = None,
    ):
        super().__init__(
            provider_name=provider_name,
            model_type="nwp",
            default_model_name=default_model_name,
            source_units=source_units,
            config=config,
        )

    def fetch_forecast(
        self,
        run_times: Optional[List[datetime]] = None,
        lead_times: Optional[List[int]] = None,
        variables: Optional[List[str]] = None,
        region: Optional[str] = None,
        file_path: Optional[Union[str, Path]] = None,
    ) -> pd.DataFrame:
        if file_path:
            return self.load_from_file(file_path)

        raise ValueError("GenericForecastProvider requires a file_path to load forecasts from.")


# =====================================================================
# Observation Provider Hierarchy
# =====================================================================

class BaseObservationProvider(ABC):
    """
    Abstract Base Class for all ground-truth meteorological observation adapters.
    """

    def __init__(
        self,
        provider_name: str,
        source_units: Optional[Dict[str, str]] = None,
        config: Optional[AppConfig] = None,
    ):
        self.provider_name = provider_name
        self.source_units = source_units or {}
        self.config = config or get_config()
        self.last_quality_report: Optional[DataQualityReport] = None

    @abstractmethod
    def fetch_observations(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        variables: Optional[List[str]] = None,
        region: Optional[str] = None,
        file_path: Optional[Union[str, Path]] = None,
    ) -> pd.DataFrame:
        """
        Fetch ground-truth observations and return DataFrame conforming to standard observation schema.
        """
        pass

    def process_raw_dataframe(
        self,
        df: pd.DataFrame,
        variable_col: str = "variable",
        value_col: str = "observed_value",
        source_units: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """
        Standardize variables, normalize units, apply quality control, and validate schema.
        """
        raw = df.copy()

        # Rename common external column aliases to standard observation schema
        rename_map = {
            "time": "timestamp",
            "obs_time": "timestamp",
            "datetime": "timestamp",
            "lat": "latitude",
            "lon": "longitude",
            "val": "observed_value",
            "value": "observed_value",
            "obs_value": "observed_value",
            "observation": "observed_value",
            "reading": "observed_value",
            "var": "variable",
            "param": "variable",
            "station": "station_id",
            "stn_id": "station_id",
            "stn": "station_id",
            "elev": "elevation",
            "alt": "elevation",
            "altitude": "elevation",
            "flag": "quality_flag",
            "qc_flag": "quality_flag",
        }
        raw = raw.rename(columns={k: v for k, v in rename_map.items() if k in raw.columns and v not in raw.columns})

        # Standardize variable names
        if variable_col in raw.columns:
            raw = standardize_dataframe_variables(raw, variable_col=variable_col, config=self.config)

        # Normalize units
        units_to_use = source_units or self.source_units
        if units_to_use:
            raw = normalize_dataframe_units(
                raw,
                value_col=value_col,
                variable_col=variable_col,
                source_units=units_to_use,
                config=self.config,
            )

        # Apply QC physical bounds & checks
        valid_df, _, report = apply_quality_control(
            raw,
            value_col=value_col,
            variable_col=variable_col,
            config=self.config,
        )
        self.last_quality_report = report

        # Final schema validation
        return validate_observation_dataframe(valid_df, enforce_utc=True)

    def load_from_file(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """
        Load observations from a local file (CSV, Parquet, NetCDF) and standardize.
        """
        df = load_weather_file(file_path, **kwargs)
        return self.process_raw_dataframe(df)


class ERA5Provider(BaseObservationProvider):
    """
    Adapter for ECMWF ERA5 Atmospheric Reanalysis products (used as gridded ground-truth).
    Default raw units: Temperature in Kelvin, Precipitation in meters, Wind Speed in m/s.
    """

    def __init__(
        self,
        provider_name: str = "era5_reanalysis",
        config: Optional[AppConfig] = None,
    ):
        source_units = {
            "temperature": "kelvin",
            "rainfall": "m",
            "wind_speed": "m/s",
        }
        super().__init__(provider_name=provider_name, source_units=source_units, config=config)

    def fetch_observations(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        variables: Optional[List[str]] = None,
        region: Optional[str] = None,
        file_path: Optional[Union[str, Path]] = None,
    ) -> pd.DataFrame:
        if file_path:
            return self.load_from_file(file_path)

        raise NotImplementedError(
            f"Direct live Copernicus CDS downloading for '{self.provider_name}' is not enabled in offline mode. "
            "Please provide a local file_path (CSV, Parquet, or NetCDF)."
        )


class StationProvider(BaseObservationProvider):
    """
    Adapter for in-situ Surface Weather Station networks (e.g. WMO GTS, NOAA ISD, IMD AWS network).
    Default raw units: Temperature in Celsius, Precipitation in mm, Wind Speed in m/s.
    """

    def __init__(
        self,
        provider_name: str = "surface_station_network",
        source_units: Optional[Dict[str, str]] = None,
        config: Optional[AppConfig] = None,
    ):
        units = source_units or {
            "temperature": "celsius",
            "rainfall": "mm",
            "wind_speed": "m/s",
        }
        super().__init__(provider_name=provider_name, source_units=units, config=config)

    def fetch_observations(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        variables: Optional[List[str]] = None,
        region: Optional[str] = None,
        file_path: Optional[Union[str, Path]] = None,
    ) -> pd.DataFrame:
        if file_path:
            return self.load_from_file(file_path)

        raise NotImplementedError(
            f"Direct live station REST API polling for '{self.provider_name}' is not enabled in offline mode. "
            "Please provide a local file_path (CSV or Parquet)."
        )


class GenericObservationProvider(BaseObservationProvider):
    """
    Universal observation file adapter with configurable units.
    """

    def __init__(
        self,
        provider_name: str = "generic_observation",
        source_units: Optional[Dict[str, str]] = None,
        config: Optional[AppConfig] = None,
    ):
        super().__init__(provider_name=provider_name, source_units=source_units, config=config)

    def fetch_observations(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        variables: Optional[List[str]] = None,
        region: Optional[str] = None,
        file_path: Optional[Union[str, Path]] = None,
    ) -> pd.DataFrame:
        if file_path:
            return self.load_from_file(file_path)

        raise ValueError("GenericObservationProvider requires a file_path to load observations from.")


class WeatherBench2ForecastProvider(BaseForecastProvider):
    """
    Adapter for WeatherBench 2 Multi-Model Forecast datasets (e.g., ECMWF HRES, GraphCast, Pangu-Weather).
    Raw units: Temperature in Kelvin, Precipitation in meters, Wind Speed in m/s.
    """

    def __init__(
        self,
        provider_name: str = "weatherbench2_forecast",
        default_model_name: str = "ecmwf_hres",
        config: Optional[AppConfig] = None,
    ):
        source_units = {
            "temperature": "kelvin",
            "rainfall": "m",
            "wind_speed": "m/s",
        }
        super().__init__(
            provider_name=provider_name,
            model_type="nwp" if "hres" in default_model_name.lower() else "ai",
            default_model_name=default_model_name,
            source_units=source_units,
            config=config,
        )

    def fetch_forecast(
        self,
        run_times: Optional[List[datetime]] = None,
        lead_times: Optional[List[int]] = None,
        variables: Optional[List[str]] = None,
        region: Optional[str] = None,
        file_path: Optional[Union[str, Path]] = None,
    ) -> pd.DataFrame:
        if file_path:
            return self.load_from_file(file_path)

        raise NotImplementedError(
            f"Direct live streaming without pre-ingestion for '{self.provider_name}' is not recommended. "
            "Please use ML/scripts/ingest_weatherbench2.py to materialize local Parquet records."
        )


class WeatherBench2ObservationProvider(BaseObservationProvider):
    """
    Adapter for WeatherBench 2 Ground-Truth Reference datasets (ERA5 Reanalysis).
    Raw units: Temperature in Kelvin, Precipitation in meters, Wind Speed in m/s.
    """

    def __init__(
        self,
        provider_name: str = "weatherbench2_era5",
        config: Optional[AppConfig] = None,
    ):
        source_units = {
            "temperature": "kelvin",
            "rainfall": "m",
            "wind_speed": "m/s",
        }
        super().__init__(provider_name=provider_name, source_units=source_units, config=config)

    def fetch_observations(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        variables: Optional[List[str]] = None,
        region: Optional[str] = None,
        file_path: Optional[Union[str, Path]] = None,
    ) -> pd.DataFrame:
        if file_path:
            return self.load_from_file(file_path)

        raise NotImplementedError(
            f"Direct live streaming without pre-ingestion for '{self.provider_name}' is not recommended. "
            "Please use ML/scripts/ingest_weatherbench2.py to materialize local Parquet records."
        )
