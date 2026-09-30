"""
Abstract Base Classes for provider-neutral forecast and observation adapters.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd

from configs import AppConfig, get_config
from data.provenance import DataProvenance, get_provenance_tracker
from ingestion.schema import (
    validate_forecast_dataframe,
    validate_observation_dataframe,
)
from ingestion.variable_mapping import standardize_dataframe_variables
from ingestion.units import normalize_dataframe_units
from ingestion.quality_control import apply_quality_control, DataQualityReport


class ForecastProvider(ABC):
    """
    Abstract Base Class for external forecast data providers.
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

    @abstractmethod
    def list_available_runs(self) -> List[datetime]:
        """
        List all available forecast run initialization timestamps.
        """
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """
        Retrieve provider and model capability metadata.
        """
        return {
            "provider_name": self.provider_name,
            "model_type": self.model_type,
            "default_model_name": self.default_model_name,
            "source_units": self.source_units,
        }

    def process_raw_dataframe(
        self,
        df: pd.DataFrame,
        model_name: Optional[str] = None,
        variable_col: str = "variable",
        value_col: str = "forecast_value",
        source_units: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """
        Standardize variables, normalize units, verify initialization/valid time relationship,
        apply quality control, and validate schema.
        """
        raw = df.copy()

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

        active_model = model_name or self.default_model_name
        if "model_name" not in raw.columns:
            raw["model_name"] = active_model

        # Ensure valid_time = initialization_time (run_time) + lead_time if lead_time or run_time are present
        if "run_time" in raw.columns and "lead_time" in raw.columns and "timestamp" not in raw.columns:
            raw["run_time"] = pd.to_datetime(raw["run_time"])
            raw["timestamp"] = raw["run_time"] + pd.to_timedelta(raw["lead_time"], unit="h")
        elif "timestamp" in raw.columns and "run_time" in raw.columns and "lead_time" not in raw.columns:
            raw["run_time"] = pd.to_datetime(raw["run_time"])
            raw["timestamp"] = pd.to_datetime(raw["timestamp"])
            raw["lead_time"] = ((raw["timestamp"] - raw["run_time"]).dt.total_seconds() / 3600).round().astype(int)

        if "lead_time" not in raw.columns:
            raw["lead_time"] = 0

        std_df = standardize_dataframe_variables(raw, variable_col=variable_col, config=self.config)
        units_to_use = source_units or self.source_units
        norm_df = normalize_dataframe_units(
            std_df,
            value_col=value_col,
            variable_col=variable_col,
            source_units=units_to_use,
            config=self.config,
        )

        valid_df, rejected_df, report = apply_quality_control(
            norm_df, value_col=value_col, variable_col=variable_col, config=self.config
        )
        self.last_quality_report = report

        return validate_forecast_dataframe(valid_df)

    def load_from_file(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """
        Load forecasts from a local file (CSV, Parquet, NetCDF, GRIB2) and standardize.
        """
        from ingestion.readers import load_weather_file
        df = load_weather_file(file_path, **kwargs)
        return self.process_raw_dataframe(df)


# Legacy backward-compatibility alias
BaseForecastProvider = ForecastProvider


class ObservationProvider(ABC):
    """
    Abstract Base Class for ground truth observation providers.
    """

    def __init__(
        self,
        provider_name: str,
        observation_type: str = "surface_stations",
        source_units: Optional[Dict[str, str]] = None,
        config: Optional[AppConfig] = None,
    ):
        self.provider_name = provider_name
        self.observation_type = observation_type  # 'surface_stations', 'reanalysis', 'satellite', 'radar'
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
        Fetch observation records and return DataFrame conforming to standard observation schema.
        """
        pass

    def list_available_stations(self) -> List[str]:
        """
        List available station or site identifiers if applicable.
        """
        return []

    def get_metadata(self) -> Dict[str, Any]:
        """
        Retrieve provider metadata.
        """
        return {
            "provider_name": self.provider_name,
            "observation_type": self.observation_type,
            "source_units": self.source_units,
        }

    def load_from_file(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """
        Load observations from a local file and standardize.
        """
        from ingestion.readers import load_weather_file
        df = load_weather_file(file_path, **kwargs)
        return self.process_raw_dataframe(df)

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

        rename_map = {
            "time": "timestamp",
            "obs_time": "timestamp",
            "datetime": "timestamp",
            "lat": "latitude",
            "lon": "longitude",
            "val": "observed_value",
            "value": "observed_value",
            "obs": "observed_value",
            "var": "variable",
            "param": "variable",
            "station": "station_id",
            "site": "station_id",
            "id": "station_id",
        }
        raw = raw.rename(columns={k: v for k, v in rename_map.items() if k in raw.columns and v not in raw.columns})

        if "station_id" not in raw.columns:
            if "latitude" in raw.columns and "longitude" in raw.columns:
                raw["station_id"] = "SITE_" + raw["latitude"].round(2).astype(str) + "_" + raw["longitude"].round(2).astype(str)
            else:
                raw["station_id"] = "OBS_STATION_001"

        std_df = standardize_dataframe_variables(raw, variable_col=variable_col, config=self.config)
        units_to_use = source_units or self.source_units
        norm_df = normalize_dataframe_units(
            std_df,
            value_col=value_col,
            variable_col=variable_col,
            source_units=units_to_use,
            config=self.config,
        )

        valid_df, rejected_df, report = apply_quality_control(
            norm_df, value_col=value_col, variable_col=variable_col, config=self.config
        )
        self.last_quality_report = report

        return validate_observation_dataframe(valid_df)


# Legacy backward-compatibility alias
BaseObservationProvider = ObservationProvider
