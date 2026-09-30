"""
Observation Provider adapters for surface weather stations and reanalysis ground truth datasets.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd

from configs import AppConfig, get_config
from data.provenance import get_provenance_tracker
from ingestion.providers.base import ObservationProvider
from ingestion.readers import load_weather_file
from ingestion.grid_readers import load_grid_weather_file


class SurfaceStationObservationProvider(ObservationProvider):
    """
    Observation provider adapter for ground-based automated weather stations (AWS).
    """

    def __init__(
        self,
        provider_name: str = "IMD_AWS_Network",
        default_file_path: Optional[Union[str, Path]] = None,
        source_units: Optional[Dict[str, str]] = None,
        license_info: str = "National Weather Service Open Station Data",
        config: Optional[AppConfig] = None,
    ):
        super().__init__(
            provider_name=provider_name,
            observation_type="surface_stations",
            source_units=source_units or {"temperature": "celsius", "rainfall": "mm", "wind_speed": "m/s"},
            config=config,
        )
        self.default_file_path = Path(default_file_path) if default_file_path else None
        self.license_info = license_info

    def list_available_stations(self, file_path: Optional[Union[str, Path]] = None) -> List[str]:
        """List distinct station identifiers in dataset."""
        target_path = Path(file_path) if file_path else self.default_file_path
        if not target_path or not target_path.exists():
            return []
        try:
            df = load_weather_file(target_path)
            for col in ["station_id", "station", "id", "site"]:
                if col in df.columns:
                    return sorted(df[col].dropna().unique().astype(str).tolist())
        except Exception:
            return []
        return []

    def fetch_observations(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        variables: Optional[List[str]] = None,
        region: Optional[str] = None,
        file_path: Optional[Union[str, Path]] = None,
    ) -> pd.DataFrame:
        """Fetch and standardize surface observation records."""
        target_path = Path(file_path) if file_path else self.default_file_path
        if not target_path or not target_path.exists():
            raise FileNotFoundError(f"Observation file not found: {target_path}")

        raw_df = load_weather_file(target_path)
        std_df = self.process_raw_dataframe(raw_df)

        if start_time is not None and "timestamp" in std_df.columns:
            std_df = std_df[pd.to_datetime(std_df["timestamp"]) >= pd.Timestamp(start_time)]

        if end_time is not None and "timestamp" in std_df.columns:
            std_df = std_df[pd.to_datetime(std_df["timestamp"]) <= pd.Timestamp(end_time)]

        if variables is not None:
            std_df = std_df[std_df["variable"].isin(variables)]

        if region is not None and region in self.config.regions:
            reg_cfg = self.config.regions[region]
            std_df = std_df[
                (std_df["latitude"] >= reg_cfg.lat_min)
                & (std_df["latitude"] <= reg_cfg.lat_max)
                & (std_df["longitude"] >= reg_cfg.lon_min)
                & (std_df["longitude"] <= reg_cfg.lon_max)
            ]

        tracker = get_provenance_tracker()
        for var in std_df["variable"].unique():
            tracker.register_dataset(
                source=self.provider_name,
                dataset_name=f"{self.provider_name}_{var}_obs",
                provider=self.provider_name,
                variable=str(var),
                units=self.config.units.temperature if var == "temperature" else (
                    self.config.units.rainfall if var == "rainfall" else self.config.units.wind_speed
                ),
                spatial_resolution="Point Stations",
                temporal_resolution="Hourly/Synoptic",
                license_info=self.license_info,
            )

        return std_df.reset_index(drop=True)


class ReanalysisObservationProvider(ObservationProvider):
    """
    Observation provider adapter for atmospheric reanalysis products (ERA5, IMDAA, MERRA2).
    """

    def __init__(
        self,
        provider_name: str = "ECMWF_ERA5_Reanalysis",
        default_file_path: Optional[Union[str, Path]] = None,
        source_units: Optional[Dict[str, str]] = None,
        spatial_resolution: str = "0.25 deg",
        license_info: str = "Copernicus Climate Change Service (C3S) License",
        config: Optional[AppConfig] = None,
    ):
        super().__init__(
            provider_name=provider_name,
            observation_type="reanalysis",
            source_units=source_units or {"temperature": "celsius", "rainfall": "mm", "wind_speed": "m/s"},
            config=config,
        )
        self.default_file_path = Path(default_file_path) if default_file_path else None
        self.spatial_resolution = spatial_resolution
        self.license_info = license_info

    def fetch_observations(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        variables: Optional[List[str]] = None,
        region: Optional[str] = None,
        file_path: Optional[Union[str, Path]] = None,
    ) -> pd.DataFrame:
        """Fetch and standardize reanalysis grid observation records."""
        target_path = Path(file_path) if file_path else self.default_file_path
        if not target_path or not target_path.exists():
            raise FileNotFoundError(f"Reanalysis dataset file not found: {target_path}")

        suffix = target_path.suffix.lower()
        if suffix in [".nc", ".netcdf", ".grb", ".grib", ".grib2"]:
            ds = load_grid_weather_file(target_path)
            raw_df = ds.to_dataframe().reset_index()
        else:
            raw_df = load_weather_file(target_path)

        std_df = self.process_raw_dataframe(raw_df)

        if start_time is not None and "timestamp" in std_df.columns:
            std_df = std_df[pd.to_datetime(std_df["timestamp"]) >= pd.Timestamp(start_time)]

        if end_time is not None and "timestamp" in std_df.columns:
            std_df = std_df[pd.to_datetime(std_df["timestamp"]) <= pd.Timestamp(end_time)]

        if variables is not None:
            std_df = std_df[std_df["variable"].isin(variables)]

        if region is not None and region in self.config.regions:
            reg_cfg = self.config.regions[region]
            std_df = std_df[
                (std_df["latitude"] >= reg_cfg.lat_min)
                & (std_df["latitude"] <= reg_cfg.lat_max)
                & (std_df["longitude"] >= reg_cfg.lon_min)
                & (std_df["longitude"] <= reg_cfg.lon_max)
            ]

        tracker = get_provenance_tracker()
        for var in std_df["variable"].unique():
            tracker.register_dataset(
                source=self.provider_name,
                dataset_name=f"{self.provider_name}_{var}_reanalysis",
                provider=self.provider_name,
                variable=str(var),
                units=self.config.units.temperature if var == "temperature" else (
                    self.config.units.rainfall if var == "rainfall" else self.config.units.wind_speed
                ),
                spatial_resolution=self.spatial_resolution,
                temporal_resolution="Hourly/3-Hourly",
                license_info=self.license_info,
            )

        return std_df.reset_index(drop=True)
