"""
Satellite Observation Provider adapter for gridded precipitation products (e.g., GPM IMERG, INSAT-3D, CMORPH).
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


class SatellitePrecipitationProvider(ObservationProvider):
    """
    Provider adapter for gridded satellite precipitation products (e.g. NASA GPM IMERG, INSAT-3D).
    """

    def __init__(
        self,
        provider_name: str = "NASA_GPM_IMERG",
        default_file_path: Optional[Union[str, Path]] = None,
        source_units: Optional[Dict[str, str]] = None,
        spatial_resolution: str = "0.10 deg",
        license_info: str = "NASA Open Data Policy (GPM Mission)",
        config: Optional[AppConfig] = None,
    ):
        super().__init__(
            provider_name=provider_name,
            observation_type="satellite",
            source_units=source_units or {"rainfall": "mm"},
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
        """Fetch and standardize gridded satellite precipitation observations."""
        target_path = Path(file_path) if file_path else self.default_file_path
        if not target_path or not target_path.exists():
            raise FileNotFoundError(f"Satellite precipitation dataset file not found: {target_path}")

        suffix = target_path.suffix.lower()
        if suffix in [".nc", ".netcdf", ".grb", ".grib", ".grib2", ".hdf5", ".h5"]:
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
                dataset_name=f"{self.provider_name}_{var}_satellite",
                provider=self.provider_name,
                variable=str(var),
                units=self.config.units.rainfall,
                spatial_resolution=self.spatial_resolution,
                temporal_resolution="30-min/3-Hourly",
                license_info=self.license_info,
            )

        return std_df.reset_index(drop=True)
