"""
Numerical Weather Prediction (NWP) forecast provider implementation supporting NetCDF, GRIB2, CSV, Parquet.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np

from configs import AppConfig, get_config
from data.provenance import get_provenance_tracker
from ingestion.providers.base import ForecastProvider
from ingestion.readers import load_weather_file
from ingestion.grid_readers import load_grid_weather_file


class NWPForecastProvider(ForecastProvider):
    """
    Provider adapter for physical Numerical Weather Prediction models (ECMWF, GFS, NCMRWF, etc.).
    """

    def __init__(
        self,
        provider_name: str = "ECMWF_IFS",
        model_name: Optional[str] = None,
        default_file_path: Optional[Union[str, Path]] = None,
        source_units: Optional[Dict[str, str]] = None,
        spatial_resolution: str = "0.25 deg",
        temporal_resolution: str = "6h",
        license_info: str = "Open Meteorological Data License",
        config: Optional[AppConfig] = None,
    ):
        super().__init__(
            provider_name=provider_name,
            model_type="nwp",
            default_model_name=model_name or provider_name,
            source_units=source_units or {"temperature": "celsius", "rainfall": "mm", "wind_speed": "m/s"},
            config=config,
        )
        self.default_file_path = Path(default_file_path) if default_file_path else None
        self.spatial_resolution = spatial_resolution
        self.temporal_resolution = temporal_resolution
        self.license_info = license_info

    def list_available_runs(self, file_path: Optional[Union[str, Path]] = None) -> List[datetime]:
        """List distinct model initialization run timestamps."""
        target_path = Path(file_path) if file_path else self.default_file_path
        if not target_path or not target_path.exists():
            return []
        try:
            df = self._read_raw_file(target_path)
            if "run_time" in df.columns:
                return sorted(pd.to_datetime(df["run_time"]).drop_duplicates().tolist())
            elif "timestamp" in df.columns:
                return sorted(pd.to_datetime(df["timestamp"]).drop_duplicates().tolist())
        except Exception:
            return []
        return []

    def fetch_forecast(
        self,
        run_times: Optional[List[datetime]] = None,
        lead_times: Optional[List[int]] = None,
        variables: Optional[List[str]] = None,
        region: Optional[str] = None,
        file_path: Optional[Union[str, Path]] = None,
    ) -> pd.DataFrame:
        """Fetch and standardize NWP forecast records."""
        target_path = Path(file_path) if file_path else self.default_file_path
        if not target_path or not target_path.exists():
            raise FileNotFoundError(f"NWP forecast file not found: {target_path}")

        raw_df = self._read_raw_file(target_path)
        std_df = self.process_raw_dataframe(raw_df, model_name=self.default_model_name)

        # Filters
        if run_times is not None and "run_time" in std_df.columns:
            rt_series = pd.to_datetime(std_df["run_time"])
            target_rts = [pd.Timestamp(rt) for rt in run_times]
            std_df = std_df[rt_series.isin(target_rts)]

        if lead_times is not None and "lead_time" in std_df.columns:
            std_df = std_df[std_df["lead_time"].isin(lead_times)]

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

        # Register provenance
        tracker = get_provenance_tracker()
        for var in std_df["variable"].unique():
            tracker.register_dataset(
                source=self.provider_name,
                dataset_name=f"{self.provider_name}_{var}_forecast",
                provider=self.provider_name,
                variable=str(var),
                units=self.config.units.temperature if var == "temperature" else (
                    self.config.units.rainfall if var == "rainfall" else self.config.units.wind_speed
                ),
                spatial_resolution=self.spatial_resolution,
                temporal_resolution=self.temporal_resolution,
                license_info=self.license_info,
            )

        return std_df.reset_index(drop=True)

    def _read_raw_file(self, file_path: Path) -> pd.DataFrame:
        suffix = file_path.suffix.lower()
        if suffix in [".nc", ".netcdf", ".grb", ".grib", ".grib2"]:
            ds = load_grid_weather_file(file_path)
            return ds.to_dataframe().reset_index()
        else:
            return load_weather_file(file_path)
