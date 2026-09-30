"""
Observation ingestion interfaces and providers.

Defines the abstract contract for ground-truth weather observations and
implements base and provider adapters (e.g. ERA5, StationProvider, GenericObservationProvider).
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

from ingestion.schema import validate_observation_dataframe
from ingestion.providers import (
    BaseObservationProvider,
    ERA5Provider,
    StationProvider,
    GenericObservationProvider,
)


class ObservationSource(BaseObservationProvider):
    """
    Abstract Base Class for observational data sources.
    Maintains compatibility with Phase 1 fetch/validate methods.
    """

    def __init__(self, source_name: str, config: Optional[Dict] = None):
        super().__init__(provider_name=source_name)
        self.source_name = source_name
        self.config_dict = config or {}

    def fetch(
        self,
        start_time: datetime,
        end_time: datetime,
        variables: Optional[List[str]] = None,
        region: Optional[str] = None,
    ) -> pd.DataFrame:
        return self.fetch_observations(start_time=start_time, end_time=end_time, variables=variables, region=region)

    def fetch_observations(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        variables: Optional[List[str]] = None,
        region: Optional[str] = None,
        file_path: Optional[str] = None,
    ) -> pd.DataFrame:
        raise NotImplementedError(
            f"Live retrieval from '{self.source_name}' is not implemented in this phase. "
            "Use file_path to load observations or use the synthetic generator."
        )

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate that the fetched data conforms to the standard schema.
        """
        return validate_observation_dataframe(df)


class StationObservationSource(ObservationSource):
    """
    Observational source adapter for surface weather station networks
    (e.g., WMO GTS, NOAA ISD, IMD AWS network).
    """

    def __init__(self, source_name: str = "surface_station_network", config: Optional[Dict] = None):
        super().__init__(source_name=source_name, config=config)

    def fetch_observations(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        variables: Optional[List[str]] = None,
        region: Optional[str] = None,
        file_path: Optional[str] = None,
    ) -> pd.DataFrame:
        if file_path:
            return self.load_from_file(file_path)
        raise NotImplementedError(
            f"Live retrieval from {self.source_name} is not implemented in this MVP phase. "
            "Use file_path to load records or use SyntheticObservationSource."
        )


class GriddedReanalysisObservationSource(ObservationSource):
    """
    Observational source adapter for gridded reanalysis / satellite products
    (e.g., ERA5, IMD Gridded Rainfall, GPM IMERG).
    """

    def __init__(self, source_name: str = "gridded_reanalysis", config: Optional[Dict] = None):
        super().__init__(source_name=source_name, config=config)

    def fetch_observations(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        variables: Optional[List[str]] = None,
        region: Optional[str] = None,
        file_path: Optional[str] = None,
    ) -> pd.DataFrame:
        if file_path:
            return self.load_from_file(file_path)
        raise NotImplementedError(
            f"Gridded reanalysis live retrieval for {self.source_name} is planned for future releases. "
            "Use file_path to load records or use SyntheticObservationSource."
        )


__all__ = [
    "BaseObservationProvider",
    "ObservationSource",
    "StationObservationSource",
    "GriddedReanalysisObservationSource",
    "ERA5Provider",
    "StationProvider",
    "GenericObservationProvider",
]
