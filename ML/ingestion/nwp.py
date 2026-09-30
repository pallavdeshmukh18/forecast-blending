"""
Numerical Weather Prediction (NWP) ingestion interfaces and base providers.

Defines the abstract contract for all model forecast sources and implements
NWP-specific base classes and provider adapters (e.g. GFS, ECMWF IFS, WRF, GEFS).
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

from ingestion.schema import validate_forecast_dataframe
from ingestion.providers import (
    BaseForecastProvider,
    GFSProvider,
    ECMWFProvider,
    WRFProvider,
    GenericForecastProvider,
)


class ForecastSource(BaseForecastProvider):
    """
    Abstract Base Class for all forecast model data sources (NWP, AI, Statistical).
    Maintains compatibility with Phase 1 fetch/validate methods.
    """

    def __init__(self, source_name: str, model_type: str = "nwp", config: Optional[Dict] = None):
        super().__init__(provider_name=source_name, model_type=model_type)
        self.source_name = source_name
        self.config_dict = config or {}

    def fetch(
        self,
        run_times: List[datetime],
        lead_times: List[int],
        variables: Optional[List[str]] = None,
        region: Optional[str] = None,
    ) -> pd.DataFrame:
        return self.fetch_forecast(run_times=run_times, lead_times=lead_times, variables=variables, region=region)

    def fetch_forecast(
        self,
        run_times: Optional[List[datetime]] = None,
        lead_times: Optional[List[int]] = None,
        variables: Optional[List[str]] = None,
        region: Optional[str] = None,
        file_path: Optional[str] = None,
    ) -> pd.DataFrame:
        raise NotImplementedError(
            f"Live external retrieval for '{self.source_name}' is not active in this baseline phase. "
            "Use file_path to load records or use the synthetic generator."
        )

    def validate(self, df: pd.DataFrame, allow_nans: bool = False) -> pd.DataFrame:
        """
        Validate that the fetched forecast DataFrame conforms to the internal standard schema.
        """
        return validate_forecast_dataframe(df, allow_nans=allow_nans)


class NWPForecastSource(ForecastSource):
    """
    Base class and placeholder for deterministic Numerical Weather Prediction models
    (e.g., NOAA GFS, ECMWF HRES/IFS, NCMRWF Unified Model, WRF).
    """

    def __init__(
        self,
        source_name: str,
        resolution_km: float = 12.0,
        config: Optional[Dict] = None,
    ):
        super().__init__(source_name=source_name, model_type="nwp", config=config)
        self.resolution_km = resolution_km

    def fetch_forecast(
        self,
        run_times: Optional[List[datetime]] = None,
        lead_times: Optional[List[int]] = None,
        variables: Optional[List[str]] = None,
        region: Optional[str] = None,
        file_path: Optional[str] = None,
    ) -> pd.DataFrame:
        if file_path:
            return self.load_from_file(file_path)
        raise NotImplementedError(
            f"Live external NWP retrieval for '{self.source_name}' is not active in this baseline phase. "
            "Use the synthetic generator or provide pre-processed forecast archives."
        )


class EnsembleForecastSource(NWPForecastSource):
    """
    Base class and placeholder for multi-member Ensemble Prediction Systems
    (e.g., GEFS 31-member, ECMWF EPS 51-member).
    """

    def __init__(
        self,
        source_name: str,
        num_members: int = 30,
        resolution_km: float = 25.0,
        config: Optional[Dict] = None,
    ):
        super().__init__(source_name=source_name, resolution_km=resolution_km, config=config)
        self.num_members = num_members
        self.model_type = "ensemble"

    def fetch_forecast(
        self,
        run_times: Optional[List[datetime]] = None,
        lead_times: Optional[List[int]] = None,
        variables: Optional[List[str]] = None,
        region: Optional[str] = None,
        file_path: Optional[str] = None,
    ) -> pd.DataFrame:
        if file_path:
            return self.load_from_file(file_path)
        raise NotImplementedError(
            f"Ensemble extraction for '{self.source_name}' ({self.num_members} members) is slated for future phases."
        )


__all__ = [
    "BaseForecastProvider",
    "ForecastSource",
    "NWPForecastSource",
    "EnsembleForecastSource",
    "GFSProvider",
    "ECMWFProvider",
    "WRFProvider",
    "GenericForecastProvider",
]
