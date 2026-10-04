"""
Data provider adapter package for provider-neutral forecast and observation ingestion.
"""

from ingestion.providers.base import (
    ForecastProvider,
    ObservationProvider,
    BaseForecastProvider,
    BaseObservationProvider,
)
from ingestion.providers.nwp import NWPForecastProvider
from ingestion.providers.observations import (
    SurfaceStationObservationProvider,
    ReanalysisObservationProvider,
)
from ingestion.providers.satellite import SatellitePrecipitationProvider
from ingestion.providers.radar import RadarPrecipitationProvider

# Compatibility provider implementations for legacy imports
class GFSProvider(NWPForecastProvider):
    def __init__(self, provider_name: str = "gfs", default_model_name: str = "nwp_model_a", config=None):
        super().__init__(
            provider_name=provider_name,
            model_name=default_model_name,
            source_units={"temperature": "kelvin", "rainfall": "m", "wind_speed": "m/s"},
            config=config,
        )

class ECMWFProvider(NWPForecastProvider):
    def __init__(self, provider_name: str = "ecmwf_ifs", default_model_name: str = "nwp_model_b", config=None):
        super().__init__(
            provider_name=provider_name,
            model_name=default_model_name,
            source_units={"temperature": "kelvin", "rainfall": "m", "wind_speed": "m/s"},
            config=config,
        )

class WRFProvider(NWPForecastProvider):
    def __init__(self, provider_name: str = "wrf_mesoscale", default_model_name: str = "nwp_model_a", config=None):
        super().__init__(
            provider_name=provider_name,
            model_name=default_model_name,
            source_units={"temperature": "celsius", "rainfall": "mm", "wind_speed": "m/s"},
            config=config,
        )

class AIProvider(ForecastProvider):
    def __init__(self, provider_name: str = "ai_model_c", model_type: str = "ai", default_model_name: str = "ai_model_c", architecture: str = "graph_neural_network", config=None):
        super().__init__(
            provider_name=provider_name,
            model_type=model_type,
            default_model_name=default_model_name,
            source_units={"temperature": "celsius", "rainfall": "mm", "wind_speed": "m/s"},
            config=config,
        )
        self.architecture = architecture
    def fetch_forecast(self, run_times=None, lead_times=None, variables=None, region=None, file_path=None):
        if file_path:
            return self.load_from_file(file_path)
        raise ValueError("AIProvider requires a valid file_path to load forecasts.")
    def list_available_runs(self):
        return []

class GenericForecastProvider(ForecastProvider):
    def __init__(self, provider_name: str = "generic_forecast", source_units=None, config=None):
        super().__init__(provider_name=provider_name, source_units=source_units, config=config)
    def fetch_forecast(self, run_times=None, lead_times=None, variables=None, region=None, file_path=None):
        if file_path:
            return self.load_from_file(file_path)
        raise ValueError("GenericForecastProvider requires a file_path.")
    def list_available_runs(self):
        return []

class ERA5Provider(ReanalysisObservationProvider):
    def __init__(self, provider_name: str = "era5_reanalysis", config=None):
        super().__init__(
            provider_name=provider_name,
            source_units={"temperature": "kelvin", "rainfall": "m", "wind_speed": "m/s"},
            config=config,
        )

class StationProvider(SurfaceStationObservationProvider):
    def __init__(self, provider_name: str = "surface_stations", config=None):
        super().__init__(
            provider_name=provider_name,
            source_units={"temperature": "celsius", "rainfall": "mm", "wind_speed": "m/s"},
            config=config,
        )

class GenericObservationProvider(ObservationProvider):
    def __init__(self, provider_name: str = "generic_observation", source_units=None, config=None):
        super().__init__(provider_name=provider_name, source_units=source_units, config=config)
    def fetch_observations(self, start_time=None, end_time=None, variables=None, region=None, file_path=None):
        if file_path:
            return self.load_from_file(file_path)
        raise ValueError("GenericObservationProvider requires a file_path.")


class WeatherBench2ForecastProvider(ForecastProvider):
    def __init__(self, provider_name: str = "weatherbench2_forecast", default_model_name: str = "ecmwf_hres", config=None):
        super().__init__(
            provider_name=provider_name,
            model_type="nwp" if "hres" in default_model_name.lower() else "ai",
            default_model_name=default_model_name,
            source_units={"temperature": "kelvin", "rainfall": "m", "wind_speed": "m/s"},
            config=config,
        )
    def fetch_forecast(self, run_times=None, lead_times=None, variables=None, region=None, file_path=None):
        if file_path:
            return self.load_from_file(file_path)
        raise ValueError("WeatherBench2ForecastProvider requires a file_path or pre-ingested parquet.")
    def list_available_runs(self):
        return []


class WeatherBench2ObservationProvider(ObservationProvider):
    def __init__(self, provider_name: str = "weatherbench2_era5", config=None):
        super().__init__(
            provider_name=provider_name,
            source_units={"temperature": "kelvin", "rainfall": "m", "wind_speed": "m/s"},
            config=config,
        )
    def fetch_observations(self, start_time=None, end_time=None, variables=None, region=None, file_path=None):
        if file_path:
            return self.load_from_file(file_path)
        raise ValueError("WeatherBench2ObservationProvider requires a file_path or pre-ingested parquet.")


__all__ = [
    "ForecastProvider",
    "ObservationProvider",
    "BaseForecastProvider",
    "BaseObservationProvider",
    "NWPForecastProvider",
    "SurfaceStationObservationProvider",
    "ReanalysisObservationProvider",
    "SatellitePrecipitationProvider",
    "RadarPrecipitationProvider",
    "GFSProvider",
    "ECMWFProvider",
    "WRFProvider",
    "AIProvider",
    "GenericForecastProvider",
    "ERA5Provider",
    "StationProvider",
    "GenericObservationProvider",
    "WeatherBench2ForecastProvider",
    "WeatherBench2ObservationProvider",
]
