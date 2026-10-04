"""
Ingestion module for observations, NWP forecasts, AI weather predictions, format readers,
unit conversions, and variable standardization.
"""

from ingestion.schema import (
    ForecastRecord,
    ObservationRecord,
    REQUIRED_FORECAST_COLUMNS,
    REQUIRED_OBSERVATION_COLUMNS,
    OPTIONAL_FORECAST_COLUMNS,
    OPTIONAL_OBSERVATION_COLUMNS,
    validate_forecast_dataframe,
    validate_observation_dataframe,
)
from ingestion.variable_mapping import (
    standardize_variable_name,
    standardize_dataframe_variables,
)
from ingestion.units import (
    convert_units,
    convert_temperature,
    convert_rainfall,
    convert_wind_speed,
    normalize_dataframe_units,
)
from ingestion.quality_control import (
    apply_quality_control,
    format_quality_report_text,
    DataQualityReport,
)
from ingestion.readers import (
    read_weather_csv,
    read_weather_parquet,
    read_weather_netcdf,
    read_weather_grib2,
    load_weather_file,
)
from ingestion.providers import (
    BaseForecastProvider,
    BaseObservationProvider,
    GFSProvider,
    ECMWFProvider,
    WRFProvider,
    AIProvider,
    GenericForecastProvider,
    ERA5Provider,
    StationProvider,
    GenericObservationProvider,
    WeatherBench2ForecastProvider,
    WeatherBench2ObservationProvider,
)
from ingestion.observations import (
    ObservationSource,
    StationObservationSource,
    GriddedReanalysisObservationSource,
)
from ingestion.nwp import (
    ForecastSource,
    NWPForecastSource,
    EnsembleForecastSource,
)
from ingestion.ai_forecasts import AIWeatherForecastSource
from ingestion.synthetic import (
    SyntheticDataGenerator,
    SyntheticObservationSource,
    SyntheticForecastSource,
)
from ingestion.grid_readers import (
    load_forecast_grid,
    load_observation_grid,
    standardize_grid_variables,
)

__all__ = [
    "ForecastRecord",
    "ObservationRecord",
    "REQUIRED_FORECAST_COLUMNS",
    "REQUIRED_OBSERVATION_COLUMNS",
    "OPTIONAL_FORECAST_COLUMNS",
    "OPTIONAL_OBSERVATION_COLUMNS",
    "validate_forecast_dataframe",
    "validate_observation_dataframe",
    "standardize_variable_name",
    "standardize_dataframe_variables",
    "convert_units",
    "convert_temperature",
    "convert_rainfall",
    "convert_wind_speed",
    "normalize_dataframe_units",
    "apply_quality_control",
    "format_quality_report_text",
    "DataQualityReport",
    "read_weather_csv",
    "read_weather_parquet",
    "read_weather_netcdf",
    "read_weather_grib2",
    "load_weather_file",
    "load_forecast_grid",
    "load_observation_grid",
    "standardize_grid_variables",
    "BaseForecastProvider",
    "BaseObservationProvider",
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
    "ObservationSource",
    "StationObservationSource",
    "GriddedReanalysisObservationSource",
    "ForecastSource",
    "NWPForecastSource",
    "EnsembleForecastSource",
    "AIWeatherForecastSource",
    "SyntheticDataGenerator",
    "SyntheticObservationSource",
    "SyntheticForecastSource",
]
