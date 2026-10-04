"""
Configuration loader and schema definition for the Hybrid Weather Blending System.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
import yaml
from pydantic import BaseModel, Field


class VariableConfig(BaseModel):
    standard_name: str
    unit: str
    min_physical_bound: float
    max_physical_bound: float


class RegionConfig(BaseModel):
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    description: str


class ModelMetadataConfig(BaseModel):
    type: str  # 'nwp', 'ai', 'ensemble', 'statistical'
    resolution_km: float
    update_frequency_hours: int
    description: str


class EvaluationConfig(BaseModel):
    metrics: List[str]
    grouping_keys: List[str]
    default_baseline: str = "equal_weight_blend"


class RainfallThresholds(BaseModel):
    moderate: float = 15.6
    heavy: float = 35.5
    very_heavy: float = 64.5
    extreme: float = 124.5


class HeatwaveThresholds(BaseModel):
    mild: float = 40.0
    severe: float = 45.0


class WindThresholds(BaseModel):
    strong_breeze: float = 10.8
    gale: float = 17.2
    storm: float = 24.5


class ExtremeThresholdsConfig(BaseModel):
    rainfall: RainfallThresholds = Field(default_factory=RainfallThresholds)
    heatwave: HeatwaveThresholds = Field(default_factory=HeatwaveThresholds)
    wind: WindThresholds = Field(default_factory=WindThresholds)


class SpatialConfig(BaseModel):
    enabled: bool = False
    target_resolution: float = 0.25
    interpolation: str = "bilinear"  # 'nearest' or 'bilinear'
    max_missing_fraction: float = 0.20
    lat_min: float = 10.0
    lat_max: float = 36.0
    lon_min: float = 70.0
    lon_max: float = 86.0


class SpatialCNNConfig(BaseModel):
    enabled: bool = False
    channels: List[int] = Field(default_factory=lambda: [32, 32, 64])
    kernel_size: int = 3
    dropout: float = 0.10
    learning_rate: float = 0.001
    epochs: int = 50
    batch_size: int = 16
    early_stopping_patience: int = 8
    random_seed: int = 42


class ProbabilisticConfig(BaseModel):
    enabled: bool = True
    quantiles: List[float] = Field(default_factory=lambda: [0.10, 0.25, 0.50, 0.75, 0.90])
    distribution_type: str = "gaussian"


class EMOSConfig(BaseModel):
    enabled: bool = False
    fit_method: str = "crps"  # 'crps', 'nll', or 'least_squares'
    min_spread_epsilon: float = 1e-4


class CalibrationConfig(BaseModel):
    enabled: bool = False
    bins: int = 10
    min_samples_per_bin: int = 5


class ExtremesEvaluationConfig(BaseModel):
    rainfall_thresholds: List[float] = Field(default_factory=lambda: [10.0, 25.0, 50.0, 100.0])
    wind_thresholds: List[float] = Field(default_factory=lambda: [10.0, 15.0, 20.0, 25.0])
    temperature_thresholds: List[float] = Field(default_factory=lambda: [0.0, 5.0, 35.0, 40.0])
    hot_temperature_thresholds: List[float] = Field(default_factory=lambda: [35.0, 38.0, 40.0])
    cold_temperature_thresholds: List[float] = Field(default_factory=lambda: [0.0, 5.0])


class SpatiotemporalConfig(BaseModel):
    enabled: bool = True
    sequence_length: int = 4
    hidden_dim: int = 32
    recurrent_type: str = "gru"  # 'gru' or 'lstm'
    dropout: float = 0.10
    learning_rate: float = 0.001
    epochs: int = 40
    batch_size: int = 8
    early_stopping_patience: int = 6
    random_seed: int = 42


class ExtremeCalibrationConfig(BaseModel):
    enabled: bool = True
    method: str = "zero_inflated_gamma"  # 'gamma_emos', 'zero_inflated_gamma', 'gev'
    censoring_threshold: float = 0.1
    min_positive_samples: int = 20
    enable_gev_tail: bool = True
    gev_min_samples: int = 50


class PipelineConfig(BaseModel):
    mode: str = "synthetic"  # 'synthetic', 'historical', 'real'
    save_provenance: bool = True
    quality_control_strict: bool = True
    enable_temporal_memory: bool = True
    enable_extreme_calibration: bool = True


class PathsConfig(BaseModel):
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    forecasts_data_dir: str = "data/forecasts"
    synthetic_observations_file: str = "data/processed/synthetic_observations.parquet"
    synthetic_forecasts_file: str = "data/forecasts/synthetic_forecasts.parquet"
    synthetic_spatial_forecasts_file: str = "data/forecasts/synthetic_spatial_forecasts.nc"
    synthetic_spatial_observations_file: str = "data/processed/synthetic_spatial_observations.nc"
    verification_dataset_file: str = "data/processed/verification_dataset.parquet"
    error_db_file: str = "data/processed/historical_error_db.parquet"
    evaluation_results_dir: str = "data/processed/evaluation"
    reports_dir: str = "reports"
    provenance_file: str = "reports/data_provenance.json"
    quality_report_file: str = "reports/data_quality.json"


class SyntheticDataConfig(BaseModel):
    seed: int = 42
    start_date: str = "2026-01-01"
    end_date: str = "2026-01-30"
    time_step_hours: int = 6
    stations_per_region: int = 3
    spatial_grid_shape: List[int] = Field(default_factory=lambda: [16, 16])


class DataConfig(BaseModel):
    mode: str = "synthetic"  # 'synthetic' or 'real'
    real_forecasts_path: Optional[str] = "data/processed/real/weatherbench2/real_forecasts.parquet"
    real_observations_path: Optional[str] = "data/processed/real/weatherbench2/real_observations.parquet"
    verification_dataset_path: Optional[str] = "data/processed/real/weatherbench2/real_verification_dataset.parquet"


class WeatherBench2Config(BaseModel):
    enabled: bool = True
    era5_source: str = "gs://weatherbench2/datasets/era5/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr"
    hres_source: str = "gs://weatherbench2/datasets/hres/2016-2022-0012-240x121_equiangular_with_poles_conservative.zarr"
    graphcast_source: str = "gs://weatherbench2/datasets/graphcast_v2/2020-240x121_equiangular_with_poles_conservative.zarr"
    pangu_source: str = "gs://weatherbench2/datasets/pangu/2018-2022_0012_240x121_equiangular_with_poles_conservative.zarr"
    variables: List[str] = Field(default_factory=lambda: ["temperature", "wind_speed", "rainfall"])
    start_date: str = "2020-01-01"
    end_date: str = "2020-01-05"
    forecast_models: List[str] = Field(default_factory=lambda: ["ecmwf_hres", "graphcast"])
    lead_times_hours: List[int] = Field(default_factory=lambda: [6, 12, 24, 48])
    region: RegionConfig = Field(
        default_factory=lambda: RegionConfig(
            lat_min=10.0,
            lat_max=30.0,
            lon_min=70.0,
            lon_max=90.0,
            description="South Asia / Indian Subcontinent Domain",
        )
    )
    resolution: str = "1.5_deg_equiangular"
    raw_dir: str = "data/raw/weatherbench2"
    processed_dir: str = "data/processed/real/weatherbench2"
    provenance_dir: str = "data/provenance/weatherbench2"


class SpatialAlignmentConfig(BaseModel):
    method: str = "nearest"
    max_distance_km: float = 50.0


class TemporalAlignmentConfig(BaseModel):
    tolerance_minutes: int = 30


class UnitsConfig(BaseModel):
    temperature: str = "celsius"
    rainfall: str = "mm"
    wind_speed: str = "m/s"


class SkillEstimationConfig(BaseModel):
    method: str = "inverse_variance"  # "inverse_variance" or "inverse_mae"
    mode: str = "rolling"             # "rolling" or "expanding"
    window_size: int = 30             # Historical time steps / observations in window
    minimum_samples: int = 20         # Minimum sample threshold for scoped skill before fallback
    epsilon: float = 1e-6             # Stability parameter avoiding division by zero
    weight_smoothing: float = 0.10    # Alpha regularization factor toward equal weight (1/N)
    fallback_hierarchy: List[str] = Field(
        default_factory=lambda: [
            "model_variable_lead_region",
            "model_variable_lead",
            "model_variable",
            "model",
        ]
    )


class RegressionConfig(BaseModel):
    enabled: bool = True
    model: str = "ridge"
    alpha: float = 1.0
    fit_intercept: bool = True
    constrained_weights: bool = True


class WeatherRegimeThresholdsConfig(BaseModel):
    heavy_rain_mm: float = 25.0
    convective_rain_mm: float = 10.0
    high_wind_mps: float = 12.0
    hot_temperature_c: float = 35.0
    cold_temperature_c: float = 5.0
    priority_order: List[str] = Field(
        default_factory=lambda: [
            "heavy_rain",
            "convective",
            "high_wind",
            "hot_extreme",
            "cold_extreme",
            "stable",
        ]
    )


class WeatherRegimeClusteringConfig(BaseModel):
    n_clusters: int = 4
    random_state: int = 42
    features: List[str] = Field(
        default_factory=lambda: [
            "temperature",
            "rainfall",
            "wind_speed",
            "relative_humidity",
            "surface_pressure",
            "diurnal_sin",
            "diurnal_cos",
            "seasonal_sin",
            "seasonal_cos",
        ]
    )


class WeatherRegimeConfig(BaseModel):
    method: str = "rule_based"  # 'rule_based' or 'clustering'
    default_regime: str = "stable"
    smoothing_window: int = 1
    clustering: WeatherRegimeClusteringConfig = Field(default_factory=WeatherRegimeClusteringConfig)
    thresholds: WeatherRegimeThresholdsConfig = Field(default_factory=WeatherRegimeThresholdsConfig)


class NeuralGatingConfig(BaseModel):
    enabled: bool = True
    hidden_sizes: List[int] = Field(default_factory=lambda: [64, 32])
    dropout: float = 0.10
    learning_rate: float = 0.001
    batch_size: int = 64
    epochs: int = 100
    early_stopping_patience: int = 10
    weight_decay: float = 0.0001
    random_seed: int = 42
    use_residual_head: bool = False


class AppConfig(BaseModel):
    project_name: str = "Hybrid AI–NWP Multi-Model Forecast Blending System"
    version: str = "0.6.0"
    environment: str = "research"
    data: DataConfig = Field(default_factory=DataConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    variables: Dict[str, VariableConfig]
    units: UnitsConfig = Field(default_factory=UnitsConfig)
    variable_mappings: Dict[str, List[str]] = Field(default_factory=dict)
    spatial_alignment: SpatialAlignmentConfig = Field(default_factory=SpatialAlignmentConfig)
    temporal_alignment: TemporalAlignmentConfig = Field(default_factory=TemporalAlignmentConfig)
    lead_time_bins: Dict[str, List[int]] = Field(
        default_factory=lambda: {
            "short": [0, 24],
            "medium": [24, 72],
            "extended": [72, 120],
            "long_range": [120, 1000],
        }
    )
    seasons: Dict[str, List[int]] = Field(
        default_factory=lambda: {
            "winter": [12, 1, 2],
            "spring": [3, 4, 5],
            "summer": [6, 7, 8],
            "autumn": [9, 10, 11],
        }
    )
    skill_estimation: SkillEstimationConfig = Field(default_factory=SkillEstimationConfig)
    regression: RegressionConfig = Field(default_factory=RegressionConfig)
    weather_regime: WeatherRegimeConfig = Field(default_factory=WeatherRegimeConfig)
    neural_gating: NeuralGatingConfig = Field(default_factory=NeuralGatingConfig)
    spatial: SpatialConfig = Field(default_factory=SpatialConfig)
    cnn: SpatialCNNConfig = Field(default_factory=SpatialCNNConfig)
    spatiotemporal: SpatiotemporalConfig = Field(default_factory=SpatiotemporalConfig)
    probabilistic: ProbabilisticConfig = Field(default_factory=ProbabilisticConfig)
    emos: EMOSConfig = Field(default_factory=EMOSConfig)
    extreme_calibration: ExtremeCalibrationConfig = Field(default_factory=ExtremeCalibrationConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    extremes: ExtremesEvaluationConfig = Field(default_factory=ExtremesEvaluationConfig)
    lead_times_hours: List[int]
    regions: Dict[str, RegionConfig]
    models: Dict[str, ModelMetadataConfig]
    evaluation: EvaluationConfig
    extreme_event_thresholds: ExtremeThresholdsConfig
    paths: PathsConfig
    synthetic_data: SyntheticDataConfig
    weatherbench2: WeatherBench2Config = Field(default_factory=WeatherBench2Config)


_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def load_config(config_path: Optional[Path | str] = None) -> AppConfig:
    """
    Load and validate configuration from a YAML file.
    Environment variable DATA_MODE overrides data.mode if present.
    """
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw_data = yaml.safe_load(f)

    # Check for DATA_MODE environment variable override
    env_data_mode = os.environ.get("DATA_MODE")
    if env_data_mode:
        if "data" not in raw_data or not isinstance(raw_data["data"], dict):
            raw_data["data"] = {}
        raw_data["data"]["mode"] = env_data_mode.lower()

    return AppConfig(**raw_data)


# Cached singleton instance for convenient access across the project
_CONFIG_INSTANCE: Optional[AppConfig] = None


def get_config(reload: bool = False) -> AppConfig:
    """Get the active configuration instance (cached singleton)."""
    global _CONFIG_INSTANCE
    if _CONFIG_INSTANCE is None or reload:
        _CONFIG_INSTANCE = load_config()
    return _CONFIG_INSTANCE
