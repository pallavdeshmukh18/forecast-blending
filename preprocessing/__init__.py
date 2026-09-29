"""
Preprocessing module for meteorological data cleaning, alignment, spatial/temporal matching,
lead-time binning, season classification, and feature extraction.
"""

from preprocessing.cleaning import clip_to_physical_bounds, remove_invalid_records
from preprocessing.alignment import align_forecasts_and_observations, pivot_models_wide
from preprocessing.spatial import haversine_distance_km, match_spatial_nearest
from preprocessing.temporal import validate_and_compute_valid_time, match_temporal_tolerance
from preprocessing.features import (
    add_temporal_features,
    add_lead_time_category,
    add_lead_time_bins,
    add_season_feature,
)
from preprocessing.regimes import WeatherRegimeClassifier
from preprocessing.matching import match_forecasts_to_observations
from preprocessing.weather_regimes import (
    extract_atmospheric_features,
    classify_rule_based_regime,
    RuleBasedRegimeClassifier,
    REGIME_TO_ID,
    ID_TO_REGIME,
)
from preprocessing.regime_clustering import (
    KMeansRegimeClassifier,
    DEFAULT_CLUSTER_FEATURES,
    evaluate_cluster_range,
)

__all__ = [
    "clip_to_physical_bounds",
    "remove_invalid_records",
    "align_forecasts_and_observations",
    "pivot_models_wide",
    "haversine_distance_km",
    "match_spatial_nearest",
    "validate_and_compute_valid_time",
    "match_temporal_tolerance",
    "add_temporal_features",
    "add_lead_time_category",
    "add_lead_time_bins",
    "add_season_feature",
    "WeatherRegimeClassifier",
    "match_forecasts_to_observations",
    "extract_atmospheric_features",
    "classify_rule_based_regime",
    "RuleBasedRegimeClassifier",
    "REGIME_TO_ID",
    "ID_TO_REGIME",
    "KMeansRegimeClassifier",
    "DEFAULT_CLUSTER_FEATURES",
    "evaluate_cluster_range",
]


