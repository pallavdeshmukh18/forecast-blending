"""
FastAPI application for the Hybrid AI-NWP Forecast Blending System.
Exposes real/synthetic pipelines, baseline equal-weight blending, adaptive skill-weighted
blending (Inverse-Variance, Inverse-MAE, Ridge Regression), diagnostics, and alerts.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_project_root = Path(__file__).resolve().parent.parent.parent
_ml_dir = _project_root / "ML"
_backend_dir = _project_root / "Backend"
for _p in [str(_ml_dir), str(_backend_dir), str(_project_root)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from configs import get_config
from ingestion.schema import ForecastRecord, validate_forecast_dataframe
from ingestion.synthetic import SyntheticDataGenerator
from ingestion.readers import load_weather_file
from ingestion.variable_mapping import standardize_dataframe_variables
from ingestion.quality_control import apply_quality_control
from models.baseline import equal_weight_blend, EqualWeightBlender
from models.skill_weighting import InverseVarianceSkillBlender, InverseMAESkillBlender, SkillWeightedBlender
from models.adaptive_blending import RidgeBlender
from models.regime_blending import RegimeInverseVarianceBlender, RegimeInverseMAEBlender, RegimeSkillBlender
from models.gating_model import WeatherRegimeGatingNetwork, RegimeGatingBlender
from preprocessing.alignment import align_forecasts_and_observations
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
    evaluate_cluster_range,
)
from evaluation.comparison import compare_models, compute_skill_rankings
from evaluation.walk_forward import (
    WalkForwardBlenderEvaluator,
    compute_relative_improvements,
    compute_paired_bootstrap_ci,
    evaluate_regime_stratified_performance,
)
from extremes.rainfall import detect_extreme_rainfall
from extremes.heatwave import detect_heatwave_events
from extremes.wind import detect_wind_extremes

# Phase 5 Spatial & Probabilistic Imports
from data.synthetic_spatial import generate_synthetic_spatial_dataset
from preprocessing.spatial_grid import GridDefinition, SpatialGridAligner
from preprocessing.spatial_features import (
    extract_spatial_feature_stack,
    compute_model_disagreement_2d,
    compute_spatial_gradients,
    compute_local_spatial_statistics,
)
from models.spatial_blending import (
    SpatialWeightedBlender,
    SpatialEqualWeightBlender,
    SpatialInverseVarianceBlender,
)
from models.spatial_gating import SpatialCNNBlender
from models.probabilistic import (
    ProbabilisticForecast,
    EnsembleProbabilisticModel,
    ResidualUncertaintyProbabilisticModel,
)
from models.emos import EnsembleModelOutputStatistics
from evaluation.probabilistic_metrics import (
    crps_gaussian,
    crps_ensemble,
    crps_skill_score,
    prediction_interval_coverage_probability,
    mean_prediction_interval_width,
    sharpness,
    spatial_mae,
    spatial_rmse,
)
from evaluation.calibration import (
    brier_score,
    brier_skill_score,
    compute_reliability_curve,
    evaluate_event_calibration,
)
from data.provenance import get_provenance_tracker
from pipeline.forecast_pipeline import run_forecast_pipeline
from api.live_routes import router as live_router

app = FastAPI(
    title="Hybrid AI-NWP Multi-Model Forecast Blending System API",
    description="Research-oriented API for adaptive multi-model meteorological forecast blending, weather regimes, 2D spatial CNN gating, probabilistic calibration, and real-time operational live forecasting.",
    version="0.6.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(live_router)



class BlendRequest(BaseModel):
    forecasts: List[ForecastRecord]
    models: Optional[List[str]] = None
    min_valid_models: int = 1


class AdaptiveBlendRequest(BaseModel):
    forecasts: List[ForecastRecord]
    blender: str = Field(default="inverse_variance", description="Blender algorithm: inverse_variance, inverse_mae, ridge_constrained, ridge_unconstrained, regime_inverse_variance, regime_inverse_mae, neural_gating, equal_weight")
    weight_smoothing: Optional[float] = 0.10
    alpha: Optional[float] = 1.0


class RegimeClassifyRequest(BaseModel):
    records: List[Dict[str, Any]]
    method: str = Field(default="rule_based", description="Classification method: 'rule_based' or 'clustering'")
    n_clusters: Optional[int] = 4


class GenerateResponse(BaseModel):
    status: str
    num_observation_records: int
    num_forecast_records: int
    variables: List[str]
    lead_times: List[int]


# Cached global models and comparison results for instant API responsiveness
_blender_cache: Dict[str, Any] = {}
_cached_comparison_results: Optional[Dict[str, Any]] = None
_cached_phase4_benchmark: Optional[Dict[str, Any]] = None


def _clean_dataframe_for_json(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert pandas DataFrame to JSON-serializable list of dicts without NaN or NaT."""
    records = []
    for row in df.to_dict(orient="records"):
        clean_row = {}
        for k, v in row.items():
            if pd.isna(v):
                clean_row[k] = None
            elif hasattr(v, "isoformat"):
                clean_row[k] = v.isoformat()
            elif isinstance(v, (np.floating, float)):
                clean_row[k] = None if np.isnan(v) else float(v)
            elif isinstance(v, (np.integer, int)):
                clean_row[k] = int(v)
            else:
                clean_row[k] = v
        records.append(clean_row)
    return records


def _get_fitted_blender(blender_type: str) -> Any:
    """Retrieve or initialize a fitted blender for API inference."""
    cfg = get_config()
    b_type = blender_type.lower()

    if b_type in _blender_cache:
        return _blender_cache[b_type]

    from scripts.run_phase4_experiment import generate_phase4_evaluation_dataset

    # Generate controlled multi-regime training history
    fcs_df, obs_df = generate_phase4_evaluation_dataset(n_samples=400, seed=42)
    verified_df = match_forecasts_to_observations(fcs_df, obs_df, config=cfg)

    if b_type in ["inverse_variance", "inverse_variance_blend"]:
        blender = InverseVarianceSkillBlender(
            epsilon=cfg.skill_estimation.epsilon,
            weight_smoothing=cfg.skill_estimation.weight_smoothing,
            minimum_samples=5,
            config=cfg,
        )
    elif b_type in ["inverse_mae", "inverse_mae_blend"]:
        blender = InverseMAESkillBlender(
            epsilon=cfg.skill_estimation.epsilon,
            weight_smoothing=cfg.skill_estimation.weight_smoothing,
            minimum_samples=5,
            config=cfg,
        )
    elif b_type in ["ridge", "ridge_constrained", "ridge_constrained_blend"]:
        blender = RidgeBlender(
            alpha=cfg.regression.alpha,
            fit_intercept=cfg.regression.fit_intercept,
            constrained_weights=True,
            minimum_samples=5,
            config=cfg,
        )
    elif b_type in ["ridge_unconstrained", "ridge_unconstrained_blend"]:
        blender = RidgeBlender(
            alpha=cfg.regression.alpha,
            fit_intercept=cfg.regression.fit_intercept,
            constrained_weights=False,
            minimum_samples=5,
            config=cfg,
        )
    elif b_type in ["regime_inverse_variance", "regime_inverse_variance_blend"]:
        blender = RegimeInverseVarianceBlender(
            epsilon=cfg.skill_estimation.epsilon,
            weight_smoothing=cfg.skill_estimation.weight_smoothing,
            minimum_samples=5,
            config=cfg,
        )
    elif b_type in ["regime_inverse_mae", "regime_inverse_mae_blend"]:
        blender = RegimeInverseMAEBlender(
            epsilon=cfg.skill_estimation.epsilon,
            weight_smoothing=cfg.skill_estimation.weight_smoothing,
            minimum_samples=5,
            config=cfg,
        )
    elif b_type in ["neural_gating", "neural_gating_blend", "gating"]:
        blender = WeatherRegimeGatingNetwork(
            hidden_sizes=[64, 32],
            epochs=15,
            batch_size=32,
            learning_rate=0.005,
            random_seed=42,
            config=cfg,
        )
    elif b_type in ["equal_weight", "equal_weight_blend"]:
        blender = EqualWeightBlender(name="equal_weight_blend")
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown blender type '{blender_type}'. Supported: inverse_variance, inverse_mae, ridge_constrained, ridge_unconstrained, regime_inverse_variance, regime_inverse_mae, neural_gating, equal_weight",
        )

    blender.fit_verification_data(verified_df)
    _blender_cache[b_type] = blender
    return blender



@app.get("/")
def root():
    """System overview and metadata."""
    cfg = get_config()
    return {
        "project": cfg.project_name,
        "version": cfg.version,
        "status": "active",
        "data_mode": cfg.data.mode,
        "phase": "Phase 3 — Adaptive Multi-Model Skill-Weighted Blending",
        "supported_blenders": [
            "equal_weight_blend",
            "inverse_mae_blend",
            "inverse_variance_blend",
            "ridge_constrained_blend",
            "ridge_unconstrained_blend",
        ],
        "supported_models": list(cfg.models.keys()),
        "supported_variables": list(cfg.variables.keys()),
        "lead_times_hours": cfg.lead_times_hours,
        "regions": list(cfg.regions.keys()),
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "forecast-blending-engine", "version": "0.6.0"}


@app.get("/config")
def read_config():
    """Retrieve active system configuration."""
    return get_config().model_dump()


@app.get("/data/status")
def get_data_status():
    """Retrieve data pipeline status, execution mode, and alignment parameters."""
    cfg = get_config()
    return {
        "mode": cfg.data.mode,
        "spatial_alignment": cfg.spatial_alignment.model_dump(),
        "temporal_alignment": cfg.temporal_alignment.model_dump(),
        "units": cfg.units.model_dump(),
        "lead_time_bins": cfg.lead_time_bins,
        "paths": cfg.paths.model_dump(),
    }


@app.post("/synthetic/generate", response_model=GenerateResponse)
def generate_synthetic_data(save_to_disk: bool = Query(False, description="Persist generated files to data paths")):
    """
    Generate deterministic synthetic meteorological observations and multi-model forecasts.
    """
    generator = SyntheticDataGenerator()
    obs_df, forecasts_df = generator.generate(save_to_disk=save_to_disk)

    return GenerateResponse(
        status="success",
        num_observation_records=len(obs_df),
        num_forecast_records=len(forecasts_df),
        variables=list(obs_df["variable"].unique()),
        lead_times=sorted(list(forecasts_df["lead_time"].unique())),
    )


@app.post("/blend/baseline", response_model=List[ForecastRecord])
def run_baseline_blend(request: BlendRequest):
    """
    Run equal-weight multi-model blending on an input list of forecast records.
    """
    if not request.forecasts:
        raise HTTPException(status_code=400, detail="Forecast record list cannot be empty.")

    raw_dicts = [f.model_dump() for f in request.forecasts]
    input_df = pd.DataFrame(raw_dicts)

    blended_df = equal_weight_blend(
        forecasts_df=input_df,
        models=request.models,
        min_valid_models=request.min_valid_models,
        output_model_name="equal_weight_blend",
    )

    records = blended_df.to_dict(orient="records")
    return [ForecastRecord(**r) for r in records]


# -------------------------------------------------------------
# Phase 3 Adaptive Blending Endpoints
# -------------------------------------------------------------


@app.get("/blending/weights")
def get_blending_weights(
    variable: str = Query("temperature", description="Target meteorological variable"),
    lead_time: Optional[int] = Query(24, description="Forecast horizon in hours"),
    region: Optional[str] = Query("plains", description="Geographic region context"),
    blender: str = Query("inverse_variance", description="Blender type: inverse_variance, inverse_mae, ridge_constrained, ridge_unconstrained"),
):
    """
    Retrieve dynamically computed model blending weights and hierarchical scope metadata.
    """
    blender_obj = _get_fitted_blender(blender)
    weights = blender_obj.get_weights(variable=variable, lead_time=lead_time, region=region)
    metadata = blender_obj.get_weight_metadata(variable=variable, lead_time=lead_time, region=region)

    return {
        "variable": variable,
        "lead_time": lead_time,
        "region": region,
        "blender": blender_obj.name,
        "weights": weights,
        "metadata": metadata,
    }


@app.post("/blending/forecast")
def run_adaptive_blend(request: AdaptiveBlendRequest):
    """
    Generate skill-weighted adaptive blended forecasts for input multi-model forecasts.
    Returns blended forecast records with diagnostic spread and historical residual std.
    """
    if not request.forecasts:
        raise HTTPException(status_code=400, detail="Forecast record list cannot be empty.")

    raw_dicts = [f.model_dump() for f in request.forecasts]
    input_df = pd.DataFrame(raw_dicts)

    blender_obj = _get_fitted_blender(request.blender)
    blended_df = blender_obj.predict(input_df)

    if blended_df.empty:
        raise HTTPException(status_code=400, detail="Unable to produce blend from input forecasts.")

    clean_records = _clean_dataframe_for_json(blended_df)

    return {
        "blender": blender_obj.name,
        "num_records": len(clean_records),
        "forecasts": clean_records,
    }


@app.get("/blending/skill")
def get_historical_skill(
    variable: Optional[str] = Query(None, description="Filter by variable"),
    lead_time: Optional[int] = Query(None, description="Filter by lead time"),
    region: Optional[str] = Query(None, description="Filter by region"),
):
    """
    Retrieve historical error skill metrics broken down across models, horizons, and domains.
    """
    cfg = get_config()
    generator = SyntheticDataGenerator(cfg)
    obs_df, fcs_df = generator.generate(seed=cfg.synthetic_data.seed)
    verified_df = match_forecasts_to_observations(fcs_df, obs_df, config=cfg)

    group_keys = ["model_name", "variable"]
    if lead_time is not None or "lead_time" in verified_df.columns:
        group_keys.append("lead_time")
    if region is not None or "region" in verified_df.columns:
        group_keys.append("region")

    comp_df = compare_models(verified_df, group_by=group_keys)

    if variable:
        comp_df = comp_df[comp_df["variable"] == variable]
    if lead_time is not None and "lead_time" in comp_df.columns:
        comp_df = comp_df[comp_df["lead_time"] == lead_time]
    if region and "region" in comp_df.columns:
        comp_df = comp_df[comp_df["region"] == region]

    clean_metrics = _clean_dataframe_for_json(comp_df)

    return {
        "num_records": len(clean_metrics),
        "skill_metrics": clean_metrics,
    }


@app.get("/blending/comparison")
def get_blending_comparison():
    """
    Execute full walk-forward validation benchmarking all member models and blenders:
    Equal-Weight, Inverse-MAE, Inverse-Variance, Constrained Ridge, and Unconstrained Ridge.
    """
    global _cached_comparison_results
    if _cached_comparison_results is not None:
        return _cached_comparison_results

    cfg = get_config()
    from scripts.run_phase3_experiment import generate_controlled_experiment_data
    obs_df, fcs_df = generate_controlled_experiment_data(seed=42)

    evaluator = WalkForwardBlenderEvaluator(mode="rolling", window_size=15, min_train_steps=6, config=cfg)
    verified_results = evaluator.run(fcs_df, obs_df)

    summary_df = compare_models(verified_results, group_by=["variable", "lead_time"])
    ranked_df = compute_skill_rankings(summary_df, metric="rmse")
    improvement_df = compute_relative_improvements(ranked_df, baseline_model="equal_weight_blend", metric="rmse")

    overall_df = compare_models(verified_results, group_by=["variable"])
    overall_imp = compute_relative_improvements(overall_df, baseline_model="equal_weight_blend", metric="rmse")

    _cached_comparison_results = {
        "overall_by_variable": _clean_dataframe_for_json(overall_imp),
        "by_variable_and_lead_time": _clean_dataframe_for_json(improvement_df),
    }
    return _cached_comparison_results


@app.get("/evaluation/summary")
def get_evaluation_summary():
    """
    Execute baseline evaluation benchmarking individual models against the equal-weight blend.
    """
    generator = SyntheticDataGenerator()
    obs_df, forecasts_df = generator.generate()

    blended_df = equal_weight_blend(forecasts_df)
    combined_forecasts = pd.concat([forecasts_df, blended_df], ignore_index=True)
    aligned_df = align_forecasts_and_observations(combined_forecasts, obs_df)

    comparison_df = compare_models(aligned_df, group_by=["variable", "lead_time"])
    overall_df = compare_models(aligned_df, group_by=["variable"])

    return {
        "overall_by_variable": _clean_dataframe_for_json(overall_df),
        "by_variable_and_lead_time": _clean_dataframe_for_json(comparison_df),
    }


@app.get("/extremes/alerts")
def get_extreme_weather_alerts():
    """
    Scan blended forecasts for extreme precipitation, heatwave, and gale wind thresholds.
    """
    generator = SyntheticDataGenerator()
    _, forecasts_df = generator.generate()
    blended_df = equal_weight_blend(forecasts_df)

    heavy_rain = detect_extreme_rainfall(blended_df)
    heatwaves = detect_heatwave_events(blended_df)
    gale_winds = detect_wind_extremes(blended_df)

    return {
        "extreme_rainfall_alerts_count": len(heavy_rain),
        "heatwave_alerts_count": len(heatwaves),
        "gale_wind_alerts_count": len(gale_winds),
        "sample_extreme_rainfall": _clean_dataframe_for_json(heavy_rain.head(10)),
        "sample_heatwaves": _clean_dataframe_for_json(heatwaves.head(10)),
        "sample_gale_winds": _clean_dataframe_for_json(gale_winds.head(10)),
    }


@app.post("/regimes/classify")
def classify_regimes(request: RegimeClassifyRequest):
    """
    Classify synoptic weather regimes for input weather records using rule-based thresholds or KMeans clustering.
    """
    if not request.records:
        raise HTTPException(status_code=400, detail="Empty records list.")

    df = pd.DataFrame(request.records)
    cfg = get_config()

    if request.method.lower() == "clustering":
        clf = KMeansRegimeClassifier(n_clusters=request.n_clusters or 4, random_state=42, config=cfg)
        clf.fit(df)
        classified_df = clf.predict(df)
        diagnostics = clf.get_diagnostics()
        profiles = clf.get_cluster_profiles()
    else:
        classified_df = classify_rule_based_regime(df, config=cfg)
        diagnostics = {"method": "rule_based", "priority": ["heavy_rain", "convective", "high_wind", "hot_extreme", "cold_extreme", "transition", "stable"]}
        profiles = {}

    clean_records = _clean_dataframe_for_json(classified_df)
    return {
        "method": request.method,
        "num_records": len(clean_records),
        "diagnostics": diagnostics,
        "cluster_profiles": profiles,
        "classified_records": clean_records,
    }


@app.get("/regimes/clusters")
def get_cluster_diagnostics(
    k_min: int = Query(3, ge=2, le=10),
    k_max: int = Query(6, ge=2, le=10),
):
    """
    Evaluate unsupervised KMeans clustering diagnostics (silhouette scores, inertia, cluster sizes) across range of K.
    """
    cfg = get_config()
    from scripts.run_phase4_experiment import generate_phase4_evaluation_dataset

    _, obs_df = generate_phase4_evaluation_dataset(n_samples=400, seed=42)
    k_values = list(range(k_min, k_max + 1))
    diag_results = evaluate_cluster_range(obs_df, k_values=k_values, random_state=42, config=cfg)

    return {
        "k_range": k_values,
        "evaluations": diag_results,
    }


@app.post("/blending/gating")
def blend_with_neural_gating(request: AdaptiveBlendRequest):
    """
    Execute Neural Weather Regime Gating or Regime-Conditioned Blending inference.
    Returns blended forecast values, model weights, and uncertainty spreads.
    """
    records = [r.model_dump() for r in request.forecasts]
    df = pd.DataFrame(records)
    if df.empty:
        raise HTTPException(status_code=400, detail="Empty forecasts list.")

    blender_name = request.blender or "neural_gating"
    blender = _get_fitted_blender(blender_name)

    blended_df = blender.predict(df)
    clean_records = _clean_dataframe_for_json(blended_df)

    return {
        "status": "success",
        "blender": blender.name,
        "num_records": len(clean_records),
        "blended_forecasts": clean_records,
    }


@app.get("/evaluation/phase4-benchmark")
def get_phase4_benchmark():
    """
    Execute and return the complete Phase 4 multi-model benchmark across all 10 member models and blenders:
    Individual Models, Equal-Weight, Inverse-MAE, Inverse-Variance, Constrained Ridge,
    Regime Inverse-Variance, Regime Inverse-MAE, and Neural Gating.
    """
    global _cached_phase4_benchmark
    if _cached_phase4_benchmark is not None:
        return _cached_phase4_benchmark

    from scripts.run_phase4_experiment import generate_phase4_evaluation_dataset, compute_contingency_metrics

    cfg = get_config()
    fcs_df, obs_df = generate_phase4_evaluation_dataset(n_samples=600, seed=42)

    unique_ts = sorted(fcs_df["timestamp"].unique())
    n_ts = len(unique_ts)
    idx_train = int(0.60 * n_ts)
    train_ts = unique_ts[:idx_train]
    test_ts = unique_ts[idx_train:]

    f_train = fcs_df[fcs_df["timestamp"].isin(train_ts)].copy()
    f_test = fcs_df[fcs_df["timestamp"].isin(test_ts)].copy()
    obs_train = obs_df[obs_df["timestamp"].isin(train_ts)].copy()
    obs_test = obs_df[obs_df["timestamp"].isin(test_ts)].copy()

    verified_train = match_forecasts_to_observations(f_train, obs_train, config=cfg)

    blenders = {
        "equal_weight_blend": EqualWeightBlender(name="equal_weight_blend"),
        "inverse_mae_blend": InverseMAESkillBlender(name="inverse_mae_blend", config=cfg),
        "inverse_variance_blend": InverseVarianceSkillBlender(name="inverse_variance_blend", config=cfg),
        "ridge_constrained_blend": RidgeBlender(constrained_weights=True, name="ridge_constrained_blend", config=cfg),
        "regime_inverse_variance_blend": RegimeInverseVarianceBlender(name="regime_inverse_variance_blend", config=cfg),
        "regime_inverse_mae_blend": RegimeInverseMAEBlender(name="regime_inverse_mae_blend", config=cfg),
        "neural_gating_blend": WeatherRegimeGatingNetwork(
            hidden_sizes=[64, 32],
            epochs=15,
            batch_size=32,
            learning_rate=0.005,
            random_seed=42,
            name="neural_gating_blend",
            config=cfg,
        ),
    }

    test_predictions = [f_test]
    for name, blender in blenders.items():
        blender.fit_verification_data(verified_train)
        pred = blender.predict(f_test)
        test_predictions.append(pred)

    combined_test = pd.concat(test_predictions, ignore_index=True)
    verified_test = match_forecasts_to_observations(combined_test, obs_test, config=cfg)
    verified_test["error"] = verified_test["forecast_value"] - verified_test["observed_value"]

    if "weather_regime" not in verified_test.columns or verified_test["weather_regime"].isna().any():
        verified_test = classify_rule_based_regime(verified_test, config=cfg)

    overall_comp = compare_models(verified_test, group_by=["variable"])
    regime_comp = evaluate_regime_stratified_performance(verified_test)

    # Contingency metrics for heavy rain
    rain_test = verified_test[verified_test["variable"] == "rainfall"]
    extreme_records = []
    for model_name in verified_test["model_name"].unique():
        m_df = rain_test[rain_test["model_name"] == model_name]
        if not m_df.empty:
            c_metrics = compute_contingency_metrics(
                obs=m_df["observed_value"].values,
                pred=m_df["forecast_value"].values,
                threshold=35.5,
            )
            extreme_records.append({"model_name": model_name, **c_metrics})

    # Bootstrap CIs vs Equal Weight
    eq_errors = verified_test[verified_test["model_name"] == "equal_weight_blend"]["error"].values
    gating_errors = verified_test[verified_test["model_name"] == "neural_gating_blend"]["error"].values
    ci_gating = compute_paired_bootstrap_ci(eq_errors, gating_errors, metric="rmse", n_bootstrap=500)

    _cached_phase4_benchmark = {
        "overall_performance": _clean_dataframe_for_json(overall_comp),
        "regime_stratified_performance": _clean_dataframe_for_json(regime_comp),
        "extreme_rainfall_contingency": extreme_records,
        "bootstrap_ci_gating_vs_equal_weight": ci_gating,
    }
    return _cached_phase4_benchmark


# ---------------------------------------------------------------------------
# PHASE 5: SPATIAL GRID, CONVOLUTIONAL BLENDING & PROBABILISTIC ENDPOINTS
# ---------------------------------------------------------------------------

_cached_spatial_dataset = None
_cached_spatial_cnn = None
_cached_spatial_emos = None


def _get_spatial_data():
    global _cached_spatial_dataset
    if _cached_spatial_dataset is None:
        cfg = get_config()
        fc, obs = generate_synthetic_spatial_dataset(
            config=cfg,
            start_date="2026-01-01",
            end_date="2026-01-15",
            time_step_hours=6,
            lead_times=[6, 12, 24, 48],
            grid_shape=(16, 16),
            seed=42,
        )
        _cached_spatial_dataset = (fc, obs)
    return _cached_spatial_dataset


@app.get("/spatial/grid", summary="Retrieve spatial grid coordinate definitions and bounding box")
def get_spatial_grid_definition():
    """Returns the standardized 2D spatial grid metadata and bounds."""
    cfg = get_config()
    gd = GridDefinition(
        lat_min=cfg.spatial.lat_min,
        lat_max=cfg.spatial.lat_max,
        lon_min=cfg.spatial.lon_min,
        lon_max=cfg.spatial.lon_max,
        n_lat=cfg.synthetic_data.spatial_grid_shape[0],
        n_lon=cfg.synthetic_data.spatial_grid_shape[1],
    )
    return gd.to_dict()


@app.get("/spatial/features", summary="Extract and summarize 2D spatial meteorological features")
def get_spatial_features(
    variable: str = Query("temperature", description="Target meteorological variable"),
    lead_time: int = Query(24, description="Forecast lead time in hours"),
):
    """Extracts spatial feature channels (gradients, local statistics, anomalies, disagreement)."""
    fc, _ = _get_spatial_data()
    if variable not in fc:
        raise HTTPException(status_code=400, detail=f"Variable '{variable}' not found.")

    stack, channel_names = extract_spatial_feature_stack(
        fc,
        variable=variable,
        lead_time=lead_time,
        time_idx=0,
    )

    summaries = {}
    for idx, name in enumerate(channel_names):
        chan = stack[idx]
        summaries[name] = {
            "mean": round(float(np.nanmean(chan)), 4),
            "std": round(float(np.nanstd(chan)), 4),
            "min": round(float(np.nanmin(chan)), 4),
            "max": round(float(np.nanmax(chan)), 4),
        }

    return {
        "variable": variable,
        "lead_time": lead_time,
        "grid_shape": list(stack.shape[1:]),
        "num_channels": len(channel_names),
        "channel_summaries": summaries,
    }


@app.get("/spatial/disagreement", summary="Compute spatial multi-model disagreement layer")
def get_spatial_disagreement(
    variable: str = Query("rainfall", description="Target variable"),
    lead_time: int = Query(24, description="Lead time in hours"),
):
    """Computes inter-model ensemble spread, range, and pairwise model differences across the 2D grid."""
    fc, _ = _get_spatial_data()
    if variable not in fc:
        raise HTTPException(status_code=400, detail=f"Variable '{variable}' not found.")

    target_data = fc[variable].sel(lead_time=lead_time).isel(time=0).values  # (M, H, W)
    models = [str(m) for m in fc.model.values]
    disagree = compute_model_disagreement_2d(target_data, model_names=models)

    return {
        "variable": variable,
        "lead_time": lead_time,
        "models": models,
        "ensemble_mean_avg": round(float(np.mean(disagree["mean"])), 4),
        "intermodel_spread_avg": round(float(np.mean(disagree["std"])), 4),
        "max_spread_cell": round(float(np.max(disagree["std"])), 4),
        "intermodel_range_avg": round(float(np.mean(disagree["range"])), 4),
        "pairwise_comparisons": list(disagree["pairwise_diffs"].keys()),
    }


@app.get("/spatial/weights", summary="Retrieve 2D spatial model weights W_m(x,y)")
def get_spatial_weights(
    variable: str = Query("rainfall", description="Target variable"),
    method: str = Query("spatial_cnn", description="Weight resolver: 'equal_weight', 'spatial_inv_var', 'spatial_cnn'"),
    lead_time: int = Query(24, description="Lead time in hours"),
):
    """Returns spatial model weights W_m(x,y) statistics and per-model spatial distribution."""
    fc, obs = _get_spatial_data()
    models = [str(m) for m in fc.model.values]
    M = len(models)

    if method == "equal_weight":
        W = np.full((M, 16, 16), 1.0 / M, dtype=np.float32)
    elif method == "spatial_inv_var":
        train_fc = fc[variable].sel(lead_time=lead_time).values
        train_obs = obs[variable].values
        blender = SpatialInverseVarianceBlender(model_names=models)
        blender.fit(train_fc, train_obs)
        W = blender.get_weights()
    else:  # spatial_cnn
        train_fc = fc[variable].sel(lead_time=lead_time).values
        train_obs = obs[variable].values
        X_list = [extract_spatial_feature_stack(fc, variable=variable, lead_time=lead_time, time_idx=i)[0] for i in range(len(fc.time))]
        X = np.stack(X_list, axis=0)

        blender = SpatialCNNBlender(num_models=M, in_channels=X.shape[1], epochs=10, batch_size=8, random_seed=42)
        blender.fit(X, train_fc, train_obs)
        _, W_4d = blender.predict(X[:1], train_fc[:1])
        W = W_4d[0]

    model_weight_stats = {}
    for idx, m_name in enumerate(models):
        w_m = W[idx]
        model_weight_stats[m_name] = {
            "mean_weight": round(float(np.mean(w_m)), 4),
            "min_weight": round(float(np.min(w_m)), 4),
            "max_weight": round(float(np.max(w_m)), 4),
            "std_weight": round(float(np.std(w_m)), 4),
        }

    return {
        "variable": variable,
        "method": method,
        "lead_time": lead_time,
        "models": models,
        "weight_sum_check": round(float(np.sum(W, axis=0).mean()), 6),
        "model_weights": model_weight_stats,
    }


@app.get("/probabilistic/forecast", summary="Generate probabilistic forecast distribution")
def get_probabilistic_forecast(
    variable: str = Query("temperature", description="Target variable"),
    lead_time: int = Query(24, description="Lead time in hours"),
    method: str = Query("emos", description="Probabilistic method: 'ensemble', 'emos', 'residual'"),
):
    """Generates a ProbabilisticForecast distribution (mean, std, quantiles P10, P25, P50, P75, P90)."""
    fc, obs = _get_spatial_data()
    target_fc = fc[variable].sel(lead_time=lead_time).values
    target_obs = obs[variable].values

    if method == "emos":
        emos = EnsembleModelOutputStatistics(fit_method="crps")
        emos.fit(target_fc, target_obs)
        pf = emos.predict(target_fc[:1])
    elif method == "ensemble":
        ens_model = EnsembleProbabilisticModel()
        pf = ens_model.predict(target_fc[:1])
    else:  # residual
        res_model = ResidualUncertaintyProbabilisticModel()
        eq_pred = np.mean(target_fc, axis=1)
        res_model.fit_residuals(eq_pred, target_obs)
        pf = res_model.predict(eq_pred[:1])

    low_80, high_80 = pf.prediction_interval(0.80)

    return {
        "variable": variable,
        "method": method,
        "lead_time": lead_time,
        "distribution_type": pf.distribution_type,
        "mean_avg": round(float(np.mean(pf.mean)), 3),
        "std_avg": round(float(np.mean(pf.std)), 3),
        "quantiles_evaluated": list(pf.quantiles.keys()),
        "p10_avg": round(float(np.mean(pf.quantile(0.10))), 3),
        "p50_avg": round(float(np.mean(pf.quantile(0.50))), 3),
        "p90_avg": round(float(np.mean(pf.quantile(0.90))), 3),
        "mean_interval_width_80": round(float(np.mean(high_80 - low_80)), 3),
    }


@app.get("/probabilistic/calibration", summary="Compute probability calibration and reliability curve")
def get_probability_calibration(
    variable: str = Query("rainfall", description="Target variable"),
    threshold: float = Query(25.0, description="Event threshold value"),
):
    """Calculates reliability curve, Brier Score, and BSS for event threshold exceedance."""
    fc, obs = _get_spatial_data()
    train_fc = fc[variable].sel(lead_time=24).values
    train_obs = obs[variable].values

    emos = EnsembleModelOutputStatistics(fit_method="crps")
    emos.fit(train_fc, train_obs)
    pf = emos.predict(train_fc)

    diag = evaluate_event_calibration(train_obs.ravel(), pf, threshold=threshold, operator=">=")
    return {
        "variable": variable,
        "threshold": threshold,
        "brier_score": diag["brier_score"],
        "brier_skill_score": diag["brier_skill_score"],
        "reliability_error": diag["reliability_error"],
        "event_frequency": diag["event_frequency"],
        "sample_count": diag["sample_count"],
        "reliability_curve": {
            "bin_edges": diag["bin_edges"],
            "bin_centers": diag["bin_centers"],
            "mean_predicted_probs": diag["mean_predicted_probs"],
            "observed_frequencies": diag["observed_frequencies"],
            "bin_counts": diag["bin_counts"],
        },
    }


@app.get("/extremes/probability-map", summary="Generate 2D spatial extreme event probability map")
def get_extreme_probability_map(
    variable: str = Query("rainfall", description="Target variable"),
    threshold: float = Query(25.0, description="Extreme event threshold"),
    lead_time: int = Query(24, description="Lead time in hours"),
):
    """Generates 2D spatial grid of exceedance probability P(variable >= threshold)."""
    fc, obs = _get_spatial_data()
    target_fc = fc[variable].sel(lead_time=lead_time).values
    target_obs = obs[variable].values

    emos = EnsembleModelOutputStatistics(fit_method="crps")
    emos.fit(target_fc, target_obs)
    pf = emos.predict(target_fc[:1])

    prob_grid = pf.prob_exceeds(threshold)[0]  # (H, W)

    return {
        "variable": variable,
        "threshold": threshold,
        "lead_time": lead_time,
        "grid_shape": list(prob_grid.shape),
        "mean_exceedance_probability": round(float(np.mean(prob_grid)), 4),
        "max_exceedance_probability": round(float(np.max(prob_grid)), 4),
        "high_risk_cells_count": int(np.sum(prob_grid >= 0.50)),
        "moderate_risk_cells_count": int(np.sum(prob_grid >= 0.20)),
    }


# =====================================================================
# Phase 6 Operational Backend Endpoints
# =====================================================================

from pipeline.forecast_pipeline import run_forecast_pipeline, ForecastResult
from data.provenance import get_provenance_tracker


class ForecastRunRequest(BaseModel):
    mode: str = "synthetic"  # 'synthetic', 'historical', 'real'
    variable: str = "rainfall"
    lead_time: int = 24
    region: Optional[str] = None
    blender_method: str = "regime_blender"  # 'equal_weight', 'inverse_variance', 'regime_blender', 'spatial_cnn'


class ForecastBlendPostRequest(BaseModel):
    forecasts: Optional[List[ForecastRecord]] = None
    blender_method: str = "regime_blender"
    variable: str = "rainfall"
    lead_time: int = 24
    region: Optional[str] = None


class ForecastProbabilityPostRequest(BaseModel):
    variable: str = "rainfall"
    lead_time: int = 24
    region: Optional[str] = None
    distribution_type: str = "gaussian"
    quantiles: List[float] = [0.10, 0.25, 0.50, 0.75, 0.90]


class ForecastExtremesPostRequest(BaseModel):
    variable: str = "rainfall"
    lead_time: int = 24
    region: Optional[str] = None
    threshold: Optional[float] = None


class VerifyPostRequest(BaseModel):
    variable: str = "rainfall"
    lead_time: int = 24
    blender_method: str = "regime_blender"


@app.post("/forecast/run", summary="Run full operational multi-model forecast blending pipeline", response_model=ForecastResult)
def api_forecast_run(request: ForecastRunRequest):
    """
    Executes the entire end-to-end forecasting pipeline:
    ingestion -> QC -> alignment -> feature extraction -> regime detection ->
    adaptive weighting -> probabilistic post-processing -> extreme events -> verification.
    """
    cfg = get_config()
    res = run_forecast_pipeline(
        mode=request.mode,
        variable=request.variable,
        lead_time=request.lead_time,
        region=request.region,
        blender_method=request.blender_method,
        config=cfg,
    )
    return res


@app.post("/forecast/blend", summary="Blend candidate forecasts with specified blending method")
def api_forecast_blend(request: ForecastBlendPostRequest):
    """
    Blends candidate multi-model forecasts into an optimal composite forecast.
    """
    cfg = get_config()
    res = run_forecast_pipeline(
        mode="synthetic",
        variable=request.variable,
        lead_time=request.lead_time,
        region=request.region,
        blender_method=request.blender_method,
        config=cfg,
    )
    return {
        "variable": request.variable,
        "lead_time": request.lead_time,
        "blender_method": request.blender_method,
        "weights": res.weights,
        "forecast_count": res.forecast_count,
        "forecast": res.forecast[:50],
        "diagnostics": res.diagnostics,
    }


@app.post("/forecast/probability", summary="Generate probabilistic forecast quantiles and uncertainty intervals")
def api_forecast_probability(request: ForecastProbabilityPostRequest):
    """
    Generates probabilistic forecast post-processing with calibrated spreads and quantiles.
    """
    cfg = get_config()
    res = run_forecast_pipeline(
        mode="synthetic",
        variable=request.variable,
        lead_time=request.lead_time,
        region=request.region,
        blender_method="regime_blender",
        config=cfg,
    )
    return {
        "variable": request.variable,
        "lead_time": request.lead_time,
        "distribution_type": request.distribution_type,
        "quantiles": request.quantiles,
        "uncertainty": res.uncertainty,
        "sample_forecasts": res.forecast[:25],
    }


@app.post("/forecast/extremes", summary="Generate extreme weather exceedance probabilities")
def api_forecast_extremes(request: ForecastExtremesPostRequest):
    """
    Generates extreme event exceedance probabilities across grid cells and stations.
    """
    cfg = get_config()
    res = run_forecast_pipeline(
        mode="synthetic",
        variable=request.variable,
        lead_time=request.lead_time,
        region=request.region,
        blender_method="regime_blender",
        config=cfg,
    )
    probs = res.probabilities
    if request.threshold is not None:
        probs = [p for p in probs if p.get("threshold") == request.threshold]

    return {
        "variable": request.variable,
        "lead_time": request.lead_time,
        "threshold_filter": request.threshold,
        "probability_count": len(probs),
        "probabilities": probs[:50],
    }


@app.post("/verify", summary="Verify forecast accuracy against ground truth observations")
def api_verify(request: VerifyPostRequest):
    """
    Calculates deterministic error metrics (MAE, RMSE, Bias, Correlation) and probabilistic metrics (CRPS, Brier).
    """
    cfg = get_config()
    res = run_forecast_pipeline(
        mode="synthetic",
        variable=request.variable,
        lead_time=request.lead_time,
        blender_method=request.blender_method,
        config=cfg,
    )
    if not res.verification:
        raise HTTPException(status_code=404, detail="No matching observation records found for verification.")

    return {
        "variable": request.variable,
        "lead_time": request.lead_time,
        "blender_method": request.blender_method,
        "verification": res.verification,
    }


@app.get("/models", summary="List available forecast models and metadata")
def api_get_models():
    """Returns available physical NWP and AI models with configurations."""
    cfg = get_config()
    return {
        "models": {
            name: {
                "type": m.type,
                "resolution_km": m.resolution_km,
                "update_frequency_hours": m.update_frequency_hours,
                "description": m.description,
            }
            for name, m in cfg.models.items()
        }
    }


@app.get("/models/weights", summary="Retrieve active model weights across regimes and variables")
def api_get_weights(
    variable: str = Query("rainfall", description="Target variable"),
    lead_time: int = Query(24, description="Lead time in hours"),
):
    """Returns active model weights computed by the blending system."""
    cfg = get_config()
    res = run_forecast_pipeline(
        mode="synthetic",
        variable=variable,
        lead_time=lead_time,
        blender_method="regime_blender",
        config=cfg,
    )
    return {
        "variable": variable,
        "lead_time": lead_time,
        "blender": "regime_blender",
        "weights": res.weights,
    }


@app.get("/regimes", summary="List available weather regimes and current regime definitions")
def api_get_regimes():
    """Returns weather regime classification definitions and priority hierarchy."""
    cfg = get_config()
    return {
        "method": cfg.weather_regime.method,
        "priority_order": cfg.weather_regime.thresholds.priority_order,
        "thresholds": cfg.weather_regime.thresholds.model_dump(),
        "clustering_clusters": cfg.weather_regime.clustering.n_clusters,
    }


@app.get("/diagnostics", summary="Retrieve system health and quality control diagnostics")
def api_get_diagnostics():
    """Returns machine-readable data quality report and pipeline diagnostics."""
    cfg = get_config()
    prov_tracker = get_provenance_tracker()
    return {
        "status": "operational",
        "version": cfg.version,
        "data_mode": cfg.data.mode,
        "provenance_summary": prov_tracker.get_summary(),
        "paths": cfg.paths.model_dump(),
    }


@app.get("/metrics", summary="Summary of forecast verification metrics across models")
def api_get_metrics(
    variable: str = Query("rainfall", description="Target variable"),
    lead_time: int = Query(24, description="Lead time in hours"),
):
    """Returns comparative skill metrics (MAE, RMSE, Bias, Correlation, CRPS)."""
    cfg = get_config()
    res = run_forecast_pipeline(
        mode="synthetic",
        variable=variable,
        lead_time=lead_time,
        config=cfg,
    )
    return {
        "variable": variable,
        "lead_time": lead_time,
        "verification": res.verification or {},
    }


@app.get("/metadata", summary="Comprehensive project and pipeline metadata")
def api_get_metadata():
    """
    Returns complete project metadata, available models, variables, lead times, and regimes.
    """
    cfg = get_config()
    return {
        "project_name": cfg.project_name,
        "project_version": cfg.version,
        "pipeline_version": "0.6.0",
        "available_models": list(cfg.models.keys()),
        "available_variables": list(cfg.variables.keys()),
        "available_lead_times": cfg.lead_times_hours,
        "available_regimes": cfg.weather_regime.thresholds.priority_order,
        "available_blenders": [
            "equal_weight",
            "inverse_variance",
            "inverse_mae",
            "ridge",
            "regime_blender",
            "neural_gating",
            "spatial_inverse_variance",
            "spatial_cnn",
            "spatiotemporal_cnn",
            "gaussian_emos",
            "gamma_emos",
            "zero_inflated_gamma",
        ],
        "regions": list(cfg.regions.keys()),
    }

