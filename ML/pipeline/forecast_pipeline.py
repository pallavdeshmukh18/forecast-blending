"""
Operational Forecast Pipeline Orchestrator for Phase 6.
Executes the unified end-to-end multi-model blending workflow across synthetic,
historical, and real data modes.
"""

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from configs import AppConfig, get_config
from data.provenance import get_provenance_tracker
from ingestion.synthetic import SyntheticDataGenerator
from ingestion.quality_control import apply_quality_control, DataQualityReport
from preprocessing.weather_regimes import extract_atmospheric_features, RuleBasedRegimeClassifier
from models.baseline import equal_weight_blend
from models.skill_weighting import InverseVarianceSkillBlender
from models.regime_blending import RegimeInverseVarianceBlender
from models.probabilistic import ProbabilisticForecastEngine
from models.extreme_calibration import ExtremeProbabilityEngine
from evaluation.metrics import compute_all_metrics
from evaluation.probabilistic_metrics import crps_gaussian
from evaluation.calibration import brier_score


class ForecastResult(BaseModel):
    """
    Standard structured result object returned by the operational forecast pipeline.
    """
    pipeline_version: str = "0.6.0"
    mode: str = "synthetic"
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    target_variable: str
    target_lead_time: int
    forecast_count: int
    forecast: List[Dict[str, Any]]
    weights: Dict[str, Any]
    uncertainty: Dict[str, Any]
    probabilities: List[Dict[str, Any]]
    regime: Dict[str, Any]
    metadata: Dict[str, Any]
    verification: Optional[Dict[str, Any]] = None
    diagnostics: Dict[str, Any]


class ForecastPipelineOrchestrator:
    """
    Single operational pipeline execution engine.
    """

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or get_config()
        self.prob_engine = ProbabilisticForecastEngine(self.config)
        self.extreme_engine = ExtremeProbabilityEngine(self.config)

    def execute(
        self,
        mode: str = "synthetic",
        variable: str = "rainfall",
        lead_time: int = 24,
        region: Optional[str] = None,
        blender_method: str = "regime_blender",  # 'equal_weight', 'inverse_variance', 'regime_blender', 'spatial_cnn'
        forecast_df: Optional[pd.DataFrame] = None,
        observation_df: Optional[pd.DataFrame] = None,
    ) -> ForecastResult:
        """
        Execute full end-to-end pipeline run.
        """
        start_time = time.time()
        diagnostics = {}

        # 1. Load data according to mode
        if forecast_df is None or observation_df is None:
            if mode == "synthetic":
                gen = SyntheticDataGenerator(config=self.config)
                o_df, f_df = gen.generate(save_to_disk=False)
            elif mode in ["historical", "real"]:
                f_path = Path(self.config.data.real_forecasts_path or "data/forecasts/synthetic_forecasts.parquet")
                o_path = Path(self.config.data.real_observations_path or "data/processed/synthetic_observations.parquet")
                if not f_path.exists() or not o_path.exists():
                    gen = SyntheticDataGenerator(config=self.config)
                    o_df, f_df = gen.generate(save_to_disk=False)
                    diagnostics["fallback_to_synthetic"] = True
                else:
                    f_df = pd.read_parquet(f_path) if f_path.suffix == ".parquet" else pd.read_csv(f_path)
                    o_df = pd.read_parquet(o_path) if o_path.suffix == ".parquet" else pd.read_csv(o_path)
            else:
                raise ValueError(f"Unsupported pipeline mode: {mode}")
        else:
            f_df = forecast_df.copy()
            o_df = observation_df.copy()

        # 2. Filter target variable, lead time, and region
        f_sub = f_df[(f_df["variable"] == variable) & (f_df["lead_time"] == lead_time)].copy()
        if f_sub.empty:
            f_sub = f_df[f_df["variable"] == variable].copy()
            if not f_sub.empty:
                lead_time = int(f_sub["lead_time"].iloc[0])
            else:
                f_sub = f_df.copy()
                variable = str(f_sub["variable"].iloc[0])
                lead_time = int(f_sub["lead_time"].iloc[0])

        if region and region in self.config.regions:
            reg_cfg = self.config.regions[region]
            f_sub = f_sub[
                (f_sub["latitude"] >= reg_cfg.lat_min)
                & (f_sub["latitude"] <= reg_cfg.lat_max)
                & (f_sub["longitude"] >= reg_cfg.lon_min)
                & (f_sub["longitude"] <= reg_cfg.lon_max)
            ]

        # 3. Quality Control
        valid_f, rej_f, qc_report = apply_quality_control(
            f_sub,
            value_col="forecast_value",
            variable_col="variable",
            config=self.config,
            save_report_path=self.config.paths.quality_report_file,
        )
        diagnostics["qc_valid_records"] = qc_report.valid_records
        diagnostics["qc_rejected_records"] = len(rej_f)

        # 4. Spatiotemporal Alignment & Feature Generation
        feat_df = extract_atmospheric_features(valid_f)
        regime_classifier = RuleBasedRegimeClassifier(config=self.config)
        regime_df = regime_classifier.classify(feat_df)

        # 5. Multi-Model Forecast Blending
        # Determine available models in slice
        models_present = valid_f["model_name"].unique().tolist()
        weights_dict = {}

        if blender_method == "equal_weight" or len(models_present) <= 1:
            blended_df = equal_weight_blend(regime_df)
            weights_dict = {m: round(1.0 / len(models_present), 4) for m in models_present}
        elif blender_method == "inverse_variance":
            blender = InverseVarianceSkillBlender(config=self.config)
            # Fit on available slice or fallback
            blender.fit(regime_df)
            blended_df = blender.predict(regime_df)
            weights_dict = blender.get_weights(variable=variable, lead_time=lead_time)
        else:  # default 'regime_blender'
            blender = RegimeInverseVarianceBlender(config=self.config)
            blender.fit(regime_df)
            blended_df = blender.predict(regime_df)
            weights_dict = blender.get_weights(variable=variable, lead_time=lead_time)

        # 6. Probabilistic & Extreme Post-Processing
        # Ensure blend_value and spread are present
        if "blend_value" not in blended_df.columns:
            if "forecast_value" in blended_df.columns:
                blended_df["blend_value"] = blended_df["forecast_value"]
            else:
                blended_df["blend_value"] = 0.0

        if "timestamp" in blended_df.columns:
            blended_df["timestamp"] = pd.to_datetime(blended_df["timestamp"], utc=True)
        if "timestamp" in regime_df.columns:
            regime_df["timestamp"] = pd.to_datetime(regime_df["timestamp"], utc=True)

        if "spread" not in blended_df.columns:
            # Estimate spread from model disagreement if multiple models exist
            if len(models_present) > 1:
                spread_series = regime_df.groupby(["timestamp", "latitude", "longitude"])["forecast_value"].std().fillna(1.0)
                blended_df = blended_df.merge(spread_series.rename("spread"), on=["timestamp", "latitude", "longitude"], how="left")
                blended_df["spread"] = blended_df["spread"].fillna(1.0)
            else:
                blended_df["spread"] = 1.0

        # Calibrate extreme probabilities
        if o_df is not None and not o_df.empty:
            o_clean = o_df.copy()
            if "timestamp" in o_clean.columns:
                o_clean["timestamp"] = pd.to_datetime(o_clean["timestamp"], utc=True)
            self.extreme_engine.calibrate_on_history(
                pd.concat([blended_df, o_clean], axis=0, ignore_index=True) if "observed_value" in o_clean.columns else blended_df
            )

        extreme_probs_df = self.extreme_engine.compute_extreme_probabilities(
            blended_df, blend_col="blend_value", spread_col="spread"
        )

        # 7. Verification against Observations if available
        verification_metrics = None
        if o_df is not None and not o_df.empty:
            o_sub = o_df[(o_df["variable"] == variable)].copy()
            if not o_sub.empty:
                if "timestamp" in o_sub.columns:
                    o_sub["timestamp"] = pd.to_datetime(o_sub["timestamp"], utc=True)
                # Merge on timestamp and coordinates
                matched = pd.merge(
                    blended_df,
                    o_sub[["timestamp", "latitude", "longitude", "observed_value"]],
                    on=["timestamp", "latitude", "longitude"],
                    how="inner",
                )
                if not matched.empty:
                    obs_vals = matched["observed_value"].values
                    pred_vals = matched["blend_value"].values
                    det_metrics = compute_all_metrics(obs_vals, pred_vals)

                    # Probabilistic metrics
                    spread_vals = matched["spread"].values
                    mean_crps = float(np.mean(crps_gaussian(obs_vals, pred_vals, spread_vals)))

                    verification_metrics = {
                        "matched_sample_count": len(matched),
                        "crps": round(mean_crps, 4),
                        **det_metrics,
                    }

        # 8. Assemble structured result
        exec_time_ms = round((time.time() - start_time) * 1000.0, 2)
        diagnostics["execution_time_ms"] = exec_time_ms
        diagnostics["blender_method"] = blender_method
        diagnostics["models_incorporated"] = models_present

        # Regime summary
        regime_counts = regime_df["regime"].value_counts().to_dict() if "regime" in regime_df.columns else {"stable": len(blended_df)}
        dominant_regime = max(regime_counts, key=regime_counts.get) if regime_counts else "stable"

        # Uncertainty summary
        spread_arr = blended_df["spread"].values
        uncertainty_summary = {
            "mean_spread": round(float(np.mean(spread_arr)), 4),
            "min_spread": round(float(np.min(spread_arr)), 4),
            "max_spread": round(float(np.max(spread_arr)), 4),
            "q10_spread": round(float(np.percentile(spread_arr, 10)), 4),
            "q90_spread": round(float(np.percentile(spread_arr, 90)), 4),
        }

        # Format output sample records
        sample_records = blended_df.head(100)[
            [c for c in ["timestamp", "latitude", "longitude", "lead_time", "variable", "blend_value", "spread", "regime"] if c in blended_df.columns]
        ].to_dict(orient="records")

        # Format sample probabilities
        prob_sample = extreme_probs_df.head(100).to_dict(orient="records")

        return ForecastResult(
            pipeline_version="0.6.0",
            mode=mode,
            target_variable=variable,
            target_lead_time=lead_time,
            forecast_count=len(blended_df),
            forecast=sample_records,
            weights=weights_dict,
            uncertainty=uncertainty_summary,
            probabilities=prob_sample,
            regime={
                "dominant_regime": dominant_regime,
                "regime_distribution": regime_counts,
            },
            metadata={
                "project_name": self.config.project_name,
                "models_available": list(self.config.models.keys()),
                "variables_available": list(self.config.variables.keys()),
                "lead_times_available": self.config.lead_times_hours,
                "region": region or "all",
            },
            verification=verification_metrics,
            diagnostics=diagnostics,
        )


def run_forecast_pipeline(
    mode: str = "synthetic",
    variable: str = "rainfall",
    lead_time: int = 24,
    region: Optional[str] = None,
    blender_method: str = "regime_blender",
    config: Optional[AppConfig] = None,
    forecast_df: Optional[pd.DataFrame] = None,
    observation_df: Optional[pd.DataFrame] = None,
) -> ForecastResult:
    """
    Convenience functional entrypoint for executing the operational forecast pipeline.
    """
    orchestrator = ForecastPipelineOrchestrator(config=config)
    return orchestrator.execute(
        mode=mode,
        variable=variable,
        lead_time=lead_time,
        region=region,
        blender_method=blender_method,
        forecast_df=forecast_df,
        observation_df=observation_df,
    )
