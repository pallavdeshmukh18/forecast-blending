"""
Skill-based Multi-Model Forecast Blenders: Inverse-Variance and Inverse-MAE Weighting.

Implements interpretable, hierarchical, lead-time and region-conditioned skill estimation
with weight smoothing and missing-model renormalization.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from configs import AppConfig, get_config
from ingestion.schema import validate_forecast_dataframe
from preprocessing.matching import match_forecasts_to_observations
from models.base import BaseBlender


class SkillWeightedBlender(BaseBlender):
    """
    Skill-weighted forecast blender dynamically weighting member models based on
    historical forecast accuracy (Inverse-Variance or Inverse-MAE).
    """

    def __init__(
        self,
        method: str = "inverse_variance",  # 'inverse_variance' or 'inverse_mae'
        epsilon: float = 1e-6,
        weight_smoothing: float = 0.10,
        minimum_samples: int = 20,
        condition_on_lead_time: bool = True,
        condition_on_region: bool = True,
        name: Optional[str] = None,
        config: Optional[AppConfig] = None,
    ):
        blender_name = name or f"{method}_blend"
        super().__init__(name=blender_name)
        self.method = method.lower()
        self.epsilon = epsilon
        self.weight_smoothing = weight_smoothing
        self.minimum_samples = minimum_samples
        self.condition_on_lead_time = condition_on_lead_time
        self.condition_on_region = condition_on_region
        self.config = config or get_config()

        # Skill storage tables by hierarchical scope
        self.skill_scope_4: Dict[Tuple[str, str, int, str], Dict[str, float]] = {}  # (model, var, lead, region)
        self.skill_scope_3: Dict[Tuple[str, str, int], Dict[str, float]] = {}       # (model, var, lead)
        self.skill_scope_2: Dict[Tuple[str, str], Dict[str, float]] = {}            # (model, var)
        self.skill_scope_1: Dict[str, Dict[str, float]] = {}                        # (model,)
        self.known_models: List[str] = []

    def _compute_model_skill(self, errors: np.ndarray) -> float:
        """Compute scalar skill metric (Mean Squared Error / Variance or MAE) from forecast errors."""
        if len(errors) == 0:
            return 1.0

        if self.method in ["inverse_variance", "variance", "var", "mse"]:
            return float(np.mean(errors**2))
        elif self.method in ["inverse_mae", "mae"]:
            return float(np.mean(np.abs(errors)))
        else:
            return float(np.mean(errors**2))

    def fit_verification_data(self, verified_df: pd.DataFrame) -> "SkillWeightedBlender":
        """
        Compute historical skill statistics across hierarchical scopes from verified records.
        """
        if verified_df.empty:
            self.is_fitted = True
            return self

        df = verified_df.copy()
        if "error" not in df.columns:
            if "forecast_error" in df.columns:
                df["error"] = df["forecast_error"]
            elif "forecast_value" in df.columns and "observed_value" in df.columns:
                df["error"] = df["forecast_value"] - df["observed_value"]
            else:
                self.known_models = sorted(list(df["model_name"].dropna().unique())) if "model_name" in df.columns else []
                self.is_fitted = True
                return self

        self.known_models = sorted(list(df["model_name"].dropna().unique()))

        # Scope 1: Global per model
        for m_name, grp in df.groupby("model_name", observed=True):
            errs = grp["error"].dropna().values
            self.skill_scope_1[m_name] = {
                "skill_metric": self._compute_model_skill(errs),
                "sample_count": len(errs),
                "var_error": float(np.var(errs)) if len(errs) > 1 else 1.0,
            }

        # Scope 2: (model, variable)
        for (m_name, var), grp in df.groupby(["model_name", "variable"], observed=True):
            errs = grp["error"].dropna().values
            self.skill_scope_2[(m_name, var)] = {
                "skill_metric": self._compute_model_skill(errs),
                "sample_count": len(errs),
                "var_error": float(np.var(errs)) if len(errs) > 1 else 1.0,
            }

        # Scope 3: (model, variable, lead_time)
        if "lead_time" in df.columns:
            for (m_name, var, lt), grp in df.groupby(["model_name", "variable", "lead_time"], observed=True):
                errs = grp["error"].dropna().values
                self.skill_scope_3[(m_name, var, int(lt))] = {
                    "skill_metric": self._compute_model_skill(errs),
                    "sample_count": len(errs),
                    "var_error": float(np.var(errs)) if len(errs) > 1 else 1.0,
                }

        # Scope 4: (model, variable, lead_time, region)
        if "lead_time" in df.columns and "region" in df.columns:
            for (m_name, var, lt, reg), grp in df.groupby(["model_name", "variable", "lead_time", "region"], observed=True):
                errs = grp["error"].dropna().values
                self.skill_scope_4[(m_name, var, int(lt), str(reg))] = {
                    "skill_metric": self._compute_model_skill(errs),
                    "sample_count": len(errs),
                    "var_error": float(np.var(errs)) if len(errs) > 1 else 1.0,
                }

        self.is_fitted = True
        return self

    def fit(self, historical_forecasts: pd.DataFrame, historical_observations: Optional[pd.DataFrame] = None) -> "SkillWeightedBlender":
        """
        Align historical forecasts and observations, then fit skill weighting parameters.
        If historical_observations is None and historical_forecasts contains observed_value,
        fits directly from verification DataFrame.
        """
        if historical_observations is None:
            return self.fit_verification_data(historical_forecasts)

        verified = match_forecasts_to_observations(
            forecasts_df=historical_forecasts,
            observations_df=historical_observations,
            config=self.config,
        )
        return self.fit_verification_data(verified)

    def _get_model_skill_and_scope(
        self,
        model_name: str,
        variable: str,
        lead_time: Optional[int] = None,
        region: Optional[str] = None,
    ) -> Tuple[float, int, str, float]:
        """
        Retrieve model skill with hierarchical fallback for sparse data.
        Returns: (skill_metric, sample_count, skill_scope_name, var_error)
        """
        # Scope 4: (model, variable, lead_time, region)
        if self.condition_on_region and region and lead_time is not None:
            key4 = (model_name, variable, int(lead_time), str(region))
            if key4 in self.skill_scope_4 and self.skill_scope_4[key4]["sample_count"] >= self.minimum_samples:
                res = self.skill_scope_4[key4]
                return res["skill_metric"], res["sample_count"], "model_variable_lead_region", res["var_error"]

        # Scope 3: (model, variable, lead_time)
        if self.condition_on_lead_time and lead_time is not None:
            key3 = (model_name, variable, int(lead_time))
            if key3 in self.skill_scope_3 and self.skill_scope_3[key3]["sample_count"] >= self.minimum_samples:
                res = self.skill_scope_3[key3]
                return res["skill_metric"], res["sample_count"], "model_variable_lead", res["var_error"]

        # Scope 2: (model, variable)
        key2 = (model_name, variable)
        if key2 in self.skill_scope_2 and self.skill_scope_2[key2]["sample_count"] >= self.minimum_samples:
            res = self.skill_scope_2[key2]
            return res["skill_metric"], res["sample_count"], "model_variable", res["var_error"]

        # Scope 1: model global
        if model_name in self.skill_scope_1 and self.skill_scope_1[model_name]["sample_count"] >= 1:
            res = self.skill_scope_1[model_name]
            return res["skill_metric"], res["sample_count"], "model_global", res["var_error"]

        # Fallback if entirely unseen
        return 1.0, 0, "fallback_unseen", 1.0

    def get_weights(
        self,
        variable: str,
        lead_time: Optional[int] = None,
        region: Optional[str] = None,
        available_models: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Calculate normalized and smoothed model weights for a specific context.
        """
        models = available_models or self.known_models
        if not models:
            return {}

        n_models = len(models)
        if n_models == 1:
            return {models[0]: 1.0}

        inv_skills = []
        for m in models:
            skill, _, _, _ = self._get_model_skill_and_scope(m, variable, lead_time, region)
            inv_skills.append(1.0 / (skill + self.epsilon))

        sum_inv = sum(inv_skills)
        if sum_inv <= 0 or np.isnan(sum_inv):
            raw_weights = [1.0 / n_models] * n_models
        else:
            raw_weights = [s / sum_inv for s in inv_skills]

        # Apply weight smoothing: w' = (1 - alpha)*w + alpha*(1/N)
        alpha = np.clip(self.weight_smoothing, 0.0, 1.0)
        smoothed_weights = [(1.0 - alpha) * w + alpha * (1.0 / n_models) for w in raw_weights]

        # Ensure exact sum to 1.0
        total_smoothed = sum(smoothed_weights)
        norm_weights = [w / total_smoothed for w in smoothed_weights]

        return dict(zip(models, norm_weights))

    def get_weight_metadata(
        self,
        variable: str,
        lead_time: Optional[int] = None,
        region: Optional[str] = None,
        available_models: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve diagnostic weight details including sample counts and active skill scopes.
        """
        models = available_models or self.known_models
        weights = self.get_weights(variable, lead_time, region, available_models=models)

        model_details = {}
        for m in models:
            skill, count, scope, var_err = self._get_model_skill_and_scope(m, variable, lead_time, region)
            model_details[m] = {
                "weight": weights.get(m, 0.0),
                "skill_metric_value": skill,
                "historical_sample_count": count,
                "active_skill_scope": scope,
            }

        return {
            "blender": self.name,
            "method": self.method,
            "variable": variable,
            "lead_time": lead_time,
            "region": region,
            "weight_smoothing_alpha": self.weight_smoothing,
            "models": model_details,
        }

    def predict(self, forecasts_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate skill-weighted blended forecasts with diagnostic uncertainty metrics.
        """
        df = validate_forecast_dataframe(forecasts_df, allow_nans=True)

        group_cols = ["timestamp", "latitude", "longitude", "lead_time", "variable"]
        meta_cols = ["run_time", "region", "season", "weather_regime"]
        active_meta = [c for c in meta_cols if c in df.columns]
        all_group_cols = group_cols + active_meta

        records = []
        for keys, grp in df.groupby(all_group_cols, dropna=False, observed=True):
            if not isinstance(keys, tuple):
                keys = (keys,)
            key_dict = dict(zip(all_group_cols, keys))

            # Filter non-null forecasts
            valid_grp = grp[grp["forecast_value"].notna()].copy()
            if valid_grp.empty:
                continue

            avail_models = list(valid_grp["model_name"].values)
            avail_values = valid_grp["forecast_value"].values

            # Get normalized weights
            weights_dict = self.get_weights(
                variable=key_dict["variable"],
                lead_time=key_dict["lead_time"],
                region=key_dict.get("region"),
                available_models=avail_models,
            )

            w_vec = np.array([weights_dict.get(m, 0.0) for m in avail_models])
            if w_vec.sum() > 0:
                w_vec = w_vec / w_vec.sum()
            else:
                w_vec = np.ones(len(avail_models)) / len(avail_models)

            # Blended forecast calculation
            blended_val = float(np.sum(w_vec * avail_values))

            # Diagnostic uncertainty spread
            if len(avail_values) > 1:
                spread = float(np.sqrt(np.sum(w_vec * (avail_values - blended_val) ** 2)))
            else:
                spread = 0.0

            # Residual std from historical skill
            res_stds = []
            for m in avail_models:
                _, _, _, v_err = self._get_model_skill_and_scope(
                    m, key_dict["variable"], key_dict["lead_time"], key_dict.get("region")
                )
                res_stds.append(np.sqrt(max(0.0, v_err)))
            hist_res_std = float(np.sum(w_vec * np.array(res_stds)))

            rec = {
                **key_dict,
                "model_name": self.name,
                "forecast_value": round(blended_val, 4),
                "forecast_spread": round(spread, 4),
                "historical_residual_std": round(hist_res_std, 4),
            }
            records.append(rec)

        if not records:
            return pd.DataFrame()

        out_df = pd.DataFrame(records)
        return validate_forecast_dataframe(out_df, allow_nans=False)


class InverseVarianceSkillBlender(SkillWeightedBlender):
    """
    Convenience wrapper for Inverse-Variance skill weighting.
    """

    def __init__(
        self,
        epsilon: float = 1e-6,
        weight_smoothing: float = 0.10,
        minimum_samples: int = 20,
        condition_on_lead_time: bool = True,
        condition_on_region: bool = True,
        name: str = "inverse_variance_blend",
        config: Optional[AppConfig] = None,
    ):
        super().__init__(
            method="inverse_variance",
            epsilon=epsilon,
            weight_smoothing=weight_smoothing,
            minimum_samples=minimum_samples,
            condition_on_lead_time=condition_on_lead_time,
            condition_on_region=condition_on_region,
            name=name,
            config=config,
        )


class InverseMAESkillBlender(SkillWeightedBlender):
    """
    Convenience wrapper for Inverse-MAE skill weighting.
    """

    def __init__(
        self,
        epsilon: float = 1e-6,
        weight_smoothing: float = 0.10,
        minimum_samples: int = 20,
        condition_on_lead_time: bool = True,
        condition_on_region: bool = True,
        name: str = "inverse_mae_blend",
        config: Optional[AppConfig] = None,
    ):
        super().__init__(
            method="inverse_mae",
            epsilon=epsilon,
            weight_smoothing=weight_smoothing,
            minimum_samples=minimum_samples,
            condition_on_lead_time=condition_on_lead_time,
            condition_on_region=condition_on_region,
            name=name,
            config=config,
        )


__all__ = [
    "SkillWeightedBlender",
    "InverseVarianceSkillBlender",
    "InverseMAESkillBlender",
]
