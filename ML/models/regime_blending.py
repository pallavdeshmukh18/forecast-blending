"""
Weather-Regime Conditioned Skill-Weighted Multi-Model Forecast Blenders.
Extends skill-based weighting to condition dynamically on atmospheric/weather regimes
(weather_regime x variable x lead_time x region) with hierarchical sparse-data fallback.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from configs import AppConfig, get_config
from ingestion.schema import validate_forecast_dataframe
from preprocessing.matching import match_forecasts_to_observations
from preprocessing.weather_regimes import classify_rule_based_regime
from models.base import BaseBlender
from models.skill_weighting import SkillWeightedBlender


class RegimeSkillBlender(SkillWeightedBlender):
    """
    Skill-weighted blender dynamically conditioned on synoptic weather regimes,
    variables, lead times, and geographic regions.
    """

    def __init__(
        self,
        method: str = "inverse_variance",  # 'inverse_variance' or 'inverse_mae'
        epsilon: float = 1e-6,
        weight_smoothing: float = 0.10,
        minimum_samples: int = 20,
        condition_on_regime: bool = True,
        condition_on_lead_time: bool = True,
        condition_on_region: bool = True,
        name: Optional[str] = None,
        config: Optional[AppConfig] = None,
    ):
        blender_name = name or f"regime_{method}_blend"
        super().__init__(
            method=method,
            epsilon=epsilon,
            weight_smoothing=weight_smoothing,
            minimum_samples=minimum_samples,
            condition_on_lead_time=condition_on_lead_time,
            condition_on_region=condition_on_region,
            name=blender_name,
            config=config,
        )
        self.condition_on_regime = condition_on_regime

        # Scope 5: (model, variable, lead_time, region, weather_regime)
        self.skill_scope_5: Dict[Tuple[str, str, int, str, str], Dict[str, float]] = {}
        # Scope 4: (model, variable, lead_time, weather_regime)
        self.skill_scope_4_regime: Dict[Tuple[str, str, int, str], Dict[str, float]] = {}

    def fit_verification_data(self, verified_df: pd.DataFrame) -> "RegimeSkillBlender":
        """
        Fit historical skill statistics across regime-conditioned hierarchical scopes.
        """
        if verified_df.empty:
            self.is_fitted = True
            return self

        df = verified_df.copy()

        # Classify weather regimes if not already populated
        if "weather_regime" not in df.columns or df["weather_regime"].isna().all():
            df = classify_rule_based_regime(df, config=self.config)

        # Fit Base Scopes (1, 2, 3, 4 without regime)
        super().fit_verification_data(df)

        if "error" not in df.columns:
            if "forecast_error" in df.columns:
                df["error"] = df["forecast_error"]
            elif "forecast_value" in df.columns and "observed_value" in df.columns:
                df["error"] = df["forecast_value"] - df["observed_value"]
            else:
                self.is_fitted = True
                return self

        # Scope 4 Regime: (model, variable, lead_time, weather_regime)
        if "lead_time" in df.columns and "weather_regime" in df.columns:
            for (m_name, var, lt, regm), grp in df.groupby(
                ["model_name", "variable", "lead_time", "weather_regime"], observed=True
            ):
                errs = grp["error"].dropna().values
                self.skill_scope_4_regime[(m_name, var, int(lt), str(regm))] = {
                    "skill_metric": self._compute_model_skill(errs),
                    "sample_count": len(errs),
                    "var_error": float(np.var(errs)) if len(errs) > 1 else float(np.mean(errs**2)),
                }

        # Scope 5: (model, variable, lead_time, region, weather_regime)
        if "lead_time" in df.columns and "region" in df.columns and "weather_regime" in df.columns:
            for (m_name, var, lt, reg, regm), grp in df.groupby(
                ["model_name", "variable", "lead_time", "region", "weather_regime"], observed=True
            ):
                errs = grp["error"].dropna().values
                self.skill_scope_5[(m_name, var, int(lt), str(reg), str(regm))] = {
                    "skill_metric": self._compute_model_skill(errs),
                    "sample_count": len(errs),
                    "var_error": float(np.var(errs)) if len(errs) > 1 else float(np.mean(errs**2)),
                }

        self.is_fitted = True
        return self

    def _get_model_skill_and_scope(
        self,
        model_name: str,
        variable: str,
        lead_time: Optional[int] = None,
        region: Optional[str] = None,
        weather_regime: Optional[str] = None,
    ) -> Tuple[float, int, str, float]:
        """
        Retrieve model skill with full 6-level hierarchical fallback:
        1. model + var + lead + region + regime (Scope 5)
        2. model + var + lead + regime (Scope 4 Regime)
        3. model + var + lead + region (Scope 4 Region)
        4. model + var + lead (Scope 3)
        5. model + var (Scope 2)
        6. model global (Scope 1)
        """
        # Scope 5: (model, variable, lead, region, regime)
        if self.condition_on_regime and weather_regime and self.condition_on_region and region and lead_time is not None:
            key5 = (model_name, variable, int(lead_time), str(region), str(weather_regime))
            if key5 in self.skill_scope_5 and self.skill_scope_5[key5]["sample_count"] >= self.minimum_samples:
                res = self.skill_scope_5[key5]
                return res["skill_metric"], res["sample_count"], "model_variable_lead_region_regime", res["var_error"]

        # Scope 4 Regime: (model, variable, lead, regime)
        if self.condition_on_regime and weather_regime and lead_time is not None:
            key4_r = (model_name, variable, int(lead_time), str(weather_regime))
            if key4_r in self.skill_scope_4_regime and self.skill_scope_4_regime[key4_r]["sample_count"] >= self.minimum_samples:
                res = self.skill_scope_4_regime[key4_r]
                return res["skill_metric"], res["sample_count"], "model_variable_lead_regime", res["var_error"]

        # Base fallbacks: Scope 4 Region, Scope 3 Lead, Scope 2 Var, Scope 1 Global
        return super()._get_model_skill_and_scope(model_name, variable, lead_time, region)

    def get_weights(
        self,
        variable: str,
        lead_time: Optional[int] = None,
        region: Optional[str] = None,
        weather_regime: Optional[str] = None,
        available_models: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Calculate normalized and smoothed model weights for a specific regime context.
        """
        models = available_models or self.known_models
        if not models:
            return {}

        n_models = len(models)
        if n_models == 1:
            return {models[0]: 1.0}

        inv_skills = []
        for m in models:
            skill, _, _, _ = self._get_model_skill_and_scope(m, variable, lead_time, region, weather_regime)
            inv_skills.append(1.0 / (skill + self.epsilon))

        sum_inv = sum(inv_skills)
        if sum_inv <= 0 or np.isnan(sum_inv):
            raw_weights = [1.0 / n_models] * n_models
        else:
            raw_weights = [s / sum_inv for s in inv_skills]

        # Apply weight smoothing: w' = (1 - alpha)*w + alpha*(1/N)
        alpha = np.clip(self.weight_smoothing, 0.0, 1.0)
        smoothed_weights = [(1.0 - alpha) * w + alpha * (1.0 / n_models) for w in raw_weights]

        total_smoothed = sum(smoothed_weights)
        norm_weights = [w / total_smoothed for w in smoothed_weights]

        return dict(zip(models, norm_weights))

    def get_weight_metadata(
        self,
        variable: str,
        lead_time: Optional[int] = None,
        region: Optional[str] = None,
        weather_regime: Optional[str] = None,
        available_models: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve diagnostic metadata distinguishing active regime scope and historical samples.
        """
        models = available_models or self.known_models
        weights = self.get_weights(
            variable=variable,
            lead_time=lead_time,
            region=region,
            weather_regime=weather_regime,
            available_models=models,
        )

        model_details = {}
        for m in models:
            skill, count, scope, var_err = self._get_model_skill_and_scope(
                m, variable, lead_time, region, weather_regime
            )
            model_details[m] = {
                "weight": round(weights.get(m, 0.0), 4),
                "skill_metric_value": round(skill, 4),
                "historical_sample_count": count,
                "active_skill_scope": scope,
            }

        return {
            "blender": self.name,
            "method": self.method,
            "variable": variable,
            "lead_time": lead_time,
            "region": region,
            "weather_regime": weather_regime,
            "weight_smoothing_alpha": self.weight_smoothing,
            "models": model_details,
        }

    def predict(self, forecasts_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate regime-conditioned skill-weighted blended forecasts with diagnostic uncertainty metrics.
        """
        df = validate_forecast_dataframe(forecasts_df, allow_nans=True)

        if "weather_regime" not in df.columns or df["weather_regime"].isna().all():
            df = classify_rule_based_regime(df, config=self.config)

        group_cols = ["timestamp", "latitude", "longitude", "lead_time", "variable"]
        meta_cols = ["run_time", "region", "season", "weather_regime"]
        active_meta = [c for c in meta_cols if c in df.columns]
        all_group_cols = group_cols + active_meta

        records = []
        for keys, grp in df.groupby(all_group_cols, dropna=False, observed=True):
            if not isinstance(keys, tuple):
                keys = (keys,)
            key_dict = dict(zip(all_group_cols, keys))

            valid_grp = grp[grp["forecast_value"].notna()].copy()
            if valid_grp.empty:
                continue

            avail_models = list(valid_grp["model_name"].values)
            avail_values = valid_grp["forecast_value"].values

            # Retrieve regime-conditioned weights
            weights_dict = self.get_weights(
                variable=key_dict["variable"],
                lead_time=key_dict["lead_time"],
                region=key_dict.get("region"),
                weather_regime=key_dict.get("weather_regime"),
                available_models=avail_models,
            )

            w_vec = np.array([weights_dict.get(m, 0.0) for m in avail_models])
            if w_vec.sum() > 0:
                w_vec = w_vec / w_vec.sum()
            else:
                w_vec = np.ones(len(avail_models)) / len(avail_models)

            blended_val = float(np.sum(w_vec * avail_values))

            # Diagnostic spread
            spread = float(np.sqrt(np.sum(w_vec * (avail_values - blended_val) ** 2))) if len(avail_values) > 1 else 0.0

            # Residual std from historical skill
            res_stds = []
            for m in avail_models:
                _, _, _, v_err = self._get_model_skill_and_scope(
                    m,
                    key_dict["variable"],
                    key_dict["lead_time"],
                    key_dict.get("region"),
                    key_dict.get("weather_regime"),
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


class RegimeInverseVarianceBlender(RegimeSkillBlender):
    """
    Convenience wrapper for Regime-Conditioned Inverse-Variance Blending.
    """

    def __init__(
        self,
        epsilon: float = 1e-6,
        weight_smoothing: float = 0.10,
        minimum_samples: int = 20,
        name: str = "regime_inverse_variance_blend",
        config: Optional[AppConfig] = None,
    ):
        super().__init__(
            method="inverse_variance",
            epsilon=epsilon,
            weight_smoothing=weight_smoothing,
            minimum_samples=minimum_samples,
            name=name,
            config=config,
        )


class RegimeInverseMAEBlender(RegimeSkillBlender):
    """
    Convenience wrapper for Regime-Conditioned Inverse-MAE Blending.
    """

    def __init__(
        self,
        epsilon: float = 1e-6,
        weight_smoothing: float = 0.10,
        minimum_samples: int = 20,
        name: str = "regime_inverse_mae_blend",
        config: Optional[AppConfig] = None,
    ):
        super().__init__(
            method="inverse_mae",
            epsilon=epsilon,
            weight_smoothing=weight_smoothing,
            minimum_samples=minimum_samples,
            name=name,
            config=config,
        )


__all__ = [
    "RegimeSkillBlender",
    "RegimeInverseVarianceBlender",
    "RegimeInverseMAEBlender",
]
