"""
Adaptive Regression Multi-Model Blending using Standard and Constrained Ridge Regression.

Implements regularized linear blending:
    y = beta_0 + sum(beta_i * F_i)
with support for both standard Ridge and strictly constrained non-negative normalized blending (w_i >= 0, sum(w_i) = 1).
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge

from configs import AppConfig, get_config
from ingestion.schema import validate_forecast_dataframe
from preprocessing.matching import match_forecasts_to_observations
from preprocessing.alignment import pivot_models_wide
from models.base import BaseBlender


class RidgeBlender(BaseBlender):
    """
    Adaptive multi-model blender utilizing regularized Ridge regression.
    Supports standard unconstrained Ridge and constrained non-negative simplex blending.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        constrained_weights: bool = True,
        minimum_samples: int = 20,
        condition_on_lead_time: bool = True,
        condition_on_region: bool = True,
        name: str = "ridge_blend",
        config: Optional[AppConfig] = None,
    ):
        super().__init__(name=name)
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.constrained_weights = constrained_weights
        self.minimum_samples = minimum_samples
        self.condition_on_lead_time = condition_on_lead_time
        self.condition_on_region = condition_on_region
        self.config = config or get_config()

        # Context-dependent fitted parameters
        # Key: (variable, lead_time, region) or (variable, lead_time) or (variable,)
        self.models_by_context: Dict[Tuple, Dict[str, Any]] = {}
        self.known_models: List[str] = []
        self.global_intercept: float = 0.0
        self.global_weights: Dict[str, float] = {}

    def _fit_linear_slice(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_names: List[str],
    ) -> Dict[str, Any]:
        """
        Fit Ridge parameters for a feature matrix X and target vector y.
        """
        n_samples, n_features = X.shape
        if n_samples < 2 or n_features == 0:
            equal_w = 1.0 / max(1, n_features)
            return {
                "intercept": 0.0,
                "weights": {m: equal_w for m in model_names},
                "sample_count": n_samples,
                "residual_std": 1.0,
            }

        if not self.constrained_weights:
            # Mode A: Standard unconstrained Ridge
            reg = Ridge(alpha=self.alpha, fit_intercept=self.fit_intercept)
            reg.fit(X, y)
            coefs = reg.coef_
            intercept = float(reg.intercept_) if self.fit_intercept else 0.0
            residuals = y - reg.predict(X)
            res_std = float(np.std(residuals)) if len(residuals) > 1 else 1.0

            weights_dict = {m: float(c) for m, c in zip(model_names, coefs)}
            return {
                "intercept": intercept,
                "weights": weights_dict,
                "sample_count": n_samples,
                "residual_std": res_std,
            }
        else:
            # Mode B: Constrained non-negative normalized blending (w >= 0, sum(w) = 1)
            # Objective: 0.5 * mean((y - (beta_0 + X @ w))^2) + 0.5 * alpha * sum(w^2)
            if self.fit_intercept:
                # Parameters: [beta_0, w_1, ..., w_K]
                def objective(params):
                    b0 = params[0]
                    w = params[1:]
                    pred = b0 + X @ w
                    loss = 0.5 * np.mean((y - pred) ** 2) + 0.5 * self.alpha * np.sum(w ** 2)
                    return loss

                x0 = np.zeros(n_features + 1)
                x0[0] = float(np.mean(y) - np.mean(X))
                x0[1:] = 1.0 / n_features

                bounds = [(None, None)] + [(0.0, 1.0) for _ in range(n_features)]
                constraints = [{"type": "eq", "fun": lambda p: np.sum(p[1:]) - 1.0}]

                res = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
                if res.success:
                    intercept = float(res.x[0])
                    weights = res.x[1:]
                else:
                    intercept = float(np.mean(y) - np.mean(X))
                    weights = np.ones(n_features) / n_features
            else:
                # Parameters: [w_1, ..., w_K]
                def objective(w):
                    pred = X @ w
                    loss = 0.5 * np.mean((y - pred) ** 2) + 0.5 * self.alpha * np.sum(w ** 2)
                    return loss

                x0 = np.ones(n_features) / n_features
                bounds = [(0.0, 1.0) for _ in range(n_features)]
                constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

                res = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
                intercept = 0.0
                weights = res.x if res.success else np.ones(n_features) / n_features

            # Ensure strict sum to 1.0 and non-negativity
            weights = np.clip(weights, 0.0, 1.0)
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
            else:
                weights = np.ones(n_features) / n_features

            pred_y = intercept + X @ weights
            res_std = float(np.std(y - pred_y)) if len(y) > 1 else 1.0

            weights_dict = {m: float(w) for m, w in zip(model_names, weights)}
            return {
                "intercept": intercept,
                "weights": weights_dict,
                "sample_count": n_samples,
                "residual_std": res_std,
            }

    def fit_verification_data(self, verified_df: pd.DataFrame) -> "RidgeBlender":
        """
        Fit Ridge regression models on wide-format multi-model verification data.
        """
        if verified_df.empty:
            self.is_fitted = True
            return self

        df = verified_df.copy()
        self.known_models = sorted(list(df["model_name"].dropna().unique()))

        if len(self.known_models) < 1:
            self.is_fitted = True
            return self

        # Convert long verified pairs to wide format for regression
        index_keys = ["timestamp", "latitude", "longitude", "lead_time", "variable"]
        meta_keys = [c for c in ["region", "observed_value"] if c in df.columns]
        wide_index = index_keys + [c for c in meta_keys if c not in index_keys]

        wide_df = df.pivot_table(
            index=[c for c in wide_index if c in df.columns],
            columns="model_name",
            values="forecast_value",
            aggfunc="first",
        ).reset_index()

        wide_df.columns.name = None

        # Filter complete cases across known models
        model_cols = [m for m in self.known_models if m in wide_df.columns]
        complete_wide = wide_df.dropna(subset=model_cols + ["observed_value"]).reset_index(drop=True)

        if complete_wide.empty:
            # Fallback to equal weights
            self.global_weights = {m: 1.0 / len(self.known_models) for m in self.known_models}
            self.is_fitted = True
            return self

        # 1. Global Model Fit
        X_glob = complete_wide[model_cols].values
        y_glob = complete_wide["observed_value"].values
        glob_fit = self._fit_linear_slice(X_glob, y_glob, model_cols)
        self.global_intercept = glob_fit["intercept"]
        self.global_weights = glob_fit["weights"]

        # 2. Contextual Fits: (variable, lead_time, region) and (variable, lead_time) and (variable,)
        group_slices = [
            ("var_lead_reg", ["variable", "lead_time", "region"]),
            ("var_lead", ["variable", "lead_time"]),
            ("var", ["variable"]),
        ]

        for slice_name, cols in group_slices:
            active_cols = [c for c in cols if c in complete_wide.columns]
            if not active_cols:
                continue

            for keys, grp in complete_wide.groupby(active_cols, observed=True):
                if not isinstance(keys, tuple):
                    keys = (keys,)
                if len(grp) >= self.minimum_samples:
                    X_grp = grp[model_cols].values
                    y_grp = grp["observed_value"].values
                    fit_res = self._fit_linear_slice(X_grp, y_grp, model_cols)
                    self.models_by_context[(slice_name, *keys)] = fit_res

        self.is_fitted = True
        return self

    def fit(self, historical_forecasts: pd.DataFrame, historical_observations: pd.DataFrame) -> "RidgeBlender":
        """
        Align historical multi-model forecasts with observations, then train Ridge blenders.
        """
        verified = match_forecasts_to_observations(
            forecasts_df=historical_forecasts,
            observations_df=historical_observations,
            config=self.config,
        )
        return self.fit_verification_data(verified)

    def _get_context_params(
        self,
        variable: str,
        lead_time: Optional[int] = None,
        region: Optional[str] = None,
    ) -> Tuple[float, Dict[str, float], str, int, float]:
        """
        Retrieve fitted intercept and model weights with hierarchical scope fallback.
        Returns: (intercept, weights_dict, scope_name, sample_count, residual_std)
        """
        # Scope 1: var_lead_reg
        if self.condition_on_region and region and lead_time is not None:
            key1 = ("var_lead_reg", variable, int(lead_time), str(region))
            if key1 in self.models_by_context:
                res = self.models_by_context[key1]
                return res["intercept"], res["weights"], "var_lead_reg", res["sample_count"], res["residual_std"]

        # Scope 2: var_lead
        if self.condition_on_lead_time and lead_time is not None:
            key2 = ("var_lead", variable, int(lead_time))
            if key2 in self.models_by_context:
                res = self.models_by_context[key2]
                return res["intercept"], res["weights"], "var_lead", res["sample_count"], res["residual_std"]

        # Scope 3: var
        key3 = ("var", variable)
        if key3 in self.models_by_context:
            res = self.models_by_context[key3]
            return res["intercept"], res["weights"], "var", res["sample_count"], res["residual_std"]

        # Scope 4: Global fit
        return self.global_intercept, self.global_weights, "global", len(self.known_models), 1.0

    def get_weights(
        self,
        variable: str,
        lead_time: Optional[int] = None,
        region: Optional[str] = None,
        available_models: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Retrieve model weights normalized across available models.
        """
        models = available_models or self.known_models
        if not models:
            return {}

        _, weights_dict, _, _, _ = self._get_context_params(variable, lead_time, region)

        raw_w = [weights_dict.get(m, 1.0 / len(models)) for m in models]

        if self.constrained_weights:
            raw_w = np.clip(raw_w, 0.0, None)
            total = np.sum(raw_w)
            if total > 0:
                norm_w = raw_w / total
            else:
                norm_w = np.ones(len(models)) / len(models)
            return dict(zip(models, [float(w) for w in norm_w]))
        else:
            return dict(zip(models, [float(w) for w in raw_w]))

    def get_weight_metadata(
        self,
        variable: str,
        lead_time: Optional[int] = None,
        region: Optional[str] = None,
        available_models: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve diagnostic metadata distinguishing intercept, weights, and scope.
        """
        intercept, weights_dict, scope, count, res_std = self._get_context_params(variable, lead_time, region)
        models = available_models or self.known_models

        weights_out = {m: weights_dict.get(m, 1.0 / max(1, len(models))) for m in models}

        return {
            "blender": self.name,
            "regression_model": "ridge",
            "alpha": self.alpha,
            "fit_intercept": self.fit_intercept,
            "constrained_weights": self.constrained_weights,
            "intercept": round(intercept, 4),
            "weights": {m: round(w, 4) for m, w in weights_out.items()},
            "scope": scope,
            "sample_count": count,
            "residual_std": round(res_std, 4),
        }

    def predict(self, forecasts_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate adaptive regression blended forecasts.
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

            valid_grp = grp[grp["forecast_value"].notna()].copy()
            if valid_grp.empty:
                continue

            avail_models = list(valid_grp["model_name"].values)
            avail_values = valid_grp["forecast_value"].values

            intercept, _, _, _, res_std = self._get_context_params(
                variable=key_dict["variable"],
                lead_time=key_dict["lead_time"],
                region=key_dict.get("region"),
            )

            weights = self.get_weights(
                variable=key_dict["variable"],
                lead_time=key_dict["lead_time"],
                region=key_dict.get("region"),
                available_models=avail_models,
            )

            w_vec = np.array([weights.get(m, 0.0) for m in avail_models])

            if self.constrained_weights:
                blended_val = float(intercept + np.sum(w_vec * avail_values))
                # Diagnostic spread
                spread = float(np.sqrt(np.sum(w_vec * (avail_values - blended_val) ** 2))) if len(avail_values) > 1 else 0.0
            else:
                blended_val = float(intercept + np.sum(w_vec * avail_values))
                spread = float(np.std(avail_values)) if len(avail_values) > 1 else 0.0

            rec = {
                **key_dict,
                "model_name": self.name,
                "forecast_value": round(blended_val, 4),
                "forecast_spread": round(spread, 4),
                "historical_residual_std": round(res_std, 4),
            }
            records.append(rec)

        if not records:
            return pd.DataFrame()

        out_df = pd.DataFrame(records)
        return validate_forecast_dataframe(out_df, allow_nans=False)


# Legacy alias
AdaptiveLinearBlender = RidgeBlender

__all__ = [
    "RidgeBlender",
    "AdaptiveLinearBlender",
]
