"""
Chronological Walk-Forward Validation Engine and Bootstrap Resampling Evaluator.
Strictly prevents future observation leakage in historical model skill estimation.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from configs import AppConfig, get_config
from ingestion.schema import validate_forecast_dataframe, validate_observation_dataframe
from preprocessing.matching import match_forecasts_to_observations
from models.base import BaseBlender
from models.baseline import EqualWeightBlender
from models.skill_weighting import InverseVarianceSkillBlender, InverseMAESkillBlender
from models.adaptive_blending import RidgeBlender
from models.regime_blending import RegimeInverseVarianceBlender, RegimeInverseMAEBlender
from models.gating_model import WeatherRegimeGatingNetwork
from evaluation.comparison import compare_models
from evaluation.metrics import root_mean_squared_error, mean_absolute_error, mean_bias_error, pearson_correlation


class WalkForwardBlenderEvaluator:
    """
    Chronological walk-forward validation evaluator for multi-model forecast blenders.
    At each forward step T_k, blenders are fitted strictly on observations verified before T_k.
    """

    def __init__(
        self,
        blenders: Optional[List[BaseBlender]] = None,
        mode: str = "rolling",  # 'rolling' or 'expanding'
        window_size: int = 30,
        min_train_steps: int = 5,
        include_phase4: bool = False,
        config: Optional[AppConfig] = None,
    ):
        self.config = config or get_config()
        self.mode = mode.lower()
        self.window_size = window_size
        self.min_train_steps = min_train_steps

        if blenders is not None:
            self.blenders = blenders
        else:
            # Default suite of benchmark blenders
            self.blenders = [
                EqualWeightBlender(name="equal_weight_blend"),
                InverseMAESkillBlender(name="inverse_mae_blend", config=self.config),
                InverseVarianceSkillBlender(name="inverse_variance_blend", config=self.config),
                RidgeBlender(constrained_weights=True, name="ridge_constrained_blend", config=self.config),
                RidgeBlender(constrained_weights=False, name="ridge_unconstrained_blend", config=self.config),
            ]
            if include_phase4:
                self.blenders.extend([
                    RegimeInverseVarianceBlender(name="regime_inverse_variance_blend", config=self.config),
                    RegimeInverseMAEBlender(name="regime_inverse_mae_blend", config=self.config),
                    WeatherRegimeGatingNetwork(name="neural_gating_blend", config=self.config),
                ])


    def run(
        self,
        forecasts_df: pd.DataFrame,
        observations_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Execute chronological walk-forward validation across all available time steps.

        Parameters
        ----------
        forecasts_df : pd.DataFrame
            Multi-model forecasts conforming to standard schema.
        observations_df : pd.DataFrame
            Ground-truth observations conforming to standard schema.

        Returns
        -------
        pd.DataFrame
            Combined verified predictions across individual models, baselines, and adaptive blenders.
        """
        # Pre-align verification records for individual models
        verified_all = match_forecasts_to_observations(
            forecasts_df=forecasts_df,
            observations_df=observations_df,
            config=self.config,
        )

        if verified_all.empty:
            return pd.DataFrame()

        f_df = validate_forecast_dataframe(forecasts_df)
        unique_times = sorted(list(f_df["timestamp"].dropna().unique()))
        n_times = len(unique_times)

        if n_times <= self.min_train_steps:
            # If dataset is too short for split, fit on all available history
            blended_dfs = []
            for blender in self.blenders:
                blender.fit_verification_data(verified_all)
                b_pred = blender.predict(f_df)
                blended_dfs.append(b_pred)

            combined_forecasts = pd.concat([f_df] + blended_dfs, ignore_index=True)
            return match_forecasts_to_observations(combined_forecasts, observations_df, config=self.config)

        # Collect out-of-sample predictions for each step
        out_of_sample_blend_forecasts = []

        for k in range(self.min_train_steps, n_times):
            test_ts = unique_times[k]

            # 1. Historical Training Slice (Strictly BEFORE test_ts)
            if self.mode == "rolling":
                start_k = max(0, k - self.window_size)
                train_times = unique_times[start_k:k]
            else:
                train_times = unique_times[:k]

            train_history = verified_all[verified_all["timestamp"].isin(train_times)].copy()

            # 2. Test Slice Forecasts
            test_forecasts = f_df[f_df["timestamp"] == test_ts].copy()
            if test_forecasts.empty or train_history.empty:
                continue

            # 3. Fit and predict each blender
            for blender in self.blenders:
                blender.fit_verification_data(train_history)
                pred_df = blender.predict(test_forecasts)
                if not pred_df.empty:
                    out_of_sample_blend_forecasts.append(pred_df)

        if not out_of_sample_blend_forecasts:
            return verified_all

        all_blended_forecasts = pd.concat(out_of_sample_blend_forecasts, ignore_index=True)
        combined_all_forecasts = pd.concat([f_df, all_blended_forecasts], ignore_index=True)

        # Merge with observations to create final verified test evaluation dataset
        final_verified = match_forecasts_to_observations(
            forecasts_df=combined_all_forecasts,
            observations_df=observations_df,
            config=self.config,
        )

        return final_verified


def compute_paired_bootstrap_ci(
    baseline_errors: Union[np.ndarray, List[float]],
    blend_errors: Union[np.ndarray, List[float]],
    metric: str = "rmse",
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Compute paired bootstrap difference and empirical confidence interval between two models.
    Difference is calculated as: delta = Metric(blend) - Metric(baseline).
    A negative delta indicates the blend achieved a lower error (improvement).

    Parameters
    ----------
    baseline_errors : array-like
        Array of baseline forecast errors (e_base = F_base - y).
    blend_errors : array-like
        Array of blended forecast errors (e_blend = F_blend - y).
    metric : str, default 'rmse'
        Target metric ('rmse' or 'mae').
    n_bootstrap : int, default 1000
        Number of bootstrap resampling iterations.
    confidence_level : float, default 0.95
        Confidence level for two-sided interval.
    seed : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, float]
        Dictionary with 'mean_diff', 'ci_lower', 'ci_upper', 'p_value_superiority'.
    """
    e_base = np.asarray(baseline_errors, dtype=float).ravel()
    e_blend = np.asarray(blend_errors, dtype=float).ravel()

    valid_mask = np.isfinite(e_base) & np.isfinite(e_blend)
    e_base = e_base[valid_mask]
    e_blend = e_blend[valid_mask]

    n = len(e_base)
    if n < 2:
        return {"mean_diff": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "p_value_superiority": 0.5}

    rng = np.random.default_rng(seed)
    diffs = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        b_sample = e_base[idx]
        m_sample = e_blend[idx]

        if metric.lower() == "mae":
            score_base = np.mean(np.abs(b_sample))
            score_blend = np.mean(np.abs(m_sample))
        else:
            score_base = np.sqrt(np.mean(b_sample**2))
            score_blend = np.sqrt(np.mean(m_sample**2))

        diffs.append(score_blend - score_base)

    diffs = np.array(diffs)
    alpha = (1.0 - confidence_level) / 2.0
    ci_lower = float(np.percentile(diffs, alpha * 100))
    ci_upper = float(np.percentile(diffs, (1.0 - alpha) * 100))
    mean_diff = float(np.mean(diffs))

    # Probability that blend achieves lower error (diff < 0)
    p_superiority = float(np.mean(diffs < 0))

    return {
        "mean_diff": round(mean_diff, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "p_value_superiority": round(p_superiority, 4),
    }


def compute_relative_improvements(
    comparison_df: pd.DataFrame,
    baseline_model: str = "equal_weight_blend",
    metric: str = "rmse",
) -> pd.DataFrame:
    """
    Calculate relative percentage skill improvement against a baseline model:
        Improvement (%) = ((Metric_base - Metric_blend) / Metric_base) * 100
    """
    df = comparison_df.copy()
    m = metric.lower()
    if m not in df.columns:
        return df

    partition_keys = [c for c in ["variable", "lead_time", "region"] if c in df.columns]
    results = []

    for keys, grp in df.groupby(partition_keys, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_dict = dict(zip(partition_keys, keys))

        base_row = grp[grp["model_name"] == baseline_model]
        if base_row.empty:
            continue

        base_val = base_row[m].iloc[0]
        if base_val <= 0 or np.isnan(base_val):
            continue

        for _, row in grp.iterrows():
            m_val = row[m]
            if m in ["rmse", "mae"]:
                # Lower is better
                rel_imp = ((base_val - m_val) / base_val) * 100.0
            elif m == "correlation":
                # Higher is better
                rel_imp = ((m_val - base_val) / max(0.01, abs(base_val))) * 100.0
            else:
                rel_imp = 0.0

            r = row.to_dict()
            r[f"{m}_rel_improvement_vs_{baseline_model}"] = round(rel_imp, 2)
            results.append(r)

    return pd.DataFrame(results) if results else df


def evaluate_regime_stratified_performance(
    verified_df: pd.DataFrame,
    models: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute comprehensive verification metrics stratified across weather regimes.
    Reports MAE, RMSE, Bias, and Pearson Correlation for each model and regime.
    """
    if verified_df.empty:
        return pd.DataFrame()

    df = verified_df.copy()
    if "weather_regime" not in df.columns:
        df["weather_regime"] = "overall"

    if models is not None:
        df = df[df["model_name"].isin(models)]

    # 1. Regime-specific evaluation
    regime_metrics = compare_models(df, group_by=["variable", "weather_regime"])

    # 2. Overall evaluation
    df_overall = df.copy()
    df_overall["weather_regime"] = "ALL (Overall)"
    overall_metrics = compare_models(df_overall, group_by=["variable", "weather_regime"])

    combined = pd.concat([overall_metrics, regime_metrics], ignore_index=True)
    return combined


__all__ = [
    "WalkForwardBlenderEvaluator",
    "compute_paired_bootstrap_ci",
    "compute_relative_improvements",
    "evaluate_regime_stratified_performance",
]

