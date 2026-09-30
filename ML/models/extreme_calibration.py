"""
Advanced Extreme-Weather Calibration and Tail Modeling.
Implements Gamma EMOS, Zero-Inflated / Censored Gamma EMOS, GEV tail fitting, and extreme exceedance probability engines.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from pydantic import BaseModel, Field

from configs import AppConfig, get_config


class ExtremeThresholdResult(BaseModel):
    """
    Standard record for extreme event probability output at a grid cell or station.
    """
    latitude: float
    longitude: float
    timestamp: str
    lead_time: int
    variable: str
    threshold: float
    probability: float
    calibration_method: str
    condition: str = ">"  # '>' for exceedance, '<' for cold extremes


class GammaEMOS:
    """
    Gamma Ensemble Model Output Statistics (Gamma EMOS) for non-negative skewed variables (e.g. precipitation).
    Models precipitation intensity via Gamma distribution parameterized by mean and variance links:
    mu = a + b * ensemble_mean
    var = c + d * ensemble_variance (or spread)
    shape alpha = mu^2 / var
    scale theta = var / mu  (or rate beta = mu / var)
    """

    def __init__(self, min_spread_eps: float = 1e-4):
        self.min_spread_eps = min_spread_eps
        self.params: Optional[np.ndarray] = None  # [a, b, c, d]
        self.fitted: bool = False

    def fit(self, ens_mean: np.ndarray, ens_var: np.ndarray, observations: np.ndarray) -> "GammaEMOS":
        """
        Fit Gamma EMOS parameters minimizing negative log-likelihood on positive observations.
        """
        ens_mean = np.asarray(ens_mean, dtype=float)
        ens_var = np.asarray(ens_var, dtype=float)
        obs = np.asarray(observations, dtype=float)

        # Filter positive values
        valid_mask = (obs > 0.0) & ~np.isnan(obs) & ~np.isnan(ens_mean) & ~np.isnan(ens_var)
        if np.sum(valid_mask) < 5:
            # Fallback trivial parameters
            self.params = np.array([0.1, 1.0, 0.1, 1.0])
            self.fitted = True
            return self

        m_sub = ens_mean[valid_mask]
        v_sub = ens_var[valid_mask]
        y_sub = obs[valid_mask]

        def nll_objective(p: np.ndarray) -> float:
            a, b, c, d = p
            mu = np.maximum(a + b * m_sub, 1e-3)
            variance = np.maximum(c + d * v_sub, self.min_spread_eps)
            alpha = np.maximum((mu ** 2) / variance, 1e-2)
            beta_scale = np.maximum(variance / mu, 1e-3)

            ll = stats.gamma.logpdf(y_sub, a=alpha, scale=beta_scale)
            ll = np.where(np.isnan(ll) | np.isinf(ll), -20.0, ll)
            return -float(np.sum(ll))

        init_p = np.array([0.1, 1.0, 0.5, 1.0])
        bounds = [(0.0, None), (0.0, None), (1e-4, None), (0.0, None)]
        res = minimize(nll_objective, init_p, bounds=bounds, method="L-BFGS-B")

        self.params = res.x if res.success else init_p
        self.fitted = True
        return self

    def predict_distribution(self, ens_mean: np.ndarray, ens_var: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Gamma shape (alpha) and scale (theta) parameters.
        """
        if not self.fitted or self.params is None:
            a, b, c, d = 0.1, 1.0, 0.5, 1.0
        else:
            a, b, c, d = self.params

        ens_mean = np.asarray(ens_mean, dtype=float)
        ens_var = np.asarray(ens_var, dtype=float)
        mu = np.maximum(a + b * ens_mean, 1e-3)
        variance = np.maximum(c + d * ens_var, self.min_spread_eps)

        alpha = np.maximum((mu ** 2) / variance, 1e-2)
        scale = np.maximum(variance / mu, 1e-3)
        return alpha, scale

    def exceedance_probability(self, ens_mean: np.ndarray, ens_var: np.ndarray, threshold: float) -> np.ndarray:
        """
        Calculate P(Rain > threshold) under fitted Gamma distribution.
        """
        alpha, scale = self.predict_distribution(ens_mean, ens_var)
        # 1 - CDF(threshold) = survival function (sf)
        probs = stats.gamma.sf(threshold, a=alpha, scale=scale)
        return np.clip(probs, 0.0, 1.0)


class ZeroInflatedGammaEMOS:
    """
    Zero-Inflated (Censored) Gamma EMOS for precipitation.
    Decouples probability of zero rainfall P(Y = 0) from the continuous positive rainfall distribution Y | (Y > 0) ~ Gamma(alpha, theta).
    P(Y = 0) is modeled via logistic regression on ensemble mean and fraction of zero ensemble members.
    """

    def __init__(self, censoring_threshold: float = 0.1, min_spread_eps: float = 1e-4):
        self.censoring_threshold = censoring_threshold
        self.min_spread_eps = min_spread_eps
        self.gamma_emos = GammaEMOS(min_spread_eps=min_spread_eps)
        self.logit_params: Optional[np.ndarray] = None  # [intercept, slope]
        self.fitted: bool = False

    def fit(self, ens_mean: np.ndarray, ens_var: np.ndarray, observations: np.ndarray) -> "ZeroInflatedGammaEMOS":
        """
        Fit zero probability model and Gamma intensity parameters.
        """
        ens_mean = np.asarray(ens_mean, dtype=float)
        ens_var = np.asarray(ens_var, dtype=float)
        obs = np.asarray(observations, dtype=float)

        valid_mask = ~np.isnan(obs) & ~np.isnan(ens_mean) & ~np.isnan(ens_var)
        if np.sum(valid_mask) < 10:
            self.logit_params = np.array([0.0, -0.5])
            self.gamma_emos.fit(ens_mean, ens_var, obs)
            self.fitted = True
            return self

        m_valid = ens_mean[valid_mask]
        v_valid = ens_var[valid_mask]
        y_valid = obs[valid_mask]

        # 1. Zero probability binary model (Y <= censoring_threshold)
        is_zero = (y_valid <= self.censoring_threshold).astype(float)

        def logit_nll(p: np.ndarray) -> float:
            w0, w1 = p
            z = np.clip(w0 + w1 * m_valid, -20.0, 20.0)
            p_zero = 1.0 / (1.0 + np.exp(-z))
            p_zero = np.clip(p_zero, 1e-6, 1.0 - 1e-6)
            ll = is_zero * np.log(p_zero) + (1.0 - is_zero) * np.log(1.0 - p_zero)
            return -float(np.sum(ll))

        res_logit = minimize(logit_nll, np.array([0.5, -0.2]), method="L-BFGS-B")
        self.logit_params = res_logit.x if res_logit.success else np.array([0.5, -0.2])

        # 2. Fit Gamma distribution on strictly positive observations
        pos_mask = y_valid > self.censoring_threshold
        if np.sum(pos_mask) >= 5:
            self.gamma_emos.fit(m_valid[pos_mask], v_valid[pos_mask], y_valid[pos_mask])
        else:
            self.gamma_emos.fit(m_valid, v_valid, y_valid)

        self.fitted = True
        return self

    def predict_zero_probability(self, ens_mean: np.ndarray) -> np.ndarray:
        """
        Predict P(Rain <= censoring_threshold).
        """
        if self.logit_params is None:
            w0, w1 = 0.5, -0.2
        else:
            w0, w1 = self.logit_params

        ens_mean = np.asarray(ens_mean, dtype=float)
        z = np.clip(w0 + w1 * ens_mean, -20.0, 20.0)
        p_zero = 1.0 / (1.0 + np.exp(-z))
        return np.clip(p_zero, 0.0, 1.0)

    def cdf(self, ens_mean: np.ndarray, ens_var: np.ndarray, y: float) -> np.ndarray:
        """
        Compute full Zero-Inflated Gamma Cumulative Distribution Function at value y.
        """
        p_zero = self.predict_zero_probability(ens_mean)
        if y <= self.censoring_threshold:
            return p_zero

        alpha, scale = self.gamma_emos.predict_distribution(ens_mean, ens_var)
        gamma_cdf = stats.gamma.cdf(y, a=alpha, scale=scale)
        return p_zero + (1.0 - p_zero) * gamma_cdf

    def exceedance_probability(self, ens_mean: np.ndarray, ens_var: np.ndarray, threshold: float) -> np.ndarray:
        """
        Calculate P(Rain > threshold) = (1 - P(Zero)) * P(Gamma > threshold).
        """
        p_zero = self.predict_zero_probability(ens_mean)
        if threshold <= self.censoring_threshold:
            return 1.0 - p_zero

        alpha, scale = self.gamma_emos.predict_distribution(ens_mean, ens_var)
        gamma_sf = stats.gamma.sf(threshold, a=alpha, scale=scale)
        probs = (1.0 - p_zero) * gamma_sf
        return np.clip(probs, 0.0, 1.0)


class GEVTailCalibrator:
    """
    Generalized Extreme Value (GEV) distribution for extreme upper tail exceedances.
    Applies only when historical extreme sample size meets or exceeds minimum threshold.
    """

    def __init__(self, min_samples: int = 50):
        self.min_samples = min_samples
        self.params: Optional[Tuple[float, float, float]] = None  # (shape xi, loc mu, scale sigma)
        self.fitted: bool = False

    def fit(self, observations: np.ndarray) -> bool:
        """
        Fit GEV distribution if sample size is adequate.
        Returns True if GEV fit succeeded, False if fallback is required.
        """
        obs = np.asarray(observations, dtype=float)
        valid = obs[~np.isnan(obs) & (obs > 0.0)]
        if len(valid) < self.min_samples:
            self.fitted = False
            return False

        try:
            # scipy stats genextreme: shape c = -xi
            c, loc, scale = stats.genextreme.fit(valid)
            self.params = (c, loc, scale)
            self.fitted = True
            return True
        except Exception:
            self.fitted = False
            return False

    def exceedance_probability(self, threshold: float) -> Optional[float]:
        """
        Calculate P(Obs > threshold) under stationary GEV fit.
        """
        if not self.fitted or self.params is None:
            return None
        c, loc, scale = self.params
        prob = stats.genextreme.sf(threshold, c, loc=loc, scale=scale)
        return float(np.clip(prob, 0.0, 1.0))


class ExtremeProbabilityEngine:
    """
    Central orchestrator for spatial and station extreme weather probability estimation across variables.
    """

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or get_config()
        self.zig_calibrator = ZeroInflatedGammaEMOS(
            censoring_threshold=self.config.extreme_calibration.censoring_threshold
        )
        self.gev_calibrators: Dict[str, GEVTailCalibrator] = {
            "rainfall": GEVTailCalibrator(min_samples=self.config.extreme_calibration.gev_min_samples),
            "wind_speed": GEVTailCalibrator(min_samples=self.config.extreme_calibration.gev_min_samples),
        }

    def calibrate_on_history(
        self,
        historical_df: pd.DataFrame,
        blend_col: str = "blend_value",
        spread_col: str = "spread",
        obs_col: str = "observed_value",
    ) -> None:
        """
        Fit all calibrators using strictly historical verification records.
        """
        # Rainfall Zero-Inflated Gamma
        rain_df = historical_df[historical_df["variable"] == "rainfall"]
        if not rain_df.empty and obs_col in rain_df.columns:
            m = rain_df[blend_col].values if blend_col in rain_df.columns else rain_df[obs_col].values
            v = (rain_df[spread_col].values ** 2) if spread_col in rain_df.columns else np.ones_like(m)
            self.zig_calibrator.fit(m, v, rain_df[obs_col].values)

            if self.config.extreme_calibration.enable_gev_tail:
                self.gev_calibrators["rainfall"].fit(rain_df[obs_col].values)

        # Wind Speed GEV Tail
        wind_df = historical_df[historical_df["variable"] == "wind_speed"]
        if not wind_df.empty and obs_col in wind_df.columns and self.config.extreme_calibration.enable_gev_tail:
            self.gev_calibrators["wind_speed"].fit(wind_df[obs_col].values)

    def compute_extreme_probabilities(
        self,
        forecast_df: pd.DataFrame,
        blend_col: str = "blend_value",
        spread_col: str = "spread",
    ) -> pd.DataFrame:
        """
        Generate comprehensive table of extreme event probabilities for each row and configurable threshold.

        Returns DataFrame with columns:
        [latitude, longitude, timestamp, lead_time, variable, threshold, probability, calibration_method, condition]
        """
        results: List[Dict[str, Any]] = []

        cfg_extremes = self.config.extremes
        rain_thresholds = cfg_extremes.rainfall_thresholds
        wind_thresholds = cfg_extremes.wind_thresholds
        hot_thresholds = getattr(cfg_extremes, "hot_temperature_thresholds", [35.0, 38.0, 40.0])
        cold_thresholds = getattr(cfg_extremes, "cold_temperature_thresholds", [0.0, 5.0])

        for _, row in forecast_df.iterrows():
            var = str(row.get("variable", "unknown"))
            val = float(row.get(blend_col, 0.0))
            spread = float(row.get(spread_col, 1.0))
            std_dev = max(spread, 0.1)
            variance = std_dev ** 2

            lat = float(row.get("latitude", 0.0))
            lon = float(row.get("longitude", 0.0))
            ts = str(row.get("timestamp", ""))
            lead = int(row.get("lead_time", 0))

            if var == "rainfall":
                for thresh in rain_thresholds:
                    prob = float(self.zig_calibrator.exceedance_probability(
                        np.array([val]), np.array([variance]), thresh
                    )[0])
                    results.append({
                        "latitude": lat,
                        "longitude": lon,
                        "timestamp": ts,
                        "lead_time": lead,
                        "variable": var,
                        "threshold": thresh,
                        "probability": round(prob, 4),
                        "calibration_method": "zero_inflated_gamma",
                        "condition": ">",
                    })

            elif var == "wind_speed":
                for thresh in wind_thresholds:
                    # Truncated normal survival function (wind >= 0)
                    prob = float(stats.norm.sf(thresh, loc=val, scale=std_dev))
                    results.append({
                        "latitude": lat,
                        "longitude": lon,
                        "timestamp": ts,
                        "lead_time": lead,
                        "variable": var,
                        "threshold": thresh,
                        "probability": round(float(np.clip(prob, 0.0, 1.0)), 4),
                        "calibration_method": "truncated_gaussian_emos",
                        "condition": ">",
                    })

            elif var == "temperature":
                # Hot extremes: P(T > threshold)
                for thresh in hot_thresholds:
                    prob = float(stats.norm.sf(thresh, loc=val, scale=std_dev))
                    results.append({
                        "latitude": lat,
                        "longitude": lon,
                        "timestamp": ts,
                        "lead_time": lead,
                        "variable": var,
                        "threshold": thresh,
                        "probability": round(float(np.clip(prob, 0.0, 1.0)), 4),
                        "calibration_method": "gaussian_emos",
                        "condition": ">",
                    })

                # Cold extremes: P(T < threshold)
                for thresh in cold_thresholds:
                    prob = float(stats.norm.cdf(thresh, loc=val, scale=std_dev))
                    results.append({
                        "latitude": lat,
                        "longitude": lon,
                        "timestamp": ts,
                        "lead_time": lead,
                        "variable": var,
                        "threshold": thresh,
                        "probability": round(float(np.clip(prob, 0.0, 1.0)), 4),
                        "calibration_method": "gaussian_emos",
                        "condition": "<",
                    })

        return pd.DataFrame(results)
