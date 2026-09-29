"""
Probabilistic Forecast Representation and Distribution Models for Phase 5.

Provides ProbabilisticForecast container, EnsembleProbabilisticModel, and
ResidualUncertaintyProbabilisticModel to represent forecast uncertainty via
parametric distributions, ensemble quantiles, and historical residuals.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.stats as stats

from configs import AppConfig, get_config


class ProbabilisticForecast:
    """
    Standardized container representing a probabilistic weather forecast distribution.
    Supports point estimates (mean, median), spread (std), quantiles (P10, P25, P50, P75, P90),
    exceedance probabilities, and prediction intervals across scalar or multi-dimensional grid fields.
    """

    def __init__(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        quantiles: Optional[Dict[float, np.ndarray]] = None,
        distribution_type: str = "gaussian",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.maximum(1e-6, np.asarray(std, dtype=np.float32))
        self.distribution_type = distribution_type.lower()
        self.metadata = metadata or {}

        if quantiles is not None:
            self.quantiles = {float(q): np.asarray(v, dtype=np.float32) for q, v in quantiles.items()}
        else:
            # Default parametric Gaussian quantiles
            default_qs = [0.10, 0.25, 0.50, 0.75, 0.90]
            self.quantiles = {}
            for q in default_qs:
                z = float(stats.norm.ppf(q))
                self.quantiles[q] = (self.mean + z * self.std).astype(np.float32)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.mean.shape

    def cdf(self, threshold: float) -> np.ndarray:
        """Compute P(Y <= threshold)."""
        if self.distribution_type == "gaussian":
            z = (threshold - self.mean) / self.std
            return stats.norm.cdf(z).astype(np.float32)
        elif 0.50 in self.quantiles:
            # Simple approximation using normal fit
            z = (threshold - self.mean) / self.std
            return stats.norm.cdf(z).astype(np.float32)
        else:
            z = (threshold - self.mean) / self.std
            return stats.norm.cdf(z).astype(np.float32)

    def prob_exceeds(self, threshold: float) -> np.ndarray:
        """Compute P(Y >= threshold) = 1 - CDF(threshold)."""
        return (1.0 - self.cdf(threshold)).astype(np.float32)

    def prob_interval(self, low: float, high: float) -> np.ndarray:
        """Compute P(low <= Y <= high)."""
        return np.maximum(0.0, self.cdf(high) - self.cdf(low)).astype(np.float32)

    def quantile(self, q: float) -> np.ndarray:
        """Get quantile value for q in (0, 1)."""
        q = float(q)
        if q in self.quantiles:
            return self.quantiles[q]
        z = float(stats.norm.ppf(q))
        return (self.mean + z * self.std).astype(np.float32)

    def prediction_interval(self, coverage: float = 0.80) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (lower_bound, upper_bound) for central prediction interval of given coverage.
        E.g. coverage=0.80 -> (P10, P90).
        """
        alpha = (1.0 - coverage) / 2.0
        q_low = round(alpha, 3)
        q_high = round(1.0 - alpha, 3)
        return self.quantile(q_low), self.quantile(q_high)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "distribution_type": self.distribution_type,
            "mean_shape": list(self.mean.shape),
            "mean_avg": float(np.nanmean(self.mean)),
            "std_avg": float(np.nanmean(self.std)),
            "quantiles": list(self.quantiles.keys()),
            "metadata": self.metadata,
        }


class EnsembleProbabilisticModel:
    """
    Constructs a ProbabilisticForecast directly from multi-model ensemble member forecasts.
    """

    def __init__(
        self,
        quantiles: Optional[List[float]] = None,
        min_spread_epsilon: float = 1e-4,
    ):
        self.quantiles = quantiles or [0.10, 0.25, 0.50, 0.75, 0.90]
        self.min_spread_epsilon = min_spread_epsilon

    def predict(
        self,
        ensemble_members: Union[List[np.ndarray], np.ndarray],
        weights: Optional[np.ndarray] = None,
    ) -> ProbabilisticForecast:
        """
        Compute probabilistic distribution from ensemble members.

        Parameters
        ----------
        ensemble_members : 3D array (M, H, W) or 4D array (N, M, H, W) or list of arrays
        weights : optional weights (M,) or (M, H, W) or (N, M, H, W)

        Returns
        -------
        ProbabilisticForecast
        """
        if isinstance(ensemble_members, list):
            E = np.stack(ensemble_members, axis=0).astype(np.float32)
        else:
            E = np.asarray(ensemble_members, dtype=np.float32)

        # Model dimension is axis=0 for 2D (M, N) or 3D (M, H, W); axis=1 for 4D (N, M, H, W)
        model_axis = 0 if E.ndim in [2, 3] else 1

        if weights is not None:
            w = np.asarray(weights, dtype=np.float32)
            # Normalize weights along model axis
            w_sum = np.sum(w, axis=model_axis, keepdims=True)
            w_norm = w / np.maximum(w_sum, 1e-6)
            mean = np.sum(w_norm * E, axis=model_axis)
            variance = np.sum(w_norm * (E - np.expand_dims(mean, axis=model_axis)) ** 2, axis=model_axis)
            std = np.sqrt(np.maximum(self.min_spread_epsilon, variance))
        else:
            mean = np.mean(E, axis=model_axis)
            std = np.maximum(self.min_spread_epsilon, np.std(E, axis=model_axis))

        # Compute empirical quantiles along model axis
        q_dict = {}
        for q in self.quantiles:
            q_dict[q] = np.quantile(E, q, axis=model_axis).astype(np.float32)

        return ProbabilisticForecast(
            mean=mean,
            std=std,
            quantiles=q_dict,
            distribution_type="ensemble",
            metadata={"num_members": E.shape[model_axis]},
        )


class ResidualUncertaintyProbabilisticModel:
    """
    Constructs a ProbabilisticForecast from a deterministic blended forecast
    combined with estimated historical residual variance sigma_res^2.
    """

    def __init__(
        self,
        residual_std: float = 1.0,
        quantiles: Optional[List[float]] = None,
    ):
        self.residual_std = max(1e-4, float(residual_std))
        self.quantiles = quantiles or [0.10, 0.25, 0.50, 0.75, 0.90]

    def fit_residuals(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
    ) -> "ResidualUncertaintyProbabilisticModel":
        """Fit residual standard deviation on historical training data."""
        res = np.asarray(predictions) - np.asarray(ground_truth)
        self.residual_std = max(1e-4, float(np.nanstd(res)))
        return self

    def predict(
        self,
        blended_forecast: np.ndarray,
        spatial_residual_std: Optional[np.ndarray] = None,
    ) -> ProbabilisticForecast:
        """
        Produce a ProbabilisticForecast around blended_forecast.
        """
        mean = np.asarray(blended_forecast, dtype=np.float32)
        if spatial_residual_std is not None:
            std = np.maximum(1e-4, np.asarray(spatial_residual_std, dtype=np.float32))
        else:
            std = np.full_like(mean, self.residual_std, dtype=np.float32)

        q_dict = {}
        for q in self.quantiles:
            z = float(stats.norm.ppf(q))
            q_dict[q] = (mean + z * std).astype(np.float32)

        return ProbabilisticForecast(
            mean=mean,
            std=std,
            quantiles=q_dict,
            distribution_type="gaussian",
            metadata={"residual_std": self.residual_std},
        )


class ProbabilisticForecastEngine:
    """
    Unified engine for constructing and calibrating probabilistic forecasts.
    """

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or get_config()
        self.quantiles = self.config.probabilistic.quantiles

    def predict(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        quantiles: Optional[List[float]] = None,
        distribution_type: str = "gaussian",
    ) -> ProbabilisticForecast:
        """Construct standard ProbabilisticForecast container."""
        qs = quantiles or self.quantiles
        q_dict = {}
        mean_arr = np.asarray(mean, dtype=np.float32)
        std_arr = np.maximum(1e-6, np.asarray(std, dtype=np.float32))

        for q in qs:
            z = float(stats.norm.ppf(q))
            q_dict[q] = (mean_arr + z * std_arr).astype(np.float32)

        return ProbabilisticForecast(
            mean=mean_arr,
            std=std_arr,
            quantiles=q_dict,
            distribution_type=distribution_type,
        )
