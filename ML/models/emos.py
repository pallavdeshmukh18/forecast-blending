"""
Ensemble Model Output Statistics (EMOS) Module for Phase 5.

Implements Gaussian EMOS calibration:
    Y ~ N(mu, sigma^2)
    mu = a + b * F_bar
    sigma^2 = c + d * S^2
where F_bar is the ensemble mean, S^2 is the ensemble spread variance, and (a,b,c,d)
are parameters fitted by minimizing CRPS or negative log-likelihood on historical training data.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.optimize as opt
import scipy.stats as stats

from configs import AppConfig, get_config
from models.probabilistic import ProbabilisticForecast


def gaussian_crps_loss(
    obs: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> float:
    """
    Calculate mean Continuous Ranked Probability Score (CRPS) for a Gaussian distribution N(mu, sigma^2).
    """
    sigma = np.maximum(1e-5, sigma)
    z = (obs - mu) / sigma
    phi = stats.norm.pdf(z)
    Phi = stats.norm.cdf(z)
    crps = sigma * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / np.sqrt(np.pi))
    return float(np.nanmean(crps))


class EnsembleModelOutputStatistics:
    """
    Gaussian Ensemble Model Output Statistics (EMOS) Post-Processor.
    Calibrates multi-model ensemble forecasts by learning affine mean correction (a, b)
    and non-negative spread calibration (c, d) on historical training data.
    """

    def __init__(
        self,
        fit_method: str = "crps",  # "crps", "nll", or "least_squares"
        min_spread_epsilon: float = 1e-4,
        quantiles: Optional[List[float]] = None,
    ):
        self.fit_method = fit_method.lower()
        self.min_spread_epsilon = min_spread_epsilon
        self.quantiles = quantiles or [0.10, 0.25, 0.50, 0.75, 0.90]

        # Parameters (a, b, c, d) where mu = a + b*F_bar, sigma^2 = c + d*S^2
        self.a: float = 0.0
        self.b: float = 1.0
        self.c: float = 1.0
        self.d: float = 1.0
        self.is_fitted: bool = False
        self.training_loss: Optional[float] = None

    def fit(
        self,
        ensemble_forecasts: np.ndarray,
        observations: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> "EnsembleModelOutputStatistics":
        """
        Fit EMOS parameters (a, b, c, d) strictly on historical training data.

        Parameters
        ----------
        ensemble_forecasts : np.ndarray (N, M) or (N, M, H, W)
        observations : np.ndarray (N,) or (N, H, W)
        weights : optional weights for ensemble members
        """
        E = np.asarray(ensemble_forecasts, dtype=np.float64)
        Y = np.asarray(observations, dtype=np.float64)

        if E.ndim == 4:
            # Flatten spatial dims to samples
            N, M, H, W = E.shape
            E = E.transpose(0, 2, 3, 1).reshape(-1, M)
            Y = Y.reshape(-1)

        valid_mask = ~np.isnan(Y) & ~np.isnan(E).any(axis=1)
        E_valid = E[valid_mask]
        Y_valid = Y[valid_mask]

        if len(Y_valid) < 5:
            # Fallback default identity parameters
            self.a, self.b, self.c, self.d = 0.0, 1.0, 1.0, 1.0
            self.is_fitted = True
            return self

        # Compute ensemble mean and ensemble variance
        if weights is not None:
            w = np.asarray(weights, dtype=np.float64)
            w_norm = w / np.maximum(np.sum(w), 1e-6)
            f_bar = np.sum(w_norm * E_valid, axis=1)
            s_sq = np.sum(w_norm * (E_valid - f_bar[:, np.newaxis]) ** 2, axis=1)
        else:
            f_bar = np.mean(E_valid, axis=1)
            s_sq = np.var(E_valid, axis=1, ddof=1) if E_valid.shape[1] > 1 else np.zeros_like(f_bar)

        s_sq = np.maximum(self.min_spread_epsilon, s_sq)

        # Initial parameter guess [a, b, c, d]
        init_params = [0.0, 1.0, float(np.var(Y_valid - f_bar)), 1.0]
        # Parameter bounds: a in (-inf, inf), b in (0.1, 2.0), c >= min_spread, d >= 0
        bounds = [
            (-50.0, 50.0),
            (0.01, 3.0),
            (self.min_spread_epsilon, 100.0),
            (0.0, 10.0),
        ]

        if self.fit_method == "crps":
            def objective(theta):
                a_val, b_val, c_val, d_val = theta
                mu_pred = a_val + b_val * f_bar
                sigma_sq = np.maximum(self.min_spread_epsilon, c_val + d_val * s_sq)
                sigma_pred = np.sqrt(sigma_sq)
                return gaussian_crps_loss(Y_valid, mu_pred, sigma_pred)

        elif self.fit_method == "nll":
            def objective(theta):
                a_val, b_val, c_val, d_val = theta
                mu_pred = a_val + b_val * f_bar
                sigma_sq = np.maximum(self.min_spread_epsilon, c_val + d_val * s_sq)
                # Negative log-likelihood of Gaussian
                nll = 0.5 * np.log(2 * np.pi * sigma_sq) + ((Y_valid - mu_pred) ** 2) / (2 * sigma_sq)
                return float(np.nanmean(nll))

        else:  # least_squares
            def objective(theta):
                a_val, b_val, _, _ = theta
                mu_pred = a_val + b_val * f_bar
                return float(np.mean((Y_valid - mu_pred) ** 2))

        result = opt.minimize(
            objective,
            x0=init_params,
            bounds=bounds,
            method="L-BFGS-B",
        )

        if result.success or result.x is not None:
            self.a, self.b, self.c, self.d = [float(p) for p in result.x]
            self.training_loss = float(result.fun)
        else:
            self.a, self.b, self.c, self.d = init_params[0], init_params[1], init_params[2], init_params[3]
            self.training_loss = float(objective(init_params))

        self.is_fitted = True
        return self

    def predict(
        self,
        ensemble_forecasts: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> ProbabilisticForecast:
        """
        Produce a calibrated ProbabilisticForecast using learned EMOS parameters.

        Parameters
        ----------
        ensemble_forecasts : np.ndarray (M, H, W) or (N, M, H, W) or (N, M)
        """
        if not self.is_fitted:
            raise RuntimeError("EnsembleModelOutputStatistics must be fitted before predict().")

        E = np.asarray(ensemble_forecasts, dtype=np.float32)

        # Determine model axis
        if E.ndim == 2:  # (N, M)
            model_axis = 1
        elif E.ndim == 3:  # (M, H, W)
            model_axis = 0
        elif E.ndim == 4:  # (N, M, H, W)
            model_axis = 1
        else:
            raise ValueError(f"Unsupported ensemble shape: {E.shape}")

        if weights is not None:
            w = np.asarray(weights, dtype=np.float32)
            w_sum = np.sum(w, axis=model_axis, keepdims=True)
            w_norm = w / np.maximum(w_sum, 1e-6)
            f_bar = np.sum(w_norm * E, axis=model_axis)
            s_sq = np.sum(w_norm * (E - np.expand_dims(f_bar, axis=model_axis)) ** 2, axis=model_axis)
        else:
            f_bar = np.mean(E, axis=model_axis)
            s_sq = np.var(E, axis=model_axis)

        # Apply EMOS calibration formulas:
        # mu = a + b * F_bar
        # sigma = sqrt(c + d * S^2)
        mu = self.a + self.b * f_bar
        sigma_sq = np.maximum(self.min_spread_epsilon, self.c + self.d * s_sq)
        sigma = np.sqrt(sigma_sq)

        # Quantiles
        q_dict = {}
        for q in self.quantiles:
            z = float(stats.norm.ppf(q))
            q_dict[q] = (mu + z * sigma).astype(np.float32)

        return ProbabilisticForecast(
            mean=mu.astype(np.float32),
            std=sigma.astype(np.float32),
            quantiles=q_dict,
            distribution_type="emos_gaussian",
            metadata={
                "emos_params": {
                    "a": self.a,
                    "b": self.b,
                    "c": self.c,
                    "d": self.d,
                },
                "training_loss": self.training_loss,
            },
        )


# Alias for Gaussian EMOS
GaussianEMOS = EnsembleModelOutputStatistics
