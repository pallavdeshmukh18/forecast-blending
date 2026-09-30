"""
Spatial Multi-Model Blending Baselines and Weight Resolvers for Phase 5.

Implements SpatialWeightedBlender, SpatialEqualWeightBlender, SpatialInverseVarianceBlender,
and SpatialRegimeSkillBlender for 2D gridded multi-model forecast blending.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import xarray as xr

from configs import AppConfig, get_config


class SpatialWeightedBlender:
    """
    Base class for 2D gridded multi-model forecast blending using spatial weight fields W_m(x,y).
    """

    def __init__(
        self,
        model_names: Optional[List[str]] = None,
        epsilon: float = 1e-6,
    ):
        self.model_names = model_names or []
        self.epsilon = epsilon

    def blend_fields(
        self,
        forecast_fields: Union[Dict[str, np.ndarray], np.ndarray],
        weight_fields: Union[Dict[str, np.ndarray], np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Blend 2D forecast fields using 2D weight fields.

        Parameters
        ----------
        forecast_fields : Dict[model_name -> (H, W)] or 3D array (M, H, W)
        weight_fields : Dict[model_name -> (H, W)] or 3D array (M, H, W)

        Returns
        -------
        blended_field : 2D array (H, W)
        normalized_weights : 3D array (M, H, W)
        """
        if isinstance(forecast_fields, dict):
            models = list(forecast_fields.keys())
            F = np.stack([forecast_fields[m] for m in models], axis=0).astype(np.float32)
        else:
            F = np.asarray(forecast_fields, dtype=np.float32)
            models = self.model_names or [f"model_{i}" for i in range(F.shape[0])]

        if isinstance(weight_fields, dict):
            W = np.stack([weight_fields.get(m, np.zeros_like(F[0])) for m in models], axis=0).astype(np.float32)
        else:
            W = np.asarray(weight_fields, dtype=np.float32)

        M, H, W_dim = F.shape
        if W.shape != (M, H, W_dim):
            raise ValueError(f"Shape mismatch: F has shape {F.shape} but W has shape {W.shape}")

        # Ensure non-negativity
        W = np.maximum(0.0, W)

        # Handle NaNs in forecast fields: mask out invalid models per cell
        valid_mask = ~np.isnan(F)
        W_effective = W * valid_mask

        # Sum of effective weights per cell
        W_sum = np.sum(W_effective, axis=0, keepdims=True)

        # Renormalize weights where sum > 0
        W_norm = np.where(W_sum > self.epsilon, W_effective / np.maximum(W_sum, self.epsilon), 0.0)

        # For cells where no valid model had positive weight, fallback to equal weight over valid models
        valid_count = np.sum(valid_mask, axis=0, keepdims=True)
        fallback_mask = (W_sum <= self.epsilon) & (valid_count > 0)
        if np.any(fallback_mask):
            equal_w = np.where(valid_mask, 1.0 / np.maximum(valid_count, 1.0), 0.0)
            W_norm = np.where(fallback_mask, equal_w, W_norm)

        # Compute blended field: sum_m W_m * F_m
        F_zeroed = np.nan_to_num(F, nan=0.0)
        blended = np.sum(W_norm * F_zeroed, axis=0)

        # If a cell had all NaNs, return NaN
        all_nan_mask = valid_count[0] == 0
        blended[all_nan_mask] = np.nan

        return blended.astype(np.float32), W_norm.astype(np.float32)


class SpatialEqualWeightBlender(SpatialWeightedBlender):
    """
    Equal-weight 2D spatial blender: W_m(x,y) = 1/M.
    """

    def predict(
        self,
        forecast_fields: Union[Dict[str, np.ndarray], np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(forecast_fields, dict):
            models = list(forecast_fields.keys())
            F = np.stack([forecast_fields[m] for m in models], axis=0).astype(np.float32)
        else:
            F = np.asarray(forecast_fields, dtype=np.float32)
            models = self.model_names or [f"model_{i}" for i in range(F.shape[0])]

        M, H, W = F.shape
        equal_weights = np.full((M, H, W), 1.0 / M, dtype=np.float32)
        return self.blend_fields(F, equal_weights)


class SpatialInverseVarianceBlender(SpatialWeightedBlender):
    """
    Spatial Inverse-Variance Blender:
    Calculates historical error variance sigma_m^2(x,y) at each grid cell from historical
    training grids, and computes convex spatial weights:
        W_m(x,y) propto 1 / (sigma_m^2(x,y) + epsilon)
    with hierarchical fallback to domain-wide skill if local sample count is insufficient,
    and alpha weight smoothing regularization toward 1/M.
    """

    def __init__(
        self,
        model_names: Optional[List[str]] = None,
        minimum_samples: int = 10,
        weight_smoothing: float = 0.10,
        epsilon: float = 1e-6,
    ):
        super().__init__(model_names=model_names, epsilon=epsilon)
        self.minimum_samples = minimum_samples
        self.weight_smoothing = weight_smoothing

        self.cell_variances: Optional[np.ndarray] = None  # (M, H, W)
        self.global_variances: Optional[np.ndarray] = None  # (M,)
        self.sample_counts: Optional[np.ndarray] = None  # (M, H, W)

    def fit(
        self,
        historical_forecasts: np.ndarray,
        historical_observations: np.ndarray,
    ) -> "SpatialInverseVarianceBlender":
        """
        Fit historical grid-cell error variances strictly on historical training grids.

        Parameters
        ----------
        historical_forecasts : np.ndarray of shape (N, M, H, W)
        historical_observations : np.ndarray of shape (N, H, W) or (N, 1, H, W)
        """
        if historical_observations.ndim == 4:
            historical_observations = historical_observations.squeeze(1)

        N, M, H, W = historical_forecasts.shape
        errors = historical_forecasts - historical_observations[:, np.newaxis, :, :]  # (N, M, H, W)

        # Compute cell-wise MSE / variance
        cell_mse = np.nanmean(errors**2, axis=0)  # (M, H, W)
        sample_counts = np.sum(~np.isnan(errors), axis=0)  # (M, H, W)

        # Compute global domain-wide variance per model
        global_mse = np.nanmean(errors**2, axis=(0, 2, 3))  # (M,)

        # Fallback to global MSE for cells with low sample counts
        fallback_mask = sample_counts < self.minimum_samples
        for m in range(M):
            cell_mse[m] = np.where(fallback_mask[m], global_mse[m], cell_mse[m])

        self.cell_variances = cell_mse.astype(np.float32)
        self.global_variances = global_mse.astype(np.float32)
        self.sample_counts = sample_counts.astype(np.int32)
        return self

    def get_weights(self, shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Compute static learned spatial weights W_m(x,y) of shape (M, H, W).
        """
        if self.cell_variances is None:
            raise RuntimeError("SpatialInverseVarianceBlender must be fitted before getting weights.")

        # Inverse variance
        inv_var = 1.0 / (self.cell_variances + self.epsilon)
        raw_weights = inv_var / np.sum(inv_var, axis=0, keepdims=True)

        # Apply weight smoothing toward 1/M
        M = raw_weights.shape[0]
        smoothed_weights = (1.0 - self.weight_smoothing) * raw_weights + self.weight_smoothing * (1.0 / M)
        return smoothed_weights.astype(np.float32)

    def predict(
        self,
        forecast_fields: Union[Dict[str, np.ndarray], np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Blend forecast fields using learned spatial inverse-variance weights.
        """
        weights = self.get_weights()
        return self.blend_fields(forecast_fields, weights)
