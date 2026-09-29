"""
Spatial Feature Engineering and Model Disagreement Module for Phase 5.

Extracts 2D spatial gradients, local spatial statistics, spatial anomalies, and multi-model
disagreement layers (mean, standard deviation spread, range, pairwise model differences)
to provide rich spatial context for CNN encoders and spatial gating networks.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.ndimage as ndimage
import xarray as xr

from configs import AppConfig, get_config


def compute_spatial_gradients(
    field_2d: np.ndarray,
    dx: float = 1.0,
    dy: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute horizontal gradients of a 2D scalar field.

    Parameters
    ----------
    field_2d : 2D np.ndarray (H, W)
    dx : grid spacing along columns (x/lon)
    dy : grid spacing along rows (y/lat)

    Returns
    -------
    grad_y : 2D array (dF/dy)
    grad_x : 2D array (dF/dx)
    grad_mag : 2D array sqrt(grad_y^2 + grad_x^2)
    """
    field = np.asarray(field_2d, dtype=np.float32)
    # Handle NaNs by filling with mean for gradient calculation
    if np.isnan(field).any():
        valid_mean = np.nanmean(field)
        field_filled = np.where(np.isnan(field), valid_mean, field)
    else:
        field_filled = field

    grad_y, grad_x = np.gradient(field_filled, dy, dx)
    grad_mag = np.sqrt(grad_y**2 + grad_x**2)
    return grad_y.astype(np.float32), grad_x.astype(np.float32), grad_mag.astype(np.float32)


def compute_local_spatial_statistics(
    field_2d: np.ndarray,
    window_size: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute local spatial mean and local spatial standard deviation using a 2D moving window.

    Parameters
    ----------
    field_2d : 2D np.ndarray (H, W)
    window_size : int, size of square filter window

    Returns
    -------
    local_mean : 2D array
    local_std : 2D array
    """
    field = np.asarray(field_2d, dtype=np.float32)
    if np.isnan(field).any():
        valid_mean = float(np.nanmean(field))
        field = np.where(np.isnan(field), valid_mean, field)

    local_mean = ndimage.uniform_filter(field, size=window_size, mode="reflect")
    local_sq_mean = ndimage.uniform_filter(field**2, size=window_size, mode="reflect")
    variance = np.maximum(0.0, local_sq_mean - local_mean**2)
    local_std = np.sqrt(variance)
    return local_mean.astype(np.float32), local_std.astype(np.float32)


def compute_spatial_anomaly(field_2d: np.ndarray) -> np.ndarray:
    """
    Compute spatial anomaly: field - spatial_mean(field).
    """
    field = np.asarray(field_2d, dtype=np.float32)
    mean_val = float(np.nanmean(field)) if field.size > 0 else 0.0
    return (field - mean_val).astype(np.float32)


def compute_model_disagreement_2d(
    model_fields: Union[Dict[str, np.ndarray], np.ndarray],
    model_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute spatial model disagreement layer across multi-model forecast fields.

    Parameters
    ----------
    model_fields : Dict[str, 2D array] or 3D array of shape (M, H, W)
    model_names : list of str, optional

    Returns
    -------
    disagreement_dict : dict containing:
        - "mean": 2D array (inter-model ensemble mean)
        - "std": 2D array (inter-model spread / standard deviation)
        - "range": 2D array (inter-model max - min)
        - "pairwise_diffs": Dict[str, 2D array] of |Model_i - Model_j|
        - "max_pairwise_diff": 2D array
    """
    if isinstance(model_fields, dict):
        names = list(model_fields.keys())
        stacked = np.stack([model_fields[m] for m in names], axis=0).astype(np.float32)
    elif isinstance(model_fields, np.ndarray):
        stacked = model_fields.astype(np.float32)
        names = model_names or [f"model_{i}" for i in range(stacked.shape[0])]
    else:
        raise ValueError("model_fields must be a dict or 3D numpy array.")

    M, H, W = stacked.shape
    ens_mean = np.mean(stacked, axis=0)
    ens_std = np.std(stacked, axis=0)
    ens_range = np.ptp(stacked, axis=0)

    pairwise_diffs = {}
    diff_stack = []
    for i in range(M):
        for j in range(i + 1, M):
            pair_key = f"{names[i]}_vs_{names[j]}"
            diff = np.abs(stacked[i] - stacked[j])
            pairwise_diffs[pair_key] = diff
            diff_stack.append(diff)

    max_pairwise = np.max(np.stack(diff_stack, axis=0), axis=0) if diff_stack else np.zeros((H, W), dtype=np.float32)

    return {
        "mean": ens_mean.astype(np.float32),
        "std": ens_std.astype(np.float32),
        "range": ens_range.astype(np.float32),
        "pairwise_diffs": pairwise_diffs,
        "max_pairwise_diff": max_pairwise.astype(np.float32),
        "model_names": names,
    }


def extract_spatial_feature_stack(
    forecast_ds: xr.Dataset,
    variable: str = "temperature",
    lead_time: int = 6,
    time_idx: int = 0,
    include_disagreement: bool = True,
    include_gradients: bool = True,
    include_local_stats: bool = True,
    include_auxiliary_vars: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract a multi-channel 2D spatial feature tensor for a given timestamp and lead time.

    Parameters
    ----------
    forecast_ds : xr.Dataset with coords (time, lead_time, model, latitude, longitude)
    variable : target meteorological variable
    lead_time : forecast lead time in hours
    time_idx : index into time coordinate
    include_disagreement : bool
    include_gradients : bool
    include_local_stats : bool
    include_auxiliary_vars : bool

    Returns
    -------
    feature_stack : 3D np.ndarray of shape (Channels, H, W)
    channel_names : list of str describing each feature channel
    """
    t_val = forecast_ds.time.values[time_idx]
    ds_slice = forecast_ds.sel(time=t_val, lead_time=lead_time)

    models = [str(m) for m in ds_slice.model.values]
    channels = []
    channel_names = []

    # 1. Base model forecast fields for target variable
    target_data = ds_slice[variable].values  # shape: (M, H, W)
    for m_idx, m_name in enumerate(models):
        channels.append(target_data[m_idx])
        channel_names.append(f"{m_name}_{variable}")

    # 2. Inter-model disagreement features
    if include_disagreement and len(models) > 1:
        disagree = compute_model_disagreement_2d(target_data, model_names=models)
        channels.append(disagree["mean"])
        channel_names.append(f"{variable}_ensemble_mean")
        channels.append(disagree["std"])
        channel_names.append(f"{variable}_intermodel_std")
        channels.append(disagree["range"])
        channel_names.append(f"{variable}_intermodel_range")
        channels.append(disagree["max_pairwise_diff"])
        channel_names.append(f"{variable}_max_pairwise_diff")

    # 3. Spatial gradients on ensemble mean
    ens_mean = np.mean(target_data, axis=0)
    if include_gradients:
        grad_y, grad_x, grad_mag = compute_spatial_gradients(ens_mean)
        channels.append(grad_y)
        channel_names.append(f"{variable}_grad_lat")
        channels.append(grad_x)
        channel_names.append(f"{variable}_grad_lon")
        channels.append(grad_mag)
        channel_names.append(f"{variable}_grad_mag")

    # 4. Local spatial statistics & anomaly
    if include_local_stats:
        local_mean, local_std = compute_local_spatial_statistics(ens_mean)
        anomaly = compute_spatial_anomaly(ens_mean)
        channels.append(local_mean)
        channel_names.append(f"{variable}_local_mean")
        channels.append(local_std)
        channel_names.append(f"{variable}_local_std")
        channels.append(anomaly)
        channel_names.append(f"{variable}_spatial_anomaly")

    # 5. Auxiliary variables if available (e.g. pressure, humidity, rain, wind)
    if include_auxiliary_vars:
        for aux_var in ["surface_pressure", "relative_humidity", "rainfall", "wind_speed"]:
            if aux_var != variable and aux_var in ds_slice:
                aux_data = ds_slice[aux_var].values  # (M, H, W)
                aux_mean = np.mean(aux_data, axis=0)
                channels.append(aux_mean)
                channel_names.append(f"aux_{aux_var}_mean")

    stack = np.stack(channels, axis=0).astype(np.float32)
    return stack, channel_names
