"""
Comprehensive Final Multi-Dimensional Validation Protocol for Phase 6.
Evaluates multi-model blending systems across chronological temporal holdouts,
spatial out-of-domain regions, weather regimes, seasons, and lead times.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from configs import AppConfig, get_config
from evaluation.probabilistic_metrics import crps_gaussian, compute_crpss, evaluate_probabilistic_forecast
from evaluation.calibration import brier_score, reliability_diagram, brier_skill_score


class MultiDimensionalHoldoutValidator:
    """
    Orchestrator for rigorous multi-dimensional holdout validation.
    Enforces strict chronological time splits and cross-domain spatial generalization testing.
    """

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or get_config()

    def split_chronological(
        self,
        df: pd.DataFrame,
        train_frac: float = 0.70,
        val_frac: float = 0.15,
        time_col: str = "timestamp",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Partition dataset strictly chronologically into Train, Validation, and Test splits.
        Never random shuffles time series data.
        """
        sorted_df = df.sort_values(by=time_col).reset_index(drop=True)
        unique_times = sorted_df[time_col].drop_duplicates().tolist()
        n_times = len(unique_times)

        n_train = int(n_times * train_frac)
        n_val = int(n_times * (train_frac + val_frac))

        train_times = set(unique_times[:n_train])
        val_times = set(unique_times[n_train:n_val])
        test_times = set(unique_times[n_val:])

        train_df = sorted_df[sorted_df[time_col].isin(train_times)].reset_index(drop=True)
        val_df = sorted_df[sorted_df[time_col].isin(val_times)].reset_index(drop=True)
        test_df = sorted_df[sorted_df[time_col].isin(test_times)].reset_index(drop=True)

        return train_df, val_df, test_df

    def split_regional(
        self,
        df: pd.DataFrame,
        train_regions: List[str],
        test_region: str,
        region_col: str = "region",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Partition dataset for spatial out-of-domain generalization (Train on Regions A+B, Test on Region C).
        """
        train_df = df[df[region_col].isin(train_regions)].reset_index(drop=True)
        test_df = df[df[region_col] == test_region].reset_index(drop=True)
        return train_df, test_df

    def evaluate_multi_dimension(
        self,
        df: pd.DataFrame,
        obs_col: str = "observed_value",
        pred_col: str = "forecast_value",
        model_col: str = "model_name",
        dimension: str = "regime",  # 'regime', 'lead_time', 'season', 'region'
    ) -> pd.DataFrame:
        """
        Evaluate deterministic forecast error metrics broken down across a specific meteorological dimension.
        """
        if dimension not in df.columns:
            return pd.DataFrame()

        records = []
        for (dim_val, var, model), grp in df.groupby([dimension, "variable", model_col], observed=True):
            if grp.empty or len(grp) < 2:
                continue
            obs = grp[obs_col].values
            pred = grp[pred_col].values
            try:
                metrics = compute_all_metrics(obs, pred)
                records.append({
                    "dimension": dimension,
                    "dimension_value": str(dim_val),
                    "variable": var,
                    "model_name": model,
                    "sample_count": len(grp),
                    **metrics,
                })
            except Exception:
                continue

        return pd.DataFrame(records)

    def evaluate_extreme_contingency(
        self,
        obs: np.ndarray,
        probs: np.ndarray,
        threshold: float,
        prob_threshold: float = 0.5,
    ) -> Dict[str, float]:
        """
        Compute extreme-event 2x2 contingency table metrics: Precision, Recall, F1, CSI (Threat Score).
        """
        actual_event = (obs >= threshold).astype(int)
        pred_event = (probs >= prob_threshold).astype(int)

        tp = int(np.sum((actual_event == 1) & (pred_event == 1)))
        fp = int(np.sum((actual_event == 0) & (pred_event == 1)))
        fn = int(np.sum((actual_event == 1) & (pred_event == 0)))
        tn = int(np.sum((actual_event == 0) & (pred_event == 0)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

        brier = float(np.mean((probs - actual_event) ** 2))

        return {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "csi_threat_score": round(csi, 4),
            "brier_score": round(brier, 4),
        }
