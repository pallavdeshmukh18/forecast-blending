"""
Evaluation module providing meteorological metrics, model skill comparison,
probabilistic metrics, calibration diagnostics, validation splitters, and block bootstrap inference.
"""

from evaluation.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    mean_bias_error,
    pearson_correlation,
    compute_all_metrics,
)
from evaluation.comparison import compare_models, compute_skill_rankings
from evaluation.validation import TimeSeriesSplitter
from evaluation.error_database import HistoricalErrorDatabase
from evaluation.walk_forward import (
    WalkForwardBlenderEvaluator,
    compute_paired_bootstrap_ci,
    compute_relative_improvements,
)
from evaluation.probabilistic_metrics import (
    crps_gaussian,
    crps_ensemble,
    compute_crpss,
    evaluate_probabilistic_forecast,
    picp_mpiw,
)
from evaluation.calibration import (
    brier_score,
    reliability_diagram,
    brier_skill_score,
    probability_calibration_error,
)
from evaluation.model_comparison import (
    compute_effective_sample_size,
    block_bootstrap_ci,
    evaluate_paired_model_comparison,
)
from evaluation.final_validation import MultiDimensionalHoldoutValidator

__all__ = [
    "mean_absolute_error",
    "root_mean_squared_error",
    "mean_bias_error",
    "pearson_correlation",
    "compute_all_metrics",
    "compare_models",
    "compute_skill_rankings",
    "TimeSeriesSplitter",
    "HistoricalErrorDatabase",
    "WalkForwardBlenderEvaluator",
    "compute_paired_bootstrap_ci",
    "compute_relative_improvements",
    "crps_gaussian",
    "brier_score",
    "compute_crpss",
    "evaluate_probabilistic_forecast",
    "picp_mpiw",
    "reliability_diagram",
    "brier_skill_score",
    "probability_calibration_error",
    "compute_effective_sample_size",
    "block_bootstrap_ci",
    "evaluate_paired_model_comparison",
    "MultiDimensionalHoldoutValidator",
]
