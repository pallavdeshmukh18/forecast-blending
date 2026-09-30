"""
Forecasting, spatial neural gating, probabilistic post-processing, and extreme-weather calibration models.
"""

from models.base import BaseBlender
from models.baseline import (
    equal_weight_blend,
    single_model_forecast,
    EqualWeightBlender,
)
from models.skill_weighting import (
    SkillWeightedBlender,
    InverseVarianceSkillBlender,
    InverseMAESkillBlender,
)
from models.adaptive_blending import (
    RidgeBlender,
    AdaptiveLinearBlender,
)
from models.regime_blending import (
    RegimeSkillBlender,
    RegimeInverseVarianceBlender,
    RegimeInverseMAEBlender,
)
from models.gating_model import (
    GatingPyTorchModule,
    WeatherRegimeGatingNetwork,
    RegimeGatingBlender,
)
from models.spatial_encoder import SpatialCNNEncoder
from models.spatial_gating import (
    SpatialGatingNetwork,
    SpatialCNNBlender,
)
from models.spatial_blending import (
    SpatialInverseVarianceBlender,
    SpatialEqualWeightBlender,
)
from models.probabilistic import ProbabilisticForecastEngine
from models.emos import GaussianEMOS
from models.extreme_calibration import (
    GammaEMOS,
    ZeroInflatedGammaEMOS,
    GEVTailCalibrator,
    ExtremeProbabilityEngine,
    ExtremeThresholdResult,
)
from models.temporal_encoder import (
    ConvGRUCell,
    SpatioTemporalEncoder,
)
from models.spatiotemporal_gating import (
    SpatioTemporalGatingNetwork,
    SpatioTemporalCNNBlender,
)

__all__ = [
    "BaseBlender",
    "equal_weight_blend",
    "single_model_forecast",
    "EqualWeightBlender",
    "SkillWeightedBlender",
    "InverseVarianceSkillBlender",
    "InverseMAESkillBlender",
    "RidgeBlender",
    "AdaptiveLinearBlender",
    "RegimeSkillBlender",
    "RegimeInverseVarianceBlender",
    "RegimeInverseMAEBlender",
    "GatingPyTorchModule",
    "WeatherRegimeGatingNetwork",
    "RegimeGatingBlender",
    "SpatialCNNEncoder",
    "SpatialGatingNetwork",
    "SpatialCNNBlender",
    "SpatialInverseVarianceBlender",
    "SpatialEqualWeightBlender",
    "ProbabilisticForecastEngine",
    "GaussianEMOS",
    "GammaEMOS",
    "ZeroInflatedGammaEMOS",
    "GEVTailCalibrator",
    "ExtremeProbabilityEngine",
    "ExtremeThresholdResult",
    "ConvGRUCell",
    "SpatioTemporalEncoder",
    "SpatioTemporalGatingNetwork",
    "SpatioTemporalCNNBlender",
]
