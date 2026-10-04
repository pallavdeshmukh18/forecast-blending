"""
Tests for Real-Data Benchmark Configuration & Experiment Design.

Verifies:
- Benchmark configuration YAML loading & validity
- Non-overlapping chronological Train / Validation / Test partitions
- Strict causality (all Test dates strictly follow Train and Validation)
- Common model & lead-time intersection logic
- Variable-specific method eligibility (e.g. ZIG EMOS for rainfall, Gaussian for temperature)
- Regime and extreme threshold validity without test leakage
"""

from datetime import datetime
from pathlib import Path
import sys
import yaml
import pytest

# Setup project import paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "ML"))
sys.path.insert(0, str(PROJECT_ROOT / "Backend"))
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def benchmark_config():
    """Load real benchmark YAML configuration."""
    config_path = PROJECT_ROOT / "ML" / "configs" / "real_benchmark.yaml"
    assert config_path.exists(), f"Configuration file not found: {config_path}"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


# =====================================================================
# Test A: Benchmark Configuration Loads
# =====================================================================

def test_benchmark_config_loads(benchmark_config):
    """Verify that real_benchmark.yaml contains all required top-level sections."""
    required_sections = [
        "dataset",
        "region",
        "variables",
        "models",
        "lead_times_hours",
        "split",
        "leakage_policy",
        "regimes",
        "extreme_events",
        "method_eligibility",
        "metrics",
    ]
    for sec in required_sections:
        assert sec in benchmark_config, f"Missing required section '{sec}' in real_benchmark.yaml"

    assert benchmark_config["dataset"]["mode"] == "real"
    assert "temperature" in benchmark_config["variables"]
    assert "rainfall" in benchmark_config["variables"]
    assert "wind_speed" in benchmark_config["variables"]


# =====================================================================
# Test B & C: Chronological Split & No Temporal Leakage
# =====================================================================

def test_benchmark_split_non_overlapping(benchmark_config):
    """Verify Train, Validation, and Test partitions do not overlap and maintain chronological order."""
    split = benchmark_config["split"]
    train_start = datetime.strptime(split["train"]["start_date"], "%Y-%m-%d")
    train_end = datetime.strptime(split["train"]["end_date"], "%Y-%m-%d")
    val_start = datetime.strptime(split["validation"]["start_date"], "%Y-%m-%d")
    val_end = datetime.strptime(split["validation"]["end_date"], "%Y-%m-%d")
    test_start = datetime.strptime(split["test"]["start_date"], "%Y-%m-%d")
    test_end = datetime.strptime(split["test"]["end_date"], "%Y-%m-%d")

    # Train window validity
    assert train_start <= train_end

    # Strict causality: Train precedes Validation
    assert train_end < val_start, "Train end date must strictly precede Validation start date"
    assert val_start <= val_end

    # Strict causality: Validation precedes Test
    assert val_end < test_start, "Validation end date must strictly precede Test start date"
    assert test_start <= test_end

    # Test leakage policy flags
    leakage = benchmark_config["leakage_policy"]
    assert leakage["enforce_chronological_split"] is True
    assert leakage["allow_random_shuffle"] is False
    assert leakage["freeze_weights_for_test"] is True


# =====================================================================
# Test D: Common Model & Lead-Time Intersection
# =====================================================================

def test_common_model_lead_time_intersection(benchmark_config):
    """Verify common lead-time intersection across models."""
    common_leads = benchmark_config["lead_times_hours"]["common_intersection"]
    eval_leads = benchmark_config["lead_times_hours"]["evaluation_leads"]

    assert len(common_leads) >= 4
    for lt in common_leads:
        assert lt in eval_leads, f"Common lead time {lt} must be in evaluation leads"

    # Primary models have complete variable coverage
    primary_models = benchmark_config["models"]["primary_set"]
    assert "ecmwf_hres" in primary_models
    assert "graphcast_v2" in primary_models


# =====================================================================
# Test E: Variable-Specific Method Eligibility
# =====================================================================

def test_variable_method_eligibility(benchmark_config):
    """Verify probabilistic methods are scientifically eligible per variable."""
    eligibility = benchmark_config["method_eligibility"]

    # Zero-Inflated Gamma is eligible for rainfall, not temperature
    assert "zero_inflated_gamma_emos" in eligibility["rainfall"]["probabilistic"]
    assert "zero_inflated_gamma_emos" not in eligibility["temperature"]["probabilistic"]

    # Gaussian EMOS is eligible for temperature and wind, not rainfall
    assert "gaussian_emos" in eligibility["temperature"]["probabilistic"]
    assert "gaussian_emos" in eligibility["wind_speed"]["probabilistic"]
    assert "gaussian_emos" not in eligibility["rainfall"]["probabilistic"]


# =====================================================================
# Test F & G: Regimes & Extreme Event Thresholds
# =====================================================================

def test_regime_and_extreme_thresholds_validity(benchmark_config):
    """Verify 6-hour precipitation and extreme thresholds are physically defined."""
    thresholds = benchmark_config["regimes"]["thresholds_6hr"]
    assert thresholds["heavy_rain_mm_6h"] > thresholds["convective_rain_mm_6h"]
    assert thresholds["heavy_rain_mm_6h"] == 15.0  # 15 mm/6h ~ 60 mm/24h

    extreme_rain = benchmark_config["extreme_events"]["rainfall_thresholds_mm_6h"]
    assert sorted(extreme_rain) == extreme_rain
    assert len(extreme_rain) >= 3
    assert extreme_rain[0] < extreme_rain[-1]
