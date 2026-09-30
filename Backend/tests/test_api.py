import numpy as np
import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_api_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "project" in data
    assert "supported_models" in data


def test_api_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_api_config():
    response = client.get("/config")
    assert response.status_code == 200
    data = response.json()
    assert "variables" in data
    assert "lead_times_hours" in data


def test_api_synthetic_generate():
    response = client.post("/synthetic/generate?save_to_disk=false")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["num_observation_records"] > 0
    assert data["num_forecast_records"] > 0


def test_api_evaluation_summary():
    response = client.get("/evaluation/summary")
    assert response.status_code == 200
    data = response.json()
    assert "overall_by_variable" in data
    assert "by_variable_and_lead_time" in data


def test_api_extremes_alerts():
    response = client.get("/extremes/alerts")
    assert response.status_code == 200
    data = response.json()
    assert "extreme_rainfall_alerts_count" in data


def test_api_blending_weights():
    response = client.get("/blending/weights?variable=temperature&lead_time=24&region=plains&blender=inverse_variance")
    assert response.status_code == 200
    data = response.json()
    assert data["variable"] == "temperature"
    assert data["lead_time"] == 24
    assert "weights" in data
    assert len(data["weights"]) >= 3
    # Check weights sum to ~1.0
    total_w = sum(data["weights"].values())
    assert abs(total_w - 1.0) < 1e-4
    assert "metadata" in data


def test_api_blending_forecast():
    # Provide a sample multi-model forecast payload
    payload = {
        "blender": "inverse_variance",
        "forecasts": [
            {
                "timestamp": "2026-01-05T12:00:00Z",
                "run_time": "2026-01-04T12:00:00Z",
                "latitude": 28.61,
                "longitude": 77.20,
                "lead_time": 24,
                "model_name": "nwp_model_a",
                "variable": "temperature",
                "forecast_value": 24.5,
                "region": "plains",
            },
            {
                "timestamp": "2026-01-05T12:00:00Z",
                "run_time": "2026-01-04T12:00:00Z",
                "latitude": 28.61,
                "longitude": 77.20,
                "lead_time": 24,
                "model_name": "nwp_model_b",
                "variable": "temperature",
                "forecast_value": 25.1,
                "region": "plains",
            },
            {
                "timestamp": "2026-01-05T12:00:00Z",
                "run_time": "2026-01-04T12:00:00Z",
                "latitude": 28.61,
                "longitude": 77.20,
                "lead_time": 24,
                "model_name": "ai_model_c",
                "variable": "temperature",
                "forecast_value": 24.8,
                "region": "plains",
            },
        ],
    }
    response = client.post("/blending/forecast", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["blender"] == "inverse_variance_blend"
    assert data["num_records"] == 1
    f_rec = data["forecasts"][0]
    assert 24.0 <= f_rec["forecast_value"] <= 26.0
    assert "forecast_spread" in f_rec
    assert "historical_residual_std" in f_rec


def test_api_blending_skill():
    response = client.get("/blending/skill?variable=temperature&lead_time=24")
    assert response.status_code == 200
    data = response.json()
    assert "skill_metrics" in data
    assert data["num_records"] > 0


def test_api_blending_comparison():
    response = client.get("/blending/comparison")
    assert response.status_code == 200
    data = response.json()
    assert "overall_by_variable" in data
    assert "by_variable_and_lead_time" in data


def test_api_regimes_classify_rule_based():
    payload = {
        "records": [
            {
                "timestamp": "2026-07-01T12:00:00Z",
                "temperature": 25.0,
                "rainfall": 60.0,
                "wind_speed": 5.0,
            },
            {
                "timestamp": "2026-07-01T18:00:00Z",
                "temperature": 20.0,
                "rainfall": 0.0,
                "wind_speed": 18.0,
            },
        ],
        "method": "rule_based",
    }
    response = client.post("/regimes/classify", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["num_records"] == 2
    assert data["classified_records"][0]["weather_regime"] == "heavy_rain"
    assert data["classified_records"][1]["weather_regime"] == "high_wind"


def test_api_regimes_clusters_diagnostics():
    response = client.get("/regimes/clusters?k_min=3&k_max=5")
    assert response.status_code == 200
    data = response.json()
    assert "evaluations" in data
    assert len(data["evaluations"]) == 3
    assert data["k_range"] == [3, 4, 5]


def test_api_blending_gating():
    payload = {
        "blender": "regime_inverse_variance",
        "forecasts": [
            {
                "timestamp": "2026-01-05T12:00:00Z",
                "latitude": 28.61,
                "longitude": 77.20,
                "lead_time": 24,
                "model_name": "gfs_seamless",
                "variable": "rainfall",
                "forecast_value": 45.0,
                "weather_regime": "convective",
            },
            {
                "timestamp": "2026-01-05T12:00:00Z",
                "latitude": 28.61,
                "longitude": 77.20,
                "lead_time": 24,
                "model_name": "ai_graphcast_ensemble",
                "variable": "rainfall",
                "forecast_value": 42.0,
                "weather_regime": "convective",
            },
        ],
    }
    response = client.post("/blending/gating", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["num_records"] == 1
    assert "forecast_value" in data["blended_forecasts"][0]


# ---------------------------------------------------------------------------
# PHASE 5 API TESTS
# ---------------------------------------------------------------------------


def test_api_spatial_grid():
    response = client.get("/spatial/grid")
    assert response.status_code == 200
    data = response.json()
    assert "lat_min" in data
    assert "lat_max" in data
    assert "shape" in data


def test_api_spatial_features():
    response = client.get("/spatial/features?variable=rainfall&lead_time=24")
    assert response.status_code == 200
    data = response.json()
    assert data["variable"] == "rainfall"
    assert "channel_summaries" in data
    assert data["num_channels"] > 0


def test_api_spatial_disagreement():
    response = client.get("/spatial/disagreement?variable=rainfall&lead_time=24")
    assert response.status_code == 200
    data = response.json()
    assert "ensemble_mean_avg" in data
    assert "intermodel_spread_avg" in data


def test_api_spatial_weights():
    response = client.get("/spatial/weights?variable=rainfall&method=equal_weight&lead_time=24")
    assert response.status_code == 200
    data = response.json()
    assert "model_weights" in data
    assert np.isclose(data["weight_sum_check"], 1.0)


def test_api_probabilistic_forecast():
    response = client.get("/probabilistic/forecast?variable=temperature&lead_time=24&method=emos")
    assert response.status_code == 200
    data = response.json()
    assert data["variable"] == "temperature"
    assert "mean_avg" in data
    assert "std_avg" in data


def test_api_probabilistic_calibration():
    response = client.get("/probabilistic/calibration?variable=rainfall&threshold=25.0")
    assert response.status_code == 200
    data = response.json()
    assert data["threshold"] == 25.0
    assert "brier_score" in data
    assert "reliability_curve" in data


def test_api_extremes_probability_map():
    response = client.get("/extremes/probability-map?variable=rainfall&threshold=25.0&lead_time=24")
    assert response.status_code == 200
    data = response.json()
    assert data["variable"] == "rainfall"
    assert "mean_exceedance_probability" in data
    assert "grid_shape" in data



