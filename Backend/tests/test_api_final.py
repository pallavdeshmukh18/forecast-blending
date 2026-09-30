"""
Unit and Integration tests for Phase 6 FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_api_health():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["version"] == "0.6.0"


def test_api_metadata():
    res = client.get("/metadata")
    assert res.status_code == 200
    data = res.json()
    assert data["pipeline_version"] == "0.6.0"
    assert "available_models" in data
    assert "available_variables" in data
    assert "available_lead_times" in data
    assert "available_regimes" in data
    assert "available_blenders" in data


def test_api_forecast_run():
    payload = {
        "mode": "synthetic",
        "variable": "rainfall",
        "lead_time": 24,
        "blender_method": "regime_blender",
    }
    res = client.post("/forecast/run", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["pipeline_version"] == "0.6.0"
    assert data["target_variable"] == "rainfall"
    assert len(data["forecast"]) > 0
    assert "dominant_regime" in data["regime"]
    assert "mean_spread" in data["uncertainty"]


def test_api_forecast_blend():
    payload = {
        "variable": "temperature",
        "lead_time": 24,
        "blender_method": "equal_weight",
    }
    res = client.post("/forecast/blend", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["variable"] == "temperature"
    assert "weights" in data
    assert len(data["forecast"]) > 0


def test_api_forecast_probability():
    payload = {
        "variable": "rainfall",
        "lead_time": 24,
        "quantiles": [0.10, 0.50, 0.90],
    }
    res = client.post("/forecast/probability", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["variable"] == "rainfall"
    assert "uncertainty" in data


def test_api_forecast_extremes():
    payload = {
        "variable": "rainfall",
        "lead_time": 24,
        "threshold": 25.0,
    }
    res = client.post("/forecast/extremes", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["variable"] == "rainfall"
    assert "probabilities" in data


def test_api_verify():
    payload = {
        "variable": "rainfall",
        "lead_time": 24,
        "blender_method": "regime_blender",
    }
    res = client.post("/verify", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "verification" in data
    assert "mae" in data["verification"]


def test_api_models():
    res = client.get("/models")
    assert res.status_code == 200
    data = res.json()
    assert "models" in data
    assert len(data["models"]) >= 3


def test_api_models_weights():
    res = client.get("/models/weights?variable=rainfall&lead_time=24")
    assert res.status_code == 200
    data = res.json()
    assert "weights" in data


def test_api_regimes():
    res = client.get("/regimes")
    assert res.status_code == 200
    data = res.json()
    assert "priority_order" in data


def test_api_diagnostics():
    res = client.get("/diagnostics")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "operational"


def test_api_metrics():
    res = client.get("/metrics?variable=rainfall&lead_time=24")
    assert res.status_code == 200
    data = res.json()
    assert "verification" in data
