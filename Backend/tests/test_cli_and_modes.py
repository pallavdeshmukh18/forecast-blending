"""
Integration tests for CLI scripts and DATA_MODE execution modes.
"""

import os
from pathlib import Path
import subprocess
import sys
import pandas as pd
import pytest

from configs import load_config, get_config


def test_data_mode_environment_override(monkeypatch):
    monkeypatch.setenv("DATA_MODE", "real")
    cfg = load_config()
    assert cfg.data.mode == "real"

    monkeypatch.setenv("DATA_MODE", "synthetic")
    cfg = load_config()
    assert cfg.data.mode == "synthetic"


def test_validate_real_dataset_script_cli():
    obs_file = Path("data/samples/sample_observations.csv")
    assert obs_file.exists()

    result = subprocess.run(
        [sys.executable, "scripts/validate_real_dataset.py", "--input", str(obs_file)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "DATA QUALITY REPORT" in result.stdout
    assert "Valid Records:" in result.stdout
    assert "[PASS]" in result.stdout


def test_build_verification_dataset_script_cli(tmp_path: Path):
    obs_file = Path("data/samples/sample_observations.csv")
    fcs_file = Path("data/samples/sample_all_forecasts.csv")
    out_file = tmp_path / "test_verification.parquet"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_verification_dataset.py",
            "--forecast", str(fcs_file),
            "--observations", str(obs_file),
            "--output", str(out_file),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert out_file.exists()
    assert "Produced" in result.stdout
    assert "Verification dataset build complete" in result.stdout

    # Verify output parquet schema
    df = pd.read_parquet(out_file)
    assert len(df) > 0
    assert "forecast_error" in df.columns
    assert "absolute_error" in df.columns
    assert "squared_error" in df.columns
    assert "lead_time_bin" in df.columns
    assert "season" in df.columns
    assert "weather_regime" in df.columns
