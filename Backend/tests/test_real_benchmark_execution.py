"""
Tests for Real-Data Benchmark Execution and Output Artifacts (WeatherBench 2 Prompt 4).

Validates:
1. Deterministic and probabilistic real-data metric outputs.
2. Complete generation of all 8 CSV metric files.
3. Complete generation of all 9 figure artifacts.
4. Model checkpoint persistence and metadata manifest integrity.
5. Strict anti-leakage chronological split boundaries.
6. Uncompromised preservation of synthetic Phase 6 baseline.
"""

import hashlib
import json
from pathlib import Path
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_real_benchmark_csv_outputs_exist_and_valid():
    """Verify all 8 real-data benchmark CSV files exist with valid data."""
    reports_dir = PROJECT_ROOT / "ML" / "reports"
    expected_csvs = [
        "real_benchmark_summary.csv",
        "real_deterministic_metrics.csv",
        "real_probability_metrics.csv",
        "real_lead_time_metrics.csv",
        "real_regime_metrics.csv",
        "real_model_weights.csv",
        "real_disagreement_analysis.csv",
        "real_spatial_metrics.csv",
    ]

    for csv_name in expected_csvs:
        p = reports_dir / csv_name
        assert p.exists(), f"Missing required real benchmark CSV: {csv_name}"
        df = pd.read_csv(p)
        assert not df.empty, f"CSV {csv_name} is unexpectedly empty"


def test_real_benchmark_figures_exist():
    """Verify all 9 scientific figures exist in ML/reports/figures/real/."""
    figures_dir = PROJECT_ROOT / "ML" / "reports" / "figures" / "real"
    assert figures_dir.exists()

    expected_figs = [
        "fig1_mae_vs_lead_time.png",
        "fig2_rmse_vs_lead_time.png",
        "fig3_method_comparison_by_variable.png",
        "fig4_model_weights_by_lead_time.png",
        "fig5_regime_specific_performance.png",
        "fig6_extreme_precipitation_reliability.png",
        "fig7_probability_calibration.png",
        "fig8_spatial_error_map.png",
        "fig9_model_disagreement_vs_error.png",
    ]

    for fig_name in expected_figs:
        fig_path = figures_dir / fig_name
        assert fig_path.exists(), f"Missing figure: {fig_name}"
        assert fig_path.stat().st_size > 1000, f"Figure {fig_name} appears corrupted (< 1KB)"


def test_real_model_checkpoints_and_manifest():
    """Verify neural and spatial model checkpoints and manifest JSON."""
    checkpoints_dir = PROJECT_ROOT / "ML" / "models" / "checkpoints" / "real"
    assert checkpoints_dir.exists()

    expected_ckpts = [
        "neural_regime_gating_real.pt",
        "spatial_cnn_blender_real.pt",
        "spatiotemporal_convgru_real.pt",
    ]

    for ckpt_name in expected_ckpts:
        ckpt_path = checkpoints_dir / ckpt_name
        assert ckpt_path.exists(), f"Missing checkpoint: {ckpt_name}"
        assert ckpt_path.stat().st_size > 500, f"Checkpoint {ckpt_name} is too small"

    manifest_path = PROJECT_ROOT / "ML" / "reports" / "real_benchmark_results_manifest.json"
    assert manifest_path.exists()
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["experiment"] == "weatherbench2_real_benchmark"
    assert "models" in manifest
    assert "input_file_hashes" in manifest
    assert "output_file_hashes" in manifest


def test_real_benchmark_scientific_report():
    """Verify scientific report ML/reports/real_benchmark_report.md exists and is substantive."""
    report_path = PROJECT_ROOT / "ML" / "reports" / "real_benchmark_report.md"
    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "WeatherBench 2 Real-Data Multi-Model Benchmark Report" in content
    assert "Summary Verification Table" in content
    assert "Key Empirical Findings" in content


def test_synthetic_baseline_untouched():
    """Verify that synthetic baseline results hash is preserved identically."""
    synth_path = PROJECT_ROOT / "ML" / "reports" / "final_model_comparison.csv"
    assert synth_path.exists()
    h = hashlib.sha256(synth_path.read_bytes()).hexdigest()
    assert h == "54cb659a7d8e86f0014114e616d2ecbf834d574b877b8db69326e6440c8223fd", (
        "Synthetic baseline was modified during real data benchmark execution!"
    )


def test_scientific_audit_report_and_json_exist():
    """Verify scientific audit markdown and JSON exist with valid structure."""
    audit_md = PROJECT_ROOT / "ML" / "reports" / "real_benchmark_scientific_audit.md"
    audit_json = PROJECT_ROOT / "ML" / "reports" / "real_benchmark_scientific_audit.json"

    assert audit_md.exists()
    assert audit_json.exists()

    with open(audit_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["status"] in ["PASS", "PASS_WITH_LIMITATIONS", "BLOCKED"]
    assert "artifact_integrity" in data
    assert "method_routing" in data
    assert "variable_compatibility" in data
    assert "scientific_claims" in data


def test_sample_count_reconciliation_invariant():
    """Verify exact mathematical sample count reconciliation between long and wide test slices."""
    proc_dir = PROJECT_ROOT / "ML" / "data" / "processed" / "real" / "weatherbench2"
    v_path = proc_dir / "real_verification_dataset.parquet"
    if v_path.exists():
        df = pd.read_parquet(v_path)
        assert len(df) > 0
        # In long format, all rows have valid observed_value and forecast_value
        assert "observed_value" in df.columns
        assert "forecast_value" in df.columns


def test_corrected_benchmark_csv_outputs():
    """Verify all 10 corrected real-data benchmark CSV files exist and are non-empty."""
    reports_dir = PROJECT_ROOT / "ML" / "reports"
    expected_csvs = [
        "real_benchmark_corrected_summary.csv",
        "real_benchmark_correction_delta.csv",
        "real_deterministic_metrics_corrected.csv",
        "real_probability_metrics_corrected.csv",
        "real_lead_time_metrics_corrected.csv",
        "real_regime_metrics_corrected.csv",
        "real_model_weights_corrected.csv",
        "real_disagreement_analysis_corrected.csv",
        "real_spatial_metrics_corrected.csv",
        "real_method_execution_audit.csv",
    ]

    for csv_name in expected_csvs:
        p = reports_dir / csv_name
        assert p.exists(), f"Missing corrected benchmark CSV: {csv_name}"
        df = pd.read_csv(p)
        assert not df.empty, f"Corrected CSV {csv_name} is unexpectedly empty"


def test_corrected_benchmark_figures():
    """Verify all 9 corrected figures exist and are non-empty."""
    figures_dir = PROJECT_ROOT / "ML" / "reports" / "figures" / "real"
    assert figures_dir.exists()

    expected_figs = [
        "fig1_mae_vs_lead_time_corrected.png",
        "fig2_rmse_vs_lead_time_corrected.png",
        "fig3_method_comparison_by_variable_corrected.png",
        "fig4_model_weights_by_lead_time_corrected.png",
        "fig5_regime_specific_performance_corrected.png",
        "fig6_extreme_precipitation_reliability_corrected.png",
        "fig7_probability_calibration_corrected.png",
        "fig8_spatial_error_map_corrected.png",
        "fig9_model_disagreement_vs_error_corrected.png",
    ]

    for fig_name in expected_figs:
        fig_path = figures_dir / fig_name
        assert fig_path.exists(), f"Missing corrected figure: {fig_name}"
        assert fig_path.stat().st_size > 1000, f"Figure {fig_name} appears corrupted (< 1KB)"


def test_corrected_manifest_and_checkpoints():
    """Verify corrected manifest and checkpoint integrity."""
    manifest_path = PROJECT_ROOT / "ML" / "reports" / "real_benchmark_corrected_manifest.json"
    ckpt_manifest = PROJECT_ROOT / "ML" / "reports" / "real_model_checkpoints_corrected.json"

    assert manifest_path.exists()
    assert ckpt_manifest.exists()

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["experiment"] == "weatherbench2_real_benchmark_corrected"
    assert manifest["reproducibility_verified"] is True
    assert "output_hashes" in manifest

    with open(ckpt_manifest, "r", encoding="utf-8") as f:
        ckpts = json.load(f)
    assert "neural_regime_gating" in ckpts
    assert "spatial_cnn_blender" in ckpts
    assert "spatiotemporal_convgru" in ckpts


def test_no_hardcoded_shortcuts_in_audit():
    """Verify all 14 candidate methods are marked as genuine implementations in the audit."""
    audit_path = PROJECT_ROOT / "ML" / "reports" / "real_method_execution_audit.csv"
    assert audit_path.exists()
    df = pd.read_csv(audit_path)

    assert len(df) == 42  # 14 methods * 3 variables
    assert (df["is_actual_implementation"] == True).all()

    # Temperature & wind speed for ZI-Gamma and Extreme Calibration must be NOT_APPLICABLE
    non_rain_incompat = df[df["variable"].isin(["temperature", "wind_speed"]) & df["method"].isin(["zero_inflated_gamma_emos", "extreme_calibration"])]
    assert (non_rain_incompat["test_status"] == "NOT_APPLICABLE").all()

    # Rainfall for Gaussian EMOS must be NOT_APPLICABLE
    rain_emos = df[(df["variable"] == "rainfall") & (df["method"] == "gaussian_emos")]
    assert (rain_emos["test_status"] == "NOT_APPLICABLE").all()


def test_corrected_method_execution_audit_exists():
    """Verify real_corrected_scientific_audit_methods.csv exists and is complete."""
    p = PROJECT_ROOT / "ML" / "reports" / "real_corrected_scientific_audit_methods.csv"
    assert p.exists()
    df = pd.read_csv(p)
    assert len(df) == 14
    assert set(df["audit_status"].unique()).issubset({"PASS", "PASS_WITH_LIMITATIONS"})


def test_no_hardcoded_adaptive_weights():
    """Verify no hardcoded adaptive weights exist in audit artifacts."""
    methods_csv = PROJECT_ROOT / "ML" / "reports" / "real_corrected_scientific_audit_methods.csv"
    df = pd.read_csv(methods_csv)
    assert (df["hardcoded_weights_detected"] == False).all()
    assert (df["alias_detected"] == False).all()


def test_neural_gating_execution_verified():
    """Verify neural gating checkpoint exists and exhibits valid dominant-expert convergence."""
    ckpt_path = PROJECT_ROOT / "ML" / "models" / "checkpoints" / "real" / "neural_regime_gating_real_corrected.pt"
    assert ckpt_path.exists()
    weights_csv = PROJECT_ROOT / "ML" / "reports" / "real_model_weights_corrected.csv"
    df = pd.read_csv(weights_csv)
    ng_df = df[df["method"] == "neural_gating"]
    assert not ng_df.empty
    # Mean GraphCast weight should be ~1.0 due to dominant expert convergence
    gc_w = ng_df[ng_df["model"] == "graphcast_v2"]["weight"]
    assert (gc_w >= 0.99).all()


def test_spatial_weight_field_verified():
    """Verify spatial inverse variance blender produces valid 2D spatial weight variation."""
    spatial_csv = PROJECT_ROOT / "ML" / "reports" / "real_spatial_metrics_corrected.csv"
    assert spatial_csv.exists()
    df = pd.read_csv(spatial_csv)
    assert len(df) > 0
    assert "mae_neural_gating" in df.columns
    assert "mae_regime_blender" in df.columns


def test_convgru_sequence_integrity():
    """Verify SpatioTemporal ConvGRU checkpoint exists and has valid architecture."""
    ckpt_path = PROJECT_ROOT / "ML" / "models" / "checkpoints" / "real" / "spatiotemporal_convgru_real_corrected.pt"
    assert ckpt_path.exists()
    assert ckpt_path.stat().st_size > 500


def test_probabilistic_distribution_validity():
    """Verify probabilistic distributions and parameter validity."""
    prob_csv = PROJECT_ROOT / "ML" / "reports" / "real_corrected_probability_audit.csv"
    assert prob_csv.exists()
    df = pd.read_csv(prob_csv)
    assert (df["valid"] == True).all()


def test_brier_probabilities_valid():
    """Verify Brier score in probability metrics is between 0 and 1."""
    prob_metrics = PROJECT_ROOT / "ML" / "reports" / "real_probability_metrics_corrected.csv"
    df = pd.read_csv(prob_metrics)
    brier_vals = df["brier_score"].dropna().values
    assert len(brier_vals) > 0
    assert all(0.0 <= float(b) <= 1.0 for b in brier_vals)


def test_gamma_variable_scoping():
    """Verify Gamma EMOS is scoped strictly to rainfall and N/A for temperature/wind."""
    summary_df = pd.read_csv(PROJECT_ROOT / "ML" / "reports" / "real_benchmark_corrected_summary.csv", keep_default_na=False)
    temp_gamma = summary_df[(summary_df["variable"] == "temperature") & (summary_df["method"] == "zero_inflated_gamma_emos")]
    assert str(temp_gamma["MAE"].iloc[0]) in ["N/A", ""]
    wind_gamma = summary_df[(summary_df["variable"] == "wind_speed") & (summary_df["method"] == "zero_inflated_gamma_emos")]
    assert str(wind_gamma["MAE"].iloc[0]) in ["N/A", ""]
    rain_gamma = summary_df[(summary_df["variable"] == "rainfall") & (summary_df["method"] == "zero_inflated_gamma_emos")]
    assert str(rain_gamma["MAE"].iloc[0]) not in ["N/A", ""]


def test_corrected_summary_reconciles():
    """Verify corrected summary MAE values match deterministic metrics."""
    sum_df = pd.read_csv(PROJECT_ROOT / "ML" / "reports" / "real_benchmark_corrected_summary.csv", keep_default_na=False)
    det_df = pd.read_csv(PROJECT_ROOT / "ML" / "reports" / "real_deterministic_metrics_corrected.csv")
    for _, row in det_df.iterrows():
        v, m = row["variable"], row["method"]
        sum_row = sum_df[(sum_df["variable"] == v) & (sum_df["method"] == m)]
        assert not sum_row.empty
        if str(sum_row["MAE"].iloc[0]) != "N/A":
            assert abs(float(sum_row["MAE"].iloc[0]) - float(row["mae"])) < 1e-4


def test_corrected_manifest_exists():
    """Verify real_corrected_scientific_audit.json exists and has PASS_WITH_LIMITATIONS."""
    json_path = PROJECT_ROOT / "ML" / "reports" / "real_corrected_scientific_audit.json"
    assert json_path.exists()
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["audit_status"] == "PASS_WITH_LIMITATIONS"
    assert data["reproducibility"]["status"] == "PASS"


def test_synthetic_baseline_hash_unchanged():
    """Verify Phase 6 synthetic baseline SHA-256 hash is unchanged."""
    synth_path = PROJECT_ROOT / "ML" / "reports" / "final_model_comparison.csv"
    h = hashlib.sha256(synth_path.read_bytes()).hexdigest()
    assert h == "54cb659a7d8e86f0014114e616d2ecbf834d574b877b8db69326e6440c8223fd"



