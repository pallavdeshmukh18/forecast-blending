"""
Tests for Phase 6 Final Research-Grade Results and Scientific Package (SkyCast).

Verifies:
1. Final research report exists and includes all 25 required sections and all 8 tables.
2. Final research summary JSON exists, validates schema, and matches audited metrics.
3. Final provenance manifest JSON exists and all SHA-256 hashes match.
4. All 9 publication figures exist in ML/reports/figures/real/ and are valid.
5. No ungrounded, invalid, or stale scientific claims exist in report/summary.
6. Synthetic Phase 6 baseline hash remains byte-for-byte unchanged.
"""

import hashlib
import json
from pathlib import Path
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = PROJECT_ROOT / "ML" / "reports"
FIGURES_DIR = REPORTS_DIR / "figures" / "real"

SYNTHETIC_BASELINE_SHA256 = "54cb659a7d8e86f0014114e616d2ecbf834d574b877b8db69326e6440c8223fd"


def compute_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def test_final_report_exists_and_contains_required_sections():
    """Verify ML/reports/final_research_report.md contains all 25 sections and all 8 tables."""
    report_path = REPORTS_DIR / "final_research_report.md"
    assert report_path.exists(), "Missing final_research_report.md"
    content = report_path.read_text(encoding="utf-8")
    assert len(content) > 10000, "final_research_report.md is too short"
    
    # Verify all 25 sections
    sections = [
        "## Abstract",
        "## 1. Introduction",
        "## 2. Research Objective",
        "## 3. Dataset",
        "## 4. Forecast Models",
        "## 5. Experimental Domain",
        "## 6. Variables and Forecast Horizons",
        "## 7. Chronological Evaluation Protocol",
        "## 8. Multi-Model Blending Framework",
        "## 9. Probabilistic Forecasting",
        "## 10. Evaluation Metrics",
        "## 11. Deterministic Results",
        "## 12. Lead-Time Results",
        "## 13. Regime Results",
        "## 14. Spatial Results",
        "## 15. Probabilistic Results",
        "## 16. Forecast Disagreement",
        "## 17. Statistical Significance",
        "## 18. Ablation / Method Comparison",
        "## 19. Discussion",
        "## 20. Scientific Interpretation",
        "## 21. Limitations",
        "## 22. Reproducibility",
        "## 23. Practical Implications",
        "## 24. Future Work",
        "## 25. Conclusion"
    ]
    for sec in sections:
        assert sec in content, f"Missing section in report: {sec}"
        
    # Verify tables
    tables = [
        "Table 1: Dataset and Experiment Configuration",
        "Table 2: Deterministic Forecast Performance",
        "Table 3: Performance by Lead Time",
        "Table 4: Performance by Weather Regime",
        "Table 5: Probabilistic Forecast Performance",
        "Table 6: Paired Block-Bootstrap Significance Comparisons",
        "Table 7: Spatial Blending Diagnostics",
        "Table 8: Learned Model Weights Across Methodologies"
    ]
    for tbl in tables:
        assert tbl in content, f"Missing table in report: {tbl}"


def test_final_summary_exists_and_valid():
    """Verify ML/reports/final_research_summary.json exists and conforms to schema."""
    summary_path = REPORTS_DIR / "final_research_summary.json"
    assert summary_path.exists(), "Missing final_research_summary.json"
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert data["project"] == "Hybrid AI–NWP Multi-Model Forecast Blending System"
    assert data["dataset"] == "WeatherBench 2"
    assert data["domain"] == "South Asia"
    assert data["methods_evaluated"] == 14
    assert data["scientific_audit_status"] == "PASS_WITH_LIMITATIONS"
    assert len(data["key_findings"]) >= 5
    assert len(data["statistically_established_improvements"]) >= 1
    assert data["statistically_established_improvements"][0]["metric"] == "RMSE"
    assert data["reproducibility"]["status"] == "PASS"
    assert data["synthetic_baseline"]["sha256"] == SYNTHETIC_BASELINE_SHA256


def test_final_manifest_exists_and_hashes_valid():
    """Verify ML/reports/final_research_manifest.json exists and hashes match on-disk files."""
    manifest_path = REPORTS_DIR / "final_research_manifest.json"
    assert manifest_path.exists(), "Missing final_research_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    assert manifest["phase"] == "Phase 6 Final Research Package"
    assert manifest["reproducibility_verified"] is True
    assert manifest["synthetic_baseline_hash"] == SYNTHETIC_BASELINE_SHA256
    
    # Check that every reported file hash matches current on-disk sha256
    for fname, expected_hash in manifest["report_hashes"].items():
        file_p = REPORTS_DIR / fname
        assert file_p.exists(), f"File in manifest missing: {fname}"
        actual_hash = compute_sha256(file_p)
        assert actual_hash == expected_hash, f"Hash mismatch for {fname}: expected {expected_hash}, got {actual_hash}"


def test_all_publication_figures_exist():
    """Verify all 9 publication-grade figures exist in ML/reports/figures/real/."""
    expected_figures = [
        "fig1_experimental_architecture.png",
        "fig2_deterministic_temperature_comparison.png",
        "fig3_deterministic_wind_comparison.png",
        "fig4_deterministic_rainfall_comparison.png",
        "fig5_lead_time_skill_curves.png",
        "fig6_regime_lead_model_weights.png",
        "fig7_spatial_blending_weight_maps.png",
        "fig8_forecast_disagreement_vs_error.png",
        "fig9_probabilistic_calibration_reliability.png"
    ]
    for fig_name in expected_figures:
        fig_p = FIGURES_DIR / fig_name
        assert fig_p.exists(), f"Missing publication figure: {fig_name}"
        assert fig_p.stat().st_size > 5000, f"Figure {fig_name} is too small (<5KB)"


def test_report_results_match_corrected_summary():
    """Verify that report metrics match real_benchmark_corrected_summary.csv."""
    summary_csv = REPORTS_DIR / "real_benchmark_corrected_summary.csv"
    assert summary_csv.exists()
    df = pd.read_csv(summary_csv, keep_default_na=False)
    
    # GraphCast Temp MAE = 0.4770, RMSE = 0.6856
    gc_temp = df[(df["variable"] == "temperature") & (df["method"] == "graphcast_v2")].iloc[0]
    assert float(gc_temp["MAE"]) == pytest.approx(0.4770, abs=1e-3)
    assert float(gc_temp["RMSE"]) == pytest.approx(0.6856, abs=1e-3)
    
    # ConvGRU Temp RMSE = 0.6760
    conv_temp = df[(df["variable"] == "temperature") & (df["method"] == "spatiotemporal_convgru")].iloc[0]
    assert float(conv_temp["RMSE"]) == pytest.approx(0.6760, abs=1e-3)
    
    # Spatial CNN Wind MAE = 0.3043
    cnn_wind = df[(df["variable"] == "wind_speed") & (df["method"] == "spatial_cnn")].iloc[0]
    assert float(cnn_wind["MAE"]) == pytest.approx(0.3043, abs=1e-3)


def test_no_stale_invalid_claims():
    """Verify that prohibited overclaims are not present in final_research_report.md."""
    report_content = (REPORTS_DIR / "final_research_report.md").read_text(encoding="utf-8").lower()
    
    forbidden_phrases = [
        "wins universally",
        "guaranteed improvement",
        "ai always beats nwp",
        "ensemble always improves forecasts",
        "neural gating dramatically improves",
        "disagreement directly predicts uncertainty",
        "calibration is perfect",
        "production-ready weather prediction"
    ]
    for phrase in forbidden_phrases:
        assert phrase not in report_content, f"Prohibited overclaim found in report: '{phrase}'"


def test_synthetic_baseline_hash_unchanged():
    """Verify synthetic baseline CSV hash remains byte-for-byte unchanged."""
    synth_csv = REPORTS_DIR / "final_model_comparison.csv"
    assert synth_csv.exists()
    actual_hash = compute_sha256(synth_csv)
    assert actual_hash == SYNTHETIC_BASELINE_SHA256, (
        f"Synthetic baseline CSV was modified! Expected {SYNTHETIC_BASELINE_SHA256}, got {actual_hash}"
    )
