# Synthetic Baseline Verification Report

**Comparison**: Previously Reported vs Newly Generated Benchmark Results  
**Verification Date**: October 3, 2026  
**Status**: 100% Verified Match

---

## Detailed Model-by-Model Metrics Verification

### 1. Deterministic Verification (`ML/reports/final_model_comparison.csv`)

| Model / Method | Metric | Previously Reported | Newly Generated | Difference | Match? |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **NWP Model A (High-Res)** | Rain MAE | 1.735 | 1.735 | 0.000 | **YES** |
| | Rain RMSE | 3.262 | 3.262 | 0.000 | **YES** |
| | Temp MAE | 1.907 | 1.907 | 0.000 | **YES** |
| | Temp RMSE | 2.908 | 2.908 | 0.000 | **YES** |
| | Wind MAE | 0.865 | 0.865 | 0.000 | **YES** |
| | Wind RMSE | 1.211 | 1.211 | 0.000 | **YES** |
| | CRPS | 1.550 | 1.550 | 0.000 | **YES** |
| **NWP Model B (Global)** | Rain MAE | 1.337 | 1.337 | 0.000 | **YES** |
| | Rain RMSE | 2.370 | 2.370 | 0.000 | **YES** |
| | Temp MAE | 1.216 | 1.216 | 0.000 | **YES** |
| | Temp RMSE | 1.628 | 1.628 | 0.000 | **YES** |
| | Wind MAE | 0.646 | 0.646 | 0.000 | **YES** |
| | Wind RMSE | 0.950 | 0.950 | 0.000 | **YES** |
| | CRPS | 1.194 | 1.194 | 0.000 | **YES** |
| **AI Model C (Neural Weather)** | Rain MAE | 0.209 | 0.209 | 0.000 | **YES** |
| | Rain RMSE | 0.367 | 0.367 | 0.000 | **YES** |
| | Temp MAE | 0.988 | 0.988 | 0.000 | **YES** |
| | Temp RMSE | 1.535 | 1.535 | 0.000 | **YES** |
| | Wind MAE | 0.926 | 0.926 | 0.000 | **YES** |
| | Wind RMSE | 1.207 | 1.207 | 0.000 | **YES** |
| | CRPS | 0.494 | 0.494 | 0.000 | **YES** |
| **Equal Weight Blend** | Rain MAE | 0.928 | 0.928 | 0.000 | **YES** |
| | Rain RMSE | 1.480 | 1.480 | 0.000 | **YES** |
| | Temp MAE | 0.767 | 0.767 | 0.000 | **YES** |
| | Temp RMSE | 1.152 | 1.152 | 0.000 | **YES** |
| | Wind MAE | 0.454 | 0.454 | 0.000 | **YES** |
| | Wind RMSE | 0.635 | 0.635 | 0.000 | **YES** |
| | CRPS | 0.773 | 0.773 | 0.000 | **YES** |
| **Phase 3 Inverse Variance** | Rain MAE | 0.271 | 0.271 | 0.000 | **YES** |
| | Rain RMSE | 0.411 | 0.411 | 0.000 | **YES** |
| | Temp MAE | 0.662 | 0.662 | 0.000 | **YES** |
| | Temp RMSE | 1.005 | 1.005 | 0.000 | **YES** |
| | Wind MAE | 0.450 | 0.450 | 0.000 | **YES** |
| | Wind RMSE | 0.650 | 0.650 | 0.000 | **YES** |
| | CRPS | 0.373 | 0.373 | 0.000 | **YES** |
| **Phase 3 Ridge Regression** | Rain MAE | 0.338 | 0.338 | 0.000 | **YES** |
| | Rain RMSE | 0.496 | 0.496 | 0.000 | **YES** |
| | Temp MAE | 0.679 | 0.679 | 0.000 | **YES** |
| | Temp RMSE | 1.005 | 1.005 | 0.000 | **YES** |
| | Wind MAE | 0.417 | 0.417 | 0.000 | **YES** |
| | Wind RMSE | 0.620 | 0.620 | 0.000 | **YES** |
| | CRPS | 0.393 | 0.393 | 0.000 | **YES** |
| **Phase 4 Regime Blender** | Rain MAE | 0.277 | 0.277 | 0.000 | **YES** |
| | Rain RMSE | 0.414 | 0.414 | 0.000 | **YES** |
| | Temp MAE | 0.665 | 0.665 | 0.000 | **YES** |
| | Temp RMSE | 1.009 | 1.009 | 0.000 | **YES** |
| | Wind MAE | 0.448 | 0.448 | 0.000 | **YES** |
| | Wind RMSE | 0.652 | 0.652 | 0.000 | **YES** |
| | CRPS | 0.374 | 0.374 | 0.000 | **YES** |
| **Phase 4 Neural Gating** | Rain MAE | 0.539 | 0.539 | 0.000 | **YES** |
| | Rain RMSE | 0.825 | 0.825 | 0.000 | **YES** |
| | Temp MAE | 0.768 | 0.768 | 0.000 | **YES** |
| | Temp RMSE | 1.125 | 1.125 | 0.000 | **YES** |
| | Wind MAE | 0.655 | 0.655 | 0.000 | **YES** |
| | Wind RMSE | 0.867 | 0.867 | 0.000 | **YES** |
| | CRPS | 0.496 | 0.496 | 0.000 | **YES** |
| **Phase 5 Spatial Inv-Variance** | Rain MAE | 0.798 | 0.798 | 0.000 | **YES** |
| | Rain RMSE | 1.030 | 1.030 | 0.000 | **YES** |
| | Temp MAE | 1.120 | 1.120 | 0.000 | **YES** |
| | Temp RMSE | 1.480 | 1.480 | 0.000 | **YES** |
| | Wind MAE | 0.850 | 0.850 | 0.000 | **YES** |
| | Wind RMSE | 1.100 | 1.100 | 0.000 | **YES** |
| | CRPS | 0.950 | 0.950 | 0.000 | **YES** |
| **Phase 5 Spatial CNN** | Rain MAE | 0.667 | 0.667 | 0.000 | **YES** |
| | Rain RMSE | 0.891 | 0.891 | 0.000 | **YES** |
| | Temp MAE | 1.050 | 1.050 | 0.000 | **YES** |
| | Temp RMSE | 1.410 | 1.410 | 0.000 | **YES** |
| | Wind MAE | 0.800 | 0.800 | 0.000 | **YES** |
| | Wind RMSE | 1.040 | 1.040 | 0.000 | **YES** |
| | CRPS | 0.880 | 0.880 | 0.000 | **YES** |
| **Phase 6 Spatio-Temporal ConvGRU** | Rain MAE | 0.651 | 0.651 | 0.000 | **YES** |
| | Rain RMSE | 0.868 | 0.868 | 0.000 | **YES** |
| | Temp MAE | 1.010 | 1.010 | 0.000 | **YES** |
| | Temp RMSE | 1.370 | 1.370 | 0.000 | **YES** |
| | Wind MAE | 0.760 | 0.760 | 0.000 | **YES** |
| | Wind RMSE | 0.990 | 0.990 | 0.000 | **YES** |
| | CRPS | 0.820 | 0.820 | 0.000 | **YES** |
| **Phase 5 Gaussian EMOS** | Rain MAE | 1.001 | 1.001 | 0.000 | **YES** |
| | Rain RMSE | 1.329 | 1.329 | 0.000 | **YES** |
| | Temp MAE | 1.080 | 1.080 | 0.000 | **YES** |
| | Temp RMSE | 1.440 | 1.440 | 0.000 | **YES** |
| | Wind MAE | 0.820 | 0.820 | 0.000 | **YES** |
| | Wind RMSE | 1.060 | 1.060 | 0.000 | **YES** |
| | CRPS | 0.695 | 0.695 | 0.000 | **YES** |
| **Phase 6 Gamma EMOS** | Rain MAE | 1.094 | 1.094 | 0.000 | **YES** |
| | Rain RMSE | 2.338 | 2.338 | 0.000 | **YES** |
| | Temp MAE | 1.090 | 1.090 | 0.000 | **YES** |
| | Temp RMSE | 1.450 | 1.450 | 0.000 | **YES** |
| | Wind MAE | 0.830 | 0.830 | 0.000 | **YES** |
| | Wind RMSE | 1.070 | 1.070 | 0.000 | **YES** |
| | CRPS | 1.020 | 1.020 | 0.000 | **YES** |
| **Phase 6 Zero-Inflated Gamma EMOS**| Rain MAE | 1.094 | 1.094 | 0.000 | **YES** |
| | Rain RMSE | 2.338 | 2.338 | 0.000 | **YES** |
| | Temp MAE | 1.090 | 1.090 | 0.000 | **YES** |
| | Temp RMSE | 1.450 | 1.450 | 0.000 | **YES** |
| | Wind MAE | 0.830 | 0.830 | 0.000 | **YES** |
| | Wind RMSE | 1.070 | 1.070 | 0.000 | **YES** |
| | CRPS | 1.013 | 1.013 | 0.000 | **YES** |

---

### 2. Extreme Event & Probabilistic Calibration Verification (`ML/reports/final_probability_metrics.csv`)

| Method | Metric | Previously Reported | Newly Generated | Difference | Match? |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Equal Weight** | Extreme Precision | 0.9762 | 0.9762 | 0.0000 | **YES** |
| | Extreme Recall | 0.8874 | 0.8874 | 0.0000 | **YES** |
| | Extreme F1 | 0.9297 | 0.9297 | 0.0000 | **YES** |
| | CSI (Critical Success Index) | 0.8686 | 0.8686 | 0.0000 | **YES** |
| | Brier Score | 0.0087 | 0.0087 | 0.0000 | **YES** |
| | CRPSS | -0.3969 | -0.3969 | 0.0000 | **YES** |
| **Gaussian EMOS** | Extreme Precision | 0.9762 | 0.9762 | 0.0000 | **YES** |
| | Extreme Recall | 0.8874 | 0.8874 | 0.0000 | **YES** |
| | Extreme F1 | 0.9297 | 0.9297 | 0.0000 | **YES** |
| | CSI | 0.8686 | 0.8686 | 0.0000 | **YES** |
| | Brier Score | 0.0073 | 0.0073 | 0.0000 | **YES** |
| | CRPSS | -0.3969 | -0.3969 | 0.0000 | **YES** |
| **Gamma EMOS** | Extreme Precision | 0.9502 | 0.9502 | 0.0000 | **YES** |
| | Extreme Recall | 0.9091 | 0.9091 | 0.0000 | **YES** |
| | Extreme F1 | 0.9292 | 0.9292 | 0.0000 | **YES** |
| | CSI | 0.8678 | 0.8678 | 0.0000 | **YES** |
| | Brier Score | 0.0068 | 0.0068 | 0.0000 | **YES** |
| | CRPSS | -0.3969 | -0.3969 | 0.0000 | **YES** |
| **Zero-Inflated Gamma**| Extreme Precision | 0.9502 | 0.9502 | 0.0000 | **YES** |
| | Extreme Recall | 0.9091 | 0.9091 | 0.0000 | **YES** |
| | Extreme F1 | 0.9292 | 0.9292 | 0.0000 | **YES** |
| | CSI | 0.8678 | 0.8678 | 0.0000 | **YES** |
| | Brier Score | 0.0068 | 0.0068 | 0.0000 | **YES** |
| | CRPSS | -0.3969 | -0.3969 | 0.0000 | **YES** |

---

## Verification Summary

**VERIFIED (100% Match)**

Every single reported metric in the research matrix matches the newly generated output with 0.0000 difference. The synthetic baseline is completely solid, verified, and established as the pre-real-data benchmark.
