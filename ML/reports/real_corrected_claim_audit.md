# Scientific Claim-by-Claim Integrity Audit

**Document Under Audit**: `ML/reports/real_benchmark_corrected_report.md`  
**Audit Phase**: Phase 5.5 Independent Scientific Audit  
**Evaluation**: Evidence-based claim categorization based on genuine corrected benchmark execution.  

---

## Summary Table of Claim Verifications

| Claim # | Benchmark Claim Statement | Audit Status | Evidence & Quantitative Justification |
| :---: | :--- | :---: | :--- |
| **1** | *GraphCast v2 significantly outperforms ECMWF HRES across all 3 variables in South Asia.* | **SUPPORTED** | GraphCast MAE is 47.0% lower on Temperature (0.4770 vs 0.9003 °C), 20.7% lower on Wind (0.3413 vs 0.4305 m/s), and 45.9% lower on Rain (0.4268 vs 0.7889 mm). |
| **2** | *Equal weighting baseline (50/50 blend) improves forecast skill over single best model.* | **CONTRADICTED** | Equal weighting is worse than GraphCast v2 alone on Temperature (0.5994 vs 0.4770 °C) and Rainfall (0.5569 vs 0.4268 mm), because ECMWF HRES error is substantially larger. Only on Wind Speed does equal weight (0.3114) slightly improve over GraphCast (0.3413). |
| **3** | *Adaptive Skill Weighting (Inverse-Variance, Inverse-MAE, Ridge) dynamically shifts weight to the more accurate model.* | **SUPPORTED** | Learned weights on Train dynamically assign higher weight to GraphCast v2 (W_GC ≈ 0.73 on Temp, 0.58 on Wind, 0.76 on Rain), outperforming equal weighting across all variables. |
| **4** | *Neural Gating produces dramatic forecast improvements over GraphCast.* | **CONTRADICTED / PARTIALLY_SUPPORTED** | Neural Gating test MAE is exactly equal to GraphCast v2 (0.4770, 0.3413, 0.4268). The network learned to route virtually 100% of weight to GraphCast (W_GC = 0.999999). It matched GraphCast but did not exceed it. |
| **5** | *Spatio-Temporal ConvGRU achieves the lowest overall RMSE on 2m Temperature.* | **SUPPORTED** | ConvGRU achieves RMSE 0.6760 °C (MAE 0.4722 °C), achieving a statistically established ~1.4% improvement over GraphCast v2 (RMSE 0.6856 °C). |
| **6** | *Spatial CNN blender outperforms simple spatial baselines for surface wind speed.* | **SUPPORTED** | Spatial CNN achieves MAE 0.3043 m/s vs Equal Weight 0.3114 m/s and GraphCast 0.3413 m/s, capturing coastal vs plains error variance patterns. |
| **7** | *Gaussian EMOS provides well-calibrated probabilistic forecasts for temperature and wind.* | **SUPPORTED** | Gaussian EMOS achieves 90% PICP of 89.54% (Temp) and 89.82% (Wind) with nominal 90% target, and sharp MPIW intervals (2.68 °C, 1.51 m/s). |
| **8** | *Zero-Inflated Gamma EMOS is calibrated for strictly non-negative precipitation.* | **SUPPORTED** | Parametric Zero-Inflated Gamma EMOS achieves CRPS of 0.4156 mm and Brier score 0.0382 for heavy rain exceedances (P >= 5mm), avoiding negative support artifacts. |
| **9** | *Model disagreement (|HRES - GraphCast|) reliably correlates with blend forecast error.* | **PARTIALLY_SUPPORTED** | Correlation is strong on Rainfall (r = 0.5310, p < 1e-50) and Temperature (r = 0.4498, p < 1e-35), but moderate on Wind Speed (r = 0.2858, p = 3.6e-16). Claims of universal r > 0.35 are refined to reflect wind's lower correlation. |
| **10** | *The benchmark is 100% bit-for-bit reproducible and isolated from synthetic data.* | **SUPPORTED** | Duplicate benchmark executions with seed=42 produce bit-for-bit identical hashes across all 10 CSVs, JSON manifests, and checkpoints. Synthetic baseline SHA-256 hash remains unchanged. |

---