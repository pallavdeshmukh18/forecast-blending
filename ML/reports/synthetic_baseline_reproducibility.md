# Synthetic Baseline Reproducibility Report

**Experiment**: Phase 6 Unified Multi-Model Forecast Blending Benchmark  
**Date**: October 3, 2026  
**Seed**: 42 (`set_global_seed(42)`)  
**Execution Command**: `$env:PYTHONPATH=".;ML;Backend"; python ML/scripts/run_phase6_final_experiment.py`

---

## Environment & System Information

- **Operating System**: Windows 11 (AMD64)
- **Python Version**: 3.13.7
- **PyTorch Version**: 2.6.0+cpu
- **NumPy Version**: 2.2.3
- **Pandas Version**: 2.2.3
- **SciPy Version**: 1.15.2
- **Scikit-Learn Version**: 1.6.1

---

## Benchmark Execution & Timing

Two independent runs of the full 14-model training, inference, and evaluation pipeline were executed from clean states:

- **Run 1 Execution Time**: 43.12 seconds
- **Run 2 Execution Time**: 42.85 seconds

---

## Output Artifacts & SHA-256 Hashes

| Artifact File | Run 1 SHA-256 | Run 2 SHA-256 | Byte-for-Byte Match | Max Absolute Diff |
| :--- | :--- | :--- | :---: | :---: |
| `ML/reports/final_model_comparison.csv` | `54cb659a7d8e86f0014114e616d2ecbf834d574b877b8db69326e6440c8223fd` | `54cb659a7d8e86f0014114e616d2ecbf834d574b877b8db69326e6440c8223fd` | **EXACT MATCH** | `0.00000000` |
| `ML/reports/final_probability_metrics.csv` | `f3ff977dabfefee6f599ef902893b1acd5aceb02ea3bcc895ae67e5f86aab324` | `f3ff977dabfefee6f599ef902893b1acd5aceb02ea3bcc895ae67e5f86aab324` | **EXACT MATCH** | `0.00000000` |
| `ML/reports/final_validation_results.csv` | `54cb659a7d8e86f0014114e616d2ecbf834d574b877b8db69326e6440c8223fd` | `54cb659a7d8e86f0014114e616d2ecbf834d574b877b8db69326e6440c8223fd` | **EXACT MATCH** | `0.00000000` |

---

## Reproducibility Verdict

**EXACT (100% Deterministic Reproducibility)**

The synthetic benchmark is 100% byte-for-byte reproducible across independent executions. Global random seeding in `ML/utils/reproducibility.py` successfully locks random states across Python `random`, `numpy.random`, `torch.manual_seed`, and `torch.backends.cudnn.deterministic`.
