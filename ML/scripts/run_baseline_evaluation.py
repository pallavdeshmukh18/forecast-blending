"""
Command-line script to run multi-model baseline forecast evaluation and skill benchmarking.
"""

import argparse
import sys
from pathlib import Path

# Add repo root to sys.path for direct script execution
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from configs import get_config
from ingestion.synthetic import SyntheticDataGenerator
from models.baseline import equal_weight_blend
from preprocessing.alignment import align_forecasts_and_observations
from evaluation.comparison import compare_models, compute_skill_rankings


def main():
    parser = argparse.ArgumentParser(description="Execute baseline multi-model forecast evaluation.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for synthetic generation (if files not cached)")
    parser.add_argument("--metric", type=str, default="rmse", choices=["rmse", "mae", "bias", "correlation"], help="Ranking metric")
    args = parser.parse_args()

    cfg = get_config()
    print("=" * 70)
    print(" Hybrid AI–NWP Multi-Model Forecast Blending System: Baseline Evaluation")
    print("=" * 70)

    # 1. Ingest Data
    generator = SyntheticDataGenerator(cfg)
    obs_df, forecasts_df = generator.generate(seed=args.seed)
    print(f"Loaded {len(obs_df):,} observations and {len(forecasts_df):,} forecast records.")

    # 2. Compute Equal-Weight Blend
    print("Calculating baseline equal-weight multi-model blend...")
    blended_df = equal_weight_blend(forecasts_df)
    combined_forecasts = pd.concat([forecasts_df, blended_df], ignore_index=True)

    # 3. Align with Ground Truth
    aligned_df = align_forecasts_and_observations(combined_forecasts, obs_df)

    # 4. Evaluate Models Across Lead Times
    comparison_df = compare_models(aligned_df, group_by=["variable", "lead_time"])
    ranked_df = compute_skill_rankings(comparison_df, metric=args.metric)

    print("\n" + "=" * 70)
    print(f" Multi-Model Skill Benchmark Ranked by {args.metric.upper()} (Lower is Better for RMSE/MAE)")
    print("=" * 70)

    for var in sorted(ranked_df["variable"].unique()):
        print(f"\n--- Variable: {var.upper()} ---")
        sub = ranked_df[ranked_df["variable"] == var]
        cols_to_show = ["lead_time", "rank", "model_name", "mae", "rmse", "bias", "correlation", "sample_count"]
        print(sub[cols_to_show].to_string(index=False))

    # 5. Overall summary by variable
    print("\n" + "=" * 70)
    print(" Overall Aggregate Verification Metrics Across All Lead Times")
    print("=" * 70)
    overall_df = compare_models(aligned_df, group_by=["variable"])
    print(overall_df.to_string(index=False))


if __name__ == "__main__":
    main()
