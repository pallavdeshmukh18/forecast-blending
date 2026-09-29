"""
Command-line script to generate deterministic synthetic weather observations and forecasts.
"""

import argparse
import sys
from pathlib import Path

# Add repo root to sys.path for direct script execution
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs import get_config
from ingestion.synthetic import SyntheticDataGenerator


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic meteorological dataset.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed (default: 42)")
    parser.add_argument("--save", action="store_true", default=True, help="Persist datasets to data/ directories")
    args = parser.parse_args()

    cfg = get_config()
    print(f"Generating synthetic dataset using seed={args.seed}...")
    generator = SyntheticDataGenerator(cfg)
    obs_df, forecasts_df = generator.generate(seed=args.seed, save_to_disk=args.save)

    print(f"Successfully generated:")
    print(f"  - Ground truth observations: {len(obs_df):,} records -> {cfg.paths.synthetic_observations_file}")
    print(f"  - Multi-model forecasts:     {len(forecasts_df):,} records -> {cfg.paths.synthetic_forecasts_file}")
    print(f"  - Unique models:             {list(forecasts_df['model_name'].unique())}")
    print(f"  - Lead times (hours):        {sorted(list(forecasts_df['lead_time'].unique()))}")
    print(f"  - Variables:                 {list(obs_df['variable'].unique())}")


if __name__ == "__main__":
    main()
