"""
Unit tests for lead-time binning and boundary condition verification.
"""

import pandas as pd
from preprocessing.features import add_lead_time_bins


def test_lead_time_bins_boundary_conditions():
    # Test boundary values: 0, 24, 25, 72, 73, 120, 121, 240
    df = pd.DataFrame({
        "lead_time": [0, 6, 24, 25, 48, 72, 73, 96, 120, 121, 240],
    })

    binned = add_lead_time_bins(df)

    expected = [
        "short",      # 0
        "short",      # 6
        "short",      # 24 (inclusive upper bound)
        "medium",     # 25 (> 24)
        "medium",     # 48
        "medium",     # 72 (inclusive upper bound)
        "extended",   # 73 (> 72)
        "extended",   # 96
        "extended",   # 120 (inclusive upper bound)
        "long_range", # 121 (> 120)
        "long_range", # 240
    ]

    assert list(binned["lead_time_bin"]) == expected
