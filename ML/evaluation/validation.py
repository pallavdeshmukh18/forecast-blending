"""
Validation and split strategies for meteorological time series forecasting.
"""

from typing import Generator, List, Tuple
import pandas as pd


class TimeSeriesSplitter:
    """
    Chronological expanding or rolling window splitter for temporal forecast validation.
    Prevents temporal data leakage in weather forecasting experiments.
    """

    def __init__(
        self,
        n_splits: int = 4,
        min_train_ratio: float = 0.5,
        rolling: bool = False,
    ):
        self.n_splits = n_splits
        self.min_train_ratio = min_train_ratio
        self.rolling = rolling

    def split(
        self,
        df: pd.DataFrame,
        time_col: str = "timestamp",
    ) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
        """
        Generate train/test DataFrame pairs partitioned strictly by verification time.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset containing a timestamp column.
        time_col : str, default 'timestamp'
            Name of the timestamp column.

        Yields
        ------
        Tuple[pd.DataFrame, pd.DataFrame]
            (train_df, test_df)
        """
        unique_times = sorted(df[time_col].unique())
        n_times = len(unique_times)
        min_train_idx = int(n_times * self.min_train_ratio)
        test_step = max(1, (n_times - min_train_idx) // self.n_splits)

        for i in range(self.n_splits):
            split_idx = min_train_idx + i * test_step
            end_test_idx = min(n_times, split_idx + test_step) if i < self.n_splits - 1 else n_times

            if split_idx >= n_times:
                break

            train_times = unique_times[:split_idx] if not self.rolling else unique_times[max(0, split_idx - min_train_idx):split_idx]
            test_times = unique_times[split_idx:end_test_idx]

            train_df = df[df[time_col].isin(train_times)].copy()
            test_df = df[df[time_col].isin(test_times)].copy()

            if not train_df.empty and not test_df.empty:
                yield train_df, test_df
