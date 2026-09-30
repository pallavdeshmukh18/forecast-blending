"""
Multi-model forecast skill comparison and benchmarking engine.
"""

from typing import Dict, List, Optional
import pandas as pd

from evaluation.metrics import compute_all_metrics


def compare_models(
    aligned_df: pd.DataFrame,
    group_by: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Evaluate and benchmark multiple forecast models and blended forecasts against observations.

    Parameters
    ----------
    aligned_df : pd.DataFrame
        Aligned verification DataFrame containing ['model_name', 'variable', 'forecast_value', 'observed_value'].
    group_by : Optional[List[str]], default None
        List of grouping columns (e.g. ['variable', 'lead_time'] or ['variable', 'region']).
        If None, groups by ['variable', 'lead_time'] if present, else ['variable'].
    models : Optional[List[str]], default None
        List of specific models to evaluate. If None, evaluates all models present.

    Returns
    -------
    pd.DataFrame
        Structured DataFrame containing metric scores for each model and grouping slice.
    """
    df = aligned_df.copy()

    if models is not None:
        df = df[df["model_name"].isin(models)]

    if group_by is None:
        possible_keys = ["variable", "lead_time", "region"]
        group_by = [k for k in possible_keys if k in df.columns]

    full_grouping = ["model_name"] + [col for col in group_by if col != "model_name"]

    records = []
    for keys, group in df.groupby(full_grouping, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        group_dict = dict(zip(full_grouping, keys))

        try:
            metrics = compute_all_metrics(
                obs=group["observed_value"].values,
                pred=group["forecast_value"].values,
            )
            record = {**group_dict, **metrics}
            records.append(record)
        except ValueError:
            continue

    res_df = pd.DataFrame(records)
    if not res_df.empty:
        # Sort for clean presentation
        sort_cols = [c for c in group_by + ["model_name"] if c in res_df.columns]
        res_df = res_df.sort_values(by=sort_cols).reset_index(drop=True)

    return res_df


def compute_skill_rankings(comparison_df: pd.DataFrame, metric: str = "rmse") -> pd.DataFrame:
    """
    Rank models by performance for each variable and lead time.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Output of compare_models.
    metric : str, default 'rmse'
        Metric to rank by ('rmse', 'mae', 'bias', 'correlation').

    Returns
    -------
    pd.DataFrame
        Ranked comparison table with 'rank' column (1 = best model).
    """
    df = comparison_df.copy()
    partition_cols = [c for c in ["variable", "lead_time", "region"] if c in df.columns]

    ascending = metric.lower() in ["mae", "rmse"]
    if metric.lower() == "bias":
        df["_bias_abs"] = df["bias"].abs()
        df["rank"] = df.groupby(partition_cols)["_bias_abs"].rank(ascending=True)
        df = df.drop(columns=["_bias_abs"])
    else:
        df["rank"] = df.groupby(partition_cols)[metric.lower()].rank(ascending=ascending)

    return df.sort_values(by=partition_cols + ["rank"]).reset_index(drop=True)
