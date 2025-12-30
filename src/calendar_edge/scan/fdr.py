"""Benjamini-Hochberg FDR correction."""

import numpy as np
import pandas as pd

from calendar_edge.config import FDR_THRESHOLD


def apply_fdr(p_values: list[float], threshold: float = FDR_THRESHOLD) -> list[float]:
    """Apply Benjamini-Hochberg FDR correction.

    Args:
        p_values: List of p-values.
        threshold: FDR threshold (default from config).

    Returns:
        List of adjusted q-values.
    """
    n = len(p_values)
    if n == 0:
        return []

    # Convert to array
    p_arr = np.array(p_values)

    # Handle NaN values
    valid_mask = ~np.isnan(p_arr)
    valid_p = p_arr[valid_mask]

    if len(valid_p) == 0:
        return [np.nan] * n

    # Sort p-values
    sorted_indices = np.argsort(valid_p)
    sorted_p = valid_p[sorted_indices]

    # BH procedure
    m = len(sorted_p)
    ranks = np.arange(1, m + 1)

    # Adjusted p-values (q-values)
    q_values = sorted_p * m / ranks

    # Ensure monotonicity (cumulative minimum from right to left)
    q_values = np.minimum.accumulate(q_values[::-1])[::-1]

    # Cap at 1.0
    q_values = np.minimum(q_values, 1.0)

    # Map back to original order
    unsorted_q = np.empty(m)
    unsorted_q[sorted_indices] = q_values

    # Reconstruct full array with NaN placeholders
    result = np.full(n, np.nan)
    result[valid_mask] = unsorted_q

    return result.tolist()


def get_significant_signals(
    signals_df: pd.DataFrame,
    q_column: str = "fdr_q",
    threshold: float = FDR_THRESHOLD,
) -> pd.DataFrame:
    """Filter signals that pass FDR threshold.

    Args:
        signals_df: DataFrame with signal statistics.
        q_column: Column name for q-values.
        threshold: FDR threshold.

    Returns:
        Filtered DataFrame.
    """
    return signals_df[signals_df[q_column] <= threshold].copy()
