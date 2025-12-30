"""Statistical computations for calendar effects."""

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from statsmodels.stats.proportion import proportion_confint


def compute_stats(
    returns_df: pd.DataFrame,
    baseline_win_rate: float,
    direction: str,
) -> dict:
    """Compute statistics for a signal.

    Args:
        returns_df: DataFrame with 'ret_cc' and 'up' columns for signal dates.
        baseline_win_rate: Baseline win rate for comparison.
        direction: 'UP' or 'DOWN'.

    Returns:
        Dictionary with n, wins, win_rate, avg_ret, median_ret, ci_low, ci_high, p_value.
    """
    n = len(returns_df)

    if n == 0:
        return {
            "n": 0,
            "wins": 0,
            "win_rate": 0.0,
            "avg_ret": None,
            "median_ret": None,
            "ci_low": None,
            "ci_high": None,
            "p_value": None,
        }

    wins = int(returns_df["up"].sum())
    win_rate = wins / n if n > 0 else 0.0
    avg_ret = float(returns_df["ret_cc"].mean())
    median_ret = float(returns_df["ret_cc"].median())

    # Wilson confidence interval
    if n > 0:
        ci_low, ci_high = proportion_confint(wins, n, method="wilson")
    else:
        ci_low, ci_high = None, None

    # Binomial test (one-tailed, directional)
    if n > 0 and 0 < baseline_win_rate < 1:
        if direction == "UP":
            result = binomtest(wins, n, p=baseline_win_rate, alternative="greater")
        else:  # DOWN
            result = binomtest(wins, n, p=baseline_win_rate, alternative="less")
        p_value = result.pvalue
    else:
        p_value = None

    return {
        "n": n,
        "wins": wins,
        "win_rate": win_rate,
        "avg_ret": avg_ret,
        "median_ret": median_ret,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
    }


def compute_z_score(
    win_rate: float,
    baseline_win_rate: float,
    direction: str,
    n: int,
) -> float:
    """Compute Z-score for a signal.

    Args:
        win_rate: Observed win rate.
        baseline_win_rate: Baseline win rate.
        direction: 'UP' or 'DOWN'.
        n: Sample size.

    Returns:
        Z-score.
    """
    if direction == "UP":
        s = win_rate
        bs = baseline_win_rate
    else:  # DOWN
        s = 1 - win_rate
        bs = 1 - baseline_win_rate

    delta = s - bs

    if bs in (0, 1) or n == 0:
        return 0.0

    z = delta / np.sqrt(bs * (1 - bs) / n)
    return float(z)
