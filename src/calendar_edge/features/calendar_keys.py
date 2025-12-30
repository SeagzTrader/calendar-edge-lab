"""Calendar keys feature engineering."""

import pandas as pd


def build_calendar_keys(prices_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Build calendar keys from price data.

    Args:
        prices_df: DataFrame with 'date' and 'close' columns.
        symbol: Symbol identifier.

    Returns:
        DataFrame with calendar key columns.
    """
    df = prices_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Basic calendar fields
    df["symbol"] = symbol
    df["dow"] = df["date"].dt.dayofweek  # 0=Monday, 6=Sunday
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day

    # Trading day of month (tdom): 1-indexed
    df["year_month"] = df["date"].dt.to_period("M")
    df["tdom"] = df.groupby("year_month").cumcount() + 1

    # Trading day of year (tdoy): 1-indexed, resets each year
    df["year"] = df["date"].dt.year
    df["tdoy"] = df.groupby("year").cumcount() + 1

    # Month boundary flags
    # is_month_start: first trading day of the month
    df["is_month_start"] = (df["tdom"] == 1).astype(int)

    # is_month_end: last trading day of the month
    # Need to look ahead or mark the last row in each month group
    df["is_month_end"] = 0
    for ym in df["year_month"].unique():
        mask = df["year_month"] == ym
        last_idx = df[mask].index[-1]
        df.loc[last_idx, "is_month_end"] = 1

    # Clean up temporary columns
    df = df.drop(columns=["year_month", "year", "close"])

    # Convert date back to string
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    return df[["symbol", "date", "dow", "month", "day", "tdom", "tdoy", "is_month_start", "is_month_end"]]
