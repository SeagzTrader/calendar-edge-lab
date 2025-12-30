"""Returns feature engineering."""

import numpy as np
import pandas as pd


def build_returns(prices_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Build returns from price data.

    Args:
        prices_df: DataFrame with 'date' and 'close' columns.
        symbol: Symbol identifier.

    Returns:
        DataFrame with symbol, date, ret_cc, up columns.
        First row is dropped (no prior close for return calculation).
    """
    df = prices_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Calculate log returns (continuously compounded)
    df["ret_cc"] = np.log(df["close"] / df["close"].shift(1))

    # Up flag: 1 if ret_cc > 0, else 0 (0% counts as not-up)
    df["up"] = (df["ret_cc"] > 0).astype(int)

    # Drop first row (no prior close)
    df = df.iloc[1:].reset_index(drop=True)

    # Add symbol
    df["symbol"] = symbol

    # Convert date to string
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    return df[["symbol", "date", "ret_cc", "up"]]
