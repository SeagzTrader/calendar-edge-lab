"""YFinance data provider."""

import logging

import pandas as pd
import yfinance as yf

from .providers import BaseProvider

logger = logging.getLogger("calendar_edge")


class YFinanceProvider(BaseProvider):
    """Data provider using yfinance."""

    def fetch(self, symbol: str, start_date: str) -> pd.DataFrame:
        """Fetch price data from Yahoo Finance.

        Args:
            symbol: The symbol to fetch (e.g., ^GSPC).
            start_date: Start date in YYYY-MM-DD format.

        Returns:
            DataFrame with 'date' and 'close' columns.
        """
        logger.info(f"Fetching {symbol} from yfinance starting {start_date}")

        ticker = yf.Ticker(symbol)
        hist = ticker.history(start=start_date, auto_adjust=True)

        if hist.empty:
            logger.warning(f"No data returned for {symbol}")
            return pd.DataFrame(columns=["date", "close"])

        # Reset index to get date as column
        hist = hist.reset_index()

        # Normalize column names
        hist.columns = hist.columns.str.lower()

        # Select and rename columns
        df = pd.DataFrame({
            "date": pd.to_datetime(hist["date"]).dt.strftime("%Y-%m-%d"),
            "close": hist["close"],
        })

        logger.info(f"Fetched {len(df)} rows for {symbol}")
        return df
