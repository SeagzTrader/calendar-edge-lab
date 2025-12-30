"""Base provider interface for data ingestion."""

from abc import ABC, abstractmethod

import pandas as pd


class BaseProvider(ABC):
    """Abstract base class for data providers."""

    @abstractmethod
    def fetch(self, symbol: str, start_date: str) -> pd.DataFrame:
        """Fetch price data for a symbol.

        Args:
            symbol: The symbol to fetch.
            start_date: Start date in YYYY-MM-DD format.

        Returns:
            DataFrame with 'date' and 'close' columns.
        """
        pass
