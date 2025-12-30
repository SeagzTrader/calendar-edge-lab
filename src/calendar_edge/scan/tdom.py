"""Trading Day of Month (TDOM) scanner."""

import logging
from typing import Iterator

import pandas as pd

logger = logging.getLogger("calendar_edge")


class TDOMScanner:
    """Scanner for Trading Day of Month effects."""

    def __init__(self, calendar_keys_df: pd.DataFrame, returns_df: pd.DataFrame):
        """Initialize scanner.

        Args:
            calendar_keys_df: DataFrame with calendar keys.
            returns_df: DataFrame with returns.
        """
        self.keys = calendar_keys_df.copy()
        self.returns = returns_df.copy()

        # Merge for efficient lookup
        self.data = pd.merge(
            self.keys,
            self.returns[["date", "ret_cc", "up"]],
            on="date",
        )

    def scan(self) -> Iterator[tuple[dict, pd.DataFrame]]:
        """Generate all TDOM candidates.

        Yields:
            Tuple of (key_dict, returns_subset_df) for each tdom value.
        """
        # Group by tdom
        for tdom, group in self.data.groupby("tdom"):
            key = {"tdom": int(tdom)}
            yield key, group[["date", "ret_cc", "up"]]

    def get_family(self) -> str:
        """Return scanner family name."""
        return "TDOM"
