"""Calendar Day of Year (CDOY) scanner."""

import logging
from typing import Iterator

import pandas as pd

logger = logging.getLogger("calendar_edge")


class CDOYScanner:
    """Scanner for Calendar Day of Year effects (month + day combinations)."""

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
        """Generate all CDOY candidates.

        Yields:
            Tuple of (key_dict, returns_subset_df) for each month+day combination.
        """
        # Group by month and day
        for (month, day), group in self.data.groupby(["month", "day"]):
            key = {"month": int(month), "day": int(day)}
            yield key, group[["date", "ret_cc", "up"]]

    def get_family(self) -> str:
        """Return scanner family name."""
        return "CDOY"
