"""Tests for CDOY and TDOM scanners."""

from pathlib import Path

import pandas as pd
import pytest

from calendar_edge.features.calendar_keys import build_calendar_keys
from calendar_edge.features.returns import build_returns
from calendar_edge.scan.cdoy import CDOYScanner
from calendar_edge.scan.tdom import TDOMScanner


@pytest.fixture
def mini_prices():
    """Load mini prices fixture."""
    fixture_path = Path(__file__).parent / "fixtures" / "mini_prices.csv"
    return pd.read_csv(fixture_path)


@pytest.fixture
def keys_and_returns(mini_prices):
    """Build calendar keys and returns from fixture."""
    keys = build_calendar_keys(mini_prices, "TEST")
    returns = build_returns(mini_prices, "TEST")
    return keys, returns


def test_cdoy_scanner_generates_candidates(keys_and_returns):
    """Test that CDOY scanner generates candidates."""
    keys, returns = keys_and_returns
    scanner = CDOYScanner(keys, returns)

    candidates = list(scanner.scan())

    # Should have some candidates (unique month+day combinations)
    assert len(candidates) > 0

    # Each candidate should have key dict and returns subset
    for key, returns_subset in candidates:
        assert "month" in key
        assert "day" in key
        assert len(returns_subset) > 0
        assert "ret_cc" in returns_subset.columns
        assert "up" in returns_subset.columns


def test_cdoy_scanner_family_name(keys_and_returns):
    """Test CDOY scanner returns correct family name."""
    keys, returns = keys_and_returns
    scanner = CDOYScanner(keys, returns)

    assert scanner.get_family() == "CDOY"


def test_tdom_scanner_generates_candidates(keys_and_returns):
    """Test that TDOM scanner generates candidates."""
    keys, returns = keys_and_returns
    scanner = TDOMScanner(keys, returns)

    candidates = list(scanner.scan())

    # Should have some candidates (unique tdom values)
    assert len(candidates) > 0

    # Each candidate should have key dict and returns subset
    for key, returns_subset in candidates:
        assert "tdom" in key
        assert len(returns_subset) > 0
        assert "ret_cc" in returns_subset.columns
        assert "up" in returns_subset.columns


def test_tdom_scanner_family_name(keys_and_returns):
    """Test TDOM scanner returns correct family name."""
    keys, returns = keys_and_returns
    scanner = TDOMScanner(keys, returns)

    assert scanner.get_family() == "TDOM"


def test_cdoy_groups_by_month_day(keys_and_returns):
    """Test CDOY scanner groups correctly by month and day."""
    keys, returns = keys_and_returns
    scanner = CDOYScanner(keys, returns)

    candidates = list(scanner.scan())

    # Check that each candidate's returns match the key
    for key, returns_subset in candidates:
        month = key["month"]
        day = key["day"]

        # All dates in subset should match month and day
        for _, row in returns_subset.iterrows():
            date = pd.to_datetime(row["date"])
            assert date.month == month
            assert date.day == day


def test_tdom_groups_by_trading_day(keys_and_returns):
    """Test TDOM scanner groups correctly by tdom."""
    keys, returns = keys_and_returns
    scanner = TDOMScanner(keys, returns)

    candidates = list(scanner.scan())

    # Merge keys with scanner data to verify grouping
    merged = pd.merge(keys, returns[["date"]], on="date")

    for key, returns_subset in candidates:
        tdom = key["tdom"]

        # Get dates for this tdom
        tdom_dates = merged[merged["tdom"] == tdom]["date"].tolist()

        # Returns subset should match these dates
        subset_dates = returns_subset["date"].tolist()
        for d in subset_dates:
            assert d in tdom_dates
