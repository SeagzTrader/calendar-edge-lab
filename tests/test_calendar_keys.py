"""Tests for calendar keys feature engineering."""

from pathlib import Path

import pandas as pd
import pytest

from calendar_edge.features.calendar_keys import build_calendar_keys


@pytest.fixture
def mini_prices():
    """Load mini prices fixture."""
    fixture_path = Path(__file__).parent / "fixtures" / "mini_prices.csv"
    return pd.read_csv(fixture_path)


def test_build_calendar_keys_basic(mini_prices):
    """Test that calendar keys are built correctly."""
    keys = build_calendar_keys(mini_prices, "TEST")

    assert len(keys) == len(mini_prices)
    assert "symbol" in keys.columns
    assert "date" in keys.columns
    assert "dow" in keys.columns
    assert "month" in keys.columns
    assert "day" in keys.columns
    assert "tdom" in keys.columns
    assert "tdoy" in keys.columns
    assert "is_month_start" in keys.columns
    assert "is_month_end" in keys.columns


def test_tdom_resets_each_month(mini_prices):
    """Test that tdom resets to 1 at start of each month."""
    keys = build_calendar_keys(mini_prices, "TEST")

    # January 2020 first trading day should be tdom=1
    jan_first = keys[keys["date"] == "2020-01-02"]
    assert len(jan_first) == 1
    assert jan_first.iloc[0]["tdom"] == 1

    # February 2020 first trading day should also be tdom=1
    feb_first = keys[keys["date"] == "2020-02-03"]
    assert len(feb_first) == 1
    assert feb_first.iloc[0]["tdom"] == 1

    # March 2020 first trading day should also be tdom=1
    mar_first = keys[keys["date"] == "2020-03-02"]
    assert len(mar_first) == 1
    assert mar_first.iloc[0]["tdom"] == 1


def test_tdoy_increments_correctly(mini_prices):
    """Test that tdoy increments correctly within year."""
    keys = build_calendar_keys(mini_prices, "TEST")

    # Sort by date and check tdoy increments
    keys_sorted = keys.sort_values("date")
    tdoys = keys_sorted["tdoy"].tolist()

    # All trading days are in 2020, so tdoy should increment from 1
    assert tdoys[0] == 1
    for i in range(1, len(tdoys)):
        assert tdoys[i] == tdoys[i - 1] + 1


def test_month_start_flag(mini_prices):
    """Test is_month_start flag is set correctly."""
    keys = build_calendar_keys(mini_prices, "TEST")

    # First trading days of months
    month_starts = keys[keys["is_month_start"] == 1]

    # Should have 3 month starts (Jan, Feb, Mar)
    assert len(month_starts) == 3

    # Verify dates
    start_dates = set(month_starts["date"].tolist())
    assert "2020-01-02" in start_dates  # First trading day of Jan
    assert "2020-02-03" in start_dates  # First trading day of Feb
    assert "2020-03-02" in start_dates  # First trading day of Mar


def test_month_end_flag(mini_prices):
    """Test is_month_end flag is set correctly."""
    keys = build_calendar_keys(mini_prices, "TEST")

    # Last trading days of months
    month_ends = keys[keys["is_month_end"] == 1]

    # Should have 3 month ends (Jan, Feb, Mar in fixture is incomplete but has last trading days)
    # Jan 31 (last trading day of Jan), Feb 28 (last trading day of Feb)
    # March is incomplete so last date is month end
    end_dates = set(month_ends["date"].tolist())
    assert "2020-01-31" in end_dates
    assert "2020-02-28" in end_dates


def test_dow_values(mini_prices):
    """Test that dow values are 0-6."""
    keys = build_calendar_keys(mini_prices, "TEST")

    # All dow should be 0-4 (weekdays only in trading data)
    assert keys["dow"].min() >= 0
    assert keys["dow"].max() <= 6

    # 2020-01-02 was a Thursday (dow=3)
    jan2 = keys[keys["date"] == "2020-01-02"].iloc[0]
    assert jan2["dow"] == 3


def test_symbol_column(mini_prices):
    """Test that symbol column is correctly set."""
    keys = build_calendar_keys(mini_prices, "MYSYMBOL")

    assert (keys["symbol"] == "MYSYMBOL").all()
