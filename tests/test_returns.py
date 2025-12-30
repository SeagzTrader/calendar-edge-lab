"""Tests for returns feature engineering."""

import math
from pathlib import Path

import pandas as pd
import pytest

from calendar_edge.features.returns import build_returns


@pytest.fixture
def mini_prices():
    """Load mini prices fixture."""
    fixture_path = Path(__file__).parent / "fixtures" / "mini_prices.csv"
    return pd.read_csv(fixture_path)


def test_build_returns_basic(mini_prices):
    """Test that returns are built correctly."""
    returns = build_returns(mini_prices, "TEST")

    # Should have one less row than prices (first row dropped)
    assert len(returns) == len(mini_prices) - 1

    assert "symbol" in returns.columns
    assert "date" in returns.columns
    assert "ret_cc" in returns.columns
    assert "up" in returns.columns


def test_ret_cc_calculation(mini_prices):
    """Test that log returns are calculated correctly."""
    returns = build_returns(mini_prices, "TEST")

    # First return should be log(101/100) = 0.00995...
    first_return = returns[returns["date"] == "2020-01-03"].iloc[0]["ret_cc"]
    expected = math.log(101.0 / 100.0)
    assert abs(first_return - expected) < 1e-10


def test_up_flag_positive(mini_prices):
    """Test up flag is 1 for positive returns."""
    returns = build_returns(mini_prices, "TEST")

    # 2020-01-03: 101 > 100, so up=1
    jan3 = returns[returns["date"] == "2020-01-03"].iloc[0]
    assert jan3["up"] == 1

    # Verify all positive returns have up=1
    positive_returns = returns[returns["ret_cc"] > 0]
    assert (positive_returns["up"] == 1).all()


def test_up_flag_negative(mini_prices):
    """Test up flag is 0 for negative returns."""
    returns = build_returns(mini_prices, "TEST")

    # 2020-01-07: 101.50 < 102.00, so up=0
    jan7 = returns[returns["date"] == "2020-01-07"].iloc[0]
    assert jan7["up"] == 0

    # Verify all negative returns have up=0
    negative_returns = returns[returns["ret_cc"] < 0]
    assert (negative_returns["up"] == 0).all()


def test_up_flag_zero():
    """Test up flag is 0 for exactly zero returns."""
    # Create prices with a flat day
    prices = pd.DataFrame({
        "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
        "close": [100.0, 100.0, 100.0],  # Two consecutive flat days
    })

    returns = build_returns(prices, "TEST")

    # All returns should be 0 and up should be 0
    assert (returns["ret_cc"] == 0.0).all()
    assert (returns["up"] == 0).all()


def test_first_row_dropped(mini_prices):
    """Test that first row (no prior close) is dropped."""
    returns = build_returns(mini_prices, "TEST")

    # First date in returns should be second date in prices
    assert returns["date"].iloc[0] == "2020-01-03"  # Not 2020-01-02


def test_symbol_column(mini_prices):
    """Test that symbol column is correctly set."""
    returns = build_returns(mini_prices, "MYSYMBOL")

    assert (returns["symbol"] == "MYSYMBOL").all()


def test_date_format(mini_prices):
    """Test that dates are in string format."""
    returns = build_returns(mini_prices, "TEST")

    # Dates should be strings
    assert returns["date"].dtype == object
    assert returns["date"].iloc[0] == "2020-01-03"
