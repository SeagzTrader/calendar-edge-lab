"""Tests for walk-forward validation."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from calendar_edge.config import TEST_START, TRAIN_END
from calendar_edge.db import get_connection, init_db
from calendar_edge.features.calendar_keys import build_calendar_keys
from calendar_edge.features.returns import build_returns


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    init_db(db_path)
    conn = get_connection(db_path)

    yield conn, db_path

    conn.close()
    db_path.unlink()


def test_train_test_split_no_overlap(temp_db):
    """Test that train and test windows don't overlap."""
    conn, db_path = temp_db

    # Create sample data spanning train and test periods
    dates = pd.date_range("2008-01-01", "2012-12-31", freq="B")
    prices_df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "close": range(100, 100 + len(dates)),
    })

    # Build keys and returns
    keys_df = build_calendar_keys(prices_df, "TEST")
    _returns_df = build_returns(prices_df, "TEST")  # noqa: F841

    # Split by train/test
    train_keys = keys_df[keys_df["date"] <= TRAIN_END]
    test_keys = keys_df[keys_df["date"] >= TEST_START]

    train_dates = set(train_keys["date"].tolist())
    test_dates = set(test_keys["date"].tolist())

    # No overlap
    assert len(train_dates & test_dates) == 0


def test_train_ends_before_test_starts():
    """Test that TRAIN_END is before TEST_START."""
    assert TRAIN_END < TEST_START
    assert TRAIN_END == "2009-12-31"
    assert TEST_START == "2010-01-01"


def test_returns_exclude_first_row():
    """Test that returns calculation excludes first row."""
    prices_df = pd.DataFrame({
        "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
        "close": [100.0, 101.0, 102.0],
    })

    returns_df = build_returns(prices_df, "TEST")

    # Should have 2 rows, not 3
    assert len(returns_df) == 2

    # First return date should be second price date
    assert returns_df["date"].iloc[0] == "2020-01-02"


def test_calendar_keys_date_alignment():
    """Test that calendar keys align with price dates."""
    prices_df = pd.DataFrame({
        "date": ["2020-01-02", "2020-01-03", "2020-01-06"],
        "close": [100.0, 101.0, 102.0],
    })

    keys_df = build_calendar_keys(prices_df, "TEST")

    # Keys should have same dates as prices
    assert set(keys_df["date"].tolist()) == set(prices_df["date"].tolist())


def test_train_window_contains_only_train_dates():
    """Test that filtering by TRAIN_END works correctly."""
    dates = ["2009-12-30", "2009-12-31", "2010-01-04", "2010-01-05"]
    returns_df = pd.DataFrame({
        "date": dates,
        "ret_cc": [0.01, -0.01, 0.02, -0.02],
        "up": [1, 0, 1, 0],
    })

    train_returns = returns_df[returns_df["date"] <= TRAIN_END]

    # Should only have 2009 dates
    assert len(train_returns) == 2
    assert "2009-12-30" in train_returns["date"].tolist()
    assert "2009-12-31" in train_returns["date"].tolist()
    assert "2010-01-04" not in train_returns["date"].tolist()


def test_test_window_contains_only_test_dates():
    """Test that filtering by TEST_START works correctly."""
    dates = ["2009-12-30", "2009-12-31", "2010-01-04", "2010-01-05"]
    returns_df = pd.DataFrame({
        "date": dates,
        "ret_cc": [0.01, -0.01, 0.02, -0.02],
        "up": [1, 0, 1, 0],
    })

    test_returns = returns_df[returns_df["date"] >= TEST_START]

    # Should only have 2010 dates
    assert len(test_returns) == 2
    assert "2010-01-04" in test_returns["date"].tolist()
    assert "2010-01-05" in test_returns["date"].tolist()
    assert "2009-12-31" not in test_returns["date"].tolist()
