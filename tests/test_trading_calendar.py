"""Tests for trading calendar utilities."""

from datetime import date

from calendar_edge.features.trading_calendar import (
    build_future_calendar_keys,
    get_trading_sessions,
    is_trading_day,
)


class TestGetTradingSessions:
    """Tests for get_trading_sessions function."""

    def test_no_weekends_in_sessions(self):
        """Trading sessions should never include weekends."""
        # Get a range that includes multiple weekends
        sessions = get_trading_sessions(date(2026, 1, 1), date(2026, 1, 31))

        for session in sessions:
            # weekday() returns 5 for Saturday, 6 for Sunday
            assert session.weekday() < 5, f"{session} is a weekend day"

    def test_jan_1_2026_excluded(self):
        """Jan 1, 2026 is New Year's Day and should be excluded."""
        sessions = get_trading_sessions(date(2025, 12, 29), date(2026, 1, 10))

        session_dates = [s for s in sessions]
        assert date(2026, 1, 1) not in session_dates, "Jan 1, 2026 should be excluded (holiday)"

    def test_christmas_2025_excluded(self):
        """Dec 25, 2025 is Christmas and should be excluded."""
        sessions = get_trading_sessions(date(2025, 12, 20), date(2025, 12, 31))

        session_dates = [s for s in sessions]
        assert date(2025, 12, 25) not in session_dates, "Dec 25, 2025 should be excluded (holiday)"

    def test_sessions_are_sorted(self):
        """Sessions should be returned in chronological order."""
        sessions = get_trading_sessions(date(2026, 1, 1), date(2026, 1, 31))

        for i in range(len(sessions) - 1):
            assert sessions[i] < sessions[i + 1], "Sessions should be sorted"

    def test_empty_range_returns_empty(self):
        """A range with no trading days returns empty list."""
        # A Saturday-Sunday range
        sessions = get_trading_sessions(date(2026, 1, 3), date(2026, 1, 4))  # Sat-Sun
        assert sessions == []


class TestBuildFutureCalendarKeys:
    """Tests for build_future_calendar_keys function."""

    def test_tdom1_is_first_trading_day(self):
        """TDOM1 should be the first trading day of the month, not calendar day 1."""
        # January 2026: Jan 1 is a holiday, Jan 2 is Friday (first trading day)
        keys = build_future_calendar_keys(date(2026, 1, 1), date(2026, 1, 15))

        # Filter to January only
        jan_keys = keys[keys["month"] == 1]

        # First trading day of Jan 2026 should be Jan 2 (Friday)
        first_jan_row = jan_keys.iloc[0]
        assert first_jan_row["date"] == "2026-01-02", "First trading day of Jan 2026 should be Jan 2"
        assert first_jan_row["tdom"] == 1, "First trading day should have tdom=1"

    def test_tdom_resets_each_month(self):
        """TDOM should reset to 1 at the start of each month."""
        # Get sessions spanning Dec 2025 and Jan 2026
        keys = build_future_calendar_keys(date(2025, 12, 29), date(2026, 1, 10))

        # Find the first session of January
        jan_keys = keys[keys["month"] == 1]
        first_jan = jan_keys.iloc[0]

        assert first_jan["tdom"] == 1, "TDOM should reset to 1 in new month"

    def test_no_weekends_in_keys(self):
        """Calendar keys should never include weekends."""
        keys = build_future_calendar_keys(date(2026, 1, 1), date(2026, 1, 31))

        for _, row in keys.iterrows():
            assert row["dow"] < 5, f"{row['date']} has dow={row['dow']} (weekend)"

    def test_no_holidays_in_keys(self):
        """Calendar keys should exclude market holidays."""
        keys = build_future_calendar_keys(date(2025, 12, 20), date(2026, 1, 10))

        dates = keys["date"].tolist()
        assert "2025-12-25" not in dates, "Christmas should be excluded"
        assert "2026-01-01" not in dates, "New Year's Day should be excluded"

    def test_columns_present(self):
        """Calendar keys should have all required columns."""
        keys = build_future_calendar_keys(date(2026, 1, 1), date(2026, 1, 15))

        required_cols = ["date", "month", "day", "tdom", "dow"]
        for col in required_cols:
            assert col in keys.columns, f"Missing column: {col}"

    def test_empty_range_returns_empty_df(self):
        """A range with no trading days returns empty DataFrame."""
        keys = build_future_calendar_keys(date(2026, 1, 3), date(2026, 1, 4))  # Sat-Sun
        assert keys.empty


class TestIsTradingDay:
    """Tests for is_trading_day function."""

    def test_weekday_is_trading_day(self):
        """A normal weekday should be a trading day."""
        # Monday Jan 5, 2026
        assert is_trading_day(date(2026, 1, 5)) is True

    def test_saturday_is_not_trading_day(self):
        """Saturday should not be a trading day."""
        # Saturday Jan 3, 2026
        assert is_trading_day(date(2026, 1, 3)) is False

    def test_sunday_is_not_trading_day(self):
        """Sunday should not be a trading day."""
        # Sunday Jan 4, 2026
        assert is_trading_day(date(2026, 1, 4)) is False

    def test_holiday_is_not_trading_day(self):
        """A market holiday should not be a trading day."""
        # New Year's Day 2026
        assert is_trading_day(date(2026, 1, 1)) is False
        # Christmas 2025
        assert is_trading_day(date(2025, 12, 25)) is False


class TestTDOMAccuracy:
    """Tests for TDOM calculation accuracy."""

    def test_tdom_increments_correctly(self):
        """TDOM should increment by 1 for each trading day in a month."""
        keys = build_future_calendar_keys(date(2026, 1, 2), date(2026, 1, 31))

        jan_keys = keys[keys["month"] == 1]
        tdom_values = jan_keys["tdom"].tolist()

        # Should be sequential: 1, 2, 3, ...
        expected = list(range(1, len(tdom_values) + 1))
        assert tdom_values == expected, f"TDOM should be sequential: got {tdom_values}"

    def test_tdom10_is_tenth_trading_day(self):
        """TDOM10 should be the 10th trading day of the month."""
        keys = build_future_calendar_keys(date(2026, 1, 1), date(2026, 1, 31))

        jan_keys = keys[keys["month"] == 1]
        tdom10_row = jan_keys[jan_keys["tdom"] == 10].iloc[0]

        # Count: Jan 2 (1), 5 (2), 6 (3), 7 (4), 8 (5), 9 (6), 12 (7), 13 (8), 14 (9), 15 (10)
        assert tdom10_row["date"] == "2026-01-15", f"TDOM10 should be Jan 15, got {tdom10_row['date']}"
