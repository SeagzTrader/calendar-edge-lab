"""Tests for signal key formatter functions."""


# Import formatters from streamlit app
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from calendar_edge.app.streamlit_app import (
    describe_direction,
    describe_pattern_key,
    describe_signal_key,
    format_short_date,
    format_signal_key,
    get_held_up_label,
    get_internal_code,
    ordinal,
    parse_internal_code,
)


class TestOrdinal:
    """Tests for ordinal number formatting."""

    def test_first(self):
        assert ordinal(1) == "1st"

    def test_second(self):
        assert ordinal(2) == "2nd"

    def test_third(self):
        assert ordinal(3) == "3rd"

    def test_fourth_through_tenth(self):
        assert ordinal(4) == "4th"
        assert ordinal(5) == "5th"
        assert ordinal(10) == "10th"

    def test_eleventh_through_thirteenth(self):
        """11th, 12th, 13th are special cases."""
        assert ordinal(11) == "11th"
        assert ordinal(12) == "12th"
        assert ordinal(13) == "13th"

    def test_twenty_first(self):
        assert ordinal(21) == "21st"

    def test_twenty_second(self):
        assert ordinal(22) == "22nd"

    def test_twenty_third(self):
        assert ordinal(23) == "23rd"


class TestGetInternalCode:
    """Tests for get_internal_code function."""

    def test_tdom1(self):
        assert get_internal_code("TDOM", {"tdom": 1}) == "TDOM1"

    def test_tdom10(self):
        assert get_internal_code("TDOM", {"tdom": 10}) == "TDOM10"

    def test_cdoy_jan_2(self):
        assert get_internal_code("CDOY", {"month": 1, "day": 2}) == "M01D02"

    def test_cdoy_dec_26(self):
        assert get_internal_code("CDOY", {"month": 12, "day": 26}) == "M12D26"

    def test_cdoy_single_digit_day(self):
        assert get_internal_code("CDOY", {"month": 3, "day": 5}) == "M03D05"


class TestFormatSignalKey:
    """Tests for format_signal_key function."""

    def test_tdom1_human_label(self):
        result = format_signal_key("TDOM", {"tdom": 1})
        assert result == "1st trading day of month"

    def test_tdom2_human_label(self):
        result = format_signal_key("TDOM", {"tdom": 2})
        assert result == "2nd trading day of month"

    def test_tdom10_human_label(self):
        result = format_signal_key("TDOM", {"tdom": 10})
        assert result == "10th trading day of month"

    def test_cdoy_jan_2(self):
        result = format_signal_key("CDOY", {"month": 1, "day": 2})
        assert result == "Jan 2"

    def test_cdoy_dec_26(self):
        result = format_signal_key("CDOY", {"month": 12, "day": 26})
        assert result == "Dec 26"

    def test_cdoy_mar_15(self):
        result = format_signal_key("CDOY", {"month": 3, "day": 15})
        assert result == "Mar 15"


class TestDescribeSignalKey:
    """Tests for describe_signal_key function."""

    def test_up_direction(self):
        result = describe_signal_key("TDOM", {"tdom": 1}, "UP")
        assert "up" in result.lower()
        assert "close" in result.lower()

    def test_down_direction(self):
        result = describe_signal_key("CDOY", {"month": 12, "day": 26}, "DOWN")
        assert "down" in result.lower()
        assert "close" in result.lower()


class TestParseInternalCode:
    """Tests for parse_internal_code function."""

    def test_parse_tdom1(self):
        result = parse_internal_code("TDOM1")
        assert result == ("TDOM", {"tdom": 1})

    def test_parse_tdom10(self):
        result = parse_internal_code("TDOM10")
        assert result == ("TDOM", {"tdom": 10})

    def test_parse_tdom22(self):
        result = parse_internal_code("TDOM22")
        assert result == ("TDOM", {"tdom": 22})

    def test_parse_cdoy_m01d02(self):
        result = parse_internal_code("M01D02")
        assert result == ("CDOY", {"month": 1, "day": 2})

    def test_parse_cdoy_m12d26(self):
        result = parse_internal_code("M12D26")
        assert result == ("CDOY", {"month": 12, "day": 26})

    def test_parse_cdoy_m3d5(self):
        """Single digit month/day should also work."""
        result = parse_internal_code("M3D5")
        assert result == ("CDOY", {"month": 3, "day": 5})

    def test_parse_invalid_returns_none(self):
        assert parse_internal_code("INVALID") is None
        assert parse_internal_code("") is None
        assert parse_internal_code("XYZ123") is None

    def test_parse_lowercase_invalid(self):
        """Parsing is case-sensitive."""
        assert parse_internal_code("tdom1") is None
        assert parse_internal_code("m01d02") is None


class TestRoundTrip:
    """Tests that internal codes can be parsed back correctly."""

    def test_tdom_roundtrip(self):
        """TDOM codes should roundtrip through get_internal_code and parse_internal_code."""
        for tdom in [1, 2, 5, 10, 22]:
            key = {"tdom": tdom}
            code = get_internal_code("TDOM", key)
            parsed = parse_internal_code(code)
            assert parsed == ("TDOM", key)

    def test_cdoy_roundtrip(self):
        """CDOY codes should roundtrip through get_internal_code and parse_internal_code."""
        test_cases = [
            {"month": 1, "day": 2},
            {"month": 12, "day": 26},
            {"month": 3, "day": 15},
            {"month": 7, "day": 4},
        ]
        for key in test_cases:
            code = get_internal_code("CDOY", key)
            parsed = parse_internal_code(code)
            assert parsed == ("CDOY", key)


class TestDescribePatternKey:
    """Tests for describe_pattern_key function."""

    def test_tdom1(self):
        result = describe_pattern_key("TDOM1")
        assert result == "1st trading day of month"

    def test_tdom10(self):
        result = describe_pattern_key("TDOM10")
        assert result == "10th trading day of month"

    def test_cdoy_m12d26(self):
        result = describe_pattern_key("M12D26")
        assert result == "Dec 26"

    def test_cdoy_m01d02(self):
        result = describe_pattern_key("M01D02")
        assert result == "Jan 2"

    def test_invalid_returns_original(self):
        result = describe_pattern_key("INVALID")
        assert result == "INVALID"


class TestDescribeDirection:
    """Tests for describe_direction function."""

    def test_up(self):
        assert describe_direction("UP") == "Up day"

    def test_down(self):
        assert describe_direction("DOWN") == "Down day"

    def test_unknown_returns_original(self):
        assert describe_direction("SIDEWAYS") == "SIDEWAYS"


class TestFormatShortDate:
    """Tests for format_short_date function."""

    def test_already_short(self):
        assert format_short_date("2026-01-02") == "2026-01-02"

    def test_handles_string(self):
        assert format_short_date("2026-12-26") == "2026-12-26"


class TestGetHeldUpLabel:
    """Tests for get_held_up_label function."""

    def test_yes(self):
        icon, label = get_held_up_label("yes")
        assert icon == "✅"
        assert label == "Held up"

    def test_unclear(self):
        icon, label = get_held_up_label("unclear")
        assert icon == "⚠️"
        assert label == "Unclear"

    def test_no(self):
        icon, label = get_held_up_label("no")
        assert icon == "❌"
        assert label == "Did not hold"

    def test_unknown(self):
        icon, label = get_held_up_label("unknown")
        assert icon == "❓"
        assert label == "No test data"
