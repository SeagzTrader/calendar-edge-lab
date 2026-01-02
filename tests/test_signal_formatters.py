"""Tests for UI component functions."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from calendar_edge.app.ui.components import (
    format_date_human,
    format_pattern_name,
    get_badge,
    get_direction_label,
    get_internal_code,
    ordinal,
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


class TestFormatPatternName:
    """Tests for format_pattern_name function."""

    def test_tdom1_human_label(self):
        result = format_pattern_name("TDOM", {"tdom": 1})
        assert result == "1st Trading Day of Month"

    def test_tdom2_human_label(self):
        result = format_pattern_name("TDOM", {"tdom": 2})
        assert result == "2nd Trading Day of Month"

    def test_tdom10_human_label(self):
        result = format_pattern_name("TDOM", {"tdom": 10})
        assert result == "10th Trading Day of Month"

    def test_cdoy_jan_2(self):
        result = format_pattern_name("CDOY", {"month": 1, "day": 2})
        assert result == "Jan 2"

    def test_cdoy_dec_26(self):
        result = format_pattern_name("CDOY", {"month": 12, "day": 26})
        assert result == "Dec 26"

    def test_cdoy_mar_15(self):
        result = format_pattern_name("CDOY", {"month": 3, "day": 15})
        assert result == "Mar 15"


class TestGetDirectionLabel:
    """Tests for get_direction_label function."""

    def test_up(self):
        assert get_direction_label("UP") == "Closes Higher"

    def test_down(self):
        assert get_direction_label("DOWN") == "Closes Lower"

    def test_unknown_returns_original(self):
        assert get_direction_label("SIDEWAYS") == "SIDEWAYS"


class TestFormatDateHuman:
    """Tests for format_date_human function."""

    def test_standard_date(self):
        assert format_date_human("2026-01-02") == "Jan 2, 2026"

    def test_december_date(self):
        assert format_date_human("2026-12-26") == "Dec 26, 2026"

    def test_empty_returns_empty(self):
        assert format_date_human("") == ""


class TestGetBadge:
    """Tests for get_badge function."""

    def test_robust_edge(self):
        """High expectancy + high decade consistency = ROBUST EDGE."""
        badge, color, _ = get_badge(
            is_validated=True,
            avg_ret=0.001,  # > 0.0005
            decade_consistency=0.85,  # >= 0.8
            win_rate=0.60,
            baseline=0.52,
        )
        assert badge == "ROBUST EDGE"
        assert color == "green"

    def test_negative_expectancy(self):
        """High win rate but negative avg_ret = NEG EXPECTANCY."""
        badge, color, _ = get_badge(
            is_validated=True,
            avg_ret=-0.001,  # negative
            decade_consistency=0.5,
            win_rate=0.60,  # > baseline + 0.05
            baseline=0.52,
        )
        assert badge == "NEG EXPECTANCY"
        assert color == "orange"

    def test_did_not_hold(self):
        """Not validated = DID NOT HOLD."""
        badge, color, _ = get_badge(
            is_validated=False,
            avg_ret=0.001,
            decade_consistency=0.85,
            win_rate=0.60,
            baseline=0.52,
        )
        assert badge == "DID NOT HOLD"
        assert color == "red"

    def test_statistical_pattern(self):
        """Validated but not robust = STATISTICAL PATTERN."""
        badge, color, _ = get_badge(
            is_validated=True,
            avg_ret=0.0003,  # < 0.0005
            decade_consistency=0.6,  # < 0.8
            win_rate=0.55,
            baseline=0.52,
        )
        assert badge == "STATISTICAL PATTERN"
        assert color == "gray"

    def test_negative_expectancy_takes_priority(self):
        """NEG EXPECTANCY should take priority over DID NOT HOLD."""
        badge, color, _ = get_badge(
            is_validated=False,
            avg_ret=-0.001,
            decade_consistency=0.85,
            win_rate=0.60,  # > baseline + 0.05
            baseline=0.52,
        )
        # NEG EXPECTANCY check happens first
        assert badge == "NEG EXPECTANCY"
        assert color == "orange"
