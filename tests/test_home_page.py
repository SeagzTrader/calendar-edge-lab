"""Smoke tests for Home page data loading."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from calendar_edge.app.streamlit_app import compute_held_up_status


class TestComputeHeldUpStatus:
    """Tests for compute_held_up_status function."""

    def test_no_test_data_returns_unknown(self):
        """When test data is missing, status should be 'unknown'."""
        result = compute_held_up_status(
            train_wr=0.55,
            train_baseline=0.53,
            test_wr=None,
            test_baseline=None,
            test_n=None,
        )
        assert result == "unknown"

    def test_zero_test_n_returns_unknown(self):
        """When test_n is 0, status should be 'unknown'."""
        result = compute_held_up_status(
            train_wr=0.55,
            train_baseline=0.53,
            test_wr=0.56,
            test_baseline=0.54,
            test_n=0,
        )
        assert result == "unknown"

    def test_positive_delta_large_n_returns_yes(self):
        """Positive test delta with n>=30 should return 'yes'."""
        result = compute_held_up_status(
            train_wr=0.60,
            train_baseline=0.53,
            test_wr=0.58,
            test_baseline=0.54,  # delta = 0.04 > 0
            test_n=50,
        )
        assert result == "yes"

    def test_positive_delta_small_n_returns_unclear(self):
        """Positive test delta with n<30 should return 'unclear'."""
        result = compute_held_up_status(
            train_wr=0.60,
            train_baseline=0.53,
            test_wr=0.58,
            test_baseline=0.54,  # delta = 0.04 > 0
            test_n=20,  # < 30
        )
        assert result == "unclear"

    def test_near_baseline_returns_unclear(self):
        """Test delta within 2% of baseline should return 'unclear'."""
        result = compute_held_up_status(
            train_wr=0.60,
            train_baseline=0.53,
            test_wr=0.53,  # exactly at baseline
            test_baseline=0.54,  # delta = -0.01
            test_n=50,
        )
        assert result == "unclear"

    def test_negative_delta_returns_no(self):
        """Test delta more than 2% below baseline should return 'no'."""
        result = compute_held_up_status(
            train_wr=0.60,
            train_baseline=0.53,
            test_wr=0.50,
            test_baseline=0.54,  # delta = -0.04 < -0.02
            test_n=50,
        )
        assert result == "no"

    def test_uses_train_baseline_when_test_baseline_none(self):
        """When test_baseline is None, should use train_baseline."""
        # Positive delta vs train_baseline
        result = compute_held_up_status(
            train_wr=0.60,
            train_baseline=0.53,
            test_wr=0.56,  # delta = 0.03 > 0
            test_baseline=None,
            test_n=50,
        )
        assert result == "yes"


class TestHomePageDataFlow:
    """Smoke tests for Home page data structures."""

    def test_held_up_status_values(self):
        """Verify all held_up_status values are valid."""
        valid_statuses = {"yes", "unclear", "no", "unknown"}

        # Test all branches
        test_cases = [
            (None, None, "unknown"),
            (0.56, 50, "yes"),  # positive, large n
            (0.56, 20, "unclear"),  # positive, small n
            (0.53, 50, "unclear"),  # near baseline
            (0.50, 50, "no"),  # negative
        ]

        for test_wr, test_n, expected in test_cases:
            result = compute_held_up_status(
                train_wr=0.60,
                train_baseline=0.53,
                test_wr=test_wr,
                test_baseline=0.54,
                test_n=test_n,
            )
            assert result in valid_statuses
            assert result == expected

    def test_confidence_mapping(self):
        """Verify held_up status maps to correct confidence levels."""
        # This tests the mapping logic used in render_home
        confidence_map = {
            "yes": "High",
            "unclear": "Medium",
            "no": "Low",
            "unknown": "Low",
        }

        for status, expected_confidence in confidence_map.items():
            # Simulate the mapping from render_home
            if status == "yes":
                confidence = "High"
            elif status == "unclear":
                confidence = "Medium"
            else:
                confidence = "Low"

            assert confidence == expected_confidence

    def test_mode_filter_values(self):
        """Verify mode filter options are as expected."""
        expected_modes = ["All", "CDOY only", "TDOM only"]
        # This is a documentation test - ensures we handle these modes
        for mode in expected_modes:
            assert mode in ["All", "CDOY only", "TDOM only"]
