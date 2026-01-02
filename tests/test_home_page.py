"""Tests for compute_held_up_status validation logic."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from calendar_edge.app.streamlit_app import compute_held_up_status


class TestComputeHeldUpStatus:
    """Tests for compute_held_up_status function.

    The function returns True (validated) when:
    - test_delta > 0 (test WR > baseline)
    - test_n >= 20 (sufficient sample size)

    Otherwise returns False.
    """

    def test_no_test_wr_returns_false(self):
        """When test_wr is None, not validated."""
        result = compute_held_up_status(
            train_wr=0.55,
            train_baseline=0.53,
            test_wr=None,
            test_baseline=0.54,
            test_n=50,
        )
        assert result is False

    def test_no_test_n_returns_false(self):
        """When test_n is None, not validated."""
        result = compute_held_up_status(
            train_wr=0.55,
            train_baseline=0.53,
            test_wr=0.56,
            test_baseline=0.54,
            test_n=None,
        )
        assert result is False

    def test_zero_test_n_returns_false(self):
        """When test_n is 0, not validated."""
        result = compute_held_up_status(
            train_wr=0.55,
            train_baseline=0.53,
            test_wr=0.56,
            test_baseline=0.54,
            test_n=0,
        )
        assert result is False

    def test_positive_delta_sufficient_n_returns_true(self):
        """Positive test delta with n>=20 should validate."""
        result = compute_held_up_status(
            train_wr=0.60,
            train_baseline=0.53,
            test_wr=0.58,
            test_baseline=0.54,  # delta = 0.04 > 0
            test_n=50,
        )
        assert result is True

    def test_positive_delta_at_threshold_n_returns_true(self):
        """Positive test delta with exactly n=20 should validate."""
        result = compute_held_up_status(
            train_wr=0.60,
            train_baseline=0.53,
            test_wr=0.58,
            test_baseline=0.54,  # delta = 0.04 > 0
            test_n=20,
        )
        assert result is True

    def test_positive_delta_small_n_returns_false(self):
        """Positive test delta with n<20 should not validate."""
        result = compute_held_up_status(
            train_wr=0.60,
            train_baseline=0.53,
            test_wr=0.58,
            test_baseline=0.54,  # delta = 0.04 > 0
            test_n=19,  # < 20
        )
        assert result is False

    def test_zero_delta_returns_false(self):
        """Test WR exactly at baseline should not validate."""
        result = compute_held_up_status(
            train_wr=0.60,
            train_baseline=0.53,
            test_wr=0.54,  # exactly at test baseline
            test_baseline=0.54,  # delta = 0
            test_n=50,
        )
        assert result is False

    def test_negative_delta_returns_false(self):
        """Test WR below baseline should not validate."""
        result = compute_held_up_status(
            train_wr=0.60,
            train_baseline=0.53,
            test_wr=0.50,
            test_baseline=0.54,  # delta = -0.04 < 0
            test_n=50,
        )
        assert result is False

    def test_uses_train_baseline_when_test_baseline_none(self):
        """When test_baseline is None, should use train_baseline."""
        result = compute_held_up_status(
            train_wr=0.60,
            train_baseline=0.53,
            test_wr=0.56,  # delta vs train_baseline = 0.03 > 0
            test_baseline=None,
            test_n=50,
        )
        assert result is True

    def test_uses_train_baseline_negative_case(self):
        """Negative delta when using train_baseline fallback."""
        result = compute_held_up_status(
            train_wr=0.60,
            train_baseline=0.53,
            test_wr=0.50,  # delta vs train_baseline = -0.03 < 0
            test_baseline=None,
            test_n=50,
        )
        assert result is False
