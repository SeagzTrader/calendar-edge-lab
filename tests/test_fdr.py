"""Tests for FDR correction."""

import numpy as np

from calendar_edge.scan.fdr import apply_fdr


def test_apply_fdr_basic():
    """Test basic FDR correction."""
    p_values = [0.01, 0.02, 0.03, 0.04]
    q_values = apply_fdr(p_values)

    # Should return same length
    assert len(q_values) == len(p_values)

    # Q-values should be >= p-values
    for p, q in zip(p_values, q_values):
        assert q >= p


def test_apply_fdr_empty():
    """Test FDR with empty input."""
    q_values = apply_fdr([])
    assert q_values == []


def test_apply_fdr_single():
    """Test FDR with single p-value."""
    q_values = apply_fdr([0.05])
    assert len(q_values) == 1
    assert q_values[0] == 0.05  # Single value unchanged


def test_apply_fdr_with_nans():
    """Test FDR handles NaN values."""
    p_values = [0.01, np.nan, 0.03]
    q_values = apply_fdr(p_values)

    assert len(q_values) == 3
    assert not np.isnan(q_values[0])
    assert np.isnan(q_values[1])
    assert not np.isnan(q_values[2])


def test_apply_fdr_matches_expected():
    """Test FDR matches expected BH correction results."""
    # Known example from BH procedure
    # p-values: [0.01, 0.04, 0.03, 0.02]
    # Sorted: [0.01, 0.02, 0.03, 0.04]
    # Ranks: [1, 2, 3, 4]
    # m = 4
    # BH adjusted: p * m / rank
    # [0.01 * 4/1, 0.02 * 4/2, 0.03 * 4/3, 0.04 * 4/4]
    # = [0.04, 0.04, 0.04, 0.04]
    # After cummin from right: [0.04, 0.04, 0.04, 0.04]

    p_values = [0.01, 0.04, 0.03, 0.02]
    q_values = apply_fdr(p_values)

    # All should be 0.04
    assert abs(q_values[0] - 0.04) < 1e-10
    assert abs(q_values[1] - 0.04) < 1e-10
    assert abs(q_values[2] - 0.04) < 1e-10
    assert abs(q_values[3] - 0.04) < 1e-10


def test_apply_fdr_caps_at_one():
    """Test that q-values are capped at 1.0."""
    p_values = [0.5, 0.6, 0.7]
    q_values = apply_fdr(p_values)

    # All q-values should be <= 1.0
    for q in q_values:
        assert q <= 1.0


def test_apply_fdr_preserves_order():
    """Test that smaller p-values get smaller or equal q-values."""
    p_values = [0.001, 0.01, 0.05, 0.10]
    q_values = apply_fdr(p_values)

    # q-values should maintain relative ordering
    # (smaller p -> smaller or equal q)
    assert q_values[0] <= q_values[1]
    assert q_values[1] <= q_values[2]
    assert q_values[2] <= q_values[3]


def test_apply_fdr_monotonic():
    """Test that q-values are monotonic when sorted by p-values."""
    p_values = [0.02, 0.01, 0.05, 0.03, 0.04]
    q_values = apply_fdr(p_values)

    # Sort by p-value and check q-values are monotonic
    sorted_pairs = sorted(zip(p_values, q_values), key=lambda x: x[0])
    sorted_q = [pair[1] for pair in sorted_pairs]

    for i in range(1, len(sorted_q)):
        assert sorted_q[i] >= sorted_q[i - 1]
