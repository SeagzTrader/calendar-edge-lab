"""Scoring functions for calendar signals."""

from calendar_edge.config import MIN_DECADE_N


def compute_decade_consistency(
    decade_deltas: dict[str, float],
    overall_delta_sign: int,
    min_n: int = MIN_DECADE_N,
) -> float:
    """Compute decade consistency factor.

    Args:
        decade_deltas: Dict mapping decade label to delta value for that decade.
        overall_delta_sign: Sign of the overall delta (+1 or -1).
        min_n: Minimum observations per decade (not used here, filtering done upstream).

    Returns:
        Decade consistency factor (0.5 to 1.0).
    """
    if not decade_deltas:
        return 0.75  # Default when no decade data

    valid_decades = [d for d in decade_deltas.values() if d != 0]

    if len(valid_decades) < 2:
        return 0.75  # Default when fewer than 2 valid decades

    same_sign_count = sum(1 for d in valid_decades if (d > 0) == (overall_delta_sign > 0))
    share_same_sign = same_sign_count / len(valid_decades)

    dcf = 0.5 + 0.5 * share_same_sign
    return dcf


def compute_score(
    z_score: float,
    decade_consistency: float,
    fdr_q: float | None,
) -> float:
    """Compute final signal score.

    Args:
        z_score: Z-score of the signal.
        decade_consistency: Decade consistency factor (0.5-1.0).
        fdr_q: FDR-adjusted q-value (None means 1.0).

    Returns:
        Final score.
    """
    q = fdr_q if fdr_q is not None else 1.0
    score = z_score * decade_consistency * (1 - q)
    return score
