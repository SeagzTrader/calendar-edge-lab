"""Reusable UI components for Calendar Edge Lab.

This module provides consistent widgets and formatters used across all pages.
"""

from datetime import datetime

import streamlit as st

# =============================================================================
# Color Palette
# =============================================================================
COLORS = {
    "bg": "#0e1117",
    "card": "#1a1d24",
    "border": "#2d3139",
    "text": "#fafafa",
    "secondary": "#9ca3af",
    "green": "#4ade80",
    "red": "#f87171",
    "badge_green": "#166534",
    "badge_orange": "#c2410c",
    "badge_red": "#991b1b",
    "badge_gray": "#374151",
}

# Month names for formatting
MONTH_NAMES = [
    "",
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]

MONTH_FULL_NAMES = [
    "",
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


# =============================================================================
# Date Formatting
# =============================================================================
def format_date_human(date_str: str) -> str:
    """Format date string to human-readable format.

    Args:
        date_str: Date in YYYY-MM-DD format.

    Returns:
        Date formatted as "Jan 1, 2024".
    """
    if not date_str:
        return ""
    try:
        dt = datetime.strptime(str(date_str)[:10], "%Y-%m-%d")
        return f"{MONTH_NAMES[dt.month]} {dt.day}, {dt.year}"
    except (ValueError, IndexError):
        return str(date_str)


def format_date_short(date_str: str) -> tuple[str, int]:
    """Format date to show day number and month.

    Args:
        date_str: Date in YYYY-MM-DD format.

    Returns:
        Tuple of (month_name, day_number).
    """
    if not date_str:
        return ("", 0)
    try:
        dt = datetime.strptime(str(date_str)[:10], "%Y-%m-%d")
        return (MONTH_NAMES[dt.month], dt.day)
    except (ValueError, IndexError):
        return ("", 0)


# =============================================================================
# Pattern Name Formatting
# =============================================================================
def ordinal(n: int) -> str:
    """Return ordinal string for a number (1st, 2nd, 3rd, etc.)."""
    if 11 <= n % 100 <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def format_pattern_name(family: str, key: dict) -> str:
    """Convert internal key to human-readable pattern name.

    Args:
        family: "TDOM" or "CDOY".
        key: Dictionary with pattern parameters.

    Returns:
        Human-readable name like "1st Trading Day of Month" or "Dec 26".
    """
    if family == "TDOM":
        tdom = key.get("tdom", 0)
        return f"{ordinal(tdom)} Trading Day of Month"
    else:  # CDOY
        month = key.get("month", 1)
        day = key.get("day", 1)
        return f"{MONTH_NAMES[month]} {day}"


def get_internal_code(family: str, key: dict) -> str:
    """Return the internal code string for a signal key.

    Examples:
        TDOM family, {"tdom": 1} -> "TDOM1"
        CDOY family, {"month": 12, "day": 26} -> "M12D26"
    """
    if family == "TDOM":
        return f"TDOM{key.get('tdom', 0)}"
    else:  # CDOY
        return f"M{key.get('month', 0):02d}D{key.get('day', 0):02d}"


def get_direction_label(direction: str) -> str:
    """Convert UP/DOWN to friendly direction label."""
    if direction == "UP":
        return "Closes Higher"
    elif direction == "DOWN":
        return "Closes Lower"
    return direction


# =============================================================================
# Badge System
# =============================================================================
def get_badge(
    is_validated: bool,
    avg_ret: float | None,
    decade_consistency: float | None,
    win_rate: float,
    baseline: float,
) -> tuple[str, str, str]:
    """Determine badge based on pattern quality.

    Args:
        is_validated: Whether pattern held up in verification.
        avg_ret: Average return (used as expectancy proxy).
        decade_consistency: Decade consistency score (0-1).
        win_rate: Pattern win rate.
        baseline: Baseline win rate.

    Returns:
        Tuple of (badge_text, badge_color, hex_color).
    """
    expectancy = avg_ret if avg_ret is not None else 0.0
    dc = decade_consistency if decade_consistency is not None else 0.0

    # Check for negative expectancy warning first
    if win_rate > (baseline + 0.05) and expectancy < 0:
        return ("NEG EXPECTANCY", "orange", COLORS["badge_orange"])

    # Check if validated
    if not is_validated:
        return ("DID NOT HOLD", "red", COLORS["badge_red"])

    # Check for robust edge
    if is_validated and expectancy > 0.0005 and dc >= 0.8:
        return ("ROBUST EDGE", "green", COLORS["badge_green"])

    # Default: statistical pattern
    return ("STATISTICAL PATTERN", "gray", COLORS["badge_gray"])


def render_badge(badge_text: str, hex_color: str) -> str:
    """Render badge as HTML span.

    Args:
        badge_text: Text to display in badge.
        hex_color: Background color.

    Returns:
        HTML string for badge.
    """
    return f'<span style="background-color:{hex_color};color:white;padding:2px 8px;border-radius:4px;font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;">{badge_text}</span>'


# =============================================================================
# Pattern Card Component
# =============================================================================
def render_pattern_card(
    pattern_name: str,
    direction: str,
    badge_text: str,
    badge_color: str,
    win_rate: float,
    baseline: float,
    avg_ret: float | None,
    verification_wr: float | None,
    verification_n: int | None,
    next_date: str | None,
    discovery_wr: float | None = None,
    on_click_key: str | None = None,
) -> None:
    """Render a pattern card with consistent styling."""
    edge = win_rate - baseline
    edge_sign = "+" if edge >= 0 else ""
    edge_bg = "rgba(74,222,128,0.2)" if edge >= 0 else "rgba(248,113,113,0.2)"
    edge_color = COLORS["green"] if edge >= 0 else COLORS["red"]

    # Build card parts
    parts = []

    # Header with badge
    badge_html = render_badge(badge_text, badge_color)
    header = (
        f'<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px;">'
        f'<div><div style="font-size:1.1rem;font-weight:600;color:{COLORS["text"]};">{pattern_name}</div>'
        f'<div style="font-size:0.85rem;color:{COLORS["secondary"]};">{get_direction_label(direction)}</div></div>'
        f'{badge_html}</div>'
    )
    parts.append(header)

    # Win rate with edge pill
    wr_html = (
        f'<div style="display:flex;align-items:baseline;gap:12px;margin-bottom:8px;">'
        f'<span style="font-size:2rem;font-weight:700;color:{COLORS["text"]};">{win_rate:.0%}</span>'
        f'<span style="background-color:{edge_bg};color:{edge_color};padding:2px 8px;border-radius:12px;font-size:0.8rem;font-weight:500;">'
        f'{edge_sign}{edge:.1%} vs baseline</span></div>'
    )
    parts.append(wr_html)

    # Avg return
    if avg_ret is not None:
        ret_pct = avg_ret * 100
        parts.append(f'<div style="font-size:0.85rem;color:{COLORS["secondary"]};margin-bottom:4px;">Avg Return: {ret_pct:+.2f}%</div>')

    # Verification stat
    if verification_wr is not None and verification_n is not None:
        disc_wr = discovery_wr if discovery_wr is not None else win_rate
        decay = disc_wr - verification_wr
        is_decaying = decay >= 0.10
        ver_color = "#f59e0b" if is_decaying else COLORS["secondary"]
        parts.append(f'<div style="font-size:0.85rem;color:{ver_color};margin-bottom:4px;">Verified: {verification_wr:.0%} since 2010 (n={verification_n})</div>')
        if is_decaying:
            parts.append('<div style="font-size:0.8rem;color:#f59e0b;margin-bottom:4px;">Pattern shows weakening in recent data</div>')

    # Next date
    if next_date:
        parts.append(f'<div style="font-size:0.85rem;color:{COLORS["secondary"]};">Next: {format_date_human(next_date)}</div>')

    # Wrap in card container
    card_style = f'background-color:{COLORS["card"]};border:1px solid {COLORS["border"]};border-radius:8px;padding:16px;margin-bottom:12px;'
    card_html = f'<div style="{card_style}">{"".join(parts)}</div>'

    st.markdown(card_html, unsafe_allow_html=True)

    # Optional click handler
    if on_click_key:
        if st.button("View Details", key=on_click_key, use_container_width=True):
            st.session_state["selected_signal_id"] = on_click_key.replace("btn_", "")
            st.session_state["page"] = "Details"
            st.rerun()


# =============================================================================
# Heatmap Cell Rendering
# =============================================================================
def get_heatmap_color(delta: float, is_validated: bool = False) -> tuple[str, str]:
    """Get heatmap cell color based on delta vs baseline.

    Args:
        delta: Win rate minus baseline.
        is_validated: Whether this is a validated signal.

    Returns:
        Tuple of (background_color, border_style).
    """
    # Color scale based on delta
    if delta > 0.10:
        bg = "#166534"  # Dark green >+10%
    elif delta > 0.05:
        bg = "#22c55e"  # Green +5-10%
    elif delta > -0.05:
        bg = "#374151"  # Gray ±5%
    elif delta > -0.10:
        bg = "#dc2626"  # Red -5-10%
    else:
        bg = "#991b1b"  # Dark red <-10%

    # White outline for validated signals
    border = "2px solid white" if is_validated else "1px solid #2d3139"

    return (bg, border)


# =============================================================================
# Index Formatting
# =============================================================================
SYMBOL_DISPLAY_NAMES = {
    "^GSPC": "S&P 500",
    "^DJI": "Dow Jones",
    "^IXIC": "Nasdaq",
}


def format_symbol(symbol: str) -> str:
    """Format symbol to display name."""
    return SYMBOL_DISPLAY_NAMES.get(symbol, symbol)


def get_symbol_tag_color(symbol: str) -> str:
    """Get color for symbol tag."""
    colors = {
        "^GSPC": "#3b82f6",  # Blue
        "^DJI": "#8b5cf6",  # Purple
        "^IXIC": "#06b6d4",  # Cyan
    }
    return colors.get(symbol, "#6b7280")
