"""Streamlit application for Calendar Edge Lab."""

import json

# Add src to path for imports
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from calendar_edge.config import DB_PATH, TEST_START, TRAIN_END
from calendar_edge.db import CalendarKeysRepo, PricesRepo, ReturnsRepo, RunsRepo, SignalsRepo
from calendar_edge.features import build_future_calendar_keys
from calendar_edge.validation import run_all_checks


def get_check_status(r) -> str:
    """Get status from CheckResult, supporting both old and new formats.

    Handles:
    - dataclass with .status (new)
    - dataclass with .passed bool (old)
    - dict with keys {"status": ...} or {"passed": ...}
    """
    # New format: has .status attribute
    if hasattr(r, "status"):
        return r.status
    # Dict with status key
    if isinstance(r, dict) and "status" in r:
        return r["status"]
    # Old format: has .passed bool
    if hasattr(r, "passed"):
        return "pass" if r.passed else "fail"
    # Dict with passed key
    if isinstance(r, dict) and "passed" in r:
        return "pass" if r["passed"] else "fail"
    # Default to fail
    return "fail"


def is_check_warning(r) -> bool:
    """Check if result is a warning/skip status."""
    status = get_check_status(r)
    return status in ("warn", "skip")


def get_check_message(r) -> str:
    """Get message from CheckResult."""
    if hasattr(r, "message"):
        return r.message
    if isinstance(r, dict) and "message" in r:
        return r["message"]
    return ""


def get_check_name(r) -> str:
    """Get name from CheckResult."""
    if hasattr(r, "name"):
        return r.name
    if isinstance(r, dict) and "name" in r:
        return r["name"]
    return "unknown"


def get_check_details(r) -> dict | None:
    """Get details from CheckResult."""
    if hasattr(r, "details"):
        return r.details
    if isinstance(r, dict) and "details" in r:
        return r["details"]
    return None


# Friendly symbol names
SYMBOL_NAMES = {
    "^GSPC": "S&P 500 (^GSPC)",
    "^DJI": "Dow Jones Industrial Average (^DJI)",
    "^IXIC": "Nasdaq Composite (^IXIC)",
}

# Short symbol names for sentences
SYMBOL_SHORT_NAMES = {
    "^GSPC": "the S&P 500",
    "^DJI": "the Dow Jones",
    "^IXIC": "the Nasdaq",
}

# Month names for plain-English descriptions
MONTH_NAMES = [
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

# Ordinal suffixes
def ordinal(n: int) -> str:
    """Return ordinal string for a number (1st, 2nd, 3rd, etc.)."""
    if 11 <= n % 100 <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def get_pattern_description(family: str, key: dict, direction: str) -> str:
    """Generate plain-English description of a pattern.

    Examples:
    - TDOM1 UP -> "The 1st trading day of each month tends to close higher."
    - M12D26 DOWN -> "December 26th tends to close lower."
    """
    if family == "TDOM":
        tdom = key.get("tdom", 0)
        dir_word = "higher" if direction == "UP" else "lower"
        return f"The {ordinal(tdom)} trading day of each month tends to close {dir_word}."
    else:  # CDOY
        month = key.get("month", 1)
        day = key.get("day", 1)
        month_name = MONTH_NAMES[month]
        dir_word = "higher" if direction == "UP" else "lower"
        return f"{month_name} {ordinal(day)} tends to close {dir_word}."


def get_pattern_label(family: str, key: dict, direction: str) -> str:
    """Generate short pattern label.

    Examples:
    - TDOM1 UP -> "TDOM1 UP"
    - M12D26 DOWN -> "Dec 26 DOWN"
    """
    if family == "TDOM":
        return f"TDOM{key.get('tdom')} {direction}"
    else:  # CDOY
        month = key.get("month", 1)
        day = key.get("day", 1)
        month_abbr = MONTH_NAMES[month][:3]
        return f"{month_abbr} {day} {direction}"


def get_held_up_badge(
    _train_wr: float,  # Kept for potential future use
    train_baseline: float,
    test_wr: float | None,
    test_baseline: float | None,
    test_n: int | None,
) -> tuple[str, str, str]:
    """Determine 'Held up?' badge based on test performance.

    Returns:
        Tuple of (icon, label, explanation)
    """
    if test_wr is None or test_n is None or test_n == 0:
        return ("❓", "No test data", "Not enough test data to evaluate.")

    test_delta = test_wr - (test_baseline or train_baseline)

    if test_delta > 0 and test_n >= 30:
        return ("✅", "Yes", f"Test win rate beat baseline by {test_delta:.1%} (n={test_n})")
    elif test_delta > 0 and test_n < 30:
        return ("⚠️", "Maybe", f"Positive in test (+{test_delta:.1%}) but small sample (n={test_n})")
    elif test_delta > -0.02:  # Within 2% of baseline
        return ("⚠️", "Unclear", f"Test win rate near baseline ({test_delta:+.1%}, n={test_n})")
    else:
        return ("❌", "No", f"Test win rate below baseline ({test_delta:+.1%}, n={test_n})")


def get_confidence_level(score: float | None) -> tuple[str, str]:
    """Map train score to confidence level.

    Returns:
        Tuple of (level, color) - level is "High", "Medium", or "Low"
    """
    if score is None:
        return ("Low", "gray")
    if score >= 2.0:
        return ("High", "green")
    elif score >= 1.0:
        return ("Medium", "orange")
    else:
        return ("Low", "gray")


# =============================================================================
# Signal Key Formatters
# =============================================================================
# These functions convert internal signal keys to human-readable labels.
# Internal keys (TDOM1, M12D26) are stored in DB and remain unchanged.


def get_internal_code(family: str, key: dict) -> str:
    """Return the internal code string for a signal key.

    Examples:
        TDOM family, {"tdom": 1} -> "TDOM1"
        CDOY family, {"month": 12, "day": 26} -> "M12D26"
    """
    if family == "TDOM":
        return f"TDOM{key.get('tdom', 0)}"
    else:  # CDOY
        month = key.get("month", 1)
        day = key.get("day", 1)
        return f"M{month:02d}D{day:02d}"


def format_signal_key(family: str, key: dict) -> str:
    """Convert internal signal key to human-readable label.

    Examples:
        TDOM1 -> "1st trading day of month"
        TDOM2 -> "2nd trading day of month"
        M12D26 -> "Dec 26"
        M01D02 -> "Jan 2"
    """
    if family == "TDOM":
        tdom = key.get("tdom", 0)
        return f"{ordinal(tdom)} trading day of month"
    else:  # CDOY
        month = key.get("month", 1)
        day = key.get("day", 1)
        month_abbr = MONTH_NAMES[month][:3]
        return f"{month_abbr} {day}"


def describe_signal_key(family: str, key: dict, direction: str) -> str:
    """Generate plain-English sentence describing the signal.

    Examples:
        TDOM1 UP -> "Tends to close up on this day"
        M12D26 DOWN -> "Tends to close down on this day"
    """
    dir_word = "up" if direction == "UP" else "down"
    return f"Tends to close {dir_word} on this day"


def describe_pattern_key(code: str) -> str:
    """Convert internal key string to friendly label.

    This is the main display helper for converting internal codes to
    human-readable labels throughout the UI.

    Examples:
        "TDOM1" -> "1st trading day of month"
        "TDOM10" -> "10th trading day of month"
        "M12D26" -> "Dec 26"
        "M01D02" -> "Jan 2"

    Returns:
        Friendly label string, or original code if parsing fails.
    """
    parsed = parse_internal_code(code)
    if parsed is None:
        return code
    family, key = parsed
    return format_signal_key(family, key)


def describe_direction(direction: str) -> str:
    """Convert UP/DOWN to friendly direction label.

    Examples:
        "UP" -> "Up day"
        "DOWN" -> "Down day"
    """
    if direction == "UP":
        return "Up day"
    elif direction == "DOWN":
        return "Down day"
    return direction


def format_short_date(date_str: str) -> str:
    """Format date string to short format (YYYY-MM-DD).

    Input can be various formats, output is always YYYY-MM-DD.
    """
    # If already in YYYY-MM-DD format, return as-is
    if len(date_str) == 10 and date_str[4] == "-" and date_str[7] == "-":
        return date_str
    # Try to parse and reformat
    try:
        from datetime import datetime as dt
        parsed = dt.strptime(date_str[:10], "%Y-%m-%d")
        return parsed.strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return date_str


def parse_internal_code(code: str) -> tuple[str, dict] | None:
    """Parse an internal code string back to family and key dict.

    This is useful for unit testing and potential future use.

    Examples:
        "TDOM1" -> ("TDOM", {"tdom": 1})
        "M12D26" -> ("CDOY", {"month": 12, "day": 26})

    Returns:
        Tuple of (family, key_dict) or None if parsing fails.
    """
    import re

    # Try TDOM pattern
    tdom_match = re.match(r"^TDOM(\d+)$", code)
    if tdom_match:
        return ("TDOM", {"tdom": int(tdom_match.group(1))})

    # Try CDOY pattern (MxxDyy)
    cdoy_match = re.match(r"^M(\d{1,2})D(\d{1,2})$", code)
    if cdoy_match:
        return ("CDOY", {"month": int(cdoy_match.group(1)), "day": int(cdoy_match.group(2))})

    return None


def get_held_up_label(status: str) -> tuple[str, str]:
    """Get display label and icon for held-up status.

    Returns:
        Tuple of (icon, label) for display.
    """
    labels = {
        "yes": ("✅", "Held up"),
        "unclear": ("⚠️", "Unclear"),
        "no": ("❌", "Did not hold"),
        "unknown": ("❓", "No test data"),
    }
    return labels.get(status, ("❓", status))


def format_symbol(symbol: str) -> str:
    """Return friendly symbol name."""
    return SYMBOL_NAMES.get(symbol, symbol)


def render_glossary():
    """Render a glossary expander with key terms."""
    with st.expander("Glossary / How to Read This Dashboard"):
        st.markdown("""
**Calendar Effect Families:**
- **CDOY** = Calendar Day of Year (e.g., Dec 26 = M12D26)
- **TDOM** = Trading Day of Month (e.g., 1st trading day = TDOM1)

**Windows:**
- **Train** = Discovery window (data through 2009-12-31). Signals are discovered and scored here.
- **Test** = Holdout window (data from 2010-01-01 onward). Out-of-sample evaluation.
- **Full** = All available data (context only, not used for scoring).

**Metrics:**
- **Win Rate** = Fraction of days with close-to-close return > 0 (0% return counts as not-up)
- **Baseline** = Mean win rate over the window for that symbol
- **Delta** = Win rate - Baseline (positive = beats baseline)
- **Z-Score** = Standard score measuring deviation from baseline
- **DCF** = Decade Consistency Factor (0.5-1.0); higher = more consistent across decades
- **FDR-Q** = Benjamini-Hochberg adjusted p-value (lower = more significant)
- **Score** = z_score × decade_consistency × (1 - fdr_q)

**Eligibility:** A signal is "eligible" if FDR-Q ≤ 10% and n ≥ 20 in the Train window.
        """)


def main():
    st.set_page_config(
        page_title="Calendar Edge Lab",
        page_icon="📅",
        layout="wide",
    )

    st.title("📅 Calendar Edge Lab")
    st.caption("Research-grade calendar effect discovery and validation")

    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Navigation",
        ["Home", "Calendar Heatmap", "Signal Detail", "Forward Calendar", "Diagnostics", "Methodology"],
    )

    # Global toggle for showing internal codes
    st.sidebar.markdown("---")
    show_internal_codes = st.sidebar.checkbox(
        "Show internal codes",
        value=False,
        help="Display internal signal codes (e.g., TDOM1, M12D26) alongside human-readable labels",
    )

    # Check database exists
    if not DB_PATH.exists():
        st.error(f"Database not found at {DB_PATH}. Run `python -m calendar_edge.cli init-db` first.")
        return

    runs_repo = RunsRepo()

    # Get latest run
    latest_run = runs_repo.get_latest_run()
    if not latest_run:
        st.warning("No scan runs found. Run `python -m calendar_edge.cli run-scan` first.")
        return

    run_id = latest_run["run_id"]
    st.sidebar.info(f"Run ID: {run_id}")

    # Add glossary to technical pages only (not Home, Methodology, or Diagnostics)
    if page in ("Calendar Heatmap", "Signal Detail", "Forward Calendar"):
        render_glossary()

    if page == "Home":
        render_home(run_id, show_internal_codes)
    elif page == "Calendar Heatmap":
        render_heatmap(run_id)
    elif page == "Signal Detail":
        render_signal_detail(run_id, show_internal_codes)
    elif page == "Forward Calendar":
        render_forward_calendar(run_id, show_internal_codes)
    elif page == "Diagnostics":
        render_diagnostics(run_id)
    elif page == "Methodology":
        render_methodology()


def compute_held_up_status(
    train_wr: float,
    train_baseline: float,
    test_wr: float | None,
    test_baseline: float | None,
    test_n: int | None,
) -> str:
    """Compute held-up status: 'yes', 'unclear', 'no', or 'unknown'.

    Returns:
        String status for filtering and confidence mapping.
    """
    if test_wr is None or test_n is None or test_n == 0:
        return "unknown"

    test_delta = test_wr - (test_baseline or train_baseline)

    if test_delta > 0 and test_n >= 30:
        return "yes"
    elif test_delta > 0 and test_n < 30:
        return "unclear"
    elif test_delta > -0.02:  # Within 2% of baseline
        return "unclear"
    else:
        return "no"


def render_home(run_id: str, show_internal_codes: bool = False):
    """Render the Home/Briefing page with non-technical summary."""

    # =========================================================================
    # REPOSITORIES
    # =========================================================================
    prices_repo = PricesRepo()
    returns_repo = ReturnsRepo()
    signals_repo = SignalsRepo()

    symbols = prices_repo.get_symbols()

    # =========================================================================
    # ABOVE THE FOLD: Index selector + baseline
    # =========================================================================
    selected_symbol = st.selectbox(
        "Select Index",
        symbols,
        format_func=format_symbol,
    )

    if not selected_symbol:
        st.warning("No symbols found in database.")
        return

    short_name = SYMBOL_SHORT_NAMES.get(selected_symbol, selected_symbol)

    # Get baseline data
    returns_df = returns_repo.get_returns(selected_symbol)
    train_returns = returns_df[returns_df["date"] <= TRAIN_END]
    test_returns = returns_df[returns_df["date"] >= TEST_START]

    full_wr = returns_df["up"].mean()
    train_wr = train_returns["up"].mean()
    test_wr = test_returns["up"].mean()
    train_baseline = train_wr
    test_baseline = test_wr

    # One-sentence baseline
    st.markdown(f"**{short_name.title()}** closes higher about **{full_wr:.0%}** of trading days.")

    # Collapsed "How it works"
    with st.expander("How it works", expanded=False):
        st.markdown("""
        1. **Discover** — Scan historical data for patterns that beat the baseline
        2. **Verify** — Test those patterns on data they've never seen
        3. **Filter** — Only show patterns that pass statistical controls

        Patterns that work in discovery but fail in verification are marked "Did not hold" or "Unclear".
        """)

    # Baseline details (collapsed)
    with st.expander("Baseline details"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Discovery period", f"{train_wr:.1%}", f"n={len(train_returns):,}")
        with col2:
            st.metric("Verification period", f"{test_wr:.1%}", f"n={len(test_returns):,}")
        with col3:
            st.metric("All time", f"{full_wr:.1%}", f"n={len(returns_df):,}")

    st.markdown("---")

    # =========================================================================
    # GET SIGNALS AND COMPUTE HELD-UP STATUS
    # =========================================================================
    all_eligible = signals_repo.get_signal_stats(run_id, "train", eligible_only=True)
    symbol_signals = all_eligible[all_eligible["symbol"] == selected_symbol]
    test_stats = signals_repo.get_signal_stats(run_id, "test")

    # Compute held-up status and cache test data for each signal
    signal_data_map = {}  # signal_id -> {held_up, test_wr, test_n, train_wr, train_n}
    held_up_counts = {"yes": 0, "unclear": 0, "no": 0, "unknown": 0}

    for _, sig_row in symbol_signals.iterrows():
        signal_id = sig_row["signal_id"]
        train_signal_wr = float(sig_row["win_rate"])
        train_n = int(sig_row["n"])

        test_row = test_stats[test_stats["signal_id"] == signal_id]
        if not test_row.empty:
            test_row = test_row.iloc[0]
            test_signal_wr = float(test_row["win_rate"])
            test_signal_n = int(test_row["n"])
        else:
            test_signal_wr = None
            test_signal_n = None

        status = compute_held_up_status(
            train_signal_wr, train_baseline,
            test_signal_wr, test_baseline,
            test_signal_n
        )
        held_up_counts[status] += 1

        signal_data_map[signal_id] = {
            "held_up": status,
            "train_wr": train_signal_wr,
            "train_n": train_n,
            "test_wr": test_signal_wr,
            "test_n": test_signal_n,
        }

    # Summary counts
    patterns_found = len(symbol_signals)
    held_up_count = held_up_counts["yes"]
    unclear_count = held_up_counts["unclear"]

    # Compact summary line
    st.caption(
        f"Found **{patterns_found}** patterns in discovery. "
        f"**{held_up_count}** held up in verification, **{unclear_count}** unclear."
    )

    st.markdown("---")

    # =========================================================================
    # TOP VALIDATED PATTERNS (cards)
    # =========================================================================
    st.subheader("Top Validated Patterns")

    # Filter options
    col_filter, col_mode = st.columns([1, 1])
    with col_filter:
        include_unclear = st.checkbox("Include unclear", value=False)
    with col_mode:
        mode_filter = st.selectbox(
            "Type",
            ["All patterns", "Calendar dates only", "Trading days only"],
            index=0,
            label_visibility="collapsed",
        )

    # Apply filters
    display_statuses = ["yes"]
    if include_unclear:
        display_statuses.append("unclear")

    patterns_to_show = []
    for _, row in symbol_signals.iterrows():
        signal_id = row["signal_id"]
        sig_data = signal_data_map.get(signal_id, {})

        if sig_data.get("held_up") not in display_statuses:
            continue

        # Mode filter
        family = row["family"]
        if mode_filter == "Calendar dates only" and family != "CDOY":
            continue
        if mode_filter == "Trading days only" and family != "TDOM":
            continue

        patterns_to_show.append(row)

    if not patterns_to_show:
        st.info("No patterns held up for this index. Try including unclear patterns.")
    else:
        # Sort by score descending, take top 3
        patterns_df = pd.DataFrame(patterns_to_show)
        patterns_df = patterns_df.sort_values("score", ascending=False).head(3)

        cols = st.columns(len(patterns_df))

        for i, (_, row) in enumerate(patterns_df.iterrows()):
            signal_id = row["signal_id"]
            sig_data = signal_data_map[signal_id]
            key = json.loads(row["key_json"]) if isinstance(row["key_json"], str) else row["key_json"]
            family = row["family"]
            direction = row["direction"]

            # Friendly labels
            pattern_name = format_signal_key(family, key)
            internal_code = get_internal_code(family, key)
            direction_label = describe_direction(direction)
            held_icon, held_label = get_held_up_label(sig_data["held_up"])

            # Stats
            train_wr = sig_data["train_wr"]
            train_n = sig_data["train_n"]
            train_delta = train_wr - train_baseline
            test_wr = sig_data["test_wr"]
            test_n = sig_data["test_n"]
            test_delta = (test_wr - test_baseline) if test_wr else None

            with cols[i]:
                # Card header
                if show_internal_codes:
                    st.markdown(f"**{pattern_name}** ({internal_code})")
                else:
                    st.markdown(f"**{pattern_name}**")
                st.caption(direction_label)

                # Stats
                st.markdown(f"**Discovery:** {train_wr:.0%} ({train_delta:+.0%}) · n={train_n}")
                if test_wr is not None:
                    st.markdown(f"**Verification:** {test_wr:.0%} ({test_delta:+.0%}) · n={test_n}")
                else:
                    st.markdown("**Verification:** No data")

                # Show avg return if available
                train_avg_ret = row.get("avg_ret")
                if train_avg_ret is not None:
                    ret_pct = train_avg_ret * 100
                    st.caption(f"Avg return: {ret_pct:+.2f}%")

                    # Warning for negative avg return despite high win rate
                    if train_avg_ret < 0:
                        st.warning("⚠️ Negative avg return")

                # Held up badge
                st.markdown(f"{held_icon} **{held_label}**")

                # View details button
                if st.button("View details", key=f"card_{signal_id}"):
                    st.session_state["selected_signal_id"] = signal_id
                    st.info("Go to 'Signal Detail' page for full analysis.")

    st.markdown("---")

    # =========================================================================
    # UPCOMING EVENTS
    # =========================================================================
    st.subheader("Upcoming Events")
    st.caption("Dates where a discovered pattern applies. For research only.")

    # Controls
    col_days, col_failed = st.columns([1, 1])
    with col_days:
        days_ahead = st.radio("Show next", [30, 90], horizontal=True, index=0, label_visibility="collapsed")
    with col_failed:
        include_failed = st.checkbox("Include patterns that did not hold", value=False)

    today = datetime.now().date()
    end_date = today + timedelta(days=days_ahead)
    future_keys = build_future_calendar_keys(today, end_date)

    # Apply mode filter to signals for events
    filtered_signals = symbol_signals.copy()
    if mode_filter == "Calendar dates only":
        filtered_signals = filtered_signals[filtered_signals["family"] == "CDOY"]
    elif mode_filter == "Trading days only":
        filtered_signals = filtered_signals[filtered_signals["family"] == "TDOM"]

    if filtered_signals.empty or future_keys.empty:
        st.info("No upcoming events in this period.")
    else:
        events = []

        for _, key_row in future_keys.iterrows():
            date_str = format_short_date(str(key_row["date"]))
            month = int(key_row["month"])
            day = int(key_row["day"])
            tdom = int(key_row["tdom"])

            for _, sig_row in filtered_signals.iterrows():
                key = json.loads(sig_row["key_json"]) if isinstance(sig_row["key_json"], str) else sig_row["key_json"]
                family = sig_row["family"]
                matched = False

                if family == "CDOY" and key.get("month") == month and key.get("day") == day:
                    matched = True
                elif family == "TDOM" and key.get("tdom") == tdom:
                    matched = True

                if matched:
                    signal_id = sig_row["signal_id"]
                    sig_data = signal_data_map.get(signal_id, {})
                    held_status = sig_data.get("held_up", "unknown")

                    # Filter based on held-up status
                    if held_status in ("no", "unknown") and not include_failed:
                        continue

                    direction = sig_row["direction"]
                    pattern_name = format_signal_key(family, key)
                    internal_code = get_internal_code(family, key)
                    held_icon, held_label = get_held_up_label(held_status)

                    if show_internal_codes:
                        pattern_display = f"{pattern_name} ({internal_code})"
                    else:
                        pattern_display = pattern_name

                    events.append({
                        "Date": date_str,
                        "Pattern": pattern_display,
                        "Direction": describe_direction(direction),
                        "Status": f"{held_icon} {held_label}",
                    })

        if events:
            events_df = pd.DataFrame(events).sort_values("Date")
            st.dataframe(
                events_df[["Date", "Pattern", "Direction", "Status"]],
                use_container_width=True,
                hide_index=True,
            )
            st.caption(f"{len(events)} events in the next {days_ahead} days.")
        else:
            st.info("No upcoming events in this period.")

    st.markdown("---")

    # =========================================================================
    # ADVANCED: ALL SIGNALS (collapsed)
    # =========================================================================
    with st.expander("Advanced: All validated patterns"):
        st.caption("Complete list of patterns that passed filters in discovery.")

        # Re-fetch without mode filter for complete list
        all_eligible_full = signals_repo.get_signal_stats(run_id, "train", eligible_only=True)
        symbol_signals_full = all_eligible_full[all_eligible_full["symbol"] == selected_symbol]

        for family_name, family_label in [("CDOY", "Calendar Dates"), ("TDOM", "Trading Days")]:
            st.markdown(f"**{family_label}**")

            family_signals = symbol_signals_full[symbol_signals_full["family"] == family_name]
            family_signals = family_signals.sort_values("score", ascending=False).head(20)

            if family_signals.empty:
                st.info(f"No {family_label.lower()} patterns found.")
                continue

            display_list = []
            for _, row in family_signals.iterrows():
                signal_id = row["signal_id"]
                key = json.loads(row["key_json"]) if isinstance(row["key_json"], str) else row["key_json"]
                sig_data = signal_data_map.get(signal_id, {})

                pattern_name = format_signal_key(family_name, key)
                internal_code = get_internal_code(family_name, key)
                direction = row["direction"]

                if show_internal_codes:
                    pattern_display = f"{pattern_name} ({internal_code})"
                else:
                    pattern_display = pattern_name

                train_wr = sig_data.get("train_wr", row["win_rate"])
                test_wr = sig_data.get("test_wr")
                held_status = sig_data.get("held_up", "unknown")
                held_icon, held_label = get_held_up_label(held_status)

                display_list.append({
                    "Pattern": pattern_display,
                    "Direction": describe_direction(direction),
                    "Discovery": f"{train_wr:.0%}",
                    "Verification": f"{test_wr:.0%}" if test_wr else "—",
                    "Status": f"{held_icon} {held_label}",
                })

            st.dataframe(pd.DataFrame(display_list), use_container_width=True, hide_index=True)


def render_heatmap(run_id: str):
    """Render calendar heatmap page.

    Computes heatmap from underlying data (returns_daily + calendar_keys),
    NOT from the signals table. This ensures all month/day combinations
    are shown regardless of whether they passed signal filters.
    """
    st.header("Calendar Heatmap")

    prices_repo = PricesRepo()
    keys_repo = CalendarKeysRepo()
    returns_repo = ReturnsRepo()
    signals_repo = SignalsRepo()

    symbols = prices_repo.get_symbols()
    selected_symbol = st.selectbox(
        "Select Symbol",
        symbols,
        format_func=format_symbol,
    )

    if not selected_symbol:
        return

    # Window selector
    window = st.radio("Window", ["Train", "Test", "Full"], horizontal=True)

    # Get calendar keys and returns
    keys_df = keys_repo.get_keys(selected_symbol)
    returns_df = returns_repo.get_returns(selected_symbol)

    if keys_df.empty or returns_df.empty:
        st.warning("No data available for this symbol.")
        return

    # Filter by window
    if window == "Train":
        keys_df = keys_df[keys_df["date"] <= TRAIN_END]
        returns_df = returns_df[returns_df["date"] <= TRAIN_END]
    elif window == "Test":
        keys_df = keys_df[keys_df["date"] >= TEST_START]
        returns_df = returns_df[returns_df["date"] >= TEST_START]
    # Full = no filter

    # Merge keys and returns on date
    merged = pd.merge(keys_df, returns_df, on=["symbol", "date"], how="inner")

    if merged.empty:
        st.warning("No data available for this window.")
        return

    # Compute baseline
    baseline = merged["up"].mean()

    # Group by month/day and compute stats
    grouped = merged.groupby(["month", "day"]).agg(
        n=("up", "count"),
        wins=("up", "sum"),
        win_rate=("up", "mean"),
    ).reset_index()

    grouped["delta"] = grouped["win_rate"] - baseline

    # Get eligible CDOY signals for overlay (optional)
    eligible_signals = set()
    if window == "Train":
        all_signals = signals_repo.get_signal_stats(run_id, "train", eligible_only=True)
        cdoy_signals = all_signals[
            (all_signals["symbol"] == selected_symbol) &
            (all_signals["family"] == "CDOY")
        ]
        for _, row in cdoy_signals.iterrows():
            key = json.loads(row["key_json"]) if isinstance(row["key_json"], str) else row["key_json"]
            eligible_signals.add((key.get("month"), key.get("day")))

    # Create heatmap data
    heatmap_data = {}
    for _, row in grouped.iterrows():
        month = int(row["month"])
        day = int(row["day"])
        heatmap_data[(month, day)] = {
            "win_rate": row["win_rate"],
            "delta": row["delta"],
            "n": int(row["n"]),
            "eligible": (month, day) in eligible_signals,
        }

    # Create grid
    st.markdown(f"**Win Rate Delta vs Baseline ({baseline:.2%}) — {window} Window**")
    st.caption("Green = above baseline, Red = below baseline. Bold border = eligible signal (FDR ≤ 10%).")

    # Build DataFrame for display
    months = list(range(1, 13))
    days = list(range(1, 32))

    grid_data = []
    for day in days:
        row_data = {"Day": day}
        for month in months:
            if (month, day) in heatmap_data:
                data = heatmap_data[(month, day)]
                delta = data["delta"]
                # Show delta with asterisk if eligible
                if data["eligible"]:
                    row_data[f"{month:02d}"] = f"{delta:+.1%}*"
                else:
                    row_data[f"{month:02d}"] = f"{delta:+.1%}"
            else:
                row_data[f"{month:02d}"] = ""
        grid_data.append(row_data)

    grid_df = pd.DataFrame(grid_data)
    grid_df = grid_df.set_index("Day")

    # Display with styling
    def color_delta(val):
        if val == "" or val is None:
            return ""
        try:
            # Remove asterisk and % for parsing
            clean_val = val.replace("%", "").replace("+", "").replace("*", "")
            delta = float(clean_val) / 100
            if delta > 0:
                intensity = min(abs(delta) * 5, 0.8)
                return f"background-color: rgba(0, 200, 0, {intensity:.2f})"
            elif delta < 0:
                intensity = min(abs(delta) * 5, 0.8)
                return f"background-color: rgba(200, 0, 0, {intensity:.2f})"
        except (ValueError, AttributeError):
            pass
        return ""

    styled_df = grid_df.style.applymap(color_delta)
    st.dataframe(styled_df, use_container_width=True, height=800)

    # Stats summary
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Days", len(merged))
    with col2:
        st.metric("Baseline Win Rate", f"{baseline:.2%}")
    with col3:
        st.metric("Month/Day Combinations", len(heatmap_data))
    with col4:
        st.metric("Eligible Signals", len(eligible_signals))


def render_signal_detail(run_id: str, show_internal_codes: bool = False):
    """Render signal detail page."""
    st.header("Pattern Details")

    signals_repo = SignalsRepo()
    prices_repo = PricesRepo()
    returns_repo = ReturnsRepo()

    symbols = prices_repo.get_symbols()
    selected_symbol = st.selectbox(
        "Select Index",
        symbols,
        format_func=format_symbol,
    )

    if not selected_symbol:
        return

    # Get baseline for context
    returns_df = returns_repo.get_returns(selected_symbol)
    train_returns = returns_df[returns_df["date"] <= TRAIN_END]
    test_returns = returns_df[returns_df["date"] >= TEST_START]
    train_baseline_wr = train_returns["up"].mean()
    test_baseline_wr = test_returns["up"].mean()
    train_baseline_ret = train_returns["ret_cc"].mean() if "ret_cc" in train_returns.columns else None
    test_baseline_ret = test_returns["ret_cc"].mean() if "ret_cc" in test_returns.columns else None

    # Get all signals for symbol
    all_signals = signals_repo.get_signal_stats(run_id, "train")
    symbol_signals = all_signals[all_signals["symbol"] == selected_symbol]

    if symbol_signals.empty:
        st.warning("No patterns found for this index.")
        return

    # Build signal options using formatters
    signal_options = []
    for _, row in symbol_signals.iterrows():
        key = json.loads(row["key_json"]) if isinstance(row["key_json"], str) else row["key_json"]
        family = row["family"]
        pattern_name = format_signal_key(family, key)
        internal_code = get_internal_code(family, key)
        direction = row["direction"]
        direction_label = describe_direction(direction)

        if show_internal_codes:
            base_label = f"{pattern_name} ({internal_code}) - {direction_label}"
        else:
            base_label = f"{pattern_name} - {direction_label}"

        signal_options.append((row["signal_id"], base_label, row.get("score", 0) or 0))

    # Sort by score
    signal_options.sort(key=lambda x: x[2], reverse=True)

    selected_signal = st.selectbox(
        "Select Pattern",
        options=[s[0] for s in signal_options],
        format_func=lambda x: next(s[1] for s in signal_options if s[0] == x),
    )

    if not selected_signal:
        return

    # Get all window stats for this signal
    signal_data = all_signals[all_signals["signal_id"] == selected_signal].iloc[0]

    key = json.loads(signal_data["key_json"]) if isinstance(signal_data["key_json"], str) else signal_data["key_json"]
    family = signal_data["family"]
    direction = signal_data["direction"]

    # Friendly labels
    pattern_name = format_signal_key(family, key)
    internal_code = get_internal_code(family, key)
    direction_label = describe_direction(direction)
    pattern_description = get_pattern_description(family, key, direction)

    # Header with pattern info
    st.markdown("---")
    if show_internal_codes:
        st.subheader(f"{pattern_name} ({internal_code})")
    else:
        st.subheader(pattern_name)
    st.markdown(f"**{direction_label}** — {pattern_description}")

    # Get test stats
    test_stats = signals_repo.get_signal_stats(run_id, "test")
    test_row = test_stats[test_stats["signal_id"] == selected_signal]

    # Compute held-up status
    train_wr = float(signal_data["win_rate"])
    train_n = int(signal_data["n"])
    if not test_row.empty:
        test_data = test_row.iloc[0]
        test_wr = float(test_data["win_rate"])
        test_n = int(test_data["n"])
        test_avg_ret = test_data.get("avg_ret")
    else:
        test_wr = None
        test_n = None
        test_avg_ret = None

    held_status = compute_held_up_status(train_wr, train_baseline_wr, test_wr, test_baseline_wr, test_n)
    held_icon, held_label = get_held_up_label(held_status)

    # Main performance cards
    st.markdown("### Performance Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Discovery Period**")
        train_delta = train_wr - train_baseline_wr
        st.metric("Hit Rate", f"{train_wr:.1%}", f"{train_delta:+.1%} vs baseline")
        st.caption(f"n = {train_n} days")

        # Show avg return if available
        train_avg_ret = signal_data.get("avg_ret")
        if train_avg_ret and train_baseline_ret:
            ret_bps = train_avg_ret * 10000
            baseline_bps = train_baseline_ret * 10000
            st.caption(f"Avg return: {ret_bps:+.1f} bps (baseline: {baseline_bps:+.1f} bps)")

    with col2:
        st.markdown("**Verification Period**")
        if test_wr is not None:
            test_delta = test_wr - test_baseline_wr
            st.metric("Hit Rate", f"{test_wr:.1%}", f"{test_delta:+.1%} vs baseline")
            st.caption(f"n = {test_n} days")

            if test_avg_ret and test_baseline_ret:
                ret_bps = test_avg_ret * 10000
                baseline_bps = test_baseline_ret * 10000
                st.caption(f"Avg return: {ret_bps:+.1f} bps (baseline: {baseline_bps:+.1f} bps)")
        else:
            st.metric("Hit Rate", "—")
            st.caption("No data available")

    with col3:
        st.markdown("**Held Up?**")
        st.markdown(f"### {held_icon} {held_label}")
        if held_status == "yes":
            st.caption("Pattern persisted in verification data")
        elif held_status == "unclear":
            st.caption("Results inconclusive (small sample or near baseline)")
        elif held_status == "no":
            st.caption("Pattern did not persist in verification")
        else:
            st.caption("Insufficient verification data")

    # Decade breakdown section
    st.markdown("---")
    st.markdown("### Decade Breakdown")
    st.caption("Performance by decade during the discovery period (before 2010).")

    # Get all window stats for this signal to find decade_* windows
    all_window_stats = signals_repo.get_signal_stats(run_id)
    signal_all_windows = all_window_stats[all_window_stats["signal_id"] == selected_signal]

    # Filter to decade windows
    decade_rows = signal_all_windows[signal_all_windows["window"].str.startswith("decade_")]

    if decade_rows.empty:
        st.info("No decade breakdown available. Re-run scan to generate.")
    else:
        decade_data = []
        decades_with_edge = 0
        total_decades = 0

        for _, row in decade_rows.iterrows():
            window = row["window"]  # e.g., "decade_1950s"
            decade_label = window.replace("decade_", "").replace("s", "s")  # "1950s"

            win_rate = row["win_rate"]
            n = row["n"]
            delta = win_rate - train_baseline_wr

            # Checkmark if win rate > baseline
            if delta > 0:
                status = "✅"
                decades_with_edge += 1
            else:
                status = "—"

            total_decades += 1

            decade_data.append({
                "Decade": decade_label,
                "Hit Rate": f"{win_rate:.1%}",
                "vs Baseline": f"{delta:+.1%}",
                "N": n,
                "Edge": status,
            })

        # Sort by decade
        decade_data.sort(key=lambda x: x["Decade"])

        # Summary
        st.markdown(f"**{decades_with_edge} of {total_decades}** decades showed positive edge.")

        st.dataframe(pd.DataFrame(decade_data), use_container_width=True, hide_index=True)

    # Eligibility status
    st.markdown("---")
    if signal_data["eligible"]:
        st.success("✓ This pattern passed statistical filters in discovery")
    else:
        st.warning("This pattern did not pass statistical filters")

    # Advanced stats (collapsed)
    with st.expander("Advanced: Statistical Details"):
        st.caption("Technical metrics from the discovery period.")

        # Get full stats
        full_stats = signals_repo.get_signal_stats(run_id, "full")
        full_row = full_stats[full_stats["signal_id"] == selected_signal]

        comparison_data = []

        # Discovery
        comparison_data.append({
            "Period": "Discovery",
            "N": signal_data["n"],
            "Wins": signal_data["wins"],
            "Hit Rate": f"{signal_data['win_rate']:.2%}",
            "Avg Return": f"{signal_data['avg_ret']:.4%}" if signal_data['avg_ret'] else "—",
            "Z-Score": f"{signal_data['z_score']:.2f}" if signal_data['z_score'] else "—",
            "Decade Consistency": f"{signal_data['decade_consistency']:.2f}" if signal_data['decade_consistency'] else "—",
            "FDR-Q": f"{signal_data['fdr_q']:.3f}" if signal_data['fdr_q'] else "—",
        })

        # Verification
        if not test_row.empty:
            comparison_data.append({
                "Period": "Verification",
                "N": test_data["n"],
                "Wins": test_data["wins"],
                "Hit Rate": f"{test_data['win_rate']:.2%}",
                "Avg Return": f"{test_data['avg_ret']:.4%}" if test_data['avg_ret'] else "—",
                "Z-Score": "—",
                "Decade Consistency": "—",
                "FDR-Q": "—",
            })

        # Full
        if not full_row.empty:
            full_data = full_row.iloc[0]
            comparison_data.append({
                "Period": "All Time",
                "N": full_data["n"],
                "Wins": full_data["wins"],
                "Hit Rate": f"{full_data['win_rate']:.2%}",
                "Avg Return": f"{full_data['avg_ret']:.4%}" if full_data['avg_ret'] else "—",
                "Z-Score": "—",
                "Decade Consistency": "—",
                "FDR-Q": "—",
            })

        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)


def render_forward_calendar(run_id: str, show_internal_codes: bool = False):
    """Render forward calendar page.

    Shows upcoming dates where eligible signals fire.
    Includes BOTH CDOY and TDOM signals.
    Uses NYSE trading calendar (exchange-calendars) for accurate session dates.
    """
    st.header("Upcoming Pattern Dates")
    st.caption("Trading sessions where discovered patterns apply. For research only.")

    signals_repo = SignalsRepo()
    prices_repo = PricesRepo()
    returns_repo = ReturnsRepo()

    symbols = prices_repo.get_symbols()

    # Controls
    col1, col2 = st.columns(2)
    with col1:
        selected_symbols = st.multiselect(
            "Indices",
            symbols,
            default=symbols,
            format_func=format_symbol,
        )
    with col2:
        num_days = st.slider("Days Ahead", 30, 180, 60)

    if not selected_symbols:
        st.warning("Please select at least one index.")
        return

    # Get eligible signals from discovery window
    all_signals = signals_repo.get_signal_stats(run_id, "train", eligible_only=True)

    if all_signals.empty:
        st.warning("No validated patterns found.")
        return

    # Filter by selected symbols
    filtered_signals = all_signals[all_signals["symbol"].isin(selected_symbols)]

    if filtered_signals.empty:
        st.warning("No validated patterns for selected indices.")
        return

    # Get test stats and compute held-up status for each signal
    test_stats = signals_repo.get_signal_stats(run_id, "test")

    signal_status_map = {}  # signal_id -> held_up status
    for _, sig_row in filtered_signals.iterrows():
        signal_id = sig_row["signal_id"]
        symbol = sig_row["symbol"]

        # Get baseline for this symbol
        returns_df = returns_repo.get_returns(symbol)
        train_returns = returns_df[returns_df["date"] <= TRAIN_END]
        test_returns = returns_df[returns_df["date"] >= TEST_START]
        train_baseline = train_returns["up"].mean()
        test_baseline = test_returns["up"].mean()

        train_wr = float(sig_row["win_rate"])
        test_row = test_stats[test_stats["signal_id"] == signal_id]

        if not test_row.empty:
            test_wr = float(test_row.iloc[0]["win_rate"])
            test_n = int(test_row.iloc[0]["n"])
        else:
            test_wr = None
            test_n = None

        status = compute_held_up_status(train_wr, train_baseline, test_wr, test_baseline, test_n)
        signal_status_map[signal_id] = status

    # Build future calendar keys using NYSE trading calendar
    today = datetime.now().date()
    end_date = today + timedelta(days=num_days)

    future_keys = build_future_calendar_keys(today, end_date)

    if future_keys.empty:
        st.warning("No trading sessions found in the selected date range.")
        return

    # Build forward calendar data
    calendar_data = []

    for _, key_row in future_keys.iterrows():
        date_str = format_short_date(str(key_row["date"]))
        month = int(key_row["month"])
        day = int(key_row["day"])
        tdom = int(key_row["tdom"])

        # Check each symbol's signals
        for symbol in selected_symbols:
            symbol_signals = filtered_signals[filtered_signals["symbol"] == symbol]

            for _, sig_row in symbol_signals.iterrows():
                key = json.loads(sig_row["key_json"]) if isinstance(sig_row["key_json"], str) else sig_row["key_json"]
                family = sig_row["family"]
                matched = False

                if family == "CDOY" and key.get("month") == month and key.get("day") == day:
                    matched = True
                elif family == "TDOM" and key.get("tdom") == tdom:
                    matched = True

                if matched:
                    signal_id = sig_row["signal_id"]

                    # Use formatters for signal display
                    pattern_name = format_signal_key(family, key)
                    internal_code = get_internal_code(family, key)
                    direction = sig_row["direction"]

                    if show_internal_codes:
                        signal_display = f"{pattern_name} ({internal_code})"
                    else:
                        signal_display = pattern_name

                    # Get held-up status
                    held_status = signal_status_map.get(signal_id, "unknown")
                    held_icon, held_label = get_held_up_label(held_status)

                    calendar_data.append({
                        "Date": date_str,
                        "Index": SYMBOL_SHORT_NAMES.get(symbol, symbol),
                        "Pattern": signal_display,
                        "Direction": describe_direction(direction),
                        "Status": f"{held_icon} {held_label}",
                    })

    if calendar_data:
        # Sort by date
        calendar_df = pd.DataFrame(calendar_data)
        calendar_df = calendar_df.sort_values("Date")
        st.dataframe(calendar_df, use_container_width=True, hide_index=True, height=500)

        st.caption(f"{len(calendar_data)} pattern events across {len(future_keys)} trading sessions.")
    else:
        st.info("No pattern events in the selected date range.")


def render_diagnostics(run_id: str):
    """Render diagnostics page showing validation check results."""
    st.header("Diagnostics")
    st.caption("Validation checks for data integrity and calendar correctness.")

    with st.spinner("Running validation checks..."):
        results = run_all_checks(run_id)

    # Count by status using helper function for compatibility
    passed = sum(1 for r in results if get_check_status(r) == "pass")
    warned = sum(1 for r in results if get_check_status(r) in ("warn", "skip"))
    failed = sum(1 for r in results if get_check_status(r) == "fail")
    total = len(results)

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Checks", total)
    with col2:
        st.metric("Passed", passed)
    with col3:
        if warned > 0:
            st.metric("Warnings", warned)
        else:
            st.metric("Warnings", 0)
    with col4:
        if failed > 0:
            st.metric("Failed", failed, delta=f"-{failed}", delta_color="inverse")
        else:
            st.metric("Failed", 0)

    # Overall status
    if failed == 0 and warned == 0:
        st.success("All validation checks passed.")
    elif failed == 0:
        st.warning(f"All checks passed, but {warned} warning(s). See details below.")
    else:
        st.error(f"{failed} check(s) failed. See details below.")

    st.markdown("---")

    # Group results by category
    categories = {}
    for r in results:
        name = get_check_name(r)
        category = name.split("_")[0]
        if category not in categories:
            categories[category] = []
        categories[category].append(r)

    # Category descriptions
    category_info = {
        "data": ("Data Coverage", "Checks price and returns data counts and date ranges for each symbol."),
        "no": ("Trading Calendar", "Verifies NYSE trading calendar excludes weekends."),
        "jan1": ("Holiday Exclusion", "Confirms Jan 1, 2026 (holiday) is excluded from sessions."),
        "tdom1": ("TDOM1 Correctness", "Verifies TDOM1 Jan 2026 = Jan 2 (first trading day after holiday)."),
        "tdom2": ("TDOM2 Correctness", "Verifies TDOM2 Jan 2026 = Jan 5 (second trading day)."),
        "heatmap": ("Heatmap Sanity", "Checks each symbol has sufficient month/day cells with n >= 20."),
        "signal": ("Signal Detail", "Validates top signal train/test/full statistics consistency."),
        "forward": ("Forward Coverage", "Shows eligible signals and upcoming calendar matches."),
    }

    # Status display: icon + label
    status_display = {
        "pass": ("✅", "PASS"),
        "fail": ("❌", "FAIL"),
        "warn": ("⚠️", "WARN"),
        "skip": ("⏭️", "SKIP"),
    }

    for category, check_results in categories.items():
        title, desc = category_info.get(category, (category.upper(), ""))

        # Expand if any failures or warnings
        has_issues = any(get_check_status(r) in ("fail", "warn", "skip") for r in check_results)

        with st.expander(f"{title} ({len(check_results)} checks)", expanded=has_issues):
            if desc:
                st.caption(desc)

            for r in check_results:
                status = get_check_status(r)
                name = get_check_name(r)
                message = get_check_message(r)
                details = get_check_details(r)

                icon, label = status_display.get(status, ("❓", status.upper()))
                st.markdown(f"{icon} **[{label}]** {name}: {message}")

                # Show details if available
                if details:
                    with st.container():
                        detail_cols = st.columns(min(len(details), 4))
                        items = list(details.items())
                        for i, (key, value) in enumerate(items[:4]):
                            with detail_cols[i]:
                                if isinstance(value, float):
                                    st.metric(key, f"{value:.4f}")
                                elif isinstance(value, list):
                                    st.metric(key, ", ".join(str(v) for v in value[:3]) or "None")
                                else:
                                    st.metric(key, str(value))

    # Re-run button
    st.markdown("---")
    if st.button("Re-run Validation Checks"):
        st.rerun()


def render_methodology():
    """Render methodology page."""
    st.header("Methodology")

    st.markdown("""
    ## Walk-Forward Validation

    Calendar Edge Lab uses walk-forward validation to discover and validate calendar effects:

    1. **Discovery Phase (Train)**: Data through 2009-12-31
       - Scan for calendar patterns (CDOY, TDOM)
       - Apply statistical filters (n ≥ 20, delta ≥ 5%)
       - Compute z-scores and decade consistency
       - Apply Benjamini-Hochberg FDR correction at 10%

    2. **Evaluation Phase (Test)**: Data from 2010-01-01 onward
       - Evaluate discovered signals on unseen data
       - Compare train vs test performance
       - Identify signals that persist out-of-sample

    ## Calendar Effect Families

    ### CDOY (Calendar Day of Year)
    Month + Day combinations (e.g., January 15, December 26)

    **Example**: M12D26 = December 26th

    ### TDOM (Trading Day of Month)
    Trading day number within each month (e.g., 1st trading day, 2nd trading day)

    **Example**: TDOM1 = First trading day of the month

    ## Statistical Controls

    ### Benjamini-Hochberg FDR Correction
    Controls the expected proportion of false discoveries among all discoveries.
    We use a 10% threshold, meaning we expect at most 10% of our significant
    signals to be false positives.

    ### Wilson Confidence Intervals
    More accurate confidence intervals for proportions, especially for
    small sample sizes.

    ### Decade Consistency Factor
    Measures whether the signal shows consistent direction across different decades.
    Higher values indicate more robust signals.

    ## Scoring Formula

    ```
    score = z_score × decade_consistency × (1 - fdr_q)
    ```

    Where:
    - **z_score**: Standard score measuring deviation from baseline
    - **decade_consistency**: 0.5 to 1.0, based on consistency across decades
    - **fdr_q**: FDR-adjusted p-value (lower = more significant)

    ## Limitations

    - **Historical patterns may not persist**: Market dynamics change over time
    - **No transaction costs**: Real trading involves costs not modeled here
    - **Cash indices only**: Futures and ETFs may behave differently
    - **Survivorship bias**: We only analyze indices that still exist
    - **Look-ahead bias**: Care taken to avoid, but always a risk

    ## Data Sources

    - **yfinance**: Yahoo Finance API for price data
    - **Adjusted closes**: Used for accurate return calculations
    """)


if __name__ == "__main__":
    main()
