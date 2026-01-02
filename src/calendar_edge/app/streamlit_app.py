"""Calendar Edge Lab - Pattern Discovery Dashboard.

A research product for exploring validated calendar effects in equity indices.
"""

import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from calendar_edge.app.ui.components import (
    COLORS,
    MONTH_FULL_NAMES,
    MONTH_NAMES,
    format_date_human,
    format_pattern_name,
    format_symbol,
    get_badge,
    get_direction_label,
    get_internal_code,
    get_symbol_tag_color,
    render_badge,
    render_pattern_card,
)
from calendar_edge.config import (
    FDR_PROMISING,
    FDR_VALIDATED,
    MIN_N,
    P_EXPLORATORY,
    TEST_START,
    TRAIN_END,
)
from calendar_edge.db import (
    CalendarKeysRepo,
    PricesRepo,
    ReturnsRepo,
    RunsRepo,
    SignalsRepo,
)
from calendar_edge.features import build_future_calendar_keys

# =============================================================================
# Page Configuration
# =============================================================================
st.set_page_config(
    page_title="Calendar Edge Lab",
    page_icon="📅",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for styling
st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {COLORS['bg']};
    }}
    .main .block-container {{
        padding-top: 2rem;
        max-width: 1200px;
    }}
    h1, h2, h3 {{
        color: {COLORS['text']};
    }}
    .stSelectbox label, .stMultiSelect label {{
        color: {COLORS['secondary']};
    }}
    .stButton > button {{
        background-color: {COLORS['card']};
        border: 1px solid {COLORS['border']};
        color: {COLORS['text']};
    }}
    .stButton > button:hover {{
        background-color: {COLORS['border']};
        border: 1px solid {COLORS['secondary']};
    }}
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background-color: {COLORS['card']};
        padding: 8px;
        border-radius: 8px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: transparent;
        border-radius: 4px;
        color: {COLORS['secondary']};
        padding: 8px 16px;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {COLORS['border']};
        color: {COLORS['text']};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Data Loading Utilities
# =============================================================================
@st.cache_data(ttl=300)
def load_run_data():
    """Load the latest run data from database."""
    runs_repo = RunsRepo()
    run = runs_repo.get_latest_run()
    if not run:
        return None, None, None

    run_id = run["run_id"]
    signals_repo = SignalsRepo()

    # Load all window stats
    train_stats = signals_repo.get_signal_stats(run_id, "train")
    test_stats = signals_repo.get_signal_stats(run_id, "test")

    return run_id, train_stats, test_stats


@st.cache_data(ttl=300)
def load_baseline_data():
    """Load baseline win rates for each symbol."""
    returns_repo = ReturnsRepo()
    prices_repo = PricesRepo()
    symbols = prices_repo.get_symbols()

    baselines = {}
    for symbol in symbols:
        returns_df = returns_repo.get_returns(symbol)
        train_returns = returns_df[returns_df["date"] <= TRAIN_END]
        test_returns = returns_df[returns_df["date"] >= TEST_START]

        baselines[symbol] = {
            "train": train_returns["up"].mean() if len(train_returns) > 0 else 0.5,
            "test": test_returns["up"].mean() if len(test_returns) > 0 else 0.5,
        }

    return baselines


@st.cache_data(ttl=300)
def load_raw_heatmap_data(symbol: str):
    """Compute win rates from raw data for ALL CDOY and TDOM cells.

    Returns dict with:
        - cdoy_grid: {(month, day): {"wr": float, "n": int, "delta": float}}
        - tdom_grid: {tdom: {"wr": float, "n": int, "delta": float}}
        - baseline: float
    """
    returns_repo = ReturnsRepo()
    calendar_keys_repo = CalendarKeysRepo()

    # Get returns and calendar keys
    returns_df = returns_repo.get_returns(symbol)
    calendar_keys_df = calendar_keys_repo.get_keys(symbol)

    if returns_df.empty or calendar_keys_df.empty:
        return {"cdoy_grid": {}, "tdom_grid": {}, "baseline": 0.5}

    # Filter to train period only
    returns_df = returns_df[returns_df["date"] <= TRAIN_END]
    calendar_keys_df = calendar_keys_df[calendar_keys_df["date"] <= TRAIN_END]

    # Merge returns with calendar keys
    merged = pd.merge(returns_df, calendar_keys_df, on="date", how="inner")

    if merged.empty:
        return {"cdoy_grid": {}, "tdom_grid": {}, "baseline": 0.5}

    baseline = merged["up"].mean()

    # Compute CDOY stats (UP direction)
    cdoy_grid = {}
    cdoy_groups = merged.groupby(["month", "day"])
    for (month, day), group in cdoy_groups:
        n = len(group)
        if n >= 5:  # Minimum for display
            wr = group["up"].mean()
            cdoy_grid[(int(month), int(day))] = {
                "wr": wr,
                "n": n,
                "delta": wr - baseline,
            }

    # Compute TDOM stats (UP direction)
    tdom_grid = {}
    tdom_groups = merged.groupby("tdom")
    for tdom, group in tdom_groups:
        n = len(group)
        if n >= 5:  # Minimum for display
            wr = group["up"].mean()
            tdom_grid[int(tdom)] = {
                "wr": wr,
                "n": n,
                "delta": wr - baseline,
            }

    return {"cdoy_grid": cdoy_grid, "tdom_grid": tdom_grid, "baseline": baseline}


@st.cache_data(ttl=300)
def load_signal_tiers(run_id: str, symbol: str):
    """Load signal tier classifications for overlay markers.

    Returns dict with:
        - validated: set of (family, key_tuple) for q <= 0.10
        - promising: set of (family, key_tuple) for 0.10 < q <= 0.20
        - exploratory: set of (family, key_tuple) for p <= 0.05 (not in validated/promising)
    """
    signals_repo = SignalsRepo()
    train_stats = signals_repo.get_signal_stats(run_id, "train")

    if train_stats.empty:
        return {"validated": set(), "promising": set(), "exploratory": set()}

    # Filter to this symbol
    symbol_stats = train_stats[train_stats["symbol"] == symbol]

    validated = set()
    promising = set()
    exploratory = set()

    for _, row in symbol_stats.iterrows():
        if row["n"] < MIN_N:
            continue

        family = row["family"]
        key = json.loads(row["key_json"]) if isinstance(row["key_json"], str) else row["key_json"]

        # Create hashable key tuple
        if family == "CDOY":
            key_tuple = (family, key.get("month"), key.get("day"))
        else:  # TDOM
            key_tuple = (family, key.get("tdom"))

        q = row.get("fdr_q")
        p = row.get("p_value")

        if q is not None and q <= FDR_VALIDATED:
            validated.add(key_tuple)
        elif q is not None and q <= FDR_PROMISING:
            promising.add(key_tuple)
        elif p is not None and p <= P_EXPLORATORY:
            exploratory.add(key_tuple)

    return {"validated": validated, "promising": promising, "exploratory": exploratory}


@st.cache_data(ttl=300)
def load_future_calendar(days_ahead: int = 90):
    """Load future calendar events."""
    today = date.today()
    end_date = today + timedelta(days=days_ahead)
    # Cap at end of current year to avoid exchange_calendars bounds error
    max_date = date(today.year, 12, 31)
    if end_date > max_date:
        end_date = max_date
    try:
        return build_future_calendar_keys(today, end_date)
    except Exception:
        # Return empty DataFrame if calendar lookup fails
        return pd.DataFrame()


def compute_held_up_status(
    train_wr: float,
    train_baseline: float,
    test_wr: float | None,
    test_baseline: float | None,
    test_n: int | None,
) -> bool:
    """Determine if pattern held up in verification."""
    if test_wr is None or test_n is None or test_n == 0:
        return False

    baseline = test_baseline if test_baseline else train_baseline
    test_delta = test_wr - baseline

    # Held up if positive delta with decent sample
    return test_delta > 0 and test_n >= 20


def get_validated_patterns(train_stats, test_stats, baselines, include_not_held=False):
    """Get patterns with validation status computed."""
    if train_stats is None or train_stats.empty:
        return pd.DataFrame()

    patterns = []

    for _, row in train_stats.iterrows():
        if row["eligible"] != 1:
            continue

        signal_id = row["signal_id"]
        symbol = row["symbol"]
        family = row["family"]
        direction = row["direction"]
        key = json.loads(row["key_json"]) if isinstance(row["key_json"], str) else row["key_json"]

        train_baseline = baselines.get(symbol, {}).get("train", 0.5)
        test_baseline = baselines.get(symbol, {}).get("test", 0.5)

        # Get test stats
        test_row = test_stats[test_stats["signal_id"] == signal_id] if test_stats is not None else None
        if test_row is not None and not test_row.empty:
            test_data = test_row.iloc[0]
            test_wr = float(test_data["win_rate"])
            test_n = int(test_data["n"])
        else:
            test_wr = None
            test_n = None

        is_validated = compute_held_up_status(
            row["win_rate"], train_baseline, test_wr, test_baseline, test_n
        )

        if not include_not_held and not is_validated:
            continue

        # Get badge
        badge_text, badge_color, badge_hex = get_badge(
            is_validated=is_validated,
            avg_ret=row.get("avg_ret"),
            decade_consistency=row.get("decade_consistency"),
            win_rate=row["win_rate"],
            baseline=train_baseline,
        )

        patterns.append({
            "signal_id": signal_id,
            "symbol": symbol,
            "family": family,
            "direction": direction,
            "key": key,
            "pattern_name": format_pattern_name(family, key),
            "internal_code": get_internal_code(family, key),
            "train_wr": row["win_rate"],
            "train_n": row["n"],
            "train_baseline": train_baseline,
            "test_wr": test_wr,
            "test_n": test_n,
            "test_baseline": test_baseline,
            "avg_ret": row.get("avg_ret"),
            "decade_consistency": row.get("decade_consistency"),
            "score": row.get("score"),
            "is_validated": is_validated,
            "badge_text": badge_text,
            "badge_color": badge_color,
            "badge_hex": badge_hex,
        })

    return pd.DataFrame(patterns)


# =============================================================================
# Page: Overview
# =============================================================================
def render_overview():
    """Render the Overview page."""
    st.title("Calendar Edge Lab")

    run_id, train_stats, test_stats = load_run_data()
    if run_id is None:
        st.error("No scan data found. Run `python -m calendar_edge.cli run-scan` first.")
        return

    baselines = load_baseline_data()
    prices_repo = PricesRepo()
    symbols = prices_repo.get_symbols()

    # Index selector
    selected_symbol = st.selectbox(
        "Select Index",
        symbols,
        format_func=format_symbol,
        key="overview_symbol",
    )

    if not selected_symbol:
        return

    # Get validated patterns for this symbol
    all_patterns = get_validated_patterns(train_stats, test_stats, baselines, include_not_held=False)
    symbol_patterns = all_patterns[all_patterns["symbol"] == selected_symbol]

    # Also get count including not-held for context
    all_with_not_held = get_validated_patterns(train_stats, test_stats, baselines, include_not_held=True)
    symbol_all = all_with_not_held[all_with_not_held["symbol"] == selected_symbol]

    # Headline
    validated_count = len(symbol_patterns)
    total_count = len(symbol_all)

    headline_html = (
        f'<div style="background-color:{COLORS["card"]};border:1px solid {COLORS["border"]};border-radius:8px;padding:24px;margin:20px 0;text-align:center;">'
        f'<div style="font-size:3rem;font-weight:700;color:{COLORS["text"]};">{validated_count}</div>'
        f'<div style="font-size:1.2rem;color:{COLORS["secondary"]};">validated patterns detected</div>'
        f'<div style="font-size:0.9rem;color:{COLORS["secondary"]};margin-top:8px;">out of {total_count} discovered patterns for {format_symbol(selected_symbol)}</div>'
        f'</div>'
    )
    st.markdown(headline_html, unsafe_allow_html=True)

    if symbol_patterns.empty:
        st.info("No validated patterns for this index. Patterns that did not hold up in verification are hidden by default.")
        with st.expander("Show patterns that did not hold"):
            not_held = symbol_all[~symbol_all["is_validated"]]
            for _, p in not_held.iterrows():
                render_pattern_card(
                    pattern_name=p["pattern_name"],
                    direction=p["direction"],
                    badge_text=p["badge_text"],
                    badge_color=p["badge_hex"],
                    win_rate=p["train_wr"],
                    baseline=p["train_baseline"],
                    avg_ret=p["avg_ret"],
                    verification_wr=p["test_wr"],
                    verification_n=p["test_n"],
                    next_date=None,
                    discovery_wr=p["train_wr"],
                    on_click_key=f"btn_{p['signal_id']}",
                )
        return

    # Load future calendar for next dates
    future_cal = load_future_calendar(90)

    # Get next date for each pattern
    def get_next_date(row):
        if future_cal.empty:
            return None
        family = row["family"]
        key = row["key"]

        if family == "TDOM":
            matches = future_cal[future_cal["tdom"] == key.get("tdom")]
        else:  # CDOY
            matches = future_cal[
                (future_cal["month"] == key.get("month")) &
                (future_cal["day"] == key.get("day"))
            ]

        if matches.empty:
            return None
        return matches.iloc[0]["date"]

    symbol_patterns = symbol_patterns.copy()
    symbol_patterns["next_date"] = symbol_patterns.apply(get_next_date, axis=1)

    # Sort by score
    symbol_patterns = symbol_patterns.sort_values("score", ascending=False)

    # Render pattern cards in grid
    st.markdown("### Validated Patterns")

    cols = st.columns(2)
    for i, (_, p) in enumerate(symbol_patterns.iterrows()):
        with cols[i % 2]:
            render_pattern_card(
                pattern_name=p["pattern_name"],
                direction=p["direction"],
                badge_text=p["badge_text"],
                badge_color=p["badge_hex"],
                win_rate=p["train_wr"],
                baseline=p["train_baseline"],
                avg_ret=p["avg_ret"],
                verification_wr=p["test_wr"],
                verification_n=p["test_n"],
                next_date=p["next_date"],
                discovery_wr=p["train_wr"],
                on_click_key=f"btn_{p['signal_id']}",
            )


# =============================================================================
# Page: Upcoming
# =============================================================================
def render_upcoming():
    """Render the Upcoming page with forward calendar."""
    st.title("Upcoming Pattern Dates")

    run_id, train_stats, test_stats = load_run_data()
    if run_id is None:
        st.error("No scan data found.")
        return

    baselines = load_baseline_data()
    prices_repo = PricesRepo()
    symbols = prices_repo.get_symbols()

    # Filter chips for indices
    selected_symbols = st.multiselect(
        "Filter by Index",
        symbols,
        default=symbols,
        format_func=format_symbol,
    )

    if not selected_symbols:
        st.info("Select at least one index.")
        return

    # Get validated patterns
    all_patterns = get_validated_patterns(train_stats, test_stats, baselines, include_not_held=False)
    filtered_patterns = all_patterns[all_patterns["symbol"].isin(selected_symbols)]

    if filtered_patterns.empty:
        st.info("No validated patterns for selected indices.")
        return

    # Load future calendar
    future_cal = load_future_calendar(90)
    if future_cal.empty:
        st.warning("No trading days in the next 90 days.")
        return

    # Build upcoming events
    events = []
    for _, p in filtered_patterns.iterrows():
        family = p["family"]
        key = p["key"]

        if family == "TDOM":
            matches = future_cal[future_cal["tdom"] == key.get("tdom")]
        else:  # CDOY
            matches = future_cal[
                (future_cal["month"] == key.get("month")) &
                (future_cal["day"] == key.get("day"))
            ]

        for _, m in matches.iterrows():
            events.append({
                "date": m["date"],
                "month": m["month"],
                "day": m["day"],
                "pattern_name": p["pattern_name"],
                "direction": p["direction"],
                "symbol": p["symbol"],
                "badge_text": p["badge_text"],
                "badge_hex": p["badge_hex"],
                "train_wr": p["train_wr"],
                "train_baseline": p["train_baseline"],
                "test_wr": p["test_wr"],
                "signal_id": p["signal_id"],
            })

    if not events:
        st.info("No upcoming events for validated patterns in the next 90 days.")
        return

    events_df = pd.DataFrame(events)
    events_df["date_parsed"] = pd.to_datetime(events_df["date"])
    events_df = events_df.sort_values("date_parsed")
    events_df["year_month"] = events_df["date_parsed"].dt.strftime("%Y-%m")

    # Group by month
    for ym in events_df["year_month"].unique():
        month_events = events_df[events_df["year_month"] == ym]
        month_date = datetime.strptime(ym, "%Y-%m")
        month_name = f"{MONTH_FULL_NAMES[month_date.month]} {month_date.year}"

        st.markdown(f"### {month_name}")

        for _, e in month_events.iterrows():
            dt = e["date_parsed"]
            edge = e["train_wr"] - e["train_baseline"]
            verified_text = f" · Verified: {e['test_wr']:.0%}" if e["test_wr"] else ""

            # Event row
            col1, col2, col3 = st.columns([1, 4, 1])

            with col1:
                date_html = (
                    f'<div style="text-align:center;">'
                    f'<div style="font-size:2rem;font-weight:700;color:{COLORS["text"]};">{dt.day}</div>'
                    f'<div style="font-size:0.8rem;color:{COLORS["secondary"]};">{MONTH_NAMES[dt.month]}</div>'
                    f'</div>'
                )
                st.markdown(date_html, unsafe_allow_html=True)

            with col2:
                badge_html = render_badge(e["badge_text"], e["badge_hex"])
                card_html = (
                    f'<div style="background-color:{COLORS["card"]};border:1px solid {COLORS["border"]};border-radius:8px;padding:12px 16px;">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                    f'<div><span style="font-weight:600;color:{COLORS["text"]};">{e["pattern_name"]}</span>'
                    f'<span style="color:{COLORS["secondary"]};margin-left:8px;">{get_direction_label(e["direction"])}</span></div>'
                    f'{badge_html}</div>'
                    f'<div style="margin-top:8px;font-size:0.85rem;color:{COLORS["secondary"]};">'
                    f'Win Rate: {e["train_wr"]:.0%} ({edge:+.1%} vs baseline){verified_text}</div>'
                    f'</div>'
                )
                st.markdown(card_html, unsafe_allow_html=True)

            with col3:
                tag_color = get_symbol_tag_color(e["symbol"])
                tag_html = f'<div style="background-color:{tag_color};color:white;padding:4px 12px;border-radius:16px;font-size:0.8rem;font-weight:500;text-align:center;margin-top:20px;">{format_symbol(e["symbol"])}</div>'
                st.markdown(tag_html, unsafe_allow_html=True)

        st.markdown("---")


# =============================================================================
# Page: Heatmap
# =============================================================================
def render_heatmap():
    """Render the Heatmap page with calendar grid showing ALL data with tier overlays."""
    st.title("Calendar Heatmap")

    run_id, train_stats, test_stats = load_run_data()
    prices_repo = PricesRepo()
    symbols = prices_repo.get_symbols()

    # Controls row
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

    with col1:
        selected_symbol = st.selectbox(
            "Select Index",
            symbols,
            format_func=format_symbol,
            key="heatmap_symbol",
        )

    with col2:
        show_promising = st.checkbox("Show Promising", value=True, key="show_promising")

    with col3:
        show_exploratory = st.checkbox("Show Exploratory", value=False, key="show_exploratory")

    with col4:
        direction = st.selectbox("Direction", ["UP", "DOWN"], key="heatmap_direction")

    if not selected_symbol:
        return

    # Load raw heatmap data (computed from returns + calendar_keys)
    heatmap_data = load_raw_heatmap_data(selected_symbol)
    cdoy_grid = heatmap_data["cdoy_grid"]
    tdom_grid = heatmap_data["tdom_grid"]
    baseline = heatmap_data["baseline"]

    # Load signal tiers for overlays
    tiers = {"validated": set(), "promising": set(), "exploratory": set()}
    if run_id:
        tiers = load_signal_tiers(run_id, selected_symbol)

    # Legend
    st.markdown("#### Legend")
    legend_cols = st.columns(6)
    with legend_cols[0]:
        st.markdown('<span style="background:#166534;color:white;padding:2px 8px;border-radius:4px;">>+10%</span>', unsafe_allow_html=True)
    with legend_cols[1]:
        st.markdown('<span style="background:#22c55e;color:white;padding:2px 8px;border-radius:4px;">+5-10%</span>', unsafe_allow_html=True)
    with legend_cols[2]:
        st.markdown('<span style="background:#374151;color:white;padding:2px 8px;border-radius:4px;">±5%</span>', unsafe_allow_html=True)
    with legend_cols[3]:
        st.markdown('<span style="background:#dc2626;color:white;padding:2px 8px;border-radius:4px;">-5-10%</span>', unsafe_allow_html=True)
    with legend_cols[4]:
        st.markdown('<span style="background:#991b1b;color:white;padding:2px 8px;border-radius:4px;"><-10%</span>', unsafe_allow_html=True)

    st.markdown("**Overlays:** ✓ = Validated (q≤0.10) · ⚠ = Promising (q≤0.20) · • = Exploratory (p≤0.05)")

    # Count tiers
    validated_count = len(tiers["validated"])
    promising_count = len(tiers["promising"])
    exploratory_count = len(tiers["exploratory"])
    st.caption(f"Signals: {validated_count} Validated, {promising_count} Promising, {exploratory_count} Exploratory")

    # CDOY Section
    st.markdown("### Calendar Day of Year (CDOY)")
    st.caption(f"Win rate for {direction} direction vs {baseline:.1%} baseline. {len(cdoy_grid)} cells with data.")

    # Build DataFrame for heatmap with overlay markers
    def get_cdoy_display(month, day):
        cell = cdoy_grid.get((month, day))
        if cell is None:
            return ""
        wr = cell["wr"]
        # For DOWN direction, invert the win rate
        if direction == "DOWN":
            wr = 1 - wr
        # Check for overlay markers
        key_val = ("CDOY", month, day)
        marker = ""
        if key_val in tiers["validated"]:
            marker = "✓"
        elif show_promising and key_val in tiers["promising"]:
            marker = "⚠"
        elif show_exploratory and key_val in tiers["exploratory"]:
            marker = "•"
        return f"{wr:.0%}{marker}"

    def get_cdoy_value(month, day):
        cell = cdoy_grid.get((month, day))
        if cell is None:
            return None
        wr = cell["wr"]
        if direction == "DOWN":
            wr = 1 - wr
        return wr

    grid_data = {}
    for m in range(1, 13):
        col_data = [get_cdoy_display(m, day) for day in range(1, 32)]
        grid_data[MONTH_NAMES[m]] = col_data

    grid_df = pd.DataFrame(grid_data, index=range(1, 32))
    grid_df.index.name = "Day"

    # Color function for styling
    def color_cell(val):
        if val is None or (isinstance(val, float) and pd.isna(val)) or val == "":
            return "background-color: #1a1d24; color: #6b7280;"
        # Extract numeric value if string with marker
        if isinstance(val, str):
            try:
                num_str = val.rstrip("✓⚠•").rstrip("%")
                val = float(num_str) / 100
            except (ValueError, AttributeError):
                return "background-color: #1a1d24; color: #6b7280;"
        delta = val - baseline
        if direction == "DOWN":
            delta = (1 - val) - baseline  # Recalculate for DOWN
        if delta > 0.10:
            return "background-color: #166534; color: white;"
        elif delta > 0.05:
            return "background-color: #22c55e; color: white;"
        elif delta > -0.05:
            return "background-color: #374151; color: white;"
        elif delta > -0.10:
            return "background-color: #dc2626; color: white;"
        else:
            return "background-color: #991b1b; color: white;"

    styled_df = grid_df.style.map(color_cell)
    st.dataframe(styled_df, use_container_width=True, height=400)

    # TDOM Section
    st.markdown("### Trading Day of Month (TDOM)")
    st.caption(f"Win rate for {direction} direction. {len(tdom_grid)} cells with data.")

    def get_tdom_display(tdom):
        cell = tdom_grid.get(tdom)
        if cell is None:
            return ""
        wr = cell["wr"]
        if direction == "DOWN":
            wr = 1 - wr
        key_val = ("TDOM", tdom)
        marker = ""
        if key_val in tiers["validated"]:
            marker = "✓"
        elif show_promising and key_val in tiers["promising"]:
            marker = "⚠"
        elif show_exploratory and key_val in tiers["exploratory"]:
            marker = "•"
        return f"{wr:.0%}{marker}"

    tdom_data = {f"T{tdom}": get_tdom_display(tdom) for tdom in range(1, 24)}
    tdom_df = pd.DataFrame([tdom_data])
    tdom_styled = tdom_df.style.map(color_cell)
    st.dataframe(tdom_styled, use_container_width=True, hide_index=True)


# =============================================================================
# Page: Details
# =============================================================================
def render_details():
    """Render the Details page for a specific pattern."""
    st.title("Pattern Details")

    run_id, train_stats, test_stats = load_run_data()
    if run_id is None:
        st.error("No scan data found.")
        return

    baselines = load_baseline_data()
    signals_repo = SignalsRepo()
    prices_repo = PricesRepo()
    symbols = prices_repo.get_symbols()

    # Check if coming from card click
    preselected_signal = st.session_state.get("selected_signal_id")

    # Index selector
    selected_symbol = st.selectbox(
        "Select Index",
        symbols,
        format_func=format_symbol,
        key="details_symbol",
    )

    if not selected_symbol:
        return

    # Get all patterns for this symbol
    all_patterns = get_validated_patterns(train_stats, test_stats, baselines, include_not_held=True)
    symbol_patterns = all_patterns[all_patterns["symbol"] == selected_symbol]

    if symbol_patterns.empty:
        st.info("No patterns found for this index.")
        return

    # Build pattern options
    pattern_options = []
    for _, p in symbol_patterns.iterrows():
        label = f"{p['pattern_name']} - {get_direction_label(p['direction'])}"
        if p["is_validated"]:
            label += " ✓"
        pattern_options.append((p["signal_id"], label))

    # Sort validated first
    pattern_options.sort(key=lambda x: (0 if "✓" in x[1] else 1, x[1]))

    # Find preselected index
    default_idx = 0
    if preselected_signal:
        for i, (sid, _) in enumerate(pattern_options):
            if sid == preselected_signal:
                default_idx = i
                break

    selected_signal = st.selectbox(
        "Select Pattern",
        options=[p[0] for p in pattern_options],
        format_func=lambda x: next(p[1] for p in pattern_options if p[0] == x),
        index=default_idx,
        key="details_pattern",
    )

    if not selected_signal:
        return

    # Get pattern data
    pattern = symbol_patterns[symbol_patterns["signal_id"] == selected_signal].iloc[0]

    # Breadcrumb
    st.markdown(
        f'<div style="color:{COLORS["secondary"]};font-size:0.9rem;margin-bottom:16px;">{format_symbol(selected_symbol)} / {pattern["pattern_name"]}</div>',
        unsafe_allow_html=True,
    )

    # Header with badge
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"## {pattern['pattern_name']}")
        st.markdown(f"*{get_direction_label(pattern['direction'])}*")
    with col2:
        badge_html = f'<div style="text-align:right;padding-top:16px;">{render_badge(pattern["badge_text"], pattern["badge_hex"])}</div>'
        st.markdown(badge_html, unsafe_allow_html=True)

    st.markdown("---")

    # Two prominent cards side-by-side
    col1, col2 = st.columns(2)

    train_baseline = pattern["train_baseline"]
    test_baseline = pattern["test_baseline"] if pattern["test_baseline"] else train_baseline
    train_delta = pattern["train_wr"] - train_baseline
    test_delta = (pattern["test_wr"] - test_baseline) if pattern["test_wr"] else None

    # Check for alpha decay
    alpha_decay = False
    if pattern["test_wr"] is not None:
        decay = pattern["train_wr"] - pattern["test_wr"]
        alpha_decay = decay >= 0.10

    with col1:
        train_edge_bg = "rgba(74,222,128,0.2)" if train_delta >= 0 else "rgba(248,113,113,0.2)"
        train_edge_color = COLORS["green"] if train_delta >= 0 else COLORS["red"]
        discovery_html = (
            f'<div style="background-color:{COLORS["card"]};border:1px solid {COLORS["border"]};border-radius:8px;padding:24px;min-height:200px;">'
            f'<div style="font-size:0.9rem;color:{COLORS["secondary"]};margin-bottom:8px;">Discovery Period</div>'
            f'<div style="font-size:0.8rem;color:{COLORS["secondary"]};margin-bottom:16px;">1950 - 2009</div>'
            f'<div style="font-size:3rem;font-weight:700;color:{COLORS["text"]};">{pattern["train_wr"]:.0%}</div>'
            f'<div style="display:inline-block;background-color:{train_edge_bg};color:{train_edge_color};padding:4px 12px;border-radius:16px;font-size:0.9rem;margin-top:8px;">{train_delta:+.1%} vs baseline</div>'
            f'<div style="font-size:0.85rem;color:{COLORS["secondary"]};margin-top:12px;">n = {pattern["train_n"]} observations</div>'
            f'</div>'
        )
        st.markdown(discovery_html, unsafe_allow_html=True)

    with col2:
        ver_color = "#f59e0b" if alpha_decay else COLORS["text"]
        if pattern["test_wr"] is not None:
            test_edge_bg = "rgba(74,222,128,0.2)" if test_delta >= 0 else "rgba(248,113,113,0.2)"
            test_edge_color = COLORS["green"] if test_delta >= 0 else COLORS["red"]
            border_color = "#f59e0b" if alpha_decay else COLORS["border"]
            verification_html = (
                f'<div style="background-color:{COLORS["card"]};border:1px solid {border_color};border-radius:8px;padding:24px;min-height:200px;">'
                f'<div style="font-size:0.9rem;color:{COLORS["secondary"]};margin-bottom:8px;">Verification Period</div>'
                f'<div style="font-size:0.8rem;color:{COLORS["secondary"]};margin-bottom:16px;">2010 - Present</div>'
                f'<div style="font-size:3rem;font-weight:700;color:{ver_color};">{pattern["test_wr"]:.0%}</div>'
                f'<div style="display:inline-block;background-color:{test_edge_bg};color:{test_edge_color};padding:4px 12px;border-radius:16px;font-size:0.9rem;margin-top:8px;">{test_delta:+.1%} vs baseline</div>'
                f'<div style="font-size:0.85rem;color:{COLORS["secondary"]};margin-top:12px;">n = {pattern["test_n"]} observations</div>'
                f'</div>'
            )
            st.markdown(verification_html, unsafe_allow_html=True)
            if alpha_decay:
                st.warning("Pattern shows weakening in recent data")
        else:
            no_data_html = (
                f'<div style="background-color:{COLORS["card"]};border:1px solid {COLORS["border"]};border-radius:8px;padding:24px;min-height:200px;display:flex;align-items:center;justify-content:center;">'
                f'<div style="text-align:center;color:{COLORS["secondary"]};">'
                f'<div style="font-size:1.2rem;margin-bottom:8px;">No Verification Data</div>'
                f'<div style="font-size:0.9rem;">Insufficient observations since 2010</div>'
                f'</div></div>'
            )
            st.markdown(no_data_html, unsafe_allow_html=True)

    # Decade breakdown
    st.markdown("### Decade Breakdown")
    st.caption("Performance by decade during discovery period. Checkmark indicates win rate exceeded baseline.")

    # Get decade stats
    all_stats = signals_repo.get_signal_stats(run_id)
    signal_stats = all_stats[all_stats["signal_id"] == selected_signal]
    decade_stats = signal_stats[signal_stats["window"].str.startswith("decade_")]

    if decade_stats.empty:
        st.info("No decade breakdown available.")
    else:
        decades = []
        for _, row in decade_stats.iterrows():
            decade_label = row["window"].replace("decade_", "")
            delta = row["win_rate"] - train_baseline
            decades.append({
                "decade": decade_label,
                "wr": row["win_rate"],
                "n": row["n"],
                "delta": delta,
                "passed": delta > 0,
            })

        decades.sort(key=lambda x: x["decade"])

        # Count passing decades
        passing = sum(1 for d in decades if d["passed"])
        total = len(decades)

        st.markdown(f"**{passing} of {total}** decades showed positive edge")

        # Render decade grid
        cols = st.columns(len(decades))
        for i, d in enumerate(decades):
            with cols[i]:
                check = "✓" if d["passed"] else ""
                bg_color = COLORS["badge_green"] if d["passed"] else COLORS["card"]
                decade_html = (
                    f'<div style="background-color:{bg_color};border:1px solid {COLORS["border"]};border-radius:8px;padding:12px;text-align:center;">'
                    f'<div style="font-size:1.2rem;font-weight:600;color:{COLORS["text"]};">{d["decade"]}</div>'
                    f'<div style="font-size:1.5rem;font-weight:700;color:{COLORS["text"]};margin:8px 0;">{d["wr"]:.0%} {check}</div>'
                    f'<div style="font-size:0.8rem;color:{COLORS["secondary"]};">n={d["n"]}</div>'
                    f'</div>'
                )
                st.markdown(decade_html, unsafe_allow_html=True)

    # Next occurrence
    st.markdown("### Next Occurrence")

    future_cal = load_future_calendar(365)
    if not future_cal.empty:
        family = pattern["family"]
        key = pattern["key"]

        if family == "TDOM":
            matches = future_cal[future_cal["tdom"] == key.get("tdom")]
        else:
            matches = future_cal[
                (future_cal["month"] == key.get("month")) &
                (future_cal["day"] == key.get("day"))
            ]

        if not matches.empty:
            next_date = matches.iloc[0]["date"]
            next_html = f'<div style="background-color:{COLORS["card"]};border:1px solid {COLORS["border"]};border-radius:8px;padding:16px;display:inline-block;"><span style="font-size:1.1rem;color:{COLORS["text"]};">{format_date_human(next_date)}</span></div>'
            st.markdown(next_html, unsafe_allow_html=True)
        else:
            st.info("No upcoming occurrence in the next year.")
    else:
        st.info("Could not load future calendar.")


# =============================================================================
# Main App
# =============================================================================
def main():
    """Main application entry point."""
    # Initialize session state
    if "page" not in st.session_state:
        st.session_state["page"] = "Overview"
    if "selected_signal_id" not in st.session_state:
        st.session_state["selected_signal_id"] = None

    # Navigation tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Upcoming", "Heatmap", "Details"])

    with tab1:
        render_overview()

    with tab2:
        render_upcoming()

    with tab3:
        render_heatmap()

    with tab4:
        render_details()


if __name__ == "__main__":
    main()
