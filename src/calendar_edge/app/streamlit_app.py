"""Streamlit application for Calendar Edge Lab."""

import json

# Add src to path for imports
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from calendar_edge.config import DB_PATH, FDR_THRESHOLD, MIN_N, TEST_START, TRAIN_END
from calendar_edge.db import CalendarKeysRepo, PricesRepo, ReturnsRepo, RunsRepo, SignalsRepo
from calendar_edge.features import build_future_calendar_keys

# Friendly symbol names
SYMBOL_NAMES = {
    "^GSPC": "S&P 500 (^GSPC)",
    "^DJI": "Dow Jones Industrial Average (^DJI)",
    "^IXIC": "Nasdaq Composite (^IXIC)",
}


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
        ["Overview", "Calendar Heatmap", "Signal Detail", "Forward Calendar", "Methodology"],
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

    # Add glossary to all pages except Methodology
    if page != "Methodology":
        render_glossary()

    if page == "Overview":
        render_overview(run_id)
    elif page == "Calendar Heatmap":
        render_heatmap(run_id)
    elif page == "Signal Detail":
        render_signal_detail(run_id)
    elif page == "Forward Calendar":
        render_forward_calendar(run_id)
    elif page == "Methodology":
        render_methodology()


def render_overview(run_id: str):
    """Render overview page."""
    st.header("Overview")

    prices_repo = PricesRepo()
    returns_repo = ReturnsRepo()
    signals_repo = SignalsRepo()

    symbols = prices_repo.get_symbols()

    # Symbol selector with friendly names
    selected_symbol = st.selectbox(
        "Select Symbol",
        symbols,
        format_func=format_symbol,
    )

    if not selected_symbol:
        st.warning("No symbols found in database.")
        return

    # Baseline stats
    st.subheader("Baseline Statistics")

    returns_df = returns_repo.get_returns(selected_symbol)
    train_returns = returns_df[returns_df["date"] <= TRAIN_END]
    test_returns = returns_df[returns_df["date"] >= TEST_START]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Train Win Rate",
            f"{train_returns['up'].mean():.2%}",
            f"n={len(train_returns)}",
        )
    with col2:
        st.metric(
            "Test Win Rate",
            f"{test_returns['up'].mean():.2%}",
            f"n={len(test_returns)}",
        )
    with col3:
        st.metric(
            "Full Win Rate",
            f"{returns_df['up'].mean():.2%}",
            f"n={len(returns_df)}",
        )

    # Top signals by family
    st.subheader("Top 10 Signals by Family")
    st.caption("Ranked by Train score; evaluated out-of-sample on Test.")

    for family in ["CDOY", "TDOM"]:
        st.markdown(f"**{family}**")

        top_df = signals_repo.get_top_signals(run_id, "train", 10, selected_symbol, family)

        if top_df.empty:
            st.info(f"No eligible {family} signals found.")
            continue

        # Format for display
        display_df = []
        for _, row in top_df.iterrows():
            key = json.loads(row["key_json"]) if isinstance(row["key_json"], str) else row["key_json"]
            if family == "CDOY":
                key_str = f"M{key.get('month'):02d}D{key.get('day'):02d}"
            else:
                key_str = f"TDOM{key.get('tdom')}"

            display_df.append({
                "Key": key_str,
                "Direction": row["direction"],
                "N": row["n"],
                "Win Rate": f"{row['win_rate']:.2%}",
                "Z-Score": f"{row['z_score']:.2f}" if row["z_score"] else "N/A",
                "DCF": f"{row['decade_consistency']:.2f}" if row["decade_consistency"] else "N/A",
                "FDR-Q": f"{row['fdr_q']:.3f}" if row["fdr_q"] else "N/A",
                "Score": f"{row['score']:.2f}" if row["score"] else "N/A",
            })

        st.dataframe(pd.DataFrame(display_df), use_container_width=True)


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


def render_signal_detail(run_id: str):
    """Render signal detail page."""
    st.header("Signal Detail")

    signals_repo = SignalsRepo()
    prices_repo = PricesRepo()

    symbols = prices_repo.get_symbols()
    selected_symbol = st.selectbox(
        "Select Symbol",
        symbols,
        format_func=format_symbol,
    )

    if not selected_symbol:
        return

    # Get all signals for symbol
    all_signals = signals_repo.get_signal_stats(run_id, "train")
    symbol_signals = all_signals[all_signals["symbol"] == selected_symbol]

    if symbol_signals.empty:
        st.warning("No signals found for this symbol.")
        return

    # Build signal options
    signal_options = []
    for _, row in symbol_signals.iterrows():
        key = json.loads(row["key_json"]) if isinstance(row["key_json"], str) else row["key_json"]
        family = row["family"]
        if family == "CDOY":
            key_str = f"M{key.get('month'):02d}D{key.get('day'):02d}"
        else:
            key_str = f"TDOM{key.get('tdom')}"
        label = f"{family} {key_str} ({row['direction']}) - Score: {row['score']:.2f}" if row['score'] else f"{family} {key_str} ({row['direction']})"
        signal_options.append((row["signal_id"], label))

    # Sort by score
    signal_options.sort(key=lambda x: float(x[1].split("Score: ")[1]) if "Score:" in x[1] else 0, reverse=True)

    selected_signal = st.selectbox(
        "Select Signal",
        options=[s[0] for s in signal_options],
        format_func=lambda x: next(s[1] for s in signal_options if s[0] == x),
    )

    if not selected_signal:
        return

    # Get all window stats for this signal
    signal_data = all_signals[all_signals["signal_id"] == selected_signal].iloc[0]

    st.subheader("Signal Information")

    key = json.loads(signal_data["key_json"]) if isinstance(signal_data["key_json"], str) else signal_data["key_json"]
    family = signal_data["family"]
    if family == "CDOY":
        key_str = f"Month {key.get('month')}, Day {key.get('day')}"
    else:
        key_str = f"Trading Day {key.get('tdom')} of Month"

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Family", family)
    with col2:
        st.metric("Key", key_str)
    with col3:
        st.metric("Direction", signal_data["direction"])

    # Train vs Test comparison with clear labels
    st.subheader("Performance by Window")

    # Get test stats
    test_stats = signals_repo.get_signal_stats(run_id, "test")
    test_row = test_stats[test_stats["signal_id"] == selected_signal]

    full_stats = signals_repo.get_signal_stats(run_id, "full")
    full_row = full_stats[full_stats["signal_id"] == selected_signal]

    comparison_data = []

    # Train
    comparison_data.append({
        "Window": "Train (discovery)",
        "N": signal_data["n"],
        "Wins": signal_data["wins"],
        "Win Rate": f"{signal_data['win_rate']:.2%}",
        "Avg Return": f"{signal_data['avg_ret']:.4%}" if signal_data['avg_ret'] else "N/A",
        "CI": f"[{signal_data['ci_low']:.2f}-{signal_data['ci_high']:.2f}]" if signal_data['ci_low'] else "N/A",
        "Z-Score": f"{signal_data['z_score']:.2f}" if signal_data['z_score'] else "N/A",
        "DCF": f"{signal_data['decade_consistency']:.2f}" if signal_data['decade_consistency'] else "N/A",
        "FDR-Q": f"{signal_data['fdr_q']:.3f}" if signal_data['fdr_q'] else "N/A",
        "Score": f"{signal_data['score']:.2f}" if signal_data['score'] else "N/A",
    })

    # Test
    if not test_row.empty:
        test_data = test_row.iloc[0]
        comparison_data.append({
            "Window": "Test (holdout)",
            "N": test_data["n"],
            "Wins": test_data["wins"],
            "Win Rate": f"{test_data['win_rate']:.2%}",
            "Avg Return": f"{test_data['avg_ret']:.4%}" if test_data['avg_ret'] else "N/A",
            "CI": f"[{test_data['ci_low']:.2f}-{test_data['ci_high']:.2f}]" if test_data['ci_low'] else "N/A",
            "Z-Score": "-",
            "DCF": "-",
            "FDR-Q": "-",
            "Score": "-",
        })

    # Full
    if not full_row.empty:
        full_data = full_row.iloc[0]
        comparison_data.append({
            "Window": "Full (context)",
            "N": full_data["n"],
            "Wins": full_data["wins"],
            "Win Rate": f"{full_data['win_rate']:.2%}",
            "Avg Return": f"{full_data['avg_ret']:.4%}" if full_data['avg_ret'] else "N/A",
            "CI": f"[{full_data['ci_low']:.2f}-{full_data['ci_high']:.2f}]" if full_data['ci_low'] else "N/A",
            "Z-Score": "-",
            "DCF": "-",
            "FDR-Q": "-",
            "Score": "-",
        })

    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)

    # Eligibility status
    if signal_data["eligible"]:
        st.success(f"This signal passes eligibility criteria (n >= {MIN_N}, FDR-Q <= {FDR_THRESHOLD:.0%})")
    else:
        st.warning(f"This signal does not pass eligibility criteria (requires n >= {MIN_N} and FDR-Q <= {FDR_THRESHOLD:.0%})")


def render_forward_calendar(run_id: str):
    """Render forward calendar page.

    Shows upcoming dates where eligible signals fire.
    Includes BOTH CDOY and TDOM signals.
    Uses NYSE trading calendar (exchange-calendars) for accurate session dates.
    """
    st.header("Forward Calendar")
    st.caption("Upcoming trading sessions where eligible Train signals fire. Uses NYSE calendar (holidays excluded).")

    signals_repo = SignalsRepo()
    prices_repo = PricesRepo()

    symbols = prices_repo.get_symbols()

    # Controls
    col1, col2 = st.columns(2)
    with col1:
        selected_symbols = st.multiselect(
            "Symbols",
            symbols,
            default=symbols,
            format_func=format_symbol,
        )
    with col2:
        num_days = st.slider("Calendar Days Ahead", 30, 180, 60)

    if not selected_symbols:
        st.warning("Please select at least one symbol.")
        return

    # Get eligible signals from Train window
    all_signals = signals_repo.get_signal_stats(run_id, "train", eligible_only=True)

    if all_signals.empty:
        st.warning("No eligible signals found in the Train window.")
        return

    # Filter by selected symbols
    filtered_signals = all_signals[all_signals["symbol"].isin(selected_symbols)]

    if filtered_signals.empty:
        st.warning("No eligible signals for selected symbols.")
        return

    # Get test stats for comparison
    test_stats = signals_repo.get_signal_stats(run_id, "test")

    # Build a map of signal_id -> test win_rate
    test_wr_map = {}
    for _, row in test_stats.iterrows():
        test_wr_map[row["signal_id"]] = row["win_rate"]

    # Build future calendar keys using NYSE trading calendar
    today = datetime.now().date()
    end_date = today + timedelta(days=num_days)

    future_keys = build_future_calendar_keys(today, end_date)

    if future_keys.empty:
        st.warning("No trading sessions found in the selected date range.")
        return

    # Build forward calendar data
    calendar_data = []
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    for _, key_row in future_keys.iterrows():
        date_str = key_row["date"]
        month = int(key_row["month"])
        day = int(key_row["day"])
        tdom = int(key_row["tdom"])
        dow = dow_names[int(key_row["dow"])]

        # Check each symbol's signals
        for symbol in selected_symbols:
            symbol_signals = filtered_signals[filtered_signals["symbol"] == symbol]

            for _, sig_row in symbol_signals.iterrows():
                key = json.loads(sig_row["key_json"]) if isinstance(sig_row["key_json"], str) else sig_row["key_json"]
                family = sig_row["family"]
                matched = False

                if family == "CDOY":
                    if key.get("month") == month and key.get("day") == day:
                        matched = True
                        signal_name = f"M{month:02d}D{day:02d}"
                elif family == "TDOM":
                    if key.get("tdom") == tdom:
                        matched = True
                        signal_name = f"TDOM{tdom}"

                if matched:
                    # Get test win rate if available
                    test_wr = test_wr_map.get(sig_row["signal_id"])
                    test_wr_str = f"{test_wr:.2%}" if test_wr is not None else "-"

                    calendar_data.append({
                        "Date": date_str,
                        "DOW": dow,
                        "Symbol": format_symbol(symbol),
                        "Family": family,
                        "Signal": signal_name,
                        "Direction": sig_row["direction"],
                        "Train WR": f"{sig_row['win_rate']:.2%}",
                        "Test WR": test_wr_str,
                        "FDR-Q": f"{sig_row['fdr_q']:.3f}" if sig_row['fdr_q'] else "-",
                        "Score": f"{sig_row['score']:.2f}" if sig_row['score'] else "-",
                    })

    if calendar_data:
        # Sort by date
        calendar_df = pd.DataFrame(calendar_data)
        calendar_df = calendar_df.sort_values("Date")
        st.dataframe(calendar_df, use_container_width=True, height=600)

        # Summary stats
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Trading Sessions", len(future_keys))
        with col2:
            st.metric("Total Signal Events", len(calendar_data))
        with col3:
            cdoy_count = len([c for c in calendar_data if c["Family"] == "CDOY"])
            st.metric("CDOY Events", cdoy_count)
        with col4:
            tdom_count = len([c for c in calendar_data if c["Family"] == "TDOM"])
            st.metric("TDOM Events", tdom_count)
    else:
        st.info("No eligible signals fire in the selected date range.")


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
