"""Validation and sanity checks for Calendar Edge Lab."""

import json
from dataclasses import dataclass
from datetime import date, timedelta

from calendar_edge.config import (
    DEFAULT_SYMBOLS,
    MIN_N,
    TRAIN_END,
)
from calendar_edge.db import (
    CalendarKeysRepo,
    PricesRepo,
    ReturnsRepo,
    RunsRepo,
    SignalsRepo,
)
from calendar_edge.features import build_future_calendar_keys, get_trading_sessions


@dataclass
class CheckResult:
    """Result of a validation check.

    Attributes:
        name: Unique identifier for the check.
        status: One of "pass", "fail", "warn", "skip".
        message: Human-readable result message.
        details: Optional dict with additional data.
    """

    name: str
    status: str  # "pass", "fail", "warn", "skip"
    message: str
    details: dict | None = None

    @property
    def passed(self) -> bool:
        """True if status is pass or warn (not a failure)."""
        return self.status in ("pass", "warn", "skip")

    @property
    def is_warning(self) -> bool:
        """True if status is warn or skip."""
        return self.status in ("warn", "skip")


def check_data_coverage() -> list[CheckResult]:
    """Check data coverage and date ranges for all symbols."""
    results = []
    prices_repo = PricesRepo()
    returns_repo = ReturnsRepo()

    for symbol in DEFAULT_SYMBOLS:
        # Get counts and ranges
        prices_count = prices_repo.get_row_count(symbol)
        returns_count = returns_repo.get_row_count(symbol)
        prices_range = prices_repo.get_date_range(symbol)

        # Get returns date range from DataFrame
        returns_df = returns_repo.get_returns(symbol)
        if returns_df.empty:
            returns_range = (None, None)
        else:
            returns_range = (returns_df["date"].min(), returns_df["date"].max())

        # Check returns count ~ prices count - 1
        expected_returns = prices_count - 1
        diff = abs(returns_count - expected_returns)
        count_ok = diff <= 5  # Allow small variance

        details = {
            "prices_count": prices_count,
            "returns_count": returns_count,
            "prices_min": prices_range[0],
            "prices_max": prices_range[1],
            "returns_min": returns_range[0],
            "returns_max": returns_range[1],
        }

        if prices_count == 0:
            results.append(CheckResult(
                name=f"data_coverage_{symbol}",
                status="fail",
                message=f"{symbol}: No price data found",
                details=details,
            ))
        elif not count_ok:
            results.append(CheckResult(
                name=f"data_coverage_{symbol}",
                status="fail",
                message=f"{symbol}: Returns count ({returns_count}) != prices-1 ({expected_returns})",
                details=details,
            ))
        else:
            results.append(CheckResult(
                name=f"data_coverage_{symbol}",
                status="pass",
                message=f"{symbol}: {prices_count} prices, {returns_count} returns, {prices_range[0]} to {prices_range[1]}",
                details=details,
            ))

    return results


def check_forward_calendar_correctness() -> list[CheckResult]:
    """Check forward calendar uses correct NYSE trading calendar."""
    results = []

    # Fixed test window
    start = date(2025, 12, 20)
    end = date(2026, 2, 15)

    sessions = get_trading_sessions(start, end)
    future_keys = build_future_calendar_keys(start, end)

    # Check no weekends
    weekend_sessions = [s for s in sessions if s.weekday() >= 5]
    results.append(CheckResult(
        name="no_weekends_in_sessions",
        status="pass" if len(weekend_sessions) == 0 else "fail",
        message=f"No weekends in sessions: {len(weekend_sessions)} found",
        details={"weekend_dates": [str(s) for s in weekend_sessions]},
    ))

    # Check Jan 1, 2026 excluded
    jan1_excluded = date(2026, 1, 1) not in sessions
    results.append(CheckResult(
        name="jan1_2026_excluded",
        status="pass" if jan1_excluded else "fail",
        message=f"Jan 1, 2026 excluded: {jan1_excluded}",
    ))

    # Check TDOM1 for January 2026
    jan_keys = future_keys[future_keys["month"] == 1]
    if not jan_keys.empty:
        tdom1_row = jan_keys[jan_keys["tdom"] == 1]
        tdom1_date = tdom1_row.iloc[0]["date"] if not tdom1_row.empty else None
        tdom1_correct = tdom1_date == "2026-01-02"

        results.append(CheckResult(
            name="tdom1_jan2026",
            status="pass" if tdom1_correct else "fail",
            message=f"TDOM1 Jan 2026 = {tdom1_date} (expected 2026-01-02)",
            details={"tdom1_date": tdom1_date},
        ))

        # Check TDOM2 for January 2026
        tdom2_row = jan_keys[jan_keys["tdom"] == 2]
        tdom2_date = tdom2_row.iloc[0]["date"] if not tdom2_row.empty else None
        tdom2_correct = tdom2_date == "2026-01-05"

        results.append(CheckResult(
            name="tdom2_jan2026",
            status="pass" if tdom2_correct else "fail",
            message=f"TDOM2 Jan 2026 = {tdom2_date} (expected 2026-01-05)",
            details={"tdom2_date": tdom2_date},
        ))
    else:
        results.append(CheckResult(
            name="tdom1_jan2026",
            status="fail",
            message="No January 2026 sessions found",
        ))

    return results


def check_heatmap_sanity() -> list[CheckResult]:
    """Check heatmap data sanity for each symbol.

    For symbols with short train history (years < MIN_N), this check
    returns WARN/SKIP instead of FAIL since it's impossible to have
    MIN_N observations per cell.
    """
    import pandas as pd

    results = []
    keys_repo = CalendarKeysRepo()
    returns_repo = ReturnsRepo()

    for symbol in DEFAULT_SYMBOLS:
        keys_df = keys_repo.get_keys(symbol)
        returns_df = returns_repo.get_returns(symbol)

        if keys_df.empty or returns_df.empty:
            results.append(CheckResult(
                name=f"heatmap_sanity_{symbol}",
                status="fail",
                message=f"{symbol}: No data available",
            ))
            continue

        # Filter to train window
        keys_df = keys_df[keys_df["date"] <= TRAIN_END]
        returns_df = returns_df[returns_df["date"] <= TRAIN_END]

        # Merge
        merged = pd.merge(keys_df, returns_df, on=["symbol", "date"], how="inner")

        if merged.empty:
            results.append(CheckResult(
                name=f"heatmap_sanity_{symbol}",
                status="fail",
                message=f"{symbol}: No merged data in train window",
            ))
            continue

        # Compute key metrics
        total_rows = len(merged)
        baseline = merged["up"].mean()

        # Count distinct years in train window
        merged["year"] = pd.to_datetime(merged["date"]).dt.year
        years_in_window = merged["year"].nunique()

        # Group by month/day
        grouped = merged.groupby(["month", "day"]).agg(
            n=("up", "count"),
        ).reset_index()

        total_cells = len(grouped)
        max_cell_n = int(grouped["n"].max())
        sum_n = int(grouped["n"].sum())
        cells_with_min_n = len(grouped[grouped["n"] >= MIN_N])

        details = {
            "total_cells": total_cells,
            "max_cell_n": max_cell_n,
            "years_in_window": years_in_window,
            "sum_n": sum_n,
            "total_rows": total_rows,
            "baseline": baseline,
            "cells_with_min_n": cells_with_min_n,
        }

        # Basic sanity: join is healthy
        join_healthy = total_cells > 0 and sum_n > 0 and abs(sum_n - total_rows) <= 5

        if not join_healthy:
            results.append(CheckResult(
                name=f"heatmap_sanity_{symbol}",
                status="fail",
                message=f"{symbol}: Join unhealthy - total_cells={total_cells}, sum_n={sum_n}, total_rows={total_rows}",
                details=details,
            ))
            continue

        # Check if history is too short for MIN_N
        if max_cell_n < MIN_N:
            # History too short - WARN/SKIP, not FAIL
            results.append(CheckResult(
                name=f"heatmap_sanity_{symbol}",
                status="warn",
                message=f"{symbol}: Train history too short for MIN_N={MIN_N}; max_cell_n={max_cell_n}, years={years_in_window}",
                details=details,
            ))
        else:
            # Normal case: check we have enough cells with MIN_N observations
            # For symbols with sufficient history, expect most cells to qualify
            threshold = 200 if years_in_window >= 50 else 100
            passed = cells_with_min_n >= threshold

            if passed:
                results.append(CheckResult(
                    name=f"heatmap_sanity_{symbol}",
                    status="pass",
                    message=f"{symbol}: {total_cells} cells, {cells_with_min_n} with n>={MIN_N}, baseline={baseline:.4f}",
                    details=details,
                ))
            else:
                results.append(CheckResult(
                    name=f"heatmap_sanity_{symbol}",
                    status="fail",
                    message=f"{symbol}: Only {cells_with_min_n} cells with n>={MIN_N} (threshold={threshold})",
                    details=details,
                ))

    return results


def check_signal_detail_sanity(run_id: str) -> list[CheckResult]:
    """Check top signal detail sanity for each symbol."""
    results = []
    signals_repo = SignalsRepo()

    for symbol in DEFAULT_SYMBOLS:
        # Get top train-scored signal
        top_signals = signals_repo.get_top_signals(run_id, "train", 1, symbol)

        if top_signals.empty:
            results.append(CheckResult(
                name=f"signal_detail_{symbol}",
                status="pass",  # No signals is OK
                message=f"{symbol}: No eligible signals",
            ))
            continue

        top = top_signals.iloc[0]
        signal_id = top["signal_id"]
        key = json.loads(top["key_json"]) if isinstance(top["key_json"], str) else top["key_json"]
        family = top["family"]

        if family == "CDOY":
            label = f"M{key.get('month'):02d}D{key.get('day'):02d}"
        else:
            label = f"TDOM{key.get('tdom')}"

        # Get train/test/full stats
        train_stats = signals_repo.get_signal_stats(run_id, "train")
        test_stats = signals_repo.get_signal_stats(run_id, "test")
        full_stats = signals_repo.get_signal_stats(run_id, "full")

        train_row = train_stats[train_stats["signal_id"] == signal_id]
        test_row = test_stats[test_stats["signal_id"] == signal_id]
        full_row = full_stats[full_stats["signal_id"] == signal_id]

        issues = []
        details = {"signal": f"{family} {label}", "direction": top["direction"]}

        # Check train n >= MIN_N
        if not train_row.empty:
            train_n = int(train_row.iloc[0]["n"])
            train_wr = float(train_row.iloc[0]["win_rate"])
            details["train_n"] = train_n
            details["train_wr"] = f"{train_wr:.4f}"
            if train_n < MIN_N:
                issues.append(f"train n={train_n} < {MIN_N}")
        else:
            issues.append("train stats missing")

        # Check test row exists
        if not test_row.empty:
            test_n = int(test_row.iloc[0]["n"])
            test_wr = float(test_row.iloc[0]["win_rate"])
            details["test_n"] = test_n
            details["test_wr"] = f"{test_wr:.4f}"
            if test_n == 0:
                issues.append("test n=0")
        else:
            issues.append("test stats missing")

        # Check full n ~ train + test
        if not full_row.empty and not train_row.empty and not test_row.empty:
            full_n = int(full_row.iloc[0]["n"])
            expected_full = train_n + test_n
            details["full_n"] = full_n
            # Allow some variance for boundary dates
            if abs(full_n - expected_full) > 5:
                issues.append(f"full n={full_n} != train+test={expected_full}")

        if len(issues) == 0:
            msg = f"{symbol}: {family} {label} {top['direction']} - train={details.get('train_n')}/{details.get('train_wr')}, test={details.get('test_n')}/{details.get('test_wr')}"
            status = "pass"
        else:
            msg = f"{symbol}: {'; '.join(issues)}"
            status = "fail"

        results.append(CheckResult(
            name=f"signal_detail_{symbol}",
            status=status,
            message=msg,
            details=details,
        ))

    return results


def check_forward_calendar_coverage(run_id: str, num_days: int = 60) -> list[CheckResult]:
    """Check forward calendar coverage and explain empty cases."""
    results = []
    signals_repo = SignalsRepo()

    today = date.today()
    end_date = today + timedelta(days=num_days)
    future_keys = build_future_calendar_keys(today, end_date)

    # Get eligible signals
    all_signals = signals_repo.get_signal_stats(run_id, "train", eligible_only=True)

    for symbol in DEFAULT_SYMBOLS:
        symbol_signals = all_signals[all_signals["symbol"] == symbol]

        if symbol_signals.empty:
            results.append(CheckResult(
                name=f"forward_coverage_{symbol}",
                status="pass",  # Informational
                message=f"{symbol}: No eligible signals",
                details={"cdoy_count": 0, "tdom_count": 0},
            ))
            continue

        cdoy_signals = symbol_signals[symbol_signals["family"] == "CDOY"]
        tdom_signals = symbol_signals[symbol_signals["family"] == "TDOM"]

        # Count matches in forward calendar
        cdoy_matches = 0
        tdom_matches = 0
        next_events = []

        for _, key_row in future_keys.iterrows():
            month = int(key_row["month"])
            day = int(key_row["day"])
            tdom = int(key_row["tdom"])

            for _, sig in cdoy_signals.iterrows():
                key = json.loads(sig["key_json"]) if isinstance(sig["key_json"], str) else sig["key_json"]
                if key.get("month") == month and key.get("day") == day:
                    cdoy_matches += 1
                    if len(next_events) < 3:
                        next_events.append(f"{key_row['date']} M{month:02d}D{day:02d}")

            for _, sig in tdom_signals.iterrows():
                key = json.loads(sig["key_json"]) if isinstance(sig["key_json"], str) else sig["key_json"]
                if key.get("tdom") == tdom:
                    tdom_matches += 1
                    if len(next_events) < 3:
                        next_events.append(f"{key_row['date']} TDOM{tdom}")

        total_matches = cdoy_matches + tdom_matches

        results.append(CheckResult(
            name=f"forward_coverage_{symbol}",
            status="pass",  # Informational, not a pass/fail
            message=f"{symbol}: {len(cdoy_signals)} CDOY, {len(tdom_signals)} TDOM eligible; {total_matches} events in next {num_days} days",
            details={
                "cdoy_eligible": len(cdoy_signals),
                "tdom_eligible": len(tdom_signals),
                "cdoy_matches": cdoy_matches,
                "tdom_matches": tdom_matches,
                "next_events": next_events[:3],
            },
        ))

    return results


def run_all_checks(run_id: str | None = None) -> list[CheckResult]:
    """Run all validation checks."""
    results = []

    # Get run_id if not provided
    if run_id is None:
        runs_repo = RunsRepo()
        latest = runs_repo.get_latest_run()
        if latest:
            run_id = latest["run_id"]

    # Data coverage
    results.extend(check_data_coverage())

    # Forward calendar correctness
    results.extend(check_forward_calendar_correctness())

    # Heatmap sanity
    results.extend(check_heatmap_sanity())

    # Signal detail sanity (requires run_id)
    if run_id:
        results.extend(check_signal_detail_sanity(run_id))
        results.extend(check_forward_calendar_coverage(run_id))

    return results


def format_results(results: list[CheckResult]) -> str:
    """Format check results for display."""
    lines = []
    lines.append("=" * 60)
    lines.append("CALENDAR EDGE LAB VALIDATION")
    lines.append("=" * 60)

    # Count by status
    passed = sum(1 for r in results if r.status == "pass")
    failed = sum(1 for r in results if r.status == "fail")
    warned = sum(1 for r in results if r.status in ("warn", "skip"))

    # Group by category
    categories = {}
    for r in results:
        category = r.name.split("_")[0]
        if category not in categories:
            categories[category] = []
        categories[category].append(r)

    # Status display mapping
    status_labels = {
        "pass": "PASS",
        "fail": "FAIL",
        "warn": "WARN",
        "skip": "SKIP",
    }

    for category, checks in categories.items():
        lines.append("")
        lines.append(f"[{category.upper()}]")
        for r in checks:
            label = status_labels.get(r.status, r.status.upper())
            lines.append(f"  [{label}] {r.message}")

    lines.append("")
    lines.append("=" * 60)
    summary_parts = [f"{passed} passed"]
    if warned > 0:
        summary_parts.append(f"{warned} warned")
    summary_parts.append(f"{failed} failed")
    lines.append(f"SUMMARY: {', '.join(summary_parts)}")
    lines.append("=" * 60)

    return "\n".join(lines)
