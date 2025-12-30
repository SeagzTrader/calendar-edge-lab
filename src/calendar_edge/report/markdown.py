"""Markdown report generation."""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

from calendar_edge.config import DB_PATH, PROJECT_ROOT, TEST_START, TRAIN_END
from calendar_edge.db import PricesRepo, ReturnsRepo, SignalsRepo
from calendar_edge.features import build_future_calendar_keys

logger = logging.getLogger("calendar_edge")


def generate_markdown_report(
    run_id: str,
    next_days: int = 60,
    output_dir: Path | None = None,
    db_path: Path | str | None = None,
) -> Path:
    """Generate Markdown report for a run.

    Args:
        run_id: Run identifier.
        next_days: Number of days for forward calendar.
        output_dir: Output directory.
        db_path: Optional path to database. Defaults to config DB_PATH.

    Returns:
        Path to generated report.
    """
    output_dir = output_dir or PROJECT_ROOT / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    db = db_path or DB_PATH
    signals_repo = SignalsRepo(db)
    prices_repo = PricesRepo(db)
    returns_repo = ReturnsRepo(db)

    lines = []
    lines.append("# Calendar Edge Lab Report")
    lines.append("")
    lines.append(f"**Run ID:** {run_id}")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Train Period:** <= {TRAIN_END}")
    lines.append(f"**Test Period:** >= {TEST_START}")
    lines.append("")

    # Symbols and baselines
    lines.append("## Symbols & Baselines")
    lines.append("")

    symbols = prices_repo.get_symbols()
    for symbol in symbols:
        returns_df = returns_repo.get_returns(symbol)
        if returns_df.empty:
            continue

        train_returns = returns_df[returns_df["date"] <= TRAIN_END]
        test_returns = returns_df[returns_df["date"] >= TEST_START]

        train_wr = train_returns["up"].mean() if len(train_returns) > 0 else 0
        test_wr = test_returns["up"].mean() if len(test_returns) > 0 else 0
        full_wr = returns_df["up"].mean()

        train_ret = train_returns["ret_cc"].mean() if len(train_returns) > 0 else 0
        test_ret = test_returns["ret_cc"].mean() if len(test_returns) > 0 else 0
        full_ret = returns_df["ret_cc"].mean()

        lines.append(f"### {symbol}")
        lines.append("")
        lines.append("| Window | N | Win Rate | Avg Return |")
        lines.append("|--------|---|----------|------------|")
        lines.append(f"| Train | {len(train_returns)} | {train_wr:.4f} | {train_ret:.6f} |")
        lines.append(f"| Test | {len(test_returns)} | {test_wr:.4f} | {test_ret:.6f} |")
        lines.append(f"| Full | {len(returns_df)} | {full_wr:.4f} | {full_ret:.6f} |")
        lines.append("")

    # Top signals
    lines.append("## Top Signals (Train)")
    lines.append("")

    for symbol in symbols:
        for family in ["CDOY", "TDOM"]:
            top_df = signals_repo.get_top_signals(run_id, "train", 10, symbol, family)
            if top_df.empty:
                continue

            lines.append(f"### {symbol} - {family}")
            lines.append("")
            lines.append("| Key | Dir | N | Win Rate | CI | Z-Score | DCF | FDR-Q | Score |")
            lines.append("|-----|-----|---|----------|-----|---------|-----|-------|-------|")

            for _, row in top_df.iterrows():
                key_dict = json.loads(row["key_json"]) if isinstance(row["key_json"], str) else row["key_json"]
                if family == "CDOY":
                    key_str = f"M{key_dict.get('month'):02d}D{key_dict.get('day'):02d}"
                else:
                    key_str = f"TDOM{key_dict.get('tdom')}"

                ci_str = f"[{row['ci_low']:.2f}-{row['ci_high']:.2f}]" if row['ci_low'] else "N/A"
                z_str = f"{row['z_score']:.2f}" if row['z_score'] else "N/A"
                dcf_str = f"{row['decade_consistency']:.2f}" if row['decade_consistency'] else "N/A"
                fdr_str = f"{row['fdr_q']:.3f}" if row['fdr_q'] else "N/A"
                score_str = f"{row['score']:.2f}" if row['score'] else "N/A"

                lines.append(
                    f"| {key_str} | {row['direction']} | {row['n']} | {row['win_rate']:.4f} | "
                    f"{ci_str} | {z_str} | {dcf_str} | {fdr_str} | {score_str} |"
                )
            lines.append("")

    # Train vs Test comparison for top signals
    lines.append("## Train vs Test Comparison")
    lines.append("")

    for symbol in symbols:
        top_train = signals_repo.get_top_signals(run_id, "train", 10, symbol)
        if top_train.empty:
            continue

        lines.append(f"### {symbol}")
        lines.append("")
        lines.append("| Signal | Dir | Train WR | Test WR | Train N | Test N |")
        lines.append("|--------|-----|----------|---------|---------|--------|")

        for _, train_row in top_train.iterrows():
            signal_id = train_row["signal_id"]
            key_dict = json.loads(train_row["key_json"]) if isinstance(train_row["key_json"], str) else train_row["key_json"]
            family = train_row["family"]

            if family == "CDOY":
                key_str = f"M{key_dict.get('month'):02d}D{key_dict.get('day'):02d}"
            else:
                key_str = f"TDOM{key_dict.get('tdom')}"

            # Get test stats
            test_stats = signals_repo.get_signal_stats(run_id, "test")
            test_row = test_stats[test_stats["signal_id"] == signal_id]

            if not test_row.empty:
                test_row = test_row.iloc[0]
                lines.append(
                    f"| {key_str} | {train_row['direction']} | {train_row['win_rate']:.4f} | "
                    f"{test_row['win_rate']:.4f} | {train_row['n']} | {test_row['n']} |"
                )
            else:
                lines.append(
                    f"| {key_str} | {train_row['direction']} | {train_row['win_rate']:.4f} | "
                    f"N/A | {train_row['n']} | 0 |"
                )
        lines.append("")

    # Forward calendar using NYSE trading calendar
    lines.append(f"## Forward Calendar (Next {next_days} Calendar Days)")
    lines.append("")
    lines.append("*Uses NYSE trading calendar - excludes weekends and market holidays.*")
    lines.append("")

    today = datetime.now().date()
    end_date = today + timedelta(days=next_days)

    # Build future calendar keys using trading calendar
    future_keys = build_future_calendar_keys(today, end_date)

    # Get all eligible signals
    all_signals = signals_repo.get_signal_stats(run_id, "train", eligible_only=True)

    if not all_signals.empty and not future_keys.empty:
        lines.append("| Date | DOW | Family | Signal | Symbol | Direction |")
        lines.append("|------|-----|--------|--------|--------|-----------|")

        dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        for _, key_row in future_keys.iterrows():
            date_str = key_row["date"]
            month = int(key_row["month"])
            day = int(key_row["day"])
            tdom = int(key_row["tdom"])
            dow = dow_names[int(key_row["dow"])]

            # Find matching signals (both CDOY and TDOM)
            for _, row in all_signals.iterrows():
                key_dict = json.loads(row["key_json"]) if isinstance(row["key_json"], str) else row["key_json"]
                family = row["family"]

                if family == "CDOY":
                    if key_dict.get("month") == month and key_dict.get("day") == day:
                        signal_name = f"M{month:02d}D{day:02d}"
                        lines.append(f"| {date_str} | {dow} | CDOY | {signal_name} | {row['symbol']} | {row['direction']} |")
                elif family == "TDOM":
                    if key_dict.get("tdom") == tdom:
                        signal_name = f"TDOM{tdom}"
                        lines.append(f"| {date_str} | {dow} | TDOM | {signal_name} | {row['symbol']} | {row['direction']} |")

        lines.append("")
    else:
        lines.append("*No eligible signals found.*")
        lines.append("")

    # Methodology
    lines.append("## Methodology")
    lines.append("")
    lines.append("### Walk-Forward Validation")
    lines.append("")
    lines.append(f"- **Discovery (Train):** Data through {TRAIN_END}")
    lines.append(f"- **Evaluation (Test):** Data from {TEST_START} onward")
    lines.append("- Signals are discovered on train data, then evaluated on unseen test data")
    lines.append("")
    lines.append("### Statistical Controls")
    lines.append("")
    lines.append("- **FDR Correction:** Benjamini-Hochberg at 10% threshold")
    lines.append("- **Minimum N:** 20 observations required per signal")
    lines.append("- **Minimum Delta:** 5% improvement over baseline")
    lines.append("- **Confidence Intervals:** Wilson score intervals")
    lines.append("")
    lines.append("### Scoring")
    lines.append("")
    lines.append("```")
    lines.append("score = z_score * decade_consistency * (1 - fdr_q)")
    lines.append("```")
    lines.append("")
    lines.append("### Limitations")
    lines.append("")
    lines.append("- Historical patterns may not persist")
    lines.append("- No transaction cost modeling")
    lines.append("- Cash index closes only")
    lines.append("")

    # Write report
    output_path = output_dir / f"report_{run_id}.md"
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    logger.info(f"Report generated: {output_path}")
    return output_path
