"""CLI for Calendar Edge Lab."""

import argparse
import logging
import sys

from calendar_edge.config import DB_PATH, DEFAULT_SYMBOLS, LOG_LEVEL

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("calendar_edge")


def cmd_init_db(args: argparse.Namespace) -> int:
    """Initialize the database."""
    from calendar_edge.db import init_db

    logger.info("Initializing database...")
    init_db()
    logger.info(f"Database initialized at {DB_PATH}")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    """Ingest price data."""
    from calendar_edge.db import PricesRepo
    from calendar_edge.ingest import YFinanceProvider

    symbols = args.symbols or DEFAULT_SYMBOLS
    start_date = args.start or "1950-01-01"

    logger.info(f"Ingesting data for symbols: {symbols}")
    logger.info(f"Start date: {start_date}")

    provider = YFinanceProvider()
    repo = PricesRepo()  # Uses default DB_PATH

    for symbol in symbols:
        logger.info(f"Fetching {symbol}...")
        df = provider.fetch(symbol, start_date)

        if df.empty:
            logger.warning(f"No data returned for {symbol}")
            continue

        count = repo.upsert_prices(df, symbol, "yfinance", "adjusted")
        min_date, max_date = repo.get_date_range(symbol)
        logger.info(f"{symbol}: {count} rows, date range: {min_date} to {max_date}")

    return 0


def cmd_build_keys(args: argparse.Namespace) -> int:
    """Build calendar keys table."""
    from calendar_edge.db import CalendarKeysRepo, PricesRepo
    from calendar_edge.features import build_calendar_keys

    prices_repo = PricesRepo()
    keys_repo = CalendarKeysRepo()

    symbols = prices_repo.get_symbols()
    logger.info(f"Building calendar keys for {len(symbols)} symbols")

    for symbol in symbols:
        prices_df = prices_repo.get_prices(symbol)
        keys_df = build_calendar_keys(prices_df, symbol)
        count = keys_repo.upsert_keys(keys_df)
        logger.info(f"{symbol}: {count} calendar keys built")

    return 0


def cmd_build_returns(args: argparse.Namespace) -> int:
    """Build returns table."""
    from calendar_edge.db import PricesRepo, ReturnsRepo
    from calendar_edge.features import build_returns

    prices_repo = PricesRepo()
    returns_repo = ReturnsRepo()

    symbols = prices_repo.get_symbols()
    logger.info(f"Building returns for {len(symbols)} symbols")

    for symbol in symbols:
        prices_df = prices_repo.get_prices(symbol)
        returns_df = build_returns(prices_df, symbol)
        count = returns_repo.upsert_returns(returns_df)
        logger.info(f"{symbol}: {count} returns built")

    return 0


def cmd_run_scan(args: argparse.Namespace) -> int:
    """Run calendar effect scan."""
    import uuid

    from calendar_edge.db import RunsRepo
    from calendar_edge.scan import WalkForwardValidator

    run_id = str(uuid.uuid4())[:8]
    logger.info(f"Starting scan run: {run_id}")

    runs_repo = RunsRepo()
    runs_repo.create_run(run_id, notes="CLI scan run")

    validator = WalkForwardValidator(run_id)
    validator.run()
    logger.info(f"Scan complete. Run ID: {run_id}")

    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Generate reports."""
    from calendar_edge.db import RunsRepo
    from calendar_edge.report import export_csv, generate_markdown_report

    next_days = args.next or 60

    runs_repo = RunsRepo()
    run = runs_repo.get_latest_run()
    if not run:
        logger.error("No runs found. Run 'run-scan' first.")
        return 1

    run_id = run["run_id"]
    logger.info(f"Generating reports for run: {run_id}")

    # Export CSV
    csv_path = export_csv(run_id)
    logger.info(f"CSV exported to: {csv_path}")

    # Generate Markdown
    md_path = generate_markdown_report(run_id, next_days=next_days)
    logger.info(f"Markdown report: {md_path}")

    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Run validation checks on the database."""
    from calendar_edge.validation import format_results, run_all_checks

    print("Running validation checks...")
    print()

    results = run_all_checks()
    print(format_results(results))

    # Return exit code based on failures
    failed = sum(1 for r in results if not r.passed)
    return 1 if failed > 0 else 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="calendar-edge",
        description="Calendar Edge Lab - Calendar effect discovery and validation",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # init-db
    init_parser = subparsers.add_parser("init-db", help="Initialize the database")
    init_parser.set_defaults(func=cmd_init_db)

    # ingest
    ingest_parser = subparsers.add_parser("ingest", help="Ingest price data")
    ingest_parser.add_argument(
        "--symbols",
        nargs="+",
        help="Symbols to ingest (default: ^GSPC ^DJI ^IXIC)",
    )
    ingest_parser.add_argument(
        "--start",
        type=str,
        help="Start date (default: 1950-01-01)",
    )
    ingest_parser.set_defaults(func=cmd_ingest)

    # build-keys
    keys_parser = subparsers.add_parser("build-keys", help="Build calendar keys")
    keys_parser.set_defaults(func=cmd_build_keys)

    # build-returns
    returns_parser = subparsers.add_parser("build-returns", help="Build returns")
    returns_parser.set_defaults(func=cmd_build_returns)

    # run-scan
    scan_parser = subparsers.add_parser("run-scan", help="Run calendar effect scan")
    scan_parser.set_defaults(func=cmd_run_scan)

    # report
    report_parser = subparsers.add_parser("report", help="Generate reports")
    report_parser.add_argument(
        "--next",
        type=int,
        default=60,
        help="Number of days for forward calendar (default: 60)",
    )
    report_parser.set_defaults(func=cmd_report)

    # validate
    validate_parser = subparsers.add_parser("validate", help="Run validation checks")
    validate_parser.set_defaults(func=cmd_validate)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
