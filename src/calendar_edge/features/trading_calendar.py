"""Trading calendar utilities using exchange-calendars."""

from datetime import date, timedelta

import exchange_calendars as xcals
import pandas as pd

# Default calendar for US equities (NYSE)
DEFAULT_CALENDAR = "XNYS"


def get_trading_sessions(
    start_date: date,
    end_date: date,
    calendar: str = DEFAULT_CALENDAR,
) -> list[date]:
    """Get list of trading sessions between start and end dates.

    Args:
        start_date: Start date (inclusive).
        end_date: End date (inclusive).
        calendar: Exchange calendar code (default: XNYS for NYSE).

    Returns:
        List of trading session dates.
    """
    cal = xcals.get_calendar(calendar)

    # Convert to pandas Timestamp for exchange-calendars
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    # Get sessions in range
    sessions = cal.sessions_in_range(start_ts, end_ts)

    return [s.date() for s in sessions]


def build_future_calendar_keys(
    start_date: date,
    end_date: date,
    calendar: str = DEFAULT_CALENDAR,
) -> pd.DataFrame:
    """Build calendar keys for future trading sessions.

    Args:
        start_date: Start date (inclusive).
        end_date: End date (inclusive).
        calendar: Exchange calendar code (default: XNYS for NYSE).

    Returns:
        DataFrame with columns: date, month, day, tdom, dow
        where tdom is the trading day of month (1-indexed).
    """
    sessions = get_trading_sessions(start_date, end_date, calendar)

    if not sessions:
        return pd.DataFrame(columns=["date", "month", "day", "tdom", "dow"])

    # Build calendar keys
    records = []
    current_month = None
    tdom = 0

    for session in sessions:
        # Reset tdom counter on new month
        if session.month != current_month:
            current_month = session.month
            tdom = 1
        else:
            tdom += 1

        records.append({
            "date": session.strftime("%Y-%m-%d"),
            "month": session.month,
            "day": session.day,
            "tdom": tdom,
            "dow": session.weekday(),  # 0=Monday, 6=Sunday
        })

    return pd.DataFrame(records)


def is_trading_day(
    check_date: date,
    calendar: str = DEFAULT_CALENDAR,
) -> bool:
    """Check if a date is a trading day.

    Args:
        check_date: Date to check.
        calendar: Exchange calendar code (default: XNYS for NYSE).

    Returns:
        True if the date is a trading session.
    """
    cal = xcals.get_calendar(calendar)
    ts = pd.Timestamp(check_date)

    return cal.is_session(ts)


def get_next_n_trading_days(
    start_date: date,
    n: int,
    calendar: str = DEFAULT_CALENDAR,
) -> list[date]:
    """Get the next N trading days starting from a date.

    Args:
        start_date: Start date (may or may not be a trading day).
        n: Number of trading days to return.
        calendar: Exchange calendar code (default: XNYS for NYSE).

    Returns:
        List of N trading session dates.
    """
    cal = xcals.get_calendar(calendar)

    # Look ahead enough days to ensure we get N trading sessions
    # (conservative: assume ~1.5x calendar days needed)
    end_date = start_date + timedelta(days=int(n * 2) + 30)

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    sessions = cal.sessions_in_range(start_ts, end_ts)

    return [s.date() for s in sessions[:n]]
