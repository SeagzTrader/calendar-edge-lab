"""Repository classes for database operations."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from calendar_edge.config import DB_PATH
from calendar_edge.db.connect import get_conn


class PricesRepo:
    """Repository for prices_daily table."""

    def __init__(self, db_path: Path | str | None = None):
        self.db_path = db_path or DB_PATH

    def upsert_prices(self, df: pd.DataFrame, symbol: str, source: str, close_definition: str) -> int:
        """Insert or replace price data."""
        with get_conn(self.db_path) as conn:
            cursor = conn.cursor()
            count = 0
            for _, row in df.iterrows():
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO prices_daily (symbol, date, close, source, close_definition)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (symbol, str(row["date"]), float(row["close"]), source, close_definition),
                )
                count += 1
            conn.commit()
        return count

    def get_prices(self, symbol: str, start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
        """Get prices for a symbol."""
        query = "SELECT date, close FROM prices_daily WHERE symbol = ?"
        params: list[Any] = [symbol]

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        query += " ORDER BY date"

        with get_conn(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=params)

    def get_symbols(self) -> list[str]:
        """Get list of symbols in database."""
        with get_conn(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT symbol FROM prices_daily ORDER BY symbol")
            return [row[0] for row in cursor.fetchall()]

    def get_date_range(self, symbol: str) -> tuple[str | None, str | None]:
        """Get min and max dates for a symbol."""
        with get_conn(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT MIN(date), MAX(date) FROM prices_daily WHERE symbol = ?",
                (symbol,),
            )
            row = cursor.fetchone()
            return (row[0], row[1]) if row else (None, None)

    def get_row_count(self, symbol: str) -> int:
        """Get row count for a symbol."""
        with get_conn(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM prices_daily WHERE symbol = ?", (symbol,))
            return cursor.fetchone()[0]


class CalendarKeysRepo:
    """Repository for calendar_keys table."""

    def __init__(self, db_path: Path | str | None = None):
        self.db_path = db_path or DB_PATH

    def upsert_keys(self, df: pd.DataFrame) -> int:
        """Insert or replace calendar keys."""
        with get_conn(self.db_path) as conn:
            cursor = conn.cursor()
            count = 0
            for _, row in df.iterrows():
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO calendar_keys
                    (symbol, date, dow, month, day, tdom, tdoy, is_month_start, is_month_end)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["symbol"],
                        str(row["date"]),
                        int(row["dow"]),
                        int(row["month"]),
                        int(row["day"]),
                        int(row["tdom"]),
                        int(row["tdoy"]),
                        int(row["is_month_start"]),
                        int(row["is_month_end"]),
                    ),
                )
                count += 1
            conn.commit()
        return count

    def get_keys(self, symbol: str, start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
        """Get calendar keys for a symbol."""
        query = "SELECT * FROM calendar_keys WHERE symbol = ?"
        params: list[Any] = [symbol]

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        query += " ORDER BY date"

        with get_conn(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=params)

    def get_row_count(self, symbol: str) -> int:
        """Get row count for a symbol."""
        with get_conn(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM calendar_keys WHERE symbol = ?", (symbol,))
            return cursor.fetchone()[0]


class ReturnsRepo:
    """Repository for returns_daily table."""

    def __init__(self, db_path: Path | str | None = None):
        self.db_path = db_path or DB_PATH

    def upsert_returns(self, df: pd.DataFrame) -> int:
        """Insert or replace returns."""
        with get_conn(self.db_path) as conn:
            cursor = conn.cursor()
            count = 0
            for _, row in df.iterrows():
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO returns_daily (symbol, date, ret_cc, up)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        row["symbol"],
                        str(row["date"]),
                        float(row["ret_cc"]),
                        int(row["up"]),
                    ),
                )
                count += 1
            conn.commit()
        return count

    def get_returns(self, symbol: str, start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
        """Get returns for a symbol."""
        query = "SELECT * FROM returns_daily WHERE symbol = ?"
        params: list[Any] = [symbol]

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        query += " ORDER BY date"

        with get_conn(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=params)

    def get_row_count(self, symbol: str) -> int:
        """Get row count for a symbol."""
        with get_conn(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM returns_daily WHERE symbol = ?", (symbol,))
            return cursor.fetchone()[0]


class RunsRepo:
    """Repository for runs table."""

    def __init__(self, db_path: Path | str | None = None):
        self.db_path = db_path or DB_PATH

    def create_run(self, run_id: str, params: dict | None = None, notes: str | None = None) -> None:
        """Create a new run."""
        with get_conn(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO runs (run_id, created_at, params_json, notes)
                VALUES (?, ?, ?, ?)
                """,
                (
                    run_id,
                    datetime.now().isoformat(),
                    json.dumps(params) if params else None,
                    notes,
                ),
            )
            conn.commit()

    def get_run(self, run_id: str) -> dict | None:
        """Get run by ID."""
        with get_conn(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def get_latest_run(self) -> dict | None:
        """Get the most recent run."""
        with get_conn(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM runs ORDER BY created_at DESC LIMIT 1")
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def list_runs(self) -> list[dict]:
        """List all runs."""
        with get_conn(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM runs ORDER BY created_at DESC")
            return [dict(row) for row in cursor.fetchall()]


class SignalsRepo:
    """Repository for signals and signal_stats tables."""

    def __init__(self, db_path: Path | str | None = None):
        self.db_path = db_path or DB_PATH

    def insert_signal(
        self,
        signal_id: str,
        run_id: str,
        symbol: str,
        family: str,
        direction: str,
        key_json: dict,
    ) -> None:
        """Insert a signal."""
        with get_conn(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO signals (signal_id, run_id, symbol, family, direction, key_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    signal_id,
                    run_id,
                    symbol,
                    family,
                    direction,
                    json.dumps(key_json),
                    datetime.now().isoformat(),
                ),
            )
            conn.commit()

    def insert_signal_stats(
        self,
        run_id: str,
        signal_id: str,
        window: str,
        stats: dict,
    ) -> None:
        """Insert signal statistics."""
        with get_conn(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO signal_stats
                (run_id, signal_id, window, n, wins, win_rate, avg_ret, median_ret,
                 ci_low, ci_high, p_value, fdr_q, decade_consistency, z_score, score, eligible)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    signal_id,
                    window,
                    stats.get("n", 0),
                    stats.get("wins", 0),
                    stats.get("win_rate", 0.0),
                    stats.get("avg_ret"),
                    stats.get("median_ret"),
                    stats.get("ci_low"),
                    stats.get("ci_high"),
                    stats.get("p_value"),
                    stats.get("fdr_q"),
                    stats.get("decade_consistency"),
                    stats.get("z_score"),
                    stats.get("score"),
                    stats.get("eligible", 1),
                ),
            )
            conn.commit()

    def get_signals(self, run_id: str, symbol: str | None = None, family: str | None = None) -> pd.DataFrame:
        """Get signals for a run."""
        query = "SELECT * FROM signals WHERE run_id = ?"
        params: list[Any] = [run_id]

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if family:
            query += " AND family = ?"
            params.append(family)

        with get_conn(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=params)

    def get_signal_stats(
        self,
        run_id: str,
        window: str | None = None,
        eligible_only: bool = False,
    ) -> pd.DataFrame:
        """Get signal statistics for a run."""
        query = """
            SELECT s.signal_id, s.run_id, s.symbol, s.family, s.direction, s.key_json,
                   ss.window, ss.n, ss.wins, ss.win_rate, ss.avg_ret, ss.median_ret,
                   ss.ci_low, ss.ci_high, ss.p_value, ss.fdr_q, ss.decade_consistency,
                   ss.z_score, ss.score, ss.eligible
            FROM signals s
            JOIN signal_stats ss ON s.signal_id = ss.signal_id AND s.run_id = ss.run_id
            WHERE s.run_id = ?
        """
        params: list[Any] = [run_id]

        if window:
            query += " AND ss.window = ?"
            params.append(window)
        if eligible_only:
            query += " AND ss.eligible = 1"

        query += " ORDER BY ss.score DESC NULLS LAST"

        with get_conn(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=params)

    def get_top_signals(
        self,
        run_id: str,
        window: str = "train",
        limit: int = 10,
        symbol: str | None = None,
        family: str | None = None,
    ) -> pd.DataFrame:
        """Get top signals by score."""
        query = """
            SELECT s.signal_id, s.run_id, s.symbol, s.family, s.direction, s.key_json,
                   ss.window, ss.n, ss.wins, ss.win_rate, ss.avg_ret, ss.median_ret,
                   ss.ci_low, ss.ci_high, ss.p_value, ss.fdr_q, ss.decade_consistency,
                   ss.z_score, ss.score, ss.eligible
            FROM signals s
            JOIN signal_stats ss ON s.signal_id = ss.signal_id AND s.run_id = ss.run_id
            WHERE s.run_id = ? AND ss.window = ? AND ss.eligible = 1
        """
        params: list[Any] = [run_id, window]

        if symbol:
            query += " AND s.symbol = ?"
            params.append(symbol)
        if family:
            query += " AND s.family = ?"
            params.append(family)

        query += " ORDER BY ss.score DESC LIMIT ?"
        params.append(limit)

        with get_conn(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=params)
