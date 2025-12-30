"""Database connection management for Calendar Edge Lab."""

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from calendar_edge.config import DB_PATH

logger = logging.getLogger("calendar_edge")


def get_conn(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Create a fresh SQLite connection.

    Args:
        db_path: Optional path to database. Defaults to config DB_PATH.

    Returns:
        SQLite connection with row_factory set to sqlite3.Row.
    """
    path = db_path or DB_PATH
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def conn_ctx(db_path: Path | str | None = None):
    """Context manager for database connections.

    Args:
        db_path: Optional path to database. Defaults to config DB_PATH.

    Yields:
        SQLite connection with row_factory set to sqlite3.Row.
    """
    conn = get_conn(db_path)
    try:
        yield conn
    finally:
        conn.close()


# Keep old name for backwards compatibility during transition
def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    """Deprecated: Use get_conn() or conn_ctx() instead."""
    return get_conn(db_path)


def init_db(db_path: Path | None = None) -> None:
    """Initialize the database with schema.

    Args:
        db_path: Optional path to database. Defaults to config DB_PATH.
    """
    path = db_path or DB_PATH

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Read schema
    schema_path = Path(__file__).parent / "schema.sql"
    with open(schema_path, "r") as f:
        schema = f.read()

    # Execute schema
    with conn_ctx(path) as conn:
        conn.executescript(schema)
        conn.commit()
        logger.info(f"Database initialized at {path}")
