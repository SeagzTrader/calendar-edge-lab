"""CSV export functionality."""

import json
import logging
from pathlib import Path

import pandas as pd

from calendar_edge.config import DB_PATH, PROJECT_ROOT
from calendar_edge.db import SignalsRepo

logger = logging.getLogger("calendar_edge")


def export_csv(run_id: str, output_dir: Path | None = None, db_path: Path | str | None = None) -> Path:
    """Export signal statistics to CSV.

    Args:
        run_id: Run identifier.
        output_dir: Output directory (default: PROJECT_ROOT/reports).
        db_path: Optional path to database. Defaults to config DB_PATH.

    Returns:
        Path to exported CSV file.
    """
    output_dir = output_dir or PROJECT_ROOT / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    repo = SignalsRepo(db_path or DB_PATH)

    # Get all signal stats
    stats_df = repo.get_signal_stats(run_id)

    if stats_df.empty:
        logger.warning(f"No signals found for run {run_id}")
        output_path = output_dir / f"signals_{run_id}.csv"
        pd.DataFrame().to_csv(output_path, index=False)
        return output_path

    # Parse key_json for readable columns
    def parse_key(key_json_str):
        try:
            return json.loads(key_json_str)
        except (json.JSONDecodeError, TypeError):
            return {}

    stats_df["key_parsed"] = stats_df["key_json"].apply(parse_key)

    # Flatten key fields
    for key_field in ["month", "day", "tdom"]:
        stats_df[key_field] = stats_df["key_parsed"].apply(lambda x: x.get(key_field))

    # Select and order columns
    columns = [
        "signal_id",
        "run_id",
        "symbol",
        "family",
        "direction",
        "month",
        "day",
        "tdom",
        "window",
        "n",
        "wins",
        "win_rate",
        "avg_ret",
        "median_ret",
        "ci_low",
        "ci_high",
        "p_value",
        "fdr_q",
        "decade_consistency",
        "z_score",
        "score",
        "eligible",
    ]

    # Only include columns that exist
    columns = [c for c in columns if c in stats_df.columns]
    export_df = stats_df[columns]

    output_path = output_dir / f"signals_{run_id}.csv"
    export_df.to_csv(output_path, index=False)
    logger.info(f"Exported {len(export_df)} rows to {output_path}")

    return output_path
