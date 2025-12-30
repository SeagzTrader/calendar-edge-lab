"""Walk-forward validation for calendar effects."""

import logging
import uuid
from pathlib import Path

import pandas as pd

from calendar_edge.config import (
    DB_PATH,
    FDR_THRESHOLD,
    MIN_DECADE_N,
    MIN_DELTA,
    MIN_N,
    TEST_START,
    TRAIN_END,
)
from calendar_edge.db import CalendarKeysRepo, PricesRepo, ReturnsRepo, SignalsRepo
from calendar_edge.scan.cdoy import CDOYScanner
from calendar_edge.scan.fdr import apply_fdr
from calendar_edge.scan.scoring import compute_decade_consistency, compute_score
from calendar_edge.scan.stats import compute_stats, compute_z_score
from calendar_edge.scan.tdom import TDOMScanner

logger = logging.getLogger("calendar_edge")


class WalkForwardValidator:
    """Walk-forward validation framework."""

    def __init__(self, run_id: str, db_path: Path | str | None = None):
        """Initialize validator.

        Args:
            run_id: Unique run identifier.
            db_path: Optional path to database. Defaults to config DB_PATH.
        """
        self.run_id = run_id
        self.db_path = db_path or DB_PATH
        self.prices_repo = PricesRepo(self.db_path)
        self.keys_repo = CalendarKeysRepo(self.db_path)
        self.returns_repo = ReturnsRepo(self.db_path)
        self.signals_repo = SignalsRepo(self.db_path)

    def run(self) -> None:
        """Execute walk-forward validation."""
        symbols = self.prices_repo.get_symbols()
        logger.info(f"Running walk-forward validation for {len(symbols)} symbols")

        for symbol in symbols:
            logger.info(f"Processing {symbol}...")
            self._process_symbol(symbol)

    def _process_symbol(self, symbol: str) -> None:
        """Process a single symbol."""
        # Load data
        keys_df = self.keys_repo.get_keys(symbol)
        returns_df = self.returns_repo.get_returns(symbol)

        if keys_df.empty or returns_df.empty:
            logger.warning(f"No data for {symbol}")
            return

        # Split into windows
        train_keys = keys_df[keys_df["date"] <= TRAIN_END].copy()
        train_returns = returns_df[returns_df["date"] <= TRAIN_END].copy()

        _test_keys = keys_df[keys_df["date"] >= TEST_START].copy()  # noqa: F841
        test_returns = returns_df[returns_df["date"] >= TEST_START].copy()

        # Compute baselines
        train_baseline = train_returns["up"].mean() if len(train_returns) > 0 else 0.5
        test_baseline = test_returns["up"].mean() if len(test_returns) > 0 else 0.5
        full_baseline = returns_df["up"].mean() if len(returns_df) > 0 else 0.5

        logger.info(f"{symbol} train baseline: {train_baseline:.4f}, n={len(train_returns)}")
        logger.info(f"{symbol} test baseline: {test_baseline:.4f}, n={len(test_returns)}")

        # Run scanners on train data
        scanners = [
            CDOYScanner(train_keys, train_returns),
            TDOMScanner(train_keys, train_returns),
        ]

        discovered_signals = []

        for scanner in scanners:
            family = scanner.get_family()
            candidates_before = 0
            candidates_after = 0

            for key, returns_subset in scanner.scan():
                candidates_before += 1
                n = len(returns_subset)

                if n < MIN_N:
                    continue

                wins = int(returns_subset["up"].sum())
                win_rate = wins / n

                # Determine direction
                if win_rate > train_baseline:
                    direction = "UP"
                elif win_rate < train_baseline:
                    direction = "DOWN"
                else:
                    continue  # No edge

                # Compute directional success and delta
                if direction == "UP":
                    s = win_rate
                    bs = train_baseline
                else:
                    s = 1 - win_rate
                    bs = 1 - train_baseline

                delta = s - bs

                if delta < MIN_DELTA:
                    continue

                candidates_after += 1

                # Compute train stats
                train_stats = compute_stats(returns_subset, train_baseline, direction)
                z_score = compute_z_score(win_rate, train_baseline, direction, n)

                # Compute decade consistency (on train window)
                decade_deltas = self._compute_decade_deltas(
                    keys_df[keys_df["date"] <= TRAIN_END],
                    returns_df[returns_df["date"] <= TRAIN_END],
                    key,
                    family,
                    train_baseline,
                    direction,
                )
                overall_delta_sign = 1 if delta > 0 else -1
                dcf = compute_decade_consistency(decade_deltas, overall_delta_sign)

                discovered_signals.append({
                    "key": key,
                    "family": family,
                    "direction": direction,
                    "train_stats": train_stats,
                    "z_score": z_score,
                    "decade_consistency": dcf,
                    "p_value": train_stats["p_value"],
                })

            logger.info(f"{symbol} {family}: {candidates_before} candidates, {candidates_after} passed filters")

        # Apply FDR per family
        for family in ["CDOY", "TDOM"]:
            family_signals = [s for s in discovered_signals if s["family"] == family]
            if not family_signals:
                continue

            p_values = [s["p_value"] for s in family_signals]
            q_values = apply_fdr(p_values)

            for sig, q in zip(family_signals, q_values):
                sig["fdr_q"] = q
                sig["score"] = compute_score(sig["z_score"], sig["decade_consistency"], q)
                sig["eligible"] = 1 if q <= FDR_THRESHOLD and sig["train_stats"]["n"] >= MIN_N else 0

        logger.info(f"{symbol}: {len(discovered_signals)} signals discovered, FDR applied")

        # Persist signals and compute test/full stats
        for sig in discovered_signals:
            signal_id = str(uuid.uuid4())[:8]

            # Insert signal
            self.signals_repo.insert_signal(
                signal_id=signal_id,
                run_id=self.run_id,
                symbol=symbol,
                family=sig["family"],
                direction=sig["direction"],
                key_json=sig["key"],
            )

            # Train stats
            train_stats_full = {
                **sig["train_stats"],
                "z_score": sig["z_score"],
                "decade_consistency": sig["decade_consistency"],
                "fdr_q": sig.get("fdr_q"),
                "score": sig.get("score"),
                "eligible": sig.get("eligible", 1),
            }
            self.signals_repo.insert_signal_stats(
                self.run_id, signal_id, "train", train_stats_full
            )

            # Test stats (evaluate discovered signal on test data)
            test_subset = self._get_returns_for_key(
                keys_df[keys_df["date"] >= TEST_START],
                returns_df[returns_df["date"] >= TEST_START],
                sig["key"],
                sig["family"],
            )
            test_stats = compute_stats(test_subset, test_baseline, sig["direction"])
            test_stats["eligible"] = 1 if test_stats["n"] >= MIN_N else 0
            self.signals_repo.insert_signal_stats(
                self.run_id, signal_id, "test", test_stats
            )

            # Full stats (context only)
            full_subset = self._get_returns_for_key(
                keys_df,
                returns_df,
                sig["key"],
                sig["family"],
            )
            full_stats = compute_stats(full_subset, full_baseline, sig["direction"])
            full_stats["eligible"] = 1 if full_stats["n"] >= MIN_N else 0
            self.signals_repo.insert_signal_stats(
                self.run_id, signal_id, "full", full_stats
            )

    def _get_returns_for_key(
        self,
        keys_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        key: dict,
        family: str,
    ) -> pd.DataFrame:
        """Get returns subset for a given key."""
        if family == "CDOY":
            mask = (keys_df["month"] == key["month"]) & (keys_df["day"] == key["day"])
        elif family == "TDOM":
            mask = keys_df["tdom"] == key["tdom"]
        else:
            return pd.DataFrame(columns=["date", "ret_cc", "up"])

        dates = keys_df[mask]["date"].tolist()
        return returns_df[returns_df["date"].isin(dates)][["date", "ret_cc", "up"]]

    def _compute_decade_deltas(
        self,
        keys_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        key: dict,
        family: str,
        baseline: float,
        direction: str,
    ) -> dict[str, float]:
        """Compute delta for each decade."""
        # Extract years
        returns_df = returns_df.copy()
        returns_df["year"] = pd.to_datetime(returns_df["date"]).dt.year
        returns_df["decade"] = (returns_df["year"] // 10) * 10

        keys_df = keys_df.copy()
        keys_df["year"] = pd.to_datetime(keys_df["date"]).dt.year
        keys_df["decade"] = (keys_df["year"] // 10) * 10

        result = {}

        for decade in sorted(keys_df["decade"].unique()):
            decade_keys = keys_df[keys_df["decade"] == decade]
            decade_returns = returns_df[returns_df["decade"] == decade]

            subset = self._get_returns_for_key(decade_keys, decade_returns, key, family)

            if len(subset) < MIN_DECADE_N:
                continue

            wins = subset["up"].sum()
            n = len(subset)
            win_rate = wins / n

            # Directional success
            if direction == "UP":
                s = win_rate
                bs = baseline
            else:
                s = 1 - win_rate
                bs = 1 - baseline

            delta = s - bs
            result[str(decade)] = delta

        return result
