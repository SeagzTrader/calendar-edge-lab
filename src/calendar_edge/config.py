"""Configuration constants for Calendar Edge Lab v1."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "calendar_edge.db"

# Symbols
DEFAULT_SYMBOLS = ["^GSPC", "^DJI", "^IXIC"]

# Scan params
MIN_N = 20
MIN_WIN_RATE = 0.60
MIN_DELTA = 0.05
MIN_DECADE_N = 8

# Walk-forward split
TRAIN_END = "2009-12-31"
TEST_START = "2010-01-01"

# FDR thresholds for evidence tiers
FDR_VALIDATED = 0.10  # Tier 1: Validated (q <= 0.10)
FDR_PROMISING = 0.20  # Tier 2: Promising (0.10 < q <= 0.20)
P_EXPLORATORY = 0.05  # Tier 3: Exploratory (raw p <= 0.05)

# Legacy alias
FDR_THRESHOLD = FDR_VALIDATED

# Logging
LOG_LEVEL = "INFO"
