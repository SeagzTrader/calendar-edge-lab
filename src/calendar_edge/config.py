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

# FDR
FDR_THRESHOLD = 0.10

# Logging
LOG_LEVEL = "INFO"
