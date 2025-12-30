# Calendar Edge Lab

Research-grade calendar effect discovery and validation for U.S. cash index closes.

## Overview

Calendar Edge Lab is a quantitative research tool that discovers and validates calendar-based anomalies in equity index returns. It uses walk-forward methodology with FDR correction to identify statistically robust calendar effects.

## Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

```bash
# Initialize database
python -m calendar_edge.cli init-db

# Ingest price data
python -m calendar_edge.cli ingest --symbols ^GSPC --start 1950-01-01

# Build calendar keys
python -m calendar_edge.cli build-keys

# Build returns
python -m calendar_edge.cli build-returns

# Run calendar effect scan
python -m calendar_edge.cli run-scan

# Generate report
python -m calendar_edge.cli report --next 60

# Launch Streamlit app
streamlit run src/calendar_edge/app/streamlit_app.py
```

## Methodology

### Walk-Forward Validation

- **Train Period**: Data through 2009-12-31 (discovery + FDR + scoring)
- **Test Period**: Data from 2010-01-01 onward (out-of-sample evaluation)

### Calendar Effect Families

- **CDOY (Calendar Day of Year)**: Month + Day combinations (e.g., January 15)
- **TDOM (Trading Day of Month)**: Trading day number within each month

### Statistical Controls

- Benjamini-Hochberg FDR correction at 10% threshold
- Wilson confidence intervals for win rates
- Decade consistency factor for robustness
- Minimum sample size requirements (n >= 20)

### Scoring Formula

```
score = z_score * decade_consistency_factor * (1 - fdr_q)
```

## Limitations

- Historical patterns may not persist in future markets
- No consideration of transaction costs or slippage
- Cash index closes only (no futures/ETFs)
- v1 uses yfinance data only

## License

MIT License
