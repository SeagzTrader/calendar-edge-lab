-- Calendar Edge Lab Database Schema

CREATE TABLE IF NOT EXISTS prices_daily (
    symbol TEXT NOT NULL,
    date TEXT NOT NULL,
    close REAL NOT NULL,
    source TEXT NOT NULL,
    close_definition TEXT NOT NULL,
    PRIMARY KEY (symbol, date, source, close_definition)
);

CREATE TABLE IF NOT EXISTS calendar_keys (
    symbol TEXT NOT NULL,
    date TEXT NOT NULL,
    dow INTEGER NOT NULL,
    month INTEGER NOT NULL,
    day INTEGER NOT NULL,
    tdom INTEGER NOT NULL,
    tdoy INTEGER NOT NULL,
    is_month_start INTEGER NOT NULL,
    is_month_end INTEGER NOT NULL,
    PRIMARY KEY (symbol, date)
);

CREATE TABLE IF NOT EXISTS returns_daily (
    symbol TEXT NOT NULL,
    date TEXT NOT NULL,
    ret_cc REAL NOT NULL,
    up INTEGER NOT NULL,
    PRIMARY KEY (symbol, date)
);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    params_json TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS signals (
    signal_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    family TEXT NOT NULL,
    direction TEXT NOT NULL,
    key_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS signal_stats (
    run_id TEXT NOT NULL,
    signal_id TEXT NOT NULL,
    window TEXT NOT NULL,
    n INTEGER NOT NULL,
    wins INTEGER NOT NULL,
    win_rate REAL NOT NULL,
    avg_ret REAL,
    median_ret REAL,
    ci_low REAL,
    ci_high REAL,
    p_value REAL,
    fdr_q REAL,
    decade_consistency REAL,
    z_score REAL,
    score REAL,
    eligible INTEGER DEFAULT 1,
    PRIMARY KEY (run_id, signal_id, window),
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    FOREIGN KEY (signal_id) REFERENCES signals(signal_id)
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_prices_symbol ON prices_daily(symbol);
CREATE INDEX IF NOT EXISTS idx_prices_date ON prices_daily(date);
CREATE INDEX IF NOT EXISTS idx_calendar_keys_symbol ON calendar_keys(symbol);
CREATE INDEX IF NOT EXISTS idx_returns_symbol ON returns_daily(symbol);
CREATE INDEX IF NOT EXISTS idx_signals_run ON signals(run_id);
CREATE INDEX IF NOT EXISTS idx_signal_stats_run ON signal_stats(run_id);
CREATE INDEX IF NOT EXISTS idx_signal_stats_signal ON signal_stats(signal_id);
