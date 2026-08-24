-- IDX Signal Dashboard — canonical schema.
--
-- This file is the single source of truth for the store's shape. Python is the
-- only writer; the R/Shiny app opens the same file read-only.
--
-- Everything here is CREATE ... IF NOT EXISTS: running init-db against an
-- existing store must be a no-op, never a wipe.

CREATE TABLE IF NOT EXISTS meta (
    key         VARCHAR PRIMARY KEY,
    value       VARCHAR NOT NULL,
    updated_at  TIMESTAMP NOT NULL
);

-- ---------------------------------------------------------------------------
-- tickers — the universe, append-only.
--
-- A ticker is NEVER deleted. When it leaves the source listing it is marked
-- is_active = false and stamped with delisted_at. Rebuilding the universe from
-- "what exists today" makes every backtest optimistic, because the companies
-- that failed have been quietly removed from history.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS tickers (
    ticker       VARCHAR PRIMARY KEY,   -- Yahoo form, e.g. BBCA.JK
    idx_code     VARCHAR NOT NULL,      -- IDX form, e.g. BBCA
    name         VARCHAR,
    sector       VARCHAR,
    board        VARCHAR,               -- Utama / Pengembangan / Ekonomi Baru / Pemantauan Khusus
    listing_date DATE,
    is_active    BOOLEAN NOT NULL DEFAULT TRUE,
    first_seen   DATE NOT NULL,         -- first sync that saw this ticker
    last_seen    DATE NOT NULL,         -- most recent sync that saw it
    delisted_at  DATE,                  -- set when it stops appearing; never cleared silently
    source       VARCHAR NOT NULL,
    updated_at   TIMESTAMP NOT NULL
);

-- ---------------------------------------------------------------------------
-- prices — RAW OHLCV. Not adjusted.
--
-- auto_adjust=False at the fetch layer, always explicit. Adjustment is computed
-- on read from split_factor / dividend so we can always recover the number the
-- broker screen actually showed on the day.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS prices (
    ticker       VARCHAR NOT NULL,
    date         DATE NOT NULL,
    open         DOUBLE,
    high         DOUBLE,
    low          DOUBLE,
    close        DOUBLE,
    volume       BIGINT,
    split_factor DOUBLE NOT NULL DEFAULT 1.0,
    dividend     DOUBLE NOT NULL DEFAULT 0.0,
    source       VARCHAR NOT NULL,
    ingested_at  TIMESTAMP NOT NULL,
    PRIMARY KEY (ticker, date)
);

-- ---------------------------------------------------------------------------
-- indices — market benchmarks, kept OUT of `prices` and `tickers` on purpose.
--
-- IHSG is not a stock. Storing it alongside the universe would put it into
-- every screen, every backtest and every "average stock" baseline, quietly
-- corrupting the very comparison it exists to provide.
--
-- Same raw-price rule as `prices`: no adjustment, adjustment on read.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS indices (
    symbol      VARCHAR NOT NULL,      -- Yahoo form, e.g. '^JKSE'
    name        VARCHAR,
    date        DATE NOT NULL,
    open        DOUBLE,
    high        DOUBLE,
    low         DOUBLE,
    close       DOUBLE,
    volume      BIGINT,
    source      VARCHAR NOT NULL,
    ingested_at TIMESTAMP NOT NULL,
    PRIMARY KEY (symbol, date)
);

CREATE INDEX IF NOT EXISTS idx_indices_date ON indices (date);

-- ---------------------------------------------------------------------------
-- price_quality — one row per price bar, flagging what is wrong with it.
--
-- Bad data that looks like good data is the main risk in this project, so the
-- gate lives at the store rather than in the UI. Every check FLAGS the row; no
-- check deletes it. Dropping a bar would silently change history, and a gap we
-- created ourselves is indistinguishable from a gap the exchange had.
--
-- The magnitude is stored next to each flag (how long the stale run was, how
-- big the jump was) so the scale of a problem is visible, not just its
-- existence.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS price_quality (
    ticker               VARCHAR NOT NULL,
    date                 DATE NOT NULL,
    insufficient_history BOOLEAN NOT NULL DEFAULT FALSE,
    stale_price          BOOLEAN NOT NULL DEFAULT FALSE,
    zero_volume          BOOLEAN NOT NULL DEFAULT FALSE,
    suspicious_jump      BOOLEAN NOT NULL DEFAULT FALSE,
    gap                  BOOLEAN NOT NULL DEFAULT FALSE,
    impossible_bar       BOOLEAN NOT NULL DEFAULT FALSE,  -- high<low or high/low not the extreme
    stale_run_length     INTEGER,   -- consecutive identical closes ending here
    jump_pct             DOUBLE,    -- signed close-to-close move, percent
    gap_days             INTEGER,   -- calendar days since the previous bar
    bars_available       INTEGER NOT NULL,
    is_clean             BOOLEAN NOT NULL,
    computed_at          TIMESTAMP NOT NULL,
    PRIMARY KEY (ticker, date)
);

-- ---------------------------------------------------------------------------
-- indicators — computed once, here, by Python. Never recomputed in R.
--
-- bars_available is stored per row so downstream code can always tell
-- "MA200 is genuinely below MA100" apart from "MA200 does not exist yet".
-- That distinction is the direct fix for the silent-downgrade bug (section 7.4).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS indicators (
    ticker         VARCHAR NOT NULL,
    date           DATE NOT NULL,
    close          DOUBLE,
    ma5            DOUBLE,
    ma20           DOUBLE,
    ma50           DOUBLE,
    ma100          DOUBLE,
    ma200          DOUBLE,
    rsi14          DOUBLE,              -- Wilder smoothing, ewm(alpha=1/14, adjust=False)
    macd           DOUBLE,
    macd_signal    DOUBLE,
    macd_hist      DOUBLE,
    -- Ichimoku and Donchian are DISPLAY indicators. They are stored here so the
    -- dashboard reads rather than recomputes (invariant 1), but the signal
    -- engine does not touch them — the six criteria are MA alignment and RSI,
    -- and quietly widening that definition would change what the backtest
    -- measured without anyone deciding to.
    --
    -- Senkou A/B are stored UNSHIFTED, at the date they were computed. Ichimoku
    -- draws them 26 bars forward; doing that shift on render keeps the table
    -- free of future-dated rows that would look like lookahead.
    tenkan         DOUBLE,
    kijun          DOUBLE,
    senkou_a       DOUBLE,
    senkou_b       DOUBLE,
    donchian_upper DOUBLE,
    donchian_lower DOUBLE,
    -- Bollinger(20, 2) and the Fractal Chaos Bands are display indicators too,
    -- ported from the Pine script the owner charts with. Same rule as above:
    -- stored so the UI reads rather than recomputes, and untouched by the
    -- signal engine.
    --
    -- fcb_* are stored at the date they were computed. The script draws them
    -- two bars back, because a fractal is not confirmable until two bars later;
    -- that shift happens on render, so no row here is future-dated.
    bb_basis       DOUBLE,
    bb_upper       DOUBLE,
    bb_lower       DOUBLE,
    fcb_upper      DOUBLE,
    fcb_lower      DOUBLE,
    fcb_dot_upper  DOUBLE,
    fcb_dot_lower  DOUBLE,
    -- Two lower-panel indicators fit to the owner's TradingView screenshots,
    -- with no Pine source to port (see compute/indicators.py). ribbon is a
    -- 4-state trend x momentum ribbon (0..3); capitulation is a placeholder
    -- down flag (0/1). Both approximate and display-only — the signal engine
    -- does not read them.
    ribbon         INTEGER,
    capitulation   INTEGER,
    bars_available INTEGER NOT NULL,
    computed_at    TIMESTAMP NOT NULL,
    PRIMARY KEY (ticker, date)
);

-- ---------------------------------------------------------------------------
-- signals — computed for EVERY historical date, not just today.
-- The backtest reads this table; it cannot exist as a "current view" only.
--
-- The six criteria are not six independent rules (section 7.3). They are two
-- axes:  alignment_depth in {0, 3, 4, 5}  and  rsi_gate in {true, false}.
-- The six labels are rendered from these, never stored as six flags.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS signals (
    ticker               VARCHAR NOT NULL,
    date                 DATE NOT NULL,
    alignment_depth      INTEGER NOT NULL,   -- 0 / 3 / 4 / 5
    rsi_gate             BOOLEAN,            -- NULL when rsi14 is unavailable
    above_ma5            BOOLEAN,
    days_since_ma5_cross INTEGER,            -- NULL if no cross in the trailing window
    tier                 VARCHAR,
    data_quality         VARCHAR NOT NULL,
    bars_available       INTEGER NOT NULL,
    computed_at          TIMESTAMP NOT NULL,
    PRIMARY KEY (ticker, date),
    CONSTRAINT signals_alignment_depth_valid
        CHECK (alignment_depth IN (0, 3, 4, 5)),
    -- insufficient_history is its own state, never collapsed into "failed the test".
    CONSTRAINT signals_data_quality_valid
        CHECK (data_quality IN ('ok', 'insufficient_history', 'stale', 'suspect'))
);

-- ---------------------------------------------------------------------------
-- labels — backtest outcomes, one row per (ticker, entry_date, rule_version).
-- rule_version is part of the key so re-running under a changed outcome
-- definition does not silently overwrite the previous measurement.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS labels (
    ticker           VARCHAR NOT NULL,
    entry_date       DATE NOT NULL,      -- the signal date
    rule_version     VARCHAR NOT NULL,
    fill_date        DATE,               -- next bar's open; no same-bar fills
    fill_price       DOUBLE,
    exit_date        DATE,
    exit_price       DOUBLE,
    outcome          VARCHAR,            -- target / stop / timeout / open
    bars_held        INTEGER,
    return_pct       DOUBLE,             -- net of the cost model
    gross_return_pct DOUBLE,
    mae              DOUBLE,             -- maximum adverse excursion
    mfe              DOUBLE,             -- maximum favourable excursion
    computed_at      TIMESTAMP NOT NULL,
    PRIMARY KEY (ticker, entry_date, rule_version),
    CONSTRAINT labels_outcome_valid
        CHECK (outcome IS NULL OR outcome IN ('target', 'stop', 'timeout', 'open'))
);

-- ---------------------------------------------------------------------------
-- ingest_log — EVERY fetch attempt writes a row. No silent failures.
--
-- An empty response is empty_response. It is never interpreted as "delisted",
-- and a network failure is never reported as a fact about a stock.
-- ---------------------------------------------------------------------------
CREATE SEQUENCE IF NOT EXISTS ingest_log_id_seq START 1;

CREATE TABLE IF NOT EXISTS ingest_log (
    log_id        BIGINT PRIMARY KEY DEFAULT nextval('ingest_log_id_seq'),
    run_id        VARCHAR NOT NULL,
    ticker        VARCHAR,            -- NULL for run-level rows (start / abort / summary)
    source        VARCHAR NOT NULL,
    status        VARCHAR NOT NULL,
    rows_written  INTEGER NOT NULL DEFAULT 0,
    error_message VARCHAR,
    attempt       INTEGER NOT NULL DEFAULT 1,
    started_at    TIMESTAMP NOT NULL,
    finished_at   TIMESTAMP,
    CONSTRAINT ingest_log_status_valid CHECK (status IN (
        'ok',
        'rate_limited',
        'network_error',
        'empty_response',
        'parse_error',
        'circuit_open',
        'run_started',
        'run_finished',
        'run_aborted'
    ))
);

CREATE INDEX IF NOT EXISTS idx_prices_date       ON prices (date);
CREATE INDEX IF NOT EXISTS idx_indicators_date   ON indicators (date);
CREATE INDEX IF NOT EXISTS idx_signals_date      ON signals (date);
CREATE INDEX IF NOT EXISTS idx_ingest_log_run    ON ingest_log (run_id);
CREATE INDEX IF NOT EXISTS idx_ingest_log_status ON ingest_log (status);
