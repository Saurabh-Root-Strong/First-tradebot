"""Canonical DDL for the per-day intraday DuckDB store.

Single source of truth for the schema of data/intraday/<date>.duckdb. The writer
(intraday_db) and every reader/validator that opens these files (session_replay,
footprint_validate, eod_sync, ...) should build/migrate via init_intraday() instead
of carrying their own copy of the CREATE statements.

Pure strings + a tiny apply helper — imports nothing, so any layer can use it.

Tables: ticks · candles · oi_snapshots · futures_quotes · signals · trade_setups
        · chain_snapshots (per-strike, with delta/iv greeks for faithful replay).
"""
from __future__ import annotations

INTRADAY_DDL = """
-- ── Raw tick store ────────────────────────────────────────────────────────────
-- One row per WebSocket SymbolUpdate message (Fyers L1, ~1 tick/sec per index).
-- Primary key on (ts, symbol) deduplicates WebSocket reconnect replays.
--
-- tick_vol  = volume increment from previous tick (derived, always 0 for indices
--             since NSE computed indices carry no traded volume — meaningful for
--             futures symbols if they are added later).
-- cum_vol   = Fyers cumulative day volume at this tick (raw from feed).
-- day_open/high/low  = expanding session OHLC as broadcast by exchange feed.
-- ch / chp  = points / % change from previous session close.
CREATE TABLE IF NOT EXISTS ticks (
    ts          TIMESTAMPTZ NOT NULL,
    symbol      VARCHAR     NOT NULL,
    ltp         DOUBLE      NOT NULL,
    tick_vol    BIGINT      DEFAULT 0,
    cum_vol     BIGINT      DEFAULT 0,
    day_open    DOUBLE,
    day_high    DOUBLE,
    day_low     DOUBLE,
    ch          DOUBLE,
    chp         DOUBLE,
    PRIMARY KEY (ts, symbol)
);

CREATE TABLE IF NOT EXISTS candles (
    ts         TIMESTAMPTZ NOT NULL,
    date       DATE        NOT NULL,
    symbol     VARCHAR     NOT NULL,
    resolution VARCHAR     NOT NULL,
    open       DOUBLE,
    high       DOUBLE,
    low        DOUBLE,
    close      DOUBLE,
    volume     BIGINT,
    PRIMARY KEY (ts, symbol, resolution)
);

CREATE TABLE IF NOT EXISTS oi_snapshots (
    ts             TIMESTAMPTZ NOT NULL,
    date           DATE        NOT NULL,
    symbol         VARCHAR     NOT NULL,
    spot           DOUBLE,
    atm            INTEGER,
    pcr            DOUBLE,
    total_call_oi  BIGINT,
    total_put_oi   BIGINT,
    atm_call_oi    BIGINT,
    atm_put_oi     BIGINT,
    atm_call_iv    DOUBLE,
    atm_put_iv     DOUBLE,
    atm_iv         DOUBLE,
    atm_call_prem  DOUBLE,
    atm_put_prem   DOUBLE,
    call_wall      INTEGER,
    put_wall       INTEGER,
    max_pain       INTEGER,
    near_call_oi   BIGINT,
    near_put_oi    BIGINT,
    put_skew       DOUBLE,
    PRIMARY KEY (ts, symbol)
);

CREATE TABLE IF NOT EXISTS futures_quotes (
    ts             TIMESTAMPTZ NOT NULL,
    date           DATE        NOT NULL,
    symbol         VARCHAR     NOT NULL,
    near_ltp       DOUBLE,
    next_ltp       DOUBLE,
    far_ltp        DOUBLE,
    near_basis     DOUBLE,
    next_basis     DOUBLE,
    roll_spread    DOUBLE,
    term_structure VARCHAR,
    near_vol       BIGINT,
    next_vol       BIGINT,
    PRIMARY KEY (ts, symbol)
);

CREATE TABLE IF NOT EXISTS signals (
    ts             TIMESTAMPTZ NOT NULL,
    date           DATE        NOT NULL,
    symbol         VARCHAR     NOT NULL,
    weighted_score DOUBLE,
    overall        VARCHAR,
    score_5min     DOUBLE,
    score_15min    DOUBLE,
    score_60min    DOUBLE,
    score_daily    DOUBLE,
    signal_5min    VARCHAR,
    signal_15min   VARCHAR,
    signal_60min   VARCHAR,
    signal_daily   VARCHAR,
    rsi_5min       DOUBLE,
    rsi_15min      DOUBLE,
    macd_hist_15m  DOUBLE,
    close_price    DOUBLE,
    vwap_15min     DOUBLE,
    bull_tfs       INTEGER,
    bear_tfs       INTEGER,
    PRIMARY KEY (ts, symbol)
);

CREATE TABLE IF NOT EXISTS trade_setups (
    ts              TIMESTAMPTZ NOT NULL,
    date            DATE        NOT NULL,
    symbol          VARCHAR     NOT NULL,
    timeframe       VARCHAR     NOT NULL,
    signal          VARCHAR,
    composite_score DOUBLE,
    confidence      INTEGER,
    direction       VARCHAR,
    l1_tech         DOUBLE,
    l2_oi           DOUBLE,
    l3_velocity     DOUBLE,
    l4_inst         DOUBLE,
    l5_futures      DOUBLE,
    l6_iv           DOUBLE,
    l7_pcr          DOUBLE,
    l8_maxpain      DOUBLE,
    l9_context      DOUBLE,
    agreement       INTEGER,
    phase           VARCHAR,
    spot            DOUBLE,
    atm_iv          DOUBLE,
    PRIMARY KEY (ts, symbol, timeframe)
);

-- ── Per-strike option-chain snapshots ─────────────────────────────────────────
-- Written every ~3 min by the OI background poller for strikes near ATM.
-- oich / ltpch are vs PREVIOUS SESSION CLOSE (raw Fyers chain fields), so the
-- 4-quadrant OI-premium read (writing / buildup / covering / unwinding) can be
-- reconstructed per strike for any moment of the session — the strike-level
-- detail that total-OI aggregates hide (e.g. gap-up call unwinding).
CREATE TABLE IF NOT EXISTS chain_snapshots (
    ts      TIMESTAMPTZ NOT NULL,
    date    DATE        NOT NULL,
    symbol  VARCHAR     NOT NULL,
    strike  INTEGER     NOT NULL,
    side    VARCHAR     NOT NULL,
    ltp     DOUBLE,
    ltpch   DOUBLE,
    oi      BIGINT,
    oich    BIGINT,
    volume  BIGINT,
    delta   DOUBLE,
    iv      DOUBLE,
    expiry  BIGINT NOT NULL DEFAULT 0,   -- option expiry epoch (0 = legacy single-expiry)
    PRIMARY KEY (ts, symbol, strike, side, expiry)
);
"""

# The six core capture tables, in canonical order (chain_snapshots is per-strike
# and handled on its own path — parquet-exported / queried separately).
CAPTURE_TABLES = ("ticks", "candles", "oi_snapshots", "futures_quotes", "signals", "trade_setups")

# Columns added to chain_snapshots after the table first shipped — applied as
# idempotent ALTERs so an existing duckdb (CREATE IF NOT EXISTS won't add columns)
# gains them. greeks for faithful replay (Phase 4.1). Old rows stay NULL.
CHAIN_SNAPSHOT_MIGRATIONS = ("delta DOUBLE", "iv DOUBLE", "expiry BIGINT")


def init_intraday(conn) -> None:
    """Create every table + apply idempotent migrations on an open DuckDB connection."""
    conn.execute(INTRADAY_DDL)
    for _col in CHAIN_SNAPSHOT_MIGRATIONS:
        try:
            conn.execute(f"ALTER TABLE chain_snapshots ADD COLUMN IF NOT EXISTS {_col}")
        except Exception:
            pass
