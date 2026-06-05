"""
nightly_sync.py  --  Nightly data snapshot: Daily_Cash_Market → Tradebot.

Scheduled via Windows Task Scheduler (setup_scheduler.bat) at 7:30 PM weekdays.
Reads all relevant tables from market_data.duckdb, computes derived signals
(max pain, FPI equity flows, large-cap delivery quality), and writes a compact
pre-processed snapshot to data/tradebot_context.db (SQLite).

After writing, invalidates both bridge caches so next morning's first read
starts cold and loads the freshly-synced local data instead of stale DuckDB.

Usage:
    python nightly_sync.py           # run sync (skips if already done today)
    python nightly_sync.py --force   # re-sync even if already done today
    python nightly_sync.py --status  # print last sync status and exit
"""

import argparse
import datetime
import logging
import sqlite3
import sys
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE      = Path(__file__).parent
SOURCE_DB  = Path(r"D:\Python Projects\Daily_Cash_Market\data\market_data.duckdb")
LOCAL_DB   = _HERE / "data" / "tradebot_context.db"

# Symbol table — keeps this file self-contained (no bridge import at sync time)
_SYMBOLS = {
    "NSE:NIFTY50-INDEX":    {"dcm": "NIFTY",      "idx": "Nifty 50",                  "fii_cat": "NIFTY FUTURES"},
    "NSE:NIFTYBANK-INDEX":  {"dcm": "BANKNIFTY",  "idx": "Nifty Bank",                "fii_cat": "BANKNIFTY FUTURES"},
    "NSE:FINNIFTY-INDEX":   {"dcm": "FINNIFTY",   "idx": "Nifty Financial Services",  "fii_cat": "FINNIFTY FUTURES"},
    "NSE:MIDCPNIFTY-INDEX": {"dcm": "MIDCPNIFTY", "idx": "Nifty Midcap Select",       "fii_cat": "MIDCPNIFTY FUTURES"},
}

_HEAVY_SECTORS = {
    "Financial Services", "IT", "Oil Gas & Consumable Fuels",
    "FMCG", "Automobile and Auto Components", "Healthcare",
}

log = logging.getLogger("nightly_sync")


# ── Local DB schema ───────────────────────────────────────────────────────────

_DDL = [
    # One row per sync day
    """
    CREATE TABLE IF NOT EXISTS sync_metadata (
        sync_date    TEXT PRIMARY KEY,
        synced_at    TEXT NOT NULL,
        source_date  TEXT,
        status       TEXT DEFAULT 'ok',
        error_msg    TEXT
    )
    """,
    # Market-wide context (VIX, breadth, FII aggregate, delivery quality)
    """
    CREATE TABLE IF NOT EXISTS market_context (
        sync_date                TEXT PRIMARY KEY,
        india_vix                REAL,
        india_vix_5d_chg         REAL,
        breadth_pct              REAL,
        heavy_breadth_pct        REAL,
        fii_market_5d_cr         REAL,
        fii_market_10d_cr        REAL,
        fii_market_5d_pos        INTEGER,
        nifty50_avg_deliv_pct    REAL,
        nifty50_high_deliv_count INTEGER
    )
    """,
    # Per-index context — one row per (sync_date, fyers_sym)
    # All fields pre-computed so bridges do zero heavy lifting at trading-session start
    """
    CREATE TABLE IF NOT EXISTS index_context (
        sync_date          TEXT,
        fyers_sym          TEXT,
        dcm_sym            TEXT,
        -- Key levels (from index_data, last 22 days)
        prev_close         REAL,
        eod_close          REAL,
        week_high          REAL,
        week_low           REAL,
        high_20d           REAL,
        low_20d            REAL,
        pct_5d             REAL,
        up_streak          INTEGER,
        pe_ratio           REAL,
        pb_ratio           REAL,
        -- Prediction (from prediction_log, latest row)
        pred_date          TEXT,
        direction          TEXT,
        confidence         TEXT,
        composite_score    REAL,
        signal_count       INTEGER,
        hmm_state          TEXT,
        memory_label       TEXT,
        range_low          REAL,
        range_high         REAL,
        target_close       REAL,
        expected_move_pts  REAL,
        feat_pcr           REAL,
        feat_carry         REAL,
        feat_fii_net       REAL,
        feat_fii_5d        REAL,
        feat_vix           REAL,
        feat_vix_5d_chg    REAL,
        feat_breadth       REAL,
        feat_hurst         REAL,
        feat_entropy       REAL,
        feat_oi_score      REAL,
        -- Prediction accuracy (last 30 filled predictions)
        pred_acc_30d       REAL,
        pred_hi_conf_acc   REAL,
        pred_sample        INTEGER,
        -- FII futures flow (from fii_derivatives_stats, last 10 days)
        fii_today_cr       REAL,
        fii_5d_cr          REAL,
        fii_10d_cr         REAL,
        fii_5d_pos_days    INTEGER,
        fii_oi             INTEGER,
        -- FAO participant positioning (from fao_participant, latest date)
        fii_fut_net        INTEGER,
        client_fut_net     INTEGER,
        fii_call_net       INTEGER,
        fii_put_net        INTEGER,
        -- EOD options chain aggregate (from fno_bhavcopy)
        eod_call_oi        INTEGER,
        eod_put_oi         INTEGER,
        eod_pcr            REAL,
        eod_call_chg       INTEGER,
        eod_put_chg        INTEGER,
        -- Max pain + key OI levels (DERIVED from strike-level fno_bhavcopy)
        max_pain_price     REAL,
        max_pain_dist_pct  REAL,
        nearest_expiry     TEXT,
        top_call_strike    REAL,     -- strike with highest CE OI near spot (=Resistance)
        top_put_strike     REAL,     -- strike with highest PE OI near spot (=Support)
        -- FPI equity flows (from fpi_nsdl_flows — macro signal, same for all indices)
        fpi_equity_today_cr REAL,
        fpi_equity_5d_cr   REAL,
        fpi_equity_10d_cr  REAL,
        fpi_5d_pos_days    INTEGER,
        -- Exact replica fields (matching Daily_Cash_Market IndexPrediction output)
        spot_close         REAL,     -- prediction_log.spot_close (hero price on card)
        fii_today_cts      INTEGER,  -- daily net contracts from fii_derivatives_stats
        fii_net_change_1d  INTEGER,  -- fao_today_net - fao_prev_net (COV>0, ADD<0)
        PRIMARY KEY (sync_date, fyers_sym)
    )
    """,
]


def _init_local_db(con: sqlite3.Connection) -> None:
    for ddl in _DDL:
        con.execute(ddl)
    # ── Schema migration: add new columns to existing databases ──────────────
    # SQLite does not support adding columns inside CREATE TABLE IF NOT EXISTS;
    # ALTER TABLE ADD COLUMN is idempotent — silently ignored if already present.
    _new_cols = [
        ("index_context", "top_call_strike",      "REAL"),
        ("index_context", "top_put_strike",       "REAL"),
        ("index_context", "spot_close",           "REAL"),
        ("index_context", "fii_today_cts",        "INTEGER"),
        ("index_context", "fii_net_change_1d",    "INTEGER"),
        ("index_context", "fpi_equity_today_cr",  "REAL"),
        ("index_context", "fpi_equity_5d_cr",     "REAL"),
        ("index_context", "fpi_equity_10d_cr",    "REAL"),
        ("index_context", "fpi_5d_pos_days",      "INTEGER"),
    ]
    for tbl, col, typ in _new_cols:
        try:
            con.execute(f"ALTER TABLE {tbl} ADD COLUMN {col} {typ}")
        except Exception:
            pass   # column already exists — ignore
    con.commit()


# ── Derived signal: Max Pain ───────────────────────────────────────────────────

def _compute_max_pain(duck, dcm_sym: str, spot: float) -> tuple:
    """
    Compute EOD max pain, top call strike (R) and top put strike (S).

    Algorithm matches Daily_Cash_Market's _compute_key_levels() exactly:
      - Max pain   = strike minimising total ITM intrinsic value for option writers
      - top_call   = CE strike with highest OI within [spot-5%, spot+15%] band
      - top_put    = PE strike with highest OI within [spot-15%, spot+5%] band

    Returns (max_pain_price, dist_pct, expiry_str, top_call_strike, top_put_strike)
    where dist_pct is (max_pain - spot) / spot × 100.
    All values None on failure.
    """
    try:
        row = duck.execute("""
            SELECT MIN(expiry_date)
            FROM fno_bhavcopy
            WHERE symbol      = ?
              AND instrument  = 'OPTIDX'
              AND trade_date  = (SELECT MAX(trade_date) FROM fno_bhavcopy WHERE symbol = ?)
              AND expiry_date >= (SELECT MAX(trade_date) FROM fno_bhavcopy WHERE symbol = ?)
        """, [dcm_sym, dcm_sym, dcm_sym]).fetchone()

        if not row or not row[0]:
            return None, None, None, None, None

        expiry = row[0]

        df = duck.execute("""
            SELECT strike_price, option_type, open_interest
            FROM fno_bhavcopy
            WHERE symbol      = ?
              AND instrument  = 'OPTIDX'
              AND trade_date  = (SELECT MAX(trade_date) FROM fno_bhavcopy WHERE symbol = ?)
              AND expiry_date = ?
              AND option_type IN ('CE', 'PE')
              AND open_interest > 0
        """, [dcm_sym, dcm_sym, expiry]).df()

        if df.empty:
            return None, None, str(expiry), None, None

        ce_rows = df[df["option_type"] == "CE"].copy()
        pe_rows = df[df["option_type"] == "PE"].copy()

        ce_oi = dict(zip(ce_rows["strike_price"], ce_rows["open_interest"]))
        pe_oi = dict(zip(pe_rows["strike_price"], pe_rows["open_interest"]))
        strikes = sorted(set(ce_oi) | set(pe_oi))

        if not strikes:
            return None, None, str(expiry), None, None

        # ── Max pain ──────────────────────────────────────────────────────────
        min_pain, max_pain_price = float("inf"), strikes[len(strikes) // 2]
        for price in strikes:
            pain = (sum((price - k) * oi for k, oi in ce_oi.items() if k < price) +
                    sum((k - price) * oi for k, oi in pe_oi.items() if k > price))
            if pain < min_pain:
                min_pain, max_pain_price = pain, price

        # ── Top OI strikes: band-filtered, matching DCM _compute_key_levels ──
        # Resistance: CE with max OI in [spot−5%, spot+15%]
        # Support:    PE with max OI in [spot−15%, spot+5%]
        band     = float(spot) * 0.05 if spot else 0.0
        top_call = None
        top_put  = None

        if not ce_rows.empty and spot:
            nc = ce_rows[(ce_rows["strike_price"] >= spot - band) &
                         (ce_rows["strike_price"] <= spot + band * 3)]
            if nc.empty:
                nc = ce_rows
            top_call = float(nc.loc[nc["open_interest"].idxmax(), "strike_price"])

        if not pe_rows.empty and spot:
            np_ = pe_rows[(pe_rows["strike_price"] >= spot - band * 3) &
                          (pe_rows["strike_price"] <= spot + band)]
            if np_.empty:
                np_ = pe_rows
            top_put = float(np_.loc[np_["open_interest"].idxmax(), "strike_price"])

        dist_pct = None
        if spot and spot > 0:
            dist_pct = round((max_pain_price - spot) / spot * 100, 3)

        return float(max_pain_price), dist_pct, str(expiry), top_call, top_put

    except Exception as exc:
        log.warning("max_pain failed for %s: %s", dcm_sym, exc)
        return None, None, None, None, None


# ── Derived signal: FPI equity flows ─────────────────────────────────────────

def _fetch_fpi_flows(duck) -> dict:
    """
    FPI (Foreign Portfolio Investors) net equity investment — macro driver of NIFTY.

    FPI is a superset of FII: includes FIIs, foreign ETFs, QFIs.
    Net equity flow is THE primary determinant of sustained NIFTY direction.
    5D positive + 10D positive = structural tailwind regardless of intraday noise.
    """
    try:
        df = duck.execute("""
            SELECT trade_date, net_investment_cr
            FROM fpi_nsdl_flows
            WHERE LOWER(category) = 'equity'
            ORDER BY trade_date DESC
            LIMIT 10
        """).df()

        if df.empty:
            return {}

        return {
            "fpi_equity_today_cr": round(float(df["net_investment_cr"].iloc[0]), 1),
            "fpi_equity_5d_cr":    round(float(df["net_investment_cr"].head(5).sum()), 1),
            "fpi_equity_10d_cr":   round(float(df["net_investment_cr"].head(10).sum()), 1),
            "fpi_5d_pos_days":     int((df["net_investment_cr"].head(5) > 0).sum()),
        }
    except Exception as exc:
        log.warning("FPI flow fetch failed: %s", exc)
        return {}


# ── Derived signal: Large-cap delivery quality ────────────────────────────────

def _fetch_delivery_quality(duck) -> dict:
    """
    Large-cap delivery % signals institutional accumulation vs distribution.

    High delivery % (> 60%) in large-cap stocks = institutions holding positions
    rather than intraday trading = conviction buying / accumulation.
    Low delivery % (< 35%) = high-velocity intraday activity = institutional exits.
    """
    try:
        df = duck.execute("""
            SELECT dd.symbol, dd.deliv_per
            FROM daily_data dd
            JOIN sector_master sm ON dd.symbol = sm.symbol
            WHERE dd.trade_date         = (SELECT MAX(trade_date) FROM daily_data)
              AND dd.series             = 'EQ'
              AND dd.deliv_per          IS NOT NULL
              AND dd.deliv_per          > 0
              AND dd.turnover_lacs      >= 100
              AND LOWER(sm.market_cap_category) IN ('large cap', 'largecap', 'large-cap')
        """).df()

        if df.empty:
            return {}

        return {
            "nifty50_avg_deliv_pct":     round(float(df["deliv_per"].mean()), 1),
            "nifty50_high_deliv_count":  int((df["deliv_per"] > 60).sum()),
        }
    except Exception as exc:
        log.warning("delivery quality fetch failed: %s", exc)
        return {}


# ── Main sync ─────────────────────────────────────────────────────────────────

def run_sync(force: bool = False) -> bool:
    """
    Pull all relevant tables from Daily_Cash_Market's DuckDB, compute derived
    signals, write to local SQLite snapshot.  Returns True on success.

    Idempotent: if today's sync already succeeded, returns True immediately
    unless force=True.
    """
    today = datetime.date.today().isoformat()
    LOCAL_DB.parent.mkdir(parents=True, exist_ok=True)

    # Fast-path: already synced today
    if not force:
        try:
            _c = sqlite3.connect(str(LOCAL_DB))
            _init_local_db(_c)
            row = _c.execute(
                "SELECT status FROM sync_metadata WHERE sync_date = ?", [today]
            ).fetchone()
            _c.close()
            if row and row[0] == "ok":
                log.info("Already synced for %s — use --force to re-sync", today)
                return True
        except Exception:
            pass

    if not SOURCE_DB.exists():
        log.error("Source DB not found: %s", SOURCE_DB)
        _write_error(today, "Source DB not found")
        return False

    log.info("Starting nightly sync  source=%s", SOURCE_DB)
    started = datetime.datetime.now()

    try:
        import duckdb
        duck = duckdb.connect(str(SOURCE_DB), read_only=True)
    except Exception as exc:
        log.error("Cannot connect to source DB: %s", exc)
        _write_error(today, str(exc))
        return False

    try:
        local = sqlite3.connect(str(LOCAL_DB))
        _init_local_db(local)

        # ── Source freshness check ────────────────────────────────────────────
        src_row = duck.execute(
            "SELECT MAX(trade_date) FROM index_data WHERE index_name = 'Nifty 50'"
        ).fetchone()
        source_date = str(src_row[0]) if src_row and src_row[0] else None
        log.info("Source latest trade_date: %s", source_date)

        # ── Market-wide context ───────────────────────────────────────────────
        mc: dict = {"sync_date": today}

        # India VIX: level + 5-day change
        try:
            df = duck.execute("""
                SELECT close_val FROM index_data
                WHERE index_name = 'India VIX'
                ORDER BY trade_date DESC LIMIT 10
            """).df()
            if not df.empty:
                mc["india_vix"]        = round(float(df["close_val"].iloc[0]), 2)
                mc["india_vix_5d_chg"] = round(
                    float(df["close_val"].iloc[0]) -
                    float(df["close_val"].iloc[min(5, len(df) - 1)]), 2
                )
        except Exception as exc:
            log.warning("VIX fetch failed: %s", exc)

        # Market breadth: % stocks advancing per sector
        try:
            df = duck.execute("""
                SELECT sm.sector,
                       SUM(CASE WHEN dd.close_price > dd.prev_close THEN 1 ELSE 0 END) AS adv,
                       COUNT(*) AS total
                FROM daily_data dd
                JOIN sector_master sm ON dd.symbol = sm.symbol
                WHERE dd.trade_date = (SELECT MAX(trade_date) FROM daily_data)
                  AND dd.series = 'EQ'
                  AND dd.turnover_lacs >= 5
                GROUP BY sm.sector
            """).df()
            if not df.empty:
                rates = df["adv"] / df["total"] * 100
                mc["breadth_pct"] = round(float(rates.mean()), 1)
                heavy = df[df["sector"].isin(_HEAVY_SECTORS)]
                if not heavy.empty:
                    mc["heavy_breadth_pct"] = round(
                        float((heavy["adv"] / heavy["total"] * 100).mean()), 1
                    )
        except Exception as exc:
            log.warning("Breadth fetch failed: %s", exc)

        # FII aggregate futures flow across all four index categories
        try:
            df = duck.execute("""
                SELECT trade_date,
                       SUM(buy_value_cr - sell_value_cr) AS net_cr
                FROM fii_derivatives_stats
                WHERE category IN (
                    'NIFTY FUTURES', 'BANKNIFTY FUTURES',
                    'FINNIFTY FUTURES', 'MIDCPNIFTY FUTURES'
                )
                GROUP BY trade_date
                ORDER BY trade_date DESC
                LIMIT 10
            """).df()
            if not df.empty:
                mc["fii_market_5d_cr"]  = round(float(df["net_cr"].head(5).sum()), 1)
                mc["fii_market_10d_cr"] = round(float(df["net_cr"].head(10).sum()), 1)
                mc["fii_market_5d_pos"] = int((df["net_cr"].head(5) > 0).sum())
        except Exception as exc:
            log.warning("FII aggregate fetch failed: %s", exc)

        mc.update(_fetch_delivery_quality(duck))

        local.execute(
            "INSERT OR REPLACE INTO market_context VALUES (?,?,?,?,?,?,?,?,?,?)",
            [mc.get("sync_date"),
             mc.get("india_vix"), mc.get("india_vix_5d_chg"),
             mc.get("breadth_pct"), mc.get("heavy_breadth_pct"),
             mc.get("fii_market_5d_cr"), mc.get("fii_market_10d_cr"), mc.get("fii_market_5d_pos"),
             mc.get("nifty50_avg_deliv_pct"), mc.get("nifty50_high_deliv_count")]
        )

        # ── FAO participant snapshot — last 2 dates for today + 1D delta ─────
        # fii_net_change_1d = today_net - prev_net
        #   > 0 → FII covered shorts / added longs  → label "COV" (bullish)
        #   < 0 → FII added shorts / reduced longs  → label "ADD" (bearish)
        fao_data: dict      = {}   # latest date: {client_type: {fut_net, call_net, put_net}}
        fii_net_change_1d   = 0
        try:
            rows = duck.execute("""
                SELECT trade_date, client_type,
                       fut_idx_long,      fut_idx_short,
                       opt_idx_call_long, opt_idx_call_short,
                       opt_idx_put_long,  opt_idx_put_short
                FROM fao_participant
                WHERE data_type = 'OI'
                  AND trade_date IN (
                      SELECT DISTINCT trade_date FROM fao_participant
                      WHERE data_type = 'OI'
                      ORDER BY trade_date DESC LIMIT 2
                  )
                ORDER BY trade_date DESC, client_type
            """).fetchall()

            by_date: dict = {}
            for r in rows:
                dt, ct = r[0], r[1]
                if dt not in by_date:
                    by_date[dt] = {}
                by_date[dt][ct] = {
                    "fut_net":  (r[2] or 0) - (r[3] or 0),
                    "call_net": (r[4] or 0) - (r[5] or 0),
                    "put_net":  (r[6] or 0) - (r[7] or 0),
                }

            dates = sorted(by_date.keys(), reverse=True)
            if dates:
                fao_data = by_date[dates[0]]   # latest
            if len(dates) >= 2:
                net_today = fao_data.get("FII", {}).get("fut_net", 0)
                net_prev  = by_date[dates[1]].get("FII", {}).get("fut_net", 0)
                fii_net_change_1d = net_today - net_prev

        except Exception as exc:
            log.warning("FAO fetch failed: %s", exc)

        # ── FPI equity flows (macro signal, shared across all indices) ─────────
        fpi = _fetch_fpi_flows(duck)

        # ── Per-symbol context ────────────────────────────────────────────────
        for fyers_sym, meta in _SYMBOLS.items():
            dcm_sym  = meta["dcm"]
            idx_name = meta["idx"]
            fii_cat  = meta["fii_cat"]
            rd: dict = {"sync_date": today, "fyers_sym": fyers_sym, "dcm_sym": dcm_sym}

            # Index OHLC + key levels (22 days for 20D high/low)
            try:
                df = duck.execute("""
                    SELECT trade_date, high_val, low_val, close_val,
                           prev_close, pct_chg, pe_ratio, pb_ratio
                    FROM index_data
                    WHERE index_name = ?
                    ORDER BY trade_date DESC
                    LIMIT 22
                """, [idx_name]).df()
                if not df.empty:
                    prev = float(df["prev_close"].iloc[0]) or float(df["close_val"].iloc[0])
                    rd.update({
                        "prev_close": prev,
                        "eod_close":  float(df["close_val"].iloc[0]),
                        "high_20d":   float(df["high_val"].max()),
                        "low_20d":    float(df["low_val"].min()),
                        "week_high":  float(df["high_val"].head(5).max()),
                        "week_low":   float(df["low_val"].head(5).min()),
                        "pct_5d":     round(float(df["pct_chg"].head(5).sum()), 2),
                        "pe_ratio":   float(df["pe_ratio"].iloc[0]) if df["pe_ratio"].iloc[0] else None,
                        "pb_ratio":   float(df["pb_ratio"].iloc[0]) if df["pb_ratio"].iloc[0] else None,
                    })
                    streak = 0
                    for chg in df["pct_chg"]:
                        if chg and chg > 0:
                            streak += 1
                        else:
                            break
                    rd["up_streak"] = streak
            except Exception as exc:
                log.warning("Index OHLC failed for %s: %s", dcm_sym, exc)

            # Prediction log — latest prediction + feature vector + spot_close
            try:
                r = duck.execute("""
                    SELECT trade_date, direction_pred, confidence_pred, composite_score,
                           signal_count,
                           feat_pcr, feat_carry, feat_fii_net, feat_fii_5d_cumul,
                           feat_vix, feat_vix_5d_chg, feat_breadth,
                           feat_hurst, feat_entropy, feat_oi_score,
                           hmm_state, memory_label,
                           range_low, range_high, target_close, expected_move_pts,
                           spot_close
                    FROM prediction_log
                    WHERE fno_symbol = ?
                    ORDER BY trade_date DESC
                    LIMIT 1
                """, [dcm_sym]).fetchone()
                if r:
                    rd.update({
                        "pred_date":        str(r[0]),
                        "direction":        r[1],
                        "confidence":       r[2],
                        "composite_score":  float(r[3]) if r[3] is not None else 0.0,
                        "signal_count":     r[4],
                        "feat_pcr":         r[5],   "feat_carry":     r[6],
                        "feat_fii_net":     r[7],   "feat_fii_5d":    r[8],
                        "feat_vix":         r[9],   "feat_vix_5d_chg": r[10],
                        "feat_breadth":     r[11],  "feat_hurst":     r[12],
                        "feat_entropy":     r[13],  "feat_oi_score":  r[14],
                        "hmm_state":        r[15],  "memory_label":   r[16],
                        "range_low":        float(r[17]) if r[17] is not None else None,
                        "range_high":       float(r[18]) if r[18] is not None else None,
                        "target_close":     float(r[19]) if r[19] is not None else None,
                        "expected_move_pts": float(r[20]) if r[20] is not None else None,
                        "spot_close":       float(r[21]) if r[21] is not None else None,
                    })

                # 30-day rolling accuracy for confidence weighting in B1
                r2 = duck.execute("""
                    SELECT COUNT(*),
                           SUM(CASE WHEN was_correct THEN 1 ELSE 0 END),
                           AVG(CASE WHEN confidence_pred = 'HIGH' AND was_correct THEN 1.0
                                    WHEN confidence_pred = 'HIGH'                 THEN 0.0
                               END)
                    FROM (
                        SELECT was_correct, confidence_pred
                        FROM prediction_log
                        WHERE fno_symbol    = ?
                          AND outcome_filled = TRUE
                        ORDER BY trade_date DESC
                        LIMIT 30
                    ) sub
                """, [dcm_sym]).fetchone()
                if r2 and r2[0]:
                    rd["pred_acc_30d"]     = round((r2[1] or 0) / r2[0] * 100, 1)
                    rd["pred_hi_conf_acc"] = round(r2[2] * 100, 1) if r2[2] is not None else None
                    rd["pred_sample"]      = r2[0]
            except Exception as exc:
                log.warning("Prediction log failed for %s: %s", dcm_sym, exc)

            # FII futures flow — last 10 trading days
            try:
                df = duck.execute("""
                    SELECT buy_value_cr - sell_value_cr    AS net_cr,
                           buy_contracts - sell_contracts  AS net_cts,
                           oi_contracts
                    FROM fii_derivatives_stats
                    WHERE category = ?
                    ORDER BY trade_date DESC
                    LIMIT 10
                """, [fii_cat]).df()
                if not df.empty:
                    rd.update({
                        "fii_today_cr":  round(float(df["net_cr"].iloc[0]), 1),
                        "fii_today_cts": int(df["net_cts"].iloc[0])
                                         if df["net_cts"].iloc[0] is not None else 0,
                        "fii_5d_cr":       round(float(df["net_cr"].head(5).sum()), 1),
                        "fii_10d_cr":      round(float(df["net_cr"].head(10).sum()), 1),
                        "fii_5d_pos_days": int((df["net_cr"].head(5) > 0).sum()),
                        "fii_oi":          int(df["oi_contracts"].iloc[0]),
                    })
            except Exception as exc:
                log.warning("FII flow failed for %s: %s", dcm_sym, exc)

            # FAO: FII vs retail positioning + 1D change for ADD/COV display
            fii_p  = fao_data.get("FII",    {})
            client = fao_data.get("Client", {})
            rd.update({
                "fii_fut_net":       fii_p.get("fut_net",  0),
                "client_fut_net":    client.get("fut_net", 0),
                "fii_call_net":      fii_p.get("call_net", 0),
                "fii_put_net":       fii_p.get("put_net",  0),
                "fii_net_change_1d": fii_net_change_1d,
            })

            # EOD option chain aggregate (PCR + OI change via 2-day delta)
            # chg_in_oi from NSE bhavcopy is often 0; compute delta from consecutive days.
            try:
                df = duck.execute("""
                    SELECT trade_date, option_type, SUM(open_interest) AS oi
                    FROM fno_bhavcopy
                    WHERE symbol     = ?
                      AND instrument  = 'OPTIDX'
                      AND trade_date IN (
                          SELECT DISTINCT trade_date FROM fno_bhavcopy
                          WHERE symbol = ? ORDER BY trade_date DESC LIMIT 2
                      )
                      AND option_type IN ('CE', 'PE')
                    GROUP BY trade_date, option_type
                    ORDER BY trade_date DESC
                """, [dcm_sym, dcm_sym]).df()
                if not df.empty:
                    dates = sorted(df["trade_date"].unique(), reverse=True)
                    t0 = df[df["trade_date"] == dates[0]]
                    t1 = df[df["trade_date"] == dates[1]] if len(dates) > 1 else None
                    def _s(sub, opt): return int(sub[sub["option_type"] == opt]["oi"].sum()) if not sub[sub["option_type"] == opt].empty else 0
                    c_oi  = _s(t0, "CE"); p_oi = _s(t0, "PE")
                    c_chg = c_oi - _s(t1, "CE") if t1 is not None else 0
                    p_chg = p_oi - _s(t1, "PE") if t1 is not None else 0
                    rd.update({
                        "eod_call_oi":  c_oi,
                        "eod_put_oi":   p_oi,
                        "eod_pcr":      round(p_oi / c_oi, 3) if c_oi else 0.0,
                        "eod_call_chg": c_chg,
                        "eod_put_chg":  p_chg,
                    })
            except Exception as exc:
                log.warning("OI aggregate failed for %s: %s", dcm_sym, exc)

            # Max pain + top OI strikes — pass spot_close for accurate band filtering
            spot_ref = rd.get("spot_close") or rd.get("eod_close") or rd.get("prev_close") or 0
            mp_price, mp_dist, mp_expiry, top_call, top_put = _compute_max_pain(
                duck, dcm_sym, spot_ref
            )
            rd.update({
                "max_pain_price":    mp_price,
                "max_pain_dist_pct": mp_dist,
                "nearest_expiry":    mp_expiry,
                "top_call_strike":   top_call,
                "top_put_strike":    top_put,
            })

            # FPI equity flows (macro — same value for all four index symbols)
            rd.update(fpi)

            # Write this symbol's row to local DB
            cols         = list(rd.keys())
            placeholders = ",".join("?" * len(cols))
            local.execute(
                f"INSERT OR REPLACE INTO index_context "
                f"({','.join(cols)}) VALUES ({placeholders})",
                [rd[c] for c in cols]
            )
            log.info("  %-30s  pred=%s  fii5d=%s Cr  fpi5d=%s Cr  maxpain=%s",
                     fyers_sym,
                     rd.get("direction", "?"),
                     rd.get("fii_5d_cr", "?"),
                     rd.get("fpi_equity_5d_cr", "?"),
                     rd.get("max_pain_price", "?"))

        # ── Sync metadata ─────────────────────────────────────────────────────
        elapsed = (datetime.datetime.now() - started).total_seconds()
        local.execute(
            "INSERT OR REPLACE INTO sync_metadata VALUES (?,?,?,'ok',NULL)",
            [today, datetime.datetime.now().isoformat(), source_date]
        )
        local.commit()
        log.info("Sync complete in %.1fs → %s", elapsed, LOCAL_DB)

        _invalidate_bridge_caches()
        return True

    except Exception as exc:
        log.error("Sync failed: %s", exc, exc_info=True)
        _write_error(today, str(exc))
        return False

    finally:
        try:
            duck.close()
        except Exception:
            pass


def _write_error(today: str, msg: str) -> None:
    try:
        LOCAL_DB.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(str(LOCAL_DB))
        _init_local_db(con)
        con.execute(
            "INSERT OR REPLACE INTO sync_metadata VALUES (?,?,NULL,'error',?)",
            [today, datetime.datetime.now().isoformat(), msg[:500]]
        )
        con.commit()
        con.close()
    except Exception:
        pass


def _invalidate_bridge_caches() -> None:
    """
    Reset in-process bridge cache TTLs.  Only effective when nightly_sync is
    imported by the same Python process as the bridges (e.g. in tests).
    When run as a standalone script the bridges live in a separate process
    and this is a no-op — that is fine: at trading-session start the bridges
    call _load() and find today's local DB is fresh, so they read from it.
    """
    try:
        from daily_context_bridge import get_bridge
        bridge = get_bridge()
        with bridge._lock:
            bridge._loaded_at = 0.0
        log.info("daily_context_bridge cache invalidated")
    except Exception:
        pass

    try:
        from market_data_bridge import _cache, _lock
        with _lock:
            _cache.clear()
        log.info("market_data_bridge cache cleared")
    except Exception:
        pass


# ── Status display ────────────────────────────────────────────────────────────

def print_status() -> None:
    if not LOCAL_DB.exists():
        print(f"No local DB at {LOCAL_DB}")
        print("Run:  python nightly_sync.py")
        return

    con = sqlite3.connect(str(LOCAL_DB))

    print("\n─── Sync Log ─────────────────────────────────────────────────────────")
    rows = con.execute(
        "SELECT sync_date, status, source_date, synced_at, error_msg "
        "FROM sync_metadata ORDER BY sync_date DESC LIMIT 7"
    ).fetchall()
    for r in rows:
        status_icon = "✓" if r[1] == "ok" else "✗"
        print(f"  {status_icon} {r[0]}  status={r[1]}  source={r[2]}  at={r[3][:19]}")
        if r[4]:
            print(f"      ERROR: {r[4]}")

    print("\n─── Market Context ───────────────────────────────────────────────────")
    row = con.execute(
        "SELECT * FROM market_context ORDER BY sync_date DESC LIMIT 1"
    ).fetchone()
    if row:
        print(f"  Date: {row[0]}")
        print(f"  India VIX: {row[1]}  5D change: {row[2]:+.2f}")
        print(f"  Breadth: {row[3]}%  Heavy-sector breadth: {row[4]}%")
        print(f"  FII futures  5D: {row[5]:+,.0f} Cr  10D: {row[6]:+,.0f} Cr  ({row[7]}/5 buying days)")
        print(f"  Large-cap avg delivery: {row[8]}%  High-deliv stocks: {row[9]}")

    print("\n─── Index Context ────────────────────────────────────────────────────")
    rows = con.execute("""
        SELECT fyers_sym, pred_date, direction, confidence, composite_score,
               fii_5d_cr, fpi_equity_5d_cr, max_pain_price, max_pain_dist_pct,
               pred_acc_30d, eod_pcr
        FROM index_context
        WHERE sync_date = (SELECT MAX(sync_date) FROM index_context)
        ORDER BY fyers_sym
    """).fetchall()
    for r in rows:
        sym_short = r[0].replace("NSE:", "").replace("-INDEX", "")
        print(
            f"  {sym_short:15s}  pred={r[1]} {r[2]:10s}[{r[3]}]  score={r[4]:+5.1f}  "
            f"FII5d={r[5]:+7,.0f}Cr  FPI5d={r[6]:+7,.0f}Cr  "
            f"MaxPain={r[7]} ({r[8]:+.2f}%)  PCR={r[10]:.2f}  Acc={r[9]}%"
        )

    con.close()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Nightly snapshot: Daily_Cash_Market → Tradebot local DB"
    )
    parser.add_argument("--force",  action="store_true",
                        help="Re-sync even if already done today")
    parser.add_argument("--status", action="store_true",
                        help="Print last sync status and exit")
    args = parser.parse_args()

    if args.status:
        print_status()
        sys.exit(0)

    ok = run_sync(force=args.force)
    sys.exit(0 if ok else 1)
