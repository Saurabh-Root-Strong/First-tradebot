"""
daily_context_bridge.py  --  EOD structural context bridge.

Connects Tradebot to Daily_Cash_Market's DuckDB (read-only) and extracts
the seven sub-signals that form Layer 9: Daily Context (17% weight).

Sub-signals (each clamped to [-4, +4]):
  B1  Prediction Engine   24-signal EOD directional forecast + accuracy weight
  B2  HMM Regime          Hidden Markov state (Bull/Bear/Sideways) + Hurst filter
  B3  FII Flow Trend       5-day + 10-day net futures flow in Rs. Cr
  B4  Market Breadth       % sectors advancing + NIFTY heavyweight confirmation
  B5  VIX Regime           India VIX level and 5-day change
  B6  EOD OI Positioning   End-of-day PCR and OI change direction
  B7  FPI Capital Flows    FPI net equity investment 5D/10D (macro driver of NIFTY)

Data source priority (fastest → most authoritative):
  1. Local SQLite snapshot  data/tradebot_context.db  written by nightly_sync.py
     at 7:30 PM every weekday — preferred when today's sync is available.
  2. Live DuckDB read from Daily_Cash_Market's market_data.duckdb — fallback
     when no local snapshot exists or snapshot is from a prior day.

Key design decisions:
  - Read-only DuckDB access — never blocks or corrupts the source DB
  - Single lock acquisition per load — all data fetched in one DB session
  - 4-hour TTL cache — EOD data is stable during market hours
  - Background thread loads silently at startup — no dashboard latency
  - Full graceful degradation — returns (0, []) when DB unavailable
  - Exposes get_key_levels() for dashboard support/resistance display

Usage:
    from daily_context_bridge import get_bridge
    score, signals = get_bridge().score_index("NSE:NIFTY50-INDEX")
    levels         = get_bridge().get_key_levels("NSE:NIFTY50-INDEX")
    panel          = get_bridge().get_panel_data("NSE:NIFTY50-INDEX")
"""

import datetime
import sqlite3
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from core.obs import warn_once   # observe silently-swallowed field-computation failures

# ── Configuration ──────────────────────────────────────────────────────────────
DB_PATH      = Path(r"D:\Python Projects\Daily_Cash_Market\data\market_data.duckdb")
LOCAL_DB     = Path(__file__).parent / "data" / "tradebot_context.db"
CACHE_TTL    = 4 * 3600   # 4 hours — EOD data, stable during trading hours

# Symbol mapping: Fyers → Daily_Cash_Market fno_symbol
_SYM = {
    "NSE:NIFTY50-INDEX":    "NIFTY",
    "NSE:NIFTYBANK-INDEX":  "BANKNIFTY",
    "NSE:FINNIFTY-INDEX":   "FINNIFTY",
    "NSE:MIDCPNIFTY-INDEX": "MIDCPNIFTY",
}

# Nifty 50 heavyweight sectors (combined weight ~80% of index)
_HEAVY_SECTORS = {
    "Financial Services", "IT", "Oil Gas & Consumable Fuels",
    "FMCG", "Automobile and Auto Components", "Healthcare",
}

# Index name in index_data table
_IDX_NAME = {
    "NIFTY":      "Nifty 50",
    "BANKNIFTY":  "Nifty Bank",
    "FINNIFTY":   "Nifty Financial Services",
    "MIDCPNIFTY": "Nifty Midcap Select",
}

# FII futures category in fii_derivatives_stats
_FII_CAT = {
    "NIFTY":      "NIFTY FUTURES",
    "BANKNIFTY":  "BANKNIFTY FUTURES",
    "FINNIFTY":   "FINNIFTY FUTURES",
    "MIDCPNIFTY": "MIDCPNIFTY FUTURES",
}


# ── Bridge ─────────────────────────────────────────────────────────────────────

class DailyContextBridge:
    """
    Thread-safe singleton.  Loads all Daily_Cash_Market context once at
    startup, caches it, and exposes scored sub-signals to trade_setup.py.

    All public methods degrade gracefully to (0.0, []) / {} when the DB
    is unreachable — Tradebot never crashes because of the bridge.
    """

    def __init__(self):
        self._lock        = threading.Lock()
        self._cache: dict = {}          # {fyers_sym: data_dict, "__common__": ...}
        self._loaded_at: float = 0.0
        self._available: bool  = False
        self._loading:   bool  = False  # prevents thundering-herd on stale cache
        # Load silently in background so dashboard startup is not delayed
        with self._lock:
            self._loading = True
        threading.Thread(target=self._load, daemon=True, name="dcb-init").start()

    # ── Public API ─────────────────────────────────────────────────────────────

    def score_index(self, fyers_sym: str) -> tuple[float, list]:
        """
        Core entry point for trade_setup.py.
        Returns (composite_score, signals) where score in [-4, +4].
        Degrades to (0.0, [building/unavailable msg]) when data not ready.
        """
        self._maybe_refresh()
        with self._lock:
            avail  = self._available
            data   = dict(self._cache.get(fyers_sym, {}))
            common = dict(self._cache.get("__common__", {}))

        if not avail:
            return 0.0, [("neut", "Daily context: DB unavailable (run Daily_Cash_Market)")]
        if not data:
            return 0.0, [("neut", "Daily context: no data for this symbol")]

        return self._build_score(data, common)

    def get_key_levels(self, fyers_sym: str) -> dict:
        """
        Daily-history key levels for dashboard support/resistance overlay.
        Returns empty dict on failure.
        """
        self._maybe_refresh()
        with self._lock:
            data = dict(self._cache.get(fyers_sym, {}))
        return {
            "prev_close":        data.get("prev_close"),
            "week_high":         data.get("week_high"),
            "week_low":          data.get("week_low"),
            "high_20d":          data.get("high_20d"),
            "low_20d":           data.get("low_20d"),
            "pred_range_low":    data.get("range_low"),
            "pred_range_high":   data.get("range_high"),
            "pred_target":       data.get("target_close"),
            "pred_move_pts":     data.get("expected_move_pts"),
            "max_pain_price":    data.get("max_pain_price"),
            "max_pain_dist_pct": data.get("max_pain_dist_pct"),
            "nearest_expiry":    data.get("nearest_expiry"),
        }

    def get_panel_data(self, fyers_sym: str) -> dict:
        """
        Full context dict for the dashboard 'Daily Context' card.
        All reads under a single lock acquisition for consistency.
        """
        self._maybe_refresh()
        with self._lock:
            data      = dict(self._cache.get(fyers_sym, {}))
            common    = dict(self._cache.get("__common__", {}))
            available = self._available
            loaded_at = self._loaded_at
        data["__common__"] = common
        data["available"]  = available
        data["loaded_at"]  = loaded_at
        return data

    def is_available(self) -> bool:
        with self._lock:
            return self._available

    # ── Cache management ───────────────────────────────────────────────────────

    def _sqlite_has_newer_data(self) -> bool:
        """
        Return True if the local SQLite snapshot was written AFTER the current
        in-memory cache was loaded.

        This lets the bridge detect a fresh nightly_sync run (typically 7:30 PM)
        without waiting for the 4-hour TTL to expire.  The check is a single
        lightweight SQLite row fetch — negligible cost at the 60-second callback
        frequency.
        """
        if not LOCAL_DB.exists():
            return False
        try:
            con = sqlite3.connect(str(LOCAL_DB), timeout=5.0)
            row = con.execute(
                "SELECT synced_at FROM sync_metadata WHERE status = 'ok' "
                "ORDER BY sync_date DESC LIMIT 1"
            ).fetchone()
            con.close()
            if not row or not row[0]:
                return False
            # synced_at is stored as ISO-8601 string, e.g. "2026-06-03T19:30:05"
            synced_ts = datetime.datetime.fromisoformat(str(row[0])[:19]).timestamp()
            with self._lock:
                loaded = self._loaded_at
            return synced_ts > loaded
        except Exception:
            return False

    def _maybe_refresh(self):
        with self._lock:
            age     = time.time() - self._loaded_at
            loading = self._loading

        if loading:
            return  # refresh already in flight

        # Trigger a reload when:
        #   (a) TTL expired (4 hours, keeps data fresh during long sessions), OR
        #   (b) nightly_sync wrote new data to SQLite since we last loaded
        #       — detects the 7:30 PM update within one 60-second callback cycle
        needs_refresh = (age > CACHE_TTL) or self._sqlite_has_newer_data()

        if not needs_refresh:
            return

        with self._lock:
            # Double-check under lock — another thread may have won the race
            if self._loading:
                return
            self._loading = True
        threading.Thread(target=self._load, daemon=True, name="dcb-refresh").start()

    def _load(self):
        """
        Populate cache from the best available data source.

        Preference:
          1. Local SQLite snapshot (data/tradebot_context.db) — written by
             nightly_sync.py at 7:30 PM.  Fast, no lock contention with
             Daily_Cash_Market while it writes.  Used when today's sync is present.
          2. Live DuckDB read — same queries as before, used as fallback when
             local snapshot is absent or from a prior trading day.
        """
        try:
            local_cache = self._try_load_from_local()
            if local_cache is not None:
                with self._lock:
                    self._cache     = local_cache
                    self._loaded_at = time.time()
                    self._available = True
                return

            # Fallback: live DuckDB read
            if not DB_PATH.exists():
                with self._lock:
                    self._available = False
                    self._loading   = False
                return
            try:
                import duckdb
                con = duckdb.connect(str(DB_PATH), read_only=True)
                try:
                    cache = {}
                    cache["__common__"] = self._fetch_common(con)
                    for fyers_sym, dcm_sym in _SYM.items():
                        cache[fyers_sym] = self._fetch_symbol(con, dcm_sym)
                finally:
                    con.close()

                with self._lock:
                    self._cache      = cache
                    self._loaded_at  = time.time()
                    self._available  = True

            except Exception as exc:
                print(f"[DCBridge] live DuckDB load failed: {exc}", file=sys.stderr)
                with self._lock:
                    self._available = False
        finally:
            with self._lock:
                self._loading = False

    def _try_load_from_local(self) -> "dict | None":
        """
        Read today's pre-computed snapshot from data/tradebot_context.db.

        Returns populated cache dict if today's sync exists with status='ok',
        or None if local DB is absent / stale / failed.
        """
        if not LOCAL_DB.exists():
            return None

        try:
            con = sqlite3.connect(str(LOCAL_DB), timeout=5.0)

            # Accept the most recent successful sync within the last 3 calendar days.
            # The nightly sync runs at 7:30 PM and stores sync_date = that calendar
            # date (e.g., Jun 3).  The dashboard starts the next morning (Jun 4) and
            # must still pick up Jun 3's snapshot rather than returning None.
            row = con.execute(
                "SELECT sync_date FROM sync_metadata WHERE status = 'ok' "
                "ORDER BY sync_date DESC LIMIT 1"
            ).fetchone()
            if not row:
                con.close()
                return None   # no successful sync at all → fall through to DuckDB

            sync_date = row[0]   # e.g., "2026-06-03"
            try:
                age_days = (datetime.date.today() -
                            datetime.date.fromisoformat(sync_date[:10])).days
            except Exception:
                age_days = 99
            if age_days > 3:
                con.close()
                return None   # sync is stale (>3 days) → fall through to DuckDB

            # ── Market-wide context ───────────────────────────────────────────
            mc_row = con.execute(
                "SELECT india_vix, india_vix_5d_chg, breadth_pct, heavy_breadth_pct, "
                "fii_market_5d_cr, fii_market_10d_cr, fii_market_5d_pos, "
                "nifty50_avg_deliv_pct, nifty50_high_deliv_count "
                "FROM market_context WHERE sync_date = ?", [sync_date]
            ).fetchone()

            common: dict = {}
            if mc_row:
                common["india_vix"]              = mc_row[0]
                common["india_vix_5d_chg"]       = mc_row[1]
                common["breadth_pct"]            = mc_row[2]
                common["heavy_breadth_pct"]      = mc_row[3]
                common["fii_market_5d_cr"]       = mc_row[4]
                common["fii_market_10d_cr"]      = mc_row[5]
                common["fii_market_5d_pos"]      = mc_row[6]
                common["avg_deliv_pct"]          = mc_row[7]
                common["high_deliv_count"]       = mc_row[8]
                common["latest_date"]            = datetime.date.fromisoformat(sync_date[:10])

            # ── Per-symbol context ────────────────────────────────────────────
            rows = con.execute("""
                SELECT
                    fyers_sym, dcm_sym,
                    prev_close, eod_close, week_high, week_low, high_20d, low_20d,
                    pct_5d, up_streak, pe_ratio, pb_ratio,
                    pred_date, direction, confidence, composite_score, signal_count,
                    hmm_state, memory_label,
                    range_low, range_high, target_close, expected_move_pts,
                    feat_pcr, feat_carry, feat_fii_net, feat_fii_5d,
                    feat_vix, feat_vix_5d_chg, feat_breadth,
                    feat_hurst, feat_entropy, feat_oi_score,
                    pred_acc_30d, pred_hi_conf_acc, pred_sample,
                    fii_today_cr, fii_5d_cr, fii_10d_cr, fii_5d_pos_days, fii_oi,
                    fii_fut_net, client_fut_net, fii_call_net, fii_put_net,
                    eod_call_oi, eod_put_oi, eod_pcr, eod_call_chg, eod_put_chg,
                    max_pain_price, max_pain_dist_pct, nearest_expiry,
                    top_call_strike, top_put_strike,
                    fpi_equity_today_cr, fpi_equity_5d_cr, fpi_equity_10d_cr, fpi_5d_pos_days,
                    spot_close, fii_today_cts, fii_net_change_1d
                FROM index_context
                WHERE sync_date = ?
            """, [sync_date]).fetchall()

            con.close()

            if not rows:
                return None

            cache: dict = {"__common__": common}
            for r in rows:
                (fyers_sym, dcm_sym,
                 prev_close, eod_close, week_high, week_low, high_20d, low_20d,
                 pct_5d, up_streak, pe_ratio, pb_ratio,
                 pred_date_s, direction, confidence, composite_score, signal_count,
                 hmm_state, memory_label,
                 range_low, range_high, target_close, expected_move_pts,
                 feat_pcr, feat_carry, feat_fii_net, feat_fii_5d,
                 feat_vix, feat_vix_5d_chg, feat_breadth,
                 feat_hurst, feat_entropy, feat_oi_score,
                 pred_acc_30d, pred_hi_conf_acc, pred_sample,
                 fii_today_cr, fii_5d_cr, fii_10d_cr, fii_5d_pos_days, fii_oi,
                 fii_fut_net, client_fut_net, fii_call_net, fii_put_net,
                 eod_call_oi, eod_put_oi, eod_pcr, eod_call_chg, eod_put_chg,
                 max_pain_price, max_pain_dist_pct, nearest_expiry,
                 top_call_strike, top_put_strike,
                 fpi_equity_today_cr, fpi_equity_5d_cr, fpi_equity_10d_cr, fpi_5d_pos_days,
                 spot_close, fii_today_cts, fii_net_change_1d) = r

                # pred_date is stored as ISO string in SQLite; restore to date object
                pred_date = None
                if pred_date_s:
                    try:
                        pred_date = datetime.date.fromisoformat(str(pred_date_s)[:10])
                    except Exception:
                        pred_date = pred_date_s

                cache[fyers_sym] = {
                    "symbol":          dcm_sym,
                    "prev_close":      prev_close,
                    "eod_close":       eod_close,
                    "week_high":       week_high,
                    "week_low":        week_low,
                    "high_20d":        high_20d,
                    "low_20d":         low_20d,
                    "pct_5d":          pct_5d,
                    "up_streak":       up_streak,
                    "pe_ratio":        pe_ratio,
                    "pb_ratio":        pb_ratio,
                    "pred_date":       pred_date,
                    "direction":       direction,
                    "confidence":      confidence,
                    "composite_score": float(composite_score) if composite_score is not None else 0.0,
                    "signal_count":    signal_count,
                    "hmm_state":       hmm_state,
                    "memory_label":    memory_label,
                    "range_low":       range_low,
                    "range_high":      range_high,
                    "target_close":    target_close,
                    "expected_move_pts": expected_move_pts,
                    "feat_pcr":        feat_pcr,
                    "feat_carry":      feat_carry,
                    "feat_fii_net":    feat_fii_net,
                    "feat_fii_5d":     feat_fii_5d,
                    "feat_vix":        feat_vix,
                    "feat_vix_5d_chg": feat_vix_5d_chg,
                    "feat_breadth":    feat_breadth,
                    "feat_hurst":      feat_hurst,
                    "feat_entropy":    feat_entropy,
                    "feat_oi_score":   feat_oi_score,
                    "pred_acc_30d":    pred_acc_30d,
                    "pred_hi_conf_acc": pred_hi_conf_acc,
                    "pred_sample":     pred_sample,
                    "fii_today_cr":    fii_today_cr,
                    "fii_5d_cr":       fii_5d_cr,
                    "fii_10d_cr":      fii_10d_cr,
                    "fii_5d_pos_days": fii_5d_pos_days,
                    "fii_oi":          fii_oi,
                    # FAO participant net OI positions (from fao_participant table)
                    "fii_fut_net":     fii_fut_net,
                    "client_fut_net":  client_fut_net,
                    "fii_call_net":    fii_call_net,
                    "fii_put_net":     fii_put_net,
                    "eod_call_oi":     eod_call_oi,
                    "eod_put_oi":      eod_put_oi,
                    "eod_pcr":         eod_pcr,
                    "eod_call_chg":    eod_call_chg,
                    "eod_put_chg":     eod_put_chg,
                    "max_pain_price":    max_pain_price,
                    "max_pain_dist_pct": max_pain_dist_pct,
                    "nearest_expiry":    nearest_expiry,
                    # Key OI levels — exact DCM replica fields
                    "top_call_strike":   top_call_strike,   # Resistance (R)
                    "top_put_strike":    top_put_strike,    # Support (S)
                    # Exact replica fields
                    "spot_close":        spot_close,         # hero price (from prediction_log)
                    "fii_today_cts":     fii_today_cts,     # daily net contracts ADD/COV
                    "fii_net_change_1d": fii_net_change_1d, # FAO delta: COV>0, ADD<0
                    # B7 FPI fields
                    "fpi_equity_today_cr": fpi_equity_today_cr,
                    "fpi_equity_5d_cr":    fpi_equity_5d_cr,
                    "fpi_equity_10d_cr":   fpi_equity_10d_cr,
                    "fpi_5d_pos_days":     fpi_5d_pos_days,
                }

            return cache

        except Exception as exc:
            print(f"[DCBridge] local snapshot read failed: {exc}", file=sys.stderr)
            return None

    # ── Data fetchers ──────────────────────────────────────────────────────────

    def _fetch_common(self, con) -> dict:
        """Market-wide metrics shared across all indices."""
        r: dict = {}

        # Latest trade date in DB
        try:
            row = con.execute(
                "SELECT MAX(trade_date) FROM index_data WHERE index_name = 'Nifty 50'"
            ).fetchone()
            r["latest_date"] = row[0] if row else None
        except Exception as _e:
            warn_once(_e)

        # India VIX — level + 5-day change
        try:
            df = con.execute("""
                SELECT close_val
                FROM index_data
                WHERE index_name = 'India VIX'
                ORDER BY trade_date DESC
                LIMIT 10
            """).df()
            if not df.empty:
                r["india_vix"]       = float(df["close_val"].iloc[0])
                r["india_vix_5d_chg"] = round(
                    float(df["close_val"].iloc[0]) -
                    float(df["close_val"].iloc[min(5, len(df) - 1)]), 2
                )
        except Exception as _e:
            warn_once(_e)

        # Market breadth: % of stocks with close > prev_close, grouped by sector
        # Uses latest daily_data + sector_master
        try:
            df = con.execute("""
                SELECT
                    sm.sector,
                    SUM(CASE WHEN dd.close_price > dd.prev_close THEN 1 ELSE 0 END) AS adv,
                    COUNT(*)                                                            AS total
                FROM daily_data dd
                JOIN sector_master sm ON dd.symbol = sm.symbol
                WHERE dd.trade_date = (SELECT MAX(trade_date) FROM daily_data)
                  AND dd.series     = 'EQ'
                  AND dd.turnover_lacs >= 5
                GROUP BY sm.sector
            """).df()
            if not df.empty:
                adv_rates = df["adv"] / df["total"] * 100
                r["breadth_pct"]       = round(float(adv_rates.mean()), 1)
                heavy = df[df["sector"].isin(_HEAVY_SECTORS)]
                if not heavy.empty:
                    r["heavy_breadth_pct"] = round(
                        float((heavy["adv"] / heavy["total"] * 100).mean()), 1
                    )
        except Exception as _e:
            warn_once(_e)

        # FII aggregate flow (all index futures combined, 5D + 10D)
        try:
            df = con.execute("""
                SELECT
                    trade_date,
                    SUM(buy_value_cr - sell_value_cr)     AS net_cr,
                    SUM(buy_contracts - sell_contracts)   AS net_cts
                FROM fii_derivatives_stats
                WHERE category IN (
                    'NIFTY FUTURES','BANKNIFTY FUTURES',
                    'FINNIFTY FUTURES','MIDCPNIFTY FUTURES'
                )
                GROUP BY trade_date
                ORDER BY trade_date DESC
                LIMIT 10
            """).df()
            if not df.empty:
                r["fii_market_5d_cr"]  = round(float(df["net_cr"].head(5).sum()), 1)
                r["fii_market_10d_cr"] = round(float(df["net_cr"].head(10).sum()), 1)
                r["fii_market_5d_pos"] = int((df["net_cr"].head(5) > 0).sum())
        except Exception as _e:
            warn_once(_e)

        # Large-cap delivery quality (institutional conviction indicator)
        try:
            df = con.execute("""
                SELECT dd.deliv_per
                FROM daily_data dd
                JOIN sector_master sm ON dd.symbol = sm.symbol
                WHERE dd.trade_date         = (SELECT MAX(trade_date) FROM daily_data)
                  AND dd.series             = 'EQ'
                  AND dd.deliv_per          IS NOT NULL
                  AND dd.deliv_per          > 0
                  AND dd.turnover_lacs      >= 100
                  AND LOWER(sm.market_cap_category) IN ('large cap','largecap','large-cap')
            """).df()
            if not df.empty:
                r["avg_deliv_pct"]    = round(float(df["deliv_per"].mean()), 1)
                r["high_deliv_count"] = int((df["deliv_per"] > 60).sum())
        except Exception as _e:
            warn_once(_e)

        return r

    def _fetch_symbol(self, con, dcm_sym: str) -> dict:
        """Per-index context from prediction_log, FII stats, index_data, fno_bhavcopy."""
        r: dict = {"symbol": dcm_sym}

        # ── prediction_log: latest prediction + 12-dim fingerprint + spot_close ─
        try:
            row = con.execute("""
                SELECT
                    trade_date, direction_pred, confidence_pred,
                    composite_score, signal_count,
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

            if row:
                (pred_date, direction, confidence, composite, sig_count,
                 pcr, carry, fii_net, fii_5d,
                 vix_f, vix_chg, breadth_f,
                 hurst, entropy, oi_score,
                 hmm, mem_label,
                 rng_lo, rng_hi, target, move_pts,
                 spot_cl) = row
                r.update({
                    "pred_date":        pred_date,
                    "direction":        direction,
                    "confidence":       confidence,
                    "composite_score":  float(composite) if composite else 0.0,
                    "signal_count":     sig_count,
                    "feat_pcr":         pcr,
                    "feat_carry":       carry,
                    "feat_fii_net":     fii_net,
                    "feat_fii_5d":      fii_5d,
                    "feat_vix":         vix_f,
                    "feat_vix_5d_chg":  vix_chg,
                    "feat_breadth":     breadth_f,
                    "feat_hurst":       hurst,
                    "feat_entropy":     entropy,
                    "feat_oi_score":    oi_score,
                    "hmm_state":        hmm,
                    "memory_label":     mem_label,
                    "range_low":        float(rng_lo)   if rng_lo   else None,
                    "range_high":       float(rng_hi)   if rng_hi   else None,
                    "target_close":     float(target)   if target   else None,
                    "expected_move_pts": float(move_pts) if move_pts else None,
                    "spot_close":       float(spot_cl)  if spot_cl  else None,
                })
        except Exception as _e:
            warn_once(_e)

        # ── prediction accuracy (last 30 filled predictions) ─────────────────
        # IMPORTANT: LIMIT on an aggregate query is a no-op — it limits the
        # single-row result, not the source rows.  Use a subquery instead.
        try:
            row = con.execute("""
                SELECT
                    COUNT(*)                                                  AS n,
                    SUM(CASE WHEN was_correct THEN 1 ELSE 0 END)            AS correct,
                    AVG(CASE WHEN confidence_pred = 'HIGH' AND was_correct THEN 1.0
                              WHEN confidence_pred = 'HIGH'                 THEN 0.0
                         END)                                                AS hi_acc
                FROM (
                    SELECT was_correct, confidence_pred
                    FROM prediction_log
                    WHERE fno_symbol    = ?
                      AND outcome_filled = TRUE
                    ORDER BY trade_date DESC
                    LIMIT 30
                ) recent
            """, [dcm_sym]).fetchone()
            if row and row[0]:
                r["pred_acc_30d"]      = round((row[1] or 0) / row[0] * 100, 1)
                r["pred_hi_conf_acc"]  = (round(row[2] * 100, 1) if row[2] is not None else None)
                r["pred_sample"]       = row[0]
        except Exception as _e:
            warn_once(_e)

        # ── index_data: daily OHLCV → key levels ─────────────────────────────
        idx_name = _IDX_NAME.get(dcm_sym)
        if idx_name:
            try:
                df = con.execute("""
                    SELECT trade_date, high_val, low_val, close_val,
                           prev_close, pct_chg, pe_ratio, pb_ratio
                    FROM index_data
                    WHERE index_name = ?
                    ORDER BY trade_date DESC
                    LIMIT 22
                """, [idx_name]).df()
                if not df.empty:
                    # eod_close  = last session's closing price (the "hero" price displayed)
                    # prev_close = the session before that (used for day-change % computation)
                    r["eod_close"]  = float(df["close_val"].iloc[0])
                    raw_prev = df["prev_close"].iloc[0]
                    r["prev_close"] = (float(raw_prev) if raw_prev is not None
                                       else float(df["close_val"].iloc[1])
                                            if len(df) > 1
                                       else r["eod_close"])
                    r["high_20d"]   = float(df["high_val"].max())
                    r["low_20d"]    = float(df["low_val"].min())
                    r["week_high"]  = float(df["high_val"].head(5).max())
                    r["week_low"]   = float(df["low_val"].head(5).min())
                    if len(df) >= 5:
                        r["pct_5d"] = round(float(df["pct_chg"].head(5).sum()), 2)
                    pe = df["pe_ratio"].iloc[0]
                    pb = df["pb_ratio"].iloc[0]
                    r["pe_ratio"] = float(pe) if pe is not None else None
                    r["pb_ratio"] = float(pb) if pb is not None else None
                    # Consecutive up days (momentum streak)
                    streak = 0
                    for chg in df["pct_chg"]:
                        if chg and chg > 0:
                            streak += 1
                        else:
                            break
                    r["up_streak"] = streak
            except Exception as _e:
                warn_once(_e)

        # ── fno_bhavcopy (max pain) — inline computation for DuckDB live path ──
        # nightly_sync pre-computes this; in the DuckDB live fallback we do it here.
        try:
            exp_row = con.execute("""
                SELECT MIN(expiry_date)
                FROM fno_bhavcopy
                WHERE symbol = ? AND instrument = 'OPTIDX'
                  AND trade_date = (SELECT MAX(trade_date) FROM fno_bhavcopy WHERE symbol = ?)
                  AND expiry_date >= (SELECT MAX(trade_date) FROM fno_bhavcopy WHERE symbol = ?)
            """, [dcm_sym, dcm_sym, dcm_sym]).fetchone()

            if exp_row and exp_row[0]:
                expiry = exp_row[0]
                mp_df  = con.execute("""
                    SELECT strike_price, option_type, open_interest
                    FROM fno_bhavcopy
                    WHERE symbol = ? AND instrument = 'OPTIDX'
                      AND trade_date = (SELECT MAX(trade_date) FROM fno_bhavcopy WHERE symbol = ?)
                      AND expiry_date = ? AND option_type IN ('CE','PE')
                      AND open_interest > 0
                """, [dcm_sym, dcm_sym, expiry]).df()

                if not mp_df.empty:
                    ce_rows_mp = mp_df[mp_df["option_type"] == "CE"].copy()
                    pe_rows_mp = mp_df[mp_df["option_type"] == "PE"].copy()
                    ce_oi = dict(zip(ce_rows_mp["strike_price"], ce_rows_mp["open_interest"]))
                    pe_oi = dict(zip(pe_rows_mp["strike_price"], pe_rows_mp["open_interest"]))
                    strikes = sorted(set(ce_oi) | set(pe_oi))
                    if strikes:
                        min_pain = float("inf")
                        mp_price = strikes[len(strikes) // 2]
                        for price in strikes:
                            pain = (
                                sum((price - k) * oi for k, oi in ce_oi.items() if k < price) +
                                sum((k - price) * oi for k, oi in pe_oi.items() if k > price)
                            )
                            if pain < min_pain:
                                min_pain, mp_price = pain, price
                        r["max_pain_price"] = float(mp_price)
                        r["nearest_expiry"] = str(expiry)
                        spot = r.get("eod_close") or r.get("prev_close") or 0.0
                        if spot > 0:
                            r["max_pain_dist_pct"] = round((mp_price - spot) / spot * 100, 3)
                        # Top OI strikes with DCM band filtering
                        band = float(spot) * 0.05 if spot else 0.0
                        if not ce_rows_mp.empty and spot:
                            nc = ce_rows_mp[(ce_rows_mp["strike_price"] >= spot - band) &
                                            (ce_rows_mp["strike_price"] <= spot + band * 3)]
                            if nc.empty: nc = ce_rows_mp
                            r["top_call_strike"] = float(
                                nc.loc[nc["open_interest"].idxmax(), "strike_price"])
                        if not pe_rows_mp.empty and spot:
                            np_ = pe_rows_mp[(pe_rows_mp["strike_price"] >= spot - band * 3) &
                                             (pe_rows_mp["strike_price"] <= spot + band)]
                            if np_.empty: np_ = pe_rows_mp
                            r["top_put_strike"] = float(
                                np_.loc[np_["open_interest"].idxmax(), "strike_price"])
        except Exception as _e:
            warn_once(_e)

        # ── fii_derivatives_stats: per-symbol FII futures flow ────────────────
        fii_cat = _FII_CAT.get(dcm_sym)
        if fii_cat:
            try:
                df = con.execute("""
                    SELECT trade_date,
                           buy_value_cr - sell_value_cr    AS net_cr,
                           buy_contracts - sell_contracts  AS net_cts,
                           oi_contracts
                    FROM fii_derivatives_stats
                    WHERE category = ?
                    ORDER BY trade_date DESC
                    LIMIT 10
                """, [fii_cat]).df()
                if not df.empty:
                    r["fii_today_cr"]    = round(float(df["net_cr"].iloc[0]), 1)
                    r["fii_5d_cr"]       = round(float(df["net_cr"].head(5).sum()), 1)
                    r["fii_10d_cr"]      = round(float(df["net_cr"].head(10).sum()), 1)
                    r["fii_5d_pos_days"] = int((df["net_cr"].head(5) > 0).sum())
                    r["fii_oi"]          = int(df["oi_contracts"].iloc[0])
            except Exception as _e:
                warn_once(_e)

        # ── fao_participant: FII net + 1D change for ADD/COV display ─────────────
        try:
            rows = con.execute("""
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
            for row in rows:
                dt, ct = row[0], row[1]
                if dt not in by_date:
                    by_date[dt] = {}
                by_date[dt][ct] = {
                    "fut_net":  (row[2] or 0) - (row[3] or 0),
                    "call_net": (row[4] or 0) - (row[5] or 0),
                    "put_net":  (row[6] or 0) - (row[7] or 0),
                }
            dates = sorted(by_date.keys(), reverse=True)
            if dates:
                latest = by_date[dates[0]]
                fii_p = latest.get("FII", {})
                r["fii_fut_net"]  = fii_p.get("fut_net",  0)
                r["fii_call_net"] = fii_p.get("call_net", 0)
                r["fii_put_net"]  = fii_p.get("put_net",  0)
                r["client_fut_net"] = latest.get("Client", {}).get("fut_net", 0)
            if len(dates) >= 2:
                net_today = by_date[dates[0]].get("FII", {}).get("fut_net", 0)
                net_prev  = by_date[dates[1]].get("FII", {}).get("fut_net", 0)
                r["fii_net_change_1d"] = net_today - net_prev
        except Exception as _e:
            warn_once(_e)

        # ── fpi_nsdl_flows: FPI equity flows (macro B7 signal) ───────────────────
        try:
            df = con.execute("""
                SELECT net_investment_cr
                FROM fpi_nsdl_flows
                WHERE LOWER(category) = 'equity'
                ORDER BY trade_date DESC
                LIMIT 10
            """).df()
            if not df.empty:
                r["fpi_equity_today_cr"] = round(float(df["net_investment_cr"].iloc[0]), 1)
                r["fpi_equity_5d_cr"]    = round(float(df["net_investment_cr"].head(5).sum()), 1)
                r["fpi_equity_10d_cr"]   = round(float(df["net_investment_cr"].head(10).sum()), 1)
                r["fpi_5d_pos_days"]     = int((df["net_investment_cr"].head(5) > 0).sum())
        except Exception as _e:
            warn_once(_e)

        # ── fii_derivatives_stats: add fii_today_cts (daily net contracts) ───
        # (already computed in the fii_5d block above — refetch for today only)
        fii_cat = _FII_CAT.get(dcm_sym)
        if fii_cat and "fii_today_cts" not in r:
            try:
                row_cts = con.execute("""
                    SELECT buy_contracts - sell_contracts AS net_cts
                    FROM fii_derivatives_stats
                    WHERE category = ?
                    ORDER BY trade_date DESC LIMIT 1
                """, [fii_cat]).fetchone()
                if row_cts and row_cts[0] is not None:
                    r["fii_today_cts"] = int(row_cts[0])
            except Exception as _e:
                warn_once(_e)

        # ── fno_bhavcopy: EOD option chain OI + OI change (2-day delta) ─────────
        # NSE bhavcopy chg_in_oi column is often 0; compute change by diffing
        # the two most recent dates so the OI direction signal fires reliably.
        try:
            df = con.execute("""
                SELECT trade_date, option_type, SUM(open_interest) AS oi
                FROM fno_bhavcopy
                WHERE symbol     = ?
                  AND instrument  = 'OPTIDX'
                  AND trade_date IN (
                      SELECT DISTINCT trade_date FROM fno_bhavcopy
                      WHERE symbol = ? ORDER BY trade_date DESC LIMIT 2
                  )
                  AND option_type IN ('CE','PE')
                GROUP BY trade_date, option_type
                ORDER BY trade_date DESC
            """, [dcm_sym, dcm_sym]).df()
            if not df.empty:
                dates = sorted(df["trade_date"].unique(), reverse=True)
                today_df = df[df["trade_date"] == dates[0]]
                prev_df  = df[df["trade_date"] == dates[1]] if len(dates) > 1 else None
                def _oi(sub, opt): return int(sub[sub["option_type"] == opt]["oi"].sum()) if not sub[sub["option_type"] == opt].empty else 0
                c_oi  = _oi(today_df, "CE")
                p_oi  = _oi(today_df, "PE")
                c_chg = c_oi - _oi(prev_df, "CE") if prev_df is not None else 0
                p_chg = p_oi - _oi(prev_df, "PE") if prev_df is not None else 0
                r["eod_call_oi"]  = c_oi
                r["eod_put_oi"]   = p_oi
                r["eod_pcr"]      = round(p_oi / c_oi, 3) if c_oi else 0.0
                r["eod_call_chg"] = c_chg
                r["eod_put_chg"]  = p_chg
        except Exception as _e:
            warn_once(_e)

        return r

    # ── Score computation ──────────────────────────────────────────────────────

    def _build_score(self, data: dict, common: dict) -> tuple[float, list]:
        signals: list = []
        score = 0.0

        score += self._b1_prediction(data, signals)
        score += self._b2_hmm(data, signals)
        score += self._b3_fii(data, signals)
        score += self._b4_breadth(common, signals)
        score += self._b5_vix(data, common, signals)
        score += self._b6_eod_oi(data, signals)
        # B7: FPI equity flows — aggregate NSDL data; full weight for NIFTY only.
        # BANKNIFTY/FINNIFTY/MIDCAP sectors can diverge from aggregate FPI flow,
        # so apply 30% weight for non-NIFTY indices (macro hint, not conviction).
        fpi_score = self._b7_fpi(data, signals)
        if data.get("symbol") != "NIFTY":
            fpi_score = round(fpi_score * 0.3, 2)
        score += fpi_score
        score += self._b8_delivery(common, signals)

        return round(min(max(score, -4.0), 4.0), 2), signals

    # ── B1: Prediction Engine ─────────────────────────────────────────────────

    def _b1_prediction(self, data: dict, signals: list) -> float:
        direction  = data.get("direction")
        confidence = data.get("confidence")
        composite  = data.get("composite_score", 0.0) or 0.0
        pred_date  = data.get("pred_date")
        accuracy   = data.get("pred_acc_30d", 50.0) or 50.0

        if not direction or not pred_date:
            signals.append(("neut",
                "Prediction engine: no forecast found — run daily ingestion"))
            return 0.0

        # Staleness: prediction older than 3 calendar days is unreliable.
        # pred_date may be a datetime.date (DuckDB path) or already parsed (SQLite path).
        try:
            if isinstance(pred_date, str):
                pred_date = datetime.date.fromisoformat(pred_date[:10])
            elif isinstance(pred_date, datetime.datetime):
                pred_date = pred_date.date()
            age_days = (datetime.date.today() - pred_date).days
        except Exception:
            age_days = 0

        if age_days > 3:
            signals.append(("neut",
                f"Prediction engine: last data is {age_days}d old — "
                f"run daily_cash_market ingestion to refresh"))
            return 0.0

        # Normalize composite -20..+20 → base contribution ±2.0
        # Confidence weight: HIGH = full, MEDIUM = 0.85, LOW = 0.65
        conf_mult = {"HIGH": 1.0, "MEDIUM": 0.85, "LOW": 0.65}.get(confidence or "", 0.75)

        # Accuracy weight: >65% accurate = trust more; <50% = trust less
        acc_mult = max(0.5, min(accuracy / 65.0, 1.15))

        base = (composite / 20.0) * 2.0 * conf_mult * acc_mult

        # Direction sanity check
        if direction == "SIDEWAYS":
            base *= 0.3     # sideways → very weak directional contribution

        base = round(min(max(base, -2.0), 2.0), 2)

        bias      = "bull" if base > 0.1 else "bear" if base < -0.1 else "neut"
        date_str  = pred_date.strftime("%d %b") if hasattr(pred_date, "strftime") else str(pred_date)
        conf_flag = f"[{confidence}]" if confidence else ""
        acc_str   = f"30D acc {accuracy:.0f}%"

        signals.append((bias,
            f"Prediction engine ({date_str}): {direction} {conf_flag} "
            f"— score {composite:+.1f}/20 · {acc_str} · "
            f"{'24-signal' if data.get('signal_count', 0) >= 24 else str(data.get('signal_count','?')) + '-signal'} model"))

        return base

    # ── B2: HMM Regime ────────────────────────────────────────────────────────

    def _b2_hmm(self, data: dict, signals: list) -> float:
        hmm   = data.get("hmm_state")
        hurst = data.get("feat_hurst") or 0.5
        entropy = data.get("feat_entropy") or 0.5

        if not hmm:
            return 0.0

        base = {"Bull": 1.0, "Bear": -1.0, "Sideways": 0.0}.get(hmm, 0.0)

        # Hurst > 0.55: trending market — amplify regime signal
        # Hurst < 0.45: mean-reverting — dampen regime signal
        if hurst > 0.55:
            base *= 1.25
            hurst_note = f"Hurst {hurst:.2f} (trending — regime amplified)"
        elif hurst < 0.45:
            base *= 0.6
            hurst_note = f"Hurst {hurst:.2f} (mean-reverting — regime muted)"
        else:
            hurst_note = f"Hurst {hurst:.2f} (neutral)"

        # Low entropy = ordered/predictable market → more reliable regime
        if entropy < 0.35:
            base *= 1.1
            ent_note = f"entropy {entropy:.2f} (ordered, signal reliable)"
        elif entropy > 0.65:
            base *= 0.85
            ent_note = f"entropy {entropy:.2f} (chaotic, reduce conviction)"
        else:
            ent_note = f"entropy {entropy:.2f}"

        base  = round(min(max(base, -1.25), 1.25), 2)
        bias  = "bull" if base > 0 else "bear" if base < 0 else "neut"
        mem   = data.get("memory_label", "")
        mem_s = f" · memory: {mem}" if mem else ""
        signals.append((bias,
            f"HMM regime: {hmm.upper()} · {hurst_note} · {ent_note}{mem_s}"))
        return base

    # ── B3: FII Flow Trend (prior 4D + 9D, today excluded) ───────────────────
    # Layer 4 (market_data_bridge, 10% weight) already scores TODAY's FII flow.
    # B3 owns the multi-day structural trend by stripping today out, so each
    # layer measures a distinct time window with no double-count.
    #
    # CONTRARIAN CAVEAT (measure before flipping): fii_positioning_backtest found
    # FII index-futures positioning is CONTRARIAN to forward Nifty (IC −0.13/−0.16,
    # hedging-driven), i.e. the "FII buying → bullish" sign below is likely
    # backwards. It is kept as-is (bounded: ±2 here, one of 8 ctx sub-signals) until
    # a sign-flip is validated OOS — do NOT invert it untested.

    def _b3_fii(self, data: dict, signals: list) -> float:
        today_cr = data.get("fii_today_cr",    0.0) or 0.0
        raw_5d   = data.get("fii_5d_cr",       0.0) or 0.0
        raw_10d  = data.get("fii_10d_cr",      0.0) or 0.0
        raw_pos  = data.get("fii_5d_pos_days", 0)   or 0

        # Exclude today so B3 covers prior 4 trading days (5D window minus today)
        fii_4d  = raw_5d  - today_cr
        fii_9d  = raw_10d - today_cr
        # If today was a buying day reduce the positive-day count by 1
        pos_days = max(0, raw_pos - (1 if today_cr > 0 else 0))

        if not (fii_4d or fii_9d):
            return 0.0

        score = 0.0

        # Prior-4D net flow — ±₹2000Cr over 4 days = meaningful structural trend
        if fii_4d > 2000:
            score += 1.5
            signals.append(("bull",
                f"FII futures prior 4D: +{fii_4d:,.0f}Cr ({pos_days}/4 buying days) "
                f"— sustained multi-day accumulation"))
        elif fii_4d > 650:
            score += 0.75
            signals.append(("bull",
                f"FII futures prior 4D: +{fii_4d:,.0f}Cr ({pos_days}/4 buying days) "
                f"— consistent structural buying"))
        elif fii_4d < -2000:
            score -= 1.5
            signals.append(("bear",
                f"FII futures prior 4D: {fii_4d:,.0f}Cr ({pos_days}/4 buying days) "
                f"— sustained multi-day distribution"))
        elif fii_4d < -650:
            score -= 0.75
            signals.append(("bear",
                f"FII futures prior 4D: {fii_4d:,.0f}Cr ({pos_days}/4 buying days) "
                f"— consistent structural selling"))
        else:
            signals.append(("neut",
                f"FII futures prior 4D: {fii_4d:+,.0f}Cr — neutral structural stance"))

        # Prior-9D structural context
        if abs(fii_9d) > 5000:
            structural_bias = "bull" if fii_9d > 0 else "bear"
            direction_word  = "accumulation" if fii_9d > 0 else "distribution"
            score += 0.5 if fii_9d > 0 else -0.5
            signals.append((structural_bias,
                f"FII 9D structural: {fii_9d:+,.0f}Cr — "
                f"sustained {direction_word} in progress"))

        return round(min(max(score, -2.0), 2.0), 2)

    # ── B4: Market Breadth ────────────────────────────────────────────────────

    def _b4_breadth(self, common: dict, signals: list) -> float:
        breadth = common.get("breadth_pct")
        heavy   = common.get("heavy_breadth_pct")

        if breadth is None:
            return 0.0

        score = 0.0

        # Market-wide breadth
        if breadth > 72:
            score += 0.75
            signals.append(("bull",
                f"Market breadth: {breadth:.0f}% sectors advancing — "
                f"broad-based move confirmed"))
        elif breadth < 28:
            score -= 0.75
            signals.append(("bear",
                f"Market breadth: {breadth:.0f}% sectors advancing — "
                f"broad-based selling confirmed"))
        elif breadth > 58:
            score += 0.3
            signals.append(("bull", f"Market breadth: {breadth:.0f}% advancing — mild bullish tilt"))
        elif breadth < 42:
            score -= 0.3
            signals.append(("bear", f"Market breadth: {breadth:.0f}% advancing — mild bearish tilt"))
        else:
            signals.append(("neut", f"Market breadth: {breadth:.0f}% advancing — neutral"))

        # Nifty heavyweight sector confirmation
        if heavy is not None:
            if heavy > 70:
                score += 0.5
                signals.append(("bull",
                    f"NIFTY heavy sectors: {heavy:.0f}% advancing "
                    f"(Fin/IT/FMCG/Auto leading) — index move confirmed by weight"))
            elif heavy < 30:
                score -= 0.5
                signals.append(("bear",
                    f"NIFTY heavy sectors: {heavy:.0f}% advancing — "
                    f"heavyweight drag, index structurally weak"))

        return round(min(max(score, -1.25), 1.25), 2)

    # ── B5: VIX Regime ────────────────────────────────────────────────────────

    def _b5_vix(self, data: dict, common: dict, signals: list) -> float:
        # India VIX level: DB value preferred; fall back to prediction_log feature.
        # Use explicit None check — VIX of 0 is impossible so "or" would be safe,
        # but being explicit avoids confusion.
        vix_db = common.get("india_vix")
        vix    = vix_db if vix_db is not None else (data.get("feat_vix") or 0.0)

        # 5D change: DB stores absolute point change; feat_vix_5d_chg is relative %.
        # Only fall back to the feature if DB value genuinely absent (not just zero).
        vix_5d_db = common.get("india_vix_5d_chg")
        vix_5d    = vix_5d_db if vix_5d_db is not None else (data.get("feat_vix_5d_chg") or 0.0)

        if not vix:
            return 0.0

        score = 0.0

        # India VIX bands (normal range 12-18%)
        if vix < 12:
            score += 0.5
            signals.append(("bull",
                f"India VIX {vix:.1f}% — very low fear, stable environment, cheap options"))
        elif vix < 15:
            score += 0.25
            signals.append(("bull", f"India VIX {vix:.1f}% — calm market, good buying conditions"))
        elif vix > 24:
            score -= 0.75
            signals.append(("bear",
                f"India VIX {vix:.1f}% — ELEVATED fear, options expensive, sharp moves possible"))
        elif vix > 19:
            score -= 0.25
            signals.append(("bear", f"India VIX {vix:.1f}% — rising anxiety, caution warranted"))
        else:
            signals.append(("neut", f"India VIX {vix:.1f}% — normal range (12-19%)"))

        # 5-day VIX trend
        if abs(vix_5d) > 2.0:
            if vix_5d > 0:
                score -= 0.25
                signals.append(("bear",
                    f"VIX +{vix_5d:.1f}% over 5 days — fear rising, reduce position size"))
            else:
                score += 0.25
                signals.append(("bull",
                    f"VIX {vix_5d:.1f}% over 5 days — fear dissipating, risk-on environment"))

        return round(min(max(score, -1.0), 1.0), 2)

    # ── B6: EOD OI Positioning ────────────────────────────────────────────────

    def _b6_eod_oi(self, data: dict, signals: list) -> float:
        pcr     = data.get("eod_pcr",      0.0) or 0.0
        ce_chg  = data.get("eod_call_chg", 0)   or 0
        pe_chg  = data.get("eod_put_chg",  0)   or 0

        if not pcr:
            return 0.0

        score = 0.0

        # EOD PCR — end-of-day option chain (more reliable than intraday snapshot)
        if pcr > 1.35:
            score += 0.5
            signals.append(("bull",
                f"EOD PCR {pcr:.2f} — heavy put writing at close, "
                f"strong floor being built for next session"))
        elif pcr > 1.05:
            score += 0.25
            signals.append(("bull",
                f"EOD PCR {pcr:.2f} — mild put writing bias, bullish tilt for next session"))
        elif pcr < 0.65:
            score -= 0.5
            signals.append(("bear",
                f"EOD PCR {pcr:.2f} — heavy call writing at close, "
                f"strong ceiling being built"))
        elif pcr < 0.85:
            score -= 0.25
            signals.append(("bear",
                f"EOD PCR {pcr:.2f} — call writing dominant, bearish tilt"))
        else:
            signals.append(("neut", f"EOD PCR {pcr:.2f} — balanced OI positioning"))

        # EOD OI change direction
        if abs(ce_chg) > 500_000 or abs(pe_chg) > 500_000:
            if pe_chg > 0 and pe_chg > ce_chg * 1.3:
                score += 0.5
                signals.append(("bull",
                    f"EOD OI build: Put +{pe_chg/1e5:.1f}L vs Call +{ce_chg/1e5:.1f}L "
                    f"— fresh put writing, structural floor at yesterday's close"))
            elif ce_chg > 0 and ce_chg > pe_chg * 1.3:
                score -= 0.5
                signals.append(("bear",
                    f"EOD OI build: Call +{ce_chg/1e5:.1f}L vs Put +{pe_chg/1e5:.1f}L "
                    f"— fresh call writing, structural ceiling from yesterday's close"))

        return round(min(max(score, -1.0), 1.0), 2)

    # ── B7: FPI Capital Flows ─────────────────────────────────────────────────

    def _b7_fpi(self, data: dict, signals: list) -> float:
        """
        FPI (Foreign Portfolio Investors) net equity investment is the single
        largest driver of sustained NIFTY directional moves.

        FPI is broader than FII futures flow (B3): it captures ALL foreign money
        entering/exiting Indian equities — ETFs, QFIs, institutional investors.
        When FPI equity 5D and FII futures 5D align → extremely high conviction.

        Thresholds calibrated to NSE historical FPI equity flow data:
          ±₹5000Cr in 5 days = strong directional flow
          ±₹2000Cr in 5 days = moderate flow
        """
        fpi_5d   = data.get("fpi_equity_5d_cr",    0.0) or 0.0
        fpi_10d  = data.get("fpi_equity_10d_cr",   0.0) or 0.0
        fpi_pos  = data.get("fpi_5d_pos_days",     0)   or 0
        today_cr = data.get("fpi_equity_today_cr", 0.0) or 0.0

        if not (fpi_5d or fpi_10d):
            return 0.0

        score = 0.0

        # 5D flow — primary signal
        if fpi_5d > 5000:
            score += 1.5
            signals.append(("bull",
                f"FPI equity 5D: +{fpi_5d:,.0f}Cr ({fpi_pos}/5 buying days) "
                f"— heavy foreign capital inflow, structural tailwind"))
        elif fpi_5d > 2000:
            score += 0.75
            signals.append(("bull",
                f"FPI equity 5D: +{fpi_5d:,.0f}Cr ({fpi_pos}/5 buying days) "
                f"— consistent FPI buying in cash market"))
        elif fpi_5d < -5000:
            score -= 1.5
            signals.append(("bear",
                f"FPI equity 5D: {fpi_5d:,.0f}Cr ({fpi_pos}/5 buying days) "
                f"— heavy foreign capital outflow, structural headwind"))
        elif fpi_5d < -2000:
            score -= 0.75
            signals.append(("bear",
                f"FPI equity 5D: {fpi_5d:,.0f}Cr ({fpi_pos}/5 buying days) "
                f"— sustained FPI selling in cash market"))
        else:
            signals.append(("neut",
                f"FPI equity 5D: {fpi_5d:+,.0f}Cr — neutral foreign flow"))

        # 10D structural context — confirms or contradicts the 5D trend
        if abs(fpi_10d) > 12000:
            structural_bias = "bull" if fpi_10d > 0 else "bear"
            direction_word  = "accumulation" if fpi_10d > 0 else "distribution"
            score += 0.5 if fpi_10d > 0 else -0.5
            signals.append((structural_bias,
                f"FPI equity 10D structural: {fpi_10d:+,.0f}Cr — "
                f"sustained {direction_word} by foreign investors"))

        return round(min(max(score, -2.0), 2.0), 2)

    # ── B8: Large-cap Delivery Quality ────────────────────────────────────────

    def _b8_delivery(self, common: dict, signals: list) -> float:
        """
        Large-cap delivery % distinguishes institutional accumulation vs distribution.

        High delivery (>60%) in large-caps = institutions holding/accumulating → bullish.
        Low delivery (<35%) = high-velocity exits, intraday-only activity → bearish.

        Threshold calibration (NSE large-cap typical):
          avg_deliv_pct > 60  = strong institutional conviction
          avg_deliv_pct 50-60 = mild accumulation bias
          avg_deliv_pct 35-50 = neutral
          avg_deliv_pct < 35  = distribution phase
          high_deliv_count    = # of large-cap stocks with >60% delivery (breadth)
        """
        avg   = common.get("avg_deliv_pct")
        count = common.get("high_deliv_count")

        if avg is None:
            return 0.0

        score = 0.0

        if avg > 60:
            score += 0.75
            signals.append(("bull",
                f"Large-cap delivery {avg:.0f}% avg ({count} stocks >60%) "
                f"— institutional accumulation, conviction buying"))
        elif avg > 50:
            score += 0.3
            signals.append(("bull",
                f"Large-cap delivery {avg:.0f}% avg — moderate institutional buying interest"))
        elif avg < 35:
            score -= 0.75
            signals.append(("bear",
                f"Large-cap delivery {avg:.0f}% avg ({count} stocks >60%) "
                f"— distribution phase, institutions exiting intraday"))
        elif avg < 45:
            score -= 0.3
            signals.append(("bear",
                f"Large-cap delivery {avg:.0f}% avg — low institutional conviction, churn"))
        else:
            signals.append(("neut",
                f"Large-cap delivery {avg:.0f}% avg — neutral institutional activity"))

        return round(min(max(score, -0.75), 0.75), 2)


# ── Module singleton ──────────────────────────────────────────────────────────

_bridge: Optional[DailyContextBridge] = None
_bridge_lock = threading.Lock()


def get_bridge() -> DailyContextBridge:
    """Return the module-level singleton bridge, creating it on first call."""
    global _bridge
    with _bridge_lock:
        if _bridge is None:
            _bridge = DailyContextBridge()
    return _bridge
