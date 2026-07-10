# -*- coding: utf-8 -*-
"""
dcm_prediction.py — Direct reader for Daily_Cash_Market prediction data.

Reads from the SAME market_data.duckdb that Daily_Cash_Market uses.
Zero intermediary — no nightly_sync, no SQLite.
When DCM ingests new EOD data, the next cache refresh picks it up.

Cache TTL: 30 minutes.  Thread-safe singleton.

Fields returned per symbol match the EXACT sources used by DCM's
IndexPrediction engine:
    spot_close       prediction_log.spot_close
    pcr              fno_bhavcopy  put_oi / call_oi  at near expiry
    carry_pct_ann    prediction_log.feat_carry  (raw annualised %)
    vix              prediction_log.feat_vix    (raw level)
    vix_5d_pct       prediction_log.feat_vix_5d_chg  (raw 5D % change)
    fii_fut_net      fao_participant  long − short  (net contracts)
    fii_net_change   fao_today − fao_prev  (COV > 0, ADD < 0)
    top_call_strike  fno_bhavcopy  max CE OI in [spot−5%, spot+15%]
    top_put_strike   fno_bhavcopy  max PE OI in [spot−15%, spot+5%]
    max_pain_price   strike minimising total ITM intrinsic value
    hurst/entropy    prediction_log.feat_hurst / feat_entropy
    hmm_state        prediction_log.hmm_state
    range_low/high   prediction_log.range_low / range_high
    target_close     prediction_log.target_close
    expected_move_pts prediction_log.expected_move_pts
"""

import datetime
import threading
import time
from pathlib import Path
from typing import Optional

from core.obs import warn_once   # observe silently-swallowed field-computation failures

# ── Config ────────────────────────────────────────────────────────────────────
DCM_DB    = Path(r"D:\Python Projects\Daily_Cash_Market\data\market_data.duckdb")
CACHE_TTL = 30 * 60   # 30 minutes — short enough to pick up post-6:30 PM ingestion

_SYMBOLS = {
    "NSE:NIFTY50-INDEX":    "NIFTY",
    "NSE:NIFTYBANK-INDEX":  "BANKNIFTY",
    "NSE:FINNIFTY-INDEX":   "FINNIFTY",
    "NSE:MIDCPNIFTY-INDEX": "MIDCPNIFTY",
}
_IDX_NAME = {
    "NIFTY":      "Nifty 50",
    "BANKNIFTY":  "Nifty Bank",
    "FINNIFTY":   "Nifty Financial Services",
    "MIDCPNIFTY": "Nifty Midcap Select",
}


# ── Cache entry — one dict per Fyers symbol ───────────────────────────────────
# Keys returned:
#   pred_date, direction, confidence, composite_score
#   spot_close, prev_close, day_change_pct
#   feat_pcr, eod_pcr, feat_carry
#   feat_vix, feat_vix_5d_chg
#   feat_hurst, feat_entropy, hmm_state
#   range_low, range_high, target_close, expected_move_pts
#   nearest_expiry, dte
#   max_pain_price, top_call_strike, top_put_strike
#   fii_fut_net, fii_net_change_1d
#   pred_acc_30d
#   india_vix (shared)

class _DCMPredictionReader:
    def __init__(self):
        self._lock      = threading.Lock()
        self._cache:   dict  = {}
        self._loaded_at: float = 0.0
        self._available: bool  = False
        self._loading:   bool  = False
        with self._lock:
            self._loading = True
        threading.Thread(target=self._load, daemon=True, name="dcm-pred-init").start()

    # ── Public API ─────────────────────────────────────────────────────────────

    def get(self, fyers_sym: str) -> dict:
        """Return prediction dict for one index. Empty dict if unavailable."""
        self._maybe_refresh()
        with self._lock:
            return dict(self._cache.get(fyers_sym, {}))

    def get_all(self) -> dict:
        """Return {fyers_sym: data_dict} for all 4 indices."""
        self._maybe_refresh()
        with self._lock:
            return {k: dict(v) for k, v in self._cache.items()}

    def is_available(self) -> bool:
        with self._lock:
            return self._available

    def loaded_at(self) -> float:
        with self._lock:
            return self._loaded_at

    # ── Cache management ───────────────────────────────────────────────────────

    def _maybe_refresh(self):
        with self._lock:
            age     = time.time() - self._loaded_at
            loading = self._loading
        if loading:
            return
        if age <= CACHE_TTL:
            return
        with self._lock:
            if self._loading:
                return
            self._loading = True
        threading.Thread(target=self._load, daemon=True, name="dcm-pred-refresh").start()

    def _load(self):
        try:
            if not DCM_DB.exists():
                with self._lock:
                    self._available = False
                return

            import duckdb
            con = duckdb.connect(str(DCM_DB), read_only=True)
            try:
                cache = self._read_all(con)
                with self._lock:
                    self._cache     = cache
                    self._loaded_at = time.time()
                    self._available = True
            finally:
                con.close()

        except Exception as exc:
            import sys
            print(f"[DCMPred] load failed: {exc}", file=sys.stderr)
            with self._lock:
                self._available = False
        finally:
            with self._lock:
                self._loading = False

    # ── All queries in ONE connection ──────────────────────────────────────────

    def _read_all(self, con) -> dict:
        cache: dict = {}

        # ── India VIX (shared across all indices) ─────────────────────────────
        india_vix = None
        try:
            rows = con.execute(
                "SELECT close_val FROM index_data "
                "WHERE index_name = 'India VIX' ORDER BY trade_date DESC LIMIT 1"
            ).fetchall()
            if rows:
                india_vix = float(rows[0][0])
        except Exception as _e:
            warn_once(_e)

        # ── FAO participant: FII net futures + 1D change ───────────────────────
        # fii_net_change_1d > 0 → COV (covering), < 0 → ADD (adding to short)
        fii_net_latest   = None
        fii_net_change   = 0
        try:
            rows = con.execute("""
                SELECT trade_date, fut_idx_long - fut_idx_short AS net
                FROM fao_participant
                WHERE data_type  = 'OI'
                  AND client_type = 'FII'
                  AND trade_date IN (
                      SELECT DISTINCT trade_date FROM fao_participant
                      WHERE data_type = 'OI' ORDER BY trade_date DESC LIMIT 2
                  )
                ORDER BY trade_date DESC
            """).fetchall()
            if rows:
                fii_net_latest = int(rows[0][1])
                if len(rows) >= 2:
                    fii_net_change = int(rows[0][1]) - int(rows[1][1])
        except Exception as _e:
            warn_once(_e)

        # ── Per-symbol reads ───────────────────────────────────────────────────
        for fyers_sym, dcm_sym in _SYMBOLS.items():
            d: dict = {
                "symbol":           dcm_sym,
                "india_vix":        india_vix,
                "fii_fut_net":      fii_net_latest,
                "fii_net_change_1d": fii_net_change,
            }
            self._read_prediction_log(con, dcm_sym, d)
            self._read_prev_close(con, dcm_sym, d)
            self._read_option_chain(con, dcm_sym, d)
            self._read_accuracy(con, dcm_sym, d)
            # Pre-compute breakout scenarios so _rp_card() can render them inline
            d["breakout"] = compute_breakout_scenarios(d)
            cache[fyers_sym] = d

        return cache

    def _read_prediction_log(self, con, dcm_sym: str, d: dict) -> None:
        """Latest prediction row — hero price, direction, all feature fields."""
        try:
            row = con.execute("""
                SELECT
                    trade_date, direction_pred, confidence_pred,
                    composite_score,
                    feat_pcr, feat_carry, feat_vix, feat_vix_5d_chg,
                    feat_hurst, feat_entropy,
                    hmm_state, memory_label,
                    range_low, range_high, target_close, expected_move_pts,
                    spot_close
                FROM prediction_log
                WHERE fno_symbol = ?
                ORDER BY trade_date DESC LIMIT 1
            """, [dcm_sym]).fetchone()
            if not row:
                return
            (pred_date, direction, confidence,
             composite,
             feat_pcr, feat_carry, feat_vix, feat_vix_5d_chg,
             feat_hurst, feat_entropy,
             hmm_state, mem_label,
             range_lo, range_hi, target, exp_pts,
             spot_cl) = row

            d.update({
                "pred_date":       pred_date,
                "direction":       direction,
                "confidence":      confidence,
                "composite_score": float(composite) if composite is not None else 0.0,
                "feat_pcr":        float(feat_pcr)  if feat_pcr  is not None else None,
                "feat_carry":      float(feat_carry) if feat_carry is not None else None,
                "feat_vix":        float(feat_vix)  if feat_vix  is not None else None,
                "feat_vix_5d_chg": float(feat_vix_5d_chg) if feat_vix_5d_chg is not None else 0.0,
                "feat_hurst":      float(feat_hurst) if feat_hurst is not None else None,
                "feat_entropy":    float(feat_entropy) if feat_entropy is not None else None,
                "hmm_state":       hmm_state,
                "memory_label":    mem_label,
                "range_low":       float(range_lo) if range_lo is not None else None,
                "range_high":      float(range_hi) if range_hi is not None else None,
                "target_close":    float(target)   if target    is not None else None,
                "expected_move_pts": float(exp_pts) if exp_pts  is not None else None,
                "spot_close":      float(spot_cl)  if spot_cl   is not None else None,
            })
        except Exception as _e:
            warn_once(_e)

    def _read_prev_close(self, con, dcm_sym: str, d: dict) -> None:
        """Previous session's close — needed for day-change %."""
        idx_name = _IDX_NAME.get(dcm_sym)
        if not idx_name:
            return
        try:
            row = con.execute("""
                SELECT close_val, prev_close
                FROM index_data
                WHERE index_name = ?
                ORDER BY trade_date DESC LIMIT 1
            """, [idx_name]).fetchone()
            if row:
                d["prev_close"] = float(row[1]) if row[1] is not None else None
        except Exception as _e:
            warn_once(_e)

    def _read_option_chain(self, con, dcm_sym: str, d: dict) -> None:
        """Near-expiry option chain: PCR, DTE, max pain, top OI strikes."""
        try:
            # Nearest unexpired expiry
            exp_row = con.execute("""
                SELECT MIN(expiry_date)
                FROM fno_bhavcopy
                WHERE symbol = ? AND instrument = 'OPTIDX'
                  AND trade_date = (SELECT MAX(trade_date) FROM fno_bhavcopy WHERE symbol = ?)
                  AND expiry_date >= (SELECT MAX(trade_date) FROM fno_bhavcopy WHERE symbol = ?)
            """, [dcm_sym, dcm_sym, dcm_sym]).fetchone()

            if not exp_row or not exp_row[0]:
                return

            expiry = exp_row[0]
            d["nearest_expiry"] = str(expiry)

            # DTE = (near_expiry − pred_date).days
            pred_date = d.get("pred_date")
            if pred_date:
                try:
                    exp_dt  = (expiry if isinstance(expiry, datetime.date)
                               else datetime.date.fromisoformat(str(expiry)[:10]))
                    prd_dt  = (pred_date if isinstance(pred_date, datetime.date)
                               else datetime.date.fromisoformat(str(pred_date)[:10]))
                    d["dte"] = max(0, (exp_dt - prd_dt).days)
                except Exception as _e:
                    warn_once(_e)

            # All strikes for near expiry
            mp_df = con.execute("""
                SELECT strike_price, option_type, open_interest
                FROM fno_bhavcopy
                WHERE symbol = ? AND instrument = 'OPTIDX'
                  AND trade_date = (SELECT MAX(trade_date) FROM fno_bhavcopy WHERE symbol = ?)
                  AND expiry_date = ? AND option_type IN ('CE','PE')
                  AND open_interest > 0
            """, [dcm_sym, dcm_sym, expiry]).df()

            if mp_df.empty:
                return

            ce_rows = mp_df[mp_df["option_type"] == "CE"].copy()
            pe_rows = mp_df[mp_df["option_type"] == "PE"].copy()

            # EOD PCR = put_oi / call_oi
            total_ce = int(ce_rows["open_interest"].sum()) if not ce_rows.empty else 0
            total_pe = int(pe_rows["open_interest"].sum()) if not pe_rows.empty else 0
            d["eod_pcr"] = round(total_pe / total_ce, 2) if total_ce > 0 else None

            ce_oi = dict(zip(ce_rows["strike_price"], ce_rows["open_interest"]))
            pe_oi = dict(zip(pe_rows["strike_price"], pe_rows["open_interest"]))
            strikes = sorted(set(ce_oi) | set(pe_oi))
            if not strikes:
                return

            # Max pain
            min_pain, mp = float("inf"), strikes[len(strikes) // 2]
            for p in strikes:
                pain = (sum((p - k) * oi for k, oi in ce_oi.items() if k < p) +
                        sum((k - p) * oi for k, oi in pe_oi.items() if k > p))
                if pain < min_pain:
                    min_pain, mp = pain, p
            d["max_pain_price"] = float(mp)

            # Top OI strikes with DCM band filtering (spot ± 5% band)
            spot = d.get("spot_close") or 0.0
            band = spot * 0.05

            if not ce_rows.empty and spot:
                nc = ce_rows[(ce_rows["strike_price"] >= spot - band) &
                             (ce_rows["strike_price"] <= spot + band * 3)]
                if nc.empty:
                    nc = ce_rows
                d["top_call_strike"] = float(
                    nc.loc[nc["open_interest"].idxmax(), "strike_price"])

            if not pe_rows.empty and spot:
                np_ = pe_rows[(pe_rows["strike_price"] >= spot - band * 3) &
                              (pe_rows["strike_price"] <= spot + band)]
                if np_.empty:
                    np_ = pe_rows
                d["top_put_strike"] = float(
                    np_.loc[np_["open_interest"].idxmax(), "strike_price"])

        except Exception as _e:
            warn_once(_e)

    def _read_accuracy(self, con, dcm_sym: str, d: dict) -> None:
        """30-day rolling prediction accuracy."""
        try:
            row = con.execute("""
                SELECT COUNT(*),
                       SUM(CASE WHEN was_correct THEN 1 ELSE 0 END)
                FROM (
                    SELECT was_correct
                    FROM prediction_log
                    WHERE fno_symbol = ? AND outcome_filled = TRUE
                    ORDER BY trade_date DESC LIMIT 30
                ) s
            """, [dcm_sym]).fetchone()
            if row and row[0]:
                d["pred_acc_30d"] = round((row[1] or 0) / row[0] * 100, 1)
        except Exception as _e:
            warn_once(_e)


def compute_breakout_scenarios(d: dict) -> dict:
    """
    Consistent three-tier breakout scenario engine.

    Design principles (same methodology for BOTH directions — no asymmetry):
      1. Statistical anchor  — 2.4σ from spot in both directions.
      2. DTE gravity         — max pain acts as a magnet; strength scales with
                               proximity to expiry (30 % at 5 DTE → 75 % at 1 DTE).
      3. Upside: gravity drag reduces the 2.4σ target; FII massive-short
                 overrides gravity (forced covering accelerates the move).
      4. Call-wall cap       — if a call wall sits inside the extension zone
                               and FII is NOT massively short, cap at wall + 20%σ.
      5. Downside: three tiers so the trader sees the full risk picture:
           Tier 1 — immediate put wall (if within 200 pts of trigger)
           Tier 2 — gravity-corrected 2.4σ (dominant scenario)
           Tier 3 — full statistical 2.4σ if max-pain gravity fails entirely
      6. Break buffer        — adaptive: 25% of σ, clamped 40–75 pts.

    All inputs come from prediction_log + fno_bhavcopy (already in cache dict).
    Returns empty dict when required fields are absent.
    """
    spot   = d.get("spot_close")        or 0.0
    sigma  = d.get("expected_move_pts") or 0.0
    mp     = d.get("max_pain_price")    or 0.0
    c_wall = d.get("top_call_strike")   or 0.0
    p_wall = d.get("top_put_strike")    or 0.0
    fii    = d.get("fii_fut_net")       or 0
    dte    = d.get("dte")               or 5
    r_lo   = d.get("range_low")         or 0.0
    r_hi   = d.get("range_high")        or 0.0

    if not (spot and sigma and r_lo and r_hi):
        return {}

    # ── Break buffer ──────────────────────────────────────────────────────────
    buf = float(max(40, min(75, round(sigma * 0.25 / 25) * 25)))

    u_trigger = r_hi + buf
    d_trigger = r_lo - buf

    # ── Statistical 2.4σ targets (identical formula both directions) ──────────
    u_stat = round(spot + 2.4 * sigma)
    d_stat = round(spot - 2.4 * sigma)

    # ── DTE gravity factor ────────────────────────────────────────────────────
    # Expiry pinning intensifies in final days — option writers defend max pain
    # more aggressively as gamma risk grows.
    _g = {1: 0.75, 2: 0.60, 3: 0.50, 4: 0.40, 5: 0.30}
    gravity = _g.get(int(dte), 0.30 if dte > 5 else 0.75)

    fii_squeeze = fii < -100_000   # FII massively short → short squeeze possible

    # ── UPSIDE ────────────────────────────────────────────────────────────────
    # Max pain BELOW the upside trigger → drags price back down (gravity drag).
    u_gravity_drag = max(0.0, (u_trigger - mp) * gravity) if mp and mp < u_trigger else 0.0

    u_corrected = u_stat
    if not fii_squeeze:
        u_corrected = round(u_stat - u_gravity_drag)
        # Call wall inside extension zone caps the move (unless squeeze overrides)
        if c_wall and u_trigger < c_wall < u_stat:
            u_corrected = min(u_corrected, round(c_wall + sigma * 0.20))
    # FII squeeze: forced covering overrides all gravity — full 2.4σ extension
    u_corrected = max(u_corrected, round(u_trigger + 10))  # always at least 10 pts

    # ── DOWNSIDE ─────────────────────────────────────────────────────────────
    # Max pain ABOVE the downside trigger → pulls price back up (gravity lift).
    d_gravity_pull = max(0.0, (mp - d_trigger) * gravity) if mp and mp > d_trigger else 0.0

    # Tier 2: gravity-corrected target (dominant scenario — max pain is respected)
    d_tier2 = round(d_stat + d_gravity_pull)
    d_tier2  = min(d_tier2, round(d_trigger - 10))  # always at least 10 pts below trigger

    # Tier 3: pure statistical (gravity fully fails — max pain cannot hold)
    d_tier3 = d_stat

    # Tier 1: immediate put wall floor (if a significant put concentration sits
    # within 200 pts below the trigger, it is the first line of defense)
    d_tier1: "int | None" = None
    if p_wall and 0 < (d_trigger - p_wall) <= 200:
        d_tier1 = round(p_wall)

    return {
        # Inputs reflected for display
        "sigma":           round(sigma),
        "buf":             int(buf),
        "gravity":         gravity,
        "dte":             dte,
        "mp":              round(mp) if mp else None,
        "fii_net":         fii,
        "squeeze":         fii_squeeze,
        # Upside
        "u_trigger":       round(u_trigger),
        "u_corrected":     round(u_corrected),    # gravity-corrected target
        "u_stat":          u_stat,                 # full 2.4σ (squeeze scenario)
        "u_ext_pts":       round(u_corrected) - round(u_trigger),
        "u_drag_pts":      round(u_gravity_drag),
        # Downside — three tiers
        "d_trigger":       round(d_trigger),
        "d_tier1":         d_tier1,                # put wall immediate floor (may be None)
        "d_tier2":         round(d_tier2),         # gravity-corrected (dominant)
        "d_tier3":         d_tier3,                # full statistical (gravity fails)
        "d_ext_pts":       round(d_trigger) - round(d_tier2),
        "d_pull_pts":      round(d_gravity_pull),
    }


# ── Module singleton ───────────────────────────────────────────────────────────
_reader: Optional[_DCMPredictionReader] = None
_reader_lock = threading.Lock()


def get_dcm_reader() -> _DCMPredictionReader:
    global _reader
    with _reader_lock:
        if _reader is None:
            _reader = _DCMPredictionReader()
    return _reader
