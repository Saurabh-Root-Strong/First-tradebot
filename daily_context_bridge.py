"""
daily_context_bridge.py  --  EOD structural context bridge.

Connects Tradebot to Daily_Cash_Market's DuckDB (read-only) and extracts
the six sub-signals that form Layer 9: Daily Context (17% weight).

Sub-signals (each clamped to [-4, +4]):
  B1  Prediction Engine   24-signal EOD directional forecast + accuracy weight
  B2  HMM Regime          Hidden Markov state (Bull/Bear/Sideways) + Hurst filter
  B3  FII Flow Trend       5-day + 10-day net futures flow in Rs. Cr
  B4  Market Breadth       % sectors advancing + NIFTY heavyweight confirmation
  B5  VIX Regime           India VIX level and 5-day change
  B6  EOD OI Positioning   End-of-day PCR and OI change direction

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
import threading
import time
from pathlib import Path
from typing import Optional

# ── Configuration ──────────────────────────────────────────────────────────────
DB_PATH   = Path(r"D:\Python Projects\Daily_Cash_Market\data\market_data.duckdb")
CACHE_TTL = 4 * 3600   # 4 hours — EOD data, stable during trading hours

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
        self._lock      = threading.Lock()
        self._cache: dict = {}          # {fyers_sym: data_dict, "__common__": ...}
        self._loaded_at: float = 0.0
        self._available: bool  = False
        # Load silently in background so dashboard startup is not delayed
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
        }

    def get_panel_data(self, fyers_sym: str) -> dict:
        """
        Full context dict for the dashboard 'Daily Context' card.
        """
        self._maybe_refresh()
        with self._lock:
            data   = dict(self._cache.get(fyers_sym, {}))
            common = dict(self._cache.get("__common__", {}))
        data["__common__"] = common
        data["available"]  = self._available
        data["loaded_at"]  = self._loaded_at
        return data

    def is_available(self) -> bool:
        with self._lock:
            return self._available

    # ── Cache management ───────────────────────────────────────────────────────

    def _maybe_refresh(self):
        with self._lock:
            age = time.time() - self._loaded_at
        if age > CACHE_TTL:
            threading.Thread(target=self._load, daemon=True, name="dcb-refresh").start()

    def _load(self):
        """Fetch all context in a single DuckDB session and update cache atomically."""
        if not DB_PATH.exists():
            with self._lock:
                self._available = False
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

        except Exception:
            with self._lock:
                self._available = False

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
        except Exception:
            pass

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
                    float(df["close_val"].iloc[min(4, len(df) - 1)]), 2
                )
        except Exception:
            pass

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
        except Exception:
            pass

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
        except Exception:
            pass

        return r

    def _fetch_symbol(self, con, dcm_sym: str) -> dict:
        """Per-index context from prediction_log, FII stats, index_data, fno_bhavcopy."""
        r: dict = {"symbol": dcm_sym}

        # ── prediction_log: latest prediction + 12-dim fingerprint ───────────
        try:
            row = con.execute("""
                SELECT
                    trade_date, direction_pred, confidence_pred,
                    composite_score, signal_count,
                    feat_pcr, feat_carry, feat_fii_net, feat_fii_5d_cumul,
                    feat_vix, feat_vix_5d_chg, feat_breadth,
                    feat_hurst, feat_entropy, feat_oi_score,
                    hmm_state, memory_label,
                    range_low, range_high, target_close, expected_move_pts
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
                 rng_lo, rng_hi, target, move_pts) = row
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
                    "range_low":        float(rng_lo)  if rng_lo  else None,
                    "range_high":       float(rng_hi)  if rng_hi  else None,
                    "target_close":     float(target)  if target  else None,
                    "expected_move_pts": float(move_pts) if move_pts else None,
                })
        except Exception:
            pass

        # ── prediction accuracy (last 30 filled predictions) ─────────────────
        try:
            row = con.execute("""
                SELECT
                    COUNT(*)                                                  AS n,
                    SUM(CASE WHEN was_correct THEN 1 ELSE 0 END)            AS correct,
                    AVG(CASE WHEN confidence_pred = 'HIGH' AND was_correct THEN 1.0
                              WHEN confidence_pred = 'HIGH'                 THEN 0.0
                         END)                                                AS hi_acc
                FROM prediction_log
                WHERE fno_symbol    = ?
                  AND outcome_filled = TRUE
                ORDER BY trade_date DESC
                LIMIT 30
            """, [dcm_sym]).fetchone()
            if row and row[0]:
                r["pred_acc_30d"]      = round((row[1] or 0) / row[0] * 100, 1)
                r["pred_hi_conf_acc"]  = (round(row[2] * 100, 1) if row[2] is not None else None)
                r["pred_sample"]       = row[0]
        except Exception:
            pass

        # ── index_data: daily OHLCV → key levels ─────────────────────────────
        idx_name = _IDX_NAME.get(dcm_sym)
        if idx_name:
            try:
                df = con.execute("""
                    SELECT trade_date, high_val, low_val, close_val, pct_chg
                    FROM index_data
                    WHERE index_name = ?
                    ORDER BY trade_date DESC
                    LIMIT 22
                """, [idx_name]).df()
                if not df.empty:
                    r["prev_close"] = float(df["close_val"].iloc[0])
                    r["high_20d"]   = float(df["high_val"].max())
                    r["low_20d"]    = float(df["low_val"].min())
                    r["week_high"]  = float(df["high_val"].head(5).max())
                    r["week_low"]   = float(df["low_val"].head(5).min())
                    if len(df) >= 5:
                        r["pct_5d"] = round(float(df["pct_chg"].head(5).sum()), 2)
                    # Consecutive up days (momentum streak)
                    streak = 0
                    for chg in df["pct_chg"]:
                        if chg > 0:
                            streak += 1
                        else:
                            break
                    r["up_streak"] = streak
            except Exception:
                pass

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
            except Exception:
                pass

        # ── fno_bhavcopy: EOD option chain OI + OI change ─────────────────────
        try:
            df = con.execute("""
                SELECT option_type,
                       SUM(open_interest) AS oi,
                       SUM(chg_in_oi)     AS oi_chg
                FROM fno_bhavcopy
                WHERE symbol     = ?
                  AND instrument  = 'OPTIDX'
                  AND trade_date  = (
                      SELECT MAX(trade_date) FROM fno_bhavcopy WHERE symbol = ?
                  )
                  AND option_type IN ('CE','PE')
                GROUP BY option_type
            """, [dcm_sym, dcm_sym]).df()
            if not df.empty:
                ce  = df[df["option_type"] == "CE"]
                pe  = df[df["option_type"] == "PE"]
                c_oi  = int(ce["oi"].iloc[0])    if not ce.empty else 0
                p_oi  = int(pe["oi"].iloc[0])    if not pe.empty else 0
                c_chg = int(ce["oi_chg"].iloc[0]) if not ce.empty else 0
                p_chg = int(pe["oi_chg"].iloc[0]) if not pe.empty else 0
                r["eod_call_oi"]  = c_oi
                r["eod_put_oi"]   = p_oi
                r["eod_pcr"]      = round(p_oi / c_oi, 3) if c_oi else 0.0
                r["eod_call_chg"] = c_chg
                r["eod_put_chg"]  = p_chg
        except Exception:
            pass

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

        # Staleness: prediction older than 3 calendar days is unreliable
        try:
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

    # ── B3: FII Flow Trend ────────────────────────────────────────────────────

    def _b3_fii(self, data: dict, signals: list) -> float:
        fii_5d   = data.get("fii_5d_cr",       0.0) or 0.0
        fii_10d  = data.get("fii_10d_cr",      0.0) or 0.0
        pos_days = data.get("fii_5d_pos_days", 0)   or 0
        today_cr = data.get("fii_today_cr",    0.0) or 0.0

        if not (fii_5d or fii_10d):
            return 0.0

        score = 0.0

        # 5-day net flow — threshold ±₹1000Cr = significant institutional activity
        if fii_5d > 2500:
            score += 1.5
            signals.append(("bull",
                f"FII futures 5D: +{fii_5d:,.0f}Cr ({pos_days}/5 buying days) "
                f"— sustained institutional accumulation"))
        elif fii_5d > 800:
            score += 0.75
            signals.append(("bull",
                f"FII futures 5D: +{fii_5d:,.0f}Cr ({pos_days}/5 buying days) "
                f"— consistent FII buying"))
        elif fii_5d < -2500:
            score -= 1.5
            signals.append(("bear",
                f"FII futures 5D: {fii_5d:,.0f}Cr ({pos_days}/5 buying days) "
                f"— sustained institutional distribution"))
        elif fii_5d < -800:
            score -= 0.75
            signals.append(("bear",
                f"FII futures 5D: {fii_5d:,.0f}Cr ({pos_days}/5 buying days) "
                f"— consistent FII selling"))
        else:
            signals.append(("neut",
                f"FII futures 5D: {fii_5d:+,.0f}Cr — neutral institutional stance"))

        # 10-day structural context (confirms or contradicts 5D)
        if abs(fii_10d) > 6000:
            structural_bias = "bull" if fii_10d > 0 else "bear"
            direction_word  = "accumulation" if fii_10d > 0 else "distribution"
            score += 0.5 if fii_10d > 0 else -0.5
            signals.append((structural_bias,
                f"FII 10D structural: {fii_10d:+,.0f}Cr — "
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
        # India VIX from DB > fall back to prediction_log feature
        vix     = common.get("india_vix") or data.get("feat_vix") or 0.0
        vix_5d  = common.get("india_vix_5d_chg") or data.get("feat_vix_5d_chg") or 0.0

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
