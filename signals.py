"""
signals.py  --  Multi-timeframe trade signal engine for NSE indices.

Timeframes : 5-min (Intraday scalp), 15-min (Intraday swing),
             60-min (BTST / short positional), Daily (Positional)
Indicators : EMA 9/21/50, MACD 12/26/9, RSI 14, VWAP (intraday resets)
Output     : Signal grid + ranked option recommendations with Entry/SL/Target
"""

import datetime
import time
import threading
from pathlib import Path

import requests
import pandas as pd

# ── Config ─────────────────────────────────────────────────────────────────────
APP_ID     = "WVDZUTO6HL-100"
TOKEN_FILE = Path("access_token.txt")
from core.constants import IST   # single source of truth  # noqa: E402

HISTORY_URL = "https://api-t1.fyers.in/data/history"

# Maps Fyers history resolution string → candle_store resolution key.
# "D" maps to None because daily candles don't need a forming-candle append.
_STORE_RES_MAP = {"5S": "5sec", "15S": "15sec", "30S": "30sec",
                  "1": "1min", "5": "5min", "15": "15min", "60": "60min", "D": None}

INDEX_LABELS = {
    "NSE:NIFTY50-INDEX":    "NIFTY 50",
    "NSE:NIFTYBANK-INDEX":  "BANK NIFTY",
    "NSE:FINNIFTY-INDEX":   "FIN NIFTY",
    "NSE:MIDCPNIFTY-INDEX": "MIDCAP NIFTY",
}

TIMEFRAMES = {
    "5min":  {"res": "5",  "days": 3,  "label": "5 Min",   "trade": "INTRADAY",   "min_candles": 30},
    "15min": {"res": "15", "days": 5,  "label": "15 Min",  "trade": "INTRADAY",   "min_candles": 26},
    "60min": {"res": "60", "days": 10, "label": "1 Hour",  "trade": "BTST",       "min_candles": 26},
    "daily": {"res": "D",  "days": 60, "label": "Daily",   "trade": "POSITIONAL", "min_candles": 30},
}

# ── Helpers ────────────────────────────────────────────────────────────────────
def _fmt_oi(v) -> str:
    """Format an OI/volume number. Handles negative values (OI unwinding)."""
    if not v: return "—"
    sign = "-" if v < 0 else ""
    a = abs(v)
    if a >= 1_000_000: return f"{sign}{a / 1_000_000:.2f}M"
    if a >= 1_000:     return f"{sign}{a / 1_000:.1f}K"
    return f"{sign}{int(a)}"


# ── Auth ───────────────────────────────────────────────────────────────────────
def _auth() -> str:
    raw = TOKEN_FILE.read_text(encoding="utf-8").strip() if TOKEN_FILE.exists() else ""
    return f"{APP_ID}:{raw}"


# ── OHLCV fetch (with 60-second cache) ────────────────────────────────────────
_ohlcv_cache: dict = {}
_ohlcv_lock  = threading.Lock()


def fetch_ohlcv(sym: str, resolution: str, days: int) -> pd.DataFrame:
    """
    Fetch completed OHLCV candles from Fyers API, then append the live
    forming candle from candle_store so signals always include the
    current partial bar — no more 1-candle lag.

    Cache: 30 s for intraday resolutions (was 60 s), 120 s for daily.
    """
    key    = f"{sym}_{resolution}"
    is_eod = resolution in ("D", "W", "M")
    ttl    = 120 if is_eod else 30

    with _ohlcv_lock:
        c = _ohlcv_cache.get(key)
        if c and time.time() - c["ts"] < ttl:
            return c["df"]

    now        = datetime.datetime.now(tz=IST)
    range_to   = now.strftime("%Y-%m-%d")
    range_from = (now - datetime.timedelta(days=days)).strftime("%Y-%m-%d")

    try:
        resp = requests.get(
            HISTORY_URL,
            headers={"Authorization": _auth(), "Content-Type": "application/json", "version": "3"},
            params={"symbol": sym, "resolution": resolution, "date_format": "1",
                    "range_from": range_from, "range_to": range_to, "cont_flag": "1"},
            timeout=15,
        )
        data    = resp.json()
        candles = data.get("candles", [])
        if data.get("s") != "ok" or not candles:
            return pd.DataFrame()
        df = pd.DataFrame(candles, columns=["ts", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata")
        df = df.reset_index(drop=True)
    except Exception:
        return pd.DataFrame()

    # ── Append live forming candle from WebSocket-driven candle store ──────────
    store_res = _STORE_RES_MAP.get(resolution)
    if store_res and not is_eod:
        try:
            from intraday_store import candle_store
            forming = candle_store.forming_candle(sym, store_res)
            if forming and "ts" in forming:
                # Normalise to pd.Timestamp so the comparison is type-safe
                # regardless of whether intraday_store uses ZoneInfo or
                # datetime.timezone for its IST constant.
                f_ts = pd.Timestamp(forming["ts"]).tz_convert("Asia/Kolkata")
                # only append if forming candle is NEWER than last API candle
                if df.empty or f_ts > df["ts"].iloc[-1]:
                    f_row = pd.DataFrame([{
                        "ts":     f_ts,
                        "open":   forming["open"],
                        "high":   forming["high"],
                        "low":    forming["low"],
                        "close":  forming["close"],
                        "volume": forming["volume"],
                    }])
                    df = pd.concat([df, f_row], ignore_index=True)
        except Exception:
            pass  # candle_store not yet warm — degrade gracefully

    with _ohlcv_lock:
        _ohlcv_cache[key] = {"df": df, "ts": time.time()}
    return df


# ── Technical indicators ───────────────────────────────────────────────────────
def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    g = d.clip(lower=0).ewm(alpha=1/n, min_periods=n).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1/n, min_periods=n).mean()
    return 100 - 100 / (1 + g / l)


def _macd(s: pd.Series, fast=12, slow=26, sig=9):
    m = _ema(s, fast) - _ema(s, slow)
    sl = m.ewm(span=sig, adjust=False).mean()
    return m, sl, m - sl


def _vwap(df: pd.DataFrame) -> pd.Series:
    tp      = (df["high"] + df["low"] + df["close"]) / 3
    grp     = df["ts"].dt.date
    cum_vol = df["volume"].groupby(grp).cumsum()
    # Guard: zero cumulative volume (session open before any volume received, or
    # index bars with no traded volume) returns NaN so the VWAP signal is skipped
    # rather than firing a spurious bearish score every session open.
    return (tp * df["volume"]).groupby(grp).cumsum() / cum_vol.replace(0, float("nan"))


def _bollinger(s: pd.Series, n=20, k=2):
    mid = s.rolling(n).mean()
    std = s.rolling(n).std()
    return mid - k*std, mid, mid + k*std


# ── Single timeframe analysis ──────────────────────────────────────────────────
def _analyze(df: pd.DataFrame, min_candles: int) -> dict:
    if len(df) < min_candles:
        return {"signal": "INSUFFICIENT DATA", "score": 0, "confidence": 0, "reasons": []}

    close = df["close"]
    vol   = df["volume"]

    e9  = _ema(close, 9)
    e21 = _ema(close, 21)
    e50 = _ema(close, 50)
    m, ms, mh = _macd(close)
    r   = _rsi(close)
    vw  = _vwap(df)
    bb_lo, bb_mid, bb_hi = _bollinger(close)

    # Current values
    c  = close.iloc[-1];   cp = close.iloc[-2]
    e9n = e9.iloc[-1];     e9p = e9.iloc[-2]
    e21n= e21.iloc[-1];    e21p= e21.iloc[-2]
    e50n= e50.iloc[-1]
    mn  = m.iloc[-1];      mp  = m.iloc[-2]
    msn = ms.iloc[-1];     msp = ms.iloc[-2]
    mhn = mh.iloc[-1];     mhp = mh.iloc[-2]
    rn  = r.iloc[-1];      rp  = r.iloc[-2]
    vwn = vw.iloc[-1]
    bbn = bb_hi.iloc[-1];  bln = bb_lo.iloc[-1]
    avg_vol = vol.rolling(20).mean().iloc[-1]
    cur_vol = vol.iloc[-1]

    score   = 0.0
    reasons = []   # (bias, text)

    # 1. EMA 9/21 cross
    if e9p < e21p and e9n > e21n:
        score += 2
        reasons.append(("bull", "EMA 9 crossed ABOVE EMA 21 (fresh bullish cross)"))
    elif e9p > e21p and e9n < e21n:
        score -= 2
        reasons.append(("bear", "EMA 9 crossed BELOW EMA 21 (fresh bearish cross)"))
    elif e9n > e21n:
        score += 1
        reasons.append(("bull", f"EMA 9 ({e9n:.0f}) above EMA 21 ({e21n:.0f}) — uptrend"))
    else:
        score -= 1
        reasons.append(("bear", f"EMA 9 ({e9n:.0f}) below EMA 21 ({e21n:.0f}) — downtrend"))

    # 2. EMA 50 trend filter
    if c > e50n:
        score += 1
        reasons.append(("bull", f"Price above EMA 50 ({e50n:.0f}) — bullish structure"))
    else:
        score -= 1
        reasons.append(("bear", f"Price below EMA 50 ({e50n:.0f}) — bearish structure"))

    # 3. MACD histogram
    if mhp < 0 and mhn > 0:
        score += 2
        reasons.append(("bull", f"MACD histogram crossed ZERO upward (strong bullish signal)"))
    elif mhp > 0 and mhn < 0:
        score -= 2
        reasons.append(("bear", f"MACD histogram crossed ZERO downward (strong bearish signal)"))
    elif mhn > 0 and mhn > mhp:
        score += 1
        reasons.append(("bull", f"MACD histogram expanding positive ({mhn:.2f}) — momentum up"))
    elif mhn < 0 and mhn < mhp:
        score -= 1
        reasons.append(("bear", f"MACD histogram expanding negative ({mhn:.2f}) — momentum down"))

    # 4. RSI
    if rn < 30:
        score += 2
        reasons.append(("bull", f"RSI {rn:.1f} — oversold, bounce likely"))
    elif rn > 70:
        score -= 2
        reasons.append(("bear", f"RSI {rn:.1f} — overbought, reversal risk"))
    elif rn > 55 and rn > rp:
        score += 1
        reasons.append(("bull", f"RSI {rn:.1f} rising in bullish zone"))
    elif rn < 45 and rn < rp:
        score -= 1
        reasons.append(("bear", f"RSI {rn:.1f} falling in bearish zone"))
    else:
        reasons.append(("neut", f"RSI {rn:.1f} — neutral zone"))

    # 5. VWAP — only score when VWAP is defined. A volume-less session (pre-open, or
    #    any instrument Fyers serves without volume) yields NaN VWAP; `c > NaN` is
    #    False, which would silently score a spurious bearish −1 every such bar.
    if pd.notna(vwn):
        if c > vwn:
            score += 1
            reasons.append(("bull", f"Price ({c:.1f}) above VWAP ({vwn:.1f}) — buyers in control"))
        else:
            score -= 1
            reasons.append(("bear", f"Price ({c:.1f}) below VWAP ({vwn:.1f}) — sellers in control"))
    else:
        reasons.append(("neut", "VWAP undefined (no volume) — skipped"))

    # 6. Volume confirmation
    if cur_vol > avg_vol * 1.5 and c > cp:
        score += 1
        reasons.append(("bull", f"High volume ({cur_vol/avg_vol:.1f}x avg) on up candle — confirmed move"))
    elif cur_vol > avg_vol * 1.5 and c < cp:
        score -= 1
        reasons.append(("bear", f"High volume ({cur_vol/avg_vol:.1f}x avg) on down candle — confirmed move"))

    # 7. Bollinger Band position
    if c > bbn:
        score -= 0.5
        reasons.append(("bear", f"Price above upper Bollinger Band — extended, mean reversion risk"))
    elif c < bln:
        score += 0.5
        reasons.append(("bull", f"Price below lower Bollinger Band — oversold, reversion expected"))

    # Verdict
    score = round(score, 1)
    if   score >= 4:  signal, clr = "STRONG BUY",  "#22c55e"
    elif score >= 2:  signal, clr = "BUY",          "#4ade80"
    elif score >= 0.5:signal, clr = "MILD BUY",     "#86efac"
    elif score <= -4: signal, clr = "STRONG SELL",  "#ef4444"
    elif score <= -2: signal, clr = "SELL",          "#f87171"
    elif score <=-0.5:signal, clr = "MILD SELL",    "#fca5a5"
    else:             signal, clr = "NEUTRAL",       "#94a3b8"

    confidence = round(min(abs(score) / 6 * 100, 95), 0)

    return {
        "signal":     signal,
        "color":      clr,
        "score":      score,
        "confidence": confidence,
        "reasons":    reasons,
        "close":      c,
        "rsi":        rn,
        "vwap":       vwn,
        "ema9":       e9n,
        "ema21":      e21n,
        "ema50":      e50n,
        "macd_hist":  mhn,
    }


# ── Multi-timeframe consensus ─────────────────────────────────────────────────
_signal_cache: dict = {}
_signal_lock  = threading.Lock()


def analyze_index(sym: str) -> dict:
    """Full multi-timeframe analysis for one index. Cached 60s."""
    with _signal_lock:
        c = _signal_cache.get(sym)
        if c and time.time() - c["ts"] < 60:
            return c["data"]

    result = {"symbol": sym, "label": INDEX_LABELS.get(sym, sym), "timeframes": {}}

    for tf_key, cfg in TIMEFRAMES.items():
        df  = fetch_ohlcv(sym, cfg["res"], cfg["days"])
        ana = _analyze(df, cfg["min_candles"])
        ana["label"]      = cfg["label"]
        ana["trade_type"] = cfg["trade"]
        result["timeframes"][tf_key] = ana

    # Consensus score (weighted: daily 30%, 60min 25%, 15min 25%, 5min 20%)
    weights = {"5min": 0.20, "15min": 0.25, "60min": 0.25, "daily": 0.30}
    w_score = sum(
        result["timeframes"][k]["score"] * weights[k]
        for k in TIMEFRAMES
        if result["timeframes"][k].get("score") is not None
    )

    # Count agreements
    tfs = result["timeframes"]
    bull_count = sum(1 for t in tfs.values() if t["score"] > 0.5)
    bear_count = sum(1 for t in tfs.values() if t["score"] < -0.5)
    agreement  = max(bull_count, bear_count)  # how many TFs agree

    result["weighted_score"]  = round(w_score, 2)
    result["bull_timeframes"] = bull_count
    result["bear_timeframes"] = bear_count
    result["agreement"]       = agreement

    # Overall direction
    if   w_score >= 2.5: result["overall"] = ("STRONG BUY",  "#22c55e")
    elif w_score >= 1.0: result["overall"] = ("BUY",          "#4ade80")
    elif w_score >= 0.3: result["overall"] = ("MILD BUY",    "#86efac")
    elif w_score <=-2.5: result["overall"] = ("STRONG SELL", "#ef4444")
    elif w_score <=-1.0: result["overall"] = ("SELL",         "#f87171")
    elif w_score <=-0.3: result["overall"] = ("MILD SELL",   "#fca5a5")
    else:                result["overall"] = ("NEUTRAL",      "#94a3b8")

    # Persist to intraday DB (non-blocking; fires only on fresh computation)
    try:
        from intraday_db import idb
        idb.write_signal(result)
    except Exception:
        pass

    with _signal_lock:
        _signal_cache[sym] = {"data": result, "ts": time.time()}
    return result


# NOTE: the legacy `recommend_option()` (hardcoded 30%/+50%/+100% premium geometry)
# was removed in the 2026-06-20 audit. The single authoritative recommender is
# trade_setup.build_recommendation (session-phase-shaped SL/targets, 11 layers).
# Do not reintroduce a second recommender here — one engine, one risk model.


# ── Full analysis for all 4 indices ──────────────────────────────────────────
def run_full_analysis(index_symbols: list[str]) -> dict:
    """Run analysis for all indices. Returns results dict keyed by symbol."""
    results = {}
    for sym in index_symbols:
        try:
            results[sym] = analyze_index(sym)
        except Exception as e:
            results[sym] = {"symbol": sym, "label": INDEX_LABELS.get(sym, sym),
                            "error": str(e), "timeframes": {}}
    return results
