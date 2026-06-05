"""
indicators.py  —  Extended technical indicator library for the NIFTY trading bot.

Complements signals.py by adding indicators missing from the hand-rolled set:
  Supertrend       (Olivier Seban — the #1 trend-following tool in Indian markets)
  ATR              (via `ta` library — Average True Range for volatility / stop-loss)
  ADX              (Average Directional Index — trend strength filter)
  Stochastic       (%K / %D — momentum overbought / oversold)
  CCI              (Commodity Channel Index — divergence signals)
  OBV              (On Balance Volume — institutional accumulation)
  VWAP Band        (VWAP ± 1σ / 2σ — intraday mean-reversion levels)
  Ichimoku Cloud   (Tenkan / Kijun / Senkou A+B / Chikou — multi-dimension view)
  Candlestick      (12 key price-action patterns — pure pandas, no TA-Lib needed)
  TradingView      (live BUY/SELL/NEUTRAL summary from TradingView for any symbol)

Usage:
    from indicators import supertrend, candlestick_patterns, get_tradingview_signal
    from indicators import add_all_indicators    # adds every column to df in-place

Requires:  ta>=0.11  tradingview-ta>=3.3  (both Python 3.14 compatible)
           pip install ta tradingview-ta   (already installed in this venv)

All indicator functions:
  - Take a pandas DataFrame with columns [open, high, low, close, volume]
  - Return a pandas Series or DataFrame column-aligned with the input
  - Are NaN-safe (first N rows = NaN due to look-back period)
  - Never raise — return NaN Series on any error
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

try:
    import ta as _ta
    _TA_OK = True
except ImportError:
    _TA_OK = False


# ── Internal guard ────────────────────────────────────────────────────────────

def _req_ta(name: str) -> None:
    if not _TA_OK:
        raise ImportError(f"{name} requires the `ta` package — pip install ta")


# ══════════════════════════════════════════════════════════════════════════════
# 1. SUPERTREND
# ══════════════════════════════════════════════════════════════════════════════

def supertrend(
    df: pd.DataFrame,
    period: int = 10,
    multiplier: float = 3.0,
) -> pd.DataFrame:
    """
    Supertrend — Olivier Seban's trend-following indicator.

    Parameters
    ----------
    df          : DataFrame with high / low / close columns
    period      : ATR look-back period (default 10 — standard Indian intraday)
    multiplier  : ATR multiplier for band width (default 3.0)

    Returns
    -------
    DataFrame with columns:
      supertrend        float  — the trailing stop value
      supertrend_dir    int    — +1 = uptrend (price above ST), -1 = downtrend
      supertrend_bull   bool   — True when in uptrend
      supertrend_signal str    — 'BUY' on uptrend flip, 'SELL' on downtrend flip

    Usage
    -----
    st = supertrend(df)
    current_trend = st["supertrend_dir"].iloc[-1]   # +1 or -1
    signal = st["supertrend_signal"].iloc[-1]        # 'BUY' / 'SELL' / ''

    Theory
    ------
    Upper Band = HL2 + multiplier × ATR(period)
    Lower Band = HL2 − multiplier × ATR(period)
    Trend flips when price crosses the active band.
    Active band self-adjusts (only moves in the direction of trend).
    """
    _req_ta("supertrend")

    hl2  = (df["high"] + df["low"]) / 2.0
    atr  = _ta.volatility.AverageTrueRange(
        df["high"], df["low"], df["close"], window=period
    ).average_true_range()

    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr

    n = len(df)
    st    = np.full(n, np.nan)
    st_d  = np.zeros(n, dtype=int)

    final_upper = upper.values.copy()
    final_lower = lower.values.copy()
    close = df["close"].values

    for i in range(1, n):
        if np.isnan(atr.iloc[i]):
            continue

        # Final upper band: only moves down
        final_upper[i] = upper.iloc[i] if (
            upper.iloc[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]
        ) else final_upper[i - 1]

        # Final lower band: only moves up
        final_lower[i] = lower.iloc[i] if (
            lower.iloc[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]
        ) else final_lower[i - 1]

        # Determine trend direction.
        # st_d == 1 → uptrend  (ST is lower band, price above it)
        # st_d == -1 → downtrend (ST is upper band, price below it)
        if np.isnan(st[i - 1]):
            st_d[i] = 1
        elif st[i - 1] == final_upper[i - 1]:
            # Was in downtrend (ST = upper band).
            # Bullish reversal: price crosses ABOVE upper band → switch to uptrend.
            st_d[i] = 1 if close[i] > final_upper[i] else -1
        else:
            # Was in uptrend (ST = lower band).
            # Bearish reversal: price crosses BELOW lower band → switch to downtrend.
            st_d[i] = -1 if close[i] < final_lower[i] else 1

        # Supertrend value is the active band
        st[i] = final_lower[i] if st_d[i] == 1 else final_upper[i]

    # Convert direction: 1 = downtrend (price below ST), -1 → we want +1 = uptrend
    # Convention: price > ST → bullish (+1), price < ST → bearish (-1)
    direction = np.where(st_d == 1, 1, -1)          # 1 = bull, -1 = bear

    # Signal: 'BUY' when direction flips +1, 'SELL' when flips -1
    direction_series = pd.Series(direction, index=df.index)
    prev_dir = direction_series.shift(1)
    signals  = np.where(
        (direction_series == 1)  & (prev_dir == -1), "BUY",
        np.where(
            (direction_series == -1) & (prev_dir == 1),  "SELL", ""
        )
    )

    return pd.DataFrame({
        "supertrend":        pd.Series(st,         index=df.index),
        "supertrend_dir":    direction_series,
        "supertrend_bull":   direction_series == 1,
        "supertrend_signal": pd.Series(signals,    index=df.index),
    })


# ══════════════════════════════════════════════════════════════════════════════
# 2. STANDARD INDICATORS via `ta` library
# ══════════════════════════════════════════════════════════════════════════════

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range — volatility and stop-loss sizing."""
    _req_ta("atr")
    return _ta.volatility.AverageTrueRange(
        df["high"], df["low"], df["close"], window=period
    ).average_true_range().rename("atr")


def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    ADX — Average Directional Index.
    Returns: adx (trend strength), adx_pos (+DI), adx_neg (-DI).
    ADX > 25 = trending.  ADX < 20 = ranging (avoid trend-following signals).
    """
    _req_ta("adx")
    ind = _ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=period)
    return pd.DataFrame({
        "adx":     ind.adx(),
        "adx_pos": ind.adx_pos(),
        "adx_neg": ind.adx_neg(),
    })


def stochastic(df: pd.DataFrame, k: int = 14, d: int = 3) -> pd.DataFrame:
    """
    Stochastic Oscillator.
    Returns: stoch_k, stoch_d.
    K > 80 = overbought.  K < 20 = oversold.
    """
    _req_ta("stochastic")
    ind = _ta.momentum.StochasticOscillator(
        df["high"], df["low"], df["close"],
        window=k, smooth_window=d
    )
    return pd.DataFrame({
        "stoch_k": ind.stoch(),
        "stoch_d": ind.stoch_signal(),
    })


def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    CCI — Commodity Channel Index.
    >100 = overbought / strong trend.  <-100 = oversold / strong downtrend.
    """
    _req_ta("cci")
    return _ta.trend.CCIIndicator(
        df["high"], df["low"], df["close"], window=period
    ).cci().rename("cci")


def obv(df: pd.DataFrame) -> pd.Series:
    """On Balance Volume — cumulative volume pressure (institutional accumulation)."""
    _req_ta("obv")
    return _ta.volume.OnBalanceVolumeIndicator(
        df["close"], df["volume"]
    ).on_balance_volume().rename("obv")


def mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Money Flow Index — volume-weighted RSI. >80 overbought, <20 oversold."""
    _req_ta("mfi")
    return _ta.volume.MFIIndicator(
        df["high"], df["low"], df["close"], df["volume"], window=period
    ).money_flow_index().rename("mfi")


def vwap_bands(df: pd.DataFrame) -> pd.DataFrame:
    """
    VWAP with ±1σ / ±2σ Bollinger-style bands.
    Resets at midnight IST (matches exchange session boundary).
    Requires a 'ts' column (tz-aware or naive IST).
    """
    if "ts" not in df.columns:
        return pd.DataFrame(index=df.index)

    tp  = (df["high"] + df["low"] + df["close"]) / 3.0
    pv  = tp * df["volume"]

    if hasattr(df["ts"].iloc[0], "date"):
        grp = df["ts"].dt.date
    else:
        grp = pd.to_datetime(df["ts"]).dt.date

    cum_pv  = pv.groupby(grp).cumsum()
    cum_vol = df["volume"].groupby(grp).cumsum()
    # Guard: replace zero cumulative volume with NaN so first-bar VWAP = NaN
    # instead of inf/error when indices or zero-volume bars are present.
    vwap    = cum_pv / cum_vol.replace(0, float("nan"))

    # Rolling intraday σ of typical price (resets by date)
    deviation   = (tp - vwap) ** 2
    cum_dev     = deviation.groupby(grp).cumsum()
    count       = df.groupby(grp).cumcount() + 1
    rolling_std = (cum_dev / count) ** 0.5

    return pd.DataFrame({
        "vwap":     vwap,
        "vwap_1u":  vwap + 1 * rolling_std,
        "vwap_1d":  vwap - 1 * rolling_std,
        "vwap_2u":  vwap + 2 * rolling_std,
        "vwap_2d":  vwap - 2 * rolling_std,
    })


def ichimoku(df: pd.DataFrame,
             tenkan: int = 9, kijun: int = 26,
             senkou_b: int = 52) -> pd.DataFrame:
    """
    Ichimoku Cloud.
    Tenkan (9) = fast signal line.
    Kijun  (26) = slow baseline.
    Senkou A/B form the cloud (Kumo).
    Chikou = close shifted back 26 bars (lagging span — for confirmation).
    """
    def _hl_avg(n: int) -> pd.Series:
        return (df["high"].rolling(n).max() + df["low"].rolling(n).min()) / 2.0

    tenkan_sen  = _hl_avg(tenkan)
    kijun_sen   = _hl_avg(kijun)
    senkou_a    = ((tenkan_sen + kijun_sen) / 2.0).shift(kijun)
    senkou_b    = _hl_avg(senkou_b).shift(kijun)
    # Chikou (lagging span) is the current close plotted 26 bars in the PAST for
    # chart display purposes — achieved by shifting the series forward by kijun.
    # shift(-kijun) reads FUTURE data and must NEVER be used in live signal logic.
    # ichi_chikou  = display column (future look-ahead — use for charts only)
    # ichi_chikou_signal = signal-safe column (future region replaced with NaN)
    chikou        = df["close"].shift(-kijun)
    chikou_signal = df["close"].shift(-kijun).where(
        pd.RangeIndex(len(df)) < len(df) - kijun, other=float("nan")
    )

    return pd.DataFrame({
        "ichi_tenkan":          tenkan_sen,
        "ichi_kijun":           kijun_sen,
        "ichi_senkou_a":        senkou_a,
        "ichi_senkou_b":        senkou_b,
        "ichi_chikou":          chikou,          # display only — contains future data
        "ichi_chikou_signal":   chikou_signal,   # signal-safe — NaN for future bars
        # Derived: price vs cloud
        "ichi_above_cloud": (df["close"] > senkou_a) & (df["close"] > senkou_b),
        "ichi_below_cloud": (df["close"] < senkou_a) & (df["close"] < senkou_b),
        "ichi_in_cloud":    ~(
            ((df["close"] > senkou_a) & (df["close"] > senkou_b)) |
            ((df["close"] < senkou_a) & (df["close"] < senkou_b))
        ),
    })


# ══════════════════════════════════════════════════════════════════════════════
# 3. CANDLESTICK PATTERN RECOGNITION  (pure pandas — no TA-Lib required)
# ══════════════════════════════════════════════════════════════════════════════

def candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect 12 high-probability candlestick patterns for NSE intraday trading.

    Returns a DataFrame of boolean columns — True on the bar where the
    pattern completes.  All patterns are aligned to the CLOSING bar of the
    formation (single-candle patterns reference the current bar;
    multi-candle patterns reference the last bar).

    Patterns:
      Bullish:  hammer, inv_hammer, bull_engulfing, morning_star,
                bull_marubozu, bull_spinning_top, bullish_doji_star
      Bearish:  hanging_man, shooting_star, bear_engulfing, evening_star,
                bear_marubozu, bear_spinning_top, bearish_doji_star, doji

    Parameters for a "significant" body vs wick:
      body_ratio   >= 0.6  → Marubozu (big body, small wicks)
      body_ratio   <= 0.1  → Doji / spinning top (small body)
      wick_ratio   >= 0.5  → Hammer / Shooting Star (long wick)
    """
    o = df["open"]
    h = df["high"]
    l = df["low"]
    c = df["close"]

    rng        = h - l                              # full candle range
    body       = (c - o).abs()                      # body size
    upper_wick = h - df[["open","close"]].max(axis=1)
    lower_wick = df[["open","close"]].min(axis=1) - l

    body_ratio  = body  / rng.replace(0, np.nan)    # body as fraction of range
    upper_ratio = upper_wick / rng.replace(0, np.nan)
    lower_ratio = lower_wick / rng.replace(0, np.nan)

    bull_candle = c >= o    # green
    bear_candle = c < o     # red

    # ── Single-candle patterns ───────────────────────────────────────────────

    # Doji: body very small relative to range
    doji = body_ratio <= 0.10

    # Hammer / Hanging Man: small body at top, long lower wick (≥ 2× body)
    _hammer_shape = (
        (body_ratio <= 0.35) &
        (lower_wick >= 2 * body) &
        (upper_wick <= body * 0.5)
    )
    hammer      = _hammer_shape & bull_candle   # appears after downtrend (bullish)
    hanging_man = _hammer_shape & bear_candle   # appears after uptrend (bearish)

    # Inverted Hammer / Shooting Star: small body at bottom, long upper wick
    _inv_hammer_shape = (
        (body_ratio <= 0.35) &
        (upper_wick >= 2 * body) &
        (lower_wick <= body * 0.5)
    )
    inv_hammer   = _inv_hammer_shape & bull_candle  # after downtrend (bullish)
    shooting_star = _inv_hammer_shape & bear_candle # after uptrend  (bearish)

    # Marubozu: nearly all body, tiny wicks
    bull_marubozu = bull_candle & (body_ratio >= 0.90)
    bear_marubozu = bear_candle & (body_ratio >= 0.90)

    # Spinning Top: small body, balanced long wicks
    spinning_top = (
        (body_ratio <= 0.30) &
        (upper_ratio >= 0.25) &
        (lower_ratio >= 0.25) &
        ~doji
    )
    bull_spinning_top = spinning_top & bull_candle
    bear_spinning_top = spinning_top & bear_candle

    # ── Two-candle patterns ──────────────────────────────────────────────────

    prev_o = o.shift(1)
    prev_c = c.shift(1)
    prev_bull = bull_candle.shift(1)
    prev_bear = bear_candle.shift(1)
    prev_body = body.shift(1)

    # Bullish Engulfing: big green candle fully engulfs previous red candle
    bull_engulfing = (
        bull_candle &
        prev_bear.fillna(False) &
        (o <= prev_c) &            # opens at or below prev close
        (c >= prev_o)              # closes at or above prev open
    )

    # Bearish Engulfing: big red candle fully engulfs previous green candle
    bear_engulfing = (
        bear_candle &
        prev_bull.fillna(False) &
        (o >= prev_c) &
        (c <= prev_o)
    )

    # Doji Star patterns (doji after a big candle)
    bullish_doji_star = doji & prev_bear.fillna(False) & (prev_body > body * 3)
    bearish_doji_star = doji & prev_bull.fillna(False) & (prev_body > body * 3)

    # ── Three-candle patterns ────────────────────────────────────────────────

    pp_o = o.shift(2)
    pp_c = c.shift(2)
    pp_bull = bull_candle.shift(2)
    pp_bear = bear_candle.shift(2)

    # Morning Star: big bear | small body gap | big bull
    morning_star = (
        bull_candle &
        prev_bear.fillna(False) &   # middle: small body (gap down)
        pp_bear.fillna(False) &     # first: big bear
        (body > pp_o.fillna(0) - pp_c.fillna(0)) &   # third closes back into first body
        (body_ratio.shift(1) <= 0.35)                 # middle has small body
    )

    # Evening Star: big bull | small body gap | big bear
    evening_star = (
        bear_candle &
        prev_bull.fillna(False) &
        pp_bull.fillna(False) &
        (body > pp_c.fillna(0) - pp_o.fillna(0)) &
        (body_ratio.shift(1) <= 0.35)
    )

    out = pd.DataFrame({
        # Bullish
        "cdl_hammer":          hammer,
        "cdl_inv_hammer":      inv_hammer,
        "cdl_bull_engulfing":  bull_engulfing,
        "cdl_morning_star":    morning_star,
        "cdl_bull_marubozu":   bull_marubozu,
        "cdl_bull_spinning":   bull_spinning_top,
        "cdl_bullish_doji":    bullish_doji_star,
        # Bearish
        "cdl_hanging_man":     hanging_man,
        "cdl_shooting_star":   shooting_star,
        "cdl_bear_engulfing":  bear_engulfing,
        "cdl_evening_star":    evening_star,
        "cdl_bear_marubozu":   bear_marubozu,
        "cdl_bear_spinning":   bear_spinning_top,
        "cdl_bearish_doji":    bearish_doji_star,
        "cdl_doji":            doji,
    }, index=df.index)

    # Summary columns
    bullish_cols = [c for c in out.columns if c.startswith("cdl_") and
                    any(x in c for x in ("bull","hammer","inv_hammer","morning","bullish_doji"))]
    bearish_cols = [c for c in out.columns if c.startswith("cdl_") and
                    any(x in c for x in ("bear","hanging","shooting","evening","bearish_doji"))]

    out["cdl_any_bull"] = out[bullish_cols].any(axis=1)
    out["cdl_any_bear"] = out[bearish_cols].any(axis=1)
    out["cdl_active"]   = out["cdl_any_bull"] | out["cdl_any_bear"] | out["cdl_doji"]

    # Human-readable name of the active pattern (last one wins if multiple)
    _name_map = {
        "cdl_hammer":         "Hammer",
        "cdl_inv_hammer":     "Inv Hammer",
        "cdl_bull_engulfing": "Bull Engulfing",
        "cdl_morning_star":   "Morning Star",
        "cdl_bull_marubozu":  "Bull Marubozu",
        "cdl_bull_spinning":  "Bull Spinning Top",
        "cdl_bullish_doji":   "Bullish Doji Star",
        "cdl_hanging_man":    "Hanging Man",
        "cdl_shooting_star":  "Shooting Star",
        "cdl_bear_engulfing": "Bear Engulfing",
        "cdl_evening_star":   "Evening Star",
        "cdl_bear_marubozu":  "Bear Marubozu",
        "cdl_bear_spinning":  "Bear Spinning Top",
        "cdl_bearish_doji":   "Bearish Doji Star",
        "cdl_doji":           "Doji",
    }
    def _pattern_name(row) -> str:
        names = [v for k, v in _name_map.items() if row.get(k, False)]
        return " + ".join(names) if names else ""

    out["cdl_name"] = out.apply(
        lambda row: _pattern_name(row), axis=1
    )

    return out


# ══════════════════════════════════════════════════════════════════════════════
# 4. TRADINGVIEW LIVE SIGNAL
# ══════════════════════════════════════════════════════════════════════════════

_TV_INTERVAL_MAP = {
    "1min":  "1m",
    "5min":  "5m",
    "15min": "15m",
    "60min": "1h",
    "daily": "1D",
    "1m":    "1m",
    "5m":    "5m",
    "15m":   "15m",
    "1h":    "1h",
    "1D":    "1D",
}

_TV_SYM_MAP = {
    "NSE:NIFTY50-INDEX":    "NIFTY",
    "NSE:NIFTYBANK-INDEX":  "BANKNIFTY",
    "NSE:FINNIFTY-INDEX":   "NIFTY_FIN_SERVICE",  # TradingView NSE symbol
    "NSE:MIDCPNIFTY-INDEX": "NIFTY_MID_SELECT",   # TradingView NSE symbol
}

def get_tradingview_signal(
    symbol: str,
    interval: str = "15min",
    screener: str = "india",
    exchange: str = "NSE",
    timeout: float = 5.0,
) -> dict:
    """
    Fetch TradingView's live technical analysis summary for a symbol.

    Returns a dict with:
      recommendation  str   — 'STRONG_BUY' / 'BUY' / 'NEUTRAL' / 'SELL' / 'STRONG_SELL'
      buy             int   — number of bullish indicators
      sell            int   — number of bearish indicators
      neutral         int   — number of neutral indicators
      rsi             float — RSI value from TradingView
      macd_signal     str   — 'BUY' / 'SELL' / 'NEUTRAL'
      stoch_signal    str   — 'BUY' / 'SELL' / 'NEUTRAL'
      adx             float
      indicators      dict  — full indicator dict from TradingView
      error           str   — error message if call failed (empty on success)

    Usage
    -----
    tv = get_tradingview_signal("NSE:NIFTY50-INDEX", "15min")
    print(tv["recommendation"])    # 'STRONG_BUY'
    print(tv["buy"], tv["sell"])   # 14  2

    Parameters
    ----------
    symbol    : Fyers symbol OR TradingView symbol (NIFTY, BANKNIFTY, etc.)
    interval  : '5min' / '15min' / '60min' / 'daily'
    screener  : 'india' (default for NSE)
    exchange  : 'NSE' (default)
    timeout   : request timeout in seconds
    """
    try:
        from tradingview_ta import TA_Handler, Interval
    except ImportError:
        return {"error": "tradingview-ta not installed — pip install tradingview-ta",
                "recommendation": "NEUTRAL", "buy": 0, "sell": 0, "neutral": 0}

    # Resolve symbol
    tv_sym = _TV_SYM_MAP.get(symbol, symbol)

    # Resolve interval
    tv_interval_str = _TV_INTERVAL_MAP.get(interval, interval)
    _interval_obj_map = {
        "1m":  Interval.INTERVAL_1_MINUTE,
        "5m":  Interval.INTERVAL_5_MINUTES,
        "15m": Interval.INTERVAL_15_MINUTES,
        "1h":  Interval.INTERVAL_1_HOUR,
        "1D":  Interval.INTERVAL_1_DAY,
    }
    tv_interval = _interval_obj_map.get(tv_interval_str, Interval.INTERVAL_15_MINUTES)

    try:
        handler = TA_Handler(
            symbol=tv_sym,
            screener=screener,
            exchange=exchange,
            interval=tv_interval,
            timeout=timeout,
        )
        analysis = handler.get_analysis()
        summ = analysis.summary
        inds = analysis.indicators

        def _ind_signal(key: str) -> str:
            """Convert a TradingView oscillator/ma value to BUY/SELL/NEUTRAL."""
            v = inds.get(key)
            if v is None: return "NEUTRAL"
            if isinstance(v, str): return v.upper()
            return "NEUTRAL"

        return {
            "recommendation": summ.get("RECOMMENDATION", "NEUTRAL"),
            "buy":            summ.get("BUY",    0),
            "sell":           summ.get("SELL",   0),
            "neutral":        summ.get("NEUTRAL", 0),
            "rsi":            inds.get("RSI", None),
            "macd_signal":    _ind_signal("MACD.macd"),
            "stoch_signal":   _ind_signal("Stoch.K"),
            "adx":            inds.get("ADX", None),
            "ema20":          inds.get("EMA20", None),
            "ema50":          inds.get("EMA50", None),
            "close":          inds.get("close", None),
            "volume":         inds.get("volume", None),
            "indicators":     inds,
            "error":          "",
        }

    except Exception as exc:
        return {
            "recommendation": "NEUTRAL",
            "buy": 0, "sell": 0, "neutral": 0,
            "indicators": {},
            "error": str(exc),
        }


# ══════════════════════════════════════════════════════════════════════════════
# 5. CONVENIENCE: add_all_indicators
# ══════════════════════════════════════════════════════════════════════════════

def add_all_indicators(df: pd.DataFrame,
                       supertrend_period: int = 10,
                       supertrend_mult: float = 3.0) -> pd.DataFrame:
    """
    Add all indicators from this module to a copy of df.

    Appends columns:
      atr, adx, adx_pos, adx_neg
      stoch_k, stoch_d
      cci, obv, mfi
      supertrend, supertrend_dir, supertrend_bull, supertrend_signal
      vwap, vwap_1u/1d/2u/2d        (only if 'ts' column present)
      ichi_tenkan, ichi_kijun, ...   (Ichimoku)
      cdl_*                          (candlestick patterns)

    Also preserves hand-rolled indicators already in df (EMA 9/21/50, MACD, RSI, VWAP).

    Returns a new DataFrame (does not modify input).
    """
    out = df.copy()

    # ta-based indicators
    if _TA_OK:
        try:
            out["atr"] = atr(df)
        except Exception: pass
        try:
            adx_df = adx(df)
            for col in adx_df.columns:
                out[col] = adx_df[col]
        except Exception: pass
        try:
            stoch_df = stochastic(df)
            for col in stoch_df.columns:
                out[col] = stoch_df[col]
        except Exception: pass
        try:
            out["cci"] = cci(df)
        except Exception: pass
        try:
            out["obv"] = obv(df)
        except Exception: pass
        try:
            out["mfi"] = mfi(df)
        except Exception: pass
        # Supertrend
        try:
            st_df = supertrend(df, period=supertrend_period, multiplier=supertrend_mult)
            for col in st_df.columns:
                out[col] = st_df[col]
        except Exception: pass

    # VWAP bands (if 'ts' present)
    if "ts" in df.columns:
        try:
            vb = vwap_bands(df)
            for col in vb.columns:
                out[col] = vb[col]
        except Exception: pass

    # Ichimoku
    try:
        ichi_df = ichimoku(df)
        for col in ichi_df.columns:
            out[col] = ichi_df[col]
    except Exception: pass

    # Candlestick patterns
    try:
        cdl_df = candlestick_patterns(df)
        for col in cdl_df.columns:
            out[col] = cdl_df[col]
    except Exception: pass

    return out


# ══════════════════════════════════════════════════════════════════════════════
# 6. QUICK SELF-TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    print(f"Python {sys.version.split()[0]}  |  ta={_TA_OK}\n")

    np.random.seed(0)
    n = 250
    prices = 23000 + np.cumsum(np.random.randn(n) * 12)
    df_test = pd.DataFrame({
        "ts":     pd.date_range("2026-06-01 09:15", periods=n, freq="5min"),
        "open":   prices + np.random.randn(n) * 4,
        "close":  prices + np.random.randn(n) * 4,
        "volume": np.random.randint(50_000, 2_000_000, n),
    })
    df_test["high"] = df_test[["open","close"]].max(axis=1) + np.abs(np.random.randn(n) * 15)
    df_test["low"]  = df_test[["open","close"]].min(axis=1) - np.abs(np.random.randn(n) * 15)

    print("── Supertrend ──────────────────────────────────────")
    st = supertrend(df_test)
    last = st.iloc[-1]
    print(f"  Value = {last['supertrend']:.2f}  Dir = {last['supertrend_dir']}  "
          f"Bull = {last['supertrend_bull']}  Signal = '{last['supertrend_signal']}'")

    print("\n── ATR / ADX / Stochastic / CCI ───────────────────")
    print(f"  ATR    = {atr(df_test).iloc[-1]:.2f}")
    adx_r = adx(df_test)
    print(f"  ADX    = {adx_r['adx'].iloc[-1]:.2f}  "
          f"+DI={adx_r['adx_pos'].iloc[-1]:.2f}  -DI={adx_r['adx_neg'].iloc[-1]:.2f}")
    stoch_r = stochastic(df_test)
    print(f"  Stoch  K={stoch_r['stoch_k'].iloc[-1]:.1f}  D={stoch_r['stoch_d'].iloc[-1]:.1f}")
    print(f"  CCI    = {cci(df_test).iloc[-1]:.2f}")
    print(f"  OBV    = {obv(df_test).iloc[-1]:,.0f}")
    print(f"  MFI    = {mfi(df_test).iloc[-1]:.1f}")

    print("\n── Ichimoku ────────────────────────────────────────")
    ichi = ichimoku(df_test)
    print(f"  Tenkan = {ichi['ichi_tenkan'].iloc[-1]:.2f}  "
          f"Kijun = {ichi['ichi_kijun'].iloc[-1]:.2f}  "
          f"Above cloud = {ichi['ichi_above_cloud'].iloc[-1]}")

    print("\n── VWAP Bands ──────────────────────────────────────")
    vb = vwap_bands(df_test)
    print(f"  VWAP  = {vb['vwap'].iloc[-1]:.2f}  "
          f"±1σ [{vb['vwap_1d'].iloc[-1]:.2f}, {vb['vwap_1u'].iloc[-1]:.2f}]")

    print("\n── Candlestick Patterns ────────────────────────────")
    cdl = candlestick_patterns(df_test)
    active = cdl[cdl["cdl_active"]]
    print(f"  {len(active)} patterns detected in {n} bars")
    print(f"  Last 5 active bars:")
    for _, row in active.tail(5).iterrows():
        print(f"    {row.name}  →  {row['cdl_name']}")

    print("\n── add_all_indicators ──────────────────────────────")
    enriched = add_all_indicators(df_test)
    ta_cols = [c for c in enriched.columns if c not in df_test.columns]
    print(f"  Added {len(ta_cols)} indicator columns")
    print(f"  Columns: {', '.join(ta_cols[:12])} ...")

    print("\n── TradingView (import check only — market closed) ─")
    tv = get_tradingview_signal("NSE:NIFTY50-INDEX", "15min")
    if tv["error"]:
        print(f"  Error (expected when market closed): {tv['error'][:80]}")
    else:
        print(f"  Recommendation: {tv['recommendation']}  "
              f"B:{tv['buy']} S:{tv['sell']} N:{tv['neutral']}")
    print("\nAll tests complete.")
