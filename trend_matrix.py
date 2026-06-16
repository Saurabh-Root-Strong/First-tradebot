"""
trend_matrix.py — multi-timeframe TREND ribbon (intraday → positional).

Not a prediction tool (the directional signal has no cost-survivable edge) — a CONTEXT
tool. Its value is alignment vs divergence across horizons:

    5m up + daily up   → aligned, ride with the trend
    5m up + daily down → counter-trend bounce, SCALP ONLY, don't hold

So it spans 5m / 15m / 1h / daily / weekly and issues one alignment verdict. The
higher-TF (daily/weekly) structure is the piece the intraday panels lack — and it's
what tells you whether to trust an intraday move.

Trend per TF (structural, not oscillator):
    direction = EMA stacking  (9>21>50 up, 9<21<50 down)
    strength  = ATR-normalised slope of EMA21 over the last ~10 bars
    → STRONG UP / UP / SIDEWAYS / DOWN / STRONG DOWN

  .venv\\Scripts\\python.exe trend_matrix.py
"""
from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

# Intraday TFs via live fetch (label, fyers_resolution, days). Daily/weekly come from
# the downloaded 4-yr history (Fyers caps a daily call ~250 bars — too few for weekly).
TFS = [("5m", "5", 5), ("15m", "15", 15), ("1h", "60", 60)]
from core.constants import (HIST_DIR, INDEX_SYMBOLS, LABELS,           # noqa: E402
                            LABEL_TO_SYM, safe_sym as _safe)
from core.ta import ema as _ema, atr as _atr                          # noqa: E402

HIST_DAILY = HIST_DIR / "daily"
_STRONG = 0.9      # ATR-normalised slope ≥ this → strong trend
_TREND  = 0.35     # below this (and EMAs not stacked) → sideways


def trend_from_df(df: pd.DataFrame, min_bars: int = 55) -> dict:
    """Structural trend: EMA stacking for direction, ATR-normalised slope for strength."""
    if df is None or len(df) < min_bars:
        return {"dir": 0, "strength": 0.0, "label": "—"}
    c = df["close"].astype(float)
    e9, e21, e50 = _ema(c, 9), _ema(c, 21), _ema(c, 50)
    last = float(c.iloc[-1])
    atrp = float(_atr(df).iloc[-1]) / last * 100 if last else 1.0
    n = min(10, len(c) - 2)
    slope = (float(e21.iloc[-1]) - float(e21.iloc[-1 - n])) / float(e21.iloc[-1 - n]) * 100
    strength = abs(slope) / max(atrp, 1e-6)             # slope measured in ATRs

    up = e9.iloc[-1] > e21.iloc[-1] > e50.iloc[-1]
    dn = e9.iloc[-1] < e21.iloc[-1] < e50.iloc[-1]
    if up:   d = 1
    elif dn: d = -1
    else:    d = (1 if last > e50.iloc[-1] else -1) if strength >= _TREND else 0

    if d == 0 or (not up and not dn and strength < _TREND):
        d, label = 0, "SIDE"
    else:
        strong = strength >= _STRONG and (up or dn)
        label = ("▲▲ STRONG" if d > 0 else "▼▼ STRONG") if strong else ("▲ UP" if d > 0 else "▼ DN")
    return {"dir": d, "strength": round(strength, 2), "label": label, "slope": round(slope, 3)}


def _load_daily(sym: str, fetch) -> pd.DataFrame | None:
    """4-yr daily history (downloaded parquet) + today's live daily bar, deduped by date.
    Gives enough bars for both a daily and a weekly 50-EMA."""
    frames = []
    p = HIST_DAILY / f"{_safe(sym)}_daily.parquet"
    if p.exists():
        b = pd.read_parquet(p)
        b["ts"] = pd.to_datetime(b["ts"]).dt.tz_localize(None)
        frames.append(b[["ts", "open", "high", "low", "close", "volume"]])
    live = fetch(sym, "D", 250)
    if live is not None and len(live):
        l = live.copy()
        l["ts"] = pd.to_datetime(l["ts"]).dt.tz_localize(None)
        frames.append(l[["ts", "open", "high", "low", "close", "volume"]])
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df["ts"] = df["ts"].dt.normalize()
    return df.drop_duplicates(subset=["ts"], keep="last").sort_values("ts").reset_index(drop=True)


def _to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.copy()
    d["ts"] = pd.to_datetime(d["ts"])
    w = d.set_index("ts").resample("W-FRI").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    return w.reset_index()


def trend_index(sym: str, fetch) -> dict:
    """Trend per TF (5m/15m/1h/daily) + weekly (resampled), and an alignment verdict.
    `fetch(sym, resolution, days) -> OHLCV df` is injected (signals.fetch_ohlcv live)."""
    ribbon = []
    for key, res, days in TFS:
        ribbon.append((key, trend_from_df(fetch(sym, res, days))))
    daily_df = _load_daily(sym, fetch)
    ribbon.append(("1D", trend_from_df(daily_df)))
    if daily_df is not None and len(daily_df):
        ribbon.append(("1W", trend_from_df(_to_weekly(daily_df), min_bars=30)))

    dirs = {k: t["dir"] for k, t in ribbon}
    live = [d for d in dirs.values() if d != 0]
    up = sum(1 for d in live if d > 0)
    dn = sum(1 for d in live if d < 0)
    short = [dirs.get(k, 0) for k in ("5m", "15m")]
    higher = dirs.get("1D", 0) or dirs.get("1h", 0)
    s_sign = short[0] if short and all(s == short[0] and s != 0 for s in short) else 0

    if up and not dn:
        verdict, vclr = "ALIGNED UP — trend with you", "#22c55e"
    elif dn and not up:
        verdict, vclr = "ALIGNED DOWN — trend with you", "#ef4444"
    elif s_sign and higher and s_sign != higher:
        kind = "bounce" if higher < 0 else "pullback"
        verdict = f"COUNTER-TREND {kind} — {'up' if s_sign>0 else 'down'} intraday vs {'down' if higher<0 else 'up'} daily — SCALP ONLY"
        vclr = "#f59e0b"
    else:
        verdict, vclr = "MIXED / transition — no clean trend", "#94a3b8"
    return {"sym": sym, "label": LABELS.get(sym, sym), "ribbon": ribbon,
            "verdict": verdict, "vclr": vclr, "up": up, "down": dn}


def main() -> None:
    import signals
    syms = (sys.argv[1:] and [LABEL_TO_SYM.get(a.upper(), a) for a in sys.argv[1:]]) or INDEX_SYMBOLS
    print("=" * 72)
    print("  MULTI-TIMEFRAME TREND  (5m · 15m · 1h · daily · weekly)")
    print("=" * 72)
    for sym in syms:
        r = trend_index(sym, signals.fetch_ohlcv)
        cells = "  ".join(f"{k}:{t['label']:<9}" for k, t in r["ribbon"])
        print(f"\n  {r['label']:<13}")
        print(f"    {cells}")
        print(f"    └─ {r['verdict']}")
    print()


if __name__ == "__main__":
    main()
