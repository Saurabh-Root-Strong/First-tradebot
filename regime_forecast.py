"""
regime_forecast.py — Forward-looking regime-change forecast.

Where intraday_shock.py detects a regime shift *after* it fires, this module
tries to anticipate one: it reads the drift of the structural factors the desk
watches — OI flow, ATM premium, PCR, futures basis, and volume/price — and asks
"is the current regime weakening, and if so, how soon does it flip?"

Method (honest, bucketed — no fake single-minute precision):
  • For each index, compute a directional read over a LONG window (~60 min, the
    established regime) and a SHORT window (~20 min, the freshest drift).
  • When the short read diverges from the long one, the regime is weakening;
    the size of that divergence is the "flip pressure".
  • Flip pressure → a stage (STABLE / BUILDING / IMMINENT) and an ETA bucket.
  • Confidence rises with pressure magnitude, volume confirmation, and — for the
    market-wide read — how many of the 4 indices are drifting coherently.

Everything reads the lock-free Parquet snapshot mirrors in data/intraday/live/
(refreshed ~every 10s by the dashboard), so it is fully decoupled from the live
in-memory stores and supports `as_of` reconstruction for the 30-min time-machine.

All functions degrade gracefully to {"has_data": False} when history is thin.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Optional

try:
    import pandas as pd
    _PANDAS = True
except Exception:                       # pragma: no cover
    _PANDAS = False

from core.constants import IST   # single source of truth  # noqa: E402
_LIVE_DIR = Path(__file__).parent / "data" / "intraday" / "live"

INDEX_SYMBOLS = [
    "NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX",
    "NSE:FINNIFTY-INDEX", "NSE:MIDCPNIFTY-INDEX",
]

# Tunables
_LONG_MIN   = 60     # established-regime window
_SHORT_MIN  = 20     # fresh-drift window
_MIN_SNAPS  = 4      # need at least this many snapshots in the session
_OI_SCALE   = 300_000   # OI delta that maps to a "full" directional unit
_PREM_SCALE = 12.0      # ATM premium delta (pts) for a full unit
_PCR_SCALE  = 0.12      # PCR delta for a full unit


# ── Parquet readers (lock-free) ────────────────────────────────────────────────
def _today() -> str:
    return datetime.datetime.now(tz=IST).date().isoformat()


def _read(tbl: str, date: Optional[str] = None) -> "pd.DataFrame | None":
    if not _PANDAS:
        return None
    p = _LIVE_DIR / f"{date or _today()}_{tbl}.parquet"
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p)
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        return df
    except Exception:
        return None


def _snaps(sym: str, as_of: Optional[datetime.datetime],
           date: Optional[str] = None) -> "pd.DataFrame | None":
    df = _read("oi_snapshots", date)
    if df is None or df.empty:
        return None
    df = df[df["symbol"] == sym].sort_values("ts")
    if as_of is not None:
        cutoff = as_of.astimezone(datetime.timezone.utc)
        df = df[df["ts"] <= cutoff]
    return df if len(df) >= _MIN_SNAPS else None


def _nearest_before(df: "pd.DataFrame", ref_ts, minutes: int):
    """Row nearest to `minutes` before the last ts (≥ that target if possible)."""
    target = ref_ts - pd.Timedelta(minutes=minutes)
    older = df[df["ts"] <= target]
    return older.iloc[-1] if len(older) else df.iloc[0]


# ── Directional read over a window ─────────────────────────────────────────────
def _clip(x: float, lo=-1.0, hi=1.0) -> float:
    return max(lo, min(hi, x))


def _window_read(df: "pd.DataFrame", minutes: int) -> tuple[float, list]:
    """
    Directional structure score over the last `minutes`, in [-1, +1].
    + bullish (put-writing / PCR up / call-premium up), − bearish.
    Returns (score, factor_messages).
    """
    now = df.iloc[-1]
    ref = _nearest_before(df, df["ts"].iloc[-1], minutes)

    d_put_oi  = (now["total_put_oi"]  or 0) - (ref["total_put_oi"]  or 0)
    d_call_oi = (now["total_call_oi"] or 0) - (ref["total_call_oi"] or 0)
    d_pcr     = (now["pcr"]           or 0) - (ref["pcr"]           or 0)
    d_cprem   = (now["atm_call_prem"] or 0) - (ref["atm_call_prem"] or 0)
    d_pprem   = (now["atm_put_prem"]  or 0) - (ref["atm_put_prem"]  or 0)

    oi_u   = _clip((d_put_oi - d_call_oi) / _OI_SCALE)       # net writing pressure
    pcr_u  = _clip(d_pcr / _PCR_SCALE)
    prem_u = _clip((d_cprem - d_pprem) / _PREM_SCALE)        # call vs put premium drift

    score = _clip(0.45 * oi_u + 0.30 * pcr_u + 0.25 * prem_u)

    f: list = []
    if abs(oi_u) > 0.25:
        f.append(("bull" if oi_u > 0 else "bear",
                  f"OI {'put-writing' if oi_u>0 else 'call-writing'} over {minutes}m"))
    if abs(pcr_u) > 0.25:
        f.append(("bull" if pcr_u > 0 else "bear",
                  f"PCR {'rising' if pcr_u>0 else 'falling'} {d_pcr:+.2f} ({minutes}m)"))
    if abs(prem_u) > 0.25:
        f.append(("bull" if prem_u > 0 else "bear",
                  f"{'Call' if prem_u>0 else 'Put'} premium leading ({minutes}m)"))
    return round(score, 3), f


def _vol_confirm(sym: str, as_of: Optional[datetime.datetime],
                 date: Optional[str] = None) -> float:
    """Recent 1-min price direction as a light confirmation factor, in [-0.3, 0.3]."""
    df = _read("candles", date)
    if df is None or df.empty or "resolution" not in df.columns:
        return 0.0
    df = df[(df["symbol"] == sym) & (df["resolution"] == "1min")].sort_values("ts")
    if as_of is not None:
        df = df[df["ts"] <= as_of.astimezone(datetime.timezone.utc)]
    if len(df) < 4:
        return 0.0
    closes = df["close"].tolist()
    net = closes[-1] - closes[-4]
    return _clip(net / max(abs(closes[-1]) * 0.001, 1e-6) / 100, -0.3, 0.3)


# ── Stage / ETA mapping ────────────────────────────────────────────────────────
def _stage_eta(pressure: float) -> tuple[str, str]:
    p = abs(pressure)
    if   p >= 0.90: return "IMMINENT", "<10 min"
    elif p >= 0.60: return "BUILDING", "~10–20 min"
    elif p >= 0.35: return "BUILDING", "~20–40 min"
    else:           return "STABLE",   ">40 min"


def _label(score: float) -> str:
    return "BULLISH" if score > 0.15 else "BEARISH" if score < -0.15 else "NEUTRAL"


# ── Per-index forecast ─────────────────────────────────────────────────────────
def forecast_index(sym: str, as_of: Optional[datetime.datetime] = None,
                   date: Optional[str] = None) -> dict:
    """
    Regime + change-forecast for one index.  `date` (ISO) replays a past captured
    session; `as_of` cuts the read off at a moment within it (lookahead-free).

    Returns dict with: has_data, regime, bias, stage, eta, next_dir, confidence,
    factors, asof.
    """
    out = {"sym": sym, "has_data": False, "asof": as_of.isoformat() if as_of else None}
    df = _snaps(sym, as_of, date)
    if df is None:
        return out

    long_s,  long_f  = _window_read(df, _LONG_MIN)
    short_s, short_f = _window_read(df, _SHORT_MIN)
    vol = _vol_confirm(sym, as_of, date)
    short_s = _clip(short_s + vol)

    regime = _label(long_s)
    # Flip pressure: the short read opposing the established long read.
    opposing = (long_s * short_s) < 0
    if opposing:
        pressure = min(abs(short_s) + abs(short_s - long_s) * 0.5, 1.0)
        next_dir = _label(short_s)
    else:
        # Same direction — regime persisting; pressure only if the trend is
        # decelerating hard (short much weaker than long).
        decel = max(0.0, abs(long_s) - abs(short_s))
        pressure = min(decel, 0.5) if abs(long_s) > 0.3 else 0.0
        next_dir = "NEUTRAL" if pressure >= 0.35 else ""

    stage, eta = _stage_eta(pressure)
    if stage == "STABLE":
        next_dir, eta = "", ">40 min"

    # Confidence: pressure magnitude + volume agreement, 40–85.
    conf = int(_clip(40 + pressure * 45 + abs(vol) * 30, 0, 85))

    factors = (short_f or long_f)[:3]
    return {
        "sym": sym, "has_data": True, "regime": regime, "bias": long_s,
        "short": short_s, "stage": stage, "eta": eta, "next_dir": next_dir,
        "confidence": conf, "pressure": round(pressure, 2),
        "factors": factors, "asof": out["asof"],
    }


# ── Market-wide forecast ───────────────────────────────────────────────────────
def market_forecast(as_of: Optional[datetime.datetime] = None) -> dict:
    """
    Aggregate regime forecast across all four indices. Coherent drift across
    indices is the strongest tell, so confidence scales with agreement.
    """
    per = {s: forecast_index(s, as_of) for s in INDEX_SYMBOLS}
    live = [f for f in per.values() if f.get("has_data")]
    if not live:
        return {"has_data": False, "per_index": per,
                "asof": as_of.isoformat() if as_of else None}

    building = [f for f in live if f["stage"] in ("BUILDING", "IMMINENT")]
    # Coherent flip direction among those that are flipping.
    dirs = [f["next_dir"] for f in building if f["next_dir"] in ("BULLISH", "BEARISH")]
    coherent_dir, coherent_n = "", 0
    for d in ("BULLISH", "BEARISH"):
        n = dirs.count(d)
        if n > coherent_n:
            coherent_dir, coherent_n = d, n

    pressure = sum(f["pressure"] for f in live) / len(live)
    # Coherence boost: more indices pointing the same way → higher pressure/conf.
    pressure = min(pressure * (1 + 0.25 * max(0, coherent_n - 1)), 1.0)
    stage, eta = _stage_eta(pressure)
    base_regime = _label(sum(f["bias"] for f in live) / len(live))
    conf = int(min(45 + pressure * 40 + (coherent_n - 1) * 8, 88)) if building else \
           int(40 + pressure * 30)

    return {
        "has_data": True, "stage": stage, "eta": eta if stage != "STABLE" else ">40 min",
        "regime": base_regime, "next_dir": coherent_dir if stage != "STABLE" else "",
        "confidence": conf, "pressure": round(pressure, 2),
        "coherence": f"{coherent_n}/4 indices" if coherent_n else "mixed",
        "building_n": len(building), "per_index": per,
        "asof": as_of.isoformat() if as_of else None,
    }


# ── Per-trade regime risk (for old/open trades) ───────────────────────────────
def trade_regime_risk(trade: dict, fc: Optional[dict] = None) -> Optional[dict]:
    """
    For an OPEN trade, return a regime-risk read ONLY when the forecast points to
    a regime change AGAINST the trade's thesis (CE=bullish, PE=bearish) that is
    BUILDING or IMMINENT. Returns None otherwise.

    `fc` may be a pre-computed forecast_index() result for this index (lets the
    caller compute the 4 index forecasts once instead of per-card).
    """
    direction = trade.get("direction")
    if direction not in ("CE", "PE"):
        return None
    if fc is None:
        fc = forecast_index(trade.get("index_sym", ""))
    if not fc.get("has_data") or fc["stage"] == "STABLE":
        return None

    thesis_dir = "BULLISH" if direction == "CE" else "BEARISH"
    against = (fc["next_dir"] and fc["next_dir"] != thesis_dir) or \
              (not fc["next_dir"] and fc["regime"] and fc["regime"] != thesis_dir)
    if not against:
        return None

    # Trade age (older trades on a turning regime are the real worry).
    age_min = None
    try:
        opened = datetime.datetime.fromisoformat(trade["opened_ts"])
        age_min = int((datetime.datetime.now(tz=IST) - opened).total_seconds() / 60)
    except Exception:
        pass

    return {
        "eta": fc["eta"], "confidence": fc["confidence"], "stage": fc["stage"],
        "next_dir": fc["next_dir"] or fc["regime"], "factors": fc["factors"],
        "age_min": age_min,
        "msg": (f"⏳ REGIME RISK — {fc['stage']} shift to {fc['next_dir'] or fc['regime']} "
                f"in {fc['eta']} ({fc['confidence']}%) vs this {direction}"
                + (f"; trade {age_min}m old" if age_min is not None else "")),
    }


# ── 30-minute checkpoint times (for the time-machine dropdown) ────────────────
def checkpoint_times() -> list[datetime.datetime]:
    """Half-hour marks from 09:30 up to the latest passed mark (capped at 15:30)."""
    now = datetime.datetime.now(tz=IST)
    marks = []
    t = now.replace(hour=9, minute=30, second=0, microsecond=0)
    end = now.replace(hour=15, minute=30, second=0, microsecond=0)
    while t <= now and t <= end:
        marks.append(t)
        t = t + datetime.timedelta(minutes=30)
    return marks
