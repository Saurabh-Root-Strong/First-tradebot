"""
intraday_shock.py — Market Shock / Regime-Shift detection (signal Layer 10).

The velocity layer (trade_setup._velocity_layer) reads OI / wall / PCR *slopes*
over smooth 15–60 min windows. It catches trends, not shocks. This module
detects ABRUPT regime changes inside the last few minutes — the moments the
desk actually reacts to:

  • OI shock      — z-score of the LATEST OI delta vs the session's own pace.
                    A sudden burst of put-writing (bullish) / call-writing
                    (bearish), or a sudden unwind of either wall.
  • Volume surge  — latest 1-min candle volume vs the trailing median, paired
                    with price direction (surge + up = bullish thrust;
                    surge + down = bearish distribution).
  • Price impulse — 1-min price velocity vs recent ATR (a violent leg).

`analyze_shock()` fuses these into a directional score, an alert level, and a
list of human-readable signals. The recommendation engine consumes it as a
conviction MODULATOR / OVERRIDE (see trade_setup.apply_shock_override), and the
trade tracker uses `shock_against()` to flag & tighten open trades when a shock
fires against them.

Scale-free by design: OI uses z-scores relative to each index's own session
pace, so NIFTY and MIDCPNIFTY are judged on the same footing without per-index
magnitude tuning. A small absolute floor guards against dead-market noise.
"""

from __future__ import annotations

import datetime
import statistics
from typing import Optional

try:
    from intraday_store import oi_store, candle_store, IST
    _STORE_AVAILABLE = True
except Exception:                       # pragma: no cover
    _STORE_AVAILABLE = False
    from core.constants import IST   # single source of truth

try:
    from signals import _fmt_oi
except Exception:                       # pragma: no cover
    def _fmt_oi(v) -> str:
        return f"{v:,.0f}"


# ── Tunables ───────────────────────────────────────────────────────────────────
_OI_Z_THRESH      = 2.0       # z-score above which the latest OI delta is a shock
_OI_ABS_FLOOR     = 30_000    # min |latest delta| to count (kills dead-market noise)
_OI_RATIO_MIN     = 1.6       # latest |delta| must be ≥ this × median prior |delta|
_VOL_SURGE_MULT   = 2.0       # last-candle volume ≥ this × trailing median = surge
_VOL_STRONG_MULT  = 3.0       # ... and ≥ this is an extreme surge
_PRICE_ATR_MULT   = 1.5       # |1-min move| ≥ this × recent ATR = impulse
_MIN_OI_SNAPS     = 5         # need this many same-day snapshots for OI z-scores
_MIN_VOL_CANDLES  = 8         # need this many completed 1-min candles for volume


# ── Small helpers ──────────────────────────────────────────────────────────────
def _zscore(latest: float, prior: list[float]) -> float:
    """z of `latest` against the distribution of `prior` (sample stdev)."""
    if len(prior) < 2:
        return 0.0
    mu = statistics.fmean(prior)
    try:
        sd = statistics.stdev(prior)
    except statistics.StatisticsError:
        return 0.0
    if sd <= 0:
        return 0.0
    return (latest - mu) / sd


def _today_series(sym: str) -> list:
    """Same-day OI snapshots, oldest→newest."""
    try:
        series = oi_store.series(sym)
    except Exception:
        return []
    today = datetime.datetime.now(tz=IST).date()
    return [s for s in series if getattr(s, "ts", None) and s.ts.date() == today]


# ── 1. OI shock ────────────────────────────────────────────────────────────────
def detect_oi_shock(sym: str) -> dict:
    """
    Sudden acceleration in OI build/unwind on either leg, judged against the
    session's own pace via a z-score on the most recent snapshot-to-snapshot
    delta.

    Returns dict: has_shock, score [-2..+2], z, leg, kind, msg.
    Sign convention (score): + bullish, − bearish.
    """
    out = {"has_shock": False, "score": 0.0, "z": 0.0, "leg": "", "kind": "", "msg": ""}
    if not _STORE_AVAILABLE:
        return out
    series = _today_series(sym)
    if len(series) < _MIN_OI_SNAPS:
        return out

    def _deltas(attr: str) -> list[float]:
        vals = [getattr(s, attr, 0) or 0 for s in series]
        return [vals[i] - vals[i - 1] for i in range(1, len(vals))]

    put_d  = _deltas("total_put_oi")
    call_d = _deltas("total_call_oi")
    if len(put_d) < 3:
        return out

    cand = []   # (abs_z, signed_score, leg, kind, msg)
    for leg, deltas in (("PUT", put_d), ("CALL", call_d)):
        latest = deltas[-1]
        prior  = deltas[:-1]
        if abs(latest) < _OI_ABS_FLOOR:
            continue
        med_prior = statistics.median(abs(d) for d in prior) if prior else 0
        if med_prior and abs(latest) < _OI_RATIO_MIN * med_prior:
            continue
        z = _zscore(latest, prior)
        # Ratio fallback: a near-linear prior build gives stdev≈0 → z≈0 even when
        # the latest delta is a clear outlier. Treat a large ratio as an effective
        # shock so we never miss a burst just because the run-up was smooth.
        ratio = abs(latest) / med_prior if med_prior else 0
        eff_z = abs(z)
        if eff_z < _OI_Z_THRESH and ratio >= 3.0:
            eff_z = _OI_Z_THRESH * min(ratio / 3.0, 2.0)  # map ratio→pseudo-z
            z = eff_z if latest > 0 else -eff_z
        if eff_z < _OI_Z_THRESH:
            continue

        # Map (leg, build/unwind) → direction.
        #   PUT  build (+)  → floor forming      → bullish
        #   PUT  unwind (−) → floor pulled       → bearish
        #   CALL build (+)  → ceiling forming     → bearish
        #   CALL unwind (−) → ceiling pulled / SC → bullish
        building = latest > 0
        if leg == "PUT":
            bullish = building
            verb = "writing surge" if building else "unwind"
            why  = "floor forming fast" if building else "support being pulled — breakdown risk"
        else:  # CALL
            bullish = not building
            verb = "writing surge" if building else "unwind"
            why  = "ceiling capping fast" if building else "ceiling lifting — short covering"

        mag = min(abs(z) / _OI_Z_THRESH, 2.0)          # 1.0 at threshold → up to 2.0
        signed = (mag if bullish else -mag)
        z_disp = max(min(z, 9.9), -9.9)                 # clamp for display sanity
        msg = (f"OI SHOCK · {leg} {verb} {_fmt_oi(abs(latest))} in last snapshot "
               f"(z={z_disp:+.1f}) — {why}")
        cand.append((abs(z), round(signed, 2), leg, verb, msg))

    if not cand:
        return out
    cand.sort(reverse=True)                            # strongest |z| wins
    _, signed, leg, kind, msg = cand[0]
    out.update(has_shock=True, score=signed, z=cand[0][0], leg=leg, kind=kind, msg=msg)
    return out


# ── 2. Volume surge ────────────────────────────────────────────────────────────
def detect_volume_surge(sym: str) -> dict:
    """
    Relative-volume spike on the last completed 1-min candle vs the trailing
    median, paired with price direction over the last few minutes.

    Returns dict: has_surge, score [-2..+2], rel_vol, direction, msg.
    """
    out = {"has_surge": False, "score": 0.0, "rel_vol": 0.0, "direction": "", "msg": ""}
    if not _STORE_AVAILABLE:
        return out
    try:
        df = candle_store.to_dataframe(sym, "1min", include_forming=False)
    except Exception:
        return out
    if df is None or len(df) < _MIN_VOL_CANDLES:
        return out

    vols = df["volume"].tolist()
    last_vol  = vols[-1]
    prior     = vols[-_MIN_VOL_CANDLES:-1] or vols[:-1]
    med = statistics.median(v for v in prior if v >= 0) if prior else 0
    if med <= 0 or last_vol < _VOL_SURGE_MULT * med:
        return out

    rel = last_vol / med
    # Direction from the net move over the last 3 candles (robust vs a single wick).
    closes = df["close"].tolist()
    ref    = closes[-4] if len(closes) >= 4 else closes[0]
    net    = closes[-1] - ref
    if abs(net) < 1e-9:
        # Surge with no net move = churn/absorption — informative but non-directional.
        out.update(has_surge=True, score=0.0, rel_vol=round(rel, 1), direction="flat",
                   msg=(f"VOLUME SURGE · {rel:.1f}× median on flat price — "
                        f"absorption/churn, await resolution"))
        return out

    bullish = net > 0
    strong  = rel >= _VOL_STRONG_MULT
    mag     = min(rel / _VOL_SURGE_MULT, 2.0)          # 1.0 at surge → up to 2.0
    signed  = (mag if bullish else -mag)
    arrow   = "↑" if bullish else "↓"
    tag     = "EXTREME" if strong else "surge"
    out.update(
        has_surge=True, score=round(signed, 2), rel_vol=round(rel, 1),
        direction="up" if bullish else "down",
        msg=(f"VOLUME {tag} · {rel:.1f}× median vol with price {arrow}{abs(net):.0f}pt "
             f"(3m) — {'demand thrust' if bullish else 'supply hitting'}"),
    )
    return out


# ── 3. Price impulse ───────────────────────────────────────────────────────────
def detect_price_impulse(sym: str) -> dict:
    """
    A violent 1-min leg: net move over the last ~2 candles vs recent ATR.

    Returns dict: has_impulse, score [-1.5..+1.5], direction, msg.
    """
    out = {"has_impulse": False, "score": 0.0, "direction": "", "msg": ""}
    if not _STORE_AVAILABLE:
        return out
    try:
        df = candle_store.to_dataframe(sym, "1min", include_forming=False)
    except Exception:
        return out
    if df is None or len(df) < _MIN_VOL_CANDLES:
        return out

    highs = df["high"].tolist(); lows = df["low"].tolist(); closes = df["close"].tolist()
    rngs  = [h - l for h, l in zip(highs[-10:], lows[-10:]) if h >= l]
    atr   = statistics.fmean(rngs) if rngs else 0
    if atr <= 0:
        return out
    net = closes[-1] - (closes[-3] if len(closes) >= 3 else closes[0])
    if abs(net) < _PRICE_ATR_MULT * atr:
        return out

    bullish = net > 0
    mag = min(abs(net) / atr / _PRICE_ATR_MULT, 1.5)
    out.update(
        has_impulse=True, score=round(mag if bullish else -mag, 2),
        direction="up" if bullish else "down",
        msg=(f"PRICE IMPULSE · {abs(net):.0f}pt in 2m ({abs(net)/atr:.1f}× ATR) "
             f"{'up' if bullish else 'down'} — momentum leg"),
    )
    return out


# ── Fusion ─────────────────────────────────────────────────────────────────────
# Sub-detector weights within the shock layer (sum need not be 1; score is clamped).
_W_OI, _W_VOL, _W_PRICE = 1.0, 0.8, 0.6


def analyze_shock(sym: str) -> dict:
    """
    Fuse the three detectors into one regime read.

    Returns:
      {
        "level":     "none" | "watch" | "alert",
        "score":     float [-3, +3]   (+ bullish, − bearish),
        "direction": "CE" | "PE" | "",
        "signals":   [(tag, msg), ...],   tag ∈ bull/bear/neut
        "oi","vol","price": the raw sub-detector dicts,
      }
    """
    oi    = detect_oi_shock(sym)
    vol   = detect_volume_surge(sym)
    price = detect_price_impulse(sym)

    raw = (oi["score"] * _W_OI) + (vol["score"] * _W_VOL) + (price["score"] * _W_PRICE)
    score = round(min(max(raw, -3.0), 3.0), 2)

    fired = sum(1 for d, k in ((oi, "has_shock"), (vol, "has_surge"),
                               (price, "has_impulse")) if d.get(k))
    extreme = (oi.get("z", 0) >= 3.0) or (vol.get("rel_vol", 0) >= _VOL_STRONG_MULT)

    if fired >= 2 or (fired >= 1 and extreme):
        level = "alert"
    elif fired == 1:
        level = "watch"
    else:
        level = "none"

    signals: list[tuple[str, str]] = []
    for d, key in ((oi, "has_shock"), (vol, "has_surge"), (price, "has_impulse")):
        if d.get(key) and d.get("msg"):
            s = d.get("score", 0)
            tag = "bull" if s > 0 else "bear" if s < 0 else "neut"
            signals.append((tag, d["msg"]))

    direction = "CE" if score > 0 else "PE" if score < 0 else ""
    return {
        "level": level, "score": score, "direction": direction,
        "signals": signals, "oi": oi, "vol": vol, "price": price,
        "fired": fired,
    }


def shock_against(sym: str, trade_direction: str) -> Optional[dict]:
    """
    For the trade tracker: return the shock read ONLY when an ALERT-level shock
    opposes an open trade's direction (long-CE hit by a bearish shock, or
    long-PE hit by a bullish shock). Returns None otherwise.
    """
    if trade_direction not in ("CE", "PE"):
        return None
    sh = analyze_shock(sym)
    if sh["level"] != "alert" or not sh["direction"]:
        return None
    if sh["direction"] != trade_direction:        # shock points opposite the trade
        return sh
    return None
