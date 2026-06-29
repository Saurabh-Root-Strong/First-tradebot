"""
price_structure.py — price-action / market-structure read off the index bars.

The scout (intraday_scout) is a pure OI-flow + futures engine — it has NO sense of
WHERE price sits in its own structure. backtest_scout MEASURED the consequence: buying
the directional arrow BLEEDS −2..−5% net/trade because the flow lags price and the arrow
fires you INTO resistance / into the chop, where intraday mean-reversion stops you out
(the multiple SL hits you watched). The OI book says "demand building" only after the
move already ran.

This module supplies the orthogonal family the scout was missing — support/resistance
from swing pivots, ATR, a consolidation (coil) detector, and a breakout flag — read off
the SAME footprint_chart.build_series bars the charts plot, with as_of so it is
lookahead-free and replays identically (parity by construction).

It does NOT emit a directional call (price-action direction alone is its own un-validated
problem). Its single job is a VETO / confluence gate on the scout's existing arrow:

  • CE arrow fired with tiny headroom under resistance  → the classic stop-out → VETO
  • PE arrow fired just above support                   → VETO
  • inside a coil with no breakout                      → the whipsaw zone → VETO
  • breakout in the arrow's direction                   → CONFLUENCE (do NOT veto)

DISPLAY / VETO-MEASURED ONLY until backtest_scout shows the veto cuts the option-P&L
bleed OUT of sample. Same discipline that killed the crossover / target estimators:
structure does not move the live arrow until it grades positive. The live wiring is
behind intraday_scout._STRUCT_VETO (default OFF).
"""
from __future__ import annotations

import math
from typing import Optional

# ── tunables (one source; mirrored from scout's style) ───────────────────────────────
_ATR_N = 14                # bars for ATR (uses what's available early in the session)
_PIVOT_K = 2               # fractal half-width: a swing needs K closed bars each side
_COIL_WIN = 6              # bars in the consolidation window
_COIL_MAX = 2.0            # window range < this × ATR = coiled / consolidating
_BREAK_LOOK = 6            # prior closed bars whose extreme a breakout must clear
_NEAR_ATR = 0.35           # within this × ATR of a level = "right at" it (no headroom)
_MIN_BARS = 6              # below this many closed bars structure is unreliable


def _clean(seq) -> list[float]:
    return [float(v) for v in (seq or []) if v is not None and not _isnan(v)]


def _isnan(v) -> bool:
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return True


def _atr(highs, lows, closes, n: int = _ATR_N) -> Optional[float]:
    """Wilder true-range average over the last n CLOSED bars. None if too few bars."""
    if len(closes) < 2:
        return None
    trs = []
    for i in range(1, len(closes)):
        h, l, pc = highs[i], lows[i], closes[i - 1]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    if not trs:
        return None
    w = trs[-n:]
    return sum(w) / len(w)


def _swings(highs, lows, k: int = _PIVOT_K) -> tuple[list[float], list[float]]:
    """Confirmed swing highs/lows via a simple K-bar fractal. A bar is a swing high
    if its high is the max of the [i-k, i+k] window (needs k bars each side, so the
    forming/last few bars can't be swings — that is what makes it lookahead-free)."""
    sh, sl = [], []
    n = len(highs)
    for i in range(k, n - k):
        win_h = highs[i - k:i + k + 1]
        win_l = lows[i - k:i + k + 1]
        if highs[i] >= max(win_h):
            sh.append(highs[i])
        if lows[i] <= min(win_l):
            sl.append(lows[i])
    return sh, sl


def analyze(ser: dict) -> Optional[dict]:
    """Structure read at instant t from build_series bars. Lookahead-free: ser was
    built with ts<=as_of and we drop the forming (last) bar before reading structure.
    Returns None when there are too few closed bars to trust."""
    highs = _clean(ser.get("high"))
    lows = _clean(ser.get("low"))
    closes = _clean(ser.get("close"))
    n = min(len(highs), len(lows), len(closes))
    if n < _MIN_BARS + 1:
        return None
    highs, lows, closes = highs[:n], lows[:n], closes[:n]
    # drop the last (forming) bar from STRUCTURE; price = its latest close (live level)
    price = closes[-1]
    chighs, clows, ccloses = highs[:-1], lows[:-1], closes[:-1]
    if len(ccloses) < _MIN_BARS:
        return None

    atr = _atr(chighs, clows, ccloses)
    if not atr or atr <= 0:
        return None

    sh, sl = _swings(chighs, clows)
    res_levels = sorted(v for v in sh if v > price)          # resistance ABOVE price
    sup_levels = sorted((v for v in sl if v < price), reverse=True)  # support BELOW
    nearest_res = res_levels[0] if res_levels else None
    nearest_sup = sup_levels[0] if sup_levels else None
    dist_res_atr = (nearest_res - price) / atr if nearest_res else None
    dist_sup_atr = (price - nearest_sup) / atr if nearest_sup else None

    # ── consolidation: range of the last _COIL_WIN closed bars vs ATR ────────────────
    win_h = chighs[-_COIL_WIN:]
    win_l = clows[-_COIL_WIN:]
    rng = (max(win_h) - min(win_l)) if win_h and win_l else None
    coil_ratio = (rng / atr) if (rng is not None and atr) else None
    consolidating = bool(coil_ratio is not None and coil_ratio < _COIL_MAX)

    # ── breakout: last CLOSED bar clears the prior-window extreme WITH range expansion ─
    breakout = None
    if len(ccloses) >= _BREAK_LOOK + 1:
        prior_h = max(chighs[-_BREAK_LOOK - 1:-1])
        prior_l = min(clows[-_BREAK_LOOK - 1:-1])
        last_tr = chighs[-1] - clows[-1]
        expanded = last_tr > atr
        if ccloses[-1] > prior_h and expanded:
            breakout = "UP"
        elif ccloses[-1] < prior_l and expanded:
            breakout = "DOWN"

    return {
        "price": round(price, 2),
        "atr": round(atr, 2),
        "atr_pct": round(atr / price * 100.0, 3) if price else None,
        "nearest_res": round(nearest_res, 2) if nearest_res else None,
        "nearest_sup": round(nearest_sup, 2) if nearest_sup else None,
        "dist_res_atr": round(dist_res_atr, 2) if dist_res_atr is not None else None,
        "dist_sup_atr": round(dist_sup_atr, 2) if dist_sup_atr is not None else None,
        "consolidating": consolidating,
        "coil_ratio": round(coil_ratio, 2) if coil_ratio is not None else None,
        "breakout": breakout,
        "n_res": len(res_levels), "n_sup": len(sup_levels),
    }


def _ema(vals: list[float], span: int) -> list[float]:
    """Simple EMA over vals (oldest→newest). [] if empty."""
    if not vals:
        return []
    a = 2.0 / (span + 1.0)
    out = [vals[0]]
    for v in vals[1:]:
        out.append(a * v + (1 - a) * out[-1])
    return out


def regime(ser: dict, span: int = 5, slope_bars: int = 3) -> dict:
    """Intraday DAY TREND off the index bars — the orthogonal family the SL audit
    (2026-06-29: 32/34 trades were CE on a DOWN day) said was missing. Two agreeing
    votes, robust to a single noisy bar:
      • LEVEL  : last close vs the day reference (first bar's open ≈ day open)
      • SLOPE  : sign of a short EMA's recent slope (momentum)
    UP only if BOTH agree up, DOWN only if both agree down, else NEUTRAL (chop/turn).
    Lookahead-free — uses only the bars build_series gave for ts<=as_of."""
    closes = _clean(ser.get("close"))
    opens = _clean(ser.get("open"))
    if len(closes) < slope_bars + 2:
        return {"trend": "NEUTRAL", "ref": None, "ema_slope": None, "votes": 0}
    price = closes[-1]
    ref = opens[0] if opens else closes[0]
    ema = _ema(closes, span)
    j = max(0, len(ema) - 1 - slope_bars)
    slope = ema[-1] - ema[j]
    v_level = 1 if price > ref else -1 if price < ref else 0
    v_slope = 1 if slope > 0 else -1 if slope < 0 else 0
    votes = v_level + v_slope
    trend = "UP" if votes >= 2 else "DOWN" if votes <= -2 else "NEUTRAL"
    return {"trend": trend, "ref": round(ref, 2), "price": round(price, 2),
            "ema_slope": round(slope, 2), "votes": votes,
            "day_pct": round((price / ref - 1.0) * 100.0, 2) if ref else None}


def trend_veto(reg: Optional[dict], direction: str) -> tuple[bool, str]:
    """Don't fight the day trend: veto a CE (long) in a confirmed DOWN day, a PE
    (short) in a confirmed UP day. NEUTRAL never vetoes (let the other gates rule).
    Pure read — caller decides whether to ACT (intraday_scout._TREND_VETO)."""
    if not reg or direction not in ("CE", "PE"):
        return False, ""
    t = reg.get("trend")
    if direction == "CE" and t == "DOWN":
        return True, (f"VETO long: day trend DOWN ({reg.get('day_pct')}% vs open, "
                      f"EMA slope {reg.get('ema_slope')}) — buying calls into a falling tape")
    if direction == "PE" and t == "UP":
        return True, (f"VETO short: day trend UP ({reg.get('day_pct')}% vs open, "
                      f"EMA slope {reg.get('ema_slope')}) — buying puts into a rising tape")
    return False, ""


def veto(struct: Optional[dict], direction: str) -> tuple[bool, str]:
    """Should the scout's arrow be vetoed by structure? direction = 'CE' (long) | 'PE'
    (short). Returns (veto?, reason). A breakout in the arrow's direction is CONFLUENCE
    and never vetoes; tiny headroom into the opposing level, or a no-breakout coil,
    vetoes. Pure read — the caller decides whether to ACT on it (see _STRUCT_VETO)."""
    if not struct or direction not in ("CE", "PE"):
        return False, ""
    brk = struct.get("breakout")
    consolidating = struct.get("consolidating")

    if direction == "CE":
        if brk == "UP":
            return False, f"breakout UP confirms — structure backs the long"
        d = struct.get("dist_res_atr")
        if d is not None and d < _NEAR_ATR:
            return True, (f"VETO long: price at resistance {struct.get('nearest_res')} "
                          f"({d:.2f} ATR headroom) — classic stop-out into the wall")
        if consolidating and not brk:
            return True, (f"VETO long: coiled (range {struct.get('coil_ratio')}×ATR), no "
                          f"breakout — the chop/whipsaw zone")
    else:  # PE
        if brk == "DOWN":
            return False, f"breakout DOWN confirms — structure backs the short"
        d = struct.get("dist_sup_atr")
        if d is not None and d < _NEAR_ATR:
            return True, (f"VETO short: price at support {struct.get('nearest_sup')} "
                          f"({d:.2f} ATR room) — selling into the floor, bounce risk")
        if consolidating and not brk:
            return True, (f"VETO short: coiled (range {struct.get('coil_ratio')}×ATR), no "
                          f"breakout — the chop/whipsaw zone")
    return False, ""


def summary(struct: Optional[dict]) -> str:
    """One-line DISPLAY string of where price sits in its structure (context)."""
    if not struct:
        return ""
    bits = []
    if struct.get("breakout"):
        bits.append(f"BREAKOUT {struct['breakout']}")
    elif struct.get("consolidating"):
        bits.append(f"coiled ({struct.get('coil_ratio')}×ATR)")
    r, s = struct.get("nearest_res"), struct.get("nearest_sup")
    if r is not None:
        bits.append(f"res {r} (+{struct.get('dist_res_atr')} ATR)")
    if s is not None:
        bits.append(f"sup {s} (−{struct.get('dist_sup_atr')} ATR)")
    return "structure: " + ", ".join(bits) if bits else ""
