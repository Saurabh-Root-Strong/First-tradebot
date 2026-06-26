"""
intraday_scout.py — multi-index intraday TRADE/NO-TRADE scanner (chart-driven).

The left-pane "where is the trade right now?" engine. For a chosen timeframe it
scans ALL four indices off the SAME series the charts plot (footprint_chart.
build_series / build_futures_series) and answers the question you ask when you
look at the board: "is there a clean setup on any index, and on which one?"

Per index it reads the captured chart data at instant t and looks for the
structural events you watch for on the charts — CE/PE OI crossover, price-vs-OI
divergence, the delta-adjusted buy/write/cover flow, futures-basis lean — fuses
them into a signed strength, and gates HARD: a trade is emitted only when enough
independent signals AGREE. Otherwise the index is NO-TRADE (positioning context
only). That is what makes "no trade on NIFTY / MIDCAP, trade on BANKNIFTY" fall
out naturally instead of forcing a call on every index every cycle.

PARITY BY CONSTRUCTION: it consumes build_series(..., as_of=t), which reads the
lock-free mirrors with ts<=t. So the SAME function runs live (t=now) and under
the replay clock (t=scrubbed past instant) with no lookahead. Predict at 11:30 on
a past day in the dashboard's Replay mode, then advance the clock to watch it play
out; the offline scoreboard (does the call actually pay?) is backtest_suggestion.py
/ backtest_crossover.py, which replay this exact structural read across every
captured day with a day-block CI.

HONESTY: index DIRECTION has been null/contrarian in every test in this repo
(see backtest_crossover, hour_forecast, the verdict-outcomes work). So the scout
is decision-support, the standing OI tilt is weighted as CONTEXT not a call, and
the hour_forecast RANGE band (the trustworthy product) rides alongside every row.
Treat a TRADE row as "this is the cleanest structure on the board right now,"
not a guarantee — the replay scoreboard is the arbiter.

    .venv\\Scripts\\python.exe intraday_scout.py                 # live scan, 15m
    .venv\\Scripts\\python.exe intraday_scout.py 2026-06-25 11:30 15
"""
from __future__ import annotations

import datetime
from typing import Optional

from core.constants import INDEX_SYMBOLS, STRIKE_STEP, LABELS, IST
import footprint_chart as fc
import hour_forecast as hf

# ── Signal weights (sum to 1.0). The delta-adjusted FLOW carries the most weight:
# it strips delta·Δindex from the premium, so it reads true demand, not the price
# echo. The standing OI TILT is NOT scored — it tested mildly CONTRARIAN / ~zero IC
# here (backtest_crossover), so its directional sign is unreliable; it rides along as
# DISPLAYED context only, never moves strength. ───────────────────────────────────
_W = {"flow": 0.40, "div": 0.25, "cross": 0.15, "fut": 0.20}
_RECENT = 3                 # bars treated as "recent" for flow / divergence
_TILT_DB = 0.05            # |g| below this = balanced book, no tilt call
_TRADE_TH = 0.22           # min |strength| to consider a trade
_MIN_AGREE = 3             # min independent signals agreeing with the sign to TRADE


def _last(seq, n: int = 1):
    """Last n non-None values of a list, oldest→newest. [] if none."""
    out = [v for v in seq if v is not None]
    return out[-n:] if out else []


def _atm(spot: float, sym: str) -> Optional[int]:
    step = STRIKE_STEP.get(sym)
    if not spot or not step:
        return None
    return int(round(spot / step) * step)


# Directional bullishness of each (leg, action) from build_series._act, on the
# DELTA-ADJUSTED residual (true demand, not the price echo):
#   CALL buy=long calls (+bull) · write=ceiling (−) · cover=ceiling shorts out (+) ·
#        unwind=upside longs leaving (−)
#   PUT  buy=long puts (−bear)  · write=floor (+)   · cover=floor writers out (−)   ·
#        unwind=downside longs leaving (+)
# NOTE: it is NOT "puts=bullish/calls=bearish" — that only holds for writing. Buying
# a put is bearish, buying a call is bullish; covering/unwinding flip again. Getting
# this wrong silently inverts the dominant signal.
_CALL_VAL = {"buy": 1.0, "write": -0.8, "cover": 0.6, "unwind": -0.4, "flat": 0.0}
_PUT_VAL  = {"buy": -1.0, "write": 0.8, "cover": -0.6, "unwind": 0.4, "flat": 0.0}
_ACT_WORD = {"buy": "buying", "write": "writing", "cover": "covering",
             "unwind": "unwinding", "flat": "flat"}


def _dom_action(acts: list, valmap: dict) -> str:
    """The action that drove this leg's net score over the window — grouped by
    action type and summed, so a single outlier bar can't mislabel a leg that was
    mostly the other way. Verb then agrees with the leg's net direction."""
    agg: dict = {}
    for a in acts:
        if a != "flat":
            agg[a] = agg.get(a, 0.0) + valmap.get(a, 0.0)
    return max(agg, key=lambda a: abs(agg[a])) if agg else "flat"


def _flow_signal(ser: dict) -> tuple[float, str]:
    """Delta-adjusted buy/write/cover/unwind over the recent bars → signed lean,
    correctly signed PER LEG PER ACTION (see _CALL_VAL/_PUT_VAL)."""
    ce, pe = ser.get("ce_act") or [], ser.get("pe_act") or []
    ce_r, pe_r = ce[-_RECENT:], pe[-_RECENT:]
    score = sum(_CALL_VAL.get(a, 0.0) for a in ce_r) + sum(_PUT_VAL.get(a, 0.0) for a in pe_r)
    n = max(len(ce_r) + len(pe_r), 1)
    s = max(-1.0, min(1.0, score / n))
    dom_ce = _dom_action(ce_r, _CALL_VAL)
    dom_pe = _dom_action(pe_r, _PUT_VAL)
    lean = "bullish" if s > 0.05 else "bearish" if s < -0.05 else "mixed"
    reason = (f"flow: CALL {_ACT_WORD.get(dom_ce, dom_ce)} / "
              f"PUT {_ACT_WORD.get(dom_pe, dom_pe)} ({lean})")
    return s, reason


def _divergence_signal(ser: dict) -> tuple[float, str]:
    """Price vs OI build over recent bars. Price UP while CE OI builds (ceiling
    written into strength) = bearish divergence; price DOWN while PE OI builds
    (floor) = bullish divergence; price/flow agreeing = confirmation, not div."""
    spot = _last(ser.get("spot") or [], _RECENT + 1)
    if len(spot) < 2:
        return 0.0, ""
    dpx = spot[-1] - spot[0]
    d_ce = _last(ser.get("d_oi_ce") or [], _RECENT)
    d_pe = _last(ser.get("d_oi_pe") or [], _RECENT)
    ce_build = sum(d_ce) if d_ce else 0.0
    pe_build = sum(d_pe) if d_pe else 0.0
    s, reason = 0.0, ""
    if dpx > 0 and ce_build > 0 and ce_build > pe_build:
        s = -0.7; reason = "price up into rising CALL OI — ceiling capping (bearish div)"
    elif dpx < 0 and pe_build > 0 and pe_build > ce_build:
        s = 0.7;  reason = "price down into rising PUT OI — floor building (bullish div)"
    elif dpx > 0 and pe_build > ce_build:
        s = 0.4;  reason = "price up, puts being written — floor following price (bullish)"
    elif dpx < 0 and ce_build > pe_build:
        s = -0.4; reason = "price down, calls being written — ceiling following (bearish)"
    return s, reason


def _crossover_signal(ser: dict) -> tuple[float, str]:
    """CE/PE OI tilt g and whether it FLIPPED recently (the crossover event).
    g=(putOI-callOI)/(putOI+callOI). Returns the flip event lean (not the standing
    tilt — that is _tilt_signal)."""
    oc = _last(ser.get("oi_ce") or [], _RECENT + 2)
    op = _last(ser.get("oi_pe") or [], _RECENT + 2)
    if len(oc) < 3 or len(op) < 3:
        return 0.0, ""
    def g(i):
        tot = oc[i] + op[i]
        return (op[i] - oc[i]) / tot if tot else 0.0
    g_now, g_prev = g(-1), g(0)
    if abs(g_now) > _TILT_DB and abs(g_prev) > _TILT_DB and (g_now > 0) != (g_prev > 0):
        if g_now > 0:
            return 0.8, "PUT OI just crossed ABOVE call OI — floor forming (bullish flip)"
        return -0.8, "CALL OI just crossed ABOVE put OI — ceiling forming (bearish flip)"
    return 0.0, ""


def _tilt_signal(ser: dict) -> tuple[float, str]:
    """Standing CE/PE OI tilt — CONTEXT only (tested mildly contrarian/~0 IC),
    small weight. g>0 floor-heavy book, g<0 ceiling-heavy."""
    oc = _last(ser.get("oi_ce") or [], 1)
    op = _last(ser.get("oi_pe") or [], 1)
    if not oc or not op or (oc[-1] + op[-1]) <= 0:
        return 0.0, ""
    g = (op[-1] - oc[-1]) / (oc[-1] + op[-1])
    if abs(g) < _TILT_DB:
        return 0.0, ""
    if g > 0:
        return min(g, 0.5), f"book put-OI heavy (PCR-tilt {g:+.2f}) — floor below (context)"
    return max(g, -0.5), f"book call-OI heavy (PCR-tilt {g:+.2f}) — ceiling above (context)"


def _futures_signal(sym: str, tf_min: int, date, as_of) -> tuple[float, str]:
    """Futures positioning: basis (contango bullish), futures own move, and the
    OI-derived fut_act over recent bars (long-buildup/cover = bullish, short-
    buildup/unwind = bearish). Thin signal, modest weight."""
    try:
        f = fc.build_futures_series(sym, tf_min, date, as_of)
    except Exception:
        return 0.0, ""
    if not f.get("has_data"):
        return 0.0, ""
    s, bits = 0.0, []
    # 1) basis sign: contango (fut>spot) bullish, backwardation bearish
    basis = _last(f.get("basis") or [], 1)
    if basis:
        b = basis[-1]
        if abs(b) > 1.0:
            s += 0.35 if b > 0 else -0.35
            bits.append(f"basis {b:+.0f} ({'contango' if b > 0 else 'backwardation'})")
    # 2) futures own move over recent bars
    near = _last(f.get("near") or f.get("close") or [], _RECENT + 1)
    if len(near) > 1 and near[-1] != near[0]:
        s += 0.25 if near[-1] > near[0] else -0.25
    # 3) OI-derived positioning (long/short/cover/unwind) over recent bars
    fut_act = (f.get("fut_act") or [])[-_RECENT:]
    val = {"long": 0.5, "cover": 0.3, "short": -0.5, "unwind": -0.3, "flat": 0.0}
    if fut_act:
        fa = sum(val.get(a, 0.0) for a in fut_act) / len(fut_act)
        s += 0.4 * fa
        last_fa = fut_act[-1]
        if last_fa != "flat":
            bits.append(f"fut {last_fa}")
    s = max(-1.0, min(1.0, s))
    if abs(s) < 0.12:
        return 0.0, ""
    return s, ("futures: " + ", ".join(bits)) if bits else "futures lean"


def scan_index(sym: str, tf_min: int, date=None, as_of=None) -> dict:
    """Per-index structural read at instant t. Returns a row dict (always)."""
    label = LABELS.get(sym, sym)
    try:
        ser = fc.build_series(sym, tf_min, date, as_of)
    except Exception as exc:
        return {"sym": sym, "label": label, "has_data": False, "note": f"error: {exc}"}
    if not ser.get("has_data"):
        return {"sym": sym, "label": label, "has_data": False,
                "note": ser.get("note", "warming up")}

    parts = {
        "flow":  _flow_signal(ser),
        "div":   _divergence_signal(ser),
        "cross": _crossover_signal(ser),
        "fut":   _futures_signal(sym, tf_min, date, as_of),
    }
    tilt = _tilt_signal(ser)                                     # DISPLAY-only context
    strength = sum(_W[k] * parts[k][0] for k in _W)              # signed [-1,1]
    sgn = 1 if strength > 0 else -1 if strength < 0 else 0
    # agreement = scored signals sharing the strength's sign
    agree = sum(1 for k in _W if parts[k][0] != 0 and (parts[k][0] > 0) == (sgn > 0))
    active = sum(1 for k in _W if parts[k][0] != 0)

    spot = (_last(ser.get("spot") or [], 1) or [None])[-1]
    try:
        fcst = hf.forecast(sym, as_of=as_of, date=date)         # honest range band
    except Exception:
        fcst = {}
    atm = _atm(spot, sym)

    # ── verdict gate: TRADE only on enough agreeing structure ────────────────────
    if abs(strength) >= _TRADE_TH and agree >= _MIN_AGREE and sgn != 0:
        direction = "CE" if sgn > 0 else "PE"
        verdict = f"TRADE {direction}"
    else:
        direction = ""
        verdict = "NO-TRADE"
    conf = int(round(min(abs(strength) / 0.6, 1.0) * 100))

    reasons = [parts[k][1] for k in ("flow", "div", "cross", "fut") if parts[k][1]]
    if tilt[1]:
        reasons.append(tilt[1])
    return {
        "sym": sym, "label": label, "has_data": True,
        "tf": tf_min, "spot": spot, "atm": atm,
        "strength": round(strength, 3), "agree": agree, "active": active,
        "verdict": verdict, "direction": direction, "confidence": conf,
        "instrument": (f"{atm} {direction}" if direction and atm else ""),
        "reasons": reasons,
        "parts": {k: round(parts[k][0], 3) for k in _W},
        "range_lo": fcst.get("lo"), "range_hi": fcst.get("hi"),
        "range_pct": fcst.get("exp_move_pct"),
        "fc_dir": fcst.get("direction"), "p_up": fcst.get("p_up"),
    }


def scan(tf_min: int, date=None, as_of=None) -> list[dict]:
    """Scan all four indices; rank tradables first by |strength|·agreement."""
    rows = [scan_index(s, tf_min, date, as_of) for s in INDEX_SYMBOLS]

    def _rank(r):
        if not r.get("has_data"):
            return (-1.0,)
        traded = 1 if r["verdict"].startswith("TRADE") else 0
        return (traded, abs(r["strength"]) * (1 + r["agree"]))
    return sorted(rows, key=_rank, reverse=True)


if __name__ == "__main__":
    import sys
    date = sys.argv[1] if len(sys.argv) > 1 and ":" not in sys.argv[1] else None
    as_of = None
    tf = 15
    if date and len(sys.argv) > 2:
        hh, mm = sys.argv[2].split(":")
        as_of = datetime.datetime.fromisoformat(date).replace(
            hour=int(hh), minute=int(mm), tzinfo=IST)
    if len(sys.argv) > 3:
        tf = int(sys.argv[3])

    when = f"{date} {as_of:%H:%M}" if as_of else "LIVE now"
    print(f"\nINTRADAY SCOUT — {tf}m bars @ {when}")
    print("=" * 74)
    for r in scan(tf, date, as_of):
        if not r.get("has_data"):
            print(f"  {r['label']:14s} {r['note']}")
            continue
        head = (f"  {r['label']:14s} {r['verdict']:9s} str {r['strength']:+.2f} "
                f"agree {r['agree']}/{r['active']} conf {r['confidence']}%")
        if r["instrument"]:
            head += f"  -> {r['instrument']}"
        print(head)
        if r["range_lo"] is not None:
            print(f"      range[{r['range_lo']}, {r['range_hi']}]  "
                  f"({r['fc_dir']} p_up={r['p_up']})")
        for why in r["reasons"][:4]:
            print(f"      - {why}")
    print("=" * 74)
    print("Direction is decision-support (null/contrarian in backtests); the RANGE")
    print("band is the trustworthy product. Validate calls via backtest_suggestion.py.")
