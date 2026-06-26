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
out. The verdict on whether the call PAYS is backtest_scout.py, which replays THIS
exact strength+gate across every captured day with a day-block CI.

HONESTY — MEASURED, do not trade the arrow: backtest_scout.py (8 days, n=73 trades)
grades the ACTUAL option you'd buy (ATM CE/PE entry at t, exit at t+H, net 3% round-
trip). Result: buying the arrow WINS only ~14-23% and BLEEDS −2% to −5% mean net per
trade — the 5m and 30m loss CIs EXCLUDE zero (a statistically significant LOSER), not
merely "no edge". IC(strength, fwd_ret) CIs straddle 0 (5m slightly contrarian). The
ONLY validated product is the hour_forecast RANGE band (~70% close-in-band at 15m).
So: "TRADE CE/PE" is a STRUCTURAL LEAN / decision-support label, NOT a tradeable
directional signal — trade the range band + structural levels, never buy a naked
option off the arrow until backtest_scout shows POSITIVE mean net P&L OUT of sample.
Standing OI tilt = context only.

    .venv\\Scripts\\python.exe intraday_scout.py                 # live scan, 15m
    .venv\\Scripts\\python.exe intraday_scout.py 2026-06-25 11:30 15
"""
from __future__ import annotations

import datetime
import math
import sys
from typing import Optional

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from core.constants import INDEX_SYMBOLS, STRIKE_STEP, LABELS, IST, NIFTY
from core.mirror_io import read_mirror as _read_mirror
import footprint_chart as fc
import hour_forecast as hf

# NSE killed weekly expiries for BANKNIFTY/FINNIFTY/MIDCAP — only NIFTY has a weekly;
# the others are MONTHLY only. The capture stores one (nearest tradeable) expiry per
# index, so this is just the honest label for what you'd actually trade.
def _expiry_kind(sym: str) -> str:
    return "weekly" if sym == NIFTY else "monthly"


# Premium SL / first-target as % of entry, by timeframe (from trade_setup.TF_PROFILES).
# Wider stop + further target the longer the hold.
_SLT = {5: (0.30, 0.50), 15: (0.32, 0.55), 60: (0.35, 0.65)}
_MKT_OPEN = datetime.time(9, 15)

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


def _spot_at(sym: str, date, t) -> Optional[float]:
    """Last traded price at or before t from the tick mirror (lookahead-bounded)."""
    try:
        tk = _read_mirror("ticks", date, t, sym)
    except Exception:
        return None
    if tk is None or not len(tk):
        return None
    s = tk[tk["ts"] <= pd.Timestamp(t)]
    if not len(s):
        return None
    return float(s.iloc[-1]["ltp"])


# Round-trip option cost as % of premium (entry half-spread + exit half-spread +
# brokerage/slippage). Index weekly ATM is liquid but the bid/ask still bites; 3%
# of premium is a conservative-but-realistic intraday round trip. The P&L "win" is
# net of this — a directionally-right call that didn't move the option past costs
# is correctly graded a LOSS (the whole point of the option-P&L audit).
_OPT_RT_COST = 3.0


def _opt_premium(sym: str, date, t, strike, side: str, expiry="weekly") -> Optional[float]:
    """ATM option LTP for (strike, side) at or before t from the chain mirror.
    Same expiry filter build_series uses, so entry/exit are the same instrument."""
    if not strike:
        return None
    try:
        ch = _read_mirror("chain_snapshots", date, t, sym)
    except Exception:
        return None
    if ch is None or not len(ch) or "ltp" not in ch.columns:
        return None
    ch, ok = fc._filter_expiry(ch, expiry)
    if not ok or ch is None or not len(ch):
        return None
    sub = ch[(ch["side"] == side) & (ch["strike"] == strike)
             & (ch["ts"] <= pd.Timestamp(t))]
    if not len(sub):
        return None
    v = sub.sort_values("ts").iloc[-1]["ltp"]
    return float(v) if pd.notna(v) and float(v) > 0 else None


def _verify(sym: str, date, as_of, horizon_min: int, spot0: float,
            pdir: str, lo, hi, atm=None) -> Optional[dict]:
    """Grade the forward call against what ACTUALLY happened horizon_min after t.
    Returns None when the future is not available yet (LIVE, or replay too close to
    the close) — so live shows a pending prediction, replay shows a graded one.
    The future read is the answer key only; it never feeds the prediction.

    Includes OPTION P&L for directional calls: buy the ATM CE/PE at t, exit at t+H,
    net of round-trip cost. opt_win is the HONEST grade — index ticking your way is
    not the same as the option making money after spread+theta."""
    if as_of is None or not spot0:
        return None                       # live: t+H hasn't happened
    t_end = as_of + datetime.timedelta(minutes=horizon_min)
    try:
        tk = _read_mirror("ticks", date, t_end, sym)
    except Exception:
        return None
    if tk is None or not len(tk):
        return None
    mx = tk["ts"].max()
    # require ticks to actually reach ~t+H, else the horizon isn't resolved yet
    if pd.isna(mx) or mx < pd.Timestamp(t_end) - pd.Timedelta(minutes=2):
        return None
    s = tk[tk["ts"] <= pd.Timestamp(t_end)]
    if not len(s):
        return None
    actual = float(s.iloc[-1]["ltp"])
    move = (actual / spot0 - 1.0) * 100.0
    in_band = (lo is not None and hi is not None and lo <= actual <= hi)
    if pdir == "UP":
        dir_hit = actual > spot0
    elif pdir == "DOWN":
        dir_hit = actual < spot0
    else:                                 # RANGE call: hit if it stayed in band
        dir_hit = in_band
    out = {"actual": round(actual, 2), "move_pct": round(move, 3),
           "dir_hit": bool(dir_hit), "band_hit": bool(in_band)}

    # ── OPTION P&L: the trade you'd actually take, net of cost ────────────────────
    if pdir in ("UP", "DOWN") and atm:
        side = "CE" if pdir == "UP" else "PE"
        entry = _opt_premium(sym, date, as_of, atm, side)
        exit_ = _opt_premium(sym, date, t_end, atm, side)
        if entry and exit_:
            pnl = (exit_ / entry - 1.0) * 100.0
            net = pnl - _OPT_RT_COST
            out.update({"opt_side": side, "opt_entry": round(entry, 2),
                        "opt_exit": round(exit_, 2), "opt_pnl_pct": round(pnl, 1),
                        "opt_net_pct": round(net, 1), "opt_win": bool(net > 0)})
    return out


def _lifecycle(sym, tf_min, date, as_of, direction, horizon_min,
               cur_strength) -> Optional[dict]:
    """Trade lifecycle for an OPEN directional call: when it TRIGGERED (scan back to
    the first contiguous same-direction TRADE bar), the entry strike/premium it would
    have been taken at, SL/target on the premium, live P&L, and a CLOSE/HOLD manage
    verdict. The manage logic exists to CUT the loser fast — the arrow is measured
    negative-EV, so a disciplined exit on reversal/theta is the only edge available.

    Replay-only (needs as_of); live shows the trade with manage='HOLD' until graded."""
    if as_of is None or direction not in ("CE", "PE"):
        return None
    side = direction
    # ── trigger time: walk back on the tf grid while it stayed the same TRADE ─────
    trig_t = as_of
    for i in range(1, 13):
        t_i = as_of - datetime.timedelta(minutes=tf_min * i)
        if t_i.time() < _MKT_OPEN:
            break
        r_i = scan_index(sym, tf_min, date=date, as_of=t_i,
                         horizon_min=horizon_min, with_lifecycle=False)
        if r_i.get("verdict") == f"TRADE {direction}":
            trig_t = t_i
        else:
            break
    # ── entry strike = the ATM at trigger (what you'd actually have bought) ───────
    spot_trig = _spot_at(sym, date, trig_t)
    trig_atm = _atm(spot_trig, sym) if spot_trig else None
    # premium reads use the captured chain (single nearest expiry, stored under the
    # legacy/"weekly" bucket); _expiry_kind is the honest DISPLAY label only.
    entry = _opt_premium(sym, date, trig_t, trig_atm, side) if trig_atm else None
    cur = _opt_premium(sym, date, as_of, trig_atm, side) if trig_atm else None
    sl_pct, t1_pct = _SLT.get(tf_min, (0.32, 0.55))
    out = {"trigger": trig_t.strftime("%H:%M"), "entry_strike": trig_atm,
           "entry_prem": round(entry, 2) if entry else None,
           "cur_prem": round(cur, 2) if cur else None,
           "sl": round(entry * (1 - sl_pct), 2) if entry else None,
           "target": round(entry * (1 + t1_pct), 2) if entry else None,
           "pnl_pct": None, "manage": "HOLD"}

    manage = "HOLD"
    if entry and cur:
        out["pnl_pct"] = round((cur / entry - 1.0) * 100.0, 1)
        if cur <= entry * (1 - sl_pct):
            manage = "CLOSE · SL hit"
        elif cur >= entry * (1 + t1_pct):
            manage = "BOOK · target hit"
    # structure reversed against the position → exit
    if manage == "HOLD" and cur_strength is not None:
        want_up = direction == "CE"
        if (cur_strength > 0) != want_up and abs(cur_strength) >= _TRADE_TH * 0.5:
            manage = "CLOSE · flow reversed"
    # sideways + theta bleed: held a while, index barely moved, premium decaying
    if manage == "HOLD" and spot_trig and entry and cur:
        spot_now = _spot_at(sym, date, as_of)
        if spot_now:
            mv = abs(spot_now / spot_trig - 1.0) * 100.0
            held = (as_of - trig_t).total_seconds() / 60.0
            if held >= 2 * tf_min and mv < 0.05 and cur < entry:
                manage = "CLOSE · sideways, theta bleed"
    out["manage"] = manage
    return out


def _forward(direction: str, spot, range_pct_60, horizon_min: int) -> dict:
    """Forward prediction over horizon_min: UP/DOWN/RANGE + target + band.
    Band scales the 60m realised-vol forecast by sqrt(H/60) (vol ~ sqrt time)."""
    pdir = "UP" if direction == "CE" else "DOWN" if direction == "PE" else "RANGE"
    if not spot or range_pct_60 is None:
        return {"pdir": pdir, "target": None, "pred_lo": None, "pred_hi": None,
                "move_pct": None}
    band_pct = float(range_pct_60) * math.sqrt(max(horizon_min, 1) / 60.0)
    band = spot * band_pct / 100.0
    target = (round(spot + band, 1) if pdir == "UP"
              else round(spot - band, 1) if pdir == "DOWN" else None)
    return {"pdir": pdir, "target": target,
            "pred_lo": round(spot - band, 1), "pred_hi": round(spot + band, 1),
            "move_pct": round(band_pct, 3)}


def scan_index(sym: str, tf_min: int, date=None, as_of=None,
               horizon_min: Optional[int] = None, with_lifecycle: bool = True) -> dict:
    """Per-index structural read at instant t + forward prediction over horizon_min
    (defaults to the bar timeframe) + verify (replay only) + trade lifecycle
    (trigger time / SL / target / CLOSE-HOLD). with_lifecycle=False inside the
    lookback scan to avoid recursion. Returns a row dict."""
    horizon_min = int(horizon_min or tf_min)
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

    # ── forward prediction over the selected horizon + replay verify ─────────────
    fwd = _forward(direction, spot, fcst.get("exp_move_pct"), horizon_min)
    verify = _verify(sym, date, as_of, horizon_min, spot,
                     fwd["pdir"], fwd["pred_lo"], fwd["pred_hi"], atm=atm)
    lifecycle = (_lifecycle(sym, tf_min, date, as_of, direction, horizon_min, strength)
                 if (with_lifecycle and direction) else None)
    return {
        "sym": sym, "label": label, "has_data": True,
        "tf": tf_min, "horizon": horizon_min, "spot": spot, "atm": atm,
        "expiry": _expiry_kind(sym),
        "strength": round(strength, 3), "agree": agree, "active": active,
        "verdict": verdict, "direction": direction, "confidence": conf,
        "instrument": (f"{atm} {direction}" if direction and atm else ""),
        "lifecycle": lifecycle,
        "reasons": reasons,
        "parts": {k: round(parts[k][0], 3) for k in _W},
        "range_lo": fcst.get("lo"), "range_hi": fcst.get("hi"),
        "range_pct": fcst.get("exp_move_pct"),
        "fc_dir": fcst.get("direction"), "p_up": fcst.get("p_up"),
        # forward call (what happens NEXT over `horizon` min) + its grading
        "pred_dir": fwd["pdir"], "pred_target": fwd["target"],
        "pred_lo": fwd["pred_lo"], "pred_hi": fwd["pred_hi"],
        "pred_move_pct": fwd["move_pct"], "verify": verify,
    }


def scan(tf_min: int, date=None, as_of=None,
         horizon_min: Optional[int] = None) -> list[dict]:
    """Scan all four indices; rank tradables first by |strength|·agreement."""
    rows = [scan_index(s, tf_min, date, as_of, horizon_min) for s in INDEX_SYMBOLS]

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
    rows = scan(tf, date, as_of)
    hits = sum(1 for r in rows if r.get("verify") and r["verify"]["dir_hit"])
    graded = sum(1 for r in rows if r.get("verify"))
    sb = f"  ·  scoreboard {hits}/{graded} dir-hit" if graded else ""
    print(f"\nINTRADAY SCOUT — predict next {tf}m @ {when}{sb}")
    print("=" * 74)
    for r in rows:
        if not r.get("has_data"):
            print(f"  {r['label']:14s} {r['note']}")
            continue
        head = (f"  {r['label']:14s} {r['verdict']:9s} str {r['strength']:+.2f} "
                f"agree {r['agree']}/{r['active']} conf {r['confidence']}%")
        if r["instrument"]:
            head += f"  -> {r['instrument']}"
        print(head)
        # forward call + grade
        pl = (f"      PREDICT next {r['horizon']}m: {r['pred_dir']}"
              + (f" -> {r['pred_target']}" if r['pred_target'] else "")
              + (f"  band[{r['pred_lo']}, {r['pred_hi']}]" if r['pred_lo'] else ""))
        v = r.get("verify")
        if v:
            mark = "HIT ✓" if v["dir_hit"] else "MISS ✗"
            pl += (f"   => actual {v['actual']} ({v['move_pct']:+.2f}%) "
                   f"{mark}  band {'✓' if v['band_hit'] else '✗'}")
            if "opt_net_pct" in v:
                owin = "WIN" if v["opt_win"] else "LOSS"
                pl += (f"  | opt {v['opt_side']} {v['opt_entry']}→{v['opt_exit']} "
                       f"net {v['opt_net_pct']:+.0f}% {owin}")
        else:
            pl += "   => (pending — future not reached)"
        print(pl)
        lc = r.get("lifecycle")
        if lc:
            pnl = f" ({lc['pnl_pct']:+.0f}%)" if lc.get("pnl_pct") is not None else ""
            print(f"      TRADE {r['expiry']}: triggered {lc['trigger']}  "
                  f"entry {lc['entry_strike']} {r['direction']} ₹{lc['entry_prem']} "
                  f"→ now ₹{lc['cur_prem']}{pnl}  SL ₹{lc['sl']} T ₹{lc['target']}  "
                  f"=> {lc['manage']}")
        for why in r["reasons"][:3]:
            print(f"      - {why}")
    print("=" * 74)
    print("Direction is decision-support (null/contrarian in backtests); the RANGE")
    print("band is the trustworthy product. Validate calls via backtest_suggestion.py.")
