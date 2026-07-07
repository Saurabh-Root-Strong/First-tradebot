"""
intraday_scout.py — multi-index intraday TRADE/NO-TRADE scanner (chart-driven).

The left-pane "where is the trade right now?" engine. For a chosen timeframe it
scans ALL four indices off the SAME series the charts plot (footprint_chart.
build_series / build_futures_series) and answers the question you ask when you
look at the board: "is there a clean setup on any index, and on which one?"

Per index it reads the captured chart data at instant t and looks for the
structural events you watch for on the charts — CE/PE OI crossover, price-vs-OI
divergence, the delta-adjusted buy/write/cover flow, futures-basis lean — fuses
them into a signed strength, and gates HARD on INDEPENDENT corroboration: a trade is
emitted only when the anchor (delta-adjusted flow) agrees AND at least one INDEPENDENT
family (OI-tilt crossover, or futures) agrees too. div (price×ΔOI) shares the ΔOI
term with flow, so it feeds strength but is NOT counted as independent agreement — no
trade off two correlated options-OI signals. Otherwise NO-TRADE (positioning context
only). That is what makes "no trade on NIFTY / MIDCAP, trade on BANKNIFTY" fall out
naturally instead of forcing a call on every index every cycle.

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

from core.constants import INDEX_SYMBOLS, STRIKE_STEP, LABELS, IST, NIFTY, FINNIFTY
from core.mirror_io import read_mirror as _read_mirror
import footprint_chart as fc
import hour_forecast as hf
import price_structure as ps
import regime_classifier as rc

# Structurally THIN F&O index — sparse OI (audit: ~50% strikes no OI), tiny futures
# OI (~29k vs NIFTY ~7.4M), stale LTPs on illiquid strikes. Every signal + the
# option-premium reads are LESS RELIABLE here; flagged so it is never read with the
# same trust as NIFTY/BANK.
_THIN = {FINNIFTY}

# NSE killed weekly expiries for BANKNIFTY/FINNIFTY/MIDCAP — only NIFTY has a weekly;
# the others are MONTHLY only. The capture stores one (nearest tradeable) expiry per
# index, so this is just the honest label for what you'd actually trade.
def _expiry_kind(sym: str) -> str:
    return "weekly" if sym == NIFTY else "monthly"


_EXP_DATE_CACHE: dict = {}
_DCM_SYM = {"NSE:NIFTY50-INDEX": "NIFTY", "NSE:NIFTYBANK-INDEX": "BANKNIFTY",
            "NSE:FINNIFTY-INDEX": "FINNIFTY", "NSE:MIDCPNIFTY-INDEX": "MIDCPNIFTY"}


def _expiry_date(sym: str) -> Optional[str]:
    """Nearest tradeable expiry DATE for the index (weekly for NIFTY, monthly for the
    others), from the DCM fno_bhavcopy real calendar. Cached per day, best-effort — a
    display label only, so any failure returns None (falls back to the kind alone)."""
    import datetime as _dt
    key = (sym, _dt.date.today())
    if key in _EXP_DATE_CACHE:
        return _EXP_DATE_CACHE[key]
    out = None
    try:
        import duckdb, glob
        nsym = _DCM_SYM.get(sym)
        p = glob.glob("../Daily_Cash_Market/**/market_data.duckdb", recursive=True)
        if nsym and p:
            con = duckdb.connect(p[0], read_only=True)
            r = con.execute("select min(expiry_date) from fno_bhavcopy "
                            "where symbol=? and expiry_date >= current_date", [nsym]).fetchone()
            con.close()
            if r and r[0]:
                out = r[0].strftime("%d%b")            # e.g. 03Jul
    except Exception:
        out = None
    _EXP_DATE_CACHE[key] = out
    return out


# Premium SL / first-target as % of entry, by timeframe (from trade_setup.TF_PROFILES).
# Wider stop + further target the longer the hold.
_SLT = {5: (0.30, 0.50), 15: (0.32, 0.55), 60: (0.35, 0.65)}
_MKT_OPEN = datetime.time(9, 15)
_OR_END = datetime.time(9, 30)       # opening-range window = first 15 min (09:15-09:30)
_OPEN_SETTLE = datetime.time(9, 35)  # NO trade before this — market still cooling off
_OPEN_VALID = datetime.time(9, 45)   # range/backtest calibration only validated from here
_GAP_FLAT = 0.30                     # |gap%| below this = flat open
_GAP_SHARP = 0.75                    # |gap%| at/above this = sharp gap
_WARMUP_MIN = 20                     # minutes of live data before the scout trades — from
                                     # the SESSION start, re-anchored only on a true cold-restart
_RESUME_RESET_S = 1800               # only a >30-min outage re-anchors the warmup clock. A
                                     # shorter blip (WS reconnect, a process restart) leaves the
                                     # accumulated OI/flow/price picture intact — the scout reads
                                     # the full session, so a few missing minutes don't invalidate
                                     # it. Resetting 20min on every brief blip wrongly muted a
                                     # whole session after midday restarts (2026-06-29).
_TRIG_MAX_MIN = 120        # cap the contiguous-run trigger walk-back (minutes)

# ── Signal weights (sum to 1.0). The delta-adjusted FLOW carries the most weight:
# it strips delta·Δindex from the premium, so it reads true demand, not the price
# echo. The standing OI TILT is NOT scored — it tested mildly CONTRARIAN / ~zero IC
# here (backtest_crossover), so its directional sign is unreliable; it rides along as
# DISPLAYED context only, never moves strength. ───────────────────────────────────
_W = {"flow": 0.40, "div": 0.25, "cross": 0.15, "fut": 0.20}
_RECENT = 3                 # bars treated as "recent" for flow / divergence
_TILT_DB = 0.05            # |g| below this = balanced book, no tilt call
_TRADE_TH = 0.22           # min |strength| to consider a trade
# gate (see scan_index): anchor flow must agree + >=1 INDEPENDENT family (cross/fut);
# div is the same family as flow so it never counts as independent corroboration.

# PRICE-STRUCTURE VETO: when True, a structure veto (arrow into resistance/support, or
# a no-breakout coil — see price_structure.veto) turns a TRADE into NO-TRADE. DEFAULT
# OFF: the struct read is computed + DISPLAYED + harvested on every scan so backtest_scout
# can MEASURE whether the veto cuts the option-P&L bleed OUT of sample. Flip to True only
# after that CI clears — never wire structure to the live arrow blind.
_STRUCT_VETO = False

# TREND VETO: don't fight the day trend — veto a CE in a confirmed DOWN day / a PE in a
# confirmed UP day (price_structure.regime + trend_veto). The 2026-06-29 SL audit said
# the arrow's killer is WRONG DIRECTION (32/34 CE on a down day), not S/R. Same rule:
# computed + displayed + harvested ALWAYS, acts on the verdict only when True. DEFAULT
# OFF until backtest_scout's trend-veto grade clears OOS.
_TREND_VETO = False

# MOOD VETO: the missing third mood. price_structure.regime votes UP/DOWN/NEUTRAL and
# trend_veto only fights a CONFIRMED trend — it lets the CHOP through as "NEUTRAL". But
# backtest_regime.py (2026-06-30) showed CONSOLIDATION = ~58% of engine entries and the
# loss sink in EVERY config (worst win%/exp/drawdown) = the "both-side SL hunting" the
# user named. regime_classifier (Kaufman efficiency ratio + ATR-drift) labels that chop
# explicitly and vetoes trading INTO it. Computed + displayed ALWAYS; acts only when True.
# DEFAULT OFF until backtest_scout grades it on the actual option premium OOS — the
# underlying-engine proof is necessary, not sufficient (option pays theta + 3% RT on top).
_MOOD_VETO = False
# Efficiency window for the LIVE mood read. Must be small enough to be WARM intraday:
# ER needs win+2 closed bars, so 10 → warm ~11:45 on 15m / ~10:05 on 5m (vs ER14 = only
# warm afternoon). backtest_regime confirms the chop=loss-sink ordering holds at this
# window. Mood reads CHOP (no veto, since direction-gated) until warm — fail-safe.
_MOOD_WIN = 10


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
    reason = (f"flow {s:+.2f}: CALL {_ACT_WORD.get(dom_ce, dom_ce)} / "
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
    # carry the signed value AND the raw ΔOI magnitudes (per-index, distinct even in the
    # same bucket) so the read is transparently this-index-specific, not a repeated template.
    if reason:
        reason = f"div {s:+.1f} — {reason}  [ΔOI ce{ce_build:+.0f}/pe{pe_build:+.0f}]"
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
    # 1) basis vs its OWN session norm — NOT the raw sign. Index futures sit in structural
    # contango (cost-of-carry), so basis>0 is the resting state, not a bull signal; using
    # the sign made fut vote bullish 76% of the time (audit_signals) and the term tested
    # ANTI-predictive. Only basis RICH/CHEAP vs the day's mean is directional (longs paying
    # up = rising premium = bullish). De-meaned + a 0.5σ deadband kills the structural lean.
    basis_all = [v for v in (f.get("basis") or []) if v is not None]
    if len(basis_all) >= 3:
        bmean = sum(basis_all) / len(basis_all)
        bstd = (sum((x - bmean) ** 2 for x in basis_all) / len(basis_all)) ** 0.5
        b = basis_all[-1] - bmean
        if bstd > 0 and abs(b) > 0.5 * bstd:
            s += 0.35 if b > 0 else -0.35
            bits.append(f"basis {basis_all[-1]:+.0f} ({'rich' if b > 0 else 'cheap'} vs day)")
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


def _opening_oi_build(sym: str, date, as_of) -> Optional[dict]:
    """The OPENING BOOK: net OI change vs prev close + the strikes building OI, over the
    first 20 min (09:15 → min(now, 09:35)). RAW positioning context only — oich alone
    can't tell buy from write, so the scout's delta-adjusted FLOW signal does the
    interpretation; this just shows WHAT accumulated, for transparency at the handoff."""
    if as_of is None:
        return None
    end = as_of
    settle = as_of.replace(hour=_OPEN_SETTLE.hour, minute=_OPEN_SETTLE.minute,
                           second=0, microsecond=0)
    if end > settle:
        end = settle                                  # freeze the picture at the open
    try:
        ch = _read_mirror("chain_snapshots", date, end, sym)
    except Exception:
        return None
    if ch is None or not len(ch) or "oich" not in ch.columns:
        return None
    ch = ch[ch["ts"].dt.time >= _MKT_OPEN]
    if not len(ch):
        return None
    last = ch.sort_values("ts").groupby(["strike", "side"]).last().reset_index()
    ce = last[last["side"] == "CE"]; pe = last[last["side"] == "PE"]

    def _top(df, n=3):
        return [[int(s), int(o)] for s, o in
                zip(df.nlargest(n, "oich")["strike"], df.nlargest(n, "oich")["oich"])
                if o > 0]

    return {
        "ce_oich": int(ce["oich"].sum()), "pe_oich": int(pe["oich"].sum()),
        "ce_vol": int(ce["volume"].sum()), "pe_vol": int(pe["volume"].sum()),
        "ce_walls": _top(ce), "pe_floors": _top(pe),
    }


def _opening_context(sym: str, date, as_of) -> Optional[dict]:
    """Classify the OPEN (gap vs prev close + opening range) and the session phase.
    Pure read-only CONTEXT: we have NO validated gap-trading edge (the backtest ledgers
    start ~09:45), so this DISPLAYS the open type + enforces a cool-off — it never
    invents a directional opening call. Lookahead-free (ticks <= as_of)."""
    if as_of is None:
        return None
    t = as_of.time()
    phase = ("OPENING" if t < _OPEN_SETTLE
             else "SETTLING" if t < _OPEN_VALID else "REGULAR")
    out = {"phase": phase}
    try:
        tk = _read_mirror("ticks", date, as_of, sym)
    except Exception:
        tk = None
    if tk is None or not len(tk):
        return out
    last = tk.iloc[-1]
    prev_close = None
    try:
        if "ch" in tk.columns and pd.notna(last["ch"]):
            prev_close = float(last["ltp"]) - float(last["ch"])   # ch = ltp - prev_close
    except Exception:
        prev_close = None
    day_open = None
    if "day_open" in tk.columns and pd.notna(last.get("day_open")):
        day_open = float(last["day_open"])
    if not day_open:
        day_open = float(tk.iloc[0]["ltp"])
    gap_pct = ((day_open / prev_close - 1.0) * 100.0) if (prev_close and day_open) else None
    # opening range = hi/lo of the first 15 min (09:15-09:30)
    orw = tk[(tk["ts"].dt.time >= _MKT_OPEN) & (tk["ts"].dt.time <= _OR_END)]
    or_lo = float(orw["ltp"].min()) if len(orw) else None
    or_hi = float(orw["ltp"].max()) if len(orw) else None
    if gap_pct is None:
        gtype = "OPEN"
    else:
        a = abs(gap_pct); d = "UP" if gap_pct > 0 else "DOWN"
        gtype = ("FLAT OPEN" if a < _GAP_FLAT
                 else f"SHARP GAP-{d}" if a >= _GAP_SHARP else f"GAP-{d}")
    out.update({
        "gap_pct": round(gap_pct, 2) if gap_pct is not None else None,
        "gap_type": gtype,
        "prev_close": round(prev_close, 1) if prev_close else None,
        "day_open": round(day_open, 1) if day_open else None,
        "or_lo": round(or_lo, 1) if or_lo else None,
        "or_hi": round(or_hi, 1) if or_hi else None,
    })
    # ── DATA WARMUP: anchor the cool-off to when the live feed started for the session.
    # The scout needs _WARMUP_MIN of data before it trades. Only a TRUE cold-restart (an
    # outage > _RESUME_RESET_S, i.e. >30min, that makes the session picture stale) re-anchors
    # the clock to the resume; a brief blip (WS reconnect / a process restart, common when
    # the laptop is moved or the feed flickers) does NOT — the accumulated OI/flow/price is
    # intact and the scout reads the whole session, so a few missing minutes shouldn't mute
    # it. Floored at the 09:35 settle so an early/pre-open feed can't trade before the open
    # cools off.
    settle_dt = as_of.replace(hour=_OPEN_SETTLE.hour, minute=_OPEN_SETTLE.minute,
                              second=0, microsecond=0)
    sess = tk[tk["ts"].dt.time >= _MKT_OPEN]
    if len(sess):
        gaps = sess["ts"].diff().dt.total_seconds()
        resumed = sess["ts"][gaps > _RESUME_RESET_S]        # ts after a STALE-making outage
        run_start = resumed.iloc[-1] if len(resumed) else sess["ts"].iloc[0]
        ready = max(settle_dt,
                    (run_start + pd.Timedelta(minutes=_WARMUP_MIN)).to_pydatetime())
        out["data_start"] = run_start.strftime("%H:%M")
        out["ready_at"] = ready                              # datetime (in-process use)
        out["warming"] = as_of < ready
    # opening book (net OI build + walls) — morning context only, skip later to stay light
    if t <= datetime.time(10, 30):
        out["oi_build"] = _opening_oi_build(sym, date, as_of)
    return out


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
               cur_strength, not_before=None, anchor=None) -> Optional[dict]:
    """Trade lifecycle for an OPEN directional call: when it TRIGGERED (scan back to
    the first contiguous same-direction TRADE bar), the entry strike/premium it would
    have been taken at, SL/target on the premium, live P&L, and a CLOSE/HOLD manage
    verdict. The manage logic exists to CUT the loser fast — the arrow is measured
    negative-EV, so a disciplined exit on reversal/theta is the only edge available.

    Replay-only (needs as_of); live shows the trade with manage='HOLD' until graded."""
    if as_of is None or direction not in ("CE", "PE"):
        return None
    # Clamp the walk-back reference to the last captured tick. Live `as_of` is wall-clock
    # now, which after the close (or any data stall) sits well past the last data point;
    # every dead minute in between re-evaluates to the SAME verdict on static data, so the
    # minute-by-minute walk-back wastes its whole budget rebuilding the series (the
    # post-close "scout stuck loading" hang). The honest "now" for an open trade is the
    # last data we actually have.
    _tk = _read_mirror("ticks", date, as_of, sym)
    if _tk is not None and len(_tk):
        _last_ts = _tk["ts"].max().to_pydatetime()
        if as_of > _last_ts:
            as_of = _last_ts
    side = direction
    want = f"TRADE {direction}"

    def _is_trade(t):
        return scan_index(sym, tf_min, date=date, as_of=t,
                          with_lifecycle=False, verdict_only=True).get("verdict") == want

    # ── trigger = start of the CURRENT contiguous run of THIS exact trade verdict
    # ending at as_of. Walk back MINUTE BY MINUTE; stop at the first minute that is not
    # this trade. The gate is NON-MONOTONIC on a forming bar — strength swings as the
    # partial bar fills (NIFTY 2026-06-25: TRADE@09:39-40, NO-TRADE@09:41-47, then
    # TRADE@09:48) — so the old tf-grid walk + binary-search boundary was WRONG: it
    # reported an earlier LAPSED blip (09:39) as the trigger for a trade that actually
    # (re)fired at 09:48, and priced entry off that stale minute (a phantom +P&L).
    # Minute resolution is the honest "held since". Naturally cheap — a flickery signal
    # stops on the first step back; only a genuinely persistent hold walks far (capped
    # at _TRIG_MAX_MIN). ──────────────────────────────────────────────────────────
    # Walk back at BAR granularity (step = tf_min), not per-minute. The trade is defined
    # on tf_min bars, so a minute-resolution walk just re-probes the same forming bar
    # tf_min times — ~120 build_series rebuilds for a long hold = the live "scout stuck
    # loading" hang. Stepping by tf_min checks each completed bar once (~8 probes for the
    # 120-min cap): same contiguous-run logic, trigger reported to the bar (the honest
    # grain for a tf_min trade), an order of magnitude fewer builds.
    # ABSOLUTE session grid (09:15 + k·step), NOT instants relative to as_of. A relative
    # grid moves with the viewing clock: at 13:45 it probes 12:45/11:45, at 14:00 it
    # probes 13:00/12:00 — a hold shorter than one bar then always reports trigger=as_of,
    # so the displayed "triggered" time DRIFTED to the current minute on every refresh
    # (ghost/replay made it obvious; live was masked by the alert-log overlay). Anchoring
    # probes to the fixed bar grid makes the same instants get probed every refresh →
    # the reported trigger is stable. A trade born on the still-forming bar honestly
    # shows trigger=as_of until its first grid instant passes, then settles for good.
    step = max(1, int(tf_min))
    # Budget scales with the bar: a flat 120-min cap silently TRUNCATED long holds on
    # coarse TFs — a 60m trade held since 13:15, viewed at 15:30, hit the cap before
    # reaching its true trigger and reported 14:15 with the entry REPRICED there
    # (−1% became −5%). 6 bars of budget = the whole session on 60m for 6 probes,
    # identical behaviour on 5/15m where 120 still dominates.
    _cap = max(_TRIG_MAX_MIN, 6 * step)
    _open_dt = as_of.replace(hour=_MKT_OPEN.hour, minute=_MKT_OPEN.minute,
                             second=0, microsecond=0)
    trig_t = as_of
    # ── POLLER ANCHOR: the alert poller (authoritative, sticky, holds a position through
    # a forming-bar NO-TRADE flicker) already logged when THIS position first fired. When
    # the caller hands us that instant, FREEZE the trigger there and skip the bar-grid walk.
    # Why it matters: on the coarse 60m board a trade born mid-bar has no completed TRADE
    # bar behind it, so the grid walk breaks at the first step and reports trigger=as_of —
    # entry re-priced to NOW on every 30s refresh → perpetual +0% and the trigger clock
    # drifting with the wall clock (14:44→14:50→14:53), while the ledger correctly holds
    # "since 14:44". The anchor collapses the two clocks: one frozen entry, honest running
    # P&L, board == ledger. Only live passes it (the poller log is a live artifact); replay
    # keeps the grid walk. Clamp to as_of so a just-logged anchor never sits in the future. ─
    if anchor is not None and anchor <= as_of:
        trig_t = anchor
    else:
        k = int((as_of - _open_dt).total_seconds() // 60) // step   # last grid idx <= as_of
        for i in range(k, -1, -1):
            t_i = _open_dt + datetime.timedelta(minutes=i * step)
            if (as_of - t_i).total_seconds() / 60.0 > _cap:
                break
            if not_before is not None and t_i < not_before:
                break                          # never report a trigger before data-ready
            if _is_trade(t_i):
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
    cur_spot = _spot_at(sym, date, as_of)
    out = {"trigger": trig_t.strftime("%H:%M"), "entry_strike": trig_atm,
           "entry_spot": round(spot_trig, 2) if spot_trig else None,   # INDEX level at trigger
           "cur_spot": round(cur_spot, 2) if cur_spot else None,       # INDEX level now
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


# Horizon-scaling exponent for the band — CENTRALISED in hour_forecast (band home) and
# aliased here so intraday_scout + the backtests keep reading scout._BAND_HURST. (H/60)^0.48
# holds endpoint coverage ~68% across 30/60/120/240m (2yr audit); =1 at H=60 (60m base
# preserved). See hour_forecast for the derivation.
_BAND_HURST = hf._BAND_HURST


def _forward(direction: str, spot, range_pct_60, horizon_min: int) -> dict:
    """Forward prediction over horizon_min: UP/DOWN/RANGE + target + band.
    Band scales the 60m realised-vol forecast by (H/60)^_BAND_HURST (mean-reverting, not sqrt)."""
    pdir = "UP" if direction == "CE" else "DOWN" if direction == "PE" else "RANGE"
    if not spot or range_pct_60 is None:
        return {"pdir": pdir, "target": None, "pred_lo": None, "pred_hi": None,
                "move_pct": None}
    band_pct = float(range_pct_60) * (max(horizon_min, 1) / 60.0) ** _BAND_HURST
    band = spot * band_pct / 100.0
    target = (round(spot + band, 1) if pdir == "UP"
              else round(spot - band, 1) if pdir == "DOWN" else None)
    return {"pdir": pdir, "target": target,
            "pred_lo": round(spot - band, 1), "pred_hi": round(spot + band, 1),
            "move_pct": round(band_pct, 3)}


def scan_index(sym: str, tf_min: int, date=None, as_of=None,
               horizon_min: Optional[int] = None, with_lifecycle: bool = True,
               verdict_only: bool = False, anchor=None) -> dict:
    """Per-index structural read at instant t + forward prediction over horizon_min
    (defaults to the bar timeframe) + verify (replay only) + trade lifecycle
    (trigger time / SL / target / CLOSE-HOLD). with_lifecycle=False inside the
    lookback scan to avoid recursion. verdict_only=True returns just the gate
    verdict/strength/spot (skips hour_forecast + verify + forward) — the cheap probe
    the lifecycle walk-back/refine uses, ~3-5x faster per call."""
    # Cap the evaluation at the session close FIRST — the tick feed emits after 15:30, which
    # would otherwise produce phantom post-close triggers ("held since 15:51").
    as_of = hf.eval_asof(as_of, date)
    # Live intraday resolves to None — PIN the instant. With as_of=None the opening
    # protections silently vanish (_opening_context returns None → no warmup gate;
    # in_open is False → no 09:35 settle gate), so a caller that forgets to pass
    # `now` (CLI live scan, any future headless consumer) trades into the open
    # unprotected. The dashboard/poller always passed now explicitly; this makes
    # every path equivalent.
    if as_of is None:
        as_of = datetime.datetime.now(IST)
    horizon_min = int(horizon_min or tf_min)
    # Clip to the remaining session near the close: "next 60m" at 15:18 would project past
    # 15:30 into the overnight gap — not an intraday read. The band/verify/label then honestly
    # reflect the rest-of-session window (hour_forecast.session_horizon).
    horizon_min = hf.session_horizon(horizon_min, as_of)
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

    # ── HONEST agreement over INDEPENDENT signal families ────────────────────────
    # flow and div share the per-leg ΔOI term (div = price × ΔOI, flow = residual
    # demand on the same ΔOI), so div is NOT an independent corroborator of flow —
    # counting both inflates "agree". Families: OI-flow {flow(anchor)+div}, OI-tilt
    # {cross}, futures {fut}. The agreement count = anchor flow + the INDEPENDENT
    # families (cross, fut) that share the sign. div still feeds `strength` (weighted)
    # as confirmation, but cannot manufacture a passing count.
    def _agrees(k):
        return parts[k][0] != 0 and (parts[k][0] > 0) == (sgn > 0)
    flow_ok = _agrees("flow")
    indep = [k for k in ("cross", "fut") if _agrees(k)]          # independent families
    div_confirms = _agrees("div")
    agree = (1 if flow_ok else 0) + len(indep)                   # honest, max 3
    active = ((1 if parts["flow"][0] != 0 else 0)
              + sum(1 for k in ("cross", "fut") if parts[k][0] != 0))

    spot = (_last(ser.get("spot") or [], 1) or [None])[-1]
    atm = _atm(spot, sym)

    # ── verdict gate: anchor flow MUST agree + ≥1 INDEPENDENT family corroborates ─
    # (div can't be the corroborator — same family as flow). No naked-strength trade
    # off two correlated options-OI signals.
    if (abs(strength) >= _TRADE_TH and sgn != 0 and flow_ok and len(indep) >= 1):
        direction = "CE" if sgn > 0 else "PE"
        verdict = f"TRADE {direction}"
    else:
        direction = ""
        verdict = "NO-TRADE"

    # ── OPENING cool-off: NO trade before the market settles (~09:35). The first 15-20
    # min gap/settle is unrepresentative (sharp gap-up/down, range-bound ±40-50 pts) AND
    # the range/backtest calibration is only validated from ~09:45 — a trade off opening
    # noise is trading an UNVALIDATED regime. Cheap time-check here so the hot lifecycle
    # walk-back stays fast; the rich gap/OR context is built once in the full path below.
    in_open = as_of is not None and as_of.time() < _OPEN_SETTLE
    if in_open and verdict.startswith("TRADE"):
        verdict, direction = "NO-TRADE", ""

    # Fast path for lifecycle probing — skip the expensive forecast/verify/forward.
    if verdict_only:
        return {"sym": sym, "has_data": True, "verdict": verdict,
                "direction": direction, "strength": round(strength, 3), "spot": spot}

    conf = int(round(min(abs(strength) / 0.6, 1.0) * 100))
    try:
        fcst = hf.forecast(sym, as_of=as_of, date=date)         # honest range band
    except Exception:
        fcst = {}

    # ── price-structure (S/R, coil, breakout) — the orthogonal VETO family ───────
    # The OI arrow has no sense of WHERE price sits; this reads it off the same bars.
    # Computed + displayed + harvested ALWAYS; only ACTS on the verdict when _STRUCT_VETO
    # is on (default off — backtest_scout must first show the veto cuts the bleed OOS).
    struct = ps.analyze(ser)
    struct_veto, struct_veto_reason = ps.veto(struct, direction)
    if _STRUCT_VETO and struct_veto and verdict.startswith("TRADE"):
        verdict, direction = "NO-TRADE", ""
    # day-trend regime — the orthogonal "don't fight the trend" family (SL audit fix)
    regime = ps.regime(ser)
    trend_veto, trend_veto_reason = ps.trend_veto(regime, direction)
    if _TREND_VETO and trend_veto and verdict.startswith("TRADE"):
        verdict, direction = "NO-TRADE", ""
    # mood — the third mood (CONSOLIDATION) the day-trend read lets through (backtest_regime).
    # Read off a FIXED 5-min series, NOT the display tf: the regime is an objective read that
    # must not depend on the chart timeframe, its band-width factor + veto were CALIBRATED on
    # 5-min ER10 (backtest_band_regime / backtest_regime), and on coarse bars (tf=60) ER10
    # would need 10 hours and never warm intraday. Matches the cockpit (intraday_read uses 5m).
    try:
        _mood_ser = ser if tf_min == 5 else fc.build_series(sym, 5, date, as_of)
        mood = rc.classify_from_bars(_mood_ser, n=_MOOD_WIN)
    except Exception:
        mood = rc.classify_from_bars(ser, n=_MOOD_WIN)
    mood_veto, mood_veto_reason = rc.veto(mood, direction)
    if _MOOD_VETO and mood_veto and verdict.startswith("TRADE"):
        verdict, direction = "NO-TRADE", ""

    reasons = []
    for k in ("flow", "div", "cross", "fut"):
        if not parts[k][1]:
            continue
        txt = parts[k][1]
        if k == "div" and parts["div"][0] != 0:      # mark div as confirm vs diverge
            txt += " — confirms flow" if div_confirms else " — DIVERGES from flow"
        reasons.append(txt)
    if tilt[1]:
        reasons.append(tilt[1])
    s_sum = ps.summary(struct)
    if s_sum:
        reasons.append(s_sum)
    if regime.get("trend") and regime["trend"] != "NEUTRAL":
        reasons.append(f"day trend {regime['trend']} ({regime.get('day_pct')}% vs open)")
    if struct_veto_reason:
        reasons.append(struct_veto_reason + ("" if _STRUCT_VETO else " (measured, not enforced)"))
    if trend_veto_reason:
        reasons.append(trend_veto_reason + ("" if _TREND_VETO else " (measured, not enforced)"))
    reasons.append(f"mood: {mood.short}"
                   + (f" (efficiency {mood.er:.2f})" if mood.er == mood.er else ""))
    if mood_veto_reason:
        reasons.append(mood_veto_reason + ("" if _MOOD_VETO else " (measured, not enforced)"))

    # ── opening context (gap type / opening range / session phase) ───────────────
    opening = _opening_context(sym, date, as_of)
    ready_at = opening.get("ready_at") if opening else None
    if opening:
        ph = opening["phase"]
        gt = opening.get("gap_type", "")
        if opening.get("warming"):
            # data-driven cool-off (covers the normal 09:15-09:35 open AND a late start /
            # feed resume) — needs _WARMUP_MIN of contiguous data before trading.
            if verdict.startswith("TRADE"):
                verdict, direction = "NO-TRADE", ""
            ds = opening.get("data_start")
            opening["note"] = (
                f"DATA WARMUP — live feed since {ds}; need {_WARMUP_MIN}m of data, scout "
                f"trades from {ready_at:%H:%M}" if ready_at else
                "DATA WARMUP — accumulating data before trading")
            opening["suppressed"] = abs(strength) >= _TRADE_TH
        elif ph == "SETTLING":
            opening["note"] = (f"SETTLING ({gt}) — provisional: range/edge only fully "
                               f"validated from ~09:45. Half size, confirm the opening "
                               f"range first.")

    # ── forward prediction over the selected horizon + replay verify ─────────────
    fwd = _forward(direction, spot, fcst.get("exp_move_pct"), horizon_min)
    # L4 learned per-cell multiplier × regime-conditional width (widen in a strong trend
    # where the endpoint persists past the vol estimate — measured, backtest_band_regime).
    _bm = hf.band_multiplier(sym, horizon_min)
    _rw = rc.band_width_mult(mood)
    _tw = hf.tod_width_mult(as_of, horizon_min)   # close widen — gated OFF on short clipped H
    _wf = _bm * _rw * _tw
    if _wf != 1.0 and spot and fwd.get("pred_lo") is not None:
        half = (fwd["pred_hi"] - fwd["pred_lo"]) / 2.0 * _wf
        fwd["pred_lo"], fwd["pred_hi"] = round(spot - half, 1), round(spot + half, 1)
        if fwd.get("move_pct") is not None:
            fwd["move_pct"] = round(fwd["move_pct"] * _wf, 3)
        if fwd.get("target") is not None:
            fwd["target"] = (round(spot + half, 1) if fwd["pdir"] == "UP"
                             else round(spot - half, 1) if fwd["pdir"] == "DOWN" else None)
    bcov = hf.band_coverage(sym, horizon_min)          # HONEST measured coverage for this cell
    verify = _verify(sym, date, as_of, horizon_min, spot,
                     fwd["pdir"], fwd["pred_lo"], fwd["pred_hi"], atm=atm)
    # Honor the poller anchor only when its side matches the live leg — a genuine flip
    # closes the poller episode (no open anchor), so a mismatch is stale/transient: fall
    # back to the grid walk rather than freeze entry on the wrong side.
    _anchor_t = (anchor.get("t") if isinstance(anchor, dict)
                 and anchor.get("dir") == direction else None)
    lifecycle = (_lifecycle(sym, tf_min, date, as_of, direction, horizon_min, strength,
                            not_before=ready_at, anchor=_anchor_t)
                 if (with_lifecycle and direction) else None)
    return {
        "opening": opening,
        "sym": sym, "label": label, "has_data": True,
        "tf": tf_min, "horizon": horizon_min, "spot": spot, "atm": atm,
        "expiry": _expiry_kind(sym), "expiry_date": _expiry_date(sym), "thin": sym in _THIN,
        "strength": round(strength, 3), "agree": agree, "active": active,
        "verdict": verdict, "direction": direction, "confidence": conf,
        "instrument": (f"{atm} {direction}" if direction and atm else ""),
        "lifecycle": lifecycle,
        "reasons": reasons,
        "struct": struct, "struct_veto": bool(struct_veto),
        "regime": regime, "trend_veto": bool(trend_veto),
        "mood": mood.short, "mood_full": mood.mood, "mood_er": mood.er,
        "mood_veto": bool(mood_veto),
        "parts": {k: round(parts[k][0], 3) for k in _W},
        "range_lo": fcst.get("lo"), "range_hi": fcst.get("hi"),
        "range_pct": fcst.get("exp_move_pct"),
        "fc_dir": fcst.get("direction"), "p_up": fcst.get("p_up"),
        # forward call (what happens NEXT over `horizon` min) + its grading
        "pred_dir": fwd["pdir"], "pred_target": fwd["target"],
        "pred_lo": fwd["pred_lo"], "pred_hi": fwd["pred_hi"],
        "pred_move_pct": fwd["move_pct"], "verify": verify,
        "band_cover": bcov["cover"], "band_n": bcov["n"], "band_conf": bcov["conf"],
    }


def scan(tf_min: int, date=None, as_of=None,
         horizon_min: Optional[int] = None, anchors=None) -> list[dict]:
    """Scan all four indices; rank tradables first by |strength|·agreement.
    `anchors` = {sym: frozen-trigger datetime} from the live poller ledger, so an OPEN
    position's lifecycle freezes its entry at the true first-fire minute instead of
    re-walking the coarse bar grid (see _lifecycle). None → per-index grid walk."""
    rows = [scan_index(s, tf_min, date, as_of, horizon_min,
                       anchor=(anchors or {}).get(s)) for s in INDEX_SYMBOLS]

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
        if r.get("band_cover") is not None:
            _cm = {"ok": "✓", "soft": "~", "low": "⚠ low", "thin": "· thin"}.get(r["band_conf"], "")
            pl += f"  (cover {r['band_cover']*100:.0f}% n{r['band_n']} {_cm})"
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
            print(f"      TRADE {r['expiry']}: triggered {lc['trigger']} @ index "
                  f"{lc.get('entry_spot')} → {lc.get('cur_spot')}  "
                  f"entry {lc['entry_strike']} {r['direction']} ₹{lc['entry_prem']} "
                  f"→ now ₹{lc['cur_prem']}{pnl}  SL ₹{lc['sl']} T ₹{lc['target']}  "
                  f"=> {lc['manage']}")
        for why in r["reasons"][:3]:
            print(f"      - {why}")
    print("=" * 74)
    print("Direction is decision-support (null/contrarian in backtests); the RANGE")
    print("band is the trustworthy product. Validate calls via backtest_suggestion.py.")
