"""
tradeboard.py — the LIVE fused board: one row per index, everything the desk knows, and
an HONEST verdict. Headless (the system's discipline: scan_index is headless, dashboard
renders it) — this returns a row model + a CLI render; the Dash panel is a thin view over
build_board() (step 2, not here).

WHAT IT FUSES (all lookahead-free, closed bars):
  • MOOD        regime_classifier — BIG/small trend or CHOP (the gate)
  • HTF struct  BTST-style structure() on 15m + 60m (BREAKOUT/TREND/CONSOLIDATION/RANGE)
  • PA state    the operating-TF candle at the band edge (rejection wick = the fade trigger)
  • BAND        the VALIDATED product — pred_lo/hi + honest measured coverage%
  • SETUP       the band-fade (the ONE mechanizable edge) IF it fires: side, index entry /
                target / SL, R, and the STRIKE to express it
  • OPTION ctx  OI / COI / premium / IV / delta / wall-dist / PCR / DTE at that strike
  • VERDICT     TRADE-FADE (liquid only) · RANGE-ONLY · NO-TRADE(reason) · WARMING

HONESTY GUARDRAILS (this is why the board is safe, not a signal-manufacturing machine):
  1. NO directional CE/PE arrow — that product is dead (cost floor). Only the mean-reversion
     band-fade, which is the only setup that cleared a hardened OOS grade.
  2. The fade edge is a FUTURES trade (~+0.9bps hardened, BANK/NIFTY). Expressed via an
     OPTION strike it pays the ~3% theta round-trip that KILLS it → the board names the ATM
     strike as the vehicle but LABELS the cost drag; index levels are the truth.
  3. FIN/MIDCAP fade dies on realistic futures cost → they are RANGE-ONLY, never TRADE-FADE.
  4. Regime gate: no fade INTO a BIG_TREND. Warmup gate: WARMING until the band/ER are warm.
  5. Data-staleness: the option chain dies ~11am; OI/COI older than _STALE_MIN is greyed and
     never feeds a verdict.
  6. It shows the HONEST numbers — band coverage%, fade win≈breakeven-recently — not a fake
     accuracy. Default is NO-TRADE.

    .venv\\Scripts\\python.exe tradeboard.py                       # live now
    .venv\\Scripts\\python.exe tradeboard.py --replay 2026-07-16 --at 13:30
"""
from __future__ import annotations

import argparse
import datetime
import os
import sys

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from core.constants import INDEX_SYMBOLS, LABELS, IST, STRIKE_STEP
from core.mirror_io import read_mirror
import footprint_chart as fc
import intraday_scout as scout
import paper_fade_logger as pf

# Liquid enough for the fade to survive realistic futures cost (hardened grade: BANK/NIFTY
# survive, FIN/MIDCAP die). Keyed by the FY short name.
_FADE_LIQUID = {"NIFTY50", "NIFTYBANK"}
_MKT_CLOSE = datetime.time(15, 30)   # NSE session close — post-close nothing is genuinely open
_STALE_MIN = 15                      # option-chain context older than this = stale/greyed

# ── LOOKBACK WINDOWS — env-toggleable A/B (2026-07-26). Two candidate configs:
#   BASELINE 20/40 (DEFAULT, live) — best on the 29-day LIVE option ledger (win 48%, idx -0.012R,
#     ex-jack +6,310) vs set C's (39%, -0.077R, -4,358) on the SAME days.
#   SET C   40/60 — best on 2yr CLEAN candles index-level (+0.100R OOS 15m-1h) + sits out whipsaws
#     (Friday 07-24: 1 trade vs baseline 7). But the 2yr gain did NOT confirm on the thin 29-day
#     live sample (yellow flag — sample-size collision; 29 days can't separate them).
# DECISION (user 2026-07-26): revert to 20/40 live; A/B set C forward, decide Fri evening on the
# config with better LIVE accuracy. Flip without a code edit:
#     $env:TRADEBOT_STRUCT_LB=40 ; $env:TRADEBOT_SL_WIN=60   (PowerShell) then restart dashboard.
# Fair test = grade BOTH on the same captured days: python compare_lookback.py
_STRUCT_LB = int(os.environ.get("TRADEBOT_STRUCT_LB", "20"))   # structure + ER lookback
_SL_WIN = int(os.environ.get("TRADEBOT_SL_WIN", "40"))         # HTF pivot window for the SL

# ── GRANDPARENT (3rd) TIMEFRAME — the frame above the confirm frame. OPT-IN filter only.
# MEASURED 2yr, lookahead-corrected (coarse bar truncated to the decision instant):
#     5m-15m  + GP 30m : −0.009R vs −0.007R baseline  → adds NOTHING
#     10m-30m + GP 1h  : −0.001R vs −0.001R baseline  → adds NOTHING
#     15m-1h  + GP 4h  : +0.045R [+0.01,+0.09] n=709 vs +0.013R baseline → the ONE lift,
#                        but win% DROPS 26.6→24.5 (fewer, bigger winners) and +0.045R is still
#                        far under the ~0.2R option cost floor.
# So it is a SELECTIVITY dial, not an edge: it keeps only setups whose grandparent frame is
# actively trending your way (it is NOT a conflict-veto — barely any trade conflicts; most
# baseline fires happen while the GP frame is merely RANGE/CONSOLIDATION).
_GP_OF = {15: 30, 30: 60, 60: 240}

# ── SCOUT BEHAVIOURAL GUARDS (25-day forensics 2026-07-17: every bad day = LOW-RANGE tape +
# SERIAL re-fires of the same failed idea; e.g. 06-22 seven range-top breaks on a 0.39% day
# = −9.2k). Thresholds IN-SAMPLE-TUNED on those 25 days (raw +81k → guarded +103k, trades
# −39%, win 46→51%) — direction robust (matches 2yr chop=loss-sink), magnitude not a promise.
_TAPE_MIN_PCT = 0.45   # DEAD-TAPE gate: no new setups while max(range-so-far, |gap|) < this %
_MAX_STRIKES = 2       # THREE-STRIKES: same (index, setup, side) fires at most twice per day


_HIST5: dict = {}
_CONT_CACHE: dict = {}     # memoize _bars_continuous within a render (called ~15-20x/render)


def _hist_5min(sym: str):
    """Cached read of the native historical 5-min parquet (continuous, to ~today). Keyed by
    the file's (mtime,size) so an EOD re-download refreshes it WITHOUT a dashboard restart
    (was a stale-until-restart bug in a long-running process)."""
    from core.constants import DATA_DIR
    from pathlib import Path
    fn = sym.replace(":", "_").replace("-", "_")
    p = Path(DATA_DIR) / "historical" / "5min" / f"{fn}_5min.parquet"
    try:
        st = p.stat(); sig = (st.st_mtime_ns, st.st_size)
    except OSError:
        return None
    hit = _HIST5.get(sym)
    if hit is not None and hit[0] == sig:
        return hit[1]
    df = pd.read_parquet(p)[["ts", "open", "high", "low", "close"]].copy()
    df["ts"] = pd.to_datetime(df["ts"])
    _HIST5[sym] = (sig, df)
    return df


def _prior_live_days(sym: str, before_day: str, want5: int):
    """FALLBACK prior-day 5-min bars from the LIVE captures (data/intraday/live/<date>_ticks),
    for when the historical parquet is MISSING/THIN — e.g. a fresh VM whose data/historical was
    never populated (2026-07-27: VM ledger fired 0 trades because the 60m couldn't warm). The VM
    captures every day, so its own prior live sessions ARE the warm-up bars; this makes the board
    self-sufficient with zero historical-store maintenance. Builds 5-min per prior captured day
    (most recent first) until >= want5 bars. Cached via _CONT_CACHE key."""
    from core.constants import LIVE_DIR
    import glob as _glob
    ck = ("_pld", sym, before_day, want5)
    hit = _CONT_CACHE.get(ck)
    if hit is not None:
        return hit
    days = []
    for p in _glob.glob(str(LIVE_DIR / "*_ticks.parquet")):
        s = os.path.basename(p).split("_")[0]
        if s < before_day:
            try:
                datetime.date.fromisoformat(s); days.append(s)
            except ValueError:
                pass
    frames = []
    got = 0
    for d in sorted(days, reverse=True):          # newest prior day first
        try:
            ser = fc.build_series(sym, 5, d, None)
            if ser.get("has_data") and ser.get("ts"):
                frames.append(pd.DataFrame({
                    "ts": pd.to_datetime(ser["ts"]), "open": ser["open"],
                    "high": ser["high"], "low": ser["low"], "close": ser["close"]}))
                got += len(ser["ts"])
        except Exception:
            continue
        if got >= want5:
            break
    res = (pd.concat(frames, ignore_index=True) if frames else None)
    _CONT_CACHE[ck] = res
    return res


def _bars_continuous(sym: str, tf: int, date, as_of, need: int = 24):
    """CONTINUOUS tf-min OHLC ending at as_of — stitches native 5-min history (PRIOR days) +
    today's LIVE 5-min session, resampled to tf by integer-chunking WITHIN each day (session-
    anchored to 09:15, cross-day continuous). This is what lets the 20-bar Kaufman ER warm on
    30m/60m, which a single-day build_series never can. Prior days come from the historical
    parquet; if that's MISSING/THIN, falls back to prior LIVE captures (self-sufficient VM).
    None if no history at all."""
    hist = _hist_5min(sym)
    frames = []
    day = date or (as_of.date().isoformat() if as_of else None)
    k = max(1, tf // 5)
    ckey = (sym, tf, day, as_of.isoformat() if as_of else None, need)
    hit = _CONT_CACHE.get(ckey)
    if hit is not None:
        return hit
    want5 = (need + 5) * k + 20                         # only this many 5-min bars are needed
    day_start = pd.Timestamp(day) if day else None
    if hist is not None and len(hist):
        # fast datetime cut (no per-row .dt.date), then TAIL before anything else — avoids
        # processing the whole 37k-row 2yr history on every call (the load bug).
        h = hist[hist["ts"] < day_start] if day_start is not None else hist
        h = h.tail(want5)
        if len(h):
            frames.append(h[["ts", "open", "high", "low", "close"]])
    # FALLBACK: historical missing/thin OR STALE → warm from prior LIVE captures. Stale check:
    # if the newest historical prior-bar is >3 days before `day`, the 60m context would read
    # weeks-old structure, so pull recent prior sessions from live captures and merge (dedup
    # below keeps them). Makes the VM self-sufficient even as its historical store ages.
    prior_have = sum(len(f) for f in frames)
    stale = False
    if day_start is not None and frames:
        newest = max(f["ts"].max() for f in frames)
        stale = (day_start - pd.Timestamp(newest)) > pd.Timedelta(days=3)
    if day_start is not None and (prior_have < want5 or stale):
        pld = _prior_live_days(sym, day, want5)
        if pld is not None and len(pld):
            frames.append(pld[pld["ts"] < day_start][["ts", "open", "high", "low", "close"]])
    try:
        ser = fc.build_series(sym, 5, date, as_of)
        if ser.get("has_data") and ser.get("ts"):
            frames.append(pd.DataFrame({
                "ts": pd.to_datetime(ser["ts"]), "open": ser["open"],
                "high": ser["high"], "low": ser["low"], "close": ser["close"]}))
    except Exception:
        pass
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True).dropna().drop_duplicates("ts").sort_values("ts")
    if as_of is not None:
        cut = pd.Timestamp(as_of).tz_localize(None) if df["ts"].dt.tz is None else pd.Timestamp(as_of)
        df = df[df["ts"] <= cut]
    df = df.tail(want5)                                # bound the resample input
    if k == 1:
        out = df
    else:
        df = df.copy(); df["d"] = df["ts"].dt.date
        rows = []
        for _, g in df.groupby("d"):
            g = g.reset_index(drop=True); grp = g.index // k
            a = g.groupby(grp).agg(ts=("ts", "first"), open=("open", "first"),
                                   high=("high", "max"), low=("low", "min"),
                                   close=("close", "last"))
            rows.append(a)
        out = pd.concat(rows, ignore_index=True)
    res = out.tail(need + 3) if len(out) else None
    if len(_CONT_CACHE) > 400:
        _CONT_CACHE.clear()
    _CONT_CACHE[ckey] = res
    return res


def _candle_pattern(o, h, l, c, same_session: bool = True) -> str:
    """Classic candlestick pattern on the LAST CLOSED candle (uses the prior bar for the 2-bar
    patterns). CONTEXT vocabulary for the user's PA — no validated machine edge. Priority:
    2-bar reversals/compression first, then single-candle. o/h/l/c = arrays, last = -1.
    same_session=False (the two bars straddle an overnight gap on a coarse TF) → 2-bar patterns
    are SUPPRESSED (a cross-gap 'engulfing/tweezer' is a gap artifact, not a real pattern)."""
    if len(c) < 2:
        return ""
    O, H, L, C = o[-1], h[-1], l[-1], c[-1]
    Op, Hp, Lp, Cp = o[-2], h[-2], l[-2], c[-2]
    rng = H - L
    if rng <= 0:
        return ""
    body = abs(C - O); uw = H - max(O, C); lw = min(O, C) - L
    br = body / rng
    bull = C > O
    pbody = abs(Cp - Op)
    # ── 2-bar patterns (only within the SAME session — never across an overnight gap) ──
    if same_session:
        if body > pbody and pbody > 0:
            if bull and Cp < Op and C >= Op and O <= Cp:
                return "bull engulfing"
            if (not bull) and Cp > Op and C <= Op and O >= Cp:
                return "bear engulfing"
        # TWEEZER = matching extreme AT A SWING (the pair's high/low is the local extreme of the
        # last ~6 bars) + opposite bodies — not just any two adjacent bars with close extremes.
        _sw = 6
        if len(h) >= _sw:
            swing_hi = max(H, Hp) >= h[-_sw:].max() - 1e-9
            swing_lo = min(L, Lp) <= l[-_sw:].min() + 1e-9
            if H > 0 and abs(H - Hp) / H < 0.0004 and not bull and swing_hi:
                return "tweezer top"
            if L > 0 and abs(L - Lp) / L < 0.0004 and bull and swing_lo:
                return "tweezer bottom"
        if H < Hp and L > Lp:
            return "inside bar"
        if H > Hp and L < Lp:
            return "outside bar"
    # ── single-candle patterns ──
    if br >= 0.85:
        return "marubozu↑" if bull else "marubozu↓"
    if br <= 0.1:
        return "doji"
    if lw >= 2 * body and uw <= body and br < 0.45:
        return "hammer"
    if uw >= 2 * body and lw <= body and br < 0.45:
        return "shooting star"
    if br < 0.35:
        return "spinning top"
    return "bull-body" if bull else "bear-body"


def _struct_full(sym: str, tf: int, date, as_of, lookback: int = _STRUCT_LB,
                 drop_forming: bool = True) -> dict:
    """Structure label + the CONTEXT a synthesis needs: the TF range hi/lo (the box price
    lives in), ER, coil tightness (recent range / ATR), n closed bars, last close. Uses a
    CONTINUOUS cross-day tf series (stitched 5-min history + today) so the 20-bar Kaufman ER
    warms on 30m/60m; falls back to single-day build_series if no history.

    drop_forming=True (default, for the STABLE display): drop the in-progress last bar. FALSE
    (for the LIVE HTF confirmation used by the scout/ledger): KEEP the forming bar so the read
    is 'what you'd see on the 1h chart right now' — the intraday-actionable confirmation the
    trigger fires on (repaints as the bar fills, by design)."""
    cont = _bars_continuous(sym, tf, date, as_of, need=lookback + 4)
    _cut = slice(None, -1) if drop_forming else slice(None)
    _ts = None
    if cont is not None and len(cont) >= 8:
        o = cont["open"].to_numpy(float)[_cut]
        c = cont["close"].to_numpy(float)[_cut]
        h = cont["high"].to_numpy(float)[_cut]
        l = cont["low"].to_numpy(float)[_cut]
        _ts = cont["ts"].to_numpy()[_cut]
    else:
        ser = fc.build_series(sym, tf, date, as_of)
        if not ser.get("has_data"):
            return {"struct": "n/a", "n": 0}
        o = np.asarray(ser.get("open") or [], float)[_cut]
        c = np.asarray(ser.get("close") or [], float)[_cut]
        h = np.asarray(ser.get("high") or [], float)[_cut]
        l = np.asarray(ser.get("low") or [], float)[_cut]
        _t = ser.get("ts")
        _ts = pd.to_datetime(_t).to_numpy()[_cut] if _t else None
    if len(c) < 6:
        return {"struct": "n/a", "n": int(len(c))}
    # same-session = the last 2 CLOSED bars are the same calendar day (no overnight gap between
    # them) — coarse-TF 2-bar patterns across a gap are artifacts.
    same_session = True
    if _ts is not None and len(_ts) >= 2:
        same_session = pd.Timestamp(_ts[-1]).date() == pd.Timestamp(_ts[-2]).date()
    pattern = _candle_pattern(o, h, l, c, same_session=same_session)
    seg = c[-lookback:]
    net = seg[-1] - seg[0]
    diffs = np.abs(np.diff(seg))
    # KAUFMAN ER, GAP-AWARE: on the continuous cross-day series an overnight gap is one huge
    # bar-to-bar step. Counting it in the PATH (denominator) treats a gap-and-go as "wasted
    # motion" and crushes coarse-TF ER toward 0 (30m/60m falsely read chop). A gap is EFFICIENT
    # displacement — keep it in NET, drop it from PATH (sum only same-session steps).
    if _ts is not None and len(_ts) >= len(seg):
        sdt = _ts[-len(seg):]
        same = np.fromiter((pd.Timestamp(sdt[i]).date() == pd.Timestamp(sdt[i - 1]).date()
                            for i in range(1, len(sdt))), bool, len(sdt) - 1)
        path = diffs[same].sum() if same.any() else diffs.sum()
    else:
        path = diffs.sum()
    er = min(1.0, abs(net) / path) if path > 0 else 0.0
    hi, lo, last = h[-lookback:], l[-lookback:], c[-1]
    rng_hi, rng_lo = float(hi.max()), float(lo.min())
    # coil tightness = recent 3-bar span / prior span (small = spring-loaded)
    coil = None
    if len(hi) >= 6:
        prior = hi[:-3].max() - lo[:-3].min()
        recent = hi[-3:].max() - lo[-3:].min()
        coil = round(recent / prior, 2) if prior > 0 else None
    if len(hi) > 1 and last > hi[:-1].max():
        struct = "BREAKOUT_UP"
    elif len(lo) > 1 and last < lo[:-1].min():
        struct = "BREAKOUT_DOWN"
    elif er >= 0.4:
        struct = "TREND_UP" if net > 0 else "TREND_DOWN"
    elif coil is not None and coil < 0.6:
        struct = "CONSOLIDATION"
    else:
        struct = "RANGE"
    return {"struct": struct, "hi": rng_hi, "lo": rng_lo, "er": round(er, 2),
            "coil": coil, "pattern": pattern, "n": int(len(c)), "last": last}


def _structure(sym: str, tf: int, date, as_of, lookback: int = _STRUCT_LB) -> str:
    """Structure label only (back-compat wrapper over _struct_full). AUDIT-FIX 2026-07-27:
    default was a HARDCODED 20 — under the env A/B toggle (40/60) the htf15/htf60 display
    labels in build_row would stay 20-bar while the rest of the board read 40-bar. Now tracks
    _STRUCT_LB so the whole board is coherent when the config flips."""
    return _struct_full(sym, tf, date, as_of, lookback)["struct"]


_TREND_UP_S = {"BREAKOUT_UP", "TREND_UP"}
_TREND_DN_S = {"BREAKOUT_DOWN", "TREND_DOWN"}
_RANGE_S = {"CONSOLIDATION", "RANGE"}


def synthesize(htf: dict, ltf: dict, spot: float) -> dict:
    """HTF x LTF structure CONFLUENCE read — the multi-TF context, NOT a signal (intraday MTF
    has no validated directional edge — [[project_price_action_60m_no_edge]]; the honest-fill
    audit killed the fade). Returns {tag, read, color, loc}. The decisive variable when the
    HTF is a range is WHERE price sits in the HTF box (loc: 0=at low, 1=at high) — an LTF
    breakout mid-range is a false-break trap; at the box edge it may be a real resolution."""
    hs, ls = htf.get("struct", "n/a"), ltf.get("struct", "n/a")
    if hs == "n/a" or htf.get("n", 0) < 6:
        return {"tag": "HTF warming", "read": "higher-TF not enough closed bars yet — no "
                "multi-TF read", "color": "#64748b", "loc": None}
    if ls == "n/a":
        return {"tag": "LTF warming", "read": "lower-TF warming", "color": "#64748b", "loc": None}
    hi, lo = htf.get("hi"), htf.get("lo")
    loc = None
    if hi and lo and hi > lo and spot:
        loc = max(0.0, min(1.0, (spot - lo) / (hi - lo)))
    near_hi = loc is not None and loc >= 0.72
    near_lo = loc is not None and loc <= 0.28
    G, R, A, B, N = "#22c55e", "#f87171", "#fbbf24", "#40c4ff", "#94a3b8"

    # ── HTF is a RANGE / CONSOLIDATION (the user's case) ──────────────────────────
    if hs in _RANGE_S:
        if ls in _RANGE_S:
            return {"tag": "NESTED SQUEEZE", "color": A, "loc": loc,
                    "read": "compression on BOTH TFs — a bigger move is loading, DIRECTION "
                    "UNKNOWN. Stand aside, mark the HTF box, wait for the break (don't predict)."}
        if ls == "BREAKOUT_UP":
            if near_hi:
                return {"tag": "RANGE-TOP BREAK (attempt)", "color": G, "loc": loc,
                        "read": "LTF breaking UP at the HTF ceiling — the ONLY location an LTF "
                        "breakout can be real. Needs to HOLD above + volume/OI confirm, else it "
                        "snaps back. This is the HTF-range resolution watch."}
            return {"tag": "FALSE-BREAK TRAP", "color": R, "loc": loc,
                    "read": "LTF pop UP inside the HTF range (price mid/low) — statistically "
                    "fades back to the range. Do NOT chase; the HTF HIGH is the real line."}
        if ls == "BREAKOUT_DOWN":
            if near_lo:
                return {"tag": "RANGE-FLOOR BREAK (attempt)", "color": R, "loc": loc,
                        "read": "LTF breaking DOWN at the HTF floor — only here can it be real. "
                        "Needs to hold below + confirm, else it snaps back up into the range."}
            return {"tag": "FALSE-BREAK TRAP", "color": R, "loc": loc,
                    "read": "LTF drop DOWN inside the HTF range (price mid/high) — statistically "
                    "reverts. Do NOT chase; the HTF LOW is the real line."}
        # LTF trending inside HTF range
        return {"tag": "DRIFT-IN-RANGE", "color": N, "loc": loc,
                "read": "LTF drifting inside the HTF box — noise until it reaches a box edge; "
                "read the HTF high/low as the levels that matter."}

    # ── HTF is TRENDING / BROKEN OUT ─────────────────────────────────────────────
    htf_up = hs in _TREND_UP_S
    if ls in _RANGE_S:
        return {"tag": "WITH-TREND CONTINUATION (loading)", "color": G if htf_up else R,
                "loc": loc,
                "read": f"HTF {'UP' if htf_up else 'DOWN'}, LTF coiling = a pullback loading for "
                f"CONTINUATION (the textbook with-trend setup). Trigger = LTF break "
                f"{'UP' if htf_up else 'DOWN'}; invalid if it breaks the other way."}
    ltf_up = ls in _TREND_UP_S
    if ltf_up == htf_up:
        return {"tag": "EXTENDED (aligned)", "color": A, "loc": loc,
                "read": f"HTF and LTF BOTH {'UP' if htf_up else 'DOWN'} — aligned but late; "
                f"chase risk. Wait for a pullback (LTF coil) rather than entering extended."}
    return {"tag": "PULLBACK vs HTF", "color": A, "loc": loc,
            "read": f"LTF {'DOWN' if htf_up else 'UP'} against an HTF {'UP' if htf_up else 'DOWN'} "
            f"— a pullback/dip zone WITH the HTF trend IF the HTF structure holds; an "
            f"early-REVERSAL warning if the HTF level breaks. Watch the HTF pivot."}


_MTF_TFS = [5, 15, 30, 60]


def _tf_read(sym: str, tf: int, date, as_of) -> dict:
    """Structure + ER + last-closed candle character on one TF (context, no arrow). Uses the
    CONTINUOUS cross-day series so the 20-bar Kaufman ER warms on 30m/60m (single-day never
    has 20 coarse bars)."""
    sf = _struct_full(sym, tf, date, as_of)          # continuous 20-bar ER + structure
    st = sf.get("struct", "n/a")
    cont = _bars_continuous(sym, tf, date, as_of, need=24)
    if cont is not None and len(cont) >= 8:
        c = cont["close"].to_numpy(float)[:-1]; h = cont["high"].to_numpy(float)[:-1]
        l = cont["low"].to_numpy(float)[:-1]; o = cont["open"].to_numpy(float)[:-1]
    else:
        ser = fc.build_series(sym, tf, date, as_of)
        c = np.asarray(ser.get("close") or [], float)[:-1]
        h = np.asarray(ser.get("high") or [], float)[:-1]
        l = np.asarray(ser.get("low") or [], float)[:-1]
        o = np.asarray(ser.get("open") or [], float)[:-1]
    char, er = "—", sf.get("er")
    if len(c) >= 12:
        seg = c[-20:]
        er = round(abs(seg[-1] - seg[0]) / max(np.abs(np.diff(seg)).sum(), 1e-9), 2)
        rng = h[-1] - l[-1]
        clr = (c[-1] - l[-1]) / rng if rng > 0 else 0.5
        body = abs(c[-1] - o[-1]) / rng if rng > 0 else 0.0
        if clr >= 0.75:
            char = "reject↓wick" if c[-1] < o[-1] else "strong-up"
        elif clr <= 0.25:
            char = "reject↑wick" if c[-1] > o[-1] else "strong-dn"
        elif body < 0.3:
            char = "doji/indecision"
        else:
            char = "up-body" if c[-1] > o[-1] else "dn-body"
    return {"tf": tf, "struct": st, "er": er, "char": char}


def _mtf(sym: str, date, as_of) -> list[dict]:
    return [_tf_read(sym, tf, date, as_of) for tf in _MTF_TFS]


def _chain_last(sym: str, date, as_of):
    """Last snapshot per (strike, side) + data age in minutes. (None, None) if absent."""
    try:
        ch = read_mirror("chain_snapshots", date, as_of, sym)
    except Exception:
        return None, None
    if ch is None or not len(ch) or "oi" not in ch.columns:
        return None, None
    try:
        age = (as_of - ch["ts"].max().to_pydatetime()).total_seconds() / 60.0
    except Exception:
        age = None
    last = ch.sort_values("ts").groupby(["strike", "side"]).last().reset_index()
    return last, age


def _strike_table(last, atm, sym, n: int = 4) -> list[dict]:
    """ATM +/- n strikes: CE & PE OI / COI / vol / premium / IV / delta, side by side."""
    step = STRIKE_STEP.get(sym)
    if last is None or not atm or not step:
        return []
    rows = []
    for k in range(atm - n * step, atm + (n + 1) * step, step):
        ce = last[(last["strike"] == k) & (last["side"] == "CE")]
        pe = last[(last["strike"] == k) & (last["side"] == "PE")]

        def _g(df, col):
            return (df.iloc[-1].get(col) if len(df) else None)
        rows.append({
            "strike": k, "atm": k == atm,
            "ce_oi": _g(ce, "oi"), "ce_oich": _g(ce, "oich"), "ce_vol": _g(ce, "volume"),
            "ce_prem": _g(ce, "ltp"), "ce_iv": _g(ce, "iv"), "ce_delta": _g(ce, "delta"),
            "pe_oi": _g(pe, "oi"), "pe_oich": _g(pe, "oich"), "pe_vol": _g(pe, "volume"),
            "pe_prem": _g(pe, "ltp"), "pe_iv": _g(pe, "iv"), "pe_delta": _g(pe, "delta"),
        })
    return rows


def _chain_stats(last, spot, sym) -> dict:
    """Call/put walls (max-OI strikes), PCR, max-pain, from the full captured chain."""
    if last is None or not len(last):
        return {}
    ce, pe = last[last["side"] == "CE"], last[last["side"] == "PE"]
    out = {}
    if len(ce) and ce["oi"].max() > 0:
        out["call_wall"] = int(ce.loc[ce["oi"].idxmax(), "strike"])
    if len(pe) and pe["oi"].max() > 0:
        out["put_wall"] = int(pe.loc[pe["oi"].idxmax(), "strike"])
    tot_ce, tot_pe = float(ce["oi"].sum()), float(pe["oi"].sum())
    out["pcr"] = round(tot_pe / tot_ce, 3) if tot_ce > 0 else None
    # max-pain = expiry strike minimizing total in-the-money option value (writer pain min)
    strikes = sorted(set(last["strike"]))
    if strikes:
        ce_map = dict(zip(ce["strike"], ce["oi"])); pe_map = dict(zip(pe["strike"], pe["oi"]))
        best_k, best_v = None, None
        for K in strikes:
            v = sum(float(ce_map.get(k, 0)) * max(0, K - k) +
                    float(pe_map.get(k, 0)) * max(0, k - K) for k in strikes)
            if best_v is None or v < best_v:
                best_v, best_k = v, K
        out["max_pain"] = int(best_k) if best_k is not None else None
    return out


def _chain_age_min(sym: str, date, as_of) -> float | None:
    try:
        ch = read_mirror("chain_snapshots", date, as_of, sym)
    except Exception:
        return None
    if ch is None or not len(ch):
        return None
    last = ch["ts"].max()
    try:
        return (as_of - last.to_pydatetime()).total_seconds() / 60.0
    except Exception:
        return None


# Validated overnight strong-close net % per index (backtest_overnight_8yr.py, 8.5yr, clr>=0.66
# → long into close, exit next open). The DAILY candle carries the ONE robust directional edge.
_ONV = {"NIFTY50": 0.155, "NIFTYBANK": 0.176, "FINNIFTY": 0.164, "MIDCPNIFTY": 0.236}
_ONV_WIN = {"NIFTY50": 71, "NIFTYBANK": 69, "FINNIFTY": 68, "MIDCPNIFTY": 79}


def _daily_read(sym: str, as_of=None) -> dict:
    """DAILY candle character + the validated OVERNIGHT lean. Source = the local daily EOD
    parquet. clr/body/wicks, 20-day Kaufman ER, 200-DMA regime, daily breakout-from-base,
    and the overnight edge this strong close implies (8.5yr-validated).

    FORMING-TODAY fix: the EOD parquet is downloaded after close, so INTRADAY its last bar is
    YESTERDAY. For the live 'hold overnight tonight' decision we build TODAY's forming daily
    (open/high/low/last from today's intraday) and read clr off it — gated to the 15:10-15:30
    entry window (before that it is still forming; after 15:35 the EOD bar is the record)."""
    from core.constants import DATA_DIR
    from pathlib import Path
    short = None
    for k in _ONV:
        if k in sym.replace(":", "_").replace("-", "_"):
            short = k
    fn = sym.replace(":", "_").replace("-", "_")
    p = Path(DATA_DIR) / "historical" / "daily" / f"{fn}_daily.parquet"
    if not p.exists():
        return {}
    df = pd.read_parquet(p).sort_values("ts").reset_index(drop=True)
    if len(df) < 25:
        return {}
    o = df["open"].to_numpy(float); h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float); c = df["close"].to_numpy(float)
    last_date = pd.Timestamp(df["ts"].iloc[-1]).date()
    # ── FORMING TODAY: if the parquet's last bar is NOT today's session, build today's daily
    #    from intraday so the overnight lean reflects TONIGHT, not last night. ──────────────
    forming = False
    today = (as_of.date() if as_of is not None else datetime.datetime.now(IST).date())
    tt = (as_of.time() if as_of is not None else datetime.datetime.now(IST).time())
    if last_date < today:
        try:
            ser = fc.build_series(sym, 5, today.isoformat(), as_of)
            tc = np.asarray(ser.get("close") or [], float)
            if ser.get("has_data") and len(tc):
                th = np.asarray(ser.get("high") or [], float)
                tl = np.asarray(ser.get("low") or [], float)
                to = np.asarray(ser.get("open") or [], float)
                o = np.append(o, to[0]); h = np.append(h, th.max())
                l = np.append(l, tl.min()); c = np.append(c, tc[-1])
                forming = True
        except Exception:
            pass
    rng = h[-1] - l[-1]
    clr = (c[-1] - l[-1]) / rng if rng > 0 else 0.5
    body = abs(c[-1] - o[-1]) / rng if rng > 0 else 0.0
    uwick = (h[-1] - max(o[-1], c[-1])) / rng if rng > 0 else 0.0
    lwick = (min(o[-1], c[-1]) - l[-1]) / rng if rng > 0 else 0.0
    in_window = tt >= datetime.time(15, 10)
    seg = c[-20:]
    er20 = round(abs(seg[-1] - seg[0]) / max(np.abs(np.diff(seg)).sum(), 1e-9), 2)
    sma200 = float(np.mean(c[-200:])) if len(c) >= 200 else float(np.mean(c))
    regime = "bull" if c[-1] >= sma200 else "bear"
    # daily structure (breakout-from-base): last close clears prior 19-bar high after a tight base
    prior_hi = h[-20:-1].max(); prior_lo = l[-20:-1].min()
    base_rng = (h[-6:-1].max() - l[-6:-1].min())
    atr = float(np.mean(h[-14:] - l[-14:]))
    tight_base = atr > 0 and base_rng < 1.5 * atr
    if c[-1] > prior_hi:
        d_struct = "BREAKOUT_UP (from base)" if tight_base else "BREAKOUT_UP"
    elif c[-1] < prior_lo:
        d_struct = "BREAKOUT_DOWN"
    elif er20 >= 0.45:
        d_struct = "TREND_UP" if seg[-1] > seg[0] else "TREND_DOWN"
    elif tight_base:
        d_struct = "TIGHT BASE (coiled)"
    else:
        d_struct = "RANGE"
    strong = clr >= 0.66
    onv = _ONV.get(short); win = _ONV_WIN.get(short)
    # actionable only when the daily is settled: a completed EOD bar, OR the forming today
    # bar INSIDE the 15:10-15:30 entry window (before that clr is not yet meaningful).
    actionable = (not forming) or in_window
    if strong and actionable:
        lean = "LONG overnight"
    elif strong and forming:
        lean = "forming strong — firms in the 15:10-15:30 window"
    else:
        lean = "no overnight signal"
    return {
        "clr": round(clr, 2), "body": round(body, 2), "uwick": round(uwick, 2),
        "lwick": round(lwick, 2), "er20": er20, "regime": regime, "struct": d_struct,
        "close": round(c[-1], 2), "sma200": round(sma200, 1),
        "strong_close": strong, "onv": onv, "onv_win": win, "lean": lean,
        "forming": forming, "in_window": in_window, "actionable": bool(actionable),
        "asof_date": str(last_date if not forming else today),
    }


# The user's THREE MASTER COMBOS — each = a lower-TF entry confirmed by its higher-TF
# structure (the nested MTF price-action lens for intraday). (ltf, htf).
_COMBOS = [(5, 15), (10, 30), (15, 60)]


def _synth_combos(sym: str, date, as_of, spot: float) -> list[dict]:
    """Phase-2 read for all three nested combos (5×15, 10×30, 15×60). Each returns the HTF×LTF
    synthesis (continuation / nested-squeeze / false-break / pullback / etc). CONTEXT for the
    user's own PA — intraday MTF has no validated machine edge (every gate failed OOS)."""
    out = []
    for ltf_tf, htf_tf in _COMBOS:
        htf = _struct_full(sym, htf_tf, date, as_of, drop_forming=False)
        ltf = _struct_full(sym, ltf_tf, date, as_of)
        s = synthesize(htf, ltf, spot)
        s.update(ltf_tf=ltf_tf, htf_tf=htf_tf, ltf_struct=ltf.get("struct"),
                 htf_struct=htf.get("struct"), ltf_pattern=ltf.get("pattern"),
                 htf_pattern=htf.get("pattern"))
        out.append(s)
    return out


# Setup-quality rank (best trade context → worst) for the cross-index scout scan.
_TAG_RANK = {
    "WITH-TREND CONTINUATION (loading)": 0,   # HTF trend + LTF coil = the textbook setup
    "RANGE-TOP BREAK (attempt)": 1, "RANGE-FLOOR BREAK (attempt)": 1,   # resolution watch
    "PULLBACK vs HTF": 2,                      # dip zone with the HTF trend
    "EXTENDED (aligned)": 3,                   # aligned but late
    "DRIFT-IN-RANGE": 4, "NESTED SQUEEZE": 5,  # wait
    "FALSE-BREAK TRAP": 6,                     # avoid
}


def scout_pa_ledger(date, as_of, ltf_tf: int = 15, htf_tf: int = 60, hold: int = 6,
                    sl_mode: str = "htf", gp_only: bool = False) -> dict:
    """sl_mode: 'htf' (default, graded baseline — stop at the 60m structure pivot) ·
    'ltf' (stop at the ENTRY-TF 15m swing — tighter, chartist-proper geometry) ·
    'ltf_trail' (15m swing stop that RATCHETS behind each new completed favorable swing —
    the dynamic structure-trailing stop; never loosens). Non-default modes are GRADED
    experiments until they beat the baseline on the 25-day honest re-grade."""
    """TradeBoard SCOUT day-ledger — the level-trades the scout fired TODAY, graded (target/
    SL/timeout on the index) for closed ones, live status for open. Mirrors the Charts scout
    ledger. Level trade (index), 2yr-graded ~breakeven (MIDCAP marginally +, backtest_scout_pa)
    — a measurement of YOUR method, not a machine fire."""
    from core.constants import INDEX_SYMBOLS, LOT_SIZES
    L = {}
    SUP = 0                                                    # guard-suppressed fires (all idx)
    d0s = (as_of.date().isoformat() if as_of else date)
    for sym in INDEX_SYMBOLS:
        contL = _bars_continuous(sym, ltf_tf, date, as_of, need=max(60, _STRUCT_LB + 8))
        contH = _bars_continuous(sym, htf_tf, date, as_of, need=_SL_WIN + 20)  # SL-pivot + warmup
        if contL is None or contH is None or len(contL) < 26 or len(contH) < 26:
            continue
        # GRANDPARENT series (opt-in filter only; fetched lazily so the default path is unchanged)
        contG = None
        if gp_only and _GP_OF.get(htf_tf):
            contG = _bars_continuous(sym, _GP_OF[htf_tf], date, as_of, need=_STRUCT_LB + 12)
            if contG is None or len(contG) < _STRUCT_LB + 2:
                continue                      # cannot verify the grandparent → do not fire
        d0 = (as_of.date() if as_of else None)
        lot = LOT_SIZES.get(sym, 1)
        # today's captured option chain (premiums only exist for captured days, and the chain
        # DIES ~11am — so a premium read is trusted only when its snapshot is FRESH (<=20min old
        # at the query time), else 'n/a'. Read once per sym.
        try:
            _chain = read_mirror("chain_snapshots", d0s, as_of, sym)
        except Exception:
            _chain = None

        _ctz = (_chain["ts"].dt.tz if (_chain is not None and len(_chain)) else None)

        def _prem(t, strike, side):
            if _chain is None or not len(_chain) or "ltp" not in _chain.columns or not strike:
                return None
            tq = pd.Timestamp(t)                               # bar times are tz-naive IST
            if _ctz is not None and tq.tz is None:             # chain ts is tz-aware → align
                tq = tq.tz_localize(_ctz)
            elif _ctz is None and tq.tz is not None:
                tq = tq.tz_localize(None)
            sub = _chain[(_chain["side"] == side) & (_chain["strike"] == strike)
                         & (_chain["ts"] <= tq)]
            if not len(sub):
                return None
            row = sub.sort_values("ts").iloc[-1]
            if (tq - row["ts"]).total_seconds() > 1200:        # stale >20min → n/a
                return None
            return round(float(row["ltp"]), 2)
        Lh = contL["high"].to_numpy(float); Ll = contL["low"].to_numpy(float)
        Lc = contL["close"].to_numpy(float); Lts = contL["ts"].to_numpy()
        Hh = contH["high"].to_numpy(float); Hl = contH["low"].to_numpy(float)
        Hc = contH["close"].to_numpy(float); Hts = contH["ts"].to_numpy()
        if contG is not None:
            Gh = contG["high"].to_numpy(float); Gl = contG["low"].to_numpy(float)
            Gc = contG["close"].to_numpy(float); Gts = contG["ts"].to_numpy()
        gjx = 0
        # today's LTF bars only (setups fired today)
        today_idx = [i for i in range(24, len(Lc))
                     if d0 is None or pd.Timestamp(Lts[i]).date() == d0]
        # tape-travel inputs: today's open + gap vs the prior session's last close
        _t0 = today_idx[0] if today_idx else None
        day_open = float(contL["open"].to_numpy(float)[_t0]) if _t0 is not None else None
        prev_close = float(Lc[_t0 - 1]) if (_t0 is not None and _t0 > 0) else None
        gap_pct = (abs(day_open / prev_close - 1) * 100.0
                   if (day_open and prev_close) else 0.0)
        hj = 0
        open_until = -1                                        # serialize: ONE position per index
        strikes: dict = {}                                     # (tag, side) -> fires today
        n_suppressed = 0
        _tf = pd.Timedelta(minutes=ltf_tf)                     # one LTF bar (close = start + _tf)
        for i in today_idx:
            if i <= open_until:                                # a trade is still live — no new fire
                continue
            while hj + 1 < len(Hts) and Hts[hj + 1] <= Lts[i]:
                hj += 1
            if hj < 24:
                continue
            ls = _struct_min(Lh, Ll, Lc, i)
            # ── LOOKAHEAD FIX (audit 2026-07-18): contH is fetched at day-end, so the hj
            # (current) HTF bar arrives COMPLETED — a 10:30 fire would read the 10:15-11:15
            # bar's full H/L/C (up to 45min of future). Rebuild the hj bar's PARTIAL OHLC
            # from the LTF bars up to the trigger, so structure/pivots/ATR see exactly what
            # the live board saw at that instant. ───────────────────────────────────────
            w_start = Hts[hj]
            j0 = i
            while j0 > 0 and Lts[j0 - 1] >= w_start:
                j0 -= 1
            Hh_t = Hh[:hj + 1].copy(); Hl_t = Hl[:hj + 1].copy(); Hc_t = Hc[:hj + 1].copy()
            Hh_t[hj] = float(Lh[j0:i + 1].max())
            Hl_t[hj] = float(Ll[j0:i + 1].min())
            Hc_t[hj] = float(Lc[i])
            hs = _struct_min(Hh_t, Hl_t, Hc_t, hj)
            if ls == "n/a" or hs == "n/a":
                continue
            spot = float(Lc[i])
            s = synthesize({"struct": hs, "hi": float(Hh_t[hj - 19:hj + 1].max()),
                            "lo": float(Hl_t[hj - 19:hj + 1].min()), "n": 30},
                           {"struct": ls, "n": 30}, spot)
            tag = s.get("tag", "")
            lean = ("UP" if "RANGE-TOP" in tag or hs in _TREND_UP_S else
                    "DOWN" if "RANGE-FLOOR" in tag or hs in _TREND_DN_S else None)
            if lean is None or not any(k in tag for k in
                                       ("CONTINUATION", "BREAK (attempt)", "PULLBACK")):
                continue
            # ── OPT-IN GRANDPARENT GATE: the 3rd frame must be actively trending WITH the
            # trade. Its current bar is TRUNCATED to this instant exactly like the HTF —
            # a completed coarse bar would leak up to 4h of future (the lookahead that
            # inflated the first version of this very test). ─────────────────────────────
            if contG is not None:
                while gjx + 1 < len(Gts) and Gts[gjx + 1] <= Lts[i]:
                    gjx += 1
                if gjx < _STRUCT_LB:
                    n_suppressed += 1
                    continue
                g0 = i
                while g0 > 0 and Lts[g0 - 1] >= Gts[gjx]:
                    g0 -= 1
                Gh_t = Gh[:gjx + 1].copy(); Gl_t = Gl[:gjx + 1].copy(); Gc_t = Gc[:gjx + 1].copy()
                Gh_t[gjx] = float(Lh[g0:i + 1].max())
                Gl_t[gjx] = float(Ll[g0:i + 1].min())
                Gc_t[gjx] = float(Lc[i])
                gs = _struct_min(Gh_t, Gl_t, Gc_t, gjx)
                aligned = ((lean == "UP" and gs in _TREND_UP_S) or
                           (lean == "DOWN" and gs in _TREND_DN_S))
                if not aligned:
                    n_suppressed += 1
                    continue
            # ── GUARD 1 · DEAD-TAPE: no new fires while the tape hasn't travelled. travel =
            # max(today's range-so-far, |gap|) — the gap IS travel (a +3.9% gap day is alive
            # at 09:45 even before intraday range builds). ────────────────────────────────
            if day_open:
                rng_so_far = (float(Lh[_t0:i + 1].max()) - float(Ll[_t0:i + 1].min())) \
                    / day_open * 100.0
                if max(rng_so_far, gap_pct) < _TAPE_MIN_PCT:
                    n_suppressed += 1
                    continue
            # ── GUARD 2 · THREE-STRIKES: the same idea (setup+side) failing twice = the market
            # said no; don't re-buy the same failed break all day (06-22 = 7 serial fires). ──
            skey = (tag[:14], lean)
            if strikes.get(skey, 0) >= _MAX_STRIKES:
                n_suppressed += 1
                continue
            his, los = _pivots(Hh_t[max(0, hj - _SL_WIN):hj + 1],
                               Hl_t[max(0, hj - _SL_WIN):hj + 1], w=3)
            atr = float(np.mean(Hh_t[hj - 13:hj + 1] - Hl_t[hj - 13:hj + 1]))
            md = 0.25 * atr
            res = min((x for x in his if x > spot + md), default=spot + atr)
            sup = max((x for x in los if x < spot - md), default=spot - atr)
            entry = spot
            band_hi, band_lo = spot + atr, spot - atr
            # SAME EXIT VOCABULARY AS THE CHARTS SCOUT LEDGER: band-touch (take-profit at the
            # band edge) / SL hit (structure stop) / flipped (LTF structure reverses) / timed
            # out (90m) / squared off at the bell. target = band edge, stop = structure S/R.
            target, stop = (band_hi, sup) if lean == "UP" else (band_lo, res)
            # ── SL-MODE experiments: the ENTRY-TF (15m) structure stop — chartist-proper
            # geometry (a 15m entry wearing a 60m stop = R:R < 1, observed live). ─────────
            if sl_mode in ("ltf", "ltf_trail"):
                atr_l = float(np.mean(Lh[i - 13:i + 1] - Ll[i - 13:i + 1]))
                his_l, los_l = _pivots(Lh[max(0, i - 30):i + 1], Ll[max(0, i - 30):i + 1], w=2)
                if lean == "UP":
                    stop = max((x for x in los_l if x < entry - 0.25 * atr_l),
                               default=entry - atr_l)
                else:
                    stop = min((x for x in his_l if x > entry + 0.25 * atr_l),
                               default=entry + atr_l)
            if abs(entry - stop) < 1e-6:
                continue
            hb = min(i + hold, len(Lc) - 1)                       # 90m hold (hold=6 x 15m)
            outcome, exitpx, exit_i = "open", None, None
            for j in range(i + 1, hb + 1):
                if lean == "UP":
                    if Ll[j] <= stop:
                        outcome, exitpx, exit_i = "SL hit", stop, j; break
                    if Lh[j] >= target:
                        outcome, exitpx, exit_i = "band ↑ upper", target, j; break
                else:
                    if Lh[j] >= stop:
                        outcome, exitpx, exit_i = "SL hit", stop, j; break
                    if Ll[j] <= target:
                        outcome, exitpx, exit_i = "band ↓ lower", target, j; break
                # TRAILING (ltf_trail): after bar j COMPLETES, ratchet the stop behind the
                # newest favorable 15m swing — tighten only, never loosen. Touch checks above
                # still see the forming bar; the ratchet itself uses completed structure.
                if sl_mode == "ltf_trail":
                    _jc2 = pd.Timestamp(Lts[j]) + _tf
                    _ok = True
                    if as_of is not None:
                        _n2 = pd.Timestamp(as_of).tz_localize(None) if pd.Timestamp(as_of).tz \
                            else pd.Timestamp(as_of)
                        _ok = _jc2 <= _n2
                    if _ok:
                        hs2, ls2 = _pivots(Lh[max(0, j - 30):j + 1], Ll[max(0, j - 30):j + 1], w=2)
                        if lean == "UP":
                            nw = max((x for x in ls2 if x < float(Lc[j])), default=None)
                            if nw is not None and nw > stop:
                                stop = nw
                        else:
                            nw = min((x for x in hs2 if x > float(Lc[j])), default=None)
                            if nw is not None and nw < stop:
                                stop = nw
                # FLIP — the LTF structure reverses against the position → exit at that close.
                # CLOSE-based read → COMPLETED bars only (a forming bar's structure repaints;
                # SL/band above stay live on the forming bar because high/low are monotonic).
                if as_of is not None:
                    _jc = pd.Timestamp(Lts[j]) + _tf
                    _nn = pd.Timestamp(as_of).tz_localize(None) if pd.Timestamp(as_of).tz \
                        else pd.Timestamp(as_of)
                    if _jc > _nn:
                        continue
                js = _struct_min(Lh, Ll, Lc, j)
                if (lean == "UP" and js in _TREND_DN_S) or (lean == "DOWN" and js in _TREND_UP_S):
                    outcome, exitpx, exit_i = "flipped · reversed out", float(Lc[j]), j; break
            if outcome == "open" and hb >= i + hold:
                # hold is in LTF bars → the wall-clock label must scale with the combo
                # (15m×6=90m default, but 5m×6=30m / 10m×6=60m on the other combos)
                outcome, exitpx, exit_i = f"timed out ({hold * ltf_tf}m)", float(Lc[hb]), hb
            elif outcome == "open" and as_of is not None and as_of.time() >= _MKT_CLOSE:
                outcome, exitpx, exit_i = "squared off at the bell", float(Lc[hb]), hb
            step = STRIKE_STEP.get(sym)
            atm = int(round(entry / step) * step) if step else None
            side = "CE" if lean == "UP" else "PE"
            # ENTRY happens at the setup bar's CLOSE (the bar must close to compute the setup) —
            # so timestamp + premium are read at bar-close (Lts[i] is the bar OPEN), and index
            # entry (Lc[i]) is already the close. This keeps entry index ↔ entry premium ↔ the
            # displayed time all at the SAME instant. Cap at the 15:30 bell.
            _tf = pd.Timedelta(minutes=ltf_tf)
            entry_ts = pd.Timestamp(Lts[i]) + _tf             # entry = setup bar CLOSE
            if entry_ts.time() >= _MKT_CLOSE:                  # can't enter at/after the bell
                continue
            # CLOSED-BAR DOCTRINE (fix 2026-07-20): never FIRE on the forming bar — structure
            # on a partial close REPAINTS (a trade could appear mid-bar and vanish if the bar
            # retreats). Touch-exits below still scan the forming bar (high/low are monotonic
            # — a pierce can't un-pierce), so SL/band detection stays ~real-time.
            if as_of is not None:
                _now_n = pd.Timestamp(as_of).tz_localize(None) if pd.Timestamp(as_of).tz \
                    else pd.Timestamp(as_of)
                if entry_ts > _now_n:
                    continue
            # OPENING WARMUP (standing rule, same as intraday_scout._OPEN_SETTLE): no trigger
            # before 09:35 — the first 20min are auction noise / the opening range forming;
            # the continuous-bar warmup is satisfied by PRIOR days so without this gate a gap
            # day could fire on the 09:15-09:30 bar itself (observed live 2026-07-20).
            if entry_ts.time() < scout._OPEN_SETTLE:
                continue
            e_prem = _prem(entry_ts, atm, side)                # premium AT ENTRY (bar close)
            r = {"sym": sym, "label": LABELS.get(sym, sym), "lean": lean, "side": side,
                 "strike": atm, "tag": tag, "e_prem": e_prem,
                 "since": entry_ts.strftime("%H:%M"),
                 "entry": round(entry, 1), "target": round(target, 1), "sl": round(stop, 1),
                 "outcome": outcome}

            def _opt_pnl(x_prem):
                if e_prem and x_prem:
                    return round((x_prem / e_prem - 1) * 100.0, 1), round((x_prem - e_prem) * lot)
                return None, None
            if outcome == "open":
                cur = float(Lc[-1])
                r["now"] = round(cur, 1)
                r["x_prem"] = _prem(as_of, atm, side)          # premium NOW (if chain still live)
                r["opt_pct"], r["opt_rs"] = _opt_pnl(r["x_prem"])
                r["pnl_pct"] = round(((cur - entry) if lean == "UP" else (entry - cur))
                                     / entry * 100.0, 2)
                r["rmult"] = round(((cur - entry) if lean == "UP" else (entry - cur))
                                   / abs(entry - stop), 2)
            else:
                r["exit"] = round(exitpx, 1)
                exit_ts = min(pd.Timestamp(Lts[exit_i]) + _tf, pd.Timestamp(Lts[exit_i]).normalize()
                              + pd.Timedelta(hours=15, minutes=30))
                r["x_prem"] = _prem(exit_ts, atm, side)        # premium AT EXIT (bar close)
                r["opt_pct"], r["opt_rs"] = _opt_pnl(r["x_prem"])
                r["held"] = f"{r['since']}→{exit_ts.strftime('%H:%M')}"
                r["pnl_pct"] = round(((exitpx - entry) if lean == "UP" else (entry - exitpx))
                                     / entry * 100.0, 2)
                r["rmult"] = round(((exitpx - entry) if lean == "UP" else (entry - exitpx))
                                   / abs(entry - stop), 2)
            L.setdefault(sym, []).append(r)
            strikes[skey] = strikes.get(skey, 0) + 1           # only ACTUAL fires count a strike
            # block new setups until THIS position resolves (closed → its exit bar; open → the
            # full hold window) — one position per index at a time, like the Charts scout.
            open_until = exit_i if exit_i is not None else hb
        SUP += n_suppressed
    rows = [r for v in L.values() for r in v]
    closed = [r for r in rows if r["outcome"] != "open"]
    openr = [r for r in rows if r["outcome"] == "open"]
    wins = sum(1 for r in closed if r["rmult"] > 0)             # profitable exit = win
    avg_r = round(np.mean([r["rmult"] for r in closed]), 3) if closed else None
    avg_pct = round(np.mean([r["pnl_pct"] for r in closed]), 2) if closed else None
    return {"open": openr, "closed": closed, "n_closed": len(closed),
            "wins": wins, "avg_r": avg_r, "avg_pct": avg_pct, "suppressed": SUP}


def _struct_min(h, l, c, i, lb=_STRUCT_LB):
    """Minimal structure label at bar i on arrays (for the ledger/backtest)."""
    if i < lb:
        return "n/a"
    seg = c[i - lb + 1:i + 1]; net = seg[-1] - seg[0]
    path = np.abs(np.diff(seg)).sum(); er = abs(net) / path if path > 0 else 0.0
    hh = h[i - lb + 1:i + 1]; ll = l[i - lb + 1:i + 1]; last = c[i]
    if last > hh[:-1].max():
        return "BREAKOUT_UP"
    if last < ll[:-1].min():
        return "BREAKOUT_DOWN"
    if er >= 0.4:
        return "TREND_UP" if net > 0 else "TREND_DOWN"
    prior = hh[:-3].max() - ll[:-3].min(); recent = hh[-3:].max() - ll[-3:].min()
    if prior > 0 and recent < 0.6 * prior:
        return "CONSOLIDATION"
    return "RANGE"


def _walls(h, l, tol: float):
    """TOUCH-COUNTED S/R: cluster swing pivots within `tol` price units → [(level, touches)].
    A 3-touch cluster = a wall the market rejected three times; 1-touch = a mere pivot. The
    chartist's dynamic S/R from the past bars, with strength = rejection count."""
    his, los = _pivots(h, l, w=2)
    lv = []
    for x in sorted(his + los):
        for c in lv:
            if abs(x - c[0]) <= tol:
                c[0] = (c[0] * c[1] + x) / (c[1] + 1); c[1] += 1
                break
        else:
            lv.append([x, 1])
    return [(round(x, 1), t) for x, t in lv]


def _pivots(h, l, w: int = 2):
    """Swing highs / lows (a bar whose high/low is the extreme of ±w neighbours)."""
    his, los = [], []
    for i in range(w, len(h) - w):
        if h[i] >= h[i - w:i + w + 1].max() - 1e-9:
            his.append(float(h[i]))
        if l[i] <= l[i - w:i + w + 1].min() + 1e-9:
            los.append(float(l[i]))
    return his, los


def _scout_levels(sym, ltf_tf, htf_tf, tag, htf, ltf, spot, date, as_of) -> dict:
    """Band + S/R (from past candle bars) + strike + entry/target/SL for the scout card. The
    S/R levels are the PA edge; the ATM CE/PE strike is the VEHICLE, flagged negative-EV (the
    naked arrow bleeds — audited); E/T/SL sit on the LEVELS, not a directional option bet.
    `spot` = the FRESH LTF-close price (NOT the up-to-60m-stale HTF close)."""
    cont = _bars_continuous(sym, htf_tf, date, as_of, need=40)
    if cont is None or len(cont) < 8:
        return {}
    h = cont["high"].to_numpy(float)[:-1]; l = cont["low"].to_numpy(float)[:-1]
    if not spot:                                   # fall back to HTF close only if no fresh spot
        spot = float(cont["close"].to_numpy(float)[-2])
    spot = float(spot)
    # ── 60m expected-range band ≈ spot ± ATR (a lightweight risk map; the validated ~70% band
    #    lives in hour_forecast, this is a fast structural proxy for the scout) ──
    atr = float(np.mean(h[-14:] - l[-14:])) if len(h) >= 14 else float(np.mean(h - l))
    band_lo, band_hi = round(spot - atr, 1), round(spot + atr, 1)
    # ── TOUCH-COUNTED support/resistance (clustered pivots over the past ~30 HTF bars) —
    #    the chartist's dynamic S/R with strength = rejection count. Nearest wall each side
    #    beyond a 0.25-ATR min-distance (no micro-swings hugging price). ──────────────────
    lv = _walls(h[-31:], l[-31:], 0.2 * atr)
    # ── the LTF (entry-TF) wall set — the chartist's 15m rejections. A 60m bar SWALLOWS
    #    four 15m swings, so intraday multi-rejection levels are often invisible to HTF
    #    pivots. Computed over ~30 LTF bars (~1.5 sessions), own-ATR cluster tol. ─────────
    lv_l = []
    contL = _bars_continuous(sym, ltf_tf, date, as_of, need=40)
    if contL is not None and len(contL) >= 12:
        hh2 = contL["high"].to_numpy(float)[:-1][-31:]
        ll2 = contL["low"].to_numpy(float)[:-1][-31:]
        atr_l = float(np.mean(hh2[-14:] - ll2[-14:])) if len(hh2) >= 14 else 0.0
        if atr_l > 0:
            lv_l = _walls(hh2, ll2, 0.2 * atr_l)
    md = 0.25 * atr
    above = [(x, t) for x, t in lv if x > spot + md]
    below = [(x, t) for x, t in lv if x < spot - md]
    res, res_t = min(above, key=lambda c: c[0]) if above else (None, 0)
    sup, sup_t = max(below, key=lambda c: c[0]) if below else (None, 0)
    step = STRIKE_STEP.get(sym)
    atm = int(round(spot / step) * step) if step else None
    # ── directional lean: a RANGE break is directional even when the HTF is consolidating, so
    #    take the break side from the TAG first; else fall back to the HTF trend. ──
    if "RANGE-TOP" in tag:
        lean = "UP"
    elif "RANGE-FLOOR" in tag:
        lean = "DOWN"
    elif htf.get("struct") in _TREND_UP_S:
        lean = "UP"
    elif htf.get("struct") in _TREND_DN_S:
        lean = "DOWN"
    else:
        lean = "NONE"
    up, dn = lean == "UP", lean == "DOWN"
    side = "CE" if up else "PE" if dn else "—"
    # ── entry / target / SL — MIRRORS THE GRADED LEDGER (audit 2026-07-20: the display was
    # advertising target=resistance-pivot R:R while the graded ruleset targets the BAND edge;
    # a shown R:R 3.85 vs graded 1.77 misleads the decision). target = band edge in the trade
    # direction, SL = structure pivot — exactly what the ledger books. ────────────────────
    entry = target = sl = None
    if "CONTINUATION" in tag or "BREAK (attempt)" in tag:
        if up:
            entry, target, sl = spot, band_hi, (sup or band_lo)
        elif dn:
            entry, target, sl = spot, band_lo, (res or band_hi)
    elif "PULLBACK" in tag:
        if up:                                    # buy the dip toward support
            entry, target, sl = (sup or band_lo), band_hi, round((sup or band_lo) * 0.997, 1)
        elif dn:
            entry, target, sl = (res or band_hi), band_lo, round((res or band_hi) * 1.003, 1)
    # RANGE / SQUEEZE / TRAP → no directional entry (wait / fade the band, handled in text)
    rr = None
    if entry and target and sl and abs(entry - sl) > 0:
        rr = round(abs(target - entry) / abs(entry - sl), 2)
    # headroom to the nearest MULTI-TOUCH wall in the TRADE's direction (ATR units) — display
    # context so the human can judge 'buying INTO the wall' vs 'buying THE BREAK of it'
    # (measured 25d: near-wall trades did NOT bleed — often the break IS the setup — so no
    # veto; the distinction is the chartist's call, shown not enforced).
    # AUDIT FIX: computed over ALL >=2-touch clusters WITHOUT the 0.25-ATR display filter —
    # that filter was HIDING the very closest (most dangerous) walls from the warning.
    # warn scans BOTH TF wall-sets (a 15m triple-rejection 0.2 ATR overhead matters even when
    # the 60m shows clear road); distance normalized in HTF-ATR (the trade's own scale).
    head = None; warn_level = None; warn_touches = 0; warn_tf = None
    if atr and lean in ("UP", "DOWN"):
        cands = ([(x, t, htf_tf) for x, t in lv if t >= 2] +
                 [(x, t, ltf_tf) for x, t in lv_l if t >= 2])
        mt = [(x, t, tf) for x, t, tf in cands if
              ((lean == "UP" and x > spot) or (lean == "DOWN" and x < spot))]
        if mt:
            wx, wt, wtf = min(mt, key=lambda c: abs(c[0] - spot))
            head = round(abs(wx - spot) / atr, 2)
            warn_level, warn_touches, warn_tf = wx, wt, wtf
    wall_warn = bool(head is not None and head < 0.5)
    # LTF nearest S/R (display) + ◈ CONFLUENCE flag (15m wall sitting ON a 60m wall = the
    # strongest level — the system's one validated level-finding: confluence works).
    l_above = [(x, t) for x, t in lv_l if x > spot]
    l_below = [(x, t) for x, t in lv_l if x < spot]
    res_l, res_l_t = min(l_above, key=lambda c: c[0]) if l_above else (None, 0)
    sup_l, sup_l_t = max(l_below, key=lambda c: c[0]) if l_below else (None, 0)
    conf_tol = 0.25 * atr if atr else 0
    confluence = [round(xl, 1) for xl, _ in lv_l
                  for xh, _ in lv if abs(xl - xh) <= conf_tol]
    return {
        "spot": round(spot, 1), "band_lo": band_lo, "band_hi": band_hi,
        "support": round(sup, 1) if sup else None, "resistance": round(res, 1) if res else None,
        "sup_touches": sup_t, "res_touches": res_t,
        "sup_l": sup_l, "sup_l_t": sup_l_t, "res_l": res_l, "res_l_t": res_l_t,
        "confluence": confluence[:3],
        "headroom_atr": head, "wall_warn": wall_warn,
        "warn_level": warn_level, "warn_touches": warn_touches, "warn_tf": warn_tf,
        "atm": atm, "side": side, "lean": lean,
        "entry": round(entry, 1) if entry else None,
        "target": round(target, 1) if target else None,
        "sl": round(sl, 1) if sl else None, "rr": rr,
    }


def day_regime(date, as_of, sym: str = "NSE:NIFTY50-INDEX") -> dict:
    """DAY-TYPE chip (display only). Kaufman ER on today's NIFTY 15m closes — morning (by
    11:00, causal) and session-to-date — plus range%. MEASURED 28d: every combo bleeds on
    CHOP/MID days; ALL the money is on TREND days. Morning ER does NOT forecast the day
    (corr +0.06, 3/13 AM-trend days stayed trend) — so this chip is a STATE reading, never a
    forecast and never an auto-gate."""
    try:
        c = _bars_continuous(sym, 15, date, as_of, need=120)   # ~5 prior sessions for the vol base
        if c is None or not len(c):
            return {}
        d0 = (datetime.date.fromisoformat(date) if isinstance(date, str)
              else (date or pd.Timestamp(as_of).date()))
        ts = pd.to_datetime(c["ts"])
        g = c[(ts.dt.date == d0).values]
        if len(g) < 1:
            return {}
        if len(g) < 3:                       # 1-2 bars: show WARMING, not a blank slot
            _o = float(g["open"].iloc[0])
            return {"label": "WARMING", "bars": len(g),
                    "rng": round((float(g["high"].max()) - float(g["low"].min())) / _o * 100, 2)
                    if _o else 0.0}
        # WARM-UP GUARD (audit 2026-07-23): ER on 3 closes is ~1.0 BY CONSTRUCTION (bias ≈
        # 1/sqrt(n)) — the raw chip screamed "TREND · EXPANDING x2" every single morning, i.e.
        # it was most wrong exactly when it's most read. No label until 6 bars (~10:45).
        n_today = len(g)
        seg = g["close"].to_numpy(float)
        er = abs(seg[-1] - seg[0]) / max(np.abs(np.diff(seg)).sum(), 1e-9)
        if n_today < 6:
            return {"label": "WARMING", "bars": n_today,
                    "rng": round((float(g["high"].max()) - float(g["low"].min()))
                                 / float(g["open"].iloc[0]) * 100, 2)
                    if float(g["open"].iloc[0]) else 0.0}
        am = g[pd.to_datetime(g["ts"]).dt.time <= datetime.time(11, 0)]
        s2 = am["close"].to_numpy(float)
        er_am = (abs(s2[-1] - s2[0]) / max(np.abs(np.diff(s2)).sum(), 1e-9)
                 if len(s2) > 2 else None)
        o = float(g["open"].iloc[0])
        rng = (float(g["high"].max()) - float(g["low"].min())) / o * 100 if o else 0.0
        lab = "TREND" if er >= 0.35 else ("CHOP" if er < 0.20 else "MID")
        # ── VOL STATE: today's live bar-size vs the PRIOR sessions' baseline (ATR14 on the
        # same 15m grid). Ratio > 1.25 = expansion (bars getting bigger — stops/targets are
        # auto-widening with it), < 0.80 = compression (coil/dead tape). Display only. ────
        rows_today = (ts.dt.date == d0).values
        prior = c[~rows_today]
        vr = vnow = None
        if len(g) >= 8 and len(prior) >= 20:     # >=8 bars: ATR14-ish needs real sample
            vnow = float(np.mean((g["high"] - g["low"]).to_numpy(float)[-14:]))
            base = float(np.mean((prior["high"] - prior["low"]).to_numpy(float)[-70:]))
            if base > 0:
                vr = vnow / base
        vlab = ("EXPANDING" if vr and vr >= 1.25 else
                "COMPRESSING" if vr and vr < 0.80 else "NORMAL" if vr else None)
        return {"er": round(er, 2), "er_am": round(er_am, 2) if er_am is not None else None,
                "rng": round(rng, 2), "label": lab, "bars": n_today,
                "vol_ratio": round(vr, 2) if vr else None, "vol_label": vlab,
                "atr_pts": round(vnow, 1) if vnow else None}
    except Exception:
        return {}


def scout_scan(date, as_of, ltf_tf: int, htf_tf: int, gp_only: bool = False) -> list[dict]:
    """Cross-index price-action SCOUT: scans all indices on the chosen combo (LTF entry × HTF
    confirm), returns each index's structure + candle pattern (both TFs) + the synthesis
    verdict, RANKED best-setup-first. A discretionary-read scanner (which index has the
    cleanest MTF confluence) — NOT a machine buy (intraday PA does not mechanize; audited)."""
    rows = []
    for sym in INDEX_SYMBOLS:
        try:
            htf = _struct_full(sym, htf_tf, date, as_of, drop_forming=False)
            ltf = _struct_full(sym, ltf_tf, date, as_of)
            spot = ltf.get("last") or 0
            s = synthesize(htf, ltf, spot)
            lv = _scout_levels(sym, ltf_tf, htf_tf, s.get("tag", ""), htf, ltf, spot, date, as_of)
            # OPT-IN grandparent state (3rd frame). Computed ONLY when asked, so the default
            # render pays nothing. drop_forming=False = same live read as the HTF confirm.
            gp_struct = gp_aligned = None
            if gp_only and _GP_OF.get(htf_tf):
                gp = _struct_full(sym, _GP_OF[htf_tf], date, as_of, drop_forming=False)
                gp_struct = gp.get("struct")
                _ln = lv.get("lean")
                gp_aligned = bool((_ln == "UP" and gp_struct in _TREND_UP_S) or
                                  (_ln == "DOWN" and gp_struct in _TREND_DN_S))
            # DEAD-TAPE state (guard 1, live): travel = max(today's range-so-far, |gap|)
            tape_pct, tape_dead = None, False
            cont5 = _bars_continuous(sym, ltf_tf, date, as_of, need=60)
            if cont5 is not None and len(cont5) > 2:
                dts = pd.to_datetime(cont5["ts"]).dt.date
                today = dts.iloc[-1]
                td = cont5[dts == today]
                if len(td) >= 2:
                    d_open = float(td["open"].iloc[0])
                    rng = (float(td["high"].max()) - float(td["low"].min())) / d_open * 100.0
                    prior = cont5[dts < today]
                    gap = (abs(d_open / float(prior["close"].iloc[-1]) - 1) * 100.0
                           if len(prior) else 0.0)
                    tape_pct = round(max(rng, gap), 2)
                    tape_dead = tape_pct < _TAPE_MIN_PCT
            rows.append({
                "sym": sym, "label": LABELS.get(sym, sym),
                "ltf_struct": ltf.get("struct"), "ltf_pattern": ltf.get("pattern"),
                "ltf_er": ltf.get("er"), "htf_struct": htf.get("struct"),
                "htf_pattern": htf.get("pattern"), "htf_er": htf.get("er"),
                "tag": s.get("tag", ""), "read": s.get("read", ""),
                "color": s.get("color", "#94a3b8"), "loc": s.get("loc"),
                "levels": lv, "tape_pct": tape_pct, "tape_dead": tape_dead,
                "gp_tf": _GP_OF.get(htf_tf), "gp_struct": gp_struct, "gp_aligned": gp_aligned,
            })
        except Exception:
            rows.append({"sym": sym, "label": LABELS.get(sym, sym), "tag": "n/a",
                         "color": "#64748b"})
    rows.sort(key=lambda r: _TAG_RANK.get(r.get("tag", ""), 9))
    return rows


def _synth_for(sym: str, date, as_of, spot: float) -> dict:
    """HTF x LTF synthesis for the board. HTF = 60m (fall back to 30m if 60m has too few
    closed bars intraday — the 60m warms only late morning); LTF = 15m."""
    htf = _struct_full(sym, 60, date, as_of, drop_forming=False)
    htf_tf = 60
    if htf.get("n", 0) < 6:
        htf = _struct_full(sym, 30, date, as_of, drop_forming=False)
        htf_tf = 30
    ltf = _struct_full(sym, 15, date, as_of)
    out = synthesize(htf, ltf, spot)
    out["htf_tf"], out["ltf_tf"] = htf_tf, 15
    out["htf_struct"], out["ltf_struct"] = htf.get("struct"), ltf.get("struct")
    out["htf_hi"], out["htf_lo"] = htf.get("hi"), htf.get("lo")
    return out


def build_row(sym: str, date, as_of) -> dict:
    short = pf._fy_key(sym)
    short = scout_short = pf.FY.get(short, short)
    r = scout.scan_index(sym, 5, date=date, as_of=as_of, horizon_min=45)
    label = LABELS.get(sym, sym)
    if not r.get("has_data") or not r.get("spot"):
        return {"sym": sym, "label": label, "verdict": "WARMING", "note": "no data yet"}
    spot, atm = r["spot"], r.get("atm")
    last, age = _chain_last(sym, date, as_of)
    stale = age is None or age > _STALE_MIN
    dte = pf.days_to_expiry(date, weekly=(scout_short == "NIFTY50"))
    row = {
        "sym": sym, "label": label, "spot": spot, "atm": atm,
        "mood": r.get("mood"), "mood_full": r.get("mood_full"), "er": r.get("mood_er"),
        "htf15": _structure(sym, 15, date, as_of), "htf60": _structure(sym, 60, date, as_of),
        "mtf": _mtf(sym, date, as_of),
        "synth": _synth_for(sym, date, as_of, spot),
        "combos": _synth_combos(sym, date, as_of, spot),
        "daily": _daily_read(sym, as_of),
        "band_lo": r.get("pred_lo"), "band_hi": r.get("pred_hi"),
        "cover": r.get("band_cover"), "cover_conf": r.get("band_conf"),
        "liquid": scout_short in _FADE_LIQUID,
        "strikes": _strike_table(last, atm, sym),
        "chain": _chain_stats(last, spot, sym),
        "chain_age": round(age, 1) if age is not None else None, "chain_stale": stale,
        "expiry_dte": dte, "weekly": scout_short == "NIFTY50",
        "setup": None, "opt": None, "verdict": "NO-TRADE", "note": "",
    }
    # ── the ONE actionable setup: the band-fade rejection (paper_fade_logger, parity) ──
    sig = pf._signal(sym, date, as_of)
    age = _chain_age_min(sym, date, as_of)
    stale = age is None or age > _STALE_MIN
    if sig is not None:
        s_atm = sig["atm"] or atm
        opt_side = "PE" if sig["side"] == "S" else "CE"        # fade short → buy PE
        risk = abs(sig["entry"] - sig["stop"])
        reward = abs(sig["target"] - sig["entry"])
        row["setup"] = {
            "side": "SHORT-fade" if sig["side"] == "S" else "LONG-fade",
            "strike": s_atm, "opt_side": opt_side,
            "entry": sig["entry"], "target": sig["target"], "sl": sig["stop"],
            "rr": round(reward / risk, 2) if risk else None,
            "band_pct": sig["band_pct"], "clr": sig["clr"], "er": sig["er"],
        }
        row["opt"] = pf._opt_ctx(sym, date, as_of, s_atm, sig["side"])
        row["opt"]["age_min"] = round(age, 1) if age is not None else None
        row["opt"]["stale"] = stale
        # 2026-07-16 REALIZABLE-FILL AUDIT: the fade's "+REJECT" edge was a LOOKAHEAD artifact
        # — it books a fill at the band EDGE on a trade selected by the touch-bar CLOSE (info
        # not known until after that fill). Enter at the realizable price (the confirmed-
        # rejection close) and it loses -6 to -11bps HARD on EVERY index. So the fade is NOT a
        # tradeable suggestion. It is shown as a CONTEXT MARKER (price stretched + rejected at
        # the band edge — read it with your own PA); the verdict is never "trade this".
        row["setup"]["context_only"] = True
        row["verdict"] = "RANGE-ONLY"
        row["note"] = ("stretch+rejection MARKER (context for your own read) — NOT a trade: "
                       "at realizable fills the fade loses ~-6 to -11bps (edge was a fill "
                       "artifact). Band = the risk map; no validated intraday direction.")
    else:
        # no setup → the validated RANGE band is the product; default posture is NO-TRADE
        if row["cover"] is not None:
            row["verdict"] = "RANGE-ONLY"
            row["note"] = "no fade trigger — trade the band as a risk map, not a direction"
        else:
            row["verdict"] = "NO-TRADE"
            row["note"] = "no setup, band not yet calibrated"
    return row


def build_board(date=None, as_of=None) -> list[dict]:
    """One row per index, FAULT-ISOLATED — a bad index degrades to a WARMING row rather
    than killing the whole board."""
    out = []
    for sym in INDEX_SYMBOLS:
        try:
            out.append(build_row(sym, date, as_of))
        except Exception as exc:
            out.append({"sym": sym, "label": LABELS.get(sym, sym), "verdict": "WARMING",
                        "note": f"row error: {exc}"})
    return out


def render_cli(rows: list[dict], as_of) -> None:
    ts = as_of.strftime("%Y-%m-%d %H:%M") if as_of else "LIVE"
    print("\n" + "=" * 108)
    print(f"  LIVE TRADEBOARD  ·  {ts} IST   (band = validated risk product · fade = only "
          f"mechanizable edge · default NO-TRADE)")
    print("=" * 108)
    for r in rows:
        if r["verdict"] == "WARMING":
            print(f"  {r['label']:<11} WARMING — {r.get('note','')}"); continue
        cov = f"{100*r['cover']:.0f}%{r['cover_conf'][:1]}" if r.get("cover") is not None else "—"
        print(f"\n  {r['label']:<11} {r['spot']:>10.2f}  mood {r['mood']:<6} "
              f"(ER {r['er'] if r['er'] is not None else '—'})   "
              f"HTF 15m={r['htf15']} 60m={r['htf60']}")
        print(f"    band [{r['band_lo']}, {r['band_hi']}] cover {cov}   "
              f"→ {r['verdict']}")
        s = r.get("setup")
        if s:
            print(f"    SETUP {s['side']}  strike {s['strike']} {s['opt_side']}  "
                  f"entry {s['entry']}  target {s['target']}  SL {s['sl']}  "
                  f"R:R {s['rr']}  (band {s['band_pct']}% · clr {s['clr']} · ER {s['er']})")
            o = r.get("opt") or {}
            if o.get("stale"):
                print(f"    OPT ctx STALE ({o.get('age_min')}m old) — not scored")
            else:
                print(f"    OPT {s['strike']}{s['opt_side']}  OI {o.get('fade_oi')}  "
                      f"COI {o.get('fade_oich')}  vol {o.get('fade_vol')}  "
                      f"prem {o.get('fade_prem')}  IV — · wall {o.get('wall_dist_pct')}%  "
                      f"PCR {o.get('pcr')}  DTE {o.get('dte')}")
        if r.get("note"):
            print(f"    · {r['note']}")
    print("\n" + "=" * 108)
    print("  READ: TRADE-FADE only on liquid BANK/NIFTY, and it is a FUTURES mean-reversion —")
    print("  the option strike is the vehicle, not the edge (theta eats it). Everything else is")
    print("  RANGE-ONLY (band = where price likely sits) or NO-TRADE. No directional arrow: dead.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay")
    ap.add_argument("--at", default="13:30", help="HH:MM for --replay")
    a = ap.parse_args()
    if a.replay:
        hh, mm = map(int, a.at.split(":"))
        d0 = datetime.date.fromisoformat(a.replay)
        as_of = datetime.datetime.combine(d0, datetime.time(hh, mm), tzinfo=IST)
        rows = build_board(a.replay, as_of)
    else:
        as_of = datetime.datetime.now(IST)
        rows = build_board(None, as_of)
    render_cli(rows, as_of)


if __name__ == "__main__":
    main()
