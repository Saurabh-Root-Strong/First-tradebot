"""
footprint_chart.py — full-session OI · Volume · ATM-premium time series for a
single index at a chosen timeframe (the data behind the click-to-popup chart on
the Intraday Footprint panel).

Reads the SAME lock-free parquet mirrors the footprint panel uses
(core.mirror_io.read_mirror), so the chart is consistent with the panel numbers
and replays any captured day via `as_of`. Pure data layer — no plotly, no Dash.

Series produced (whole session 9:15 -> now, resampled to tf-minute bars):
  open/high/low/close  underlying price OHLC per bar  — candlestick price
  spot     underlying close per bar     — price context (line)
  premium  ATM straddle (CE+PE of the strike nearest spot, per bar)  — IV/decay pulse
  oi_ce    total call OI  (lakh)        — ceiling / call-writing
  oi_pe    total put OI   (lakh)        — floor / put-writing
  volume   traded option volume in the bar (lakh)  — activity
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

from core.constants import IST, LIVE_DIR, NSE_NAME, STRIKE_DISPLAY_STEP, STRIKE_STEP, today_iso
from core.mirror_io import read_mirror as _read_raw

# ── full-day read cache (mtime-keyed) ────────────────────────────────────────────
# core.mirror_io caches PAST days but reads TODAY fresh on every call (the live file
# grows). The scout lifecycle walk-back calls build_series ~120× per render at
# different as_of cutoffs of the SAME day → 120 disk reads + re-parses of today's
# mirrors = a ~50s render (the live "scout stuck loading" bug). Here we cache the
# FULL-day frame once per (table,date,symbol), keyed by the file's (mtime,size) so a
# live append refreshes it, and apply the as_of cutoff IN MEMORY. Identical result to
# read_mirror (which just filters ts<=as_of), but the 120 walk-back reads collapse to
# one. Local to this module — global read_mirror / the capture box are untouched.
# Bounded: one entry per (table,date,symbol) (~a dozen), each a per-symbol frame.
_FULL_CACHE: dict = {}


def _read(tbl: str, date=None, as_of=None, symbol=None):
    day = date or today_iso()
    p = LIVE_DIR / f"{day}_{tbl}.parquet"
    try:
        st = p.stat()
        sig = (st.st_mtime_ns, st.st_size)
    except OSError:
        return None
    key = (tbl, day, symbol)
    hit = _FULL_CACHE.get(key)
    if hit is None or hit[0] != sig:
        full = _read_raw(tbl, day, None, symbol)        # whole day, no as_of cutoff
        if len(_FULL_CACHE) > 80:           # bound (browsing many past days) — t3.micro
            _FULL_CACHE.clear()
        _FULL_CACHE[key] = (sig, full)
    else:
        full = hit[1]
    if full is None:
        return None
    if as_of is None:
        return full
    ts_cut = pd.Timestamp(as_of)
    if ts_cut.tzinfo is None:
        ts_cut = ts_cut.tz_localize(IST)
    out = full[full["ts"] <= ts_cut]
    return out.reset_index(drop=True) if len(out) else None


def _filter_expiry(c, kind: str):
    """(filtered_chain, available) for expiry `kind` in {weekly, monthly}. Legacy
    capture (no expiry column / all 0 / all null) is the single nearest weekly, so
    weekly returns it and monthly is 'not captured yet'. Once the capture tags rows
    with the expiry epoch, weekly = soonest, monthly = furthest captured."""
    if c is None:
        return c, False
    if "expiry" not in c.columns:
        return c, (kind != "monthly")
    exps = sorted({int(e) for e in c["expiry"].dropna().tolist() if int(e) > 0})
    if not exps:                                   # legacy rows (expiry 0/null) = weekly
        return c, (kind != "monthly")
    if kind == "monthly" and len(exps) < 2:
        return c.iloc[0:0], False
    target = exps[0] if kind == "weekly" else exps[-1]
    sub = c[c["expiry"] == target]
    return sub, len(sub) > 0


from core.session import CLOSE as SESSION_CLOSE, OPEN as SESSION_OPEN, session_only


def _session_only(df, day=None):
    """Drop anything outside 09:15–15:30 IST (and off-day rows when `day` is given).

    Thin wrapper over core.session.session_only — the window lives there because
    intraday_tf needs the identical clamp and a second copy is how these drift.

    The tick capture runs wider than the session on BOTH ends and both ends poison a bar:

      • PRE-OPEN. The 09:00–09:15 call auction publishes indicative prices that never
        traded. Measured 2026-08-10 NIFTY: 504 pre-open ticks reaching 24722.5 against a
        true session high of 24618.9. At tf=5 the damage hides, because the bar index is
        gated by chain capture (first snapshot 09:14) so the pre-open buckets get dropped
        anyway. At tf=60 the (09:00,10:00] bucket contains that 09:14 snapshot, SURVIVES,
        and drags every auction tick in with it — the first candle printed a high 103
        points above anything that ever traded.
      • POST-CLOSE. Ticks kept arriving until 17:56. With label/closed="right" that makes
        a 15:35 bar at tf=5 and a 16:00 bar at tf=60 whose (15:00,16:00] window blends the
        real last half-hour with half an hour of post-close prints.

    `atm_strikes` already clamped this way; the bar builders did not."""
    return session_only(df, "ts", day)


def _gaps_after(index, tf_min: int) -> list:
    """Per-bar flag: True where the NEXT bar is further away than one bar-width, i.e.
    capture stopped and the chart is about to draw straight through the hole.

    A bar with no snapshot in it does not exist in `index` at all — plotly is handed a
    LIST of timestamps, not a continuous grid, so a missing bar is not a visible break,
    it is a straight line between its neighbours. A 6-minute hole and a 4-hour outage
    both render as clean data. Worse, the bar AFTER the hole absorbs the whole backlog
    of volume increments: measured 2026-08-10, the 348s gap at 14:24 made the 14:35 bar
    print 2.1x the day's median volume, which reads as a burst of activity and is really
    the recorder restarting. Flag it so the render can mark it rather than pretend."""
    step = pd.Timedelta(minutes=tf_min)
    out = [False] * len(index)
    for i in range(len(index) - 1):
        if index[i + 1] - index[i] > step:
            out[i] = True
    return out


def _wallclock(idx):
    """Drop tz so plotly plots naive IST wall-clock. Mirror timestamps are tz-aware
    (UTC+05:30); plotly serializes the offset and the cursor spike re-applies it,
    so a 11:32 bar shows '05:02 pm' on the crosshair while the axis ticks read IST.
    Stripping tz makes axis ticks AND cursor spikes agree on the literal wall-clock."""
    return idx.tz_localize(None) if getattr(idx, "tz", None) is not None else idx


_SERIES_CACHE: dict = {}   # (sym,tf,date,as_of_iso,expiry) -> (chain_sig, result, covered)

# How far past the cutoff both source frames must already reach before a truncated
# prefix is treated as FINAL despite a later append. Capture writes in exchange-feed
# time order, but a snapshot poll can land a row a few seconds out of order, so the
# margin buys the reordering window rather than assuming perfect ordering.
_PREFIX_FINAL_MARGIN = pd.Timedelta(seconds=60)


def _ts_ist(x):
    """as_of -> tz-aware IST Timestamp (the frames' ts column is tz-aware)."""
    t = pd.Timestamp(x)
    return t.tz_localize(IST) if t.tzinfo is None else t


def _covered_until(day, sym):
    """Newest ts present in BOTH source frames for the day, or None. Reads the FULL-day
    frames, which `_read` already caches — so this is a dict hit, not a parquet parse."""
    try:
        t = _read("ticks", day, None, sym)
        c = _read("chain_snapshots", day, None, sym)
        if t is None or c is None or not len(t) or not len(c):
            return None
        return min(t["ts"].max(), c["ts"].max())
    except Exception:
        return None


def build_series(sym: str, tf_min: int, date=None, as_of=None, expiry="weekly") -> dict:
    """Memoised wrapper: the scout lifecycle walk-back asks for the SAME (sym,tf,date,
    as_of) series repeatedly (every 30s tick re-walks the same minutes; post-close the
    clamped as_of is constant), and each rebuild is a ~0.2s pivot+loop. Cache the result
    keyed by the chain file's (mtime,size) so a live append invalidates it. This is what
    turns the live scout render from ~25s (120 rebuilds) into ~instant after the first.

    PREFIX FINALITY. The (mtime,size) signature alone is too blunt for a LIVE session:
    the chain file is appended every ~30-60s, which invalidated every cached entry —
    including the hundreds pinned to as_of cutoffs hours in the past, whose answer new
    rows cannot possibly change. The horizon-ledger walk pays this worst: 288 truncated
    builds that all had to be redone every time capture wrote a row. So an entry whose
    sources already extended past its own cutoff (plus a reordering margin) survives a
    signature change — it is a COMPLETE prefix, not a partial one."""
    day = date or today_iso()
    try:
        st = (LIVE_DIR / f"{day}_chain_snapshots.parquet").stat()
        sig = (st.st_mtime_ns, st.st_size)
    except OSError:
        sig = None
    key = (sym, tf_min, day, as_of.isoformat() if as_of is not None else None, expiry)
    hit = _SERIES_CACHE.get(key)
    if hit is not None:
        _sig, _res, _cov = hit
        if _sig == sig:
            return _res
        try:
            if (as_of is not None and _cov is not None
                    and _cov >= _ts_ist(as_of) + _PREFIX_FINAL_MARGIN):
                return _res
        except Exception:                  # never let a cache heuristic break the read
            pass
    res = _build_series_impl(sym, tf_min, date, as_of, expiry)
    _SERIES_CACHE[key] = (sig, res, _covered_until(day, sym) if as_of is not None else None)
    # Bound for the t3.micro (OOM-sensitive). EVICT OLDEST, do not clear(): one horizon
    # ledger walk is 288 entries and switching all four horizons on one day is ~1150, so
    # a wipe at the bound threw away work that had just cost 40s and made the next popup
    # cold all over again. dict preserves insertion order, so the head is the oldest.
    if len(_SERIES_CACHE) > 2000:
        for k in list(_SERIES_CACHE)[:500]:
            _SERIES_CACHE.pop(k, None)
    return res


def _build_series_impl(sym: str, tf_min: int, date=None, as_of=None, expiry="weekly") -> dict:
    """Return the bar series for `sym` at `tf_min` minutes. {'has_data': False, ...}
    until enough is captured. tf_min is the bar size AND the highlighted window.
    `expiry` in {weekly, monthly} picks the option expiry (weekly until capture is
    extended to tag/store the monthly chain)."""
    ticks = _read("ticks", date, as_of, sym)
    chain = _read("chain_snapshots", date, as_of, sym)
    if ticks is None or chain is None or "ltp" not in chain.columns:
        return {"has_data": False, "sym": sym, "tf": tf_min,
                "note": "warming up — need ticks + option-chain capture"}
    chain, ok = _filter_expiry(chain, expiry)
    if not ok:
        return {"has_data": False, "sym": sym, "tf": tf_min,
                "note": f"{expiry} expiry not captured yet"}
    oi = _read("oi_snapshots", date, as_of, sym)

    # no pre-open auction, no post-close tape, no straggler row from the day BEFORE
    ticks = _session_only(ticks, date or today_iso())
    if ticks is None or not len(ticks):
        return {"has_data": False, "sym": sym, "tf": tf_min,
                "note": "warming up — no in-session ticks yet"}
    spot_at = ticks.set_index("ts")["ltp"].sort_index()

    # Per-snapshot ATM straddle: nearest-to-spot strike present on BOTH legs.
    ce = (chain[chain["side"] == "CE"]
          .pivot_table(index="ts", columns="strike", values="ltp", aggfunc="last").sort_index())
    pe = (chain[chain["side"] == "PE"]
          .pivot_table(index="ts", columns="strike", values="ltp", aggfunc="last").sort_index())
    # per-strike delta (signed: CE>0, PE<0) — to delta-adjust the ATM premium below.
    _has_delta = "delta" in chain.columns
    ce_d = (chain[chain["side"] == "CE"].pivot_table(index="ts", columns="strike",
            values="delta", aggfunc="last").sort_index()) if _has_delta else None
    pe_d = (chain[chain["side"] == "PE"].pivot_table(index="ts", columns="strike",
            values="delta", aggfunc="last").sort_index()) if _has_delta else None
    common = np.array(sorted(set(ce.columns) & set(pe.columns)), dtype=float)
    straddle: dict = {}
    atm_pce: dict = {}
    atm_ppe: dict = {}
    atm_dce: dict = {}
    atm_dpe: dict = {}
    atm_k: dict = {}
    if common.size:
        # Only timestamps present on BOTH legs — a partial snapshot (CE without PE,
        # the known L2 capture gap) must skip, not KeyError and blank the whole chart.
        for ts in ce.index.intersection(pe.index):
            sp = spot_at.asof(ts)
            if pd.isna(sp):
                continue
            k = common[int(np.argmin(np.abs(common - float(sp))))]
            cval, pval = ce.at[ts, k], pe.at[ts, k]
            if pd.notna(cval) and pd.notna(pval):
                straddle[ts] = float(cval) + float(pval)
                atm_pce[ts] = float(cval)        # per-leg ATM premium AT strike k —
                atm_ppe[ts] = float(pval)        # same strike as the delta below
                atm_k[ts] = float(k)             # the rolling ATM strike (roll-aware resid)
            if ce_d is not None and k in ce_d.columns and ts in ce_d.index:
                dv = ce_d.at[ts, k]
                if pd.notna(dv):
                    atm_dce[ts] = float(dv)
            if pe_d is not None and k in pe_d.columns and ts in pe_d.index:
                dv = pe_d.at[ts, k]
                if pd.notna(dv):
                    atm_dpe[ts] = float(dv)
    strad = pd.Series(straddle, dtype=float).sort_index()
    atm_pce_s = pd.Series(atm_pce, dtype=float).sort_index()
    atm_ppe_s = pd.Series(atm_ppe, dtype=float).sort_index()
    atm_dce_s = pd.Series(atm_dce, dtype=float).sort_index()
    atm_dpe_s = pd.Series(atm_dpe, dtype=float).sort_index()
    atm_k_s   = pd.Series(atm_k, dtype=float).sort_index()

    # Cumulative day option volume (summed across strikes) — per-bar later via diff.
    cum_vol = chain.groupby("ts")["volume"].sum().sort_index()

    # CE/PE totals from oi_snapshots (canonical); fall back to chain if absent.
    iv_atm = prem_ce = prem_pe = None
    _oi_degraded = False          # True → OI totals came from the churning chain-sum fallback
    if oi is not None and "total_call_oi" in oi.columns:
        oi_i = oi.set_index("ts").sort_index()
        oi_ce, oi_pe = oi_i["total_call_oi"], oi_i["total_put_oi"]
        if "atm_iv" in oi_i.columns:                # single ATM IV (vol-regime context line)
            iv_atm = oi_i["atm_iv"]
        elif "atm_call_iv" in oi_i.columns:
            iv_atm = oi_i["atm_call_iv"]
        if "atm_call_prem" in oi_i.columns:         # per-leg ATM premium → buy/write splitter
            prem_ce, prem_pe = oi_i["atm_call_prem"], oi_i["atm_put_prem"]
    else:
        # ── FALLBACK — NOT equivalent to the canonical source, and it can be very wrong ──
        # oi_snapshots' totals are computed by the capturer on a stable basis. This sum is
        # taken over WHATEVER STRIKES THE CHAIN HAPPENS TO HOLD at each ts, and the capture
        # width churns between 31 and 51 strikes on 33.7% of snapshots (measured 2026-08-06,
        # 10 sessions). Every width change adds/removes ~20 strikes of standing OI at once,
        # so the bar-to-bar DIFF of this series is dominated by the strike set rather than by
        # trading: |Δ| on width-change snapshots runs 65-476x the width-stable median.
        #
        # That makes d_oi_ce/d_oi_pe — and therefore the `flow` (0.40) and `div` (0.25)
        # signal legs — unreliable in both magnitude and SIGN whenever this branch is taken.
        # It used to be taken silently. Flag it so a missing oi_snapshots shows up as a data
        # problem instead of quietly becoming a 100x error in a directional read.
        oi_ce = chain[chain["side"] == "CE"].groupby("ts")["oi"].sum().sort_index()
        oi_pe = chain[chain["side"] == "PE"].groupby("ts")["oi"].sum().sort_index()
        _oi_degraded = True
        try:
            _nw = chain.groupby("ts")["strike"].nunique()
            _churn = float((_nw.diff().abs() > 0).mean()) if len(_nw) > 1 else 0.0
        except Exception:
            _churn = 0.0
        log.warning(
            "%s %s: oi_snapshots missing — OI totals fell back to the chain sum over a "
            "strike set that churns on %.0f%% of snapshots. d_oi (and the flow/div signal "
            "legs built on it) are UNRELIABLE for this day.", sym, date or "", 100 * _churn)

    if not len(strad) and not len(cum_vol):
        return {"has_data": False, "sym": sym, "tf": tf_min, "note": "warming up"}

    idx = pd.DatetimeIndex(sorted(set(strad.index) | set(cum_vol.index)))
    df = pd.DataFrame(index=idx)
    df["premium"] = strad.reindex(idx)
    df["oi_ce"]   = oi_ce.reindex(idx, method="ffill")
    df["oi_pe"]   = oi_pe.reindex(idx, method="ffill")
    df["cum_vol"] = cum_vol.reindex(idx, method="ffill")
    df["spot"]    = spot_at.reindex(idx, method="ffill")
    df["iv_atm"]  = iv_atm.reindex(idx, method="ffill") if iv_atm is not None else np.nan
    # classifier premium = the ATM-strike (k) leg premium from the chain — SAME strike
    # as the delta below (consistent residual); fall back to oi_snapshots ATM premium.
    df["prem_ce"] = (atm_pce_s.reindex(idx, method="ffill") if len(atm_pce_s)
                     else prem_ce.reindex(idx, method="ffill") if prem_ce is not None else np.nan)
    df["prem_pe"] = (atm_ppe_s.reindex(idx, method="ffill") if len(atm_ppe_s)
                     else prem_pe.reindex(idx, method="ffill") if prem_pe is not None else np.nan)
    df["dlt_ce"]  = atm_dce_s.reindex(idx, method="ffill") if len(atm_dce_s) else np.nan
    df["dlt_pe"]  = atm_dpe_s.reindex(idx, method="ffill") if len(atm_dpe_s) else np.nan
    df["atm_k"]   = atm_k_s.reindex(idx, method="ffill") if len(atm_k_s) else np.nan

    # Resample to tf-minute bars: point-in-time (last) for level series; volume is the
    # in-bar increment of the cumulative total.
    rs = df.resample(f"{tf_min}min", label="right", closed="right")
    bar = pd.DataFrame({
        "premium": rs["premium"].last(),
        "oi_ce":   rs["oi_ce"].last(),
        "oi_pe":   rs["oi_pe"].last(),
        "spot":    rs["spot"].last(),
        "cum_vol": rs["cum_vol"].last(),
        "iv_atm":  rs["iv_atm"].last(),
        "prem_ce": rs["prem_ce"].last(),
        "prem_pe": rs["prem_pe"].last(),
        "dlt_ce":  rs["dlt_ce"].last(),
        "dlt_pe":  rs["dlt_pe"].last(),
        "atm_k":   rs["atm_k"].last(),
    }).dropna(how="all")
    bar = bar[bar[["premium", "cum_vol"]].notna().any(axis=1)]
    if not len(bar):
        return {"has_data": False, "sym": sym, "tf": tf_min, "note": "warming up"}

    # Per-bar option volume, IMMUNE to the captured strike set changing intraday.
    # Summing the cumulative total over a shifting strike set then diffing is wrong:
    # a strike ENTERING dumps its whole day-cumulative as one fake bar spike, and a
    # strike LEAVING makes the diff negative -> clip(0) wipes the bar. Instead diff
    # each (strike, side) cumulative column first, clip per-leg, then sum the
    # increments (NaN on a leg's entry/exit -> skipped, conservative).
    vpiv = (chain.pivot_table(index="ts", columns=["strike", "side"],
                              values="volume", aggfunc="last").sort_index())
    vinc = vpiv.diff()
    if len(vinc):
        vinc.iloc[0] = vpiv.iloc[0]                   # first snapshot = volume since open
    vstep = vinc.clip(lower=0).sum(axis=1)            # total incremental option volume / snapshot
    vol = (vstep.resample(f"{tf_min}min", label="right", closed="right")
           .sum().reindex(bar.index).fillna(0.0))

    # Per-bar price OHLC from the underlying tick stream (same bars as the level
    # series above) — lets the popup draw a candlestick, not just a close line.
    px = spot_at.resample(f"{tf_min}min", label="right", closed="right")
    ohlc = pd.DataFrame({"o": px.first(), "h": px.max(),
                         "l": px.min(), "c": px.last()}).reindex(bar.index)

    def _col(s):
        return [None if pd.isna(v) else round(float(v), 2) for v in s]

    # ── Positioning flow: who was AGGRESSIVE each bar, per side ──────────────────
    # Standard OI-premium 4-quadrant, per leg, but on the DELTA-ADJUSTED premium
    # residual (residual = Δprem − delta·Δindex), NOT raw Δprem:
    #   OI up   + residual up   -> long BUILDUP (aggressive BUYING)
    #   OI up   + residual<=0   -> short BUILDUP (WRITING, "eating premium")
    #   OI down + residual up   -> short COVERING ; OI down + residual<=0 -> long UNWINDING
    # WHY residual: ATM premium carries delta·(index move), so raw Δprem makes CE
    # echo "buy" on any up-bar and PE "buy" on any down-bar — the label just tracks
    # price on a trending bar. Stripping delta·Δindex isolates true demand. delta is
    # the REAL captured ATM delta (signed: CE>0, PE<0), so one uniform formula covers
    # both legs — identical to build_strike_series + overnight_reconciliation. Delta
    # absent (legacy days) -> .fillna(0) falls back to raw Δprem (old behaviour).
    # LIMITATION: this pairs TOTAL CE/PE OI change with the ATM strike's premium, so
    # the aggregate buy/write is indicative; the per-strike chart is the precise read.
    # (ATM IV is one market-wide value here — context line only, never the splitter,
    # since atm_call_iv == atm_put_iv in the feed.)
    d_oi_ce, d_oi_pe = bar["oi_ce"].diff(), bar["oi_pe"].diff()
    d_spot = bar["spot"].diff()

    # ── ROLL-AWARE delta-adjusted residual (fixes a systematic LONG bias) ─────────
    # The ATM strike k ROLLS with spot (k = nearest strike to spot, recomputed every
    # snapshot), so bar["prem_ce"].diff() compares DIFFERENT strikes on a trending bar.
    # On a down day the ATM rolls DOWN: the ATM-call series jumps UP and the ATM-put DOWN
    # purely from the strike change → res_ce reads positive → calls mislabeled BUY/COVER
    # (bullish). That faked a 95%-CALL scout across 8 days (2026-06-29: calls "bought" on
    # 33/49 down bars while the market fell and IV doubled). FIX: hold the PRIOR bar's ATM
    # strike FIXED across the bar pair and measure THAT option's actual move (what a holder
    # experienced), delta-adjusted with that strike's delta — no roll artifact.
    bar_k = bar["atm_k"]
    idxb = bar.index

    def _leg_at(piv, ts, k):
        if piv is None or k != k or k not in piv.columns:
            return np.nan
        v = piv[k].asof(ts)
        return float(v) if pd.notna(v) else np.nan

    dprem_ce = pd.Series(np.nan, index=idxb)
    dprem_pe = pd.Series(np.nan, index=idxb)
    dlt_ce_k = pd.Series(np.nan, index=idxb)
    dlt_pe_k = pd.Series(np.nan, index=idxb)
    for i in range(1, len(idxb)):
        kk = bar_k.iloc[i - 1]                     # hold the PRIOR bar's ATM strike fixed
        t0, t1 = idxb[i - 1], idxb[i]
        dprem_ce.iloc[i] = _leg_at(ce, t1, kk) - _leg_at(ce, t0, kk)
        dprem_pe.iloc[i] = _leg_at(pe, t1, kk) - _leg_at(pe, t0, kk)
        dlt_ce_k.iloc[i] = _leg_at(ce_d, t0, kk)
        dlt_pe_k.iloc[i] = _leg_at(pe_d, t0, kk)
    # fall back to the (rolling) ATM diff only where the fixed-strike read is unavailable
    res_ce = dprem_ce.fillna(bar["prem_ce"].diff()) - dlt_ce_k.fillna(bar["dlt_ce"]).fillna(0.0) * d_spot
    res_pe = dprem_pe.fillna(bar["prem_pe"].diff()) - dlt_pe_k.fillna(bar["dlt_pe"]).fillna(0.0) * d_spot

    # ── VEGA / IV common-mode strip ──────────────────────────────────────────────
    # Even at a fixed strike a down-day IV spike (vega>0) lifts BOTH legs; the feed's ATM
    # IV is one market-wide value and ATM call & put have ~equal vega, so that push is
    # COMMON-MODE to res_ce/res_pe. A genuine DIRECTIONAL flow moves the legs OPPOSITELY;
    # a vol expansion moves them TOGETHER. Subtract the per-bar common mean → keep the
    # directional (differential) demand, drop the vega artifact (no captured vega needed).
    # Per-strike build_strike_series is single-leg so this applies only to the aggregate.
    _common = (res_ce + res_pe) / 2.0
    res_ce = res_ce - _common
    res_pe = res_pe - _common
    # ── OI DEADBAND, calibrated 2026-08-06 (was 0.10 = no filtering at all) ──────────
    # `_act` branches on the SIGN of d_oi, so the sign has to mean something. Measured by
    # comparing this series against an independent reconstruction of the same quantity
    # (per-strike chain diff, summed) over 8 sessions, n=188 bars: correlation only +0.46
    # and the two agree on SIGN just 61.7% of the time. The disagreement is concentrated
    # entirely in small bars — by |d_oi| quintile the agreement runs
    #     q1 235k → 47.4%   q2 1.2M → 62.2%   q3 2.2M → 50.0%
    #     q4 5.0M → 67.6%   q5 14.6M → 81.6%
    # i.e. the smallest fifth is a coin flip. At the old 0.10 multiplier the deadband kept
    # 95.7% of bars, so that coin flip flowed straight into flow (0.40) and div (0.25).
    #
    #     mult   bars kept   sign agreement
    #     0.10       95.7%        61.1%     <- old
    #     0.50       73.9%        66.9%
    #     1.00       50.5%        68.4%     <- chosen
    #     2.00       23.9%        71.1%
    #     3.00       15.4%        62.1%     (n=29, degrades — not a real improvement)
    #
    # 1.0 is the knee: the best agreement still backed by half the sample. 2.0 buys ~3pp
    # more for another 27% of bars and 3.0 falls apart, so this is not tuned to the max.
    # HONEST COST: half of all bars now read "flat" instead of a direction. That is the
    # point — they were never distinguishable from noise.
    _OI_EPS_MULT = 1.0
    # EXPANDING, not whole-view. The threshold used to be one number — the median over
    # every bar CURRENTLY IN VIEW — which quietly made a closed bar's label depend on how
    # much of the day had elapsed. Measured 2026-08-10 NIFTY: eps_ce ran 32.7M at 10:00
    # down to 3.2M at 15:30 (10x), and bar #3 — closed before 10:00 — read "flat" at 10:00
    # and "buy" from 11:30 on. At tf=60 the median is taken over 3-7 diffs, so eps_pe even
    # moved NON-monotonically (32.4M → 9.4M → 24.2M) and bar #3 went write → flat.
    #
    # Not a lookahead (at cutoff T the median only ever saw bars <= T) but a bar you read
    # at 11:00 was not the bar you got back at 15:30, and intraday_scout feeds these labels
    # into the flow leg at weight 0.40. An EXPANDING median depends only on bars <= i, so
    # bar i's label is FINAL the moment it closes — the series is now prefix-invariant like
    # every numeric column beside it. Warmup (< _OI_EPS_MIN_N observed diffs) reads "flat",
    # the same fail-safe direction the mood classifier uses before ER is warm.
    _OI_EPS_MIN_N = 3

    def _eps(d):
        return (d.abs().expanding(min_periods=_OI_EPS_MIN_N).median()
                * _OI_EPS_MULT).clip(lower=1.0)

    eps_ce, eps_pe = _eps(d_oi_ce), _eps(d_oi_pe)

    def _act(d_oi, resid, eps):
        if pd.isna(d_oi) or pd.isna(eps) or abs(d_oi) < eps:
            return "flat"
        if d_oi > 0:                                              # OI building
            return "buy" if (pd.notna(resid) and resid > 0) else "write"
        return "cover" if (pd.notna(resid) and resid > 0) else "unwind"   # OI falling

    ce_act = [_act(o, r, e) for o, r, e in zip(d_oi_ce, res_ce, eps_ce)]
    pe_act = [_act(o, r, e) for o, r, e in zip(d_oi_pe, res_pe, eps_pe)]

    _wc = _wallclock(bar.index)
    return {
        "has_data": True, "sym": sym, "tf": tf_min,
        # True → OI totals came from the chain-sum fallback over a churning strike set, so
        # d_oi / flow / div are unreliable here. Consumers should degrade the OI read
        # rather than present it as equal-confidence.
        "oi_degraded": _oi_degraded,
        "ts":      [t.to_pydatetime() for t in _wc],
        "premium": [None if pd.isna(v) else round(float(v), 2) for v in bar["premium"]],
        "oi_ce":   [None if pd.isna(v) else round(float(v) / 1e5, 2) for v in bar["oi_ce"]],
        "oi_pe":   [None if pd.isna(v) else round(float(v) / 1e5, 2) for v in bar["oi_pe"]],
        "spot":    [None if pd.isna(v) else round(float(v), 2) for v in bar["spot"]],
        "open":    _col(ohlc["o"]), "high": _col(ohlc["h"]),
        "low":     _col(ohlc["l"]), "close": _col(ohlc["c"]),
        "volume":  [0.0 if pd.isna(v) else round(float(v) / 1e5, 3) for v in vol],
        "iv_atm":  [None if pd.isna(v) else round(float(v), 2) for v in bar["iv_atm"]],
        "d_oi_ce": [None if pd.isna(v) else round(float(v) / 1e5, 2) for v in d_oi_ce],
        "d_oi_pe": [None if pd.isna(v) else round(float(v) / 1e5, 2) for v in d_oi_pe],
        "ce_act":  ce_act, "pe_act": pe_act,
        # True on a bar whose successor is more than one bar-width away — capture stopped
        # here, the next bar's volume absorbed the backlog. Render marks it.
        "gap_after": _gaps_after(bar.index, tf_min),
        "last_ts": _wc[-1].to_pydatetime(),
    }


def build_futures_series(sym: str, tf_min: int, date=None, as_of=None, leg="near") -> dict:
    """Per-bar futures series for `sym` at `tf_min` minutes, full session.

    `leg` ∈ {near, next, far} picks which expiry's candles + volume to draw. The
    near/next/far close lines are always returned for context overlay.

    ROLLOVER: as expiry nears, positions move near→next (a roll, not an exit). The
    captured data lets us read it two honest ways:
      roll_share = next-month share of per-bar volume (rising = rollover in progress)
      roll       = next − near price (calendar spread; widens with roll demand).
    Per-expiry OI is NOT in the intraday feed (oi-spurts OI is consolidated across
    expiries; the OI/positioning panel is underlying-level). True per-expiry rollover
    OI% needs the EOD F&O bhavcopy.
    """
    f = _read("futures_quotes", date, as_of, sym)
    if f is None or "near_ltp" not in f.columns:
        return {"has_data": False, "sym": sym, "tf": tf_min,
                "note": "warming up — need futures capture"}
    f = _session_only(f, date or today_iso())     # same clamp as the option bars
    if f is None or not len(f):
        return {"has_data": False, "sym": sym, "tf": tf_min,
                "note": "warming up — no in-session futures quotes yet"}
    f = f.set_index("ts").sort_index()
    leg = leg if leg in ("near", "next", "far") else "near"
    _ltp = {"near": "near_ltp", "next": "next_ltp", "far": "far_ltp"}[leg]
    _vol = {"near": "near_vol", "next": "next_vol", "far": None}[leg]
    if _ltp not in f.columns:
        _ltp, _vol, leg = "near_ltp", "near_vol", "near"

    def _last(col):
        return (f[col].resample(f"{tf_min}min", label="right", closed="right").last()
                if col in f.columns else None)

    price = f[_ltp].dropna()
    rs = price.resample(f"{tf_min}min", label="right", closed="right")
    bar = pd.DataFrame({"o": rs.first(), "h": rs.max(), "l": rs.min(), "c": rs.last()})
    for c in ("near_ltp", "next_ltp", "far_ltp", "near_basis", "roll_spread"):
        s = _last(c)
        if s is not None:
            bar[c] = s
    bar = bar.dropna(subset=["c"])
    if not len(bar):
        return {"has_data": False, "sym": sym, "tf": tf_min, "note": "warming up"}

    # Selected-leg volume (cumulative day total → per-bar increment).
    vol = _last(_vol) if _vol else None
    if vol is not None:
        v0 = vol.iloc[0]
        vol = vol.diff(); vol.iloc[0] = v0; vol = vol.clip(lower=0)
        vol = vol.reindex(bar.index)

    # Rollover activity: next-month share of per-bar (near+next) volume.
    share = None
    nv, xv = _last("near_vol"), _last("next_vol")
    if nv is not None and xv is not None:
        ni, xi = nv.diff().clip(lower=0), xv.diff().clip(lower=0)
        ni.iloc[0], xi.iloc[0] = nv.iloc[0], xv.iloc[0]
        denom = (ni + xi).replace(0, float("nan"))
        share = (xi / denom * 100).reindex(bar.index)

    ts_term = f.get("term_structure")
    term = (ts_term.resample(f"{tf_min}min", label="right", closed="right").last()
            .reindex(bar.index)) if ts_term is not None else None

    # Consolidated futures OI (NSE oi-spurts, all expiries) × NEAR price → positioning.
    oi_lakh = d_oi = None
    fut_act = ["flat"] * len(bar)
    ofut = _read("futures_oi", date, as_of, NSE_NAME.get(sym, sym))
    if ofut is not None and "oi" in ofut.columns:
        oi_s = (ofut.set_index("ts")["oi"].sort_index()
                .resample(f"{tf_min}min", label="right", closed="right").last()
                .reindex(bar.index, method="ffill"))
        oi_lakh = oi_s / 1e5
        d_oi = oi_lakh.diff()
        d_px = (bar["near_ltp"] if "near_ltp" in bar else bar["c"]).diff()
        eps = max(0.1, 0.10 * float(d_oi.abs().median() or 0))

        def _fact(do, dp):
            if pd.isna(do) or abs(do) < eps:
                return "flat"
            if do > 0:
                return "long" if (pd.notna(dp) and dp >= 0) else "short"
            return "cover" if (pd.notna(dp) and dp >= 0) else "unwind"

        fut_act = [_fact(o, p) for o, p in zip(d_oi, d_px)]

    def _c(s):
        return [None if (s is None or pd.isna(v)) else round(float(v), 2) for v in (s if s is not None else [])]

    _wc = _wallclock(bar.index)
    return {
        "has_data": True, "sym": sym, "tf": tf_min, "leg": leg,
        "ts":     [t.to_pydatetime() for t in _wc],
        "open":   _c(bar["o"]), "high": _c(bar["h"]), "low": _c(bar["l"]), "close": _c(bar["c"]),
        "near":   _c(bar.get("near_ltp")), "next": _c(bar.get("next_ltp")), "far": _c(bar.get("far_ltp")),
        "basis":  _c(bar.get("near_basis")),
        "roll":   _c(bar.get("roll_spread")),
        "roll_share": _c(share), "has_roll": share is not None,
        "volume": [0.0 if (vol is None or pd.isna(v)) else round(float(v) / 1e5, 3)
                   for v in (vol if vol is not None else [0] * len(bar))],
        "has_vol": vol is not None,
        "term":   [None if (term is None or pd.isna(v)) else str(v) for v in (term if term is not None else [None] * len(bar))],
        "oi":     _c(oi_lakh), "d_oi": _c(d_oi), "fut_act": fut_act,
        "has_oi": oi_lakh is not None,
        "gap_after": _gaps_after(bar.index, tf_min),
        "last_ts": _wc[-1].to_pydatetime(),
    }


def atm_strikes(sym: str, date=None, as_of=None, n: int = 10, expiry="weekly") -> tuple:
    """(open_anchor, [anchor ± n round strikes]) for the picker.

    Anchor = the session OPENING price (the 09:15 open), snapped to the round step
    (STRIKE_DISPLAY_STEP, e.g. NIFTY 100). The ladder is FIXED at market open and
    spans the day's realistic range — n*step each side (default ±1000 for NIFTY) —
    rather than the moving intraday ATM, so the captured strikes don't shift as spot
    drifts. Round-number strikes only (where the real OI sits), filtered to those
    actually captured in chain_snapshots for the chosen `expiry`."""
    c = _read("chain_snapshots", date, as_of, sym)
    if c is None or "strike" not in c.columns:
        return None, []
    c, ok = _filter_expiry(c, expiry)
    if not ok or c is None or c.empty:
        return None, []
    ks = sorted(int(k) for k in c["strike"].unique())
    if not ks:
        return None, []

    step = STRIKE_DISPLAY_STEP.get(sym, STRIKE_STEP.get(sym, 50))
    # session OPENING price (first 09:15–15:30 tick's day_open, else its ltp)
    open_px = None
    t = _read("ticks", date, as_of, sym)
    if t is not None and len(t):
        sess = t[(t["ts"].dt.time >= pd.Timestamp("09:15").time()) &
                 (t["ts"].dt.time <= pd.Timestamp("15:30").time())]
        first = (sess if len(sess) else t).iloc[0]
        open_px = float(first.get("day_open") or 0) or float(first.get("ltp") or 0)
    if not open_px:                                    # fallback: oi-snapshots open ATM / mid
        o = _read("oi_snapshots", date, as_of, sym)
        if o is not None and "atm" in o.columns and len(o):
            try:
                open_px = float(o["atm"].iloc[0])
            except Exception:
                open_px = None
        if not open_px:
            open_px = ks[len(ks) // 2]

    anchor = int(round(open_px / step) * step)         # snap open to the round step
    avail = set(ks)
    ladder = [anchor + i * step for i in range(-n, n + 1) if (anchor + i * step) in avail]
    if not ladder:                                     # round strikes not captured → fall back
        i = ks.index(min(ks, key=lambda k: abs(k - anchor)))
        return ks[i], ks[max(0, i - n):i + n + 1]
    return anchor, ladder


def build_strike_series(sym: str, tf_min: int, strike: int, date=None, as_of=None,
                        expiry="weekly") -> dict:
    """Per-bar series for ONE option strike, full session: that strike's CE & PE
    OI, premium (ltp), volume, and a per-side write/buy classification via the
    DELTA-ADJUSTED premium residual (Δprem − delta·Δindex) — the index-move stripped
    out, so it reads true buy (residual>0) vs write (residual<=0), not just price.
    Index OHLC is returned for price context (with the strike level to overlay)."""
    c = _read("chain_snapshots", date, as_of, sym)
    if c is None or "strike" not in c.columns:
        return {"has_data": False, "sym": sym, "tf": tf_min, "strike": strike,
                "note": "warming up — need option-chain capture"}
    c, ok = _filter_expiry(c, expiry)
    if not ok:
        return {"has_data": False, "sym": sym, "tf": tf_min, "strike": strike,
                "note": f"{expiry} expiry not captured yet"}
    c = c[c["strike"] == int(strike)]
    if c.empty:
        return {"has_data": False, "sym": sym, "tf": tf_min, "strike": strike,
                "note": f"no capture at strike {strike}"}
    ticks = _read("ticks", date, as_of, sym)
    spot_at = ticks.set_index("ts")["ltp"].sort_index() if ticks is not None else None

    def _side(side):
        s = c[c["side"] == side].set_index("ts").sort_index()
        if s.empty:
            return None
        r = s.resample(f"{tf_min}min", label="right", closed="right")
        return pd.DataFrame({"oi": r["oi"].last(), "ltp": r["ltp"].last(),
                             "cum_vol": r["volume"].last(),
                             "delta": r["delta"].last() if "delta" in s.columns else np.nan})

    ce, pe = _side("CE"), _side("PE")
    base = ce if ce is not None else pe
    if base is None or not len(base):
        return {"has_data": False, "sym": sym, "tf": tf_min, "strike": strike, "note": "warming up"}
    idx = base.index

    px = (spot_at.resample(f"{tf_min}min", label="right", closed="right")
          if spot_at is not None else None)
    ohlc = (pd.DataFrame({"o": px.first(), "h": px.max(), "l": px.min(), "c": px.last()}).reindex(idx)
            if px is not None else pd.DataFrame(index=idx))
    d_spot = ohlc["c"].diff() if "c" in ohlc.columns else pd.Series(np.nan, index=idx)

    def _col(s):
        return [None if (s is None or pd.isna(v)) else round(float(v), 2) for v in (s if s is not None else [None] * len(idx))]

    def _legbars(df):
        """(oi_lakh, prem, vol_lakh, d_oi_lakh, actions) for one option leg.

        buy-vs-write uses the DELTA-ADJUSTED premium residual:
          residual = Δpremium − delta·Δindex   (the option move stripped of the part
          the underlying move alone explains). residual>0 = buyers paying up (BUY);
          residual<=0 = writers pressing (WRITE). Far truer than raw premium, which on
          a trending bar just echoes price. Falls back to raw Δpremium if delta absent."""
        if df is None:
            n = len(idx)
            return [None] * n, [None] * n, [0.0] * n, [None] * n, ["flat"] * n
        df = df.reindex(idx)
        d_oi = df["oi"].diff()
        d_pr = df["ltp"].diff()
        dlt = df["delta"] if "delta" in df.columns else pd.Series(np.nan, index=idx)
        resid = d_pr - dlt.fillna(0.0) * d_spot.reindex(df.index).fillna(0.0)
        vol = df["cum_vol"].diff()
        if len(vol):
            vol.iloc[0] = df["cum_vol"].iloc[0]
        vol = vol.clip(lower=0)
        eps = max(1.0, 0.10 * float(d_oi.abs().median() or 0))

        def _act(do, rz):
            if pd.isna(do) or abs(do) < eps:
                return "flat"
            if do > 0:                                   # OI building
                return "buy" if (pd.notna(rz) and rz > 0) else "write"
            return "cover" if (pd.notna(rz) and rz > 0) else "unwind"   # OI falling

        acts = [_act(o_, r_) for o_, r_ in zip(d_oi, resid)]
        return (_oilakh(df["oi"]), _col(df["ltp"]), _vollakh(vol), _oilakh(d_oi), acts)

    def _oilakh(s):
        return [None if pd.isna(v) else round(float(v) / 1e5, 2) for v in s]

    def _vollakh(s):
        return [0.0 if pd.isna(v) else round(float(v) / 1e5, 3) for v in s]

    ce_oi, ce_prem, ce_vol, ce_doi, ce_act = _legbars(ce)
    pe_oi, pe_prem, pe_vol, pe_doi, pe_act = _legbars(pe)

    _wc = _wallclock(idx)
    return {
        "has_data": True, "sym": sym, "tf": tf_min, "strike": int(strike),
        "ts":    [t.to_pydatetime() for t in _wc],
        "open":  _col(ohlc.get("o")), "high": _col(ohlc.get("h")),
        "low":   _col(ohlc.get("l")), "close": _col(ohlc.get("c")),
        "ce_oi": ce_oi, "pe_oi": pe_oi, "ce_prem": ce_prem, "pe_prem": pe_prem,
        "ce_vol": ce_vol, "pe_vol": pe_vol, "ce_doi": ce_doi, "pe_doi": pe_doi,
        "ce_act": ce_act, "pe_act": pe_act,
        "last_ts": _wc[-1].to_pydatetime(),
    }
