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

import numpy as np
import pandas as pd

from core.constants import NSE_NAME, STRIKE_DISPLAY_STEP, STRIKE_STEP
from core.mirror_io import read_mirror as _read


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


def _wallclock(idx):
    """Drop tz so plotly plots naive IST wall-clock. Mirror timestamps are tz-aware
    (UTC+05:30); plotly serializes the offset and the cursor spike re-applies it,
    so a 11:32 bar shows '05:02 pm' on the crosshair while the axis ticks read IST.
    Stripping tz makes axis ticks AND cursor spikes agree on the literal wall-clock."""
    return idx.tz_localize(None) if getattr(idx, "tz", None) is not None else idx


def build_series(sym: str, tf_min: int, date=None, as_of=None, expiry="weekly") -> dict:
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

    # Cumulative day option volume (summed across strikes) — per-bar later via diff.
    cum_vol = chain.groupby("ts")["volume"].sum().sort_index()

    # CE/PE totals from oi_snapshots (canonical); fall back to chain if absent.
    iv_atm = prem_ce = prem_pe = None
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
        oi_ce = chain[chain["side"] == "CE"].groupby("ts")["oi"].sum().sort_index()
        oi_pe = chain[chain["side"] == "PE"].groupby("ts")["oi"].sum().sort_index()

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
    res_ce = bar["prem_ce"].diff() - bar["dlt_ce"].fillna(0.0) * d_spot
    res_pe = bar["prem_pe"].diff() - bar["dlt_pe"].fillna(0.0) * d_spot
    eps_ce = max(1.0, 0.10 * float(d_oi_ce.abs().median() or 0))
    eps_pe = max(1.0, 0.10 * float(d_oi_pe.abs().median() or 0))

    def _act(d_oi, resid, eps):
        if pd.isna(d_oi) or abs(d_oi) < eps:
            return "flat"
        if d_oi > 0:                                              # OI building
            return "buy" if (pd.notna(resid) and resid > 0) else "write"
        return "cover" if (pd.notna(resid) and resid > 0) else "unwind"   # OI falling

    ce_act = [_act(o, r, eps_ce) for o, r in zip(d_oi_ce, res_ce)]
    pe_act = [_act(o, r, eps_pe) for o, r in zip(d_oi_pe, res_pe)]

    _wc = _wallclock(bar.index)
    return {
        "has_data": True, "sym": sym, "tf": tf_min,
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
