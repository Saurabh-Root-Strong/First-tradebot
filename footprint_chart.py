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

from core.mirror_io import read_mirror as _read


def build_series(sym: str, tf_min: int, date=None, as_of=None) -> dict:
    """Return the bar series for `sym` at `tf_min` minutes. {'has_data': False, ...}
    until enough is captured. tf_min is the bar size AND the highlighted window."""
    ticks = _read("ticks", date, as_of, sym)
    chain = _read("chain_snapshots", date, as_of, sym)
    if ticks is None or chain is None or "ltp" not in chain.columns:
        return {"has_data": False, "sym": sym, "tf": tf_min,
                "note": "warming up — need ticks + option-chain capture"}
    oi = _read("oi_snapshots", date, as_of, sym)

    spot_at = ticks.set_index("ts")["ltp"].sort_index()

    # Per-snapshot ATM straddle: nearest-to-spot strike present on BOTH legs.
    ce = (chain[chain["side"] == "CE"]
          .pivot_table(index="ts", columns="strike", values="ltp", aggfunc="last").sort_index())
    pe = (chain[chain["side"] == "PE"]
          .pivot_table(index="ts", columns="strike", values="ltp", aggfunc="last").sort_index())
    common = np.array(sorted(set(ce.columns) & set(pe.columns)), dtype=float)
    straddle: dict = {}
    if common.size:
        for ts in ce.index:
            sp = spot_at.asof(ts)
            if pd.isna(sp):
                continue
            k = common[int(np.argmin(np.abs(common - float(sp))))]
            cval, pval = ce.at[ts, k], pe.at[ts, k]
            if pd.notna(cval) and pd.notna(pval):
                straddle[ts] = float(cval) + float(pval)
    strad = pd.Series(straddle, dtype=float).sort_index()

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
    df["prem_ce"] = prem_ce.reindex(idx, method="ffill") if prem_ce is not None else np.nan
    df["prem_pe"] = prem_pe.reindex(idx, method="ffill") if prem_pe is not None else np.nan

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
    }).dropna(how="all")
    bar = bar[bar[["premium", "cum_vol"]].notna().any(axis=1)]
    if not len(bar):
        return {"has_data": False, "sym": sym, "tf": tf_min, "note": "warming up"}

    vol = bar["cum_vol"].diff()
    if len(vol):
        vol.iloc[0] = bar["cum_vol"].iloc[0]          # first bar = volume since open
    vol = vol.clip(lower=0)                            # guard the stale-open crossover

    # Per-bar price OHLC from the underlying tick stream (same bars as the level
    # series above) — lets the popup draw a candlestick, not just a close line.
    px = spot_at.resample(f"{tf_min}min", label="right", closed="right")
    ohlc = pd.DataFrame({"o": px.first(), "h": px.max(),
                         "l": px.min(), "c": px.last()}).reindex(bar.index)

    def _col(s):
        return [None if pd.isna(v) else round(float(v), 2) for v in s]

    # ── Positioning flow: who was AGGRESSIVE each bar, per side ──────────────────
    # Standard OI-premium 4-quadrant, per leg (the leg's OWN ATM premium):
    #   OI up   + premium up   -> long BUILDUP (aggressive BUYING)
    #   OI up   + premium flat/down -> short BUILDUP (WRITING, "eating premium")
    #   OI down + premium up   -> short COVERING ;  OI down + premium down -> long UNWINDING
    # NOTE: ATM premium also carries ~half the underlying move (delta ~0.5), so on a
    # strong directional bar the label leans with price — the convention every desk uses.
    # (ATM IV is a single market-wide value here — kept only as the context line, never
    # as the per-leg splitter, since atm_call_iv == atm_put_iv in the feed.)
    d_oi_ce, d_oi_pe = bar["oi_ce"].diff(), bar["oi_pe"].diff()
    d_pr_ce, d_pr_pe = bar["prem_ce"].diff(), bar["prem_pe"].diff()
    eps_ce = max(1.0, 0.10 * float(d_oi_ce.abs().median() or 0))
    eps_pe = max(1.0, 0.10 * float(d_oi_pe.abs().median() or 0))

    def _act(d_oi, d_pr, eps):
        if pd.isna(d_oi) or abs(d_oi) < eps:
            return "flat"
        if d_oi > 0:                                              # OI building
            return "buy" if (pd.notna(d_pr) and d_pr > 0) else "write"
        return "cover" if (pd.notna(d_pr) and d_pr > 0) else "unwind"   # OI falling

    ce_act = [_act(o, p, eps_ce) for o, p in zip(d_oi_ce, d_pr_ce)]
    pe_act = [_act(o, p, eps_pe) for o, p in zip(d_oi_pe, d_pr_pe)]

    return {
        "has_data": True, "sym": sym, "tf": tf_min,
        "ts":      [t.to_pydatetime() for t in bar.index],
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
        "last_ts": bar.index[-1].to_pydatetime(),
    }
