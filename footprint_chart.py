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
    if oi is not None and "total_call_oi" in oi.columns:
        oi_i = oi.set_index("ts").sort_index()
        oi_ce, oi_pe = oi_i["total_call_oi"], oi_i["total_put_oi"]
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

    # Resample to tf-minute bars: point-in-time (last) for level series; volume is the
    # in-bar increment of the cumulative total.
    rs = df.resample(f"{tf_min}min", label="right", closed="right")
    bar = pd.DataFrame({
        "premium": rs["premium"].last(),
        "oi_ce":   rs["oi_ce"].last(),
        "oi_pe":   rs["oi_pe"].last(),
        "spot":    rs["spot"].last(),
        "cum_vol": rs["cum_vol"].last(),
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
        "last_ts": bar.index[-1].to_pydatetime(),
    }
