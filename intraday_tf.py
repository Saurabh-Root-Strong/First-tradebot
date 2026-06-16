"""
intraday_tf.py — multi-timeframe OI · Price · Volume matrix (the "are they secretly
closing / building?" detector).

For each timeframe (5 / 10 / 15 / 60 min, anchored to now) it measures, for the
UNDERLYING and its OPTIONS (and futures price/vol), the joint move of:

    Price   Δ% over the window
    OI      Δ total option OI  (building ↑  vs  closing ↓)  + net put−call flow
    Volume  traded option volume in the window (activity)

and classifies each timeframe with the classic OI×Price regime:

    price↑ OI↑   LONG BUILDUP    fresh longs        (sustainable up)
    price↑ OI↓   SHORT COVERING  shorts closing     (up, but not fresh — fades)
    price↓ OI↑   SHORT BUILDUP   fresh shorts       (sustainable down)
    price↓ OI↓   LONG UNWINDING  longs closing      (down, positions exiting)

The POINT is cross-timeframe divergence: when the short frames say BUILDUP but the
hour says UNWINDING/COVERING — i.e. the tape rises while bigger positions quietly
close — that's distribution under a rising market. Or price up while net option
positioning is bearish (call-writing) = hidden distribution. Those are the
"secretly closing/creating" tells you act on (avoid / fade), not the price alone.

Futures show price + volume + basis. Intraday futures OI is NOT available from the
Fyers data feed (verified: absent from both /data/quotes and /data/options-chain-v3),
so the OI footprint here is option-based — which is the dominant signal for an index
anyway. Futures OI exists only EOD, in the Daily_Cash_Market F&O bhavcopy.

Reads lock-free mirrors with as_of/date, so it replays on any captured day.
"""
from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pandas as pd

IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
LIVE_DIR = Path(__file__).parent / "data" / "intraday" / "live"
TFS = [5, 10, 15, 60]
LABELS = {"NSE:NIFTY50-INDEX": "NIFTY 50", "NSE:NIFTYBANK-INDEX": "BANK NIFTY",
          "NSE:FINNIFTY-INDEX": "FIN NIFTY", "NSE:MIDCPNIFTY-INDEX": "MIDCAP NIFTY"}
LABEL_TO_SYM = {"NIFTY": "NSE:NIFTY50-INDEX", "BANKNIFTY": "NSE:NIFTYBANK-INDEX",
                "FINNIFTY": "NSE:FINNIFTY-INDEX", "MIDCPNIFTY": "NSE:MIDCPNIFTY-INDEX",
                "MIDCAPNIFTY": "NSE:MIDCPNIFTY-INDEX"}

_PX_DB  = 0.03    # |price%| deadband
_OI_DB  = 0.5     # ΔOI deadband in lakhs (×1e5 contracts)
# index symbol → NSE OI-spurts underlying name (for the futures_oi mirror)
_NSE_UL = {"NSE:NIFTY50-INDEX": "NIFTY", "NSE:NIFTYBANK-INDEX": "BANKNIFTY",
           "NSE:FINNIFTY-INDEX": "FINNIFTY", "NSE:MIDCPNIFTY-INDEX": "MIDCPNIFTY"}


def _today() -> str:
    return datetime.datetime.now(tz=IST).date().isoformat()


def _read(tbl, date, as_of):
    p = LIVE_DIR / f"{date or _today()}_{tbl}.parquet"
    if not p.exists() or p.stat().st_size < 100:
        return None
    df = pd.read_parquet(p)
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(IST)
    if as_of is not None:
        df = df[df["ts"] <= pd.Timestamp(as_of)]
    return df if len(df) else None


def _at_or_before(df, sym_col, sym, value_ts):
    sub = df[df["ts"] <= value_ts]
    return sub.iloc[-1] if len(sub) else None


def _regime(px: float, d_oi: float) -> tuple[str, str, int]:
    """(tag, texture, bullish_sign) from price vs total-OI direction."""
    pu, ou = (px > _PX_DB) - (px < -_PX_DB), (d_oi > _OI_DB) - (d_oi < -_OI_DB)
    if pu > 0 and ou > 0:  return "LONG BUILDUP",  "fresh",   +1
    if pu > 0 and ou < 0:  return "SHORT COVER",   "closing", +1
    if pu < 0 and ou > 0:  return "SHORT BUILDUP", "fresh",   -1
    if pu < 0 and ou < 0:  return "LONG UNWIND",   "closing", -1
    return "BALANCED", "flat", 0


def analyze(sym: str, date=None, as_of=None) -> dict:
    ticks = _read("ticks", date, as_of)
    oi = _read("oi_snapshots", date, as_of)
    chain = _read("chain_snapshots", date, as_of)
    fut = _read("futures_quotes", date, as_of)
    if ticks is None or oi is None:
        return {"sym": sym, "has_data": False, "note": "warming up — need ticks + OI"}
    t = ticks[ticks["symbol"] == sym].sort_values("ts")
    o = oi[oi["symbol"] == sym].sort_values("ts")
    if len(t) < 5 or len(o) < 2:
        return {"sym": sym, "has_data": False, "note": "warming up"}
    now_ts = t["ts"].iloc[-1]

    # total option volume series (sum of strike volume per snapshot)
    vol_series = None
    if chain is not None:
        c = chain[chain["symbol"] == sym]
        if len(c):
            vol_series = c.groupby("ts")["volume"].sum().sort_index()
    f = fut[fut["symbol"] == sym].sort_values("ts") if fut is not None else None
    # futures OI (from the NSE mirror — Fyers doesn't serve it)
    foi = _read("futures_oi", date, as_of)
    fo = None
    if foi is not None and _NSE_UL.get(sym):
        fo = foi[foi["symbol"] == _NSE_UL[sym]].sort_values("ts")
        if not len(fo):
            fo = None

    def price_at(ts_):
        s = t[t["ts"] <= ts_]
        return float(s["ltp"].iloc[-1]) if len(s) else None

    cells = []
    for T in TFS:
        anchor = now_ts - pd.Timedelta(minutes=T)
        p0, p1 = price_at(anchor), float(t["ltp"].iloc[-1])
        if p0 is None or p0 <= 0:
            continue
        px = (p1 - p0) / p0 * 100
        # option OI deltas
        o_prev = _at_or_before(o, "symbol", sym, anchor) ; o_now = o.iloc[-1]
        if o_prev is None: o_prev = o.iloc[0]
        d_call = (float(o_now["total_call_oi"] or 0) - float(o_prev["total_call_oi"] or 0)) / 1e5
        d_put  = (float(o_now["total_put_oi"]  or 0) - float(o_prev["total_put_oi"]  or 0)) / 1e5
        d_tot  = d_call + d_put
        net    = d_put - d_call                          # +put-writing(bull) / -call-writing(bear)
        # option volume in window
        ov = float("nan")
        if vol_series is not None and len(vol_series) >= 2:
            vs = vol_series[vol_series.index <= now_ts]
            base = vs[vs.index <= anchor]
            iv = float(vs.iloc[-1] - (base.iloc[-1] if len(base) else vs.iloc[0]))
            ov = iv / 1e5 if iv >= 0 else float("nan")    # NaN if it crosses the stale open
        # futures
        fpx = fvol = fbasis = None
        if f is not None and len(f) >= 1:
            fn = f.iloc[-1]
            fp = _at_or_before(f, "symbol", sym, anchor)
            if fp is None:
                fp = f.iloc[0]
            if float(fp["near_ltp"] or 0) > 0:
                fpx = (float(fn["near_ltp"]) - float(fp["near_ltp"])) / float(fp["near_ltp"]) * 100
                fvol = (float(fn["near_vol"] or 0) - float(fp["near_vol"] or 0)) / 1e5
                fbasis = float(fn["near_basis"] or 0) - float(fp["near_basis"] or 0)
        # futures OI delta over the window (clean buildup/covering)
        foi_d = None; foi_build = 0
        if fo is not None:
            fo_prev = fo[fo["ts"] <= anchor]
            base_oi = float(fo_prev.iloc[-1]["oi"]) if len(fo_prev) else float(fo.iloc[0]["oi"])
            foi_d = (float(fo.iloc[-1]["oi"]) - base_oi) / 1e5
            foi_build = int((foi_d > _OI_DB) - (foi_d < -_OI_DB))
        tag, texture, dir_sign = _regime(px, d_tot)       # dir_sign = bullish/bearish of regime
        oi_build = int((d_tot > _OI_DB) - (d_tot < -_OI_DB))   # +1 building, -1 closing
        pos_sign = int((net > _OI_DB) - (net < -_OI_DB))   # OI positioning direction
        cells.append({"tf": T, "px": round(px, 3), "d_call": round(d_call, 1),
                      "d_put": round(d_put, 1), "d_tot": round(d_tot, 1), "net": round(net, 1),
                      "ovol": round(ov, 1) if ov == ov else None, "tag": tag,
                      "texture": texture, "dir_sign": dir_sign, "oi_build": oi_build,
                      "pos_sign": pos_sign,
                      "fpx": round(fpx, 3) if fpx is not None else None,
                      "fvol": round(fvol, 1) if fvol is not None else None,
                      "fbasis": round(fbasis, 0) if fbasis is not None else None,
                      "foi": round(foi_d, 1) if foi_d is not None else None,
                      "foi_build": foi_build})
    if not cells:
        return {"sym": sym, "has_data": False, "note": "warming up"}

    fut_oi_day = round(float(fo.iloc[-1]["chg_oi"]) / 1e5, 1) if fo is not None else None
    verdict = _synthesize(cells)
    return {"sym": sym, "label": LABELS.get(sym, sym), "has_data": True,
            "now": now_ts.strftime("%H:%M:%S"), "spot": round(float(t["ltp"].iloc[-1]), 2),
            "fut_oi_day": fut_oi_day, "cells": cells, **verdict}


def _synthesize(cells: list[dict]) -> dict:
    by = {c["tf"]: c for c in cells}
    short = [by[t] for t in (5, 10) if t in by]
    hour = by.get(60) or by.get(15)
    flags = []

    # 1. tape rising on closing (short-cover / unwind) rather than fresh positioning
    up_short = short and all(c["px"] > _PX_DB for c in short)
    dn_short = short and all(c["px"] < -_PX_DB for c in short)
    if hour and up_short and hour["texture"] == "closing":
        flags.append(("warn", f"rally on {hour['tag'].lower()} over {hour['tf']}m — "
                              f"positions CLOSING under a rising tape, not fresh buying (fade risk)"))
    if hour and dn_short and hour["texture"] == "closing":
        flags.append(("warn", f"drop on {hour['tag'].lower()} over {hour['tf']}m — "
                              f"shorts/longs closing, not fresh selling (bounce risk)"))

    # 2. price vs OI-positioning divergence on the hour (hidden distribution/accumulation)
    if hour and hour["px"] > _PX_DB and hour["pos_sign"] < 0:
        flags.append(("warn", f"price UP but OI positioning BEARISH over {hour['tf']}m "
                              f"(call-writing) — hidden distribution"))
    if hour and hour["px"] < -_PX_DB and hour["pos_sign"] > 0:
        flags.append(("ok", f"price DOWN but OI positioning BULLISH over {hour['tf']}m "
                            f"(put-writing) — hidden accumulation / support"))

    # 3. short vs hour OI-build divergence (recent build into longer-TF unwind)
    if short and hour:
        s_b = np.sign(np.mean([c["oi_build"] for c in short]))
        if s_b > 0 and hour["oi_build"] < 0:
            flags.append(("warn", f"5–10m adding OI but {hour['tf']}m net closing — "
                                  f"late entries into a longer-TF exit"))

    # 4. clean FUTURES-OI read: futures unwinding under a rising tape (or building under a fall)
    if hour and hour.get("foi_build") is not None:
        if up_short and hour["foi_build"] < 0:
            flags.append(("warn", f"futures OI UNWINDING over {hour['tf']}m while price rises — "
                                  f"longs closing, not fresh buying (clean futures read)"))
        if dn_short and hour["foi_build"] > 0:
            flags.append(("warn", f"futures SHORT BUILDUP over {hour['tf']}m as price falls — "
                                  f"fresh shorts pressing"))

    # overall bias from OI positioning across TFs (magnitude-aware)
    net_bias = sum(c["net"] for c in cells)
    bias = "BULLISH" if net_bias > 1 else "BEARISH" if net_bias < -1 else "NEUTRAL"
    return {"flags": flags, "bias": bias}
