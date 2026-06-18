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

from core.constants import IST, LABELS, LABEL_TO_SYM, NSE_NAME as _NSE_UL   # noqa: E402
from core.mirror_io import read_mirror as _read                            # noqa: E402

TFS = [5, 10, 15, 60]
# A regime label is a DIRECTIONAL claim, so it must clear noise on BOTH axes:
#   price — a >= _Z_PX sigma move (volatility-normalised per TF), not a few points.
#           A fixed %-deadband mislabels a ~7-pt NIFTY wiggle (0.03%) as a trend;
#           1 sigma scales the bar to the day's actual vol and the window length.
#   OI    — a material build relative to the index's LIVE total OI, so the same bar
#           fits NIFTY (~hundreds of L) and MIDCAP (~ones of L) — not a flat 0.5L.
# Below either bar -> BALANCED (the honest "no confirmed regime"). A directional
# label is `confirmed` only when the move ALSO persisted through the window (not a
# terminal spike) on real option volume — otherwise it is shown as tentative.
_Z_PX      = 1.0      # price move >= 1 sigma (vol-normalised) to count as directional
_OI_FRAC   = 0.004    # |ΔOI| >= 0.4% of live total option OI to count as a build
_OI_ABS    = 0.3      # AND >= 0.3L absolute — blocks tiny builds in thin indices (FIN)
_POS_FRAC  = 0.004    # |net put-call| significance, same per-index scale
_IV_REAL   = 0.8      # realised vol ~0.8x implied (IV-implied fallback normaliser)
_SIG_FLOOR = 0.02     # min 1-min sigma % (guards divide-by-tiny early in the session)
_OI_DB     = 0.5      # futures-OI deadband in lakhs (NSE futures-OI mirror, own scale)


def _at_or_before(df, sym_col, sym, value_ts):
    sub = df[df["ts"] <= value_ts]
    return sub.iloc[-1] if len(sub) else None


def _sig1_pct(t: "pd.DataFrame") -> "float | None":
    """Realised 1-minute return stdev (%) from the day's ticks — the vol scale used
    to z-normalise each window's price move. None until enough history exists."""
    try:
        s = t.set_index("ts")["ltp"].resample("1min").last().dropna()
    except Exception:
        return None
    if len(s) < 12:
        return None
    r = s.pct_change().dropna() * 100.0
    if len(r) < 8:
        return None
    sig = float(r.std())
    return sig if sig > 0 else None


def _exp_move_pct(T: int, sig1: "float | None", atm_iv: float) -> float:
    """Expected 1-sigma price move (%) over a T-minute window. Prefers realised
    1-min vol (sqrt-time scaled); falls back to ATM-IV implied when history is thin."""
    if sig1 and sig1 > 0:
        return max(sig1, _SIG_FLOOR) * (T ** 0.5)
    if atm_iv and atm_iv > 0:
        daily = atm_iv / (252 ** 0.5)                 # annual IV% -> daily %
        return max(daily * ((T / 375.0) ** 0.5) * _IV_REAL, _SIG_FLOOR * (T ** 0.5))
    return _SIG_FLOOR * (T ** 0.5)


def _regime(z_px: float, oi_frac: float, persist: bool
            ) -> "tuple[str, str, int, bool, int, int]":
    """Vol-/scale-normalised OI x price regime.

    Returns (tag, texture, dir_sign, confirmed, pu, ou) where pu/ou are the
    SIGNIFICANT price / OI-build signs (-1/0/+1). A directional label needs BOTH a
    >= _Z_PX sigma price move AND a material OI build; otherwise BALANCED. The label
    is `confirmed` only when the move also persisted through the window."""
    pu = int((z_px >= _Z_PX) - (z_px <= -_Z_PX))
    ou = int((oi_frac >= _OI_FRAC) - (oi_frac <= -_OI_FRAC))
    if pu == 0 or ou == 0:
        return "BALANCED", "flat", 0, False, pu, ou
    if   pu > 0 and ou > 0: tag, tex, sgn = "LONG BUILDUP",  "fresh",   +1
    elif pu > 0 and ou < 0: tag, tex, sgn = "SHORT COVER",   "closing", +1
    elif pu < 0 and ou > 0: tag, tex, sgn = "SHORT BUILDUP", "fresh",   -1
    else:                   tag, tex, sgn = "LONG UNWIND",   "closing", -1
    return tag, tex, sgn, bool(persist), pu, ou


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

    first_ts = t["ts"].iloc[0]
    sig1 = _sig1_pct(t)                                   # realised 1-min vol scale (%)
    _iv  = o.iloc[-1]["atm_iv"] if "atm_iv" in o.columns else 0
    atm_iv = float(_iv) if pd.notna(_iv) else 0.0         # IV-implied fallback scale
    cells = []
    pending = []      # TFs whose lookback predates capture-start — show ETA, not silence
    for T in TFS:
        anchor = now_ts - pd.Timedelta(minutes=T)
        p0, p1 = price_at(anchor), float(t["ltp"].iloc[-1])
        if p0 is None or p0 <= 0:
            # cell unlocks once capture spans T minutes: first_tick + T
            pending.append({"tf": T, "eta": (first_ts + pd.Timedelta(minutes=T)).strftime("%H:%M")})
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
        # ── significance-gated regime ──────────────────────────────────────────
        # price: z-score vs the window's expected 1-sigma move (not a fixed %).
        sigT = _exp_move_pct(T, sig1, atm_iv)
        z_px = px / sigT if sigT > 0 else 0.0
        # OI build: fraction of the index's LIVE total OI (self-scaling per index).
        total_oi = float(o_now["total_call_oi"] or 0) + float(o_now["total_put_oi"] or 0)
        oi_frac  = (d_tot * 1e5) / total_oi if total_oi > 0 else 0.0
        pos_frac = (net   * 1e5) / total_oi if total_oi > 0 else 0.0
        if abs(d_tot) < _OI_ABS:   # dual gate: relative AND a small absolute floor
            oi_frac = 0.0
        # persistence: the 2nd half of the window must agree in sign with the full
        # move (kills terminal spikes already reversing) AND show real option volume.
        p_mid   = price_at(now_ts - pd.Timedelta(minutes=T / 2))
        half_px = ((p1 - p_mid) / p_mid * 100.0) if (p_mid and p_mid > 0) else px
        vol_ok  = (ov is None) or (ov > 0)
        persist = bool(np.sign(half_px) == np.sign(px) and px != 0 and vol_ok)
        tag, texture, dir_sign, confirmed, pu, ou = _regime(z_px, oi_frac, persist)
        oi_build = int(ou)                                 # significant build dir (+1/-1/0)
        pos_sign = int((pos_frac >= _POS_FRAC) - (pos_frac <= -_POS_FRAC))
        # Independent flow corroboration of the regime DIRECTION (price-independent):
        #   pos_ok — option writers agree (put-writing under a bull / call-writing under a bear)
        #   fut_ok — futures price agrees AND basis isn't contradicting (real aggression)
        pos_ok = bool(dir_sign != 0 and pos_sign == dir_sign)
        fut_ok = bool(dir_sign != 0 and fpx is not None and np.sign(fpx) == dir_sign
                      and (fbasis is None or np.sign(fbasis) != -dir_sign))
        cells.append({"tf": T, "px": round(px, 3), "z": round(z_px, 2),
                      "confirmed": confirmed, "pu": int(pu),
                      "pos_ok": pos_ok, "fut_ok": fut_ok,
                      "d_call": round(d_call, 1),
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

    # ── Multi-timeframe confirmation ────────────────────────────────────────────
    # A directional call is only "aligned" (tradeable) when a HIGHER timeframe
    # agrees and NONE opposes, AND an independent flow (OI positioning or futures)
    # corroborates it. Longer TFs set the bias, shorter TFs are confirmed by them —
    # so a lone 5/10m blip that the stack doesn't back is graded down, not traded.
    #   aligned   ★  persisted + higher-TF agrees (no opposition) + >=1 flow confirm
    #   confirmed ✓  persisted + (higher-TF agrees OR flow confirms)  [solid 1-TF]
    #   tentative ~  significant but unpersisted / uncorroborated      [watch only]
    by = {c["tf"]: c for c in cells}
    dir_cells = [c for c in cells if c["dir_sign"] != 0]
    wsum   = sum(c["tf"] * c["dir_sign"] for c in dir_cells)   # TF-length-weighted vote
    dominant = int(np.sign(wsum)) if dir_cells else 0
    top_tf = max(c["tf"] for c in cells)
    for c in cells:
        c["confirms"] = []
        if c["dir_sign"] == 0:
            c["grade"] = "balanced"
            continue
        d = c["dir_sign"]
        peers = [T for T in TFS if (T > c["tf"] if c["tf"] < top_tf else T < c["tf"])]
        agree  = [T for T in peers if T in by and by[T]["dir_sign"] == d]
        contra = [T for T in peers if T in by and by[T]["dir_sign"] == -d]
        stack_ok = (len(agree) >= 1 and not contra) or (c["tf"] == top_tf and not contra)
        flow = (["OI"] if c["pos_ok"] else []) + (["fut"] if c["fut_ok"] else [])
        c["confirms"] = [f"{T}m" for T in agree] + flow
        strong = c["confirmed"]                                # persisted + real volume
        if strong and stack_ok and flow:
            c["grade"] = "aligned"
        elif strong and (stack_ok or flow):
            c["grade"] = "confirmed"
        else:
            c["grade"] = "tentative"
    stack = {"dir": dominant,
             "agree": sum(1 for c in dir_cells if c["dir_sign"] == dominant),
             "total": len(dir_cells),
             "aligned": [c["tf"] for c in cells if c.get("grade") == "aligned"]}

    fut_oi_chg = int(float(fo.iloc[-1]["chg_oi"])) if fo is not None else None   # raw contracts
    fut_oi_day = round(fut_oi_chg / 1e5, 1) if fut_oi_chg is not None else None  # lakhs (compat)
    verdict = _synthesize(cells)
    return {"sym": sym, "label": LABELS.get(sym, sym), "has_data": True,
            "now": now_ts.strftime("%H:%M:%S"), "spot": round(float(t["ltp"].iloc[-1]), 2),
            "fut_oi_day": fut_oi_day, "fut_oi_chg": fut_oi_chg, "cells": cells,
            "pending": pending, "stack": stack, **verdict}


def _synthesize(cells: list[dict]) -> dict:
    by = {c["tf"]: c for c in cells}
    short = [by[t] for t in (5, 10) if t in by]
    hour = by.get(60) or by.get(15)
    flags = []

    # 1. tape rising on closing (short-cover / unwind) rather than fresh positioning
    up_short = short and all(c["pu"] > 0 for c in short)
    dn_short = short and all(c["pu"] < 0 for c in short)
    if hour and up_short and hour["texture"] == "closing":
        flags.append(("warn", f"rally on {hour['tag'].lower()} over {hour['tf']}m — "
                              f"positions CLOSING under a rising tape, not fresh buying (fade risk)"))
    if hour and dn_short and hour["texture"] == "closing":
        flags.append(("warn", f"drop on {hour['tag'].lower()} over {hour['tf']}m — "
                              f"shorts/longs closing, not fresh selling (bounce risk)"))

    # 2. price vs OI-positioning divergence on the hour (hidden distribution/accumulation)
    if hour and hour["pu"] > 0 and hour["pos_sign"] < 0:
        flags.append(("warn", f"price UP but OI positioning BEARISH over {hour['tf']}m "
                              f"(call-writing) — hidden distribution"))
    if hour and hour["pu"] < 0 and hour["pos_sign"] > 0:
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
