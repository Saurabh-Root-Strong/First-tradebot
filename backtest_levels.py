"""
backtest_levels.py — do CONFIRMED horizontal S/R levels actually have edge?

Formalises the desk method: a price is a real level when it is (1) touched by
multiple swing pivots, (2) confirmed on BOTH 5m and 15m, (3) persists across >=2
days. Detected as-of t from prior bars only (lookahead-free), rebuilt from the
captured tick stream resampled to 5m + 15m across all prior sessions + today<=t.

Two claims, each measured as a directional hit rate with a day-block bootstrap CI:
  A. REVERSION  — when price sits within a tol band of a confirmed level, does it
                  react the level's way (bounce off support / reject at resistance)?
  B. BREAKOUT   — when the last 15m bar CLOSES beyond a confirmed level by a margin,
                  does price CONTINUE (real break) rather than snap back (false)?

Both signed so >50% = the level method was right. Baseline = 50% (coin). Edge only
if the day-block CI clears 50. This is the price-structure analogue of the OI-wall
test (eod_oi_range) — does PRICE-derived confirmed structure bound/turn better than
chance, net of nothing yet (skill first, cost second).

    set LV_CACHE=...lv.parquet  &&  .venv\\Scripts\\python.exe backtest_levels.py
"""
from __future__ import annotations

import datetime
import glob
import math
import os
import sys

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from core.constants import IST, INDEX_SYMBOLS, LIVE_DIR
from core.mirror_io import read_mirror as _read

SEED = 7
PRIOR_DAYS = 5
PIVOT_K = 2                     # fractal pivot: extreme vs K bars each side
TOL_FRAC = 0.0012              # cluster / zone half-width (~0.12% of price)
PROX_FRAC = 0.0008            # "at the level" proximity band (~0.08%)
BRK_MARGIN = 0.0005           # close must clear the level by this to count as breakout
MIN_TOUCH = 3
MIN_DAYS = 2
ENTRY_TIMES = ["09:45", "10:15", "10:45", "11:15", "11:45", "12:15", "12:45", "13:15", "13:45", "14:15"]
HORIZONS = (15, 30)
CACHE = os.environ.get("LV_CACHE", "")


def _captured_days():
    out = set()
    for p in glob.glob(str(LIVE_DIR / "*_ticks.parquet")):
        if os.path.getsize(p) >= 2000:
            out.add(os.path.basename(p).split("_")[0])
    return sorted(out)


def _session_ticks(date, sym):
    t = _read("ticks", date, None, sym)
    if t is None or len(t) < 20:
        return None
    t = t[(t["ts"].dt.date == datetime.date.fromisoformat(date))
          & (t["ts"].dt.time >= datetime.time(9, 15))
          & (t["ts"].dt.time <= datetime.time(15, 30))]
    return t[["ts", "ltp"]].sort_values("ts") if len(t) else None


def _bars(ticks, tf):
    g = ticks.set_index("ts")["ltp"]
    return g.resample(f"{tf}min", origin="start_day").ohlc().dropna()


def _pivots(bars, tf):
    """Fractal swing highs/lows -> list of (ts, price, tf)."""
    hi = bars["high"].to_numpy(float); lo = bars["low"].to_numpy(float)
    idx = bars.index
    out = []
    k = PIVOT_K
    for i in range(k, len(bars) - k):
        win_hi = hi[i - k:i + k + 1]; win_lo = lo[i - k:i + k + 1]
        if hi[i] == win_hi.max() and (win_hi.argmax() == k):
            out.append((idx[i], float(hi[i]), tf))
        if lo[i] == win_lo.min() and (win_lo.argmin() == k):
            out.append((idx[i], float(lo[i]), tf))
    return out


def _confirmed_levels(pivots, as_of, spot):
    """Cluster pivots (ts<as_of) into zones; keep multi-touch, multi-TF, multi-day."""
    pv = [(ts, px, tf) for (ts, px, tf) in pivots if ts < as_of]
    if len(pv) < MIN_TOUCH:
        return []
    pv.sort(key=lambda x: x[1])
    tol = TOL_FRAC * spot
    clusters = []
    cur = [pv[0]]
    for p in pv[1:]:
        if p[1] - cur[-1][1] <= tol:
            cur.append(p)
        else:
            clusters.append(cur); cur = [p]
    clusters.append(cur)
    levels = []
    for c in clusters:
        prices = [x[1] for x in c]
        tfs = set(x[2] for x in c)
        days = set(x[0].date() for x in c)
        if len(c) >= MIN_TOUCH and (5 in tfs and 15 in tfs) and len(days) >= MIN_DAYS:
            levels.append(float(np.mean(prices)))
    return sorted(levels)


def harvest(days):
    rows = []
    for di, date in enumerate(days):
        if di < PRIOR_DAYS:
            continue
        priors = days[di - PRIOR_DAYS:di]
        d0 = datetime.date.fromisoformat(date)
        for sym in INDEX_SYMBOLS:
            today = _session_ticks(date, sym)
            if today is None:
                continue
            hist = [x for x in (_session_ticks(p, sym) for p in priors) if x is not None]
            if not hist:
                continue
            base = pd.concat(hist)
            for hhmm in ENTRY_TIMES:
                hh, mm = map(int, hhmm.split(":"))
                t = datetime.datetime.combine(d0, datetime.time(hh, mm), tzinfo=IST)
                upto = today[today["ts"] <= pd.Timestamp(t)]
                if len(upto) < 6:
                    continue
                spot = float(upto.iloc[-1]["ltp"])
                ctx = pd.concat([base, upto])
                b5 = _bars(ctx, 5); b15 = _bars(ctx, 15)
                if len(b15) < 2 * PIVOT_K + 2:
                    continue
                pivots = _pivots(b5, 5) + _pivots(b15, 15)
                levels = _confirmed_levels(pivots, pd.Timestamp(t), spot)
                if not levels:
                    continue
                la = np.array(levels)
                sup = la[la <= spot]; res = la[la >= spot]
                nsup = float(sup.max()) if len(sup) else None
                nres = float(res.min()) if len(res) else None
                # forward returns
                fwd = {}
                for H in HORIZONS:
                    sH = today[today["ts"] <= pd.Timestamp(t + datetime.timedelta(minutes=H))]
                    fwd[H] = float(sH.iloc[-1]["ltp"]) if len(sH) else None
                # last closed 15m bar (for breakout) — strictly before t
                cb = b15[b15.index < pd.Timestamp(t)]
                last_close = float(cb["close"].iloc[-1]) if len(cb) else spot
                # recent drift (regime proxy): net % move over the last ~30 min
                s30 = today[today["ts"] <= pd.Timestamp(t - datetime.timedelta(minutes=30))]
                spot30 = float(s30.iloc[-1]["ltp"]) if len(s30) else spot
                drift = (spot / spot30 - 1.0) * 100.0 if spot30 else 0.0
                rec = {"date": date, "sym": sym, "t": hhmm, "spot": spot, "drift": drift}
                # A) reversion: at support -> +ret ; at resistance -> -ret
                at_sup = nsup is not None and abs(spot - nsup) / spot < PROX_FRAC
                at_res = nres is not None and abs(spot - nres) / spot < PROX_FRAC
                rec["at_level"] = at_sup or at_res
                rev_sign = (1 if at_sup else (-1 if at_res else 0))
                # B) breakout: last 15m close beyond a confirmed level by margin
                up_brk = any(last_close > lv * (1 + BRK_MARGIN) >= 0 and
                             (spot >= lv) and (lv < last_close) and
                             abs(last_close - lv) / spot < 2 * TOL_FRAC for lv in levels)
                dn_brk = any(last_close < lv * (1 - BRK_MARGIN) and
                             (spot <= lv) and (lv > last_close) and
                             abs(last_close - lv) / spot < 2 * TOL_FRAC for lv in levels)
                brk_sign = (1 if up_brk and not dn_brk else (-1 if dn_brk and not up_brk else 0))
                rec["is_brk"] = brk_sign != 0
                for H in HORIZONS:
                    r = ((fwd[H] / spot - 1.0) * 100.0) if (fwd[H] and fwd[H] != spot) else np.nan
                    rec[f"rev{H}"] = rev_sign * r if rev_sign else np.nan
                    rec[f"brk{H}"] = brk_sign * r if brk_sign else np.nan
                rows.append(rec)
    return pd.DataFrame(rows)


def _wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * p, 100 * (c - h), 100 * (c + h)


def _dayblock(hits, days, rng, reps=2000):
    hits = np.asarray(hits, float); days = np.asarray(days)
    keep = ~np.isnan(hits); hits, days = hits[keep], days[keep]
    if len(hits) < 5:
        return float("nan"), float("nan"), float("nan"), 0
    u = np.unique(days); idx = {d: np.where(days == d)[0] for d in u}
    bs = [hits[np.concatenate([idx[d] for d in rng.choice(u, len(u), replace=True)])].mean()
          for _ in range(reps)]
    return 100 * hits.mean(), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5), len(hits)


def report(df, rng):
    print("\nCONFIRMED LEVEL BACKTEST — reversion + breakout, 5m&15m confirmed, multi-day")
    print("=" * 86)
    days = sorted(df.date.unique())
    print(f"  rows={len(df)}  days={len(days)} ({days[0]}..{days[-1]})  "
          f"at-level={int(df.at_level.sum())}  breakout={int(df.is_brk.sum())}")

    def line(col, label, mask):
        sub = df[mask]
        for H in HORIZONS:
            s = sub[f"{col}{H}"]
            sgn = (s > 0).astype(float).where(s.notna())
            n = int(sgn.notna().sum())
            if n < 5:
                print(f"   {label:11s} {H}m  n={n} (thin)"); continue
            wm, wlo, whi = _wilson(int(np.nansum(sgn)), n)
            bm, blo, bhi, _ = _dayblock(sgn, sub.date.to_numpy(), rng)
            mret = np.nanmean(s)
            flag = "+" if blo > 50 else ("-" if bhi < 50 else "0")
            print(f"   {label:11s} {H}m  hit {wm:4.0f}%  Wilson[{wlo:4.0f},{whi:4.0f}]  "
                  f"day-block[{blo:4.0f},{bhi:4.0f}][{flag}]  meanRet {mret:+.3f}%  n={n}")

    print("\n  A) REVERSION at a confirmed level (bounce/reject) — ALL")
    line("rev", "REVERSION", df.at_level)
    rng_mask = df.at_level & (df.drift.abs() < 0.10)     # quiet approach = range
    trd_mask = df.at_level & (df.drift.abs() >= 0.10)    # driving approach = trend
    print("\n  A-range) REVERSION when approached QUIETLY (|30m drift|<0.10% = range)")
    line("rev", "REV-RANGE", rng_mask)
    print("\n  A-trend) REVERSION when DRIVEN into the level (|drift|>=0.10% = trend)")
    line("rev", "REV-TREND", trd_mask)
    print("\n  B) CONTINUATION after a confirmed 15m-close breakout")
    line("brk", "BREAKOUT", df.is_brk)
    print("\n" + "=" * 86)
    print("READ: edge only where day-block CI clears 50. Reversion>50 = levels turn price;")
    print("Breakout>50 = real breaks continue. Both straddling 50 = confirmed levels are a")
    print("RANGE/RISK map (where reaction is likely) but not a >chance directional trigger.")


def main():
    rng = np.random.default_rng(SEED)
    if CACHE and os.path.exists(CACHE):
        df = pd.read_parquet(CACHE); print(f"loaded cache {CACHE} rows={len(df)}")
    else:
        days = _captured_days()
        print(f"captured days: {days}")
        df = harvest(days)
        if len(df) == 0:
            print("no rows"); return
        if CACHE:
            df.to_parquet(CACHE); print(f"cached -> {CACHE}")
    report(df, rng)


if __name__ == "__main__":
    main()
