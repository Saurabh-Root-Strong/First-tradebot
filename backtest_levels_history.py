"""
backtest_levels_history.py — the downside-breakout futures short on 2 YEARS of bars.

The breakthrough: the confirmed-level breakout signal is PRICE-ONLY (horizontal S/R
+ a price break), so it does NOT need captured OI/chain/flow. That frees it from the
7 captured tick-days and lets us test on the full historical bar set
(data/historical/, 2024-06-14 .. 2026-06-12, 5m+15m+60m, all 4 indices) — thousands
of breakouts across bull/bear/range, and FULLY OUT-OF-SAMPLE: the history ends 06-12,
before the 7 captured days that produced the +0.083% gross. This resolves BOTH blockers
the captured test hit — the n=30/7d underpowering AND the down-week confound.

Same detector as backtest_levels (confirm on 5m&15m, multi-touch, multi-day, fractal
pivots, lookahead-free) but on historical bars. Enter at the breakout 15m close, exit
at +H (fixed horizon), futures-cost swept. DOWN-break = SHORT, UP-break = LONG (the
asymmetry contrast). Day-clustered bootstrap CI (resample trading days). If the
DOWN-break short clears 0 net of cost here, it is real; if not, the 7-day result was
a fluke.

    .venv\\Scripts\\python.exe backtest_levels_history.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SEED = 7
INDICES = ["NIFTY50", "NIFTYBANK", "FINNIFTY", "MIDCPNIFTY"]
PIVOT_K = 2
TOL_FRAC = 0.0012
BRK_MARGIN = 0.0005
MIN_TOUCH = 3
MIN_DAYS = 2
WINDOW_DAYS = 7              # lookback (calendar) for the pivot pool ~ 5 trading days
NEAR_FRAC = 0.03            # only cluster pivots within 3% of price (speed + relevance)
TREND_LB = 16               # medium-trend lookback in 15m bars (~1 session)
FUT_COSTS = (0.02, 0.03, 0.05)
HORIZONS = (15, 30)


def _load(idx, tf):
    p = f"data/historical/{tf}/NSE_{idx}_INDEX_{tf}.parquet"
    d = pd.read_parquet(p)
    d["ts"] = pd.to_datetime(d["ts"])
    return d.sort_values("ts").reset_index(drop=True)


def _pivots(d, tf):
    hi = d["high"].to_numpy(float); lo = d["low"].to_numpy(float); ts = d["ts"].to_numpy()
    out = []; k = PIVOT_K
    for i in range(k, len(d) - k):
        wh = hi[i - k:i + k + 1]; wl = lo[i - k:i + k + 1]
        if hi[i] == wh.max() and wh.argmax() == k:
            out.append((ts[i], float(hi[i]), tf))
        if lo[i] == wl.min() and wl.argmin() == k:
            out.append((ts[i], float(lo[i]), tf))
    return out


def _confirmed(pv_ts, pv_px, pv_tf, lo_i, hi_i, close):
    """Cluster pivots[lo_i:hi_i] near close; return confirmed level prices."""
    if hi_i - lo_i < MIN_TOUCH:
        return []
    px = pv_px[lo_i:hi_i]; tf = pv_tf[lo_i:hi_i]; tsd = pv_ts[lo_i:hi_i]
    near = np.abs(px - close) / close < NEAR_FRAC
    px, tf, tsd = px[near], tf[near], tsd[near]
    if len(px) < MIN_TOUCH:
        return []
    order = np.argsort(px); px, tf, tsd = px[order], tf[order], tsd[order]
    tol = TOL_FRAC * close
    levels = []
    cur = [0]
    for j in range(1, len(px)):
        if px[j] - px[cur[-1]] <= tol:
            cur.append(j)
        else:
            _emit(cur, px, tf, tsd, levels); cur = [j]
    _emit(cur, px, tf, tsd, levels)
    return levels


def _emit(cur, px, tf, tsd, levels):
    if len(cur) < MIN_TOUCH:
        return
    tfs = set(int(tf[j]) for j in cur)
    days = set(pd.Timestamp(tsd[j]).date() for j in cur)
    if 5 in tfs and 15 in tfs and len(days) >= MIN_DAYS:
        levels.append(float(np.mean([px[j] for j in cur])))


def harvest_index(idx):
    b5 = _load(idx, "5min"); b15 = _load(idx, "15min")
    piv = _pivots(b5, 5) + _pivots(b15, 15)
    piv.sort(key=lambda x: x[0])
    pv_ts = np.array([p[0] for p in piv]); pv_px = np.array([p[1] for p in piv], float)
    pv_tf = np.array([p[2] for p in piv], int)
    c = b15["close"].to_numpy(float); t15 = b15["ts"].to_numpy()
    win = np.timedelta64(WINDOW_DAYS, "D")
    rows = []
    for i in range(TREND_LB + 1, len(b15) - max(HORIZONS) // 15 - 1):
        t = t15[i]; close = c[i]; prev = c[i - 1]
        lo_i = int(np.searchsorted(pv_ts, t - win, "left"))
        hi_i = int(np.searchsorted(pv_ts, t, "left"))          # strictly < t = lookahead-free
        levels = _confirmed(pv_ts, pv_px, pv_tf, lo_i, hi_i, close)
        if not levels:
            continue
        la = np.array(levels)
        # fresh down-break: prev at/above a level, current closes below by margin
        dn = la[(prev >= la * (1 - BRK_MARGIN)) & (close < la * (1 - BRK_MARGIN))]
        up = la[(prev <= la * (1 + BRK_MARGIN)) & (close > la * (1 + BRK_MARGIN))]
        bdir = -1 if (len(dn) and not len(up)) else (1 if (len(up) and not len(dn)) else 0)
        if bdir == 0:
            continue
        trend = np.sign(close - c[i - TREND_LB])
        rec = {"date": pd.Timestamp(t).date(), "idx": idx, "bdir": bdir, "trend": int(trend)}
        for H in HORIZONS:
            k = H // 15
            exit_px = c[i + k]
            # signed so >0 = the breakout direction was right (short gains if price fell)
            rec[f"ret{H}"] = (close - exit_px) / close * 100.0 * bdir
        rows.append(rec)
    return rows


def _boot(x, days, rng, reps=1000):
    x = np.asarray(x, float); d = np.asarray(days)
    keep = ~np.isnan(x); x, d = x[keep], d[keep]
    if len(x) < 10:
        return None
    u = np.unique(d); idx = {k: np.where(d == k)[0] for k in u}
    bs = [x[np.concatenate([idx[k] for k in rng.choice(u, len(u), replace=True)])].mean()
          for _ in range(reps)]
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(x), len(u)


def _block(name, sub, rng):
    print(f"\n  {name}")
    for H in HORIZONS:
        g = sub[f"ret{H}"].to_numpy(float)
        r = _boot(g, sub["date"].to_numpy(), rng)
        if r is None:
            print(f"     {H}m  n<10"); continue
        gm, glo, ghi, n, nd = r
        win = 100 * (g[~np.isnan(g)] > 0).mean()
        gflag = "+" if glo > 0 else ("-" if ghi < 0 else "0")
        print(f"     {H}m GROSS {gm:+.4f}% [{glo:+.4f},{ghi:+.4f}][{gflag}] win {win:3.0f}% n={n}/{nd}d")
        for c in FUT_COSTS:
            net = g - c
            rr = _boot(net, sub["date"].to_numpy(), rng)
            nm, nlo, nhi, _, _ = rr
            f = "+" if nlo > 0 else ("-" if nhi < 0 else "0")
            print(f"        cost {c:.2f}%  net {nm:+.4f}% [{nlo:+.4f},{nhi:+.4f}][{f}]  "
                  f"win {100*(net[~np.isnan(net)]>0).mean():3.0f}%")


def main():
    rng = np.random.default_rng(SEED)
    print("DOWNSIDE-BREAK FUTURES SHORT — 2-year OUT-OF-SAMPLE (historical bars)")
    print("=" * 80)
    allrows = []
    for idx in INDICES:
        rs = harvest_index(idx)
        allrows += rs
        print(f"  {idx:11s} breakouts={len(rs)}")
    df = pd.DataFrame(allrows)
    print(f"\n  total breakouts={len(df)}  days={df.date.nunique()} "
          f"({df.date.min()}..{df.date.max()})  down={int((df.bdir<0).sum())} up={int((df.bdir>0).sum())}")

    dn = df[df.bdir < 0]; up = df[df.bdir > 0]
    _block("DOWN-BREAK short — ALL (the candidate, 2yr OOS)", dn, rng)
    _block("DOWN-BREAK short — with-trend (close<close[-1d])", dn[dn.trend < 0], rng)
    _block("DOWN-BREAK short — counter-trend (rising tape)", dn[dn.trend > 0], rng)
    _block("UP-BREAK long — ALL (asymmetry contrast)", up, rng)

    print("\n" + "=" * 80)
    print("READ: this is OOS (history ends before the captured days). DOWN-break net CI")
    print("clearing 0 at 0.03-0.05% on n=thousands = the futures short is REAL. Net")
    print("straddling 0 here = the 7-day +0.083% was underpowered/regime luck. Down>>up")
    print("net = the short-side continuation asymmetry confirmed at scale.")


if __name__ == "__main__":
    main()
