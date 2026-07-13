"""
backtest_btst_tail.py — "what if tomorrow gaps DOWN?" Measure the damage, don't hand-wave it.

The BTST rule is long-only. It is wrong ~31% of nights, and a gap-down cannot be stopped out:
the market simply OPENS there. So the honest questions are not "will it gap down" (it will)
but:
  1. HOW BAD is the tail, in nights-of-gains and in rupees?
  2. Can the worst nights be AVOIDED -- is there a regime in which the edge inverts?
  3. What does an OTM PUT hedge cost, and does it leave anything behind?
  4. Does the edge survive the tail at all -- what does the equity curve actually do?

Anything that cannot be answered here is a reason NOT to put real money on this yet.

    .venv\\Scripts\\python.exe backtest_btst_tail.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import btst_signal as bs
from backtest_stbt import overnight_table, COST
from core.constants import LOT_SIZES

_FY = {"NIFTY": "NSE:NIFTY50-INDEX", "BANKNIFTY": "NSE:NIFTYBANK-INDEX",
       "FINNIFTY": "NSE:FINNIFTY-INDEX", "MIDCPNIFTY": "NSE:MIDCPNIFTY-INDEX"}


def build():
    df = overnight_table()
    df["long_bps"] = -(df["short_bps"] + COST) - COST      # flip short back to LONG, net
    df = df[df["clr"] >= bs.CLR_TH].sort_values("date").reset_index(drop=True)
    df["lot"] = df["sym"].map(lambda s: LOT_SIZES.get(_FY[s]))
    df["rupees"] = df["close"] * (df["long_bps"] / 1e4) * df["lot"]
    # regime, causal: today's close vs its own 50d / 200d mean, computed per index on the
    # DAILY archive (known at the close of D — no lookahead)
    reg = {}
    for s in bs.SYMS:
        d = bs._daily(s)
        d["ma50"] = d["close"].rolling(50).mean()
        d["ma200"] = d["close"].rolling(200).mean()
        d["above50"] = d["close"] > d["ma50"]
        d["above200"] = d["close"] > d["ma200"]
        # realised vol regime (20d), also causal
        d["vol20"] = d["close"].pct_change().rolling(20).std() * 1e4
        reg[s] = d.set_index("date")[["above50", "above200", "vol20"]]
    df["above50"] = [reg[r.sym]["above50"].get(r.date, np.nan) for r in df.itertuples()]
    df["above200"] = [reg[r.sym]["above200"].get(r.date, np.nan) for r in df.itertuples()]
    df["vol20"] = [reg[r.sym]["vol20"].get(r.date, np.nan) for r in df.itertuples()]
    return df


def stats(v):
    v = np.asarray(v, float)
    eq = np.cumsum(v)
    dd = (eq - np.maximum.accumulate(eq)).min()
    sh = v.mean() / v.std() * np.sqrt(252) if v.std() > 0 else 0
    return dict(n=len(v), mean=v.mean(), win=100 * (v > 0).mean(), worst=v.min(),
                dd=dd, sharpe=sh, total=v.sum())


def main():
    df = build()
    v = df["long_bps"].to_numpy(float)
    s = stats(v)
    print("=" * 88)
    print("1. HOW BAD IS THE TAIL?   (long-only, clr>=0.66, net 3bps, exit 09:30)")
    print("=" * 88)
    print(f"  {s['n']} signal-nights   mean {s['mean']:+.1f} bps   win {s['win']:.0f}%   "
          f"Sharpe {s['sharpe']:+.2f}")
    print(f"  losing nights: {100-s['win']:.0f}%   <-- you WILL gap down. it is priced in.\n")
    for q in (1, 5, 10):
        print(f"    worst {q:>2d}% of nights : {np.percentile(v, q):>+8.1f} bps")
    print(f"    WORST night ever    : {v.min():>+8.1f} bps  "
          f"= {abs(v.min())/s['mean']:.0f} nights of average gains, gone in one open")
    # rupees, on one lot
    r = df["rupees"].to_numpy(float)
    print(f"\n  in RUPEES on ONE lot:  mean {r.mean():>+8,.0f}   worst night {r.min():>+9,.0f}")
    print(f"  a 4-lot book (one per index) would have lost "
          f"{df.groupby('date')['rupees'].sum().min():>+,.0f} on its worst DAY")

    print("\n" + "=" * 88)
    print("2. CAN THE WORST NIGHTS BE AVOIDED?  (regime, all causal — known at the close)")
    print("=" * 88)
    print(f"  {'filter':34s} {'n':>5s} {'mean':>8s} {'win':>5s} {'worst':>8s} {'maxDD':>8s} {'Sharpe':>7s}")
    print("  " + "-" * 80)
    cuts = {
        "ALL signal nights (the rule)":  pd.Series(True, index=df.index),
        "index ABOVE its 200d mean":     df["above200"] == True,
        "index BELOW its 200d mean":     df["above200"] == False,
        "index ABOVE its 50d mean":      df["above50"] == True,
        "index BELOW its 50d mean":      df["above50"] == False,
        "calm vol (20d below median)":   df["vol20"] < df["vol20"].median(),
        "high vol (20d above median)":   df["vol20"] >= df["vol20"].median(),
        "drop MIDCAP":                   df["sym"] != "MIDCPNIFTY",
        "above 200d AND drop MIDCAP":    (df["above200"] == True) & (df["sym"] != "MIDCPNIFTY"),
    }
    for nm, m in cuts.items():
        x = df.loc[m.fillna(False), "long_bps"].to_numpy(float)
        if len(x) < 40:
            print(f"  {nm:34s} {len(x):>5d}   too few")
            continue
        st = stats(x)
        print(f"  {nm:34s} {st['n']:>5d} {st['mean']:>+8.1f} {st['win']:>4.0f}% "
              f"{st['worst']:>+8.0f} {st['dd']:>+8.0f} {st['sharpe']:>+7.2f}")
    print("  " + "-" * 80)

    print("\n" + "=" * 88)
    print("3. DOES IT SURVIVE THE TAIL?  equity curve, worst drawdown, recovery")
    print("=" * 88)
    eq = np.cumsum(v)
    peak = np.maximum.accumulate(eq)
    ddser = eq - peak
    i = int(np.argmin(ddser))
    print(f"  cumulative {eq[-1]:+.0f} bps over {len(v)} nights")
    print(f"  max drawdown {ddser.min():+.0f} bps = {abs(ddser.min())/s['mean']:.0f} "
          f"nights of average gains")
    rec = np.where(eq[i:] >= peak[i])[0]
    print(f"  recovered from it in {rec[0] if len(rec) else 'NOT YET'} nights"
          if len(rec) else "  NEVER recovered in-sample")
    print(f"  the edge is the MEAN of a WIDE distribution: std {v.std():.0f} bps vs "
          f"mean {s['mean']:.1f} bps.")
    print(f"  you need ~{(v.std()/s['mean'])**2:.0f} nights before the mean dominates the noise.")


if __name__ == "__main__":
    main()
