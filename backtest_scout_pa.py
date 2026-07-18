"""
backtest_scout_pa.py — does the TradeBoard SCOUT's price-action level-trade pay? (the ledger's
honest denominator, before we build a UI to display it)

The scout fires a directional setup (WITH-TREND CONTINUATION / RANGE-BREAK / PULLBACK) with an
entry + target(resistance/band) + SL(support/band) on the INDEX. This grades that EXACT trade
across 2yr of index candles: from the setup bar, does price hit TARGET or SL first within H
bars (target=win, SL=loss, else timeout)? Reports win%, expectancy in R, per-index + OOS split.
Traded as a LEVEL/futures trade (index move), so cost is tiny (~1-3bps) — this is the honest
horizon, unlike the naked option arrow (3% floor, measured negative-EV).

Reuses tradeboard's synthesize + _pivots + the structure classifier logic, recomputed on clean
2yr historical candles (data/historical/5min → resampled to LTF/HTF). Sampled every LTF bar.

    .venv\\Scripts\\python.exe backtest_scout_pa.py [--ltf 15 --htf 60 --hold 8]
"""
from __future__ import annotations

import argparse
import sys
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from backtest_continuity import _boot_ci
from tradeboard import _pivots, synthesize, _TREND_UP_S, _TREND_DN_S

FY = {"NSE:NIFTY50-INDEX": "NIFTY50", "NSE:NIFTYBANK-INDEX": "NIFTYBANK",
      "NSE:FINNIFTY-INDEX": "FINNIFTY", "NSE:MIDCPNIFTY-INDEX": "MIDCPNIFTY"}
SEED = 7


def _resample(df5, k):
    if k <= 1:
        return df5
    out = []
    for _, g in df5.groupby("d"):
        g = g.sort_values("ts").reset_index(drop=True)
        a = g.groupby(g.index // k).agg(ts=("ts", "first"), open=("open", "first"),
                                        high=("high", "max"), low=("low", "min"),
                                        close=("close", "last"))
        a["d"] = g["d"].iloc[0]; out.append(a)
    return pd.concat(out, ignore_index=True)


def _struct_at(h, l, c, i, lb=20):
    """Structure label at bar i on arrays (matches tradeboard._struct_full, gap-unaware ER is
    fine here — same-index single series). Returns (struct, hi, lo)."""
    if i < lb:
        return "n/a", None, None
    seg = c[i - lb + 1:i + 1]
    net = seg[-1] - seg[0]
    path = np.abs(np.diff(seg)).sum()
    er = abs(net) / path if path > 0 else 0.0
    hh = h[i - lb + 1:i + 1]; ll = l[i - lb + 1:i + 1]
    last = c[i]
    if last > hh[:-1].max():
        return "BREAKOUT_UP", float(hh.max()), float(ll.min())
    if last < ll[:-1].min():
        return "BREAKOUT_DOWN", float(hh.max()), float(ll.min())
    if er >= 0.4:
        return ("TREND_UP" if net > 0 else "TREND_DOWN"), float(hh.max()), float(ll.min())
    prior = hh[:-3].max() - ll[:-3].min(); recent = hh[-3:].max() - ll[-3:].min()
    if prior > 0 and recent < 0.6 * prior:
        return "CONSOLIDATION", float(hh.max()), float(ll.min())
    return "RANGE", float(hh.max()), float(ll.min())


def grade(sym, ltf, htf, hold):
    df = pd.read_parquet(f"data/historical/5min/NSE_{FY[sym]}_INDEX_5min.parquet")
    df["ts"] = pd.to_datetime(df["ts"]); df["d"] = df["ts"].dt.date
    L = _resample(df, ltf // 5); H = _resample(df, htf // 5)
    Lh, Ll, Lc = L["high"].to_numpy(float), L["low"].to_numpy(float), L["close"].to_numpy(float)
    Hh, Hl, Hc = H["high"].to_numpy(float), H["low"].to_numpy(float), H["close"].to_numpy(float)
    Lts, Hts = L["ts"].to_numpy(), H["ts"].to_numpy()
    recs = []
    hj = 0
    for i in range(24, len(Lc) - hold):
        # align HTF bar (last HTF close <= this LTF bar's time)
        while hj + 1 < len(Hts) and Hts[hj + 1] <= Lts[i]:
            hj += 1
        if hj < 24:
            continue
        ls, _, _ = _struct_at(Lh, Ll, Lc, i)
        hs, hhi, hlo = _struct_at(Hh, Hl, Hc, hj)
        if ls == "n/a" or hs == "n/a":
            continue
        spot = Lc[i]
        s = synthesize({"struct": hs, "hi": hhi, "lo": hlo, "n": 30},
                       {"struct": ls, "n": 30}, spot)
        tag = s.get("tag", "")
        # direction
        if "RANGE-TOP" in tag:
            lean = "UP"
        elif "RANGE-FLOOR" in tag:
            lean = "DOWN"
        elif hs in _TREND_UP_S:
            lean = "UP"
        elif hs in _TREND_DN_S:
            lean = "DOWN"
        else:
            lean = None
        if lean is None or not ("CONTINUATION" in tag or "BREAK (attempt)" in tag
                                or "PULLBACK" in tag):
            continue
        # levels from HTF pivots (up to hj)
        his, los = _pivots(Hh[max(0, hj - 40):hj + 1], Hl[max(0, hj - 40):hj + 1], w=3)
        atr = float(np.mean(Hh[hj - 13:hj + 1] - Hl[hj - 13:hj + 1]))
        md = 0.25 * atr
        res = min((x for x in his if x > spot + md), default=spot + atr)
        sup = max((x for x in los if x < spot - md), default=spot - atr)
        if lean == "UP":
            entry, target, stop = spot, res, sup
        else:
            entry, target, stop = spot, sup, res
        if abs(entry - stop) < 1e-6:
            continue
        # grade forward on LTF bars: target or stop first within `hold`
        outcome, exitpx = "timeout", Lc[i + hold]
        for j in range(i + 1, i + hold + 1):
            if lean == "UP":
                if Ll[j] <= stop:
                    outcome, exitpx = "stop", stop; break
                if Lh[j] >= target:
                    outcome, exitpx = "target", target; break
            else:
                if Lh[j] >= stop:
                    outcome, exitpx = "stop", stop; break
                if Ll[j] <= target:
                    outcome, exitpx = "target", target; break
        rmult = ((exitpx - entry) if lean == "UP" else (entry - exitpx)) / abs(entry - stop)
        recs.append({"date": Lts[i], "lean": lean, "tag": tag, "outcome": outcome,
                     "R": rmult, "win": int(outcome == "target")})
    return pd.DataFrame(recs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ltf", type=int, default=15); ap.add_argument("--htf", type=int, default=60)
    ap.add_argument("--hold", type=int, default=8)
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    print("=" * 82)
    print(f"  SCOUT PA LEVEL-TRADE — does it pay? (LTF {a.ltf}m entry × HTF {a.htf}m confirm, "
          f"hold {a.hold} bars, target=R / SL=S, INDEX level trade)")
    print("=" * 82)
    for sym in FY:
        try:
            d = grade(sym, a.ltf, a.htf, a.hold)
        except Exception as e:
            print(f"  {FY[sym]}: err {e}"); continue
        if len(d) < 30:
            print(f"  {FY[sym]}: thin ({len(d)})"); continue
        d["day"] = pd.to_datetime(d["date"]).dt.year
        R = d["R"].to_numpy(float)
        mr, lo, hi = _boot_ci(lambda x: x.mean(), R, reps=3000, rng=rng, groups=d["day"].to_numpy())
        v = "EDGE" if lo > 0 else ("bleed" if hi < 0 else "—")
        oc = dict(d["outcome"].value_counts())
        print(f"  {FY[sym]:<11} n={len(d):>5}  win {100*d.win.mean():4.1f}%  "
              f"exp {mr:+.3f}R [{lo:+.3f},{hi:+.3f}] {v}  {oc}")
        # OOS split
        yrs = sorted(d.day.unique()); cut = yrs[len(yrs) // 2]
        te = d[d.day >= cut]
        if len(te) > 30:
            Rt = te["R"].to_numpy(float)
            m2, l2, h2 = _boot_ci(lambda x: x.mean(), Rt, reps=3000, rng=rng,
                                  groups=te["day"].to_numpy())
            print(f"              OOS test (yr>={cut}) exp {m2:+.3f}R [{l2:+.3f},{h2:+.3f}] "
                  f"{'EDGE' if l2>0 else 'bleed' if h2<0 else '—'}  n={len(te)}")
    print("\n  READ: EDGE only if expectancy-R CI clears 0 OOS. Level trade (index/futures, tiny")
    print("  cost). If it bleeds like every prior intraday gate, the ledger will confirm the PA")
    print("  scout is context — trade the levels with YOUR discipline, not a mechanical fire.")


if __name__ == "__main__":
    main()
