"""
backtest_overnight_8yr.py — stress-test the ONE validated edge across 8.5 YEARS + regimes.

The overnight close-strength edge ([[project_btst_overnight_edge]]): a strong close (clr =
close-location-in-range >= 0.66) -> go LONG the index into the close, exit next open. It was
validated on ~2yr of RECENT (bull-heavy) data. The question 8.5yr of EOD finally answers:
does it survive the regimes it was never tested in — the 2018-20 bear, the COVID crash, the
2022 correction — or is it a bull artifact that inverts in downtrends (like the sector tilt)?

Data: DCM index_data (2018-01-01 .. today), daily OHLC. clr from the DAILY bar; overnight
gap = next_open/close - 1 (the realised overnight return, cost is a few bps on index futures
— tiny vs a 10-40bps gap, so overnight is NOT cost-floored like intraday).

Regime = today's close vs its 200-day SMA (bull if above, bear if below) — the simplest
causal regime split. Reports, per index: IC(clr, overnight_gap) overall + per regime, the
mean overnight gap for the strong-close bucket vs all, per-YEAR, and a day-block bootstrap CI.

    .venv\\Scripts\\python.exe backtest_overnight_8yr.py
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd
import duckdb

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from backtest_continuity import _spearman, _boot_ci

DCM = r"d:/Python Projects/Daily_Cash_Market/data/market_data.duckdb"
INDICES = ["Nifty 50", "Nifty Bank", "Nifty Financial Services", "Nifty Midcap 50"]
CLR_STRONG = 0.66
COST_BPS = 3.0            # overnight index-futures round-trip (small vs the gap)
SEED = 7


def _load(idx: str) -> pd.DataFrame:
    con = duckdb.connect(DCM, read_only=True)
    df = con.execute(
        "select trade_date, open_val o, high_val h, low_val l, close_val c "
        "from index_data where index_name = ? order by trade_date", [idx]).fetchdf()
    con.close()
    df = df.dropna().reset_index(drop=True)
    df["clr"] = ((df.c - df.l) / (df.h - df.l).replace(0, np.nan)).clip(0, 1)
    df["overnight"] = (df.o.shift(-1) / df.c - 1.0) * 100.0        # next open vs today close, %
    df["sma200"] = df.c.rolling(200).mean()
    df["regime"] = np.where(df.c >= df.sma200, "bull", "bear")
    df["year"] = pd.to_datetime(df.trade_date).dt.year
    return df.dropna(subset=["clr", "overnight"]).reset_index(drop=True)


def main():
    rng = np.random.default_rng(SEED)
    print("=" * 84)
    print("  OVERNIGHT CLOSE-STRENGTH — 8.5yr regime stress-test (clr>=0.66 -> long, exit open)")
    print("=" * 84)
    for idx in INDICES:
        try:
            df = _load(idx)
        except Exception as e:
            print(f"\n  {idx}: load error {e}"); continue
        if len(df) < 300:
            print(f"\n  {idx}: thin ({len(df)})"); continue
        span = f"{df.trade_date.min()}..{df.trade_date.max()}"
        print(f"\n  {idx}  n={len(df)}  {span}")
        # 1) IC(clr, overnight) overall + per regime
        ic, lo, hi = _boot_ci(_spearman, df.clr.to_numpy(float), df.overnight.to_numpy(float),
                              reps=4000, rng=rng, groups=df.year.to_numpy())
        v = "EDGE" if lo > 0 else ("anti" if hi < 0 else "—")
        print(f"    IC(clr, overnight)  {ic:+.3f} [{lo:+.3f},{hi:+.3f}] {v}")
        for reg in ("bull", "bear"):
            s = df[df.regime == reg]
            if len(s) < 100 or s.year.nunique() < 2:
                print(f"      {reg:<4} n={len(s)} thin"); continue
            r, l2, h2 = _boot_ci(_spearman, s.clr.to_numpy(float), s.overnight.to_numpy(float),
                                 reps=4000, rng=rng, groups=s.year.to_numpy())
            rv = "EDGE" if l2 > 0 else ("anti" if h2 < 0 else "—")
            print(f"      {reg:<4} IC {r:+.3f} [{l2:+.3f},{h2:+.3f}] {rv}  (n={len(s)})")
        # 2) strong-close bucket net overnight vs all
        strong = df[df.clr >= CLR_STRONG]
        allnet = df.overnight.mean() - COST_BPS / 100.0
        for lab, s in (("ALL days", df), ("strong-close", strong)):
            if len(s) < 50:
                continue
            net = s.overnight.to_numpy(float) - COST_BPS / 100.0
            mn, l3, h3 = _boot_ci(lambda a: a.mean(), net, reps=4000, rng=rng,
                                  groups=s.year.to_numpy())
            ev = "EDGE" if l3 > 0 else ("bleed" if h3 < 0 else "—")
            print(f"    {lab:<12} overnight net {mn:+.3f}% [{l3:+.3f},{h3:+.3f}] {ev}  "
                  f"win {100*(s.overnight>0).mean():.0f}%  n={len(s)}")
        # 3) strong-close net PER YEAR (regime robustness the eye can see)
        print("    strong-close net by year:", end=" ")
        for y, g in strong.groupby("year"):
            if len(g) >= 8:
                print(f"{y}:{g.overnight.mean()-COST_BPS/100:+.2f}", end="  ")
        print()
    print("\n" + "=" * 84)
    print("READ: EDGE only if the strong-close overnight net CI clears 0 AND holds in BOTH")
    print("regimes / across years. If it's positive in bull but flips negative in bear (2018-20,")
    print("2022) it's a REGIME artifact -> gate it. Overnight cost is tiny, so this is the one")
    print("horizon where a real directional edge can survive (unlike intraday).")


if __name__ == "__main__":
    main()
