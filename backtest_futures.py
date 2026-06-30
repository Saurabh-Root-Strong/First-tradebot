"""
backtest_futures.py — do the two promising signals pay NET OF FUTURES cost?

Every directional edge this arc died to the 3% OPTION round-trip wall. But that wall
is an option-buying artifact. Index FUTURES cost ~150x less: brokerage (flat, ~0%),
slippage/impact (~1 tick), STT-on-sell (~0.0125%), exchange+stamp (~0.003%) ->
realistic all-in round-trip ~0.02-0.05%. The two signals with a pulse were never
tested there:
  - DOWNSIDE-BREAKOUT short (confirmed-level break down) — 77% hit, ties to the proven
    short-side continuation asymmetry (PE +0.083R, index 2.7:1 down:up).
  - MTF-CONFIRM directional (fast trigger agreeing with multi-session higher-TF trend).

The date-keyed caches from edge_board already hold the SIGNED forward returns, and a
signed return on a directional bet IS the gross futures P&L in % (short a down-break
that continues -> price falls -> positive). So net = gross - futures_cost. We sweep
cost 0.02/0.03/0.05% and report mean net per trade + a day-block bootstrap CI + win
rate. Edge only if mean-net CI clears 0.

Approximation: index % move ~= futures % move intraday (basis ~constant). Fixed-horizon
exit (no SL) -> expectancy is unbiased; a level SL would only tighten the left tail.

    .venv\\Scripts\\python.exe backtest_futures.py
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SEED = 7
CACHE_DIR = os.path.join(os.path.dirname(__file__), "bt_cache")
FUT_COSTS = (0.02, 0.03, 0.05)        # round-trip %, conservative band for liquid index futures
HORIZONS = (15, 30)


def _latest(prefix):
    fs = sorted(glob.glob(os.path.join(CACHE_DIR, f"{prefix}_*.parquet")))
    return fs[-1] if fs else None


def _boot_mean_ci(x, days, rng, reps=2000):
    x = np.asarray(x, float); d = np.asarray(days)
    keep = ~np.isnan(x); x, d = x[keep], d[keep]
    if len(x) < 5:
        return None
    u = np.unique(d); idx = {k: np.where(d == k)[0] for k in u}
    bs = [x[np.concatenate([idx[k] for k in rng.choice(u, len(u), replace=True)])].mean()
          for _ in range(reps)]
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(x), len(u)


def _block(name, gross, days, rng):
    print(f"\n  {name}")
    g = np.asarray(gross, float)
    res = _boot_mean_ci(g, days, rng)
    if res is None:
        print("     n<5 — thin"); return
    gm, glo, ghi, n, nd = res
    wing = 100 * (g[~np.isnan(g)] > 0).mean()
    print(f"     GROSS  mean {gm:+.4f}% [{glo:+.4f},{ghi:+.4f}]  win {wing:3.0f}%  n={n}/{nd}d")
    for c in FUT_COSTS:
        net = g - c
        nm, nlo, nhi, _, _ = _boot_mean_ci(net, days, rng)
        win = 100 * (net[~np.isnan(net)] > 0).mean()
        flag = "+" if nlo > 0 else ("-" if nhi < 0 else "0")
        # rough daily yield: mean-net * (trades/day)
        per_day = nm * (n / nd)
        print(f"     cost {c:.2f}%  net {nm:+.4f}% [{nlo:+.4f},{nhi:+.4f}][{flag}]  "
              f"win {win:3.0f}%  ~{per_day:+.3f}%/day")


def main():
    rng = np.random.default_rng(SEED)
    mtf_c = _latest("mtf"); lv_c = _latest("lv")
    print("FUTURES-COST VERDICT — do the promising signals pay net of futures cost?")
    print("=" * 80)
    print(f"  caches: mtf={os.path.basename(mtf_c) if mtf_c else None}  "
          f"lv={os.path.basename(lv_c) if lv_c else None}")
    print(f"  futures round-trip cost swept: {', '.join(f'{c:.2f}%' for c in FUT_COSTS)}")
    print("  (run edge_board.py --run first to refresh caches for the latest day)")

    if lv_c:
        lv = pd.read_parquet(lv_c)
        dn = lv[(lv.cfg == "any2of4") & lv.is_brk & (lv.brk_dir < 0)]
        up = lv[(lv.cfg == "any2of4") & lv.is_brk & (lv.brk_dir > 0)]
        for H in HORIZONS:
            _block(f"DOWNSIDE-BREAK short @ {H}m  (the candidate)", dn[f"brk{H}"], dn["date"], rng)
        # contrast: up-break long (expected weak/negative — the short asymmetry)
        _block("UP-BREAK long @ 15m  (contrast — expect no edge)", up["brk15"], up["date"], rng)

    if mtf_c:
        m = pd.read_parquet(mtf_c)
        best = m[m.pair.isin(["10->30", "15->60"]) & m.confirmed & ~m.conflict]
        for H in HORIZONS:
            _block(f"MTF-CONFIRM directional @ {H}m", best[f"sret{H}"], best["date"], rng)

    print("\n" + "=" * 80)
    print("READ: a signal is FUTURES-TRADEABLE only if mean-net CI clears 0 at a realistic")
    print("cost (0.03-0.05%). GROSS>0 but net straddling 0 = the edge is real but too small")
    print("to beat even futures cost. Down-break paying while up-break does not = the")
    print("short-side asymmetry survives to a tradeable instrument.")


if __name__ == "__main__":
    main()
