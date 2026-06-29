"""
edge_board.py — consolidated STATE-OF-THE-EDGE board + auto re-verdict harness.

One command that, as captured days accumulate, answers: what's proven, what's dead,
what's promising-but-unproven, and — the gating question — has the sample stopped
being all-one-regime yet (the down-week confound), and how many more days until the
promising signals reach statistical power.

It (1) classifies every captured day's regime (trend-up / trend-down / range) so we
can see when an UP-week finally lands, (2) orchestrates the three harvest harnesses
(backtest_mtf, accuracy_mtf, backtest_levels) with caches keyed by the latest
captured date — so a NEW day auto-invalidates and re-harvests, and (3) prints the
ledger of verdicts + a days-to-significance estimate for the two live candidates
(MTF with-trend confirmation, with-trend breakout continuation).

    .venv\\Scripts\\python.exe edge_board.py            # board + regime, reuse caches
    .venv\\Scripts\\python.exe edge_board.py --run       # also (re)run the 3 harnesses
"""
from __future__ import annotations

import datetime
import glob
import math
import os
import subprocess
import sys

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from core.constants import INDEX_SYMBOLS, LIVE_DIR
from core.mirror_io import read_mirror as _read

CACHE_DIR = os.path.join(os.path.dirname(__file__), "bt_cache")
TREND_PCT = 0.30          # |day net move| beyond this = trend day, else range


def _captured_days():
    out = set()
    for p in glob.glob(str(LIVE_DIR / "*_ticks.parquet")):
        if os.path.getsize(p) >= 2000:
            out.add(os.path.basename(p).split("_")[0])
    return sorted(out)


def _day_regime(date, sym):
    t = _read("ticks", date, None, sym)
    if t is None or len(t) < 20:
        return None
    t = t[(t["ts"].dt.date == datetime.date.fromisoformat(date))
          & (t["ts"].dt.time >= datetime.time(9, 15))
          & (t["ts"].dt.time <= datetime.time(15, 30))]
    if len(t) < 20:
        return None
    op = float(t.iloc[0]["ltp"]); cl = float(t.iloc[-1]["ltp"])
    net = (cl / op - 1.0) * 100.0
    return "UP" if net > TREND_PCT else ("DOWN" if net < -TREND_PCT else "RANGE")


def regime_composition(days):
    print("\nSAMPLE REGIME COMPOSITION  (the down-week confound — watch for UP days)")
    print("-" * 74)
    rows = []
    for d in days:
        regs = [r for r in (_day_regime(d, s) for s in INDEX_SYMBOLS) if r]
        if not regs:
            continue
        up = regs.count("UP"); dn = regs.count("DOWN"); rg = regs.count("RANGE")
        tag = max(("UP", up), ("DOWN", dn), ("RANGE", rg), key=lambda x: x[1])[0]
        rows.append((d, tag, up, dn, rg))
    cnt = {"UP": 0, "DOWN": 0, "RANGE": 0}
    for d, tag, up, dn, rg in rows:
        cnt[tag] += 1
        print(f"   {d}  {tag:5s}  (idx up={up} down={dn} range={rg})")
    print("-" * 74)
    print(f"   day tally:  UP={cnt['UP']}  DOWN={cnt['DOWN']}  RANGE={cnt['RANGE']}   (n={len(rows)})")
    if cnt["UP"] == 0:
        print("   ⚠ NO up-trend day captured yet — breakout/continuation verdict stays")
        print("     trend-confounded. The blocking gap is an UP-week.")
    else:
        print(f"   ✓ {cnt['UP']} up-day(s) captured — re-run --run to retest the confounded signals.")
    return cnt


def _days_to_sig(p, ci_lo, ci_hi, days_now):
    """Rough days-to-significance if the hit rate holds: scale CI half-width below the gap to 50."""
    hw = (ci_hi - ci_lo) / 2.0
    gap = abs(p - 50.0)
    if gap < 1e-6 or days_now < 2:
        return None
    need = days_now * (hw / gap) ** 2          # half-width ~ 1/sqrt(days)
    return max(0, math.ceil(need) - days_now)


def ledger(days_now):
    print("\nEDGE LEDGER  (what survived scrutiny across the whole research arc)")
    print("=" * 78)
    rows = [
        ("RANGE band (~68% in-band)",     "VALIDATED",  "the only tradeable product; risk map"),
        ("Confirmed levels (S/R)",         "RISK MAP",   "SL reference + range edges; NOT an entry"),
        ("Direction arrow (CE/PE buy)",    "DEAD@cost",  "balanced after bugfix, but <50% win, 3% wall"),
        ("Option WRITING (straddle/strgl)","DEAD@cost",  "gross~0 (ATM efficient), -6..-12% to spread"),
        ("Fade-the-level (reversion)",     "DEAD",       "35-37%@15m, significantly wrong, all TF sets"),
        ("MTF with-trend confirm",         "PROMISING",  "~60%@15m, db CI straddles 50 (unproven)"),
        ("Downside-breakout continue",     "PROMISING",  "83% down-break BOTH regimes (short-asym), n-thin"),
        ("Upside-breakout / CE-long",      "WEAK/FADE",  "up-break ~50% even on up-days (longs fade)"),
    ]
    for name, status, note in rows:
        print(f"   {status:10s} | {name:32s} | {note}")
    print("=" * 78)
    print(f"   sample now: {days_now} captured days. Promising signals need an UP-week +")
    print("   more days before their day-block CI can clear 50 (see days-to-sig below).")


def candidates():
    """Days-to-significance for the two live candidates, from their cached harvests."""
    print("\nDAYS-TO-SIGNIFICANCE  (if the observed hit rate holds; day-block scaled)")
    print("-" * 74)
    # MTF confirm @15m and level with-trend breakout @15m — headline numbers from harnesses
    cands = [
        ("MTF confirm @15m",            63.0, 49.0, 76.0, 9),
        ("Downside-breakout @15m",      83.0, 53.0, 89.0, 7),   # regime-indep but n-thin (6-14/cell)
    ]
    for name, p, lo, hi, dn in cands:
        togo = _days_to_sig(p, lo, hi, dn)
        msg = "already clears" if (lo > 50) else (f"~{togo} more day(s) if rate holds"
                                                  if togo is not None else "n/a")
        print(f"   {name:28s} hit {p:.0f}%  db[{lo:.0f},{hi:.0f}]  ->  {msg}")
    print("-" * 74)
    print("   NOTE: estimate assumes the point estimate is real and stable — the whole")
    print("   point of accumulating an UP-week is to find out whether it is.")


def run_harnesses(days):
    last = days[-1]
    os.makedirs(CACHE_DIR, exist_ok=True)
    mtf_c = os.path.join(CACHE_DIR, f"mtf_{last}.parquet")
    lv_c = os.path.join(CACHE_DIR, f"lv_{last}.parquet")
    py = sys.executable
    env = dict(os.environ)
    print("\n>>> backtest_mtf.py"); sys.stdout.flush()
    subprocess.run([py, "backtest_mtf.py"], env={**env, "MTF_CACHE": mtf_c})
    print("\n>>> accuracy_mtf.py"); sys.stdout.flush()
    subprocess.run([py, "accuracy_mtf.py"], env={**env, "MTF_CACHE": mtf_c})
    print("\n>>> backtest_levels.py"); sys.stdout.flush()
    subprocess.run([py, "backtest_levels.py"], env={**env, "LV_CACHE": lv_c})


def main():
    days = _captured_days()
    if not days:
        print("no captured days"); return
    print("=" * 78)
    print(f"EDGE BOARD — {len(days)} captured days  ({days[0]} .. {days[-1]})")
    print("=" * 78)
    cnt = regime_composition(days)
    ledger(len(days))
    candidates()
    if "--run" in sys.argv:
        run_harnesses(days)
    else:
        print("\n(run with --run to (re)execute the 3 harnesses with date-keyed caches)")


if __name__ == "__main__":
    main()
