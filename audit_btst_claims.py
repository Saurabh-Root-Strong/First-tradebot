"""
audit_btst_claims.py — try to KILL the two claims I just made, before anyone trades them.

Last hour I overturned two pieces of standing guidance:
  A. "The edge is BIGGER below the 200d mean / in high vol -- the gap tail is the PAY."
  B. "Dropping MIDCAP is wrong -- it is a contributor, not the drag."
Both are surprising. Surprising results are usually bugs. A claim I have not tried to falsify
is not a finding, it is a hope. Four ways these could be lying:

  1. FAT RIGHT TAIL — a mean dragged up by a handful of monster nights. Check the MEDIAN and a
     trimmed mean. If the median is flat, the "edge" is a few lottery tickets.
  2. CLUSTERING — "below the 200d mean" nights arrive in RUNS (a bear phase is one event, not
     196 independent ones). A plain t-stat assumes independence and will overstate. Use a
     BLOCK bootstrap that resamples contiguous chunks, which preserves the clustering.
  3. OUT OF SAMPLE — does the claim hold in BOTH halves, or only the recent regime? This is
     what killed the STBT candidate; it must be applied to my own findings too.
  4. COST — a flat 3bps round-trip is applied to NIFTY and to MIDCPNIFTY alike. MIDCAP futures
     are far less liquid. If MIDCAP's true cost is 6-10bps, is it still a contributor?

    .venv\\Scripts\\python.exe audit_btst_claims.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import btst_signal as bs
from backtest_btst_tail import build

RNG = np.random.default_rng(7)


def block_boot(v, n=4000, block=20):
    """Bootstrap the mean with CONTIGUOUS blocks, so regime clustering is preserved.
    An i.i.d. bootstrap (or a plain t-stat) pretends 196 bear nights are 196 independent
    draws. They are not -- they are a handful of bear PHASES. This is the honest CI."""
    v = np.asarray(v, float)
    k = max(1, len(v) // block)
    out = np.empty(n)
    for i in range(n):
        starts = RNG.integers(0, max(1, len(v) - block), size=k)
        out[i] = np.concatenate([v[s:s + block] for s in starts]).mean()
    return np.percentile(out, [2.5, 50, 97.5])


def line(nm, v):
    v = np.asarray(v, float)
    if len(v) < 40:
        print(f"  {nm:32s} n={len(v):>4d}   too few")
        return
    t = v.mean() / (v.std() / np.sqrt(len(v)))
    trim = v[(v > np.percentile(v, 5)) & (v < np.percentile(v, 95))].mean()
    lo, mid, hi = block_boot(v)
    h = len(v) // 2
    a, b = v[:h], v[h:]
    ok = "SURVIVES" if lo > 0 and np.median(v) > 0 and a.mean() > 0 and b.mean() > 0 else "FRAGILE"
    print(f"  {nm:32s} n={len(v):>4d} mean {v.mean():>+6.1f} med {np.median(v):>+6.1f} "
          f"trim {trim:>+6.1f} t {t:>+5.2f}  block-CI [{lo:>+6.1f},{hi:>+6.1f}]  "
          f"H1 {a.mean():>+6.1f} H2 {b.mean():>+6.1f}  {ok}")


def main():
    df = build().sort_values("date").reset_index(drop=True)
    v = df["long_bps"].to_numpy(float)

    print("=" * 118)
    print("CLAIM A — 'the edge is BIGGER in bear / high vol; the tail is the pay'")
    print("   killers: fat right tail (median?), clustering (block CI?), OOS (both halves?)")
    print("=" * 118)
    line("ALL signal nights", v)
    line("above 200d mean", df.loc[df.above200 == True, "long_bps"])
    line("BELOW 200d mean", df.loc[df.above200 == False, "long_bps"])
    line("calm vol (below median)", df.loc[df.vol20 < df.vol20.median(), "long_bps"])
    line("HIGH vol (above median)", df.loc[df.vol20 >= df.vol20.median(), "long_bps"])
    print("\n  a claim is only real if: block-CI excludes 0, the MEDIAN is positive, and BOTH")
    print("  halves are positive. A high mean with a flat median = a few lottery nights.\n")

    print("=" * 118)
    print("CLAIM B — 'MIDCAP is a contributor, not the drag' (I told the user to DROP it)")
    print("=" * 118)
    for s in bs.SYMS:
        line(s, df.loc[df.sym == s, "long_bps"])
    print()

    print("=" * 118)
    print("THE COST ASSUMPTION — a flat 3bps is applied to NIFTY and MIDCAP alike.")
    print("   MIDCAP index futures are far less liquid. What if its true round-trip is higher?")
    print("=" * 118)
    print(f"  {'extra cost on MIDCAP':>22s} {'MIDCAP mean':>12s} {'ALL mean':>10s} "
          f"{'ALL Sharpe':>11s}  {'drop MIDCAP?':>13s}")
    print("  " + "-" * 76)
    for extra in (0, 3, 5, 8, 12):
        d2 = df.copy()
        m = d2["sym"] == "MIDCPNIFTY"
        d2.loc[m, "long_bps"] -= extra
        mid = d2.loc[m, "long_bps"].to_numpy(float)
        allv = d2["long_bps"].to_numpy(float)
        no_mid = d2.loc[~m, "long_bps"].to_numpy(float)
        sh = allv.mean() / allv.std() * np.sqrt(252)
        better = "YES — drop it" if no_mid.mean() > allv.mean() else "no — keep it"
        print(f"  {f'+{extra} bps':>22s} {mid.mean():>+12.1f} {allv.mean():>+10.1f} "
              f"{sh:>+11.2f}  {better:>13s}")
    print("  " + "-" * 76)
    print("  (3bps is the NIFTY-grade assumption. If MIDCAP really costs 8-12bps more to")
    print("   round-trip, the 'MIDCAP is a contributor' claim needs re-checking.)")

    print("\n" + "=" * 118)
    print("THE DEEPEST CHECK — is the RULE itself real, or did clr just ride the drift?")
    print("=" * 118)
    # every night vs SIGNAL nights: does clr add anything over simply being long every night?
    from backtest_stbt import overnight_table, COST
    allx = overnight_table()
    allx["long_bps"] = -(allx["short_bps"] + COST) - COST
    base = allx["long_bps"].to_numpy(float)
    sig = allx.loc[allx.clr >= bs.CLR_TH, "long_bps"].to_numpy(float)
    non = allx.loc[allx.clr < bs.CLR_TH, "long_bps"].to_numpy(float)
    line("LONG every night (no signal)", base)
    line("LONG only on clr>=0.66", sig)
    line("LONG only on clr<0.66", non)
    diff = sig.mean() - non.mean()
    se = np.sqrt(sig.var() / len(sig) + non.var() / len(non))
    print(f"\n  clr's INCREMENTAL edge over a coin-flip long: {diff:+.1f} bps  (t={diff/se:+.2f})")
    print("  If this t is small, clr is decoration and the 'edge' is just the overnight drift.")


if __name__ == "__main__":
    main()
