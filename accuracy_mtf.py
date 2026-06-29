"""
accuracy_mtf.py — how ACCURATE is the multi-session MTF confirmation, rigorously?

Reads the cached harvest from backtest_mtf.py (MTF_CACHE) and answers one question:
when the higher-TF CONFIRMS a fast trigger, how often is the trade directionally
right — and is that beating the honest baselines and stable, or one-day luck?

Accuracy = P(signed forward return > 0). Reported with BOTH a Wilson 95% CI (per
observation) and a DAY-BLOCK bootstrap CI (resamples whole days — the honest one,
since intraday checkpoints are correlated). Edge is vs two baselines: 50% (coin)
and the raw ALL-trigger hit rate (does confirmation ADD over just taking it).
Stability = leave-one-day-out range. Plus per-index, and the conflict-veto value
(how often a CONFLICT trade would have lost = correctly avoided).

    set MTF_CACHE=...mtf_df.parquet  &&  .venv\\Scripts\\python.exe accuracy_mtf.py
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SEED = 7
HORIZONS = (15, 30)
CACHE = os.environ.get("MTF_CACHE", "")


def _wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * p, 100 * (c - h), 100 * (c + h)


def _dayblock_hit(hits, days, rng, reps=2000):
    hits = np.asarray(hits, float); days = np.asarray(days)
    keep = ~np.isnan(hits); hits, days = hits[keep], days[keep]
    if len(hits) < 5:
        return float("nan"), float("nan"), float("nan"), 0
    u = np.unique(days); idx = {d: np.where(days == d)[0] for d in u}
    bs = [hits[np.concatenate([idx[d] for d in rng.choice(u, len(u), replace=True)])].mean()
          for _ in range(reps)]
    return 100 * hits.mean(), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5), len(hits)


def _loo(hits, days):
    """leave-one-day-out hit-rate range — exposes one-day dependence."""
    hits = np.asarray(hits, float); days = np.asarray(days)
    keep = ~np.isnan(hits); hits, days = hits[keep], days[keep]
    out = []
    for d in np.unique(days):
        m = days != d
        if m.sum() >= 5:
            out.append(100 * hits[m].mean())
    return (min(out), max(out)) if out else (float("nan"), float("nan"))


def hit(df, H):
    s = df[f"sret{H}"]
    return (s > 0).astype(float).where(s.notna())


def main():
    rng = np.random.default_rng(SEED)
    if not (CACHE and os.path.exists(CACHE)):
        print("set MTF_CACHE to the cached parquet (run backtest_mtf.py with MTF_CACHE first)"); return
    df = pd.read_parquet(CACHE)
    df["hit15"] = hit(df, 15); df["hit30"] = hit(df, 30)
    days = sorted(df.date.unique())
    print(f"DEEP ACCURACY — MTF confirmation   rows={len(df)}  days={len(days)} ({days[0]}..{days[-1]})")
    print("=" * 88)

    def bucket(sub, label, H):
        h = sub[f"hit{H}"]
        n = int(h.notna().sum())
        if n == 0:
            print(f"   {label:9s} {H}m  n=0"); return
        wm, wlo, whi = _wilson(int(np.nansum(h)), n)
        bm, blo, bhi, _ = _dayblock_hit(h, sub.date.to_numpy(), rng)
        llo, lhi = _loo(h, sub.date.to_numpy())
        flag = "+" if blo > 50 else ("-" if bhi < 50 else "0")
        print(f"   {label:9s} {H}m  hit {wm:4.0f}%  Wilson[{wlo:4.0f},{whi:4.0f}]  "
              f"day-block[{blo:4.0f},{bhi:4.0f}][{flag}]  LOO[{llo:4.0f},{lhi:4.0f}]  n={n}")

    # headline pairs (5->30 is noise, keep for contrast)
    for pair in ["10->30", "15->60", "5->30"]:
        p = df[df.pair == pair]
        print(f"\n PAIR {pair}")
        for H in HORIZONS:
            bucket(p, "ALL", H)
            bucket(p[p.confirmed & ~p.conflict], "CONFIRM", H)
            bucket(p[p.conflict], "CONFLICT", H)

    print("\n " + "-" * 70)
    print(" POOLED — best two pairs (10->30, 15->60)  [the defensible headline]")
    best = df[df.pair.isin(["10->30", "15->60"])]
    for H in HORIZONS:
        bucket(best, "ALL", H)
        bucket(best[best.confirmed & ~best.conflict], "CONFIRM", H)
        bucket(best[best.conflict], "CONFLICT", H)

    print("\n " + "-" * 70)
    print(" CONFIRM hit rate per index (best two pairs, 30m) — does it hold everywhere?")
    bb = best[best.confirmed & ~best.conflict]
    for sym in sorted(bb.sym.unique()):
        s = bb[bb.sym == sym]; h = s["hit30"]; n = int(h.notna().sum())
        wm, wlo, whi = _wilson(int(np.nansum(h)), n)
        print(f"   {sym.split(':')[1]:18s} hit {wm:4.0f}%  Wilson[{wlo:4.0f},{whi:4.0f}]  n={n}")

    print("\n " + "-" * 70)
    print(" CONFLICT-VETO value (best two pairs): how often a CONFLICT trade LOSES")
    cf = best[best.conflict]
    for H in HORIZONS:
        h = cf[f"hit{H}"]; n = int(h.notna().sum())
        loss = 100 * (1 - np.nansum(h) / n) if n else float("nan")
        print(f"   {H}m  CONFLICT loses {loss:4.0f}% of the time  (n={n}) -> vetoing avoids that")

    print("\n" + "=" * 88)
    print("READ: 'accurate' = day-block CI clears 50 AND CONFIRM>ALL AND LOO doesn't")
    print("dip below 50. day-block (not Wilson) is the honest CI. If day-block straddles")
    print("50, the hit rate is not yet distinguishable from a coin at this sample.")


if __name__ == "__main__":
    main()
