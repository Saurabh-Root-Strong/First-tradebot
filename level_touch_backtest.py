"""
level_touch_backtest.py — do OI walls predict where price TOUCHES intraday?

Your idea: from the first-20-min OI positioning, the nearest call wall (highest
call OI = ceiling) and put wall (highest put OI = floor) mark levels price tends
to reach within the next 30-60 min. This falsifies that against a fair null.

For each sample instant t (>=09:35) it asks: did spot touch the call wall (reach
>= call_wall) / the put wall (<= put_wall) within H minutes? Then compares the
observed touch-rate to a REFLECTION-PRINCIPLE null — the probability a driftless
random walk with the day's realised vol reaches a level at the SAME distance:

    P(touch within H) = 2 * (1 - Phi(distance / sigma_H))      (one-sided barrier)

  sigma_H = realised 1-min return stdev * sqrt(H), in points.

Interpretation of mean(observed - null), day-block bootstrap CI:
    > 0  walls are MAGNETS — price reaches them more than vol alone implies
    < 0  walls are BARRIERS/PINS — price stalls before them (max-pain pin)
    ~ 0  walls behave like any level at that distance — no special pull

Lookahead-free: walls/spot from oi_snapshots at t; the price path from the
independent tick mirror over (t, t+H].

    .venv\\Scripts\\python.exe level_touch_backtest.py
"""
from __future__ import annotations

import argparse
import datetime
import math
import sys

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from core.constants import IST, INDEX_SYMBOLS, LIVE_DIR
from core.mirror_io import read_mirror as _read
from backtest_continuity import _boot_ci

SEED = 7
SAMPLE_TIMES = ["09:40", "10:00", "10:30", "11:00"]
HORIZONS = [30, 60]
_SIG_FLOOR = 0.02     # min 1-min sigma % (guards thin early session)


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _reflect_p(dist_pts: float, sigma_pts: float) -> float:
    """Driftless first-passage: P(reach a level dist away within the window)."""
    if sigma_pts <= 0 or dist_pts <= 0:
        return np.nan
    return min(1.0, 2.0 * (1.0 - _phi(dist_pts / sigma_pts)))


def _sigma1_pct(ticks: pd.DataFrame, upto: pd.Timestamp) -> float | None:
    s = ticks[ticks["ts"] <= upto].set_index("ts")["ltp"].resample("1min").last().dropna()
    if len(s) < 12:
        return None
    r = s.pct_change().dropna() * 100.0
    return float(r.std()) if len(r) >= 8 and r.std() > 0 else None


def _captured_days() -> list[str]:
    return sorted({p.name.split("_")[0] for p in LIVE_DIR.glob("*_oi_snapshots.parquet")})


def collect(days: list[str]) -> pd.DataFrame:
    rows = []
    for date in days:
        d0 = datetime.date.fromisoformat(date)
        for sym in INDEX_SYMBOLS:
            ticks = _read("ticks", date, None, sym)
            oi = _read("oi_snapshots", date, None, sym)
            if ticks is None or oi is None or len(ticks) < 20:
                continue
            for hhmm in SAMPLE_TIMES:
                hh, mm = map(int, hhmm.split(":"))
                t = pd.Timestamp(datetime.datetime.combine(d0, datetime.time(hh, mm), tzinfo=IST))
                o = oi[oi["ts"] <= t]
                if not len(o):
                    continue
                last = o.iloc[-1]
                spot = float(last["spot"] or 0)
                cw, pw = float(last["call_wall"] or 0), float(last["put_wall"] or 0)
                if not spot:
                    continue
                sig1 = _sigma1_pct(ticks, t)
                if not sig1:
                    continue
                for h in HORIZONS:
                    sig_pts = max(sig1, _SIG_FLOOR) * (h ** 0.5) / 100.0 * spot
                    path = ticks[(ticks["ts"] > t) &
                                 (ticks["ts"] <= t + pd.Timedelta(minutes=h))]
                    if len(path) < 3:
                        continue
                    pmax, pmin = float(path["ltp"].max()), float(path["ltp"].min())
                    # call wall above spot -> magnet/barrier on the UP side
                    if cw > spot:
                        d = cw - spot
                        rows.append({"date": date, "sym": sym, "t": hhmm, "h": h,
                                     "side": "call_wall", "touched": float(pmax >= cw),
                                     "null": _reflect_p(d, sig_pts), "dist_sig": d / sig_pts})
                    if 0 < pw < spot:
                        d = spot - pw
                        rows.append({"date": date, "sym": sym, "t": hhmm, "h": h,
                                     "side": "put_wall", "touched": float(pmin <= pw),
                                     "null": _reflect_p(d, sig_pts), "dist_sig": d / sig_pts})
    return pd.DataFrame(rows)


def report(df: pd.DataFrame, reps: int, rng) -> None:
    print("\nOI WALL LEVEL-TOUCH — magnet vs barrier test (vs random-walk null)")
    print("=" * 76)
    if df.empty:
        print("  no rows — need captured days with oi_snapshots walls."); return
    df = df.dropna(subset=["null"])
    days = sorted(df.date.unique())
    print(f"  rows={len(df)}  days={len(days)} ({days[0]}..{days[-1]})  "
          f"samples/day~{len(SAMPLE_TIMES)}  median dist={df.dist_sig.median():.2f}σ")
    for side in ["call_wall", "put_wall"]:
        for h in HORIZONS:
            sub = df[(df.side == side) & (df.h == h)]
            if len(sub) < 5 or sub["date"].nunique() < 2:
                print(f"\n  {side} @{h}m: n<5"); continue
            obs = sub["touched"].to_numpy(float)
            null = sub["null"].to_numpy(float)
            diff = obs - null
            mo, _, _   = _boot_ci(lambda a: a.mean(), obs,  reps=reps, rng=rng, groups=sub["date"].to_numpy())
            mn         = float(null.mean())
            md, lo, hi = _boot_ci(lambda a: a.mean(), diff, reps=reps, rng=rng, groups=sub["date"].to_numpy())
            verdict = ("MAGNET" if lo > 0 else "BARRIER/PIN" if hi < 0 else "— (like any level)")
            print(f"\n  {side} @{h}m  (n={len(sub)})")
            print(f"    touched {100*mo:4.1f}%   null(vol) {100*mn:4.1f}%   "
                  f"excess {100*md:+4.1f}% [{100*lo:+.1f},{100*hi:+.1f}]  -> {verdict}")
    print("\n" + "=" * 76)
    print("READ: excess = observed touch-rate minus the random-walk touch-rate at the")
    print("same distance+vol, day-block bootstrap. MAGNET (>0) = a tradeable target;")
    print("BARRIER/PIN (<0) = price stalls before it; '—' = the wall adds nothing over")
    print("a plain vol band. Few days -> wide CI; re-run as full sessions accumulate.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3000)
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    days = _captured_days()
    if not days:
        print("No captured mirror days in", LIVE_DIR); return
    df = collect(days)
    report(df, args.reps, rng)


if __name__ == "__main__":
    main()
