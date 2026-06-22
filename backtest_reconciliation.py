"""
backtest_reconciliation.py  —  does the full per-strike overnight-reconciliation
engine (overnight_reconciliation.analyze_reconciliation) have intraday edge?

Same falsification discipline as backtest_continuity.py, but the signal is the
richer one: full positioned-strike set, overnight role from EOD per-strike OI,
today's 4-quadrant reconciliation, regime-shift gate. Lookahead-free:
  * baseline = DCM fno_bhavcopy per-strike OI strictly BEFORE the replay day
  * intraday oich + forward spot from the per-day duckdb (full-session ticks)

  .venv\\Scripts\\python.exe backtest_reconciliation.py
"""
from __future__ import annotations

import argparse
import datetime
import sys

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from core.constants import IST, INDEX_SYMBOLS, NSE_NAME
import overnight_reconciliation as orc
from eod_oi_range import DCM_DB
# reuse the validated harness pieces
from backtest_continuity import (
    DuckFeed, _captured_days, _spearman, _boot_ci, _walls_tables, _prior_walls,
    SAMPLE_TIMES, HORIZONS, EOD_CUT, SESSION_END, GAP_THRESH, BAND,
)

SEED = 7


def collect(days: list[str], con, walls: dict) -> pd.DataFrame:
    rows = []
    for date in days:
        d0 = datetime.date.fromisoformat(date)
        prior = d0 - datetime.timedelta(days=1)        # strictly before -> no lookahead
        feed = DuckFeed(date)
        try:
            for sym in INDEX_SYMBOLS:
                baseline = orc.eod_baseline(con, NSE_NAME[sym], as_of_date=prior)
                if not baseline:
                    continue
                wt = walls.get(sym)
                _, prior_close = _prior_walls(wt, d0) if wt is not None else (None, None)
                feed.set_time(datetime.datetime.combine(d0, datetime.time(9, 16), tzinfo=IST))
                open_spot = feed.spot(sym)
                gap = (open_spot / prior_close - 1.0) if (open_spot and prior_close) else 0.0

                for hhmm in SAMPLE_TIMES:
                    hh, mm = map(int, hhmm.split(":"))
                    t = datetime.datetime.combine(d0, datetime.time(hh, mm), tzinfo=IST)
                    feed.set_time(t)
                    spot = feed.spot(sym)
                    sm = feed.strike_map(sym)
                    if not spot or not sm:
                        continue
                    anchor = feed.pa_anchor(sym)   # session open (volume-free trend ref)
                    r = orc.analyze_reconciliation(sm, spot, baseline, gap=gap, vwap=anchor)
                    if not r.has_data:
                        continue
                    fwd = {}
                    for h in HORIZONS:
                        tt = t + datetime.timedelta(minutes=h)
                        if tt.time() > SESSION_END:
                            fwd[h] = np.nan; continue
                        feed.set_time(tt)
                        s2 = feed.spot(sym)
                        fwd[h] = (s2 / spot - 1.0) if (s2 and s2 != spot) else np.nan
                    feed.set_time(datetime.datetime.combine(d0, EOD_CUT, tzinfo=IST))
                    s_eod = feed.spot(sym)
                    fwd_eod = (s_eod / spot - 1.0) if (s_eod and s_eod != spot) else np.nan
                    feed.set_time(t)

                    rows.append({
                        "date": date, "sym": sym, "t": hhmm, "gap": gap,
                        "score": r.score, "verdict": r.verdict,
                        "regime_shift": r.regime_shift, "n_strikes": len(r.rows),
                        "pa_dir": r.pa_dir, "pa_confirmed": r.pa_confirmed,
                        "fwd30": fwd.get(30, np.nan), "fwd60": fwd.get(60, np.nan),
                        "fwd_eod": fwd_eod,
                    })
        finally:
            feed.close()
    return pd.DataFrame(rows)


def report(df: pd.DataFrame, reps: int, rng) -> None:
    print("\nOVERNIGHT RECONCILIATION — intraday edge test (lookahead-free)")
    print("=" * 78)
    if df.empty:
        print("  no rows."); return
    days = sorted(df.date.unique())
    print(f"  rows={len(df)}  days={len(days)} ({days[0]}..{days[-1]})  "
          f"indices={df.sym.nunique()}  median strikes/row={int(df.n_strikes.median())}  "
          f"regime-shift rows={int(df.regime_shift.sum())}")
    print(f"  actionable band |score|>={BAND}")

    subsets = [("ALL", df),
               ("PA-CONFIRMED (score agrees w/ VWAP)", df[df.pa_confirmed]),
               ("PA-DIVERGENT (score vs VWAP)", df[(df.score.abs() >= BAND) & (~df.pa_confirmed) & (df.pa_dir != 0)]),
               (f"GAP (|gap|>{GAP_THRESH*100:.2f}%)", df[df.gap.abs() > GAP_THRESH]),
               ("REGIME-SHIFT", df[df.regime_shift])]
    for label, sub in subsets:
        print(f"\n── {label}  (n={len(sub)}) " + "-" * 36)
        if len(sub) < 5:
            print("   too few rows."); continue
        for hcol in ["fwd30", "fwd60", "fwd_eod"]:
            d = sub.dropna(subset=[hcol])
            if len(d) < 5:
                print(f"   {hcol:7}: n<5"); continue
            score = d["score"].to_numpy(float); ret = d[hcol].to_numpy(float)
            ic, iclo, ichi = _boot_ci(_spearman, score, ret, reps=reps, rng=rng,
                                      groups=d["date"].to_numpy())
            act = d[d.score.abs() >= BAND]
            if len(act) >= 5:
                call = np.sign(act["score"].to_numpy())
                hit = (call == np.sign(act[hcol].to_numpy())).astype(float)
                hr, hlo, hhi = _boot_ci(lambda a: a.mean(), hit, reps=reps, rng=rng,
                                        groups=act["date"].to_numpy())
                hit_s = f"hit {100*hr:4.1f}% [{100*hlo:4.1f},{100*hhi:4.1f}] n={len(act)}"
                hv = "EDGE" if hlo > 0.5 else ("anti" if hhi < 0.5 else "—")
            else:
                hit_s, hv = "hit n<5", "—"
            icv = "EDGE" if iclo > 0 else ("anti" if ichi < 0 else "—")
            print(f"   {hcol:7}: IC {ic:+.3f} [{iclo:+.3f},{ichi:+.3f}] {icv:4} | {hit_s} {hv}")

    print("\n" + "=" * 78)
    print("READ: 'EDGE' = DAY-BLOCK bootstrap 95% CI excludes null. Days (not intraday")
    print("rows) are resampled, so within-day correlation no longer fabricates")
    print(f"significance — but with only {len(days)} captured days the CI is necessarily")
    print("WIDE, so an 'EDGE' on a small bucket is still fragile. Trust nothing until")
    print("the CI holds over many wide-strike days. Old days are ±15-strike too.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3000)
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    days = _captured_days()
    if not days or not DCM_DB.exists():
        print("Missing captured days or DCM DuckDB."); return
    import duckdb
    con = duckdb.connect(str(DCM_DB), read_only=True)
    walls = _walls_tables(con)
    df = collect(days, con, walls)
    report(df, args.reps, rng)


if __name__ == "__main__":
    main()
