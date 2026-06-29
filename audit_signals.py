"""
audit_signals.py — complete per-component bias + skill audit of the scout signals.

For every captured day/index/checkpoint it recomputes the FOUR scored components
(flow, div, cross, fut) + the tilt, and asks two questions per component:
  1. BIAS  — is its sign symmetric, or structurally stuck one way? (% positive, mean)
  2. SKILL — does it actually rank the forward move? IC(component, fwd_ret), day-block CI.
Also the conditional split on down-bars vs up-bars (where the strike-roll/vega/contango
biases showed up). This is the audit that proves whether the flow roll+vega fix removed
the 95%-CALL bias and whether div/cross/fut carry their own structural lean.

Lookahead-free: components use ts<=t; forward spot is the answer key only.

    .venv\\Scripts\\python.exe audit_signals.py
"""
from __future__ import annotations

import datetime
import sys

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from core.constants import IST, INDEX_SYMBOLS, LIVE_DIR
from core.mirror_io import read_mirror as _read
from backtest_continuity import _spearman, _boot_ci
import footprint_chart as fc
import intraday_scout as scout

SEED = 7
TF = 15
SAMPLE_TIMES = ["09:45", "10:15", "10:45", "11:15", "11:45", "12:15", "12:45",
                "13:15", "13:45", "14:15", "14:45"]
COMPONENTS = ["flow", "div", "cross", "fut", "tilt"]


def _captured_days():
    out = set()
    for p in LIVE_DIR.glob("*_oi_snapshots.parquet"):
        if p.stat().st_size < 1024:
            continue
        try:
            datetime.date.fromisoformat(p.name.split("_")[0]); out.add(p.name.split("_")[0])
        except ValueError:
            continue
    return sorted(out)


def _spot_at(ticks, t):
    s = ticks[ticks["ts"] <= pd.Timestamp(t)]
    return float(s.iloc[-1]["ltp"]) if len(s) else None


def harvest(days):
    rows = []
    for date in days:
        d0 = datetime.date.fromisoformat(date)
        for sym in INDEX_SYMBOLS:
            ticks = _read("ticks", date, None, sym)
            if ticks is None or len(ticks) < 20:
                continue
            ticks = ticks[(ticks["ts"].dt.time >= datetime.time(9, 15))
                          & (ticks["ts"].dt.time <= datetime.time(15, 30))]
            for hhmm in SAMPLE_TIMES:
                hh, mm = map(int, hhmm.split(":"))
                t = datetime.datetime.combine(d0, datetime.time(hh, mm), tzinfo=IST)
                try:
                    ser = fc.build_series(sym, TF, date, t)
                except Exception:
                    continue
                if not ser.get("has_data"):
                    continue
                spot = _spot_at(ticks, t)
                if not spot:
                    continue
                rec = {"date": date, "sym": sym, "t": hhmm, "spot": spot}
                rec["flow"] = scout._flow_signal(ser)[0]
                rec["div"] = scout._divergence_signal(ser)[0]
                rec["cross"] = scout._crossover_signal(ser)[0]
                rec["fut"] = scout._futures_signal(sym, TF, date, t)[0]
                rec["tilt"] = scout._tilt_signal(ser)[0]
                # bar direction at t (last bar move) + forward return
                sp = [x for x in ser.get("spot") if x is not None]
                rec["bar_dir"] = np.sign(sp[-1] - sp[-2]) if len(sp) >= 2 else 0
                for H in (5, 15, 30):
                    s_end = _spot_at(ticks, t + datetime.timedelta(minutes=H))
                    rec[f"ret{H}"] = ((s_end / spot - 1.0) * 100.0
                                     if (s_end and s_end != spot) else np.nan)
                rows.append(rec)
    return pd.DataFrame(rows)


def evaluate(df, reps, rng):
    print("\nSIGNAL COMPONENT AUDIT — bias (sign symmetry) + skill (IC)")
    print("=" * 82)
    days = sorted(df.date.unique())
    print(f"  rows={len(df)}  days={len(days)} ({days[0]}..{days[-1]})  indices={df.sym.nunique()}")

    print("\n  1) BIAS — is each component's sign symmetric or stuck one way?")
    print(f"     {'comp':6s} {'mean':>8} {'%pos':>6} {'%neg':>6} {'%zero':>6}   (balanced ≈ 50/50)")
    for c in COMPONENTS:
        v = df[c].to_numpy(float)
        nz = v[v != 0]
        pos = 100 * (nz > 0).mean() if len(nz) else 0
        neg = 100 * (nz < 0).mean() if len(nz) else 0
        zero = 100 * (v == 0).mean()
        flag = "  <-- BIASED" if len(nz) and (pos > 70 or neg > 70) else ""
        print(f"     {c:6s} {v.mean():>8.3f} {pos:>5.0f}% {neg:>5.0f}% {zero:>5.0f}%{flag}")

    print("\n  2) BIAS on DOWN bars (should lean bearish if honest) vs UP bars")
    print(f"     {'comp':6s} {'down:mean':>10} {'down %pos':>10}   {'up:mean':>9} {'up %pos':>9}")
    dn = df[df.bar_dir < 0]; up = df[df.bar_dir > 0]
    for c in COMPONENTS:
        dnz = dn[c][dn[c] != 0]; unz = up[c][up[c] != 0]
        dp = 100 * (dnz > 0).mean() if len(dnz) else float("nan")
        upp = 100 * (unz > 0).mean() if len(unz) else float("nan")
        print(f"     {c:6s} {dn[c].mean():>10.3f} {dp:>9.0f}%   {up[c].mean():>9.3f} {upp:>8.0f}%")

    print("\n  3) SKILL — IC(component, fwd_ret) [edge only if CI clears 0]")
    for c in COMPONENTS:
        line = f"     {c:6s} "
        for H in (5, 15, 30):
            sub = df.dropna(subset=[f"ret{H}"])
            sub = sub[sub[c] != 0]
            if len(sub) < 10 or sub.date.nunique() < 2:
                line += f"{H}m: n/a  "; continue
            ic, lo, hi = _boot_ci(_spearman, sub[c].to_numpy(float),
                                  sub[f"ret{H}"].to_numpy(float),
                                  reps=reps, rng=rng, groups=sub["date"].to_numpy())
            tag = "+" if lo > 0 else ("-" if hi < 0 else "0")
            line += f"{H}m:{ic:+.3f}[{tag}] "
        print(line)

    print("\n" + "=" * 82)
    print("READ: a healthy directional component is ~50/50 pos/neg overall, leans BEARISH")
    print("on down bars, and has an IC whose CI clears 0. Stuck-positive = structural long")
    print("bias (the 95%-CALL bug). IC CI straddling 0 = no directional skill (context only).")


def main():
    rng = np.random.default_rng(SEED)
    days = _captured_days()
    if not days:
        print("no days"); return
    print(f"reading {LIVE_DIR} (tf={TF})")
    df = harvest(days)
    evaluate(df, 1500, rng)


if __name__ == "__main__":
    main()
