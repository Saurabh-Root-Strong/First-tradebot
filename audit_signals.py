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


# ── MONITORED INVARIANTS (--check) ──────────────────────────────────────────────
# We assert what must be STRUCTURALLY true, never what we HOPE is true about alpha.
#   * BIAS SYMMETRY is structural: a directional component must be able to point BOTH ways.
#     It broke once (flow was 95% CALL for months — strike-roll + vega + contango-sign bugs,
#     silently buying calls into falling tapes) and the fix must stay fixed. HARD FAIL.
#   * IC is market-dependent and currently ~0 on every component (no measured skill). Asserting
#     IC>0 would fail forever and train you to ignore the alarm. NOT asserted — only a sign
#     INVERSION is warned, since that smells like a new roll/vega-class sign bug.
_GATE_ACTIVE = ["flow", "div", "fut"]   # nonzero gate weight AND actually fires (cross is inert)
_BIAS_LO, _BIAS_HI = 35.0, 65.0         # %positive of NONZERO values; 95%-CALL bug => ~95
_DEAD_ZERO_PCT = 95.0                   # >=this %zero => component contributes ~nothing
_IC_INVERT = -0.10                      # flow 15m IC below this => suspect a sign inversion
# Assert on a TRAILING WINDOW, never the pooled archive. Pooling DILUTES a fresh bias by the
# clean history and gets LESS sensitive as the archive grows — exactly backwards. Measured:
# with 60d of history, a bug holding flow at 95% pos for 10 straight days pools to 58% and
# NEVER fires. The original 95%-CALL bug ran for MONTHS, i.e. precisely the regime pooling
# hides. A 10-day window (~35 nonzero rows/day x 10 = ample) catches a persistent bias within
# about a week, while staying statistically stable.
_WINDOW_DAYS = 10
_MIN_WINDOW_ROWS = 150


def _trailing(df):
    """Last _WINDOW_DAYS captured days. Returns (sub_df, n_days)."""
    days = sorted(df["date"].astype(str).unique())
    keep = set(days[-_WINDOW_DAYS:])
    return df[df["date"].astype(str).isin(keep)], len(keep)


def check_invariants(df) -> int:
    """Assert the structural signal-health invariants. Returns the number of HARD violations
    (0 = healthy). Prints warnings for dead/inverted components (not fatal)."""
    print("\n" + "=" * 82)
    print("SIGNAL-HEALTH CHECK (structural invariants)")
    print("=" * 82)
    win, ndays = _trailing(df)
    print(f"  asserting on the TRAILING {ndays} captured day(s) "
          f"(pooled history would dilute a fresh bias); pooled shown for context.\n")
    violations = []
    for c in _GATE_ACTIVE:
        nz_all = df[c][df[c] != 0]
        nz = win[c][win[c] != 0]
        pooled = 100.0 * (nz_all > 0).mean() if len(nz_all) else float("nan")
        if len(nz) < _MIN_WINDOW_ROWS:
            print(f"  {c:6s} SKIP  (window has {len(nz)} nonzero rows < {_MIN_WINDOW_ROWS} "
                  f"— too thin to judge; pooled {pooled:.1f}%)")
            continue
        pos = 100.0 * (nz > 0).mean()
        ok = _BIAS_LO <= pos <= _BIAS_HI
        print(f"  {c:6s} window %pos={pos:5.1f}  (pooled {pooled:5.1f})  "
              f"band[{_BIAS_LO:.0f},{_BIAS_HI:.0f}]  {'OK' if ok else 'STRUCTURAL BIAS'}")
        if not ok:
            violations.append(f"{c} is {pos:.0f}% positive over the last {ndays} days — "
                              f"outside [{_BIAS_LO:.0f},{_BIAS_HI:.0f}] (the 95%-CALL bug class)")
    # warnings (never fatal)
    for c in COMPONENTS:
        zero = 100.0 * (df[c] == 0).mean()
        if zero >= _DEAD_ZERO_PCT:
            print(f"  WARN  {c} is {zero:.0f}% zero — inert; its gate weight is dead.")
    sub = df.dropna(subset=["ret15"])
    sub = sub[sub["flow"] != 0]
    if len(sub) >= 30:
        ic = _spearman(sub["flow"].to_numpy(float), sub["ret15"].to_numpy(float))
        print(f"  flow 15m IC = {ic:+.3f} (informational; skill is NOT asserted)")
        if ic < _IC_INVERT:
            print(f"  WARN  flow 15m IC {ic:+.3f} < {_IC_INVERT} — possible sign inversion "
                  "(roll/vega-class bug). Investigate.")
    print("-" * 82)
    if violations:
        print(f"SIGNAL DRIFT ({len(violations)}):")
        for v in violations:
            print(f"  {v}")
    else:
        print("ALL STRUCTURAL INVARIANTS HOLD — components can still point both ways.")
    return len(violations)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="scout signal component bias + skill audit")
    ap.add_argument("--check", action="store_true",
                    help="assert structural invariants; exit 1 on drift (for the weekly cron)")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    days = _captured_days()
    if not days:
        print("no days")
        sys.exit(2 if args.check else 0)
    print(f"reading {LIVE_DIR} (tf={TF})")
    df = harvest(days)
    evaluate(df, 1500, rng)
    if args.check:
        sys.exit(1 if check_invariants(df) else 0)


if __name__ == "__main__":
    main()
