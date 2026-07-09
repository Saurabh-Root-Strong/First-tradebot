"""
backtest_reversal.py — grade the REVERSAL-ACCUMULATION signal (smart_money.reversal_signal).

The hypothesis (user's): when writers secretly CLOSE their shorts (OI down, premium up) and
fresh BUYING enters at cheap premium, a positioning flip precedes a directional move — a
gamma-squeeze / pin-break, most violent near expiry. This harness tests whether the fire
actually leads a favourable move, and whether BUYING the arrow on it pays after cost.

Lookahead-free: reversal_signal(as_of=t) reads chain rows with ts<=t; the forward option
premium + spot at t+H are the answer key only, never fed back.

    two grades per fire (side = bullish->CE / bearish->PE):
      • SPOT   — did the index move the signalled way over the next H minutes? (the pure read)
      • OPTION — buy ATM CE/PE at t, sell at t+H, net of one round-trip cost (the money grade)
    split by CHEAP (DTE<=2 = near expiry) vs not — the user's specific 'cheap premium' case.

DISPLAY/RESEARCH ONLY. This fires rarely (day-level net-covering is uncommon) so the sample
is thin — the point is to ACCUMULATE and re-run as captured (esp. expiry) days build. Never
wire it as a trade trigger on a thin, non-clearing CI.

    .venv\\Scripts\\python.exe backtest_reversal.py
    .venv\\Scripts\\python.exe backtest_reversal.py --days 9 --step 15
"""
from __future__ import annotations

import argparse
import datetime

import numpy as np
import pandas as pd

from core.constants import IST, INDEX_SYMBOLS
from backtest_continuity import _boot_ci
from backtest_scout_trades import _dte, _dte_bkt, _target_week, _grid
import intraday_scout as scout
import smart_money as sm

SEED = 7
HORIZONS = (30, 60)


def simulate(days: list[str], step: int) -> pd.DataFrame:
    rows = []
    for day in days:
        for sym in INDEX_SYMBOLS:
            dte = _dte(sym, day)
            for t in _grid(day, step):
                rev = sm.reversal_signal(sym, date=day, as_of=t, dte=dte)
                if not rev.get("fired"):
                    continue
                side = "CE" if rev["side"] == "bullish" else "PE"
                spot0 = scout._spot_at(sym, day, t)
                atm = scout._atm(spot0, sym) if spot0 else None
                if not (spot0 and atm):
                    continue
                e_prem = scout._opt_premium(sym, day, t, atm, side)
                rec = {"date": day, "sym": sym, "t": t.strftime("%H:%M"), "side": side,
                       "dte": dte, "dte_bkt": _dte_bkt(dte), "cheap": rev.get("cheap", False),
                       "cover_L": rev.get("cover_L"), "confirmed": rev.get("confirmed"),
                       "strength": rev.get("strength")}
                for H in HORIZONS:
                    tH = t + datetime.timedelta(minutes=H)
                    spotH = scout._spot_at(sym, day, tH)
                    # signed spot move in the SIGNALLED direction (CE up / PE down = +)
                    rec[f"spot_{H}"] = (((spotH / spot0 - 1.0) * 100.0 * (1 if side == "CE" else -1))
                                        if (spotH and spot0) else np.nan)
                    xp = scout._opt_premium(sym, day, tH, atm, side)
                    rec[f"opt_{H}"] = (((xp / e_prem - 1.0) * 100.0 - scout._OPT_RT_COST)
                                       if (e_prem and xp) else np.nan)
                rows.append(rec)
    return pd.DataFrame(rows)


def _grade(df: pd.DataFrame, label: str, rng, reps: int) -> None:
    n = len(df)
    if not n:
        print(f"  {label:20s} n=0")
        return
    bits = [f"  {label:20s} n={n:<3d}"]
    for H in HORIZONS:
        sp = df[f"spot_{H}"].dropna().to_numpy(float)
        op = df[f"opt_{H}"].dropna().to_numpy(float)
        sp_hit = (sp > 0).mean() * 100 if len(sp) else float("nan")
        bits.append(f"| {H}m: spot-hit {sp_hit:3.0f}%  opt {op.mean():+5.1f}% (W{100*(op>0).mean():3.0f}%)"
                    if len(op) else f"| {H}m: n<1")
    print(" ".join(bits))


def report(df: pd.DataFrame, days: list[str], reps: int) -> None:
    rng = np.random.default_rng(SEED)
    print("\n" + "=" * 92)
    print("REVERSAL-ACCUMULATION GRADE — writers short-cover + fresh buy → forward move?")
    print("=" * 92)
    print(f"  sessions={len(days)} ({days[0]}..{days[-1]})   FIRES={len(df)}")
    if df.empty:
        print("  no fires in sample — day-level net-covering is rare; accumulate more days.")
        return
    _grade(df, "ALL fires", rng, reps)
    print("\n  BY CHEAP (near-expiry DTE<=2 = the user's cheap-premium case)")
    _grade(df[df["cheap"]], "CHEAP (DTE<=2)", rng, reps)
    _grade(df[~df["cheap"].astype(bool)], "not cheap", rng, reps)
    print("\n  BY SIDE")
    for s in ("CE", "PE"):
        _grade(df[df["side"] == s], s, rng, reps)
    print("\n  BY CONFIRMED (fresh same-side buying present)")
    _grade(df[df["confirmed"].astype(bool)], "confirmed+buy", rng, reps)
    print("\n  BY INDEX")
    for s in INDEX_SYMBOLS:
        _grade(df[df.sym == s], s.split(":")[1][:12], rng, reps)
    # honest CI on the money grade if enough
    op = df["opt_60"].dropna().to_numpy(float) if "opt_60" in df else np.empty(0)
    if len(op) >= 8 and df.dropna(subset=["opt_60"]).date.nunique() >= 2:
        mw, ml, mh = _boot_ci(lambda a: a.mean(), op,
                              reps=reps, rng=rng,
                              groups=df.dropna(subset=["opt_60"])["date"].to_numpy())
        ev = "EDGE" if ml > 0 else ("bleed" if mh < 0 else "flat")
        print(f"\n  60m option-net day-block CI: {op.mean():+.1f}% [{ml:+.1f},{mh:+.1f}] {ev}")
    print("\n" + "=" * 92)
    print("READ: fires rarely (net-covering is uncommon at day level) → SAMPLE IS THIN.")
    print("Verdict only when spot-hit clears 50% AND 60m option-net CI clears 0 OOS. Until")
    print("then: DISPLAY CONTEXT ONLY, never a trigger — a false fire near expiry is −45%.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=9)
    ap.add_argument("--step", type=int, default=15)
    ap.add_argument("--reps", type=int, default=2000)
    args = ap.parse_args()
    days = _target_week(args.days)
    if not days:
        print("no usable captured sessions found")
        return
    print(f"grading reversal signal over {len(days)} sessions: {days}")
    df = simulate(days, args.step)
    report(df, days, args.reps)


if __name__ == "__main__":
    main()
