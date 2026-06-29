"""
backtest_band_reversion.py — is the band-EDGE a tradeable fade? (the last distinct
directional hypothesis on the current data)

Every directional variant has died: the arrow (buy) bleeds, flip-buy dies on cost, the
trend gate hurts (the arrow is contrarian), the struct/trend vetoes don't pay. The ONE
validated primitive is the hour_forecast range band (~68% close-in-band). This asks the
only fresh question that builds ON that primitive instead of the dead arrow:

  after price has travelled ~one band half-width away from where it was, does it REVERT
  toward the mean — strongly enough to trade NET of cost?

The band is spot-centered at t, so position is 0 at t. We measure the PATH: take the
band half-width hw at t, the move over the trailing LB minutes scaled into band units
(position = (spot_t - spot_{t-LB}) / hw_t), and ask whether the forward H-minute return
reverses it. reversion ⇔ corr(position, fwd_ret) < 0; the fade trade earns
fade_ret = -sign(position)·fwd_ret and must clear a cost hurdle to be real.

Lookahead-free: hw_t and position use only ts<=t; the forward spot is the answer key.
Day-block bootstrap CI (a multi-comparisons-aware verdict — this is the LAST variant to
run on these ~9-15 days before the count erodes statistical integrity).

    .venv\\Scripts\\python.exe backtest_band_reversion.py
"""
from __future__ import annotations

import argparse
import datetime
import sys

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from core.constants import IST, INDEX_SYMBOLS, LIVE_DIR
from core.mirror_io import read_mirror as _read
from backtest_continuity import _spearman, _boot_ci
import hour_forecast as hf

SEED = 7
SAMPLE_TIMES = ["09:45", "10:00", "10:15", "10:30", "10:45", "11:00", "11:15",
                "11:30", "11:45", "12:00", "12:15", "12:30", "12:45", "13:00",
                "13:15", "13:30", "13:45", "14:00", "14:15", "14:30", "14:45"]
LB_MIN = 30                 # trailing window over which the "move into the edge" is measured
HORIZONS = (5, 15, 30, 60)
EDGE = 0.7                  # |position| at/above this = price is AT a band edge
COST_BPS = 6.0             # index round-trip a spot/futures fade must clear (0.06%)


def _captured_days() -> list[str]:
    out = set()
    for p in LIVE_DIR.glob("*_oi_snapshots.parquet"):
        if p.stat().st_size < 1024:
            continue
        stem = p.name.split("_")[0]
        try:
            datetime.date.fromisoformat(stem); out.add(stem)
        except ValueError:
            continue
    return sorted(out)


def _spot_at(ticks, t):
    s = ticks[ticks["ts"] <= pd.Timestamp(t)]
    return float(s.iloc[-1]["ltp"]) if len(s) else None


def harvest(days, tf_for_band: int = 15) -> pd.DataFrame:
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
                spot_t = _spot_at(ticks, t)
                spot_ref = _spot_at(ticks, t - datetime.timedelta(minutes=LB_MIN))
                if not spot_t or not spot_ref:
                    continue
                fc = hf.forecast(sym, as_of=t, date=date)
                emp = fc.get("exp_move_pct")
                if not emp or emp <= 0:
                    continue
                hw = spot_t * emp / 100.0                      # band half-width (index pts)
                if hw <= 0:
                    continue
                position = (spot_t - spot_ref) / hw            # band units travelled
                rec = {"date": date, "sym": sym, "t": hhmm, "spot": spot_t,
                       "hw": hw, "position": position, "abs_pos": abs(position)}
                for H in HORIZONS:
                    s_end = _spot_at(ticks, t + datetime.timedelta(minutes=H))
                    if s_end and s_end != spot_t:
                        ret = (s_end / spot_t - 1.0) * 100.0
                        rec[f"ret{H}"] = ret
                        # fader: short if we're above (pos>0), long if below → profit on revert
                        rec[f"fade{H}"] = -np.sign(position) * ret
                    else:
                        rec[f"ret{H}"] = np.nan; rec[f"fade{H}"] = np.nan
                rows.append(rec)
    return pd.DataFrame(rows)


def evaluate(df: pd.DataFrame, reps: int, rng) -> None:
    print("\nBAND-EDGE REVERSION — does a ~1-band move fade back? (day-block CV)")
    print("=" * 78)
    if df.empty:
        print("  no rows"); return
    days = sorted(df.date.unique())
    edge = df[df.abs_pos >= EDGE]
    print(f"  rows={len(df)}  days={len(days)} ({days[0]}..{days[-1]})  "
          f"indices={df.sym.nunique()}  at-edge(|pos|>={EDGE})={len(edge)} "
          f"({100*len(edge)/max(len(df),1):.0f}%)  LB={LB_MIN}m")

    # 1) IC(position, fwd_ret) — NEGATIVE = reversion, POSITIVE = momentum
    print("\n  1) IC( position , fwd_ret )   [reversion if CI < 0]")
    for H in HORIZONS:
        sub = df.dropna(subset=[f"ret{H}"])
        if len(sub) < 10 or sub.date.nunique() < 2:
            print(f"     {H:>2}m  n={len(sub)} insufficient"); continue
        ic, lo, hi = _boot_ci(_spearman, sub["position"].to_numpy(float),
                              sub[f"ret{H}"].to_numpy(float),
                              reps=reps, rng=rng, groups=sub["date"].to_numpy())
        v = "REVERSION" if hi < 0 else ("momentum" if lo > 0 else "—")
        print(f"     {H:>2}m  IC {ic:+.3f} [{lo:+.3f},{hi:+.3f}] {v}  (n={len(sub)})")

    # 2) FADE return AT THE EDGE — the actual trade, net of a spot/futures cost hurdle
    print(f"\n  2) FADE @ edge (|pos|>={EDGE}): short the top / long the bottom, target mid")
    for H in HORIZONS:
        sub = edge.dropna(subset=[f"fade{H}"])
        if len(sub) < 5 or sub.date.nunique() < 2:
            print(f"     {H:>2}m  n={len(sub)} too few at edge"); continue
        f = sub[f"fade{H}"].to_numpy(float)
        net = f - COST_BPS / 100.0
        mn, lm, hm = _boot_ci(lambda a: a.mean(), net, reps=reps, rng=rng,
                              groups=sub["date"].to_numpy())
        win = (f > 0).astype(float)
        wr, lw, hw = _boot_ci(lambda a: a.mean(), win, reps=reps, rng=rng,
                              groups=sub["date"].to_numpy())
        ev = "EDGE" if lm > 0 else ("bleed" if hm < 0 else "—")
        print(f"     {H:>2}m  win {100*wr:4.1f}% [{100*lw:4.1f},{100*hw:4.1f}]  "
              f"mean net {mn:+.3f}% [{lm:+.3f},{hm:+.3f}] {ev}  (n={len(sub)})")

    # 3) does being AT the edge beat the WHOLE sample? (is the edge condition adding info)
    print("\n  3) edge vs all (mean raw fade%, no cost) — does the edge filter help?")
    for H in HORIZONS:
        a = df.dropna(subset=[f"fade{H}"])[f"fade{H}"]
        e = edge.dropna(subset=[f"fade{H}"])[f"fade{H}"]
        if len(a) and len(e):
            print(f"     {H:>2}m  all {a.mean():+.3f}% (n={len(a)})   "
                  f"edge {e.mean():+.3f}% (n={len(e)})")

    print("\n" + "=" * 78)
    print("READ: a tradeable fade needs IC<0 (reversion) AND edge mean-net-CI clearing 0")
    print("at a horizon. Index-spot/futures cost (~0.06%) is far cheaper than the 3% option")
    print("round-trip — if it won't clear even THIS hurdle, no option expression saves it.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    days = _captured_days()
    if not days:
        print("No captured days in", LIVE_DIR); return
    print(f"reading mirror: {LIVE_DIR}  (band from hour_forecast, LB={LB_MIN}m)")
    df = harvest(days)
    evaluate(df, args.reps, rng)


if __name__ == "__main__":
    main()
