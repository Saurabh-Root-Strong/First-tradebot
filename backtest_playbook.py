"""
backtest_playbook.py — does the Opening Playbook's call carry forward edge?

Falsification, same discipline as backtest_continuity.py:
  * signal  = opening_playbook.playbook_index(sym, as_of=t, date=day) — the REAL
              engine, reconstructed lookahead-free from that day's mirrors at the
              sample instant t (>= 09:35, the earliest call).
  * outcome = forward spot return from t to t+60min and to EOD, from the day's
              tick mirror (independent of the signal's candle/OI reads).
  * stats   = Spearman IC(composite, fwd) + directional sign-hit, with a
              DAY-BLOCK bootstrap CI (intraday samples within a day are NOT
              independent — see backtest_continuity._boot_ci).

Also reports each FACTOR's standalone IC (or/gap/oi/prem/fut/eod), so we see
which of the six actually points the right way.

CAVEAT (honest): the EOD factor reads daily_context_bridge.score_index(), whose
cache is latest-only (not date-addressable) — so in replay it injects TODAY's EOD
context, a lookahead. EOD-factor and full-composite numbers are therefore
OPTIMISTIC; the intraday-only composite (or/gap/oi/prem/fut) is the clean read.

    .venv\\Scripts\\python.exe backtest_playbook.py
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
import opening_playbook as opb
from backtest_continuity import _spearman, _boot_ci

SEED = 7
SAMPLE_TIMES = ["09:40", "10:00", "10:30", "11:00"]   # >= PLAYBOOK_READY (09:35)
HORIZONS = [15, 30, 45, 60]        # the "valid for ~15-60 min" decay test
EOD_CUT = datetime.time(15, 25)
# intraday-only weights (drop the lookahead-contaminated EOD factor), renormalised
_INTRA_W = {"or": 0.20, "gap": 0.10, "oi": 0.25, "prem": 0.15, "fut": 0.10}
_INTRA_TOT = sum(_INTRA_W.values())


def _spot_at(ticks: pd.DataFrame, t: datetime.datetime) -> float | None:
    s = ticks[ticks["ts"] <= pd.Timestamp(t)]
    return float(s.iloc[-1]["ltp"]) if len(s) else None


def _captured_days() -> list[str]:
    days = sorted({p.name.split("_")[0] for p in LIVE_DIR.glob("*_oi_snapshots.parquet")})
    return days


def collect(days: list[str]) -> pd.DataFrame:
    rows = []
    for date in days:
        d0 = datetime.date.fromisoformat(date)
        for sym in INDEX_SYMBOLS:
            ticks = _read("ticks", date, None, sym)
            if ticks is None or len(ticks) < 20:
                continue
            for hhmm in SAMPLE_TIMES:
                hh, mm = map(int, hhmm.split(":"))
                t = datetime.datetime.combine(d0, datetime.time(hh, mm), tzinfo=IST)
                p = opb.playbook_index(sym, as_of=t, date=date)
                if not p.get("has_data"):
                    continue
                parts = p.get("parts", {})
                intra = sum(_INTRA_W[k] * parts.get(k, 0) for k in _INTRA_W) / _INTRA_TOT
                spot_t = _spot_at(ticks, t)
                if not spot_t:
                    continue
                fwd = {}
                for h in HORIZONS:
                    fh = _spot_at(ticks, t + datetime.timedelta(minutes=h))
                    fwd[f"fwd{h}"] = (fh / spot_t - 1.0) if (fh and fh != spot_t) else np.nan
                feod = _spot_at(ticks, datetime.datetime.combine(d0, EOD_CUT, tzinfo=IST))
                rows.append({
                    "date": date, "sym": sym, "t": hhmm,
                    "composite": p.get("composite", 0.0), "intra": round(intra, 3),
                    "conviction": p.get("conviction", 0), "direction": p.get("direction"),
                    **{f"f_{k}": parts.get(k, 0.0) for k in
                       ("or", "gap", "oi", "prem", "fut", "eod")},
                    **fwd,
                    "fwd_eod": (feod / spot_t - 1.0) if (feod and feod != spot_t) else np.nan,
                })
    return pd.DataFrame(rows)


def _ic_line(name, score, ret, dates, reps, rng):
    d = pd.DataFrame({"s": score, "r": ret, "g": dates}).dropna()
    if len(d) < 5 or d["g"].nunique() < 2:
        return f"   {name:10}: n<5 or <2 days"
    ic, lo, hi = _boot_ci(_spearman, d["s"].to_numpy(float), d["r"].to_numpy(float),
                          reps=reps, rng=rng, groups=d["g"].to_numpy())
    v = "EDGE" if lo > 0 else ("anti" if hi < 0 else "—")
    return f"   {name:10}: IC {ic:+.3f} [{lo:+.3f},{hi:+.3f}] {v}  (n={len(d)})"


def report(df: pd.DataFrame, reps: int, rng) -> None:
    print("\nOPENING PLAYBOOK — forward-edge test (lookahead-free, day-block CI)")
    print("=" * 74)
    if df.empty:
        print("  no rows — need captured mirror days with OR + OI."); return
    days = sorted(df.date.unique())
    print(f"  rows={len(df)}  days={len(days)} ({days[0]}..{days[-1]})  "
          f"indices={df.sym.nunique()}  samples/day~{len(SAMPLE_TIMES)}")

    dates = df["date"].to_numpy()
    # ── DECAY: does the call hold for ~15-60 min then fade? (intra-only composite)
    print("\n── HORIZON DECAY  (intra-only composite vs forward spot) " + "-" * 16)
    for hcol in [f"fwd{h}" for h in HORIZONS] + ["fwd_eod"]:
        d = df.dropna(subset=[hcol])
        if len(d) < 5 or d["date"].nunique() < 2:
            print(f"   {hcol:8} n<5"); continue
        ic, ilo, ihi = _boot_ci(_spearman, d["intra"].to_numpy(float),
                                d[hcol].to_numpy(float), reps=reps, rng=rng,
                                groups=d["date"].to_numpy())
        act = d[d.direction.isin(["BULLISH", "BEARISH"])]
        if len(act) >= 5 and act["date"].nunique() >= 2:
            call = np.where(act.direction == "BULLISH", 1.0, -1.0)
            hit = (call == np.sign(act[hcol].to_numpy())).astype(float)
            hr, hlo, hhi = _boot_ci(lambda a: a.mean(), hit, reps=reps, rng=rng,
                                    groups=act["date"].to_numpy())
            hs = f"dir-hit {100*hr:4.1f}% [{100*hlo:4.0f},{100*hhi:4.0f}] n={len(act)}"
        else:
            hs = "dir-hit n<5"
        flag = "EDGE" if ilo > 0 else ("anti" if ihi < 0 else "—")
        print(f"   {hcol:8} IC {ic:+.3f} [{ilo:+.3f},{ihi:+.3f}] {flag:4}  {hs}")

    # ── per-factor IC at the short horizon the user cares about (30 min)
    print("\n── per-factor IC @ fwd30 " + "-" * 30)
    for k in ("or", "gap", "oi", "prem", "fut", "eod"):
        tag = " (leak)" if k == "eod" else ""
        print(_ic_line(k, df[f"f_{k}"].to_numpy(float),
                       df["fwd30"].to_numpy(float), dates, reps, rng) + tag)

    print("\n" + "=" * 74)
    print("READ: 'EDGE' = day-block 95% CI excludes null. With few captured days the")
    print("CI is wide by design — trust nothing until it holds over many full days.")
    print("EOD factor + full composite are OPTIMISTIC (replay lookahead in the bridge);")
    print("the intra-only line is the honest first-20-min F&O edge.")


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
