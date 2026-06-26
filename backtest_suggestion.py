"""
backtest_suggestion.py — "at 11:45 the board suggested X; did X happen?"

Your replay-button workflow, made into an honest scoreboard. At every checkpoint of
every captured day it reconstructs the FULL board read the dashboard would show at
that instant (lookahead-free) — the fused playbook composite over OI + premium +
futures + gap + opening-range, the same number the live cockpit suggests — emits a
call, then scores it against what the price ACTUALLY did over the next 5 / 15 / 60
minutes. The forward price is read only to grade the call, never fed back into it.

This is the whole-signal counterpart to backtest_crossover (one factor) and the
multi-horizon, per-day counterpart to backtest_hour_forecast (60-min only). It
answers the exact question you asked: is the live suggestion worth acting on, and
does it actually work on a given day or only pooled?

Output per horizon:
  IC(composite, fwd_ret)  + day-block 95% CI      (does the read rank the move?)
  sign-hit on BULLISH/BEARISH calls vs 50% null   (does the arrow pay?)
  range close-in-band vs ~68% (60m only)          (the honest product)
Plus a PER-DAY scoreboard at +15m: W/L each day, so you see day-to-day stability.

Reads the LOCAL archive (data/intraday/live), the authoritative superset eod_sync
pulls home from the VM each night. Wired to nothing. Re-run as days accumulate.

    .venv\\Scripts\\python.exe backtest_suggestion.py
    .venv\\Scripts\\python.exe backtest_suggestion.py --scoreboard-h 60
"""
from __future__ import annotations

import argparse
import datetime
import sys

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from core.constants import IST, INDEX_SYMBOLS, LIVE_DIR, DATA_DIR
from core.mirror_io import read_mirror as _read
import hour_forecast as hf
from backtest_continuity import _spearman, _boot_ci

SEED = 7
SAMPLE_TIMES = ["09:45", "10:00", "10:15", "10:30", "10:45", "11:00", "11:15",
                "11:30", "11:45", "12:00", "12:15", "12:30", "12:45", "13:00",
                "13:15", "13:30", "13:45", "14:00", "14:15", "14:30"]
HORIZONS = (5, 15, 60)
LEDGER = DATA_DIR / "validation" / "suggestion_ledger.parquet"


def _captured_days() -> list[str]:
    out = set()
    for p in LIVE_DIR.glob("*_oi_snapshots.parquet"):
        if p.stat().st_size < 1024:                 # skip clobbered stubs
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


def harvest(days: list[str]) -> pd.DataFrame:
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
                pred = hf.predict(hf.features(sym, as_of=t, date=date))
                if not pred.get("has_data"):
                    continue
                spot_t = pred["spot"]
                rec = {"date": date, "sym": sym, "t": hhmm, "spot": spot_t,
                       "composite": pred["composite"], "p_up": pred["p_up"],
                       "direction": pred["direction"],
                       "exp_move_pct": pred.get("exp_move_pct"),
                       "lo": pred.get("lo"), "hi": pred.get("hi")}
                # 60m range coverage (band is a 60-min forecast)
                s60 = _spot_at(ticks, t + datetime.timedelta(minutes=60))
                lo, hi = pred.get("lo"), pred.get("hi")
                rec["close_in_band"] = float(lo is not None and hi is not None and s60 is not None
                                             and lo <= s60 <= hi)
                for H in HORIZONS:
                    s_end = _spot_at(ticks, t + datetime.timedelta(minutes=H))
                    rec[f"ret{H}"] = (s_end / spot_t - 1.0) if (s_end and s_end != spot_t) else np.nan
                rows.append(rec)
    return pd.DataFrame(rows)


def _upsert(new: pd.DataFrame) -> pd.DataFrame:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    comb = pd.concat([pd.read_parquet(LEDGER), new], ignore_index=True) if LEDGER.exists() else new
    comb = comb.drop_duplicates(subset=["date", "sym", "t"], keep="last").reset_index(drop=True)
    comb.to_parquet(LEDGER, index=False)
    return comb


def evaluate(df: pd.DataFrame, reps: int, rng, scoreboard_h: int) -> None:
    print("\nBOARD SUGGESTION — did the call work? (day-block CV vs nulls)")
    print("=" * 72)
    if df.empty:
        print("  ledger empty — run a harvest first."); return
    days = sorted(df.date.unique())
    print(f"  rows={len(df)}  days={len(days)} ({days[0]}..{days[-1]})  indices={df.sym.nunique()}")

    print("\n  DIRECTION by horizon   [EDGE only if CI clears 0 / 50%]")
    for H in HORIZONS:
        sub = df.dropna(subset=[f"ret{H}"])
        if len(sub) < 10 or sub.date.nunique() < 2:
            print(f"    {H:>2}m  n={len(sub)} insufficient"); continue
        dts = sub["date"].to_numpy()
        ic, ilo, ihi = _boot_ci(_spearman, sub["composite"].to_numpy(float),
                                sub[f"ret{H}"].to_numpy(float), reps=reps, rng=rng, groups=dts)
        icv = "EDGE" if ilo > 0 else ("anti" if ihi < 0 else "—")
        line = f"    {H:>2}m  IC {ic:+.3f} [{ilo:+.3f},{ihi:+.3f}] {icv}"
        act = sub[sub.direction.isin(["BULLISH", "BEARISH"])]
        if len(act) >= 5 and act.date.nunique() >= 2:
            call = np.where(act.direction == "BULLISH", 1.0, -1.0)
            hit = (call == np.sign(act[f"ret{H}"].to_numpy())).astype(float)
            hr, lo, hi = _boot_ci(lambda a: a.mean(), hit, reps=reps, rng=rng,
                                  groups=act["date"].to_numpy())
            hv = "EDGE" if lo > 0.5 else ("anti" if hi < 0.5 else "—")
            line += f"   sign-hit {100*hr:4.1f}% [{100*lo:4.1f},{100*hi:4.1f}] {hv} (n={len(act)})"
        print(line)

    # RANGE (60m band)
    cib = df.dropna(subset=["close_in_band"])
    if len(cib) >= 5 and cib.date.nunique() >= 2:
        a = cib["close_in_band"].to_numpy(float)
        cr, lo, hi = _boot_ci(lambda x: x.mean(), a, reps=reps, rng=rng,
                              groups=cib["date"].to_numpy())
        print(f"\n  RANGE 60m close-in-band {100*cr:.1f}% [{100*lo:.1f},{100*hi:.1f}]  "
              f"(target ~68%, n={len(cib)})")

    # PER-DAY SCOREBOARD — the replay workflow
    H = scoreboard_h
    sc = df[df.direction.isin(["BULLISH", "BEARISH"])].dropna(subset=[f"ret{H}"]).copy()
    print(f"\n  SCOREBOARD — directional calls graded at +{H}m (per day)")
    if len(sc) >= 5 and sc.date.nunique() >= 2:
        sc["hit"] = (np.where(sc.direction == "BULLISH", 1.0, -1.0) == np.sign(sc[f"ret{H}"])).astype(float)
        for d, grp in sc.groupby("date"):
            w = int(grp.hit.sum()); n = len(grp)
            bar = "+" * w + "-" * (n - w)
            print(f"    {d}   {w:>2}/{n:<2} {100*w/n:4.0f}%  {bar}")
        hit = sc["hit"].to_numpy()
        hr, lo, hi = _boot_ci(lambda a: a.mean(), hit, reps=reps, rng=rng,
                              groups=sc["date"].to_numpy())
        v = "EDGE" if lo > 0.5 else ("anti" if hi < 0.5 else "—")
        print(f"    POOLED {100*hr:.1f}% [{100*lo:.1f},{100*hi:.1f}] {v}  (n={len(sc)} calls vs 50%)")
    else:
        print(f"    n={len(sc)} insufficient")

    print("\n" + "=" * 72)
    print("READ: per-day W/L swings are day-to-day NOISE unless POOLED CI clears 50%.")
    print("One good day ≠ edge. Direction has been null in every test here; the RANGE")
    print("band (close-in-band ≈ 68%) is the trustworthy product. Go live on the band")
    print("+ structural levels, NOT the arrow, until a horizon's CI clears its null OOS.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3000)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--scoreboard-h", type=int, default=15, choices=HORIZONS)
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    if args.eval_only:
        if not LEDGER.exists():
            print("No ledger yet — run without --eval-only first."); return
        evaluate(pd.read_parquet(LEDGER), args.reps, rng, args.scoreboard_h); return

    days = _captured_days()
    if not days:
        print("No captured days in", LIVE_DIR); return
    print(f"reading mirror: {LIVE_DIR}")
    new = harvest(days)
    print(f"harvested {len(new)} rows from {len(days)} days -> upserting {LEDGER.name}")
    comb = _upsert(new) if not new.empty else (pd.read_parquet(LEDGER) if LEDGER.exists() else new)
    evaluate(comb, args.reps, rng, args.scoreboard_h)


if __name__ == "__main__":
    main()
