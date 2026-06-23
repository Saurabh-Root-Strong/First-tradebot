"""
backtest_hour_forecast.py — accumulate + validate the next-60-min forecast.

Two jobs:
  1. HARVEST  — for every captured mirror day, reconstruct hour_forecast at sample
     instants (lookahead-free) and label it with the realised next-60-min outcome,
     then UPSERT into a persistent ledger (data/validation/hour_forecast_ledger.parquet,
     dedup by date+sym+t). The ledger grows one block per captured day, building an
     honest out-of-sample track record over time.
  2. EVALUATE — day-block-bootstrap CV of the ledger:
       direction: sign-hit vs 50% coin-flip + Spearman IC(composite, ret60)
       range    : 60-min CLOSE-in-band + PATH-containment vs the ~68% nominal target

Nothing is wired into live decisions. Re-run daily (or via eod_sync) — the verdict
sharpens as days accumulate; only act once it clears the nulls out-of-sample.

    .venv\\Scripts\\python.exe backtest_hour_forecast.py            # harvest + eval
    .venv\\Scripts\\python.exe backtest_hour_forecast.py --eval-only
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
SAMPLE_TIMES = ["09:45", "10:00", "10:30", "11:00", "11:30", "12:00", "12:30", "13:00", "13:30", "14:00"]
HORIZON = 60
LEDGER = DATA_DIR / "validation" / "hour_forecast_ledger.parquet"


def _spot_at(ticks, t):
    s = ticks[ticks["ts"] <= pd.Timestamp(t)]
    return float(s.iloc[-1]["ltp"]) if len(s) else None


def _captured_days() -> list[str]:
    out = set()
    for p in LIVE_DIR.glob("*_oi_snapshots.parquet"):
        stem = p.name.split("_")[0]
        try:
            datetime.date.fromisoformat(stem)        # only real YYYY-MM-DD stems
            out.add(stem)
        except ValueError:
            continue
    return sorted(out)


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
                feat = hf.features(sym, as_of=t, date=date)
                pred = hf.predict(feat)
                if not pred.get("has_data"):
                    continue
                spot_t = pred["spot"]
                tend = t + datetime.timedelta(minutes=HORIZON)
                s_end = _spot_at(ticks, tend)
                if not s_end or s_end == spot_t:
                    continue
                ret60 = s_end / spot_t - 1.0
                path = ticks[(ticks["ts"] > pd.Timestamp(t)) & (ticks["ts"] <= pd.Timestamp(tend))]
                lo, hi = pred.get("lo"), pred.get("hi")
                close_in = (lo is not None and hi is not None and lo <= s_end <= hi)
                path_in = (lo is not None and hi is not None and len(path) > 0
                           and float(path["ltp"].min()) >= lo and float(path["ltp"].max()) <= hi)
                rows.append({
                    "date": date, "sym": sym, "t": hhmm,
                    "composite": pred["composite"], "p_up": pred["p_up"],
                    "direction": pred["direction"], "exp_move_pct": pred["exp_move_pct"],
                    "ret60": ret60, "close_in_band": float(close_in), "path_in_band": float(path_in),
                    **{k: feat.get(k) for k in feat if k.startswith("f_")},
                })
    return pd.DataFrame(rows)


def _upsert(new: pd.DataFrame) -> pd.DataFrame:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    if LEDGER.exists():
        old = pd.read_parquet(LEDGER)
        comb = pd.concat([old, new], ignore_index=True)
    else:
        comb = new
    comb = comb.drop_duplicates(subset=["date", "sym", "t"], keep="last").reset_index(drop=True)
    comb.to_parquet(LEDGER, index=False)
    return comb


def evaluate(df: pd.DataFrame, reps: int, rng) -> None:
    print("\nHOUR FORECAST — track record (day-block CV vs nulls)")
    print("=" * 70)
    if df.empty or "ret60" not in df.columns:
        print("  ledger empty — run a harvest first."); return
    df = df.dropna(subset=["ret60"])
    days = sorted(df.date.unique())
    print(f"  ledger rows={len(df)}  days={len(days)} ({days[0]}..{days[-1]})  indices={df.sym.nunique()}")

    # DIRECTION
    dates = df["date"].to_numpy()
    ic, ilo, ihi = _boot_ci(_spearman, df["composite"].to_numpy(float),
                            df["ret60"].to_numpy(float), reps=reps, rng=rng, groups=dates)
    icv = "EDGE" if ilo > 0 else ("anti" if ihi < 0 else "—")
    print(f"\n  DIRECTION")
    print(f"    IC(composite, ret60) {ic:+.3f} [{ilo:+.3f},{ihi:+.3f}] {icv}")
    act = df[df.direction.isin(["BULLISH", "BEARISH"])]
    if len(act) >= 5 and act["date"].nunique() >= 2:
        call = np.where(act.direction == "BULLISH", 1.0, -1.0)
        hit = (call == np.sign(act["ret60"].to_numpy())).astype(float)
        hr, lo, hi = _boot_ci(lambda a: a.mean(), hit, reps=reps, rng=rng, groups=act["date"].to_numpy())
        v = "EDGE" if lo > 0.5 else ("anti" if hi < 0.5 else "—")
        print(f"    sign-hit {100*hr:4.1f}% [{100*lo:4.1f},{100*hi:4.1f}] {v}  (n={len(act)} vs 50% null)")

    # RANGE (target ~68% for the 1-sigma band)
    print(f"\n  RANGE BAND (target ~68%)")
    for col, label in [("close_in_band", "close-in-band"), ("path_in_band", "path-contained")]:
        a = df[col].to_numpy(float)
        cr, lo, hi = _boot_ci(lambda x: x.mean(), a, reps=reps, rng=rng, groups=dates)
        print(f"    {label:15} {100*cr:4.1f}% [{100*lo:4.1f},{100*hi:4.1f}]  (n={len(df)})")
    print(f"    median band ±{df['exp_move_pct'].median():.2f}%")

    print("\n" + "=" * 70)
    print("READ: day-block 95% CI. DIRECTION 'EDGE' only if hit-CI clears 50% / IC>0.")
    print("RANGE close-in-band near 68% = calibrated band. Wide CI now (few days) —")
    print("the ledger accumulates; act only when direction clears the null OOS.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3000)
    ap.add_argument("--eval-only", action="store_true", help="skip harvest, just evaluate the ledger")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    if args.eval_only:
        if not LEDGER.exists():
            print("No ledger yet — run without --eval-only first."); return
        evaluate(pd.read_parquet(LEDGER), args.reps, rng); return

    days = _captured_days()
    if not days:
        print("No captured mirror days in", LIVE_DIR); return
    new = harvest(days)
    print(f"harvested {len(new)} rows from {len(days)} days -> upserting ledger {LEDGER.name}")
    comb = _upsert(new) if not new.empty else (pd.read_parquet(LEDGER) if LEDGER.exists() else new)
    evaluate(comb, args.reps, rng)


if __name__ == "__main__":
    main()
