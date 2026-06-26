"""
backtest_manage.py — do the scout's CLOSE/HOLD manage rules beat naive hold?

The arrow is measured negative-EV (backtest_scout). The only hope is risk DISCIPLINE:
exit fast on SL / flow-reversal / sideways-theta instead of holding to a fixed
horizon. This harness tests exactly that, PAIRED on the same entries:

  for every FRESH scout TRADE trigger (NO-TRADE/opposite -> TRADE), buy the ATM
  CE/PE at entry, then run TWO exits on the SAME trade:
     MANAGED  exit at the first of: SL hit · target hit · flow reversed ·
              sideways+theta bleed · max-hold   (the _lifecycle rules)
     HOLD     exit at max-hold (the naive baseline)
  both net of round-trip cost. Paired so the only difference is the exit rule.

Reports managed vs hold mean net P&L + win-rate, and a DAY-BLOCK CI on the
DIFFERENCE (managed − hold) — managing helps only if that difference clears 0.
Lookahead-free: entry/strength read ts<=t; premium path read forward only to price
the exits (the answer key), never fed back. Reads the LOCAL archive. Few days =
COUNTS; treat the CI as honest-but-wide.

    .venv\\Scripts\\python.exe backtest_manage.py
    .venv\\Scripts\\python.exe backtest_manage.py --tf 15 --maxhold 4 --reps 3000
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
from backtest_continuity import _boot_ci
import intraday_scout as scout

SEED = 7
SAMPLE_TIMES = ["09:30", "09:45", "10:00", "10:15", "10:30", "10:45", "11:00",
                "11:15", "11:30", "11:45", "12:00", "12:15", "12:30", "12:45",
                "13:00", "13:15", "13:30", "13:45", "14:00", "14:15", "14:30"]
COST = scout._OPT_RT_COST


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


def _prem_series(chain: pd.DataFrame, strike: int, side: str) -> pd.Series:
    """ltp(ts) for one (strike, side), sorted — for asof reads along the trade."""
    if chain is None or "ltp" not in chain.columns:
        return pd.Series(dtype=float)
    sub = chain[(chain["side"] == side) & (chain["strike"] == strike)]
    if not len(sub):
        return pd.Series(dtype=float)
    return sub.set_index("ts")["ltp"].sort_index()


def _asof(s: pd.Series, t):
    if s is None or not len(s):
        return None
    v = s[s.index <= pd.Timestamp(t)]
    if not len(v):
        return None
    x = float(v.iloc[-1])
    return x if (x == x and x > 0) else None


def harvest(days: list[str], tf: int, maxhold: int) -> pd.DataFrame:
    sl_pct, t1_pct = scout._SLT.get(tf, (0.32, 0.55))
    rows = []
    for date in days:
        d0 = datetime.date.fromisoformat(date)
        for sym in INDEX_SYMBOLS:
            chain = _read("chain_snapshots", date, None, sym)
            ticks = _read("ticks", date, None, sym)
            if chain is None or ticks is None or len(ticks) < 20:
                continue
            px = ticks.set_index("ts")["ltp"].sort_index()
            grid = [datetime.datetime.combine(d0, datetime.time(*map(int, hhmm.split(":"))),
                                              tzinfo=IST) for hhmm in SAMPLE_TIMES]
            # one pass: scout read at each checkpoint
            reads = []
            for t in grid:
                r = scout.scan_index(sym, tf, date=date, as_of=t, with_lifecycle=False)
                reads.append(r if r.get("has_data") else None)

            for k, r in enumerate(reads):
                if r is None or not r["verdict"].startswith("TRADE"):
                    continue
                prev = reads[k - 1] if k > 0 else None
                # fresh entry only (avoid counting every bar a trade stays open)
                if prev is not None and prev.get("verdict") == r["verdict"]:
                    continue
                side, atm0, t0 = r["direction"], r.get("atm"), grid[k]
                if not atm0:
                    continue
                ps = _prem_series(chain, atm0, side)
                entry = _asof(ps, t0)
                spot0 = _asof(px, t0)
                if not entry or not spot0:
                    continue
                want_up = side == "CE"
                managed_exit, managed_reason, jx = None, "maxhold", None
                # simulate forward along the grid
                for j in range(k + 1, min(k + 1 + maxhold, len(reads))):
                    tj, rj = grid[j], reads[j]
                    pj = _asof(ps, tj)
                    if pj is None:
                        continue
                    if pj <= entry * (1 - sl_pct):
                        managed_exit, managed_reason, jx = pj, "SL", j; break
                    if pj >= entry * (1 + t1_pct):
                        managed_exit, managed_reason, jx = pj, "target", j; break
                    if rj is not None:
                        sj = rj.get("strength", 0.0)
                        if (sj > 0) != want_up and abs(sj) >= scout._TRADE_TH * 0.5:
                            managed_exit, managed_reason, jx = pj, "flow_rev", j; break
                    spotj = _asof(px, tj)
                    held = (tj - t0).total_seconds() / 60.0
                    if (spotj and held >= 2 * tf
                            and abs(spotj / spot0 - 1) * 100 < 0.05 and pj < entry):
                        managed_exit, managed_reason, jx = pj, "theta", j; break
                # max-hold index for BOTH (naive hold exits here; managed too if no rule)
                jh = min(k + maxhold, len(reads) - 1)
                prem_hold = _asof(ps, grid[jh])
                if managed_exit is None:
                    managed_exit, jx = (prem_hold if prem_hold else entry), jh
                if not prem_hold:
                    prem_hold = entry
                rows.append({
                    "date": date, "sym": sym, "t": SAMPLE_TIMES[k], "side": side,
                    "entry": entry, "managed_exit": managed_exit, "hold_exit": prem_hold,
                    "reason": managed_reason,
                    "managed_net": (managed_exit / entry - 1) * 100 - COST,
                    "hold_net": (prem_hold / entry - 1) * 100 - COST,
                    "bars_held": (jx - k) if jx is not None else maxhold,
                })
    return pd.DataFrame(rows)


def evaluate(df: pd.DataFrame, reps: int, rng, tf: int, maxhold: int) -> None:
    print(f"\nMANAGE RULES vs NAIVE HOLD — paired on fresh TRADE entries "
          f"({tf}m, max-hold {maxhold} bars, cost {COST:.0f}%)")
    print("=" * 76)
    if df.empty:
        print("  no entries."); return
    days = sorted(df.date.unique())
    print(f"  entries={len(df)}  days={len(days)} ({days[0]}..{days[-1]})  "
          f"indices={df.sym.nunique()}")
    rc = df["reason"].value_counts().to_dict()
    print(f"  managed exit reasons: {rc}")
    print(f"  avg bars held (managed): {df['bars_held'].mean():.1f} of {maxhold}")

    def stat(col):
        a = df[col].to_numpy(float)
        win = (a > 0).astype(float)
        m, lo, hi = _boot_ci(lambda x: x.mean(), a, reps=reps, rng=rng,
                             groups=df["date"].to_numpy())
        w, _, _ = _boot_ci(lambda x: x.mean(), win, reps=reps, rng=rng,
                           groups=df["date"].to_numpy())
        return m, lo, hi, 100 * w

    for name, col in (("HOLD  (naive)", "hold_net"), ("MANAGED", "managed_net")):
        m, lo, hi, w = stat(col)
        print(f"  {name:16s} mean net {m:+5.1f}% [{lo:+5.1f},{hi:+5.1f}]   win {w:4.1f}%")

    diff = (df["managed_net"] - df["hold_net"]).to_numpy(float)
    dm, dlo, dhi = _boot_ci(lambda x: x.mean(), diff, reps=reps, rng=rng,
                            groups=df["date"].to_numpy())
    v = "MANAGING HELPS" if dlo > 0 else ("managing HURTS" if dhi < 0 else "no diff (CI straddles 0)")
    print(f"\n  DIFFERENCE (managed − hold)  {dm:+.1f}% [{dlo:+.1f},{dhi:+.1f}]  => {v}")

    print("\n" + "=" * 76)
    print("READ: managing only earns its place if the DIFFERENCE CI clears 0. Both legs")
    print("being negative just confirms the arrow is a losing trade either way — manage")
    print("can cut the bleed but cannot turn a negative-EV signal positive. Range band")
    print("stays the product. Few days = wide CI; re-run as the archive grows.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", type=int, default=15)
    ap.add_argument("--maxhold", type=int, default=4, help="max bars to hold")
    ap.add_argument("--reps", type=int, default=3000)
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    days = _captured_days()
    if not days:
        print("No captured days in", LIVE_DIR); return
    print(f"reading mirror: {LIVE_DIR}  tf={args.tf}m  maxhold={args.maxhold}")
    df = harvest(days, args.tf, args.maxhold)
    evaluate(df, args.reps, rng, args.tf, args.maxhold)


if __name__ == "__main__":
    main()
