"""
compare_lookback.py — A/B the two lookback configs on the SAME captured live days, so Friday's
"which is more accurate" question gets a fair, apples-to-apples answer (not a code-swap-and-guess).

  BASELINE 20/40  vs  SET C 40/60

Grades the scout PA ledger (15m x 60m, guards on) over every captured chain-day under BOTH
configs and prints them side by side: trades, win%, index-R, ex-jackpot net, option Rs.
Reads the LIVE captured tick data (the faithful environment), same as the board.

    .venv\\Scripts\\python.exe compare_lookback.py
"""
from __future__ import annotations
import datetime, glob, os, sys
import numpy as np, pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from core.constants import IST, DATA_DIR

JACKPOTS = ("2026-07-04", "2026-07-08")   # exclude to see the baseline, not the tail


def _days():
    out = []
    for p in glob.glob(str(DATA_DIR / "intraday" / "live" / "*_chain_snapshots.parquet")):
        if os.path.getsize(p) > 1024:
            s = os.path.basename(p).split("_")[0]
            try:
                datetime.date.fromisoformat(s); out.append(s)
            except ValueError:
                pass
    return sorted(set(out))


def _grade(struct_lb: int, sl_win: int, days) -> dict:
    # set the env BEFORE importing tradeboard fresh so the module-level constants pick it up
    os.environ["TRADEBOT_STRUCT_LB"] = str(struct_lb)
    os.environ["TRADEBOT_SL_WIN"] = str(sl_win)
    for m in [k for k in list(sys.modules) if k == "tradeboard"]:
        del sys.modules[m]
    import tradeboard as tb
    assert tb._STRUCT_LB == struct_lb and tb._SL_WIN == sl_win, "env override didn't take"
    rows, sup = [], 0
    for d in days:
        dd = datetime.date.fromisoformat(d)
        as_of = datetime.datetime.combine(dd, datetime.time(15, 35), tzinfo=IST)
        try:
            L = tb.scout_pa_ledger(d, as_of, 15, 60)
        except Exception:
            continue
        sup += L.get("suppressed", 0)
        for r in L["closed"]:
            if r.get("opt_rs") is not None:
                r["date"] = d; rows.append(r)
    df = pd.DataFrame(rows)
    if df.empty:
        return {"n": 0}
    w = df[df.opt_rs > 0]; l = df[df.opt_rs <= 0]
    jack = df[df.date.isin(JACKPOTS)].opt_rs.sum()
    dl = df.groupby("date").opt_rs.sum()
    return {"n": len(df), "sup": sup, "win": 100 * len(w) / len(df),
            "idxR": df.rmult.mean(), "net": df.opt_rs.sum(),
            "exjack": df.opt_rs.sum() - jack, "jack": jack,
            "payoff": abs(w.opt_rs.mean() / l.opt_rs.mean()) if len(l) else float("nan"),
            "green": int((dl > 0).sum()), "days": len(dl)}


def main():
    days = _days()
    print(f"A/B LOOKBACK — {len(days)} captured live days ({days[0]}..{days[-1]})")
    print("=" * 78)
    a = _grade(20, 40, days); b = _grade(40, 60, days)
    hdr = f'{"config":<14}{"n":>5}{"sup":>5}{"win%":>6}{"idxR":>8}{"net":>9}{"ex-jack":>9}{"payoff":>8}{"green":>7}'
    print(hdr); print("-" * 78)
    for nm, s in (("BASELINE 20/40", a), ("SET C 40/60", b)):
        if not s.get("n"):
            print(f"{nm:<14} no trades"); continue
        print(f'{nm:<14}{s["n"]:>5}{s["sup"]:>5}{s["win"]:>5.0f}%{s["idxR"]:>+8.3f}'
              f'{s["net"]:>+9,.0f}{s["exjack"]:>+9,.0f}{s["payoff"]:>7.2f}x{s["green"]:>4}/{s["days"]}')
    print("=" * 78)
    print("ACCURACY = win% + idxR + ex-jack (the tail-free baseline). Higher/greener wins.")
    print("The winner is the config to keep; flip live with TRADEBOT_STRUCT_LB / TRADEBOT_SL_WIN.")


if __name__ == "__main__":
    main()
