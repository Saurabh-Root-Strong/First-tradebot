"""backtest_btst_entry.py — does the BTST edge survive a REAL 15:15-15:30 entry?

The backtest prices entry at the 15:30 CLOSE (clr and overnight both need the final close).
But the trade is executed in the 15:15-15:30 window, which hides two effects:
  1. CONFIRMATION RISK — clr>=0.66 is only final at 15:30. A position taken at 15:15 acts on
     a still-forming signal that can fade below (or firm above) the threshold by the close.
  2. ENTRY SLIPPAGE — on a strong close price is climbing into 15:30, so a 15:15 fill != the
     close price the backtest assumes.

This measures both on the full 2yr history: entry at 15:15 / 15:20 / 15:25 / close vs the
same next-day 09:30-ish exit (next open, the backtest's own proxy), plus how often the
15:15 forming-clr disagrees with the final clr.

    .venv\\Scripts\\python.exe backtest_btst_entry.py
"""
from __future__ import annotations

import datetime as dt
import glob
import sys

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CLR_TH = 0.66
COST_BPS = 3.0
# reuse btst_signal's own file naming (FY short-name -> NSE_{FY}_INDEX_*.parquet)
from btst_signal import FY, DAILY, MIN5
IDX = list(FY)
ENTRY_BARS = ["15:10", "15:15", "15:20", "15:25"]  # 5min bar CLOSES; + "close" = daily close
DECISION_BARS = ["15:10", "15:15"]                 # times you'd commit on a forming clr


def _daily(idx):
    f = glob.glob(DAILY.format(FY[idx]))
    if not f:
        return None
    d = pd.read_parquet(f[0]).copy()
    d["date"] = pd.to_datetime(d[[c for c in d.columns if "date" in c.lower() or c == "ts"][0]]).dt.date
    d = d.sort_values("date").reset_index(drop=True)
    rng = (d["high"] - d["low"]).replace(0, np.nan)
    d["clr"] = (d["close"] - d["low"]) / rng
    d["next_open"] = d["open"].shift(-1)
    return d[["date", "open", "high", "low", "close", "clr", "next_open"]]


def _intraday_prices(idx):
    """Per date: the 5min close at 15:15/20/25, and running high/low/close THROUGH 15:15
    (for the forming-clr confirmation test)."""
    f = glob.glob(MIN5.format(FY[idx]))
    if not f:
        return None
    m = pd.read_parquet(f[0]).copy()
    m["ts"] = pd.to_datetime(m[[c for c in m.columns if c in ("ts",) or "time" in c.lower() or "date" in c.lower()][0]])
    m["date"] = m["ts"].dt.date
    m["hm"] = m["ts"].dt.strftime("%H:%M")
    out = {}
    for date, g in m.groupby("date"):
        g = g.sort_values("ts")
        row = {}
        # BARS ARE BAR-START labelled: the bar "15:10" spans [15:10,15:15). So the price/clr
        # KNOWN AT clock time T uses only bars that have fully CLOSED, i.e. hm < T (the bar
        # ending exactly at T is the one labelled T-5m). hm <= T would leak 5 minutes ahead.
        for T in set(ENTRY_BARS) | set(DECISION_BARS):
            upto = g[g["hm"] < T]              # strictly before -> bars completed by clock T
            if not len(upto):
                continue
            px = float(upto["close"].iloc[-1])           # last close = price AS OF T
            hi, lo = upto["high"].max(), upto["low"].min()
            row[T] = px                                  # entry price at T
            row[f"clr_{T.replace(':','')}"] = (px - lo) / (hi - lo) if hi > lo else np.nan
        out[date] = row
    return out


def run():
    print("=" * 92)
    print("BTST ENTRY-TIMING — does the edge survive a real 15:15-15:30 fill vs the 15:30 close?")
    print("=" * 92)
    allrows = []
    for idx in IDX:
        d = _daily(idx)
        intr = _intraday_prices(idx)
        if d is None or intr is None:
            print(f"  {idx}: missing daily/5min data — skip")
            continue
        for r in d.itertuples():
            # collect EVERY day with intraday + a next-open (both universes need the fade set)
            if not (r.clr == r.clr and r.next_open == r.next_open):
                continue
            px = intr.get(r.date)
            if px is None:
                continue
            allrows.append({"idx": idx, "date": r.date, "clr": r.clr,
                            "close": r.close, "next_open": r.next_open,
                            **{f"clr_{db.replace(':','')}": px.get(f"clr_{db.replace(':','')}", np.nan)
                               for db in DECISION_BARS},
                            **{hm: px.get(hm, np.nan) for hm in ENTRY_BARS}})
    dfall = pd.DataFrame(allrows).dropna(subset=["clr_1515"])
    if dfall.empty:
        print("  no days with intraday data found"); return
    df = dfall[dfall["clr"] >= CLR_TH]        # backtest universe = FINAL strong closes
    print(f"\n  days with 5min intraday: n={len(dfall)}  ({dfall.date.min()}..{dfall.date.max()}); "
          f"final-strong (clr>=0.66) n={len(df)}\n")

    # 1) ENTRY SLIPPAGE — where does price sit at 15:15/20/25 vs the 15:30 close?
    print("  1) ENTRY SLIPPAGE vs the 15:30 close (strong-close days; + = price rose INTO close):")
    for hm in ENTRY_BARS:
        sub = df.dropna(subset=[hm])
        slip = (df["close"] / df[hm] - 1.0) * 1e4    # bps the close is ABOVE the hm price
        slip = slip.dropna()
        print(f"     {hm}: close is {slip.mean():+5.1f} bps above the {hm} price "
              f"(median {slip.median():+4.1f}, n={len(slip)}) "
              f"-> entering at {hm} fills {'CHEAPER (favourable long)' if slip.mean()>0 else 'dearer'}")

    # 2) OVERNIGHT EDGE by entry time (same next-open exit, cost applied)
    print("\n  2) OVERNIGHT NET bps by ENTRY time (exit = next open, -3bps cost):")
    for hm in ENTRY_BARS + ["close"]:
        ep = df["close"] if hm == "close" else df[hm]
        net = (df["next_open"] / ep - 1.0) * 1e4 - COST_BPS
        net = net.dropna()
        print(f"     enter {hm:6s}: mean {net.mean():+6.2f} bps  win {100*(net>0).mean():4.1f}%  n={len(net)}")

    # apples-to-apples: close-entry on the SAME intraday sample as the 15:15 rows
    net_close_same = ((df["next_open"] / df["close"] - 1.0) * 1e4 - COST_BPS).dropna()
    print(f"     (same-sample close entry, n={len(net_close_same)}: mean "
          f"{net_close_same.mean():+.2f} bps -> 15:15 costs "
          f"~{net_close_same.mean() - ((df['next_open']/df['15:15']-1)*1e4-COST_BPS).dropna().mean():+.1f} bps vs close)")

    # 3) CONFIRMATION + FADE RISK at each decision time (15:10 vs 15:15)
    print("\n  3) DECIDE EARLY? confirmation + fade risk by decision minute:")
    for db in DECISION_BARS:
        col = f"clr_{db.replace(':','')}"
        entry_px = db                              # enter at that same bar's price
        setby = 100 * (df[col] >= CLR_TH).mean()   # of FINAL strong, already strong at db
        drift = (df["clr"] - df[col]).mean()
        look = dfall[dfall[col] >= CLR_TH]         # execution universe: strong AT db
        held = look["clr"] >= CLR_TH
        faded = look[~held]
        fn = ((faded["next_open"] / faded[entry_px] - 1.0) * 1e4 - COST_BPS).dropna() if len(faded) else pd.Series([], dtype=float)
        commit = ((look["next_open"] / look[entry_px] - 1.0) * 1e4 - COST_BPS).dropna()
        print(f"     ── decide at {db} ──")
        print(f"        set-by-{db}: {setby:.0f}% of final-strong already >=0.66  (drift {drift:+.3f})")
        print(f"        strong@{db} n={len(look)} -> HOLD {100*held.mean():.1f}%  FADE {100*(~held).mean():.1f}% "
              f"({len(faded)}d, faded overnight {fn.mean():+.1f}bps)")
        print(f"        COMMIT-AT-{db} (no wait): {commit.mean():+.1f} bps  win {100*(commit>0).mean():.0f}%  n={len(commit)}")
    print("\n  READ: earlier decision = more fade + weaker confirmation. Safe iff commit-at-time")
    print("  still clears +10-13 bps AND fade cost stays small.")


if __name__ == "__main__":
    run()
