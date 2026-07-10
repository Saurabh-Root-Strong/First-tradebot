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
ENTRY_BARS = ["15:15", "15:20", "15:25"]     # 5min bar CLOSES; + "close" = the daily close


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
        for hm in ENTRY_BARS:
            b = g[g["hm"] == hm]
            row[hm] = float(b["close"].iloc[0]) if len(b) else np.nan
        upto = g[g["hm"] <= "15:15"]           # session so far at 15:15
        if len(upto):
            hi, lo = upto["high"].max(), upto["low"].min()
            px = float(upto["close"].iloc[-1])
            row["clr_1515"] = (px - lo) / (hi - lo) if hi > lo else np.nan
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
                            "clr_1515": px.get("clr_1515", np.nan),
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

    # 3) CONFIRMATION — of FINAL strong, how much was already set at 15:15?
    print("\n  3a) SIGNAL SET-BY-15:15 (of FINAL strong closes):")
    strong_1515 = df["clr_1515"] >= CLR_TH
    print(f"     {100*strong_1515.mean():.0f}% were already clr>=0.66 at 15:15; "
          f"mean clr drift 15:15->close {(df['clr']-df['clr_1515']).mean():+.3f}")

    # 3b) THE EXECUTION UNIVERSE — you decide at 15:15. FADE RATE + P&L on faded trades.
    print("\n  3b) FADE RISK (the real 15:15 decision): of days STRONG AT 15:15, how many hold?")
    look_strong = dfall[dfall["clr_1515"] >= CLR_TH]
    held = look_strong["clr"] >= CLR_TH
    faded = look_strong[~held]
    print(f"     strong@15:15 n={len(look_strong)}  ->  held to close {100*held.mean():.1f}%  "
          f"FADED below 0.66 {100*(~held).mean():.1f}% ({len(faded)} days)")
    if len(faded):
        fn = ((faded["next_open"] / faded["15:15"] - 1.0) * 1e4 - COST_BPS).dropna()
        verdict = ("positive (fade is FREE)" if fn.mean() > 2 else
                   "~breakeven (fade is cheap)" if fn.mean() > -3 else "a real drag")
        print(f"     overnight on the FADED (wrongly-entered) trades: mean {fn.mean():+.1f} bps, "
              f"win {100*(fn>0).mean():.0f}% -> {verdict}")
    # blended: commit at 15:15 to EVERY strong@15:15 signal (held + faded), exit next open
    commit = ((look_strong["next_open"] / look_strong["15:15"] - 1.0) * 1e4 - COST_BPS).dropna()
    print(f"     COMMIT-AT-15:15 policy (take all strong@15:15, no confirmation): "
          f"mean {commit.mean():+.1f} bps  win {100*(commit>0).mean():.0f}%  n={len(commit)}")
    print("\n  READ: the 15:15-15:30 window is safe iff (a) the signal is mostly set by 15:15,")
    print("  (b) faded trades don't bleed, and (c) commit-at-15:15 still clears +10-13 bps.")


if __name__ == "__main__":
    run()
