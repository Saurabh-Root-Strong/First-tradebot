"""
backtest_premium_action.py — trade the OPTION's own price action, not the index's.

THE USER'S POINT, and it is a good one: the arrow decides CE/PE from INDEX signals (OI,
volume, consensus) and then manages the trade with a fixed SL / target / band / 90m cap. It
never looks at the PREMIUM's own candles -- the very series you are actually long. Standard
price-action discipline (trail the stop, exit when structure breaks, only enter on
confirmation) is applied to nothing.

THE THEORETICAL WORRY, stated up front so the test can refute it: an option premium is not an
independent series. P = f(spot, IV, t), so a premium candle is mostly
    delta x (spot move)  +  vega x (IV change)  -  theta.
"Price action on the premium" is therefore largely the INDEX's price action, leveraged and
decaying -- not new information. The one thing it holds that spot does not is demand for the
CONTRACT itself (IV expanding, OI building) -- and that IS testable.

So: hold the arrow's ENTRIES fixed (same signal, same strike, same minute) and replace the
exit -- and optionally gate the entry -- with rules read off the premium series, its OI and
its IV. If the arrow's problem is bad management, this fixes it. If the arrow's problem is
that the ENTRY has no directional edge, nothing here will save it, and we will see that too.

Every variant pays the same ~3% option round-trip and is capped at 90m / the bell.

    .venv\\Scripts\\python.exe backtest_premium_action.py
"""
from __future__ import annotations

import datetime as dt
import glob
import os
import sys

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from core.constants import IST, LIVE_DIR
from core.mirror_io import read_mirror
import dashboard as D

RT_COST = 3.0          # % of premium, round trip (the same wall the arrow always pays)
CAP_MIN = 90
BELL = dt.time(15, 30)


def chain_days():
    return sorted({os.path.basename(p).split("_")[0]
                   for p in glob.glob(str(LIVE_DIR / "*_chain_snapshots.parquet"))
                   if os.path.getsize(p) > 300_000})


def leg_series(ch, sym, strike, side, t0):
    """The premium series for ONE leg from entry to the 90m cap / bell: ltp + its OI and IV."""
    g = ch[(ch["symbol"] == sym) & (ch["strike"] == strike) & (ch["side"] == side)]
    if g.empty:
        return None
    end = min(t0 + pd.Timedelta(minutes=CAP_MIN),
              pd.Timestamp(f"{t0.date()} {BELL}", tz=IST))
    g = g[(g["ts"] >= t0) & (g["ts"] <= end)].sort_values("ts")
    return g if len(g) >= 4 else None


def candles(g, minutes=5):
    """Premium OHLC candles from the 30s ltp series -- the thing the user is asking us to read."""
    s = g.set_index("ts")["ltp"].astype(float)
    o = s.resample(f"{minutes}min").agg(["first", "max", "min", "last"]).dropna()
    o.columns = ["open", "high", "low", "close"]
    return o


def exit_asis(ep):
    """What the arrow ACTUALLY did (SL / target / band / flip / timeout) -- the incumbent."""
    return ep.get("pnl")


def exit_trail(g, pct):
    """Trail the stop `pct`% below the running MAX premium. The simplest price-action exit."""
    p = g["ltp"].astype(float).to_numpy()
    entry = p[0]
    peak = p[0]
    for x in p[1:]:
        peak = max(peak, x)
        if x <= peak * (1 - pct / 100.0):
            return (x / entry - 1) * 100 - RT_COST
    return (p[-1] / entry - 1) * 100 - RT_COST


def exit_structure(g, minutes=5):
    """STRUCTURE BREAK: exit when a premium candle CLOSES below the prior candle's LOW.
    This is the textbook price-action stop, applied to the series you are actually long."""
    c = candles(g, minutes)
    if len(c) < 2:
        return exit_trail(g, 100)
    entry = float(g["ltp"].iloc[0])
    for i in range(1, len(c)):
        if c["close"].iloc[i] < c["low"].iloc[i - 1]:
            return (c["close"].iloc[i] / entry - 1) * 100 - RT_COST
    return (float(g["ltp"].iloc[-1]) / entry - 1) * 100 - RT_COST


def entry_confirmed(g, minutes=5):
    """ENTRY GATE: is the PREMIUM itself in motion? Require the first candle to close UP
    (the contract is being bid, not just the index ticking). Costs one candle of lag."""
    c = candles(g, minutes)
    if len(c) < 2:
        return None
    if c["close"].iloc[0] <= c["open"].iloc[0]:
        return False                      # premium not being bid -> skip the trade
    return True


def entry_oi_iv(g):
    """The one thing premium holds that SPOT does not: demand for the CONTRACT.
    Require OI to be BUILDING (oich > 0) and IV not collapsing at entry."""
    h = g.iloc[:min(6, len(g))]           # first ~3 minutes
    oich = pd.to_numeric(h.get("oich"), errors="coerce").fillna(0).sum()
    iv = pd.to_numeric(h.get("iv"), errors="coerce").dropna()
    iv_ok = True if len(iv) < 2 else iv.iloc[-1] >= iv.iloc[0]
    return bool(oich > 0 and iv_ok)


def run():
    days = chain_days()
    rows = []
    for d in days:
        ch = read_mirror("chain_snapshots", d)
        if ch is None or ch.empty:
            continue
        ch["ts"] = pd.to_datetime(ch["ts"])
        try:
            _, closed = D._scout_episodes(
                d, as_of=dt.datetime.combine(dt.date.fromisoformat(d), dt.time(23, 59), tzinfo=IST))
        except Exception:
            continue
        for ep in closed:
            if not (ep.get("strike") and ep.get("dir") and ep.get("entry")):
                continue
            g = leg_series(ch, ep["sym"], int(ep["strike"]), ep["dir"], ep["open_ts"])
            if g is None:
                continue
            conf = entry_confirmed(g)
            rows.append({
                "day": d, "sym": ep["sym"], "outcome": ep["outcome"],
                "asis": exit_asis(ep),
                "trail15": exit_trail(g, 15), "trail25": exit_trail(g, 25),
                "struct5": exit_structure(g, 5), "struct15": exit_structure(g, 15),
                "confirmed": conf, "oi_iv": entry_oi_iv(g),
            })
    df = pd.DataFrame(rows)
    df = df[df["asis"].notna()]
    print("=" * 92)
    print(f"THE ARROW, MANAGED BY THE PREMIUM'S OWN PRICE ACTION   (n={len(df)} legs, "
          f"{df.day.nunique()} days)")
    print("  same entries, same strikes, same minute. only the EXIT (and the entry GATE) change.")
    print("=" * 92)
    print(f"  {'variant':46s} {'n':>4s} {'mean %':>8s} {'win%':>6s} {'worst':>8s}")
    print("  " + "-" * 78)

    def show(nm, v):
        v = pd.to_numeric(pd.Series(v), errors="coerce").dropna().to_numpy()
        if len(v) < 20:
            print(f"  {nm:46s} {len(v):>4d}   too few")
            return
        print(f"  {nm:46s} {len(v):>4d} {v.mean():>+8.1f} {100*(v>0).mean():>5.0f}% {v.min():>+8.1f}")

    show("AS-IS (SL/target/band/90m) -- the incumbent", df["asis"])
    print("  " + "-" * 78)
    show("premium TRAIL 15% off the peak", df["trail15"])
    show("premium TRAIL 25% off the peak", df["trail25"])
    show("premium STRUCTURE break (5m candle < prior low)", df["struct5"])
    show("premium STRUCTURE break (15m candle < prior low)", df["struct15"])
    print("  " + "-" * 78)
    c = df[df["confirmed"] == True]
    show("ENTRY GATE: 1st premium candle closes UP", c["asis"])
    show("   + structure exit", c["struct5"])
    show("   + trail 25%", c["trail25"])
    o = df[df["oi_iv"] == True]
    show("ENTRY GATE: option OI building + IV not falling", o["asis"])
    show("   + structure exit", o["struct5"])
    both = df[(df["confirmed"] == True) & (df["oi_iv"] == True)]
    show("BOTH gates + structure exit", both["struct5"])
    show("BOTH gates + trail 25%", both["trail25"])
    print("  " + "-" * 78)
    print(f"  every variant pays the same {RT_COST:.0f}% option round-trip.")
    print("  A better EXIT can only stop the bleeding. It cannot manufacture a directional")
    print("  edge the ENTRY does not have. Watch whether ANY variant crosses zero.")


if __name__ == "__main__":
    run()
