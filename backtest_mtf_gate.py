"""
backtest_mtf_gate.py — does the user's TWO-PHASE gate pay? (the decisive grade)

Phase 1 = the scout's OI/flow directional lean (CE/PE) — measured ANTI-EV standalone
          (backtest_scout: option net -2.6..-3.4bps, cost-aware hit 15-37%).
Phase 2 = price-action MULTI-TF confirmation: the lean only trades if the structure on
          5m / 15m / 60m ALIGNS with it (LTF confirmed by HTF — trend/breakout, not chop).

The thesis: each phase is weak, but the CONJUNCTION (flow lean AND full MTF alignment) is
selective → maybe the flow arrow finally clears the 3% option cost floor inside that subset.

This reuses the harvested scout_ledger (flow direction + ACTUAL ATM option net-of-cost per
checkpoint — a market-order fill at t, so NO edge-fill lookahead like the band-fade had) and
adds the Phase-2 MTF structure at each trade, then grades the aligned subset vs the rest with
a day-block bootstrap CI and an OOS temporal split. If aligned option-net CI clears 0 OOS,
the two-phase gate is real; if not, it is subtractive (fewer losses, not a profit) like every
prior veto (backtest_scout 3c/3d/3e — kept bucket still negative).

    .venv\\Scripts\\python.exe backtest_mtf_gate.py
"""
from __future__ import annotations

import datetime
import sys

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from core.constants import IST, DATA_DIR
from backtest_continuity import _boot_ci
import tradeboard as tb

LEDGER = DATA_DIR / "validation" / "scout_ledger.parquet"
UP = {"TREND_UP", "BREAKOUT_UP"}
DN = {"TREND_DOWN", "BREAKOUT_DOWN"}
SEED = 7


def _structs(sym, date, t):
    hh, mm = map(int, t.split(":"))
    as_of = datetime.datetime.combine(datetime.date.fromisoformat(date),
                                      datetime.time(hh, mm), tzinfo=IST)
    return {tf: tb._struct_full(sym, tf, date, as_of).get("struct", "n/a")
            for tf in (5, 15, 30, 60)}


def _gate(direction, s):
    """Return dict of Phase-2 gate flags for a CE/PE lean vs the 5/15/30/60m structures."""
    want, opp = (UP, DN) if direction == "CE" else (DN, UP)
    s5, s15, s30, s60 = s[5], s[15], s[30], s[60]
    CONSOL = {"CONSOLIDATION", "RANGE"}
    return {
        # ── the USER's pattern: HTF broke out in the lean dir, LTF now PAUSING (consolidation)
        #    = breakout->flag->continuation. NOT "both trending" (that excludes the pause).
        "contin60": (s60 in want) and (s15 in CONSOL),     # 1h trend/breakout + 15m consol
        "contin30": (s30 in want) and (s15 in CONSOL),     # 30m trend/breakout + 15m consol (feasible)
        "contin_x": (s60 in want or s30 in want) and (s15 in CONSOL),  # either HTF broke + 15m pause
        # STRICT full cascade: all three TFs in the lean's direction (60m rarely warm intraday)
        "strict": (s5 in want) and (s15 in want) and (s60 in want),
        # CASCADE: entry-TF (15) confirmed by HTF (60), 5m not opposing
        "cascade": (s15 in want) and (s60 in want) and (s5 not in opp),
        # PAIR 5x15: the intraday-FEASIBLE cascade — 5m confirmed by 15m (both warm by ~late
        # morning; drops the 60m requirement that is not warm until ~14:15)
        "pair515": (s5 in want) and (s15 in want),
        # 15-in-dir alone (loosest directional confirm)
        "ltf15": (s15 in want),
        # NO-CONFLICT: no TF actively fights the lean (HTF permission)
        "noconflict": (s5 not in opp) and (s15 not in opp) and (s60 not in opp),
    }


def _grade(df, col, reps, rng, label):
    sub = df.dropna(subset=[col])
    if len(sub) < 8 or sub["date"].nunique() < 2:
        print(f"     {label:<12} n={len(sub)} too few"); return
    net = sub[col].to_numpy(float)
    win = (net > 0).mean()
    mn, lo, hi = _boot_ci(lambda a: a.mean(), net, reps=reps, rng=rng,
                          groups=sub["date"].to_numpy())
    v = "EDGE" if lo > 0 else ("bleed" if hi < 0 else "—")
    print(f"     {label:<12} net {mn:+5.2f}bps [{lo:+5.2f},{hi:+5.2f}] {v:5}  "
          f"win {100*win:4.1f}%  n={len(sub):>3}")


def main():
    if not LEDGER.exists():
        print("no scout_ledger — run backtest_scout.py first."); return
    df = pd.read_parquet(LEDGER)
    tr = df[(df.is_trade == 1) & df.direction.isin(["CE", "PE"])].copy()
    print(f"scout ledger: {len(tr)} flow-lean trades, {tr.date.nunique()} days "
          f"({tr.date.min()}..{tr.date.max()})")
    print("computing MTF structure (5/15/60m) at each trade...")
    flags = {"strict": [], "cascade": [], "pair515": [], "ltf15": [], "noconflict": [],
             "contin60": [], "contin30": [], "contin_x": []}
    for _, r in tr.iterrows():
        try:
            g = _gate(r["direction"], _structs(r["sym"], r["date"], r["t"]))
        except Exception:
            g = {"strict": False, "cascade": False, "noconflict": False}
        for k in flags:
            flags[k].append(g[k])
    for k, v in flags.items():
        tr[k] = v
    rng = np.random.default_rng(SEED)

    for H in (15, 30, 60):
        col = f"opt_net{H}"
        if col not in tr.columns:
            continue
        print(f"\n  ── option net at +{H}m (net of ~3% RT; 'EDGE' only if CI clears 0) ──")
        _grade(tr, col, 4000, rng, "ALL leans")           # Phase-1 alone (the dead arrow)
        for gate in ("ltf15", "pair515", "contin30", "contin60", "contin_x"):
            g1 = tr[tr[gate]]
            _grade(g1, col, 4000, rng, f"+{gate}")          # Phase-1 AND Phase-2
        sel = " · ".join(f"{g} {int(tr[g].sum())}/{len(tr)} ({100*tr[g].mean():.0f}%)"
                         for g in ("contin30", "contin60", "contin_x", "pair515"))
        print(f"     selectivity: {sel}")

    # OOS temporal split on the strongest gate at +15m
    print("\n  ── OOS TEMPORAL SPLIT (cascade gate, +15m, train 1st-half / test 2nd-half) ──")
    days = sorted(tr.date.unique())
    cut = days[len(days) // 2]
    for name, mask in (("ALL leans", tr.index), ("+contin30", tr[tr.contin30].index),
                       ("+contin_x", tr[tr.contin_x].index)):
        g = tr.loc[mask].dropna(subset=["opt_net15"])
        for split, sub in (("train", g[g.date < cut]), ("test", g[g.date >= cut])):
            if len(sub) < 8 or sub.date.nunique() < 2:
                print(f"     {name:<10} {split:<5} n={len(sub)} too few"); continue
            net = sub["opt_net15"].to_numpy(float)
            mn, lo, hi = _boot_ci(lambda a: a.mean(), net, reps=4000, rng=rng,
                                  groups=sub["date"].to_numpy())
            v = "EDGE" if lo > 0 else ("bleed" if hi < 0 else "—")
            print(f"     {name:<10} {split:<5} net {mn:+5.2f}bps [{lo:+5.2f},{hi:+5.2f}] {v}  "
                  f"n={len(sub)}")

    print("\n  READ: if '+cascade'/'+strict' net-CI clears 0 (esp. OOS test), the two-phase")
    print("  gate is a REAL edge. If it only shrinks the loss (kept still <0), it is")
    print("  SUBTRACTIVE — a don't-trade filter, not a buy signal (matches every prior veto).")


if __name__ == "__main__":
    main()
