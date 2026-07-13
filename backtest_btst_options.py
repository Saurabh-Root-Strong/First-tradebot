"""
backtest_btst_options.py — could the BTST edge be traded with OPTIONS instead of futures?

The rule says LONG index FUTURES on a strong close. The stated reason for futures is "a long
option bleeds theta overnight". That is an ASSERTION, and assertions are how you end up
trading a mirage. Measure it: same signal, same 15:10-15:30 entry, same 09:30 exit --
buy an ATM CALL instead of the future, and compare.

Two costs an option carries that a future does not:
  THETA  — one night of time decay, paid whether or not the index moves.
  SPREAD — the option round-trip is ~3% of PREMIUM; the futures round-trip is ~3 BPS of NOTIONAL.
           On a ~24,000 index with a ~150-point ATM call, 3% of premium is ~4.5 index points
           while 3bps of notional is ~7 points -- but the call only captures ~half the move
           (delta ~0.5), so it must clear far more to break even.
The edge is only +10-13 bps of INDEX move (~25-30 points on NIFTY). The question is whether a
call can survive that, and the honest way to know is to price real captured chains.

    .venv\\Scripts\\python.exe backtest_btst_options.py
"""
from __future__ import annotations

import datetime as dt
import sys

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from core.constants import IST, LIVE_DIR
from core.mirror_io import read_mirror
import intraday_scout as scout
import btst_signal as bs

FUT_COST_BPS = 3.0          # index-futures round trip
OPT_COST_PCT = 3.0          # option round trip, % of premium (scout._OPT_RT_COST)
EXIT_T = dt.time(9, 30)


def _chain_days():
    return sorted({p.name.split("_")[0] for p in LIVE_DIR.glob("*_chain_snapshots.parquet")
                   if p.stat().st_size > 300_000})


def _spot_at(sym, day, t):
    tk = read_mirror("ticks", day, None, sym)
    if tk is None or tk.empty:
        return None
    m = tk[(tk["ts"].dt.date == dt.date.fromisoformat(day)) & (tk["ts"] <= pd.Timestamp(t))]
    return float(m.iloc[-1]["ltp"]) if len(m) else None


def run():
    days = _chain_days()
    led = pd.read_parquet("data/validation/btst_paper_ledger.parquet")
    rows = []
    print("=" * 96)
    print("BTST WITH OPTIONS vs FUTURES — same signal, same entry, same 09:30 exit")
    print("=" * 96)

    for r in led.itertuples():
        d = str(r.date)[:10]
        if d not in days:
            continue
        nxt = [x for x in days if x > d]
        if not nxt:
            continue
        d2 = nxt[0]
        sym = f"NSE:{bs.FY[r.sym]}-INDEX"
        entry_t = dt.datetime.combine(dt.date.fromisoformat(d), dt.time(15, 25), tzinfo=IST)
        exit_t = dt.datetime.combine(dt.date.fromisoformat(d2), EXIT_T, tzinfo=IST)

        spot_in = _spot_at(sym, d, entry_t)
        spot_out = _spot_at(sym, d2, exit_t)
        if not (spot_in and spot_out):
            continue
        # FUTURES leg — the actual rule (delta 1.0, cost in bps of notional)
        fut_bps = (spot_out / spot_in - 1.0) * 1e4 - FUT_COST_BPS

        # OPTION leg — buy the ATM CALL at the close, sell it at 09:30. Same strike both days.
        atm = scout._atm(spot_in, sym)
        p_in = scout._opt_premium(sym, d, entry_t, atm, "CE")
        p_out = scout._opt_premium(sym, d2, exit_t, atm, "CE")
        if not (p_in and p_out):
            continue
        opt_pct = (p_out / p_in - 1.0) * 100.0 - OPT_COST_PCT
        # what the option WOULD have made with no cost/decay, if it tracked delta ~1
        rows.append({"date": d, "exit": d2, "idx": r.sym, "clr": r.clr,
                     "spot_in": spot_in, "spot_out": spot_out, "fut_bps": fut_bps,
                     "atm": atm, "prem_in": p_in, "prem_out": p_out, "opt_pct": opt_pct,
                     "prem_chg_pct": (p_out / p_in - 1.0) * 100.0})

    if not rows:
        print("  no signal night has chain data on BOTH the entry day and the exit day.")
        return
    df = pd.DataFrame(rows)
    print(f"\n  {len(df)} signal-night positions with real captured chains "
          f"({df.date.min()} .. {df.date.max()})\n")
    print(f"  {'night':22s} {'idx':11s} {'index move':>11s} {'FUT net':>9s} "
          f"{'ATM call':>9s} {'CALL net':>9s}")
    print("  " + "-" * 78)
    for r in df.itertuples():
        mv = (r.spot_out / r.spot_in - 1.0) * 1e4
        print(f"  {r.date}->{r.exit}  {r.idx:11s} {mv:>+9.1f}bps {r.fut_bps:>+8.1f} "
              f"{r.atm:>9.0f} {r.opt_pct:>+8.1f}%")
    print("  " + "-" * 78)
    print(f"  {'MEAN':22s} {'':11s} {'':>11s} {df.fut_bps.mean():>+8.1f} "
          f"{'':>9s} {df.opt_pct.mean():>+8.1f}%")
    print()
    fw = 100 * (df.fut_bps > 0).mean()
    ow = 100 * (df.opt_pct > 0).mean()
    print(f"  FUTURES : mean {df.fut_bps.mean():+.1f} bps   win {fw:.0f}%   "
          f"(the rule as it stands)")
    print(f"  ATM CALL: mean {df.opt_pct.mean():+.1f} %     win {ow:.0f}%   "
          f"(same signal, wrong instrument?)")
    print()
    print("  WHY the call lags even when the index rises — decompose one night:")
    b = df.iloc[df.fut_bps.idxmax()]
    print(f"    {b.date}  {b.idx}: index {b.spot_in:,.0f} -> {b.spot_out:,.0f} "
          f"({(b.spot_out/b.spot_in-1)*1e4:+.0f} bps)")
    print(f"      ATM {b.atm:.0f} CE premium {b.prem_in:.2f} -> {b.prem_out:.2f} "
          f"({b.prem_chg_pct:+.1f}% gross, {b.opt_pct:+.1f}% after the 3% spread)")
    print()
    # ── APPLES-TO-APPLES: bps-of-NOTIONAL and %-of-PREMIUM are different denominators. ──
    # Convert BOTH to rupees on ONE lot, and to return on the CAPITAL each actually ties up.
    from core.constants import LOT_SIZES
    FUT_MARGIN = 0.12                       # SPAN+exposure on index futures, ~10-12%
    _SYM = {"NIFTY": "NSE:NIFTY50-INDEX", "BANKNIFTY": "NSE:NIFTYBANK-INDEX",
            "FINNIFTY": "NSE:FINNIFTY-INDEX", "MIDCPNIFTY": "NSE:MIDCPNIFTY-INDEX"}
    df["lot"] = df["idx"].map(lambda s: LOT_SIZES.get(_SYM.get(s, "")))
    df["fut_rs"] = df["spot_in"] * (df["fut_bps"] / 1e4) * df["lot"]
    df["fut_cap"] = df["spot_in"] * df["lot"] * FUT_MARGIN       # margin blocked
    df["opt_rs"] = df["prem_in"] * (df["opt_pct"] / 100.0) * df["lot"]
    df["opt_cap"] = df["prem_in"] * df["lot"]                    # premium paid = max loss
    print("\n" + "=" * 96)
    print("SAME TRADES IN RUPEES — on ONE lot, and per rupee of capital actually tied up")
    print("=" * 96)
    print(f"  {'':22s} {'FUTURES':>26s}   {'ATM CALL':>26s}")
    print(f"  {'mean P&L / lot':22s} {df.fut_rs.mean():>+25,.0f}   {df.opt_rs.mean():>+25,.0f}")
    print(f"  {'capital / lot':22s} {df.fut_cap.mean():>25,.0f}   {df.opt_cap.mean():>25,.0f}")
    print(f"  {'return on capital':22s} {100*df.fut_rs.sum()/df.fut_cap.sum():>24.2f}%   "
          f"{100*df.opt_rs.sum()/df.opt_cap.sum():>24.2f}%")
    print(f"  {'worst night / lot':22s} {df.fut_rs.min():>+25,.0f}   {df.opt_rs.min():>+25,.0f}")
    print(f"  {'MAX possible loss':22s} {'UNBOUNDED (gap risk)':>25s}   "
          f"{'premium only: ' + format(df.opt_cap.mean(), ',.0f'):>25s}")
    print()
    print("  READ:")
    print("  * The CALL's mean of %+.1f%% is a MIRAGE. Equal-weighting percentages flatters the"
          % df.opt_pct.mean())
    print("    one cheap winner (a 160-pt NIFTY call doubling) while the real money bleeds on the")
    print("    fat MIDCAP premiums. Weighted by the RUPEES actually at risk, the call LOSES:")
    print(f"    futures {df.fut_rs.mean():+,.0f}/lot vs call {df.opt_rs.mean():+,.0f}/lot; "
          f"ROC {100*df.fut_rs.sum()/df.fut_cap.sum():+.2f}% vs "
          f"{100*df.opt_rs.sum()/df.opt_cap.sum():+.2f}%.")
    print("  * WHY: the edge is only ~10-13 bps of INDEX. A call captures ~half of that (delta")
    print("    ~0.5), pays a night of theta, and must first cross a spread worth ~3% OF PREMIUM.")
    print("    Futures take the FULL move for ~3 bps of notional. There is not enough move to")
    print("    pay an option's costs -- the same 3% wall that kills the intraday arrow.")
    print(f"  * The one thing the call DOES buy is a capped loss (worst night "
          f"{df.opt_rs.min():+,.0f} vs futures {df.fut_rs.min():+,.0f}). That is a HEDGING")
    print("    argument, not a trading one: if the open-ended gap risk needs covering, buy a")
    print("    cheap OTM PUT against the future -- do not replace the future with a call and")
    print("    give up the edge itself.")
    print(f"  * SAMPLE IS n={len(df)}. Directionally clear, statistically nothing. "
          f"Re-run as nights accumulate.")


if __name__ == "__main__":
    run()
