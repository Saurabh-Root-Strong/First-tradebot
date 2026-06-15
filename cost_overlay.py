"""
cost_overlay.py  —  The decisive test: does the backtest edge survive real costs?

The backtest is LINEAR in the underlying, so it is really a STOCK-FUTURES backtest
(no theta). That makes it the OPTIMISTIC CEILING for the live option-buying engine:
if the edge dies here, it is deader as options (theta + wider spreads).

Cost-in-R is not guessed — it is computed per trade from the ledger:
    risk_frac = |entry_px - sl| / entry_px         (≈ 1.5*ATR/price, the stop in %)
    cost_R    = (round_trip_bps / 1e4) / risk_frac
    net_r     = r - cost_R
Tight stops (small risk_frac) amplify costs — that is exactly where this strategy
lives (median stop ≈ 0.41% of price).

Breakeven cost for a segment (bps where mean net R = 0):
    c* = mean(r) / mean(1/risk_frac) * 1e4
Compare c* to realistic stock-futures round-trip (~5-9 bps: STT 2 + slippage 2-5 +
txn/stamp/brokerage ~1-2). Options are strictly worse (add theta + bid-ask).

  .venv\\Scripts\\python.exe cost_overlay.py
"""
from __future__ import annotations
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

LEDGER = Path("data") / "backtest" / "trades.parquet"
TIERS  = [0, 2, 4, 6, 10]          # round-trip bps to evaluate


def load() -> pd.DataFrame:
    df = pd.read_parquet(LEDGER)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"])
    df["year"] = df["entry_ts"].dt.year
    df["risk_frac"] = (df["entry_px"] - df["sl"]).abs() / df["entry_px"]
    df = df[df["risk_frac"] > 0].copy()
    df["inv_rf"] = 1.0 / df["risk_frac"]
    return df


def net_exp(r: pd.Series, inv_rf: pd.Series, bps: float) -> float:
    return float((r - (bps / 1e4) * inv_rf).mean())


def breakeven_bps(r: pd.Series, inv_rf: pd.Series) -> float:
    m = inv_rf.mean()
    return float(r.mean() / m * 1e4) if m > 0 else float("nan")


def row(name: str, g: pd.DataFrame) -> dict:
    r, inv = g["r"], g["inv_rf"]
    d = {"segment": name, "trades": len(g), "gross_R": round(r.mean(), 3)}
    for b in TIERS:
        d[f"{b}bps"] = round(net_exp(r, inv, b), 3)
    d["breakeven_bps"] = round(breakeven_bps(r, inv), 2)
    return d


def show(title: str, rows: list[dict]) -> None:
    print("\n" + "=" * 88)
    print(f"  {title}")
    print("=" * 88)
    cols = ["segment", "trades", "gross_R"] + [f"{b}bps" for b in TIERS] + ["breakeven_bps"]
    w = {c: max(len(c), max(len(str(x.get(c, ""))) for x in rows)) for c in cols}
    print("  " + "  ".join(c.ljust(w[c]) for c in cols))
    print("  " + "  ".join("-" * w[c] for c in cols))
    for x in rows:
        print("  " + "  ".join(str(x.get(c, "")).ljust(w[c]) for c in cols))


def main() -> None:
    if not LEDGER.exists():
        print(f"ERROR: {LEDGER} not found — run backtest_engine.py --save first.")
        sys.exit(1)
    df = load()
    print("=" * 88)
    print("  COST / THETA OVERLAY  —  net expectancy (R) per trade vs round-trip cost")
    print("=" * 88)
    print(f"  {len(df):,} trades | stop size (risk_frac) median {df['risk_frac'].median()*100:.3f}% "
          f"of price | mean cost-multiplier E[1/risk_frac] = {df['inv_rf'].mean():.0f}")
    print("  Realistic stock-FUTURES round trip ≈ 5-9 bps; options strictly worse (theta + spread).")
    print("  Columns = net expectancy R at that round-trip cost.  breakeven_bps = cost that zeroes it.")

    show("OVERALL  /  BY DIRECTION", [
        row("ALL", df),
        row("CE long", df[df.dir == "CE"]),
        row("PE short", df[df.dir == "PE"]),
    ])

    # PE per year — is the *net* edge stable, or only gross?
    show("PE short — PER YEAR (does net survive every year?)",
         [row(str(y), df[(df.dir == "PE") & (df.year == y)]) for y in sorted(df.year.unique())])

    # The only place a real pocket could hide: high-ATR% names (loose stops → low cost_R).
    # Quartile by risk_frac; Q4 = widest stops, cheapest in R terms.
    df["rf_q"] = pd.qcut(df["risk_frac"], 4, labels=["Q1 tight", "Q2", "Q3", "Q4 wide"])
    show("PE short — BY STOP-WIDTH QUARTILE (Q4 = widest stop, best cost-resistance)",
         [row(str(q), df[(df.dir == "PE") & (df.rf_q == q)]) for q in df["rf_q"].cat.categories])

    # Index vs stocks, net at a realistic 6 bps.
    show("INDEX vs STOCKS",
         [row("indices", df[df.symbol.str.endswith('-INDEX')]),
          row("stocks", df[~df.symbol.str.endswith('-INDEX')])])

    # Verdict line
    pe = df[df.dir == "PE"]
    be_pe = breakeven_bps(pe["r"], pe["inv_rf"])
    be_all = breakeven_bps(df["r"], df["inv_rf"])
    print("\n" + "=" * 88)
    print("  VERDICT")
    print("=" * 88)
    print(f"  Breakeven round-trip cost:  ALL = {be_all:.2f} bps   PE-only = {be_pe:.2f} bps")
    print(f"  Realistic floor (futures)  ≈ 5-9 bps  →  edge is BELOW the cost floor "
          f"{'(DEAD)' if be_pe < 5 else '(marginal)'}.")
    print("  Options add theta + wider spreads on top, so the live engine is worse still.\n")


if __name__ == "__main__":
    main()
