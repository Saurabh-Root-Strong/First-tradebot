"""
analyze_backtest.py  —  Stability & robustness dissection of the saved backtest
                        ledger (data/backtest/trades.parquet).

Answers the question the headline expectancy cannot: is the edge STABLE across time,
across direction, and across the universe — or an artifact of one stretch / a few
names? Reads the ledger written by backtest_engine.py --save; no re-run needed.

  .venv\\Scripts\\python.exe analyze_backtest.py
"""
from __future__ import annotations
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

from core.constants import DATA_DIR   # anchor to the repo, not the CWD

LEDGER = DATA_DIR / "backtest" / "trades.parquet"


def stats(r: pd.Series) -> dict:
    if len(r) == 0:
        return {"trades": 0, "win%": 0.0, "expectancy_R": 0.0, "total_R": 0.0, "PF": 0.0}
    gw = r[r > 0].sum(); gl = -r[r < 0].sum()
    return {
        "trades": len(r),
        "win%": round(100 * (r > 0).mean(), 1),
        "expectancy_R": round(r.mean(), 3),
        "total_R": round(r.sum(), 1),
        "PF": round(gw / gl, 2) if gl > 0 else float("inf"),
    }


def table(title: str, rows: list[tuple[str, pd.Series]]) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)
    recs = [{"scope": name, **stats(r)} for name, r in rows]
    cols = ["scope", "trades", "win%", "expectancy_R", "total_R", "PF"]
    w = {c: max(len(c), max((len(str(x.get(c, ""))) for x in recs), default=0)) for c in cols}
    print("  " + "  ".join(c.ljust(w[c]) for c in cols))
    print("  " + "  ".join("-" * w[c] for c in cols))
    for x in recs:
        print("  " + "  ".join(str(x.get(c, "")).ljust(w[c]) for c in cols))


def main() -> None:
    if not LEDGER.exists():
        print(f"ERROR: {LEDGER} not found — run backtest_engine.py --save first.")
        sys.exit(1)
    df = pd.read_parquet(LEDGER)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"])
    df["year"] = df["entry_ts"].dt.year
    df["q"] = df["entry_ts"].dt.to_period("Q").astype(str)
    is_index = df["symbol"].str.endswith("-INDEX")

    print("=" * 72)
    print("  BACKTEST STABILITY DISSECTION")
    print("=" * 72)
    print(f"  Ledger: {len(df):,} trades   "
          f"{df['entry_ts'].min():%Y-%m-%d} → {df['entry_ts'].max():%Y-%m-%d}")

    # 1. Per calendar year — overall + by direction (the core stability test)
    years = sorted(df["year"].unique())
    table("PER YEAR — overall", [(str(y), df[df.year == y]["r"]) for y in years])
    table("PER YEAR — CE (long)",
          [(str(y), df[(df.year == y) & (df.dir == "CE")]["r"]) for y in years])
    table("PER YEAR — PE (short)",
          [(str(y), df[(df.year == y) & (df.dir == "PE")]["r"]) for y in years])

    # 2. Per quarter — more points to judge whether sign is stable
    qs = sorted(df["q"].unique())
    table("PER QUARTER — overall", [(q, df[df.q == q]["r"]) for q in qs])

    # 3. Index vs stocks
    table("INDEX vs STOCKS",
          [("indices", df[is_index]["r"]), ("stocks", df[~is_index]["r"])])

    # 4. Concentration — share of total positive/net R from top names
    by_sym = df.groupby("label")["r"].agg(["sum", "count"]).sort_values("sum", ascending=False)
    net = df["r"].sum()
    top10 = by_sym.head(10)["sum"].sum()
    bot10 = by_sym.tail(10)["sum"].sum()
    pos_syms = (by_sym["sum"] > 0).sum()
    print("\n" + "=" * 72)
    print("  CONCENTRATION")
    print("=" * 72)
    def _pct(a, b):
        return f"{100*a/b:.0f}%" if b else "n/a"
    print(f"  Net total R                 : {net:,.0f}")
    print(f"  Top-10 symbols contribute   : {top10:,.0f} R  ({_pct(top10, net)} of net)")
    print(f"  Bottom-10 symbols drag       : {bot10:,.0f} R")
    print(f"  Profitable symbols          : {pos_syms} / {len(by_sym)} "
          f"({_pct(pos_syms, len(by_sym))})")

    # 5. Direction split per year as a compact verdict matrix
    print("\n" + "=" * 72)
    print("  DIRECTION STABILITY MATRIX  (expectancy R per year)")
    print("=" * 72)
    piv = df.pivot_table(index="year", columns="dir", values="r", aggfunc="mean").round(3)
    cnt = df.pivot_table(index="year", columns="dir", values="r", aggfunc="count")
    print(piv.to_string())
    print("\n  trade counts:")
    print(cnt.to_string())

    # 6. Monthly net R — eyeball regime clustering / worst stretch
    m = df.set_index("entry_ts").groupby(pd.Grouper(freq="MS"))["r"].sum().round(0)
    print("\n" + "=" * 72)
    print("  MONTHLY NET R")
    print("=" * 72)
    for ts, v in m.items():
        bar = "#" * int(min(abs(v) / 30, 40))
        sign = "+" if v >= 0 else "-"
        print(f"  {ts:%Y-%m}  {sign}{abs(v):>6.0f}  {bar}")
    print()


if __name__ == "__main__":
    main()
