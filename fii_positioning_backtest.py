"""
fii_positioning_backtest.py — does SMART-MONEY positioning predict the index?

The truest "smart-money footprint" isn't inferred from OI/premium — it's the actual
participant-wise positioning NSE publishes EOD: FII / Pro / DII / Client long vs short
in index futures. This tests, on ~18 months of real data, the canonical question:

    does FII (and Pro) index-futures positioning predict forward Nifty returns?

Lookahead-free: participant OI for day D is public ~6 pm D, so the earliest tradeable
moment is D+1 OPEN. Every forward return is measured from the next session's open.
Client (retail) is the contrarian control — if the smart-money thesis holds, FII/Pro
should show positive predictive power and Client negative.

Signals (per participant):
  level : net index-fut OI, normalised  = (long - short) / (long + short)
  flow  : day-over-day change in that net (what they DID today)

Metrics: rank IC (Spearman) vs forward return, top-vs-bottom tercile return spread
(net of a round-trip cost), hit rate, and sample size.

  .venv\\Scripts\\python.exe fii_positioning_backtest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import duckdb

DCM = Path(__file__).resolve().parent.parent / "Daily_Cash_Market" / "data" / "market_data.duckdb"
COST_BPS = 5.0          # round-trip index-futures cost; tercile spread is long-short → 2×
HORIZONS = [1, 3, 5]    # forward sessions held (entry next-session open → close h sessions later)


def _load() -> pd.DataFrame:
    con = duckdb.connect(str(DCM), read_only=True)
    part = con.execute("""
        SELECT trade_date, client_type,
               fut_idx_long AS long, fut_idx_short AS short
        FROM fao_participant WHERE data_type='OI'
    """).df()
    px = con.execute("""
        SELECT trade_date, open_val, close_val
        FROM index_data WHERE index_name='Nifty 50' ORDER BY trade_date
    """).df()
    con.close()
    part["trade_date"] = pd.to_datetime(part["trade_date"])
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    px = px.dropna(subset=["open_val", "close_val"]).drop_duplicates("trade_date").sort_values("trade_date")
    # forward returns from the NEXT session's open (no lookahead — signal is EOD)
    px["entry"] = px["open_val"].shift(-1)
    for h in HORIZONS:
        px[f"ret{h}"] = px["close_val"].shift(-h) / px["entry"] - 1
    return part, px


def _signals(part: pd.DataFrame, ctype: str) -> pd.DataFrame:
    d = part[part["client_type"] == ctype].sort_values("trade_date").copy()
    tot = (d["long"] + d["short"]).replace(0, np.nan)
    d["level"] = (d["long"] - d["short"]) / tot          # net long ratio ∈ [-1,1]
    d["flow"] = d["level"].diff()                         # change vs prior day
    return d[["trade_date", "level", "flow"]]


def _ic(sig: pd.Series, ret: pd.Series) -> float:
    m = sig.notna() & ret.notna()
    if m.sum() < 30:
        return float("nan")
    return sig[m].rank().corr(ret[m].rank())


def _tercile_spread(sig: pd.Series, ret: pd.Series) -> tuple[float, int]:
    m = sig.notna() & ret.notna()
    s, r = sig[m], ret[m]
    if len(s) < 30:
        return float("nan"), len(s)
    try:
        q = pd.qcut(s, 3, labels=["lo", "mid", "hi"])
    except Exception:
        return float("nan"), len(s)
    hi, lo = r[q == "hi"].mean(), r[q == "lo"].mean()
    return (hi - lo) * 100, len(s)        # long-top / short-bottom spread, in %


def main() -> None:
    if not DCM.exists():
        print(f"ERROR: DCM db not found at {DCM}"); return
    part, px = _load()
    n_days = px["entry"].notna().sum()
    print("=" * 84)
    print(f"  SMART-MONEY POSITIONING → FORWARD NIFTY  ({px['trade_date'].min():%Y-%m-%d} … "
          f"{px['trade_date'].max():%Y-%m-%d}, {n_days} sessions)")
    print("=" * 84)
    print("  Signal known EOD → entry NEXT session open → no lookahead. IC = Spearman rank corr.")
    print("  tercile = top-third minus bottom-third forward return, NET of "
          f"{2*COST_BPS:.0f}bps (long-short).\n")

    for ctype in ["FII", "Pro", "Client"]:
        sig = _signals(part, ctype)
        df = sig.merge(px[["trade_date"] + [f"ret{h}" for h in HORIZONS]], on="trade_date", how="inner")
        tag = {"FII": "smart money", "Pro": "smart money", "Client": "retail (contrarian control)"}[ctype]
        print(f"  ── {ctype}  [{tag}] ──")
        print(f"     {'signal':<7}{'horizon':<9}{'IC':>8}{'tercile%':>11}{'net%':>9}{'n':>6}")
        for which in ["level", "flow"]:
            for h in HORIZONS:
                ic = _ic(df[which], df[f"ret{h}"])
                spread, n = _tercile_spread(df[which], df[f"ret{h}"])
                net = spread - 2 * COST_BPS / 100 if spread == spread else float("nan")
                print(f"     {which:<7}{h}-day{'':<4}{ic:>8.3f}{spread:>10.2f}%{net:>+8.2f}%{n:>6}")
        print()

    print("=" * 84)
    print("  READ: |IC|>~0.1 ≈ real signal (SE≈1/√n≈0.05). Smart-money thesis ⇒ FII/Pro IC>0,")
    print("  Client IC<0. A positive NET tercile spread = the signal pays after costs.")
    print("=" * 84)


if __name__ == "__main__":
    main()
