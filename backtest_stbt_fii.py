"""
backtest_stbt_fii.py — the overnight SHORT, hunted with the FII data used PROPERLY.

The first STBT pass used FII as a crude LEVEL (net < 0) and found nothing. That is the
weakest possible use of it. This project's own prior work says FII index-futures positioning
is CONTRARIAN to forward Nifty (IC -0.13/-0.16 — they hedge, they do not predict). If that is
true, the short signal is not "FII are short" but the opposite: FII CROWDED LONG, then fade
them. That test was never run. Run it.

Features, all known AT THE CLOSE of day D (leak-safe; the return is D -> D+1 09:30):
  fao_participant (client_type=FII):  fut_idx_long, fut_idx_short
      net        = long - short                       (positioning LEVEL)
      ratio      = long / (long + short)              (scale-free crowding)
      d_net      = day-over-day CHANGE in net          (the FLOW, not the level)
      z_net      = 60d z-score of net                  (is this an EXTREME?)
  fii_derivatives_stats (INDEX FUTURES): oi_contracts, buy/sell contracts + value
      fii_oi     = FII open interest in index futures
      d_oi       = its daily change
  fii_dii_cash: FII cash net (the other side of the same hand)

DISCIPLINE: every trigger is split CHRONOLOGICALLY up front. A number that only exists in one
half is a regime, not an edge — that is exactly how the last candidate (short-buildup + weak
close: +32.7 bps in H2, -2.8 in H1) died. No trigger is reported without both halves.

    .venv\\Scripts\\python.exe backtest_stbt_fii.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import btst_signal as bs
from backtest_stbt import overnight_table, DUCK, COST


def fii_features():
    import duckdb
    c = duckdb.connect(DUCK, read_only=True)
    p = c.execute("""
        select trade_date, fut_idx_long::double lng, fut_idx_short::double sht
        from fao_participant where upper(client_type) like '%FII%' order by trade_date
    """).fetchdf()
    s = c.execute("""
        select trade_date, sum(oi_contracts)::double fii_oi,
               sum(buy_contracts - sell_contracts)::double fii_flow
        from fii_derivatives_stats where upper(category) like '%INDEX FUTURES%'
        group by 1 order by 1
    """).fetchdf()
    try:
        cash = c.execute("""
            select trade_date, sum(fii_net)::double fii_cash
            from fii_dii_cash group by 1 order by 1
        """).fetchdf()
    except Exception:
        cash = None
    c.close()

    p["trade_date"] = pd.to_datetime(p["trade_date"]).dt.date
    s["trade_date"] = pd.to_datetime(s["trade_date"]).dt.date
    f = p.merge(s, on="trade_date", how="outer").sort_values("trade_date")
    f["net"] = f["lng"] - f["sht"]
    f["ratio"] = f["lng"] / (f["lng"] + f["sht"])
    f["d_net"] = f["net"].diff()
    f["d_oi"] = f["fii_oi"].diff()
    # 60d z-score: is TODAY's positioning an extreme vs the recent past? (rolling = causal)
    r = f["net"].rolling(60, min_periods=30)
    f["z_net"] = (f["net"] - r.mean()) / r.std()
    rr = f["ratio"].rolling(60, min_periods=30)
    f["z_ratio"] = (f["ratio"] - rr.mean()) / rr.std()
    if cash is not None and len(cash):
        cash["trade_date"] = pd.to_datetime(cash["trade_date"]).dt.date
        f = f.merge(cash, on="trade_date", how="left")
    return f


def split_report(df, trig: dict, label: str):
    print("=" * 100)
    print(label)
    print("=" * 100)
    print(f"  {'trigger':38s} {'n':>5s} {'mean':>8s} {'win':>5s} {'t':>6s} | "
          f"{'H1 mean':>8s} {'H1 t':>6s} | {'H2 mean':>8s} {'H2 t':>6s}  verdict")
    print("  " + "-" * 96)
    for name, m in trig.items():
        d = df[m.fillna(False)].sort_values("date")
        v = d["short_bps"].to_numpy(float)
        v = v[~np.isnan(v)]
        if len(v) < 50:
            print(f"  {name:38s} {len(v):>5d}   too few")
            continue
        t = v.mean() / (v.std() / np.sqrt(len(v)))
        h = len(v) // 2
        a, b = v[:h], v[h:]
        ta = a.mean() / (a.std() / np.sqrt(len(a)))
        tb = b.mean() / (b.std() / np.sqrt(len(b)))
        # a real edge is POSITIVE and SIGNIFICANT in BOTH halves. anything else is a regime.
        real = (a.mean() > COST and b.mean() > COST and ta > 1.5 and tb > 1.5)
        verdict = "SURVIVES BOTH HALVES" if real else "regime/noise"
        print(f"  {name:38s} {len(v):>5d} {v.mean():>+8.1f} {100*(v>0).mean():>4.0f}% {t:>+6.2f} | "
              f"{a.mean():>+8.1f} {ta:>+6.2f} | {b.mean():>+8.1f} {tb:>+6.2f}  {verdict}")
    print("  " + "-" * 96)


def main():
    df = overnight_table()
    f = fii_features()
    df = df.merge(f, left_on="date", right_on="trade_date", how="left")
    n = df["z_net"].notna().sum()
    print(f"\n  {len(df)} index-nights; FII features on {n} of them "
          f"({df.loc[df['z_net'].notna(),'date'].min()} .. {df['date'].max()})\n")

    trig = {
        # CONTRARIAN: fade a crowded FII long  (the hypothesis prior work implies)
        "FII crowded LONG (z_net > 1)":        df["z_net"] > 1,
        "FII crowded LONG (z_net > 1.5)":      df["z_net"] > 1.5,
        "FII long-ratio extreme (z > 1)":      df["z_ratio"] > 1,
        "FII crowded long + weak close":       (df["z_net"] > 1) & (df["clr"] <= 0.34),
        "FII crowded long + strong close":     (df["z_net"] > 1) & (df["clr"] >= 0.66),
        # FLOW, not level: FII AGGRESSIVELY ADDING longs today
        "FII adding longs hard (d_net>0, top q)":
            df["d_net"] > df["d_net"].quantile(0.75),
        "FII dumping longs (d_net, bottom q)":
            df["d_net"] < df["d_net"].quantile(0.25),
        # OI: FII index-futures open interest
        "FII index-fut OI rising + px up":     (df["d_oi"] > 0) & (df["ret_d"] > 0),
        "FII OI rising + weak close":          (df["d_oi"] > 0) & (df["clr"] <= 0.34),
        # raw level (the crude test that already failed — kept as the control)
        "CONTROL: FII net short (level)":      df["net"] < 0,
    }
    if "fii_cash" in df.columns and df["fii_cash"].notna().any():
        trig["FII selling CASH + weak close"] = (df["fii_cash"] < 0) & (df["clr"] <= 0.34)
    split_report(df, trig, "OVERNIGHT SHORT (STBT), conditioned on FII — split in BOTH halves")
    print("  A trigger must be positive AND significant in BOTH halves to be an edge.")
    print("  One good half is a REGIME — it is what killed the last candidate (+32.7 vs -2.8).")
    print(f"  Reminder of the hurdle: shorts pay the overnight drift; break-even needs > {COST} bps.\n")

    # the incumbent, scored the same way, as the yardstick
    lng = df.copy()
    lng["short_bps"] = -(df["short_bps"] + COST) - COST      # flip to LONG
    split_report(lng, {"BTST LONG: strong close (clr>=.66)": lng["clr"] >= 0.66},
                 "THE YARDSTICK — the incumbent rule, scored identically")


if __name__ == "__main__":
    main()
