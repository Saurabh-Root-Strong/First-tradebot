"""
backtest_stbt.py — is there ANY overnight SHORT (STBT) edge in index futures?

STBT is mechanically fine: you cannot short cash equities overnight (delivery), but index
FUTURES are symmetric, so selling at the close and covering next morning is executable. The
question is only whether an EDGE exists. The mirror of the BTST rule (short a WEAK close) is
dead -- t=0.92, indistinguishable from zero. But that was ONE trigger. This asks the honest
version: with EVERYTHING captured -- futures OI, change in OI, volume, option PCR, FII
derivative positioning, rollover into expiry, days-to-expiry -- can a short be found?

THE HURDLE. Equity indices carry a POSITIVE OVERNIGHT DRIFT (the well-documented overnight
anomaly: close->open earns, open->close does not). A short pays that drift EVERY night. So a
short signal is not competing against zero -- it must beat the drift AND the 3bps cost AND
carry an unbounded up-gap tail. This script prints the drift first, because it is the number
every short is fighting.

MULTIPLE TESTING is the trap here. Scanning N triggers and reporting the best is how one
manufactures a mirage. Every trigger is printed with its t-stat, and the Bonferroni bar for
the number tested is printed alongside -- so nothing gets to look significant by luck.

    .venv\\Scripts\\python.exe backtest_stbt.py
"""
from __future__ import annotations

import datetime as dt
import sys

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import btst_signal as bs

DUCK = r"d:\Python Projects\Daily_Cash_Market\data\market_data.duckdb"
COST = 3.0                       # futures round trip, bps
FUT = {"NIFTY": "NIFTY", "BANKNIFTY": "BANKNIFTY",
       "FINNIFTY": "FINNIFTY", "MIDCPNIFTY": "MIDCPNIFTY"}


def overnight_table():
    """One row per (index, day): the close, the NEXT morning's 09:30, and the features known
    AT THE CLOSE. Leak-safe by construction: every feature is from day D, the return is D->D+1."""
    rows = []
    for s in bs.SYMS:
        d = bs._daily(s)
        b = pd.read_parquet(bs.MIN5.format(bs.FY[s]))
        b["ts"] = pd.to_datetime(b["ts"]); b["date"] = b["ts"].dt.date
        dates = d["date"].tolist()
        for i in range(1, len(d) - 1):
            r, prev = d.iloc[i], d.iloc[i - 1]
            g = b[b["date"] == dates[i + 1]]
            if g.empty:
                continue
            x = g[g["ts"].dt.time >= bs.EXIT_T]
            o = g[g["ts"].dt.time >= dt.time(9, 15)]
            if not len(x) or not len(o):
                continue
            c = float(r["close"])
            rng = float(r["high"]) - float(r["low"])
            rows.append({
                "sym": s, "date": r["date"], "close": c,
                # SHORT P&L: you EARN the negative of the move
                "short_bps": -(float(x.iloc[0]["close"]) / c - 1.0) * 1e4 - COST,
                "gap_bps": (float(o.iloc[0]["open"]) / c - 1.0) * 1e4,
                "clr": bs._clr(r),
                "ret_d": (c / float(prev["close"]) - 1.0) * 1e4,
                "rng_pct": rng / c * 100.0 if c else np.nan,
                "vol": float(r.get("volume", np.nan)),
            })
    df = pd.DataFrame(rows)
    # volume z-score within index (a "heavy" day is relative to its own index)
    df["vol_z"] = df.groupby("sym")["vol"].transform(lambda v: (v - v.mean()) / v.std())
    return df


def add_fno(df):
    """Attach futures OI / chg-OI / PCR / FII positioning from the DCM bhavcopy (15M rows)."""
    try:
        import duckdb
    except ImportError:
        print("  (duckdb not available — OI/PCR/FII triggers skipped)")
        return df
    c = duckdb.connect(DUCK, read_only=True)
    # NEAREST-expiry index FUTURES: OI + chg OI + contracts, per symbol per day
    fut = c.execute("""
        with f as (
          select trade_date, symbol, expiry_date, open_interest, chg_in_oi, contracts,
                 row_number() over (partition by trade_date, symbol
                                    order by expiry_date) rn
          from fno_bhavcopy
          where instrument like 'FUTIDX%' and symbol in ('NIFTY','BANKNIFTY','FINNIFTY','MIDCPNIFTY')
        )
        select trade_date, symbol, expiry_date, open_interest fut_oi, chg_in_oi fut_doi,
               contracts fut_vol
        from f where rn = 1
    """).fetchdf()
    # option PCR (put OI / call OI), nearest expiry
    pcr = c.execute("""
        with o as (
          select trade_date, symbol, option_type, open_interest,
                 dense_rank() over (partition by trade_date, symbol order by expiry_date) rk
          from fno_bhavcopy
          where instrument like 'OPTIDX%' and symbol in ('NIFTY','BANKNIFTY','FINNIFTY','MIDCPNIFTY')
        )
        select trade_date, symbol,
               sum(case when option_type='PE' then open_interest else 0 end)::double /
               nullif(sum(case when option_type='CE' then open_interest else 0 end),0) pcr
        from o where rk = 1 group by 1,2
    """).fetchdf()
    # FII index-futures POSITIONING. NOTE: fii_derivatives_stats.category is the CONTRACT
    # (NIFTY FUTURES / INDEX FUTURES / ...), NOT the investor type -- filtering it on '%FII%'
    # silently matched nothing and the "FII" rows came back n=0, i.e. the test never ran.
    # The investor breakdown is in fao_participant.client_type.
    fii = c.execute("""
        select trade_date, (fut_idx_long - fut_idx_short)::double fii_net
        from fao_participant where upper(client_type) like '%FII%'
    """).fetchdf()
    c.close()
    for t in (fut, pcr):
        t["trade_date"] = pd.to_datetime(t["trade_date"]).dt.date
    fii["trade_date"] = pd.to_datetime(fii["trade_date"]).dt.date

    df = df.merge(fut, left_on=["date", "sym"], right_on=["trade_date", "symbol"], how="left")
    df = df.merge(pcr, left_on=["date", "sym"], right_on=["trade_date", "symbol"],
                  how="left", suffixes=("", "_p"))
    df = df.merge(fii, left_on="date", right_on="trade_date", how="left", suffixes=("", "_f"))
    df["dte"] = (pd.to_datetime(df["expiry_date"]).dt.date - df["date"]).map(
        lambda x: x.days if pd.notna(x) else np.nan)
    # OI buildup, the classic reading: price DOWN + OI UP = fresh SHORTS being built
    df["short_buildup"] = (df["ret_d"] < 0) & (df["fut_doi"] > 0)
    df["long_unwind"] = (df["ret_d"] < 0) & (df["fut_doi"] < 0)
    return df


def report(df):
    a = df["short_bps"].to_numpy(float)
    a = a[~np.isnan(a)]
    print("=" * 92)
    print("THE HURDLE — what every short is fighting")
    print("=" * 92)
    lo = -(a + COST)                       # undo the short: the LONG overnight return, gross
    print(f"  unconditional OVERNIGHT drift (close -> next 09:30), all {len(lo)} index-days:")
    print(f"     LONG  {lo.mean():+6.1f} bps   (t={lo.mean()/(lo.std()/np.sqrt(len(lo))):+.2f})")
    print(f"     so a SHORT starts at {-lo.mean():+.1f} bps BEFORE costs, {-lo.mean()-COST:+.1f} after.")
    print("     Indices grind UP overnight. A short pays that drift every single night.\n")

    trig = {
        "weak close (clr<=0.34)":            df["clr"] <= 0.34,
        "very weak close (clr<=0.20)":       df["clr"] <= 0.20,
        "big down day (<-100bps)":           df["ret_d"] <= -100,
        "big down + weak close":             (df["ret_d"] <= -100) & (df["clr"] <= 0.34),
        "heavy-volume down day (z>1)":       (df["ret_d"] < 0) & (df["vol_z"] > 1),
        "wide-range down day (rng>1.2%)":    (df["ret_d"] < 0) & (df["rng_pct"] > 1.2),
    }
    if "fut_doi" in df.columns and df["fut_doi"].notna().any():
        trig.update({
            "SHORT BUILDUP (px dn + OI up)":  df["short_buildup"] == True,
            "  + weak close":                 (df["short_buildup"] == True) & (df["clr"] <= 0.34),
            "long unwinding (px dn + OI dn)": df["long_unwind"] == True,
            "low PCR (<0.8) = call-heavy":    df["pcr"] < 0.8,
            "high PCR (>1.3) + down day":     (df["pcr"] > 1.3) & (df["ret_d"] < 0),
            "FII net SHORT index futs":       df["fii_net"] < 0,
            "FII short + weak close":         (df["fii_net"] < 0) & (df["clr"] <= 0.34),
            "expiry week (dte<=3) + down":    (df["dte"] <= 3) & (df["ret_d"] < 0),
        })
    n_tests = len(trig)
    bar = 2.0 + np.log(n_tests)            # rough Bonferroni-ish bar for |t|
    print("=" * 92)
    print(f"EVERY SHORT TRIGGER I CAN BUILD FROM THE DATA  ({n_tests} tested)")
    print("=" * 92)
    print(f"  {'trigger':34s} {'n':>5s} {'mean bps':>9s} {'win%':>6s} {'t':>7s} {'worst':>8s}  verdict")
    print("  " + "-" * 88)
    best = None
    for name, m in trig.items():
        v = df.loc[m.fillna(False), "short_bps"].to_numpy(float)
        v = v[~np.isnan(v)]
        if len(v) < 30:
            print(f"  {name:34s} {len(v):>5d}   too few")
            continue
        t = v.mean() / (v.std() / np.sqrt(len(v)))
        ok = "EDGE?" if (t > bar and v.mean() > COST) else "dead"
        print(f"  {name:34s} {len(v):>5d} {v.mean():>+9.1f} {100*(v>0).mean():>5.0f}% "
              f"{t:>+7.2f} {v.min():>+8.0f}  {ok}")
        if best is None or t > best[1]:
            best = (name, t)
    print("  " + "-" * 88)
    print(f"  significance bar for {n_tests} tests (multiple-testing corrected): |t| > {bar:.2f}")
    print(f"  best trigger: {best[0]}  t={best[1]:+.2f}")
    print()
    print("  READ: a t-stat below the bar is NOISE, however good the mean looks. Scanning many")
    print("  triggers and keeping the prettiest is exactly how a backtest mirage is built.")

    # ── THE DECISIVE TEST ────────────────────────────────────────────────────────────────
    # The winner of a 14-trigger scan is a CANDIDATE, never a finding. The only thing that
    # separates a real edge from a lucky cell is whether it SURVIVES OUT OF SAMPLE. Split the
    # candidate chronologically: fit nothing, just look at whether the second half still works.
    if "short_buildup" not in df.columns:
        return
    cand = df[(df["short_buildup"] == True) & (df["clr"] <= 0.34)].sort_values("date")
    if len(cand) < 60:
        return
    print("\n" + "=" * 92)
    print("OUT-OF-SAMPLE — the ONLY question that matters for the one candidate")
    print("   trigger: SHORT BUILDUP (price down + futures OI up) + weak close")
    print("=" * 92)
    half = len(cand) // 2
    for nm, part in (("IN-SAMPLE  (first half)", cand.iloc[:half]),
                     ("OUT-OF-SAMPLE (2nd half)", cand.iloc[half:])):
        v = part["short_bps"].to_numpy(float)
        t = v.mean() / (v.std() / np.sqrt(len(v)))
        print(f"  {nm:26s} {part['date'].min()} .. {part['date'].max()}  n={len(v):>3d}  "
              f"mean {v.mean():+6.1f} bps  win {100*(v>0).mean():3.0f}%  t={t:+.2f}")
    # and per-index: an "edge" carried by one index is a quirk of that index, not a rule
    print("\n  per index (an edge carried by ONE index is that index's quirk, not a rule):")
    for s in bs.SYMS:
        v = cand[cand["sym"] == s]["short_bps"].to_numpy(float)
        if len(v) < 15:
            print(f"    {s:11s} n={len(v):>3d}  too few")
            continue
        t = v.mean() / (v.std() / np.sqrt(len(v)))
        print(f"    {s:11s} n={len(v):>3d}  mean {v.mean():+6.1f} bps  "
              f"win {100*(v>0).mean():3.0f}%  t={t:+.2f}")


if __name__ == "__main__":
    df = overnight_table()
    df = add_fno(df)
    report(df)
