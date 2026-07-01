"""
backtest_vrp_live.py — REAL captured-intraday-fill test of the straddle harvest.

The modeled backtest (backtest_vrp.py) died because its B-S premium errs −75..+107%/day.
This harness removes the model: it reconstructs the ACTUAL ATM straddle from the captured
chain_snapshots (real per-bar LTP per strike) on the fat-capture days, sells it at a sweep
of entry times, and exits at the real close (intrinsic on expiry days). Real fills, no model.

⚠️ n IS TINY (≈6 days, 2 expiries). This is a MECHANISM / model-validation check, NOT
evidence of an edge. It cannot confirm or overturn the 2yr verdict (recent days may be a
calm patch). Read per-day numbers as anecdotes; the value is: (a) does the real intraday
seller P&L behave as theorised, (b) how wrong was the model, definitively.

    .venv\\Scripts\\python.exe backtest_vrp_live.py
"""
from __future__ import annotations

import glob
import os
import datetime as dt

import duckdb
import numpy as np
import pandas as pd

_STRADDLE_K = 0.79788456
STEP = 50
ENTRY_TIMES = ["09:45", "10:30", "11:30", "12:30", "13:30", "14:30"]
EXIT = "15:25"
COST_PCT = 3.0           # all-in % of entry premium (spread-inclusive); net column


def _expiry_set():
    p = glob.glob("../Daily_Cash_Market/**/market_data.duckdb", recursive=True)[0]
    con = duckdb.connect(p, read_only=True)
    s = set(pd.to_datetime(con.execute(
        "select distinct expiry_date from fno_bhavcopy where symbol='NIFTY'").df()
        ["expiry_date"]).dt.date)
    con.close()
    return s


def _load_day(path):
    con = duckdb.connect(path, read_only=True)
    df = con.execute("""
        select ts, strike, side, ltp from chain_snapshots
        where symbol like '%NIFTY50%' and ltp > 0
    """).df()
    con.close()
    df["ts"] = pd.to_datetime(df["ts"]).dt.tz_localize(None)
    df["t"] = df["ts"].dt.time
    return df


def _straddle_at(df, when: dt.time):
    """Real ATM straddle (spot via put-call parity, K nearest) at the snapshot ≤ `when`."""
    sub = df[df["t"] <= when]
    if sub.empty:
        return None
    ts = sub["ts"].max()
    snap = df[df["ts"] == ts]
    piv = snap.pivot_table(index="strike", columns="side", values="ltp", aggfunc="last")
    if not {"CE", "PE"}.issubset(piv.columns):
        return None
    piv = piv.dropna()
    if piv.empty:
        return None
    spot = float((piv.index + piv["CE"] - piv["PE"]).median())
    K = int(round(spot / STEP) * STEP)
    if K not in piv.index:
        K = int(piv.index[np.abs(piv.index.to_numpy() - spot).argmin()])
    row = piv.loc[K]
    return {"ts": ts, "spot": spot, "K": K, "ce": float(row["CE"]), "pe": float(row["PE"]),
            "straddle": float(row["CE"] + row["PE"])}


def _bs_straddle(spot, sigma, t_years):
    return _STRADDLE_K * spot * sigma * np.sqrt(max(t_years, 1e-9))


def main():
    exp = _expiry_set()
    days = []
    for f in sorted(glob.glob("data/intraday/*.duckdb")):
        if os.path.getsize(f) < 30e6:                  # full-session fat captures only
            continue
        d = dt.date.fromisoformat(os.path.basename(f)[:-7])
        try:
            df = _load_day(f)
        except Exception:
            continue
        if df.empty or df["t"].max() < dt.time(15, 0):
            continue
        days.append((d, df))
    print("=" * 78)
    print("  VRP HARVEST — REAL captured intraday fills (NIFTY)  ⚠ n tiny: mechanism check")
    print("=" * 78)
    print(f"  days: {', '.join(str(d)+('*' if d in exp else '') for d, _ in days)}   (* = expiry)\n")

    rows = []
    for d, df in days:
        is_exp = d in exp
        exit_s = _straddle_at(df, dt.time(15, 25))
        for te in ENTRY_TIMES:
            hh, mm = map(int, te.split(":"))
            en = _straddle_at(df, dt.time(hh, mm))
            if en is None or exit_s is None:
                continue
            # exit at same strike K as entry (re-read its legs at close)
            sub = df[df["t"] <= dt.time(15, 25)]
            ts2 = sub["ts"].max()
            snap2 = df[df["ts"] == ts2]
            p2 = snap2.pivot_table(index="strike", columns="side", values="ltp", aggfunc="last")
            if not {"CE", "PE"}.issubset(p2.columns) or en["K"] not in p2.index:
                continue
            exit_straddle = float(p2.loc[en["K"], "CE"] + p2.loc[en["K"], "PE"])
            gross = en["straddle"] - exit_straddle           # seller collects the decay
            net = gross - en["straddle"] * COST_PCT / 100.0
            rows.append({"day": d, "exp": is_exp, "entry": te, "K": en["K"],
                         "prem": en["straddle"], "exit": exit_straddle,
                         "gross": gross, "net": net,
                         "spot_move": abs(exit_s["spot"] - en["spot"])})
    R = pd.DataFrame(rows)
    if R.empty:
        print("  no reconstructable straddles"); return

    # per entry-time, split expiry vs non-expiry (different products)
    print("  ── REAL seller P&L (pts) by entry time — sell ATM straddle, exit 15:25 ──")
    print(f"  {'entry':6} {'EXPIRY days (hold→settle)':28} {'NON-EXPIRY (intraday round-trip)':30}")
    print(f"  {'':6} {'n  grossμ  netμ  win%':28} {'n  grossμ  netμ  win%':30}")
    for te in ENTRY_TIMES:
        e = R[(R.entry == te) & R.exp]; n = R[(R.entry == te) & ~R.exp]
        def fmt(s):
            if len(s) == 0:
                return f"{0:<2}  {'--':>6} {'--':>6} {'--':>5}"
            return (f"{len(s):<2}  {s.gross.mean():+6.0f} {s.net.mean():+6.0f} "
                    f"{100*(s.net>0).mean():4.0f}%")
        print(f"  {te:6} {fmt(e):28} {fmt(n):30}")

    print("\n  ── per expiry-day detail (the actual harvest product) ──")
    for d, df in days:
        if d not in exp:
            continue
        sub = R[(R.day == d) & R.exp]
        if sub.empty:
            continue
        print(f"  {d} (expiry):")
        for _, r in sub.iterrows():
            print(f"     sell {r['entry']}  K={r['K']}  prem {r['prem']:5.0f}  "
                  f"exit {r['exit']:5.0f}  gross {r['gross']:+5.0f}  net {r['net']:+5.0f}  "
                  f"(spot moved {r['spot_move']:.0f})")

    # definitive model-error table at 13:00 (closes the loop on why the model failed)
    print("\n  ── MODEL ERROR: real vs B-S(prior-day IV, trading-time) @ entry times ──")
    print("  (uses each day's own ATM IV inverted at entry as σ — isolates the time-convention)")
    for d, df in days:
        en = _straddle_at(df, dt.time(13, 0))
        if en is None:
            continue
        # invert real IV from the real straddle (calendar T to 15:30 settle, weekly≈same day exp)
        ne = min([e for e in exp if e >= d], default=d)
        T_cal = max((dt.datetime.combine(ne, dt.time(15, 30)) -
                     dt.datetime.combine(d, dt.time(13, 0))).total_seconds(), 60) / (365*24*3600)
        iv_real = en["straddle"] / (_STRADDLE_K * en["spot"] * np.sqrt(T_cal)) if T_cal > 0 else np.nan
        # what the modeled harness used: trading-time T with that σ
        T_tr = max((dt.datetime.combine(ne, dt.time(15, 30)) -
                    dt.datetime.combine(d, dt.time(13, 0))).total_seconds(), 60) / (252*6.25*3600)
        model = _bs_straddle(en["spot"], iv_real, T_tr)
        print(f"  {d}{'*' if d in exp else ' '}  real {en['straddle']:5.0f}  "
              f"calT-model {_bs_straddle(en['spot'], iv_real, T_cal):5.0f}  "
              f"tradeT-model {model:5.0f}  realIV {iv_real:5.1%}")

    print("\n  READ: real fills, no premium model. n≈6 days/2 expiries = ANECDOTE, not")
    print("  significance — cannot move the 2yr verdict. Confirms whether the real intraday")
    print("  decay supports the harvest and how far the model strayed.")


if __name__ == "__main__":
    main()
