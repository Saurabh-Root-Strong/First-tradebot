"""
backtest_vrp.py — the go/no-go on the ONE cost-survivable index-option edge.

THESIS (REGIME_TAXONOMY.md v2): direction is dead at the cost floor; the only intraday
edge that can survive is harvesting the VARIANCE RISK PREMIUM (NSE retail overpays for
options → IV > realised). The naive intraday straddle round-trip is cost-NEGATIVE (you
pay bid-ask on the whole premium to capture a sliver of VRP per hour). The ONLY structure
that survives is:

    SELL the ATM straddle on EXPIRY DAY at midday, HOLD TO SETTLEMENT.
    → collect the entire remaining premium as decay, pay entry spread ONCE (no exit).

and the whole bet is that a calm (HARVEST) expiry morning predicts a contained afternoon,
while a trending/volatile (DANGER) expiry is the seller's death. The regime filter is the
risk control, not a garnish.

DATA (all real, no VIX needed):
  • Expiry calendar + EOD option closes: DCM fno_bhavcopy (real, 2yr). The day-before-expiry
    ATM straddle is INVERTED (Brenner-Subrahmanyam) to a real daily ATM IV — this replaces
    the missing VIX with the market's own implied vol, per index.
  • Intraday underlying: data/historical/5min (real, 2yr) — entry spot @ 13:00, settle @ close.
  • Premium @ 13:00 on expiry day = B-S ATM straddle with T = time-to-settlement and σ = the
    PRIOR day's inverted IV (causal — expiry-day EOD straddle ≈ 0, can't invert same day).

P&L per expiry (seller): premium_collected − |settle − K| − entry_cost(one-way).
Honest: premium is MODELLED (B-S), validated by the inverted-IV sanity + the one captured
expiry; sensitivity-banded on IV and cost. This is the necessary go/no-go, not a live fill.

    .venv\\Scripts\\python.exe backtest_vrp.py
    .venv\\Scripts\\python.exe backtest_vrp.py --entry 13:00 --cost-pct 3 --iv-mult 1.0
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

import regime_classifier as rc

HIST5 = "data/historical/5min/NSE_NIFTY50_INDEX_5min.parquet"
HISTD = "data/historical/daily/NSE_NIFTY50_INDEX_daily.parquet"
STRIKE_STEP = 50
SETTLE = dt.time(15, 30)
_STRADDLE_K = 0.79788456   # 2·N'(0) = √(2/π): ATM straddle ≈ K·S·σ·√T  (Brenner-Subrahmanyam, r=0)


def _dcm_path() -> str:
    c = glob.glob("../Daily_Cash_Market/**/market_data.duckdb", recursive=True)
    if not c:
        raise FileNotFoundError("DCM market_data.duckdb not found")
    return c[0]


def daily_atm_iv() -> pd.DataFrame:
    """Real daily ATM IV for NIFTY, inverted from the DCM bhavcopy ATM straddle close.

    For each trade_date: nearest non-zero-DTE expiry, ATM strike (closest to the day's
    underlying), CE+PE close → straddle → invert B-S → σ. Returns date, expiry, spot, iv.
    """
    import duckdb
    con = duckdb.connect(_dcm_path(), read_only=True)
    # NIFTY index options only; keep the nearest expiry per trade_date later.
    q = """
        select trade_date, expiry_date, strike_price, option_type, close_price
        from fno_bhavcopy
        where instrument in ('OPTIDX','OPT') and symbol='NIFTY'
          and option_type in ('CE','PE') and close_price > 0
    """
    df = con.execute(q).df()
    con.close()
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    df["expiry_date"] = pd.to_datetime(df["expiry_date"]).dt.date
    df["dte"] = df.apply(lambda r: (r["expiry_date"] - r["trade_date"]).days, axis=1)
    df = df[df["dte"] >= 1]                                   # avoid T→0 blow-up at inversion
    # nearest expiry per trade_date
    nearest = df.groupby("trade_date")["expiry_date"].min().rename("near_exp")
    df = df.join(nearest, on="trade_date")
    df = df[df["expiry_date"] == df["near_exp"]]
    # spot proxy = strike whose |CE-PE| is smallest (put-call parity ATM); robust enough
    piv = df.pivot_table(index=["trade_date", "expiry_date", "dte", "strike_price"],
                         columns="option_type", values="close_price", aggfunc="last").reset_index()
    piv = piv.dropna(subset=["CE", "PE"])
    piv["synth_spot"] = piv["strike_price"] + piv["CE"] - piv["PE"]   # forward via parity
    rows = []
    for (td, exp, dte), g in piv.groupby(["trade_date", "expiry_date", "dte"]):
        spot = g["synth_spot"].median()
        g = g.assign(dist=(g["strike_price"] - spot).abs()).sort_values("dist")
        atm = g.iloc[0]
        straddle = atm["CE"] + atm["PE"]
        T = dte / 365.0
        iv = straddle / (_STRADDLE_K * spot * np.sqrt(T)) if (spot > 0 and T > 0) else np.nan
        rows.append({"date": td, "expiry": exp, "dte": dte, "spot": spot,
                     "atm_k": atm["strike_price"], "straddle_eod": straddle, "iv": iv})
    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return out[(out["iv"] > 0.03) & (out["iv"] < 0.80)]      # drop inversion garbage


def _bs_straddle(spot: float, sigma: float, t_years: float) -> float:
    return _STRADDLE_K * spot * sigma * np.sqrt(max(t_years, 1e-9))


def simulate(entry: dt.time, cost_pct: float, iv_mult: float,
             stop_mult: float | None) -> pd.DataFrame:
    """One row per expiry: sell ATM straddle at `entry`, hold to settlement (or stop)."""
    ivdf = daily_atm_iv()
    iv_by_date = dict(zip(ivdf["date"], ivdf["iv"]))
    expiries = sorted(ivdf["expiry"].unique())
    # prior-trading-day IV lookup (causal σ for the expiry-day entry)
    dates_sorted = ivdf["date"].tolist()
    prev_iv = {}
    for i in range(1, len(dates_sorted)):
        prev_iv[dates_sorted[i]] = iv_by_date[dates_sorted[i - 1]]

    bars = pd.read_parquet(HIST5)
    bars["ts"] = pd.to_datetime(bars["ts"]); bars["date"] = bars["ts"].dt.date
    by_date = {d: g.sort_values("ts").reset_index(drop=True) for d, g in bars.groupby("date")}

    rows = []
    for exp in expiries:
        g = by_date.get(exp)
        if g is None or len(g) < 30:
            continue
        sigma = prev_iv.get(exp)
        if sigma is None or not np.isfinite(sigma):
            continue
        sigma *= iv_mult
        g = g.set_index(g["ts"].dt.time)
        # entry spot
        ent = g[g.index <= entry]
        if ent.empty:
            continue
        s0 = float(ent.iloc[-1]["close"]); t_ent = ent.iloc[-1]["ts"]
        K = round(s0 / STRIKE_STEP) * STRIKE_STEP
        # premium at entry: intraday options decay on TRADING/SESSION time, not the
        # calendar clock. 2.5h left ≈ 0.4 of a 6.25h trading day; annualise in trading
        # time (252·6.25h). Using calendar-year T under-prices 0-DTE √T by ~2.4× and
        # falsely makes the seller bleed. Validated vs captured 0-DTE straddles (06-23=78pt).
        settle_dt = dt.datetime.combine(exp, SETTLE)
        rem_sec = max((settle_dt - t_ent.to_pydatetime().replace(tzinfo=None)).total_seconds(), 60)
        T = rem_sec / (252 * 6.25 * 3600)                    # trading-time years
        prem = _bs_straddle(s0, sigma, T)
        cost = prem * cost_pct / 100.0
        # ── regime: classify the MORNING (open→entry) — calm vs trending/volatile ──
        morn = g[g.index <= entry]
        rets = morn["close"].pct_change().dropna()
        morn_rv = float(rets.std()) if len(rets) > 3 else np.nan
        s_open = float(morn.iloc[0]["open"]); drift = abs(s0 / s_open - 1.0)
        mood = rc.classify_at(morn.reset_index(drop=True), n=10)
        # path to settlement
        post = g[(g.index > entry)]
        s_settle = float(g.iloc[-1]["close"])
        # optional intraday stop: buy back if |S - K| breaches stop_mult·prem (modelled exit cost)
        exit_px, stopped = s_settle, False
        if stop_mult is not None and len(post):
            breach = post[(post["close"] - K).abs() >= stop_mult * prem]
            if len(breach):
                exit_px = float(breach.iloc[0]["close"]); stopped = True
        intrinsic = abs(exit_px - K)
        # seller P&L: collect premium, pay intrinsic at settle, minus entry cost
        # (+ a second cost if stopped early — round-trip on that exit)
        pnl = prem - intrinsic - cost - (cost if stopped else 0.0)
        rows.append({
            "expiry": exp, "spot": s0, "K": K, "sigma": sigma, "T_hr": T * 365 * 24,
            "premium": prem, "cost": cost, "settle": s_settle, "intrinsic": intrinsic,
            "stopped": stopped, "pnl": pnl, "pnl_pct": 100 * pnl / s0,
            "morn_rv": morn_rv, "drift": drift, "mood": mood.short, "mood_full": mood.mood,
        })
    return pd.DataFrame(rows)


def _block_ci(x: np.ndarray, reps=3000, seed=7):
    rng = np.random.default_rng(seed)
    if len(x) < 4:
        return float(np.mean(x)) if len(x) else np.nan, np.nan, np.nan
    means = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(reps)]
    return float(np.mean(x)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _row(name, s):
    if len(s) == 0:
        return None
    pnl = s["pnl"].to_numpy(float)
    m, lo, hi = _block_ci(pnl)
    win = 100 * (pnl > 0).mean()
    ev = "EDGE" if lo > 0 else ("bleed" if hi < 0 else "—")
    return (f"  {name:14} n={len(s):>3}  win {win:4.0f}%  "
            f"mean {m:+6.1f}pt [{lo:+6.1f},{hi:+6.1f}] {ev:5}  "
            f"worst {pnl.min():+7.1f}  premΣ {s['premium'].mean():5.0f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", default="13:00", help="entry time HH:MM (default 13:00)")
    ap.add_argument("--cost-pct", type=float, default=3.0, help="entry cost %% of premium (one-way)")
    ap.add_argument("--iv-mult", type=float, default=1.0, help="IV sensitivity multiplier")
    ap.add_argument("--stop-mult", type=float, default=None,
                    help="intraday stop: buy back if |S-K| >= mult·premium (default: hold naked)")
    args = ap.parse_args()
    hh, mm = map(int, args.entry.split(":"))
    entry = dt.time(hh, mm)

    print("=" * 80)
    print("  VRP HARVEST — expiry-day ATM straddle SELL, hold to settlement (NIFTY)")
    print(f"  entry {args.entry}  cost {args.cost_pct}%/leg one-way  iv×{args.iv_mult}  "
          f"stop {args.stop_mult or 'none (naked to settle)'}")
    print("=" * 80)
    df = simulate(entry, args.cost_pct, args.iv_mult, args.stop_mult)
    if df.empty:
        print("  no expiries simulated (missing intraday/IV overlap)"); return
    print(f"  expiries simulated: {len(df)}  "
          f"({df['expiry'].min()} → {df['expiry'].max()})  "
          f"IV range {df['sigma'].min():.1%}-{df['sigma'].max():.1%}")
    print(f"  mean premium {df['premium'].mean():.0f}pt ({100*df['premium'].mean()/df['spot'].mean():.2f}% of spot)"
          f"  mean T {df['T_hr'].mean():.1f}h")

    # regime split: HARVEST = calm morning (low RV, no trend) ; DANGER = trending/volatile
    rv_med = df["morn_rv"].median()
    harvest = df[(df["morn_rv"] <= rv_med) & (df["drift"] <= 0.004)]
    danger  = df[~df.index.isin(harvest.index)]
    chop    = df[df["mood_full"] == rc.CHOP]

    print("\n  ── seller P&L per expiry (points), block-bootstrap 95% CI ──")
    for label, s in [("ALL expiries", df), ("HARVEST (calm)", harvest),
                     ("DANGER (trend/vol)", danger), ("mood=CHOP", chop)]:
        line = _row(label, s)
        if line:
            print(line)

    # the asymmetry that is the whole thesis
    if len(harvest) and len(danger):
        print(f"\n  HARVEST−DANGER mean spread: "
              f"{harvest['pnl'].mean() - danger['pnl'].mean():+.1f} pts/expiry")
    print("\n  READ: edge only if HARVEST mean-CI clears 0 AND beats DANGER. Premium is")
    print("  B-S-modelled from real bhavcopy-inverted IV — a model go/no-go, not a live fill.")


if __name__ == "__main__":
    main()
