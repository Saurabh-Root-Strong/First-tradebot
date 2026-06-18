"""
eod_oi_range.py  —  EOD OI-structure next-day RANGE map, and its calibration test.

GOAL (decision-support cockpit, NOT a directional arrow)
--------------------------------------------------------
Each evening, from the EOD option chain (DCM fno_bhavcopy), build tomorrow's
expected boundary map for each index:
    floor   = put wall   (strike with heaviest PE OI below spot)
    ceiling = call wall   (strike with heaviest CE OI above spot)
    magnet  = max pain    (settlement that minimises total option-writer payout)
This is the daily analogue of the (now-abandoned) intraday-level idea, but anchored
on the half that (a) has a real mechanism — dealer hedging / expiry pinning — and
(b) actually has daily history to backtest. Price action is demoted to width/regime.

THE ONLY QUESTION THAT MATTERS FIRST (falsification before we trust the map):
  Does the OI band contain tomorrow's price BETTER than a dumb band of the same
  width, and does max pain pull price more than a random strike? If not, the OI
  boundaries are decoration and the cockpit shouldn't lean on them.

Nulls (the discipline that killed the PA-level idea — kept here):
  • placement null : a band of the SAME WIDTH centered at spot. Isolates whether the
      walls' (asymmetric) PLACEMENT beats "a band of that width around the close".
  • width   null : a band centered at spot whose half-width is k*ATR, k set so the
      MEDIAN width matches the OI band. Isolates whether OI WIDTH beats volatility.
  • pinning  null : replace max pain with a random in-window strike.

Causal: map built from data <= EOD day d (close, ATR, chain). Outcome = day d+1's
low/high/close from data/historical/daily. No lookahead.

Coverage: fno_bhavcopy per-strike OI starts 2024-12 (~379 days) — modest sample.

Usage
  .venv\\Scripts\\python.exe eod_oi_range.py
  .venv\\Scripts\\python.exe eod_oi_range.py --near-expiry 5   # condition pinning on DTE<=5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from backtest_engine import _atr, _safe, DATA_DIR        # noqa: E402 (reuse)

DCM_DB = Path(r"D:\Python Projects\Daily_Cash_Market\data\market_data.duckdb")
SEED   = 7

# DCM option symbol  <->  Fyers index symbol (for the daily OHLC parquet)
IDX = {
    "NIFTY":      "NSE:NIFTY50-INDEX",
    "BANKNIFTY":  "NSE:NIFTYBANK-INDEX",
    "FINNIFTY":   "NSE:FINNIFTY-INDEX",
    "MIDCPNIFTY": "NSE:MIDCPNIFTY-INDEX",
    "NIFTYNXT50": "NSE:NIFTYNXT50-INDEX",
}
WIN = 0.08   # search walls / max pain within +/-8% of spot (kills far-OTM junk OI)


# ── EOD chain -> walls + max pain for one (date, front-expiry) ──────────────────

def _max_pain(strikes: np.ndarray, ce: np.ndarray, pe: np.ndarray) -> float:
    """Settlement strike that minimises total writer payout (classic max pain)."""
    best_k, best_pain = strikes[0], np.inf
    for k in strikes:                       # candidate settlement = each listed strike
        pain = (ce * np.maximum(k - strikes, 0)).sum() + (pe * np.maximum(strikes - k, 0)).sum()
        if pain < best_pain:
            best_pain, best_k = pain, k
    return float(best_k)


def _day_map(chain: pd.DataFrame, spot: float) -> "dict | None":
    """chain = front-expiry rows for one date; returns floor/ceiling/magnet."""
    lo, hi = spot * (1 - WIN), spot * (1 + WIN)
    c = chain[(chain.strike_price >= lo) & (chain.strike_price <= hi)]
    ce = c[c.option_type == "CE"]; pe = c[c.option_type == "PE"]
    if len(ce) < 3 or len(pe) < 3:
        return None
    # walls: heaviest OI strike on the correct side of spot
    ce_up = ce[ce.strike_price >= spot]; pe_dn = pe[pe.strike_price <= spot]
    if ce_up.empty or pe_dn.empty:
        return None
    ceiling = float(ce_up.loc[ce_up.open_interest.idxmax(), "strike_price"])
    floor   = float(pe_dn.loc[pe_dn.open_interest.idxmax(), "strike_price"])
    if ceiling <= floor:
        return None
    # max pain over the union of in-window strikes
    grid = np.sort(c.strike_price.unique())
    ce_oi = c[c.option_type == "CE"].groupby("strike_price").open_interest.sum().reindex(grid, fill_value=0).to_numpy(float)
    pe_oi = c[c.option_type == "PE"].groupby("strike_price").open_interest.sum().reindex(grid, fill_value=0).to_numpy(float)
    magnet = _max_pain(grid, ce_oi, pe_oi)
    return {"floor": floor, "ceiling": ceiling, "magnet": magnet}


# ── Assemble per-index feature/outcome table ────────────────────────────────────

def _load_daily(fy_sym: str) -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / "daily" / f"{_safe(fy_sym)}_daily.parquet")
    df["ts"] = pd.to_datetime(df["ts"]); df["date"] = df["ts"].dt.date
    df = df.sort_values("ts").reset_index(drop=True)
    df["atr"] = _atr(df)
    return df


def build(dcm_sym: str, con) -> pd.DataFrame:
    opt = con.execute(
        "select trade_date, expiry_date, strike_price, option_type, open_interest "
        "from fno_bhavcopy where symbol = ? and option_type in ('CE','PE') "
        "and open_interest > 0", [dcm_sym]).df()
    if opt.empty:
        return pd.DataFrame()
    opt["trade_date"] = pd.to_datetime(opt["trade_date"]).dt.date
    opt["expiry_date"] = pd.to_datetime(opt["expiry_date"]).dt.date

    day = _load_daily(IDX[dcm_sym]).set_index("date")
    closes = day["close"]; atrs = day["atr"]
    nxt = day[["low", "high", "close"]].shift(-1)        # outcome = next session

    rows = []
    for td, g in opt.groupby("trade_date"):
        if td not in closes.index:
            continue
        spot, atr = float(closes.loc[td]), float(atrs.loc[td])
        if not np.isfinite(atr) or atr <= 0:
            continue
        # front expiry = nearest expiry on/after the trade date
        fe = g[g.expiry_date >= td]["expiry_date"]
        if fe.empty:
            continue
        front = fe.min()
        m = _day_map(g[g.expiry_date == front], spot)
        if m is None or td not in nxt.index or pd.isna(nxt.loc[td, "close"]):
            continue
        rows.append({"date": td, "spot": spot, "atr": atr,
                     "dte": (front - td).days, **m,
                     "n_low": float(nxt.loc[td, "low"]), "n_high": float(nxt.loc[td, "high"]),
                     "n_close": float(nxt.loc[td, "close"])})
    return pd.DataFrame(rows)


# ── Metrics ─────────────────────────────────────────────────────────────────────

def _contain(lo, hi, x):                       # next-day close inside [lo,hi]
    return ((x >= lo) & (x <= hi)).astype(float)


def _boot_diff(a: np.ndarray, b: np.ndarray, reps: int, rng) -> tuple[float, float, float]:
    """mean(a-b) with bootstrap 95% CI (paired)."""
    d = a - b
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(reps)])
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def analyse(df: pd.DataFrame, reps: int, rng) -> dict:
    spot, atr = df.spot.to_numpy(), df.atr.to_numpy()
    floor, ceil_, magnet = df.floor.to_numpy(), df.ceiling.to_numpy(), df.magnet.to_numpy()
    nclose = df.n_close.to_numpy()
    w = ceil_ - floor                                  # OI band width
    k_full = np.median(w / atr)                        # ATR multiple matching median width

    oi_c   = _contain(floor, ceil_, nclose)            # OI band
    half = w / 2.0
    sp_c   = _contain(spot - half, spot + half, nclose)            # placement null (same width @ spot)
    atr_c  = _contain(spot - k_full / 2 * atr, spot + k_full / 2 * atr, nclose)  # width null (ATR @ spot)

    place = _boot_diff(oi_c, sp_c, reps, rng)          # OI vs same-width-at-spot
    width = _boot_diff(sp_c, atr_c, reps, rng)         # OI-width vs ATR-width (both @ spot)

    # ── pinning: next-day return vs gap-to-magnet, against a random-strike null ──
    ret = nclose / spot - 1.0
    gap = (magnet - spot) / spot
    sign_pain = float(np.mean(np.sign(ret) == np.sign(gap)))
    rnd_rates = []
    for _ in range(reps):
        rk = spot * (1 + rng.uniform(-WIN, WIN, len(spot)))        # random in-window strike
        rnd_rates.append(np.mean(np.sign(ret) == np.sign((rk - spot) / spot)))
    rnd = np.array(rnd_rates)

    return {
        "n": len(df), "median_w_atr": float(k_full),
        "oi_close": float(oi_c.mean()), "spot_close": float(sp_c.mean()), "atr_close": float(atr_c.mean()),
        "place_lift": place, "width_lift": width,
        "pin_rate": sign_pain, "pin_null": float(rnd.mean()),
        "pin_lift": sign_pain - float(rnd.mean()),
        "pin_null_ci": (float(np.percentile(rnd, 2.5)), float(np.percentile(rnd, 97.5))),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="EOD OI range map + calibration test")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--near-expiry", type=int, default=0, help="if >0, pinning row also cut to DTE<=N")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    import duckdb
    if not DCM_DB.exists():
        print(f"DCM DuckDB not found: {DCM_DB}"); return
    con = duckdb.connect(str(DCM_DB), read_only=True)

    print("\nEOD OI next-day range — calibration vs nulls  (daily, causal d -> d+1)")
    print("=" * 84)
    frames, pooled = {}, []
    for sym in IDX:
        df = build(sym, con)
        if df.empty:
            print(f"  {sym:11} — no data"); continue
        frames[sym] = df; pooled.append(df)

    hdr = (f"{'index':11} {'n':>4} {'wOI/ATR':>7}  {'contain next close %':^28}  "
           f"{'placement lift':^20}")
    print(hdr); print(f"{'':11} {'':>4} {'':>7}  {'OI':>8}{'spot':>9}{'ATR':>9}   {'(OI - spot, same width)':>20}")
    print("-" * 84)

    def row(name, r):
        pl, lo, hi = r["place_lift"]
        star = "*" if lo > 0 else (" " if hi > 0 else "x")  # *=OI better, x=worse
        print(f"{name:11} {r['n']:>4} {r['median_w_atr']:>7.2f}  "
              f"{100*r['oi_close']:>7.1f}%{100*r['spot_close']:>8.1f}%{100*r['atr_close']:>8.1f}%   "
              f"{100*pl:>+6.1f}pp [{100*lo:+.1f},{100*hi:+.1f}] {star}")

    results = {}
    for sym, df in frames.items():
        r = analyse(df, args.reps, rng); results[sym] = r; row(sym, r)
    print("-" * 84)
    alldf = pd.concat(pooled, ignore_index=True)
    rp = analyse(alldf, args.reps, rng); row("POOLED", rp)

    # ── pinning block ──
    print("\nMax-pain pinning  (next-day return sign agrees with gap-to-magnet)")
    print(f"{'index':11} {'pin%':>6} {'null%':>6} {'lift':>7}  null95")
    print("-" * 56)
    for sym, r in results.items():
        nlo, nhi = r["pin_null_ci"]
        v = "PASS" if r["pin_rate"] > nhi else "fail"
        print(f"{sym:11} {100*r['pin_rate']:>5.1f} {100*r['pin_null']:>6.1f} "
              f"{100*r['pin_lift']:>+6.1f} [{100*nlo:.1f},{100*nhi:.1f}] {v}")
    nlo, nhi = rp["pin_null_ci"]
    print(f"{'POOLED':11} {100*rp['pin_rate']:>5.1f} {100*rp['pin_null']:>6.1f} "
          f"{100*rp['pin_lift']:>+6.1f} [{100*nlo:.1f},{100*nhi:.1f}] "
          f"{'PASS' if rp['pin_rate'] > nhi else 'fail'}")

    if args.near_expiry > 0:
        sub = alldf[alldf.dte <= args.near_expiry]
        if len(sub) > 30:
            rn = analyse(sub, args.reps, rng); nlo, nhi = rn["pin_null_ci"]
            print(f"\n  near-expiry (DTE<={args.near_expiry}, n={rn['n']}): "
                  f"pin {100*rn['pin_rate']:.1f}% vs null {100*rn['pin_null']:.1f}% "
                  f"[{100*nlo:.1f},{100*nhi:.1f}] -> {'PASS' if rn['pin_rate']>nhi else 'fail'}")

    # ── verdict ──
    pl, lo, hi = rp["place_lift"]
    print("\n" + "=" * 84)
    print(f"PLACEMENT : OI walls vs same-width band at spot -> {100*pl:+.1f}pp "
          f"[{100*lo:+.1f}, {100*hi:+.1f}]  ({'OI better' if lo>0 else 'no edge / worse'})")
    print(f"PINNING   : max pain vs random strike          -> {100*rp['pin_lift']:+.1f}pp "
          f"({'PASS' if rp['pin_rate']>rp['pin_null_ci'][1] else 'no edge'})")


if __name__ == "__main__":
    main()
