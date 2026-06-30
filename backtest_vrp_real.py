"""
backtest_vrp_real.py — HFT-grade, REAL-PRICE backtest of the expiry straddle harvest.

WHY THIS EXISTS: backtest_vrp.py priced the straddle with a B-S model whose per-day error
(−75%..+107% vs real captured straddles) DWARFS the claimed edge → unprovable. This harness
removes the model entirely: entry premium = REAL DCM bhavcopy ATM option close/open, exit =
REAL expiry settlement. The only modelled quantity is transaction cost (an explicit rupee /
%-of-premium stack, sensitivity-swept). No look-ahead: entry uses T-1 (or expiry-open) prices
known before the outcome; settlement is the answer key.

PRODUCT (the real-data cousin of the intraday 13:00 idea):
  • entry=tminus1 : SELL ATM straddle at the PRIOR trading day's CLOSE (real), hold to expiry
                    settle. Captures the fat last-day decay; carries one overnight.
  • entry=open    : SELL at the EXPIRY-DAY OPEN (real), settle same day. Intraday-only, but
                    enters at 09:15 (max gamma) since bhavcopy is EOD-granular.

Settlement = expiry-day underlying last-30min VWAP (from 5m history) — the NSE convention —
falling back to the daily close. Selector = VRP richness (entry IV inverted from the real
straddle vs trailing realised vol). Reports per-year stability, max drawdown, worst run,
CVaR, Sharpe/expiry, block-bootstrap CI. This is the go/no-go that can actually be trusted.

    .venv\\Scripts\\python.exe backtest_vrp_real.py
    .venv\\Scripts\\python.exe backtest_vrp_real.py --entry open --cost-pct 4 --symbol NIFTY
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

_STRADDLE_K = 0.79788456
STEP = {"NIFTY": 50, "BANKNIFTY": 100, "FINNIFTY": 50, "MIDCPNIFTY": 25}
HIST5 = "data/historical/5min/NSE_{}_INDEX_5min.parquet"
HISTD = "data/historical/daily/NSE_{}_INDEX_daily.parquet"
_FY = {"NIFTY": "NIFTY50", "BANKNIFTY": "NIFTYBANK", "FINNIFTY": "FINNIFTY", "MIDCPNIFTY": "MIDCPNIFTY"}


def _dcm():
    return glob.glob("../Daily_Cash_Market/**/market_data.duckdb", recursive=True)[0]


def load_bhav(symbol: str) -> pd.DataFrame:
    import duckdb
    con = duckdb.connect(_dcm(), read_only=True)
    df = con.execute("""
        select trade_date, expiry_date, strike_price, option_type, open_price, close_price
        from fno_bhavcopy
        where symbol=? and instrument in ('OPTIDX','OPT') and option_type in ('CE','PE')
          and close_price>0
    """, [symbol]).df()
    con.close()
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    df["expiry_date"] = pd.to_datetime(df["expiry_date"]).dt.date
    return df


def underlying(symbol: str):
    d = pd.read_parquet(HISTD.format(_FY[symbol]))
    d["ts"] = pd.to_datetime(d["ts"]); d["date"] = d["ts"].dt.date
    daily = d.set_index("date")["close"]
    daily_open = d.set_index("date")["open"]                             # causal entry ref for open-entry
    rv10 = d.set_index("date")["close"].pct_change().rolling(10).std()   # trailing realised vol
    try:
        b = pd.read_parquet(HIST5.format(_FY[symbol]))
        b["ts"] = pd.to_datetime(b["ts"]); b["date"] = b["ts"].dt.date
        b["t"] = b["ts"].dt.time
        vwap = {}
        for dd, g in b.groupby("date"):
            last = g[g["t"] >= dt.time(15, 0)]
            if len(last):
                vwap[dd] = float((last["close"] * last["volume"]).sum() / max(last["volume"].sum(), 1)) \
                           if last["volume"].sum() > 0 else float(last["close"].mean())
    except Exception:
        vwap = {}
    return daily, daily_open, rv10, vwap


def costs_pct(cost_pct: float) -> float:
    """All-in round-trip cost as a fraction of premium. cost_pct is the spread-inclusive
    assumption (statutory STT 0.1% sell + exch 0.035%×2 + GST + SEBI ≈ 0.3%; the rest is
    bid-ask/slippage). Charged once on the premium (held to settlement = no exit leg)."""
    return cost_pct / 100.0


def _leg_px(g, t_entry, K, opt, col):
    r = g[(g["trade_date"] == t_entry) & (g["strike_price"] == K) & (g["option_type"] == opt)]
    return float(r.iloc[0][col]) if len(r) else None


def simulate(symbol: str, entry: str, cost_pct: float,
             structure: str = "naked", wing_pct: float = 1.0) -> pd.DataFrame:
    """One row per expiry. structure: 'naked' short straddle, or 'ironfly' short straddle
    + long OTM wings at ±wing_pct·spot (DEFINED RISK — caps the tail at the wing)."""
    bh = load_bhav(symbol)
    daily, daily_open, rv10, vwap = underlying(symbol)
    step = STEP[symbol]
    rows = []
    for exp, g in bh.groupby("expiry_date"):
        days = sorted(g["trade_date"].unique())
        if entry == "open":
            t_entry = exp if exp in days else None
        else:
            prior = [d for d in days if d < exp]
            t_entry = prior[-1] if prior else None
        if t_entry is None or t_entry not in daily.index or exp not in daily.index:
            continue
        # entry spot: open-entry uses the expiry-day OPEN (known 09:15), NOT the close, which
        # would pin the strike to the settlement (look-ahead). t-1 uses prior close.
        if entry == "open":
            if t_entry not in daily_open.index:
                continue
            spot_entry = float(daily_open.loc[t_entry])
        else:
            spot_entry = float(daily.loc[t_entry])
        K = round(spot_entry / step) * step
        col = "open_price" if entry == "open" else "close_price"
        ce0, pe0 = _leg_px(g, t_entry, K, "CE", col), _leg_px(g, t_entry, K, "PE", col)
        if not ce0 or not pe0 or (ce0 + pe0) <= 0:
            continue
        straddle = ce0 + pe0
        settle = vwap.get(exp, float(daily.loc[exp]))          # last-30min VWAP, else close
        legs_turnover = straddle                                # premium transacted (for cost)
        if structure == "ironfly":
            W = max(round((wing_pct / 100.0 * spot_entry) / step) * step, step)
            ku, kd = K + W, K - W
            cu, pd_ = _leg_px(g, t_entry, ku, "CE", col), _leg_px(g, t_entry, kd, "PE", col)
            # DATA-QUALITY GUARD: an OTM wing must be cheaper than the ATM same-side leg and
            # priced > 0. Illiquid strikes (FIN/MIDCAP, stale opening prints) violate this and
            # produce a fake NEGATIVE credit → impossible loss > naked. Skip those expiries.
            if cu is None or pd_ is None or cu <= 0 or pd_ <= 0 or cu >= ce0 or pd_ >= pe0:
                continue
            credit = straddle - (cu + pd_)                      # net credit after buying wings
            if credit <= 0:
                continue
            legs_turnover = straddle + cu + pd_
            # payoff at settle: short straddle intrinsic, long wings protect beyond ±W
            short_intr = abs(settle - K)
            long_intr = max(settle - ku, 0.0) + max(kd - settle, 0.0)
            cost = legs_turnover * costs_pct(cost_pct)
            pnl = credit - short_intr + long_intr - cost
            prem = credit
        else:
            cost = legs_turnover * costs_pct(cost_pct)
            pnl = straddle - abs(settle - K) - cost
            prem = straddle
        # VRP selector inputs (inverted IV vs trailing realised vol)
        T = max((exp - t_entry).days, 0.25) / 365.0
        iv = straddle / (_STRADDLE_K * spot_entry * np.sqrt(T)) if spot_entry > 0 else np.nan
        rv = float(rv10.get(t_entry, np.nan)); rv_ann = rv * np.sqrt(252) if rv == rv else np.nan
        vrp = iv / rv_ann if (rv_ann and rv_ann > 0) else np.nan
        rows.append({"symbol": symbol, "expiry": exp, "year": exp.year, "K": K,
                     "spot_entry": spot_entry, "settle": settle, "prem": prem, "cost": cost,
                     "pnl": pnl, "pnl_pct": 100 * pnl / spot_entry, "iv": iv,
                     "rv_ann": rv_ann, "vrp": vrp, "dte": (exp - t_entry).days})
    return pd.DataFrame(rows).sort_values("expiry").reset_index(drop=True)


def add_causal_vrp(df: pd.DataFrame, min_hist: int = 12) -> pd.DataFrame:
    """Tag each expiry RICH/not by its VRP vs the expanding median of PRIOR expiries only
    (causal — the cutoff you could have known live). Pool across symbols by expiry order."""
    df = df.sort_values("expiry").reset_index(drop=True)
    rich = []
    hist: list[float] = []
    for v in df["vrp"].tolist():
        if len(hist) >= min_hist and v == v:
            rich.append(v > float(np.median(hist)))
        else:
            rich.append(False)
        if v == v:
            hist.append(v)
    df["rich_causal"] = rich
    return df


def _ci(x, reps=5000, seed=7):
    x = np.asarray(x, float)
    if len(x) < 4:
        return float(x.mean()) if len(x) else np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(reps)]
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def _risk(s: pd.DataFrame) -> dict:
    p = s["pnl"].to_numpy(float)
    if len(p) == 0:
        return {}
    eq = np.cumsum(p); dd = float((eq - np.maximum.accumulate(eq)).min())
    # worst consecutive-loss run
    run = mx = 0
    for v in p:
        run = run + 1 if v < 0 else 0
        mx = max(mx, run)
    cvar = float(np.mean(np.sort(p)[:max(1, len(p) // 20)]))         # mean of worst 5%
    sharpe = float(p.mean() / p.std()) if p.std() > 0 else np.nan    # per-expiry
    return {"maxDD": dd, "worst_run": mx, "cvar5": cvar, "sharpe": sharpe, "worst": float(p.min())}


def _line(name, s):
    if len(s) == 0 or len(s) < 1:
        return f"  {name:18} n=0"
    m, lo, hi = _ci(s["pnl"])
    win = 100 * (s["pnl"] > 0).mean()
    r = _risk(s)
    ev = "EDGE" if lo > 0 else ("bleed" if hi < 0 else "—")
    return (f"  {name:18} n={len(s):>3}  win {win:4.0f}%  mean {m:+6.1f} "
            f"[{lo:+6.1f},{hi:+6.1f}] {ev:5}  worst {r['worst']:+6.0f}  Sh {r['sharpe']:+.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="NIFTY,BANKNIFTY,FINNIFTY,MIDCPNIFTY")
    ap.add_argument("--entry", default="open", choices=["tminus1", "open"])
    ap.add_argument("--cost-pct", type=float, default=3.0, help="all-in cost %% of premium")
    ap.add_argument("--structure", default="ironfly", choices=["naked", "ironfly"])
    ap.add_argument("--wing-pct", type=float, default=1.0, help="iron-fly wing distance, %% of spot")
    args = ap.parse_args()
    syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip() in STEP]

    print("=" * 92)
    print(f"  VRP HARVEST — REAL bhavcopy | {'+'.join(syms)} | entry={args.entry} | "
          f"{args.structure}" + (f" wing±{args.wing_pct}%" if args.structure == 'ironfly' else "")
          + f" | cost {args.cost_pct}%")
    print("=" * 92)
    parts = []
    for sym in syms:
        s = simulate(sym, args.entry, args.cost_pct, args.structure, args.wing_pct)
        if not s.empty:
            parts.append(s)
    if not parts:
        print("  no expiries"); return
    df = add_causal_vrp(pd.concat(parts, ignore_index=True))
    print(f"  expiries n={len(df)}  ({df.expiry.min()} → {df.expiry.max()})  "
          f"premμ {df.prem.mean():.0f}pt  DTEμ {df.dte.mean():.1f}")

    print("\n  ── P&L per expiry (pts), block-bootstrap CI, risk inline ──")
    print(_line("ALL (pooled)", df))
    d = df.dropna(subset=["vrp"])
    # in-sample tercile (optimistic) vs CAUSAL expanding-median rich (honest)
    if len(d) >= 12:
        q = d["vrp"].quantile([1/3, 2/3]).values
        print(_line("VRP-rich IS top⅓", d[d.vrp > q[1]]))
    print(_line("VRP-rich CAUSAL", df[df.rich_causal]))
    print(_line("  (its complement)", df[~df.rich_causal & df.vrp.notna()]))

    print("\n  ── per index (causal-rich subset) ──")
    for sym in syms:
        print(_line(sym, df[(df.symbol == sym) & df.rich_causal]))

    print("\n  ── per-year stability (causal-rich) ──")
    cr = df[df.rich_causal]
    for y in sorted(df.year.unique()):
        print(_line(str(y), cr[cr.year == y]))

    print("\n  ── naked vs ironfly tail (pooled, causal-rich) ──")
    for st in ("naked", "ironfly"):
        pp = [simulate(sym, args.entry, args.cost_pct, st, args.wing_pct) for sym in syms]
        ddf = add_causal_vrp(pd.concat([p for p in pp if not p.empty], ignore_index=True))
        cr2 = ddf[ddf.rich_causal]; r = _risk(cr2)
        m, lo, hi = _ci(cr2["pnl"])
        print(f"  {st:8} mean {m:+6.1f} [{lo:+.1f},{hi:+.1f}]  worst {r['worst']:+.0f}  "
              f"maxDD {r['maxDD']:+.0f}  CVaR5 {r['cvar5']:+.0f}  Sharpe {r['sharpe']:+.2f}")
    print("\n  READ: deployable ONLY if causal-rich CI clears 0 AND the ironfly (capped-tail)")
    print("  Sharpe stays attractive. If capping the tail kills it, the edge WAS the tail risk.")
    print("\n  READ: real entry premium, no model. Edge real ONLY if ALL/VRP-rich CI clears 0,")
    print("  holds across years, and the tail (maxDD/CVaR) is survivable at the size you'd run.")


if __name__ == "__main__":
    main()
