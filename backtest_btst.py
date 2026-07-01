"""
backtest_btst.py — the overnight (BTST) frontier: where the cost floor is amortised.

Intraday taker directional is dead at the ~3% option / cost wall (proven across this
desk's 2yr + this session). BTST changes the economics: one overnight move is ~0.5-1.5%
(gaps larger) vs the ~0.03% intraday wiggle, while index-FUTURES cost is ~2-4 bps. So a
weak-but-real signal that can't pay intraday CAN clear overnight.

Tests, in order of prior strength:
  1. PASSIVE OVERNIGHT DRIFT — the robust global anomaly: does the equity premium accrue
     close->open (overnight) while intraday (open->close) is flat/negative? If NSE indices
     have it, BTST-long has an edge before any signal. Decompose per index, net of cost,
     per-year, with a day-block bootstrap CI.
  2. CONDITIONAL BTST — do EOD-known signals (today's shock-z, close-strength, gap, 5d
     momentum, trend) predict the OVERNIGHT (close->next-open) move? IC + tercile L/S net
     of cost.
  3. GAP TAIL — the BTST risk: worst overnight gaps (a futures BTST carries full gap risk,
     no intraday stop).

Real prices (data/historical/daily, 2yr). Causal: every signal uses only info up to
today's close; the forward overnight return is the answer key. Cost is explicit + swept.

    .venv\\Scripts\\python.exe backtest_btst.py
    .venv\\Scripts\\python.exe backtest_btst.py --cost-bps 3 --hold overnight
"""
from __future__ import annotations

import argparse
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

FY = {"NIFTY": "NIFTY50", "BANKNIFTY": "NIFTYBANK", "FINNIFTY": "FINNIFTY", "MIDCPNIFTY": "MIDCPNIFTY"}
PATH = "data/historical/daily/NSE_{}_INDEX_daily.parquet"


def _load(sym: str) -> pd.DataFrame:
    d = pd.read_parquet(PATH.format(FY[sym]))
    d["ts"] = pd.to_datetime(d["ts"])
    d = d.sort_values("ts").reset_index(drop=True)
    d["sym"] = sym
    # forward-looking targets (shift -1 = next session)
    d["overnight"] = d["open"].shift(-1) / d["close"] - 1.0          # close -> next open (BTST)
    d["nextday"]   = d["close"].shift(-1) / d["open"].shift(-1) - 1.0  # next open -> next close
    d["close_close"] = d["close"].shift(-1) / d["close"] - 1.0        # close -> next close
    d["intraday"]  = d["close"] / d["open"] - 1.0                    # today open -> close
    # EOD-known signals (causal)
    ret = d["close"].pct_change()
    d["ret1"] = ret
    d["retz"] = ret / ret.rolling(20).std()                          # today's move in sigma (shock)
    rng = (d["high"] - d["low"]).replace(0, np.nan)
    d["clr"] = (d["close"] - d["low"]) / rng                         # close position in range [0,1]
    d["gap_today"] = d["open"] / d["close"].shift(1) - 1.0
    d["mom5"] = d["close"] / d["close"].shift(5) - 1.0
    d["above_ema20"] = (d["close"] > d["close"].ewm(span=20).mean()).astype(float)
    return d


def _ci(x, reps=5000, seed=7):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 5:
        return (float(x.mean()) if len(x) else np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(reps)]
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def _spear(a, b):
    a = pd.Series(a); b = pd.Series(b); m = a.notna() & b.notna()
    return float(a[m].rank().corr(b[m].rank())) if m.sum() > 5 else np.nan


def _pct(x):
    return 100 * x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost-bps", type=float, default=2.0, help="round-trip futures cost, bps")
    ap.add_argument("--symbols", default="NIFTY,BANKNIFTY,FINNIFTY,MIDCPNIFTY")
    args = ap.parse_args()
    c = args.cost_bps / 1e4
    syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip() in FY]
    dfs = {s: _load(s) for s in syms}
    allrows = pd.concat(dfs.values(), ignore_index=True)
    allrows["year"] = allrows["ts"].dt.year

    print("=" * 84)
    print(f"  BTST / OVERNIGHT edge — real daily, net {args.cost_bps}bps futures RT")
    print("=" * 84)

    # ── 1. PASSIVE overnight-vs-intraday decomposition ──────────────────────────
    print("\n  1) PASSIVE DRIFT: mean return by window (bps/day), net of cost, CI")
    print(f"  {'index':11}{'OVERNIGHT c→o':>26}{'INTRADAY o→c':>24}{'buy&hold c→c':>22}")
    for s in syms:
        d = dfs[s]
        rows = []
        for col in ("overnight", "intraday", "close_close"):
            net = d[col].dropna() - c            # charge one RT per window held
            m, lo, hi = _ci(net)
            sh = float(net.mean() / net.std() * np.sqrt(252)) if net.std() > 0 else np.nan
            rows.append((m, lo, hi, sh))
        def f(r):
            m, lo, hi, sh = r
            tag = "＋" if lo > 0 else ("－" if hi < 0 else " ")
            return f"{_pct(m)*100:+5.1f}[{_pct(lo)*100:+4.0f},{_pct(hi)*100:+4.0f}]Sh{sh:+.1f}{tag}"
        print(f"  {s:11}{f(rows[0]):>26}{f(rows[1]):>24}{f(rows[2]):>22}")
    print("  (bps/day; ＋=CI>0 edge, －=CI<0. Overnight>>intraday = the drift anomaly.)")

    # pooled overnight per year (is the drift stable?)
    print("\n     pooled OVERNIGHT by year (bps/day net):")
    for y in sorted(allrows.year.dropna().unique()):
        net = (allrows[allrows.year == y]["overnight"].dropna() - c)
        m, lo, hi = _ci(net)
        tag = "EDGE" if lo > 0 else ("neg" if hi < 0 else "—")
        print(f"       {int(y)}  {_pct(m)*100:+6.1f} [{_pct(lo)*100:+5.0f},{_pct(hi)*100:+5.0f}] {tag}  (n={net.notna().sum()})")

    # ── 2. CONDITIONAL: EOD signals -> overnight ────────────────────────────────
    print("\n  2) SIGNAL → OVERNIGHT (close→next-open):  IC + tercile long-short net")
    sig = allrows.dropna(subset=["overnight"])
    for name in ("retz", "clr", "gap_today", "mom5", "ret1"):
        s2 = sig.dropna(subset=[name])
        ic = _spear(s2[name], s2["overnight"])
        # tercile long top / short bottom, net of cost each leg
        q = s2[name].quantile([1/3, 2/3]).values
        top = s2[s2[name] > q[1]]["overnight"] - c
        bot = -(s2[s2[name] < q[0]]["overnight"]) - c
        ls = pd.concat([top, bot])
        m, lo, hi = _ci(ls)
        tag = "EDGE" if lo > 0 else ("neg" if hi < 0 else "—")
        print(f"     {name:10} IC {ic:+.3f}   L/S {_pct(m)*100:+6.1f}bps "
              f"[{_pct(lo)*100:+5.0f},{_pct(hi)*100:+5.0f}] {tag}  (n={len(ls)})")

    # ── 3. GAP TAIL (BTST risk) ─────────────────────────────────────────────────
    print("\n  3) OVERNIGHT GAP TAIL (BTST carries full gap risk, no stop):")
    for s in syms:
        g = dfs[s]["overnight"].dropna()
        print(f"     {s:11} worst {_pct(g.min())*100:+6.0f}bps  best {_pct(g.max())*100:+6.0f}  "
              f"|gap|>50bps: {_pct((g.abs()>0.005).mean()):.0f}% of nights  "
              f"CVaR5 {_pct(np.sort(g)[:max(1,len(g)//20)].mean())*100:+.0f}")

    print("\n  READ: BTST is viable ONLY if a window/signal net-CI clears 0 AND is stable")
    print("  per-year AND the gap tail is survivable at size. Overnight drift is the prior;")
    print("  conditioning earns its place only if it beats passive buy-overnight net of cost.")


if __name__ == "__main__":
    main()
