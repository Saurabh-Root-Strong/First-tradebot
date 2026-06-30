"""
backtest_regime.py — Does the SL bleed cluster in the CONSOLIDATION mood, and does
                     trading trend-only clear the cost floor?

This is the falsification harness for the trader's thesis (2026-06-30):
  "Market has three moods; consolidation = both-side stop-loss hunting; the arrow
   only works when there is a real trend to ride."

It reuses backtest_engine.py VERBATIM (same causal multi-TF consensus entries, same
R-state-machine, same no-lookahead guarantees) so the trades are identical to the
engine's. The ONLY addition: each trade is tagged with the market MOOD at its entry
(regime_classifier, measured on a higher TF, strictly causally), then the standing
stats are stratified by mood and by trend-alignment.

Reads, per mood:
  • win% / expectancy / net-of-cost  — does CONSOLIDATION underperform?
  • SL share                          — do the stops concentrate there?
  • trend-ALIGNED vs COUNTER vs chop  — is "trade with the mood" the actual edge?

If consolidation is the loss sink AND trend-aligned net-of-cost is positive, the
subtractive product is real: suppress the live arrow in chop. If not, the thesis is
falsified and we don't wire it.

Usage
  .venv\\Scripts\\python.exe backtest_regime.py                       # 4 indices
  .venv\\Scripts\\python.exe backtest_regime.py --symbols NIFTY,BANKNIFTY --cost-bps 3
  .venv\\Scripts\\python.exe backtest_regime.py --regime-tf 60min --er-hi 0.5 --cost-bps 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

import backtest_engine as BE
import regime_classifier as RC
from fno_universe import LABELS, ALL_SYMBOLS

INDEX_DEFAULT = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCAP"]


def _regime_close_table(fy: str, regime_tf: str, n: int, er_hi: float,
                        er_lo: float, drift_big: float) -> pd.DataFrame | None:
    """Mood per regime-TF bar, stamped with the time the bar has CLOSED (causal key)."""
    df = BE._load_tf(fy, regime_tf)
    if df is None or df.empty:
        return None
    tf_min = {"5min": 5, "15min": 15, "60min": 60, "daily": None}[regime_tf]
    moods = RC.classify_series(df, n=n, er_hi=er_hi, er_lo=er_lo, drift_big=drift_big)
    close_ts = (df["ts"] + pd.to_timedelta(tf_min, unit="m") if tf_min
                else df["ts"].dt.normalize() + pd.Timedelta(hours=15, minutes=30))
    out = pd.DataFrame({"close_ts": close_ts.astype("datetime64[ns]"),
                        "mood": moods["mood"].values, "er": moods["er"].values,
                        "sign": moods["sign"].values})
    return out.dropna(subset=["close_ts"]).sort_values("close_ts")


def _alignment(dir_: str, sign: int) -> str:
    """Is the CE/PE trade WITH the mood's drift, against it, or in chop (sign 0)?"""
    if sign == 0:
        return "chop"
    long = dir_ == "CE"
    return "aligned" if (long and sign > 0) or (not long and sign < 0) else "counter"


def _summ(book: pd.DataFrame, scope: str) -> dict:
    s = BE.summarize(book, scope)
    if book.empty:
        return s
    n = len(book)
    sl = int((book["status"] == "SL").sum())
    s["SL%"] = round(100 * sl / n, 1)
    return s


def _print(rows: list[dict]) -> None:
    rows = [r for r in rows if r.get("trades", 0) > 0]
    if not rows:
        print("  (no trades)"); return
    cols = ["scope", "trades", "win%", "SL%", "expectancy_R", "total_R",
            "profit_factor", "avg_win_R", "avg_loss_R", "max_dd_R"]
    w = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    print("  " + "  ".join(c.ljust(w[c]) for c in cols))
    print("  " + "  ".join("-" * w[c] for c in cols))
    for r in rows:
        print("  " + "  ".join(str(r.get(c, "")).ljust(w[c]) for c in cols))


def main() -> None:
    ap = argparse.ArgumentParser(description="Mood-stratified backtest of the engine")
    ap.add_argument("--symbols", default="", help="Comma underlyings. Default: 4 indices")
    ap.add_argument("--regime-tf", default="15min", choices=["5min", "15min", "60min", "daily"],
                    help="Timeframe the mood is measured on. Default: 15min")
    ap.add_argument("--er-window", type=int, default=RC.ER_WINDOW)
    ap.add_argument("--er-hi", type=float, default=RC.ER_HI)
    ap.add_argument("--er-lo", type=float, default=RC.ER_LO)
    ap.add_argument("--drift-big", type=float, default=RC.DRIFT_BIG)
    # engine entry geometry — same knobs as backtest_engine
    ap.add_argument("--min-band", choices=["MILD", "BUY", "STRONG"], default="BUY")
    ap.add_argument("--min-agree", type=int, default=2)
    ap.add_argument("--sl-atr", type=float, default=1.5)
    ap.add_argument("--t1-r", type=float, default=1.5)
    ap.add_argument("--t2-r", type=float, default=3.0)
    ap.add_argument("--cooldown-min", type=int, default=10)
    ap.add_argument("--cost-bps", type=float, default=0.0)
    args = ap.parse_args()

    rank = {"MILD": 1, "BUY": 2, "STRONG": 3}[args.min_band]
    cfg = {"min_rank": rank, "min_agree": args.min_agree, "sl_atr": args.sl_atr,
           "t1_r": args.t1_r, "t2_r": args.t2_r, "cooldown_min": args.cooldown_min}

    rev = {v: k for k, v in LABELS.items()}
    if args.symbols.strip():
        wanted = [s.strip().upper() for s in args.symbols.split(",")]
        fy_syms = [rev[s] if s in rev else s for s in wanted if s in rev or s in LABELS]
    else:
        fy_syms = [rev[s] for s in INDEX_DEFAULT if s in rev]

    print("=" * 72)
    print("  MOOD-STRATIFIED ENGINE BACKTEST  (does the bleed live in CHOP?)")
    print("=" * 72)
    print(f"  Symbols   : {', '.join(LABELS.get(s, s) for s in fy_syms)}")
    print(f"  Mood      : ER{args.er_window} on {args.regime_tf} | "
          f"er_hi {args.er_hi} er_lo {args.er_lo} drift_big {args.drift_big} ATR")
    print(f"  Entry     : band>={args.min_band}, >={args.min_agree} TFs | "
          f"SL {args.sl_atr}ATR T1 {args.t1_r}R T2 {args.t2_r}R")
    print(f"  Cost      : {args.cost_bps:.1f} bps ({'GROSS' if args.cost_bps == 0 else 'NET'})")
    print("=" * 72)

    books = []
    for fy in fy_syms:
        frames = {}
        for tf in BE.TF_KEYS:
            d = BE._load_tf(fy, tf)
            if d is None or d.empty:
                frames = {}; break
            frames[tf] = d
        if not frames:
            print(f"  {LABELS.get(fy, fy):<12} missing data — skip"); continue
        grid = BE.build_signal_grid(frames)
        tr = pd.DataFrame(BE.simulate(grid, LABELS.get(fy, fy), fy, cfg))
        if tr.empty:
            print(f"  {LABELS.get(fy, fy):<12} 0 trades"); continue
        # cost in R (identical formula to backtest_engine.main)
        if args.cost_bps > 0:
            risk_px = args.sl_atr * tr["atr"]
            tr["r"] = (tr["r"] - (tr["entry_px"] * args.cost_bps / 1e4)
                       / risk_px.replace(0, np.nan)).round(3)
        # tag mood as-of entry (causal merge on regime-bar close time)
        rt = _regime_close_table(fy, args.regime_tf, args.er_window,
                                 args.er_hi, args.er_lo, args.drift_big)
        tr = tr.sort_values("entry_ts")
        if rt is not None:
            tr = pd.merge_asof(tr, rt, left_on="entry_ts", right_on="close_ts",
                               direction="backward")
        tr["mood"] = tr["mood"].fillna(RC.CHOP)
        tr["sign"] = tr["sign"].fillna(0).astype(int)
        tr["align"] = [_alignment(d, s) for d, s in zip(tr["dir"], tr["sign"])]
        books.append(tr)
        print(f"  {LABELS.get(fy, fy):<12} {len(tr):>5} trades tagged")

    if not books:
        print("\n  No trades."); return
    book = pd.concat(books, ignore_index=True)

    order = [RC.BIG_UP, RC.BIG_DOWN, RC.SMALL_UP, RC.SMALL_DOWN, RC.CHOP]
    print("\n" + "=" * 72)
    print("  BY MOOD   (the headline test — is CHOP the loss sink?)")
    print("=" * 72)
    _print([_summ(book[book["mood"] == m], m) for m in order])

    print("\n" + "=" * 72)
    print("  COLLAPSED   trend (BIG+small) vs CONSOLIDATION")
    print("=" * 72)
    is_trend = book["mood"] != RC.CHOP
    _print([_summ(book[is_trend], "ALL TREND"),
            _summ(book[~is_trend], "CONSOLIDATION"),
            _summ(book, "ALL (trade-always)")])

    print("\n" + "=" * 72)
    print("  BY ALIGNMENT   (trade WITH the mood vs against vs chop)")
    print("=" * 72)
    _print([_summ(book[book["align"] == a], a) for a in ("aligned", "counter", "chop")])

    print("\n" + "=" * 72)
    print("  BIG-TREND, ALIGNED ONLY   (the user's actual rule: ride a real trend)")
    print("=" * 72)
    big_aligned = book[(book["mood"].isin(RC.BIG)) & (book["align"] == "aligned")]
    _print([_summ(big_aligned, "BIG+aligned"),
            _summ(book[book["align"] == "aligned"], "any-trend+aligned")])

    # mood distribution sanity
    print("\n  mood distribution:")
    dist = book["mood"].value_counts()
    for m in order:
        c = int(dist.get(m, 0))
        print(f"    {m:<16} {c:>6}  ({100*c/len(book):4.1f}%)")
    print()


if __name__ == "__main__":
    main()
