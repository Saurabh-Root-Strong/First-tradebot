"""
btst_signal.py — PAPER-FIRST live signal for the close-strength BTST edge.

The edge (backtest_btst.py, validated 4yr incl 2026, tradeable fills confirmed):
  A STRONG close (clr = (close−low)/(high−low) ≥ 0.66, top third of the day's range)
  → the index drifts UP overnight. Go LONG index FUTURES at ~15:25, exit next morning
  ~09:30. Long-only (a weak close is NOT a short — the short leg fights the overnight
  drift and adds tail). ~+10–13 bps/night net 3bps, Sharpe ~2.4, win ~55–66%.

This module is the bridge from backtest to reality: it emits tonight's candidates, logs
them to a PAPER ledger, reconciles the next-morning fills, and scores paper-vs-backtest.
NOTHING auto-executes. Deploy real capital only after the paper fills track the backtest
AND a gap-tail sizing/hedge is in place (worst backtest night ≈ −2%; a shock night can
exceed it — this is FUTURES, loss bounded by the gap but real).

Rule is LOCKED (no in-sample tuning): clr≥0.66, long-only, exit 09:30, per index.

    .venv\\Scripts\\python.exe btst_signal.py                 # tonight's candidates
    .venv\\Scripts\\python.exe btst_signal.py --reconcile     # fill yesterday's exits
    .venv\\Scripts\\python.exe btst_signal.py --scorecard     # paper P&L vs backtest
    .venv\\Scripts\\python.exe btst_signal.py --simulate 90   # walk-forward last 90d
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

CLR_TH = 0.66
EXIT_T = dt.time(9, 30)
COST_BPS = 3.0
SYMS = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]
FY = {"NIFTY": "NIFTY50", "BANKNIFTY": "NIFTYBANK", "FINNIFTY": "FINNIFTY", "MIDCPNIFTY": "MIDCPNIFTY"}
DAILY = "data/historical/daily/NSE_{}_INDEX_daily.parquet"
MIN5 = "data/historical/5min/NSE_{}_INDEX_5min.parquet"
LEDGER = Path("data") / "validation" / "btst_paper_ledger.parquet"


def _daily(sym):
    d = pd.read_parquet(DAILY.format(FY[sym]))
    d["ts"] = pd.to_datetime(d["ts"]); d["date"] = d["ts"].dt.date
    return d.sort_values("date").reset_index(drop=True)


def _clr(row):
    rng = row["high"] - row["low"]
    return float((row["close"] - row["low"]) / rng) if rng > 0 else 0.5


def candidates(date):
    """Strong-close BTST-long candidates as of `date`'s close."""
    out = []
    for s in SYMS:
        d = _daily(s)
        r = d[d["date"] == date]
        if r.empty:
            continue
        clr = _clr(r.iloc[0])
        if clr >= CLR_TH:
            out.append({"date": date, "sym": s, "clr": round(clr, 3),
                        "entry_px": float(r.iloc[0]["close"])})
    return out


def _exit_px(sym, next_date):
    """Real ~09:30 exit price on next_date (5min close ≥ 09:30), else that day's open."""
    try:
        b = pd.read_parquet(MIN5.format(FY[sym]))
        b["ts"] = pd.to_datetime(b["ts"])
        g = b[b["ts"].dt.date == next_date]
        if g.empty:
            return None
        aft = g[g["ts"].dt.time >= EXIT_T]
        return float((aft.iloc[0] if len(aft) else g.iloc[0])["close"])
    except Exception:
        return None


def _load_ledger():
    if LEDGER.exists():
        return pd.read_parquet(LEDGER)
    return pd.DataFrame(columns=["date", "sym", "clr", "entry_px", "exit_date",
                                 "exit_px", "gross_bps", "net_bps", "status"])


def _save_ledger(df):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(LEDGER, index=False)


def emit(date):
    cands = candidates(date)
    led = _load_ledger()
    print(f"\n  BTST LONG candidates for close {date} (exit next ~09:30, FUTURES, PAPER):")
    if not cands:
        print("    none — no strong closes (clr≥0.66). Stand aside.")
    for c in cands:
        print(f"    LONG {c['sym']:11} clr {c['clr']:.2f}  entry≈{c['entry_px']:.1f}  "
              f"→ exp +10–13bps  (size for a ~2% gap tail / hedge OTM put)")
        key = (led["date"].astype(str) == str(date)) & (led["sym"] == c["sym"])
        if not key.any():
            led = pd.concat([led, pd.DataFrame([{**c, "status": "OPEN"}])], ignore_index=True)
    _save_ledger(led)
    return cands


def reconcile():
    led = _load_ledger()
    if led.empty:
        print("  ledger empty"); return
    nfut = _daily("NIFTY")["date"].tolist()
    filled = 0
    for i, r in led[led["status"] == "OPEN"].iterrows():
        d = r["date"] if isinstance(r["date"], dt.date) else pd.to_datetime(r["date"]).date()
        nxt = [x for x in nfut if x > d]
        if not nxt:
            continue
        xp = _exit_px(r["sym"], nxt[0])
        if xp is None:
            continue
        gross = (xp / r["entry_px"] - 1.0) * 1e4
        led.at[i, "exit_date"] = nxt[0]; led.at[i, "exit_px"] = xp
        led.at[i, "gross_bps"] = round(gross, 1)
        led.at[i, "net_bps"] = round(gross - COST_BPS, 1)
        led.at[i, "status"] = "CLOSED"
        filled += 1
    _save_ledger(led)
    print(f"  reconciled {filled} position(s).")


def scorecard():
    led = _load_ledger()
    c = led[led["status"] == "CLOSED"]
    print("\n  ── PAPER SCORECARD (closed BTST-long positions) ──")
    if c.empty:
        print("    no closed positions yet — run --simulate or wait for reconcile."); return
    p = c["net_bps"].to_numpy(float)
    eq = np.cumsum(p); dd = (eq - np.maximum.accumulate(eq)).min()
    sh = p.mean() / p.std() * np.sqrt(252) if p.std() > 0 else 0
    print(f"    trades {len(p)}   win {100*(p>0).mean():.0f}%   mean {p.mean():+.1f} bps   "
          f"total {p.sum():+.0f} bps   maxDD {dd:+.0f}   Sharpe {sh:+.2f}   worst {p.min():+.0f}")
    print(f"    backtest expectation ≈ +10–13 bps, Sharpe ~2.4 — paper must track this.")
    for s in SYMS:
        cs = c[c["sym"] == s]
        if len(cs):
            print(f"      {s:11} n={len(cs):>3}  mean {cs['net_bps'].mean():+6.1f}  "
                  f"win {100*(cs['net_bps']>0).mean():3.0f}%")


def simulate(n_days):
    """Walk-forward paper sim over the last n trading days (emit EOD, exit next 09:30)."""
    global LEDGER
    LEDGER = Path("data") / "validation" / "btst_sim_ledger.parquet"
    if LEDGER.exists():
        LEDGER.unlink()
    dates = _daily("NIFTY")["date"].tolist()[-(n_days + 2):]
    for d in dates[:-1]:
        for c in candidates(d):
            led = _load_ledger()
            led = pd.concat([led, pd.DataFrame([{**c, "status": "OPEN"}])], ignore_index=True)
            _save_ledger(led)
    reconcile()
    print(f"\n  WALK-FORWARD PAPER SIM — last {n_days} trading days (rule LOCKED, no tuning):")
    scorecard()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reconcile", action="store_true")
    ap.add_argument("--scorecard", action="store_true")
    ap.add_argument("--simulate", type=int, default=0)
    ap.add_argument("--date", default="", help="emit for a specific YYYY-MM-DD close")
    args = ap.parse_args()
    print("=" * 74)
    print("  BTST close-strength signal — PAPER-FIRST (nothing auto-executes)")
    print("=" * 74)
    if args.simulate:
        simulate(args.simulate); return
    if args.reconcile:
        reconcile()
    if args.scorecard:
        scorecard()
    if not (args.reconcile or args.scorecard):
        last = _daily("NIFTY")["date"].iloc[-1] if not args.date else dt.date.fromisoformat(args.date)
        emit(last)
        print("\n  (paper-logged. run --reconcile next morning, --scorecard to track.)")


if __name__ == "__main__":
    main()
