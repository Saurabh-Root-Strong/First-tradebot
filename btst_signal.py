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


def _bar_from_mirror(sym, date):
    """Session OHLC (09:15–15:30 clr + close) from the intraday TICK MIRROR — the fallback
    for TODAY when the historical daily archive hasn't been downloaded yet (no token needed).
    Without this, an unattended EOD emit re-emits the stale last-archived day forever."""
    try:
        import datetime as _dt
        from core.mirror_io import read_mirror
        tk = read_mirror("ticks", date.isoformat(), None, f"NSE:{FY[sym]}-INDEX")
        if tk is None or len(tk) < 30:
            return None
        tk = tk[tk["ts"].dt.date == date]                          # drop cross-midnight bleed
        m = tk[(tk["ts"].dt.time >= _dt.time(9, 15)) & (tk["ts"].dt.time <= _dt.time(15, 30))]
        if len(m) < 30:
            return None
        ltp = m["ltp"].to_numpy(float)
        hi, lo, close = float(ltp.max()), float(ltp.min()), float(m.iloc[-1]["ltp"])
        return {"clr": (close - lo) / (hi - lo) if hi > lo else 0.5, "close": close}
    except Exception:
        return None


def candidates(date):
    """Strong-close BTST-long candidates as of `date`'s close. Uses the historical daily
    archive; falls back to the tick mirror for a not-yet-archived day (e.g. today, offline)."""
    out = []
    for s in SYMS:
        try:
            d = _daily(s)
            r = d[d["date"] == date]
        except Exception:
            r = None
        if r is not None and not r.empty:
            clr, close = _clr(r.iloc[0]), float(r.iloc[0]["close"])
        else:
            bar = _bar_from_mirror(s, date)
            if not bar:
                continue
            clr, close = bar["clr"], bar["close"]
        if clr >= CLR_TH:
            out.append({"date": date, "sym": s, "clr": round(clr, 3),
                        "entry_px": round(close, 2)})
    return out


def _exit_px(sym, next_date):
    """Real ~09:30 exit price on next_date. Tries the historical 5min archive first; if that
    day isn't downloaded yet (the archive lags the live capture by a day or two, which is why
    the paper loop silently stalled with positions stuck OPEN), FALLS BACK to the lock-free
    tick mirror. Self-healing + fully offline — the loop no longer needs a same-day download."""
    try:
        b = pd.read_parquet(MIN5.format(FY[sym]))
        b["ts"] = pd.to_datetime(b["ts"])
        g = b[b["ts"].dt.date == next_date]
        if not g.empty:
            aft = g[g["ts"].dt.time >= EXIT_T]
            return float((aft.iloc[0] if len(aft) else g.iloc[0])["close"])
    except Exception:
        pass
    return _exit_px_from_mirror(sym, next_date)


def _exit_px_from_mirror(sym, next_date):
    """Fallback exit: first tick at/after 09:30 from the intraday tick mirror."""
    try:
        from core.mirror_io import read_mirror
        msym = f"NSE:{FY[sym]}-INDEX"
        tk = read_mirror("ticks", next_date.isoformat(), None, msym)
        if tk is None or not len(tk):
            return None
        tk = tk[tk["ts"].dt.date == next_date]   # drop prior-evening bleed (mirror spans midnight)
        aft = tk[tk["ts"].dt.time >= EXIT_T]
        src = aft if len(aft) else tk
        return float(src.iloc[0]["ltp"]) if len(src) else None
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
        if key.any():
            # re-emit same day (e.g. an earlier run) → UPDATE the OPEN entry to the latest
            # clr/entry, so a 15:20 run corrects a premature morning run. Never touch CLOSED.
            m = key & (led["status"] == "OPEN")
            led.loc[m, "clr"] = c["clr"]; led.loc[m, "entry_px"] = c["entry_px"]
        else:
            led = pd.concat([led, pd.DataFrame([{**c, "status": "OPEN"}])], ignore_index=True)
    _save_ledger(led)
    return cands


def _next_trading_day(d: dt.date) -> dt.date | None:
    """Next NSE trading day after d (calendar-driven, capped at today). Independent of the
    historical archive — which lags live capture, the bug that stalled reconcile."""
    from core.market_calendar import is_trading_day
    nxt = d + dt.timedelta(days=1)
    today = dt.date.today()
    while nxt <= today:
        if is_trading_day(nxt):
            return nxt
        nxt += dt.timedelta(days=1)
    return None


def reconcile():
    led = _load_ledger()
    if led.empty:
        print("  ledger empty"); return
    filled = 0
    for i, r in led[led["status"] == "OPEN"].iterrows():
        d = r["date"] if isinstance(r["date"], dt.date) else pd.to_datetime(r["date"]).date()
        nd = _next_trading_day(d)
        if nd is None:
            continue                       # exit day hasn't happened yet
        nxt = [nd]
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


def _next_trading_day(d: dt.date) -> dt.date:
    from core.market_calendar import is_trading_day
    n = d + dt.timedelta(days=1)
    while not is_trading_day(n):
        n += dt.timedelta(days=1)
    return n


def _as_date(x) -> dt.date:
    return x if isinstance(x, dt.date) and not isinstance(x, dt.datetime) else \
        dt.date.fromisoformat(str(x)[:10])


def open_positions(led=None):
    """Split OPEN rows into (fresh, STALE). STALE = the exit day has already passed, i.e.
    --reconcile never ran.

    WHY THIS MATTERS: the scorecard only sums CLOSED rows, so a stale-open position is
    SILENTLY EXCLUDED from the paper P&L. A broken morning job therefore FLATTERS the record
    by dropping trades — and it did: 2026-07-07..10 the morning task was disabled/failing
    (0x80070002), one MIDCPNIFTY position sat unreconciled, and the card read
    'win 83%, mean +14.3 bps'. Reconciled, that trade was -39.6 bps (the worst in the ledger)
    and the truth was 'win 71%, mean +6.6 bps' — BELOW the +10-13 bps backtest expectation.
    Never let an unreconciled position be invisible."""
    led = _load_ledger() if led is None else led
    o = led[led["status"] == "OPEN"]
    if o.empty:
        return o, o
    from core.constants import IST
    today = dt.datetime.now(IST).date()
    stale = o["date"].map(lambda x: _next_trading_day(_as_date(x)) < today)
    return o[~stale], o[stale]


def _report_open(led) -> int:
    """Print open/stale positions. Returns the STALE count (0 = ledger integrity intact)."""
    fresh, stale = open_positions(led)
    if len(stale):
        print(f"\n  ⚠ {len(stale)} STALE-OPEN position(s) — the exit day passed but "
              "--reconcile never ran.")
        print("    They are EXCLUDED from the P&L below, which therefore OVERSTATES the edge.")
        for r in stale.itertuples():
            print(f"      {_as_date(r.date)}  {r.sym}  clr={r.clr}")
        print("    FIX: .venv\\Scripts\\python.exe btst_signal.py --reconcile")
    elif len(fresh):
        print(f"\n  {len(fresh)} open position(s) awaiting tomorrow's exit (normal).")
    return len(stale)


def scorecard():
    led = _load_ledger()
    _report_open(led)
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
    _health_monitor(c)


def _health_monitor(c, window: int = 20) -> None:
    """The HONEST memory for a LOCKED +EV rule: don't tune it — watch it for DECAY. A
    mechanical edge dies when the regime that carried it turns (BTST rides overnight risk-on
    drift; a risk-off month flips it). Walk-forward on the sim ledger PROVED sub-cell sizing
    (clr bucket / per-index) is overfit noise — the only thing worth learning is whether the
    edge is still ALIVE. Compares the trailing window vs the full history and flags decay."""
    if "date" in c.columns:
        c = c.sort_values("date")
    p = c["net_bps"].to_numpy(float)
    if len(p) < window + 5:
        print(f"\n    edge-health: n={len(p)} (<{window+5}) — accruing, no decay read yet")
        return
    import numpy as np
    full_m = p.mean()
    rec = p[-window:]
    rec_m, rec_wr = rec.mean(), 100 * (rec > 0).mean()
    rec_sh = rec.mean() / rec.std() * np.sqrt(252) if rec.std() > 0 else float("nan")
    # decay = trailing window has gone negative, or lost >60% of the full-sample edge
    decayed = rec_m <= 0 or (full_m > 0 and rec_m < 0.4 * full_m)
    flag = "⚠ DECAY — stand aside / re-backtest" if decayed else "✓ edge intact"
    print(f"\n    edge-health (last {window} vs full): trailing mean {rec_m:+.1f}bps "
          f"(full {full_m:+.1f})  win {rec_wr:.0f}%  Sharpe {rec_sh:+.2f}   {flag}")


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
    ap.add_argument("--check", action="store_true",
                    help="ledger-integrity guard: exit 1 if any position is STALE-OPEN "
                         "(exit day passed, --reconcile never ran). For the weekly cron.")
    args = ap.parse_args()
    print("=" * 74)
    print("  BTST close-strength signal — PAPER-FIRST (nothing auto-executes)")
    print("=" * 74)
    if args.check:
        # A stale-open position is silently dropped from the scorecard -> the paper record
        # overstates the edge. This is the ledger's integrity invariant; fail loudly.
        n = _report_open(_load_ledger())
        print("\n  LEDGER INTEGRITY: " + ("OK — no stale-open positions." if not n
                                          else f"BROKEN — {n} stale-open (reconcile did not run)."))
        sys.exit(1 if n else 0)
    if args.simulate:
        simulate(args.simulate); return
    if args.reconcile:
        reconcile()
    if args.scorecard:
        scorecard()
    if not (args.reconcile or args.scorecard):
        emit(_emit_date(args.date))
        print("\n  (paper-logged. run --reconcile next morning, --scorecard to track.)")


def _emit_date(explicit: str) -> dt.date:
    """The close to emit for. Explicit --date wins; else TODAY if it is a trading day (the
    unattended EOD run's intent — uses the tick-mirror fallback if the archive lags), else the
    most recent trading day. Never the stale last-archived date (the old re-emit-07-01 bug)."""
    if explicit:
        return dt.date.fromisoformat(explicit)
    from core.market_calendar import is_trading_day
    d = dt.date.today()
    for _ in range(7):
        if is_trading_day(d):
            return d
        d -= dt.timedelta(days=1)
    return d


if __name__ == "__main__":
    main()
