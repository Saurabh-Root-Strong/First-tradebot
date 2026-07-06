"""
backtest_btst_recent.py — the BTST close-strength edge on THIS WEEK's captured tick data.

BTST is an OVERNIGHT trade, not intraday: a STRONG close (clr≥0.66) → go LONG the index at
the ~15:30 close, exit next morning ~09:30. So the "60-minute" framing of the scout backtest
does not apply — the honest recent-week grade is the close→next-open hold, computed straight
from the tick mirrors (fully offline, exactly the today+last-week sessions requested).

Per night it reports what you asked, mapped to an overnight hold:
  • WHEN GIVEN   — the close instant the signal fires (fixed EOD, ~15:30)
  • ACCURACY     — win rate (overnight net > 0)
  • TARGET hit   — nights the overnight gap went your way (net>0)   [BTST has no intraday
  • "SL" hit     — nights it went against you (net<0)                stop; this is the
  • TAIL         — worst nights (the ~2% gap-down risk to size for)  overnight OUTCOME]

Also grades the clr≥0.66 GATE: strong-close nights vs weak-close nights overnight drift, so
you see whether the gate actually selected the winners on this slice.

    .venv\\Scripts\\python.exe backtest_btst_recent.py
"""
from __future__ import annotations

import datetime
import sys

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from core.constants import IST, INDEX_SYMBOLS, LIVE_DIR
from core.market_calendar import is_trading_day
from core.mirror_io import read_mirror as _read

CLR_TH = 0.66
COST_BPS = 3.0
OPEN_T = datetime.time(9, 15)
CLOSE_T = datetime.time(15, 30)
EXIT_T = datetime.time(9, 30)


def _sessions(n_back: int = 9) -> list[str]:
    """Recent captured TRADING days (real tick mirror, holidays/weekends dropped)."""
    days = []
    for p in sorted(LIVE_DIR.glob("*_ticks.parquet")):
        stem = p.name.split("_")[0]
        try:
            d = datetime.date.fromisoformat(stem)
        except ValueError:
            continue
        if p.stat().st_size < 200_000 or not is_trading_day(d):
            continue
        days.append(stem)
    return days[-n_back:]


def _session_bar(sym: str, day: str) -> dict | None:
    """OHLC of the 09:15–15:30 cash session from the tick mirror."""
    tk = _read("ticks", day, None, sym)
    if tk is None or len(tk) < 30:
        return None
    d0 = datetime.date.fromisoformat(day)
    tk = tk[tk["ts"].dt.date == d0]                   # drop prior-evening / cross-midnight bleed
    m = tk[(tk["ts"].dt.time >= OPEN_T) & (tk["ts"].dt.time <= CLOSE_T)]
    if len(m) < 30:
        return None
    ltp = m["ltp"].to_numpy(float)
    hi, lo = float(ltp.max()), float(ltp.min())
    close = float(m.iloc[-1]["ltp"])
    clr = (close - lo) / (hi - lo) if hi > lo else 0.5
    return {"open": float(m.iloc[0]["ltp"]), "high": hi, "low": lo, "close": close,
            "clr": clr, "close_t": m.iloc[-1]["ts"]}


def _next_open_px(sym: str, day: str) -> float | None:
    """First tick at/after 09:30 on `day` (the BTST exit)."""
    tk = _read("ticks", day, None, sym)
    if tk is None or not len(tk):
        return None
    tk = tk[tk["ts"].dt.date == datetime.date.fromisoformat(day)]   # drop cross-midnight bleed
    aft = tk[tk["ts"].dt.time >= EXIT_T]
    src = aft if len(aft) else tk[tk["ts"].dt.time >= OPEN_T]
    return float(src.iloc[0]["ltp"]) if len(src) else None


def run() -> pd.DataFrame:
    days = _sessions()
    rows = []
    for i, day in enumerate(days[:-1]):
        nxt = days[i + 1]
        for sym in INDEX_SYMBOLS:
            bar = _session_bar(sym, day)
            if not bar:
                continue
            exit_px = _next_open_px(sym, nxt)
            if not exit_px:
                continue
            gross = (exit_px / bar["close"] - 1.0) * 1e4
            rows.append({
                "close_day": day, "exit_day": nxt, "sym": sym,
                "close_t": bar["close_t"].strftime("%H:%M"),
                "clr": round(bar["clr"], 3), "strong": bar["clr"] >= CLR_TH,
                "entry": round(bar["close"], 1), "exit": round(exit_px, 1),
                "gross_bps": round(gross, 1), "net_bps": round(gross - COST_BPS, 1),
            })
    return pd.DataFrame(rows)


def _grade(p: np.ndarray) -> str:
    if not len(p):
        return "n=0"
    sh = p.mean() / p.std() * np.sqrt(252) if (len(p) > 1 and p.std() > 0) else float("nan")
    return (f"n={len(p):<3d} win {100*(p>0).mean():3.0f}%  mean {p.mean():+6.1f}bps  "
            f"total {p.sum():+6.0f}  Sharpe {sh:+.2f}  worst {p.min():+.0f}")


def report(df: pd.DataFrame) -> None:
    print("\n" + "=" * 88)
    print("BTST close-strength — THIS WEEK's captured tick data (overnight close→09:30 hold)")
    print("=" * 88)
    if df.empty:
        print("  no overnight pairs in the captured window"); return
    nights = sorted(df.close_day.unique())
    print(f"  close-nights={len(nights)} ({nights[0]}..{nights[-1]})  indices={df.sym.nunique()}  "
          f"overnight legs={len(df)}")

    strong = df[df.strong]
    print(f"\n  SIGNALS FIRED (clr≥{CLR_TH}) = {len(strong)} of {len(df)} possible closes")
    if len(strong):
        p = strong.net_bps.to_numpy(float)
        tgt = int((p > 0).sum()); sl = int((p < 0).sum())
        print("\n  ACCURACY / TARGET / SL  (overnight outcome — BTST holds, no intraday stop)")
        print(f"    win rate        {100*(p>0).mean():.1f}%   ({tgt} TARGET / {sl} adverse 'SL')")
        print(f"    mean net        {p.mean():+.1f} bps   median {np.median(p):+.1f}   "
              f"total {p.sum():+.0f} bps")
        sh = p.mean()/p.std()*np.sqrt(252) if (len(p) > 1 and p.std() > 0) else float('nan')
        print(f"    Sharpe (annl.)  {sh:+.2f}   best {p.max():+.0f}  worst {p.min():+.0f} (the gap tail)")
        print(f"    WHEN GIVEN      at the close ~{strong.close_t.mode().iloc[0]} "
              f"(fixed EOD entry), exit next ~09:30")

        print("\n  EVERY SIGNAL (the trade log)")
        for _, r in strong.sort_values(["close_day", "sym"]).iterrows():
            tag = "TARGET" if r.net_bps > 0 else "  SL  "
            print(f"    {r.close_day}→{r.exit_day[5:]}  {r.sym.split(':')[1][:9]:9} "
                  f"clr {r.clr:.2f}  {r.entry:>9.1f}→{r.exit:<9.1f}  {r.net_bps:+7.1f}bps  {tag}")

        print("\n  BY INDEX")
        for s in INDEX_SYMBOLS:
            q = strong[strong.sym == s].net_bps.to_numpy(float)
            print(f"    {s.split(':')[1]:14} {_grade(q)}")

    # gate check — did clr≥0.66 actually pick the overnight winners this slice?
    print("\n  GATE CHECK — strong-close vs weak-close overnight drift (does the gate select?)")
    for lbl, sub in (("strong clr≥0.66", df[df.strong]), ("weak   clr<0.66", df[~df.strong])):
        q = sub.net_bps.to_numpy(float)
        print(f"    {lbl:16} {_grade(q)}")

    print("\n" + "=" * 88)
    print("READ: one week ≈ %d signals is a TINY sample — CIs are enormous; this is a live-data"
          % len(strong))
    print("sanity check, NOT a verdict (this week was risk-on: even WEAK closes drifted up). The")
    print("159-night walk-forward sim is the anchor: +11.9 bps/night flat clr≥0.66, Sharpe ~1.66.")
    print("NOTE: sub-cell sizing (clr-bucket / per-index) INVERTS out-of-sample — keep the rule")
    print("FLAT + LOCKED; the only honest 'memory' is the edge-decay monitor, not a size knob.")


if __name__ == "__main__":
    df = run()
    report(df)
