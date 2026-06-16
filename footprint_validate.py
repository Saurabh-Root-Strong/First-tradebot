"""
footprint_validate.py — accumulating, honest validation of the smart-money footprint.

The intraday footprint (smart_money.py) is the one directional signal we have NOT been
able to test — it needs many captured option-chain days, and we only have a few. This
builds the answer the disciplined way:

  HARVEST  (append-only): for each COMPLETE captured day, replay the footprint at a few
           checkpoints (lookahead-free, as_of), record the realized forward index move,
           and append one row per (day, checkpoint, index) to a durable ledger. Idempotent
           + catch-up: re-running back-fills any un-harvested day, never double-counts.

  VERDICT: over the whole ledger, test the pre-registered primary hypothesis — does the
           footprint DIRECTION predict the forward index return, net of costs — with
           DAY-clustered counts (a day's checkpoints are correlated, so they are NOT
           independent obs) and a hard POWER GATE: no verdict until >= MIN_DAYS sessions.

So today it will say "ACCUMULATING — N/30 sessions, locked", and it becomes conclusive on
its own as the capture grows. Run --harvest daily (supervise.py does this post-close).

  .venv\\Scripts\\python.exe footprint_validate.py --harvest   # back-fill + add today
  .venv\\Scripts\\python.exe footprint_validate.py             # show the verdict / gate
"""
from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

import smart_money as SM

IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
LIVE_DIR = Path(__file__).parent / "data" / "intraday" / "live"
LEDGER = Path(__file__).parent / "data" / "validation" / "footprint_ledger.parquet"
INDEX_SYMBOLS = SM.INDEX_SYMBOLS
LABELS = SM.LABELS

CHECKPOINTS = [datetime.time(*t) for t in [(10, 30), (11, 30), (12, 30), (13, 30), (14, 15)]]
HORIZONS = [30, 60]                       # forward minutes (capped at session end)
SESS_END = datetime.time(15, 25)
COST_PCT = 0.05                           # ~5 bps futures round-trip (index move proxy)
MIN_DAYS = 30                             # power gate — no verdict below this
ROBUST_DAYS = 50


def _ticks(date: str, sym: str) -> pd.DataFrame | None:
    p = LIVE_DIR / f"{date}_ticks.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df = df[df["symbol"] == sym].copy()
    if df.empty:
        return None
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(IST)
    return df.sort_values("ts")


def _px_at(df: pd.DataFrame, t: datetime.datetime) -> float | None:
    s = df[df["ts"] <= pd.Timestamp(t)]
    return float(s["ltp"].iloc[-1]) if len(s) else None


def _complete_days() -> list[str]:
    """Captured days whose session is fully over (so forward windows exist)."""
    days = sorted({p.name[:10] for p in LIVE_DIR.glob("*_chain_snapshots.parquet")})
    today = datetime.datetime.now(tz=IST)
    out = []
    for d in days:
        done = d < today.date().isoformat() or today.time() > datetime.time(15, 35)
        if done:
            out.append(d)
    return out


def harvest() -> int:
    led = pd.read_parquet(LEDGER) if LEDGER.exists() else pd.DataFrame()
    have = set(led["date"].unique()) if len(led) else set()
    rows = []
    for date in _complete_days():
        if date in have:
            continue
        d = datetime.date.fromisoformat(date)
        ticks = {s: _ticks(date, s) for s in INDEX_SYMBOLS}
        for ct in CHECKPOINTS:
            as_of = datetime.datetime.combine(d, ct, tzinfo=IST)
            for sym in INDEX_SYMBOLS:
                fp = SM.footprint_index(sym, date=date, as_of=as_of)
                if not fp.get("has_data") or fp["direction"] == "THIN":
                    continue
                tk = ticks.get(sym)
                if tk is None:
                    continue
                p0 = _px_at(tk, as_of)
                if not p0:
                    continue
                rec = {"date": date, "time": ct.strftime("%H:%M"), "sym": sym,
                       "direction": fp["direction"], "lean": fp["lean"],
                       "leads": bool(fp["leads"]), "conviction": fp["conviction"],
                       "build_L": fp.get("build_L")}
                for h in HORIZONS:
                    t1 = min(as_of + datetime.timedelta(minutes=h),
                             datetime.datetime.combine(d, SESS_END, tzinfo=IST))
                    p1 = _px_at(tk, t1)
                    rec[f"fwd{h}"] = (p1 - p0) / p0 * 100 if (p1 and t1 > as_of) else np.nan
                rows.append(rec)
    if not rows:
        return 0
    out = pd.concat([led, pd.DataFrame(rows)], ignore_index=True) if len(led) else pd.DataFrame(rows)
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(LEDGER, index=False)
    return len(rows)


# ── Verdict ─────────────────────────────────────────────────────────────────────

def _seg(df: pd.DataFrame, h: int) -> dict:
    d = df.dropna(subset=[f"fwd{h}"])
    if d.empty:
        return {"n": 0, "days": 0}
    sgn = d["direction"].map({"BULLISH": 1, "BEARISH": -1})
    fwd = d[f"fwd{h}"]
    # rank-IC only when both have spread and the sample is non-trivial
    ic = (d["lean"].rank().corr(fwd.rank())
          if len(d) >= 20 and d["lean"].nunique() > 1 and fwd.nunique() > 1 else float("nan"))
    hit = float((np.sign(sgn) == np.sign(fwd)).mean() * 100)
    signed = (sgn * fwd)                    # gross directional edge, %
    return {"n": len(d), "days": d["date"].nunique(),
            "IC": round(ic, 3), "hit%": round(hit, 1),
            "gross%": round(signed.mean(), 3),
            "net%": round(signed.mean() - COST_PCT, 3)}


def verdict() -> None:
    if not LEDGER.exists():
        print("No ledger yet — run with --harvest first."); return
    led = pd.read_parquet(LEDGER)
    n_days = led["date"].nunique()
    print("=" * 80)
    print("  SMART-MONEY FOOTPRINT — VALIDATION SCOREBOARD")
    print("=" * 80)
    print(f"  ledger: {len(led):,} signals over {n_days} sessions "
          f"({led['date'].min()} … {led['date'].max()})")
    gate = n_days >= MIN_DAYS
    bar = "#" * int(min(n_days / MIN_DAYS, 1) * 30)
    print(f"  power:  [{bar:<30}] {n_days}/{MIN_DAYS} sessions  "
          f"{'✅ UNLOCKED' if gate else '🔒 LOCKED — accumulating'}")
    print()

    rows = [("ALL", led)] + [(LABELS[s], led[led["sym"] == s]) for s in INDEX_SYMBOLS]
    rows.append(("LEADS-only", led[led["leads"]]))
    for h in HORIZONS:
        print(f"  ── forward {h} min ──")
        print(f"     {'segment':<13}{'days':>5}{'n':>6}{'IC':>8}{'hit%':>7}{'gross%':>9}{'net%':>8}")
        for name, sub in rows:
            s = _seg(sub, h)
            if s["n"] == 0:
                continue
            print(f"     {name:<13}{s['days']:>5}{s['n']:>6}{s.get('IC',float('nan')):>8}"
                  f"{s.get('hit%',0):>7}{s.get('gross%',0):>+9}{s.get('net%',0):>+8}")
        print()

    print("=" * 80)
    if not gate:
        print(f"  🔒 VERDICT LOCKED — {n_days}/{MIN_DAYS} sessions. Numbers above are provisional")
        print(f"     and NOT to be acted on (day-count too low for significance). They unlock at")
        print(f"     {MIN_DAYS} sessions (preliminary) / {ROBUST_DAYS}+ (robust). Keep --harvest running.")
    else:
        print("  Primary test: footprint DIRECTION → forward return, net of costs.")
        print("  Real edge ⇒ IC clearly > 2·SE (SE≈1/√days), hit%>52, and net% > 0 consistently")
        print("  across indices + horizons. One lucky cell ≠ edge — demand consistency (see FII study).")
    print("=" * 80)


def main() -> None:
    ap = argparse.ArgumentParser(description="Accumulating footprint validation")
    ap.add_argument("--harvest", action="store_true", help="back-fill ledger from captured days")
    args = ap.parse_args()
    if args.harvest:
        n = harvest()
        print(f"[footprint_validate] harvested {n} new signal rows; "
              f"ledger sessions = {pd.read_parquet(LEDGER)['date'].nunique() if LEDGER.exists() else 0}")
    else:
        verdict()


if __name__ == "__main__":
    main()
