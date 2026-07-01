"""
intraday_read.py — the LIVE intraday cockpit (your operating loop).

Intraday-only, honest. For each index at instant t it reads the same causal series the
scout/charts use (footprint_chart.build_series, ts≤t, lookahead-free) and reports:

  • REGIME   — OPENING / NORMAL (~79%) / HIGH_VOL (~21%)  (feature_engine, 3-regime)
  • BAND     — the ~70%-reliable range (hour_forecast): the risk map / where the stop lives
  • ACTION   — the HONEST call: trade the band / no directional bet / size down. There is NO
               intraday directional arrow here — it loses (58-75% wrong at cost).
  • CARRY    — AFTER 15:00 ONLY: if a STRONG CLOSE is forming (top third of the day's range),
               flag the validated BTST-LONG (close-strength → overnight, exit next ~09:30).
               This is the sole path from intraday to BTST — a late signal with no time to
               finish intraday. A weak close after 15:00 = no trade.

Runs LIVE (t=now) or REPLAY (a past captured day / instant) with identical logic.

    .venv\\Scripts\\python.exe intraday_read.py                    # live, all indices, 5m
    .venv\\Scripts\\python.exe intraday_read.py 2026-06-25 14:30   # replay a past instant
"""
from __future__ import annotations

import datetime as dt
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

import footprint_chart as fc
import feature_engine as FE
import hour_forecast as hf
from core.constants import INDEX_SYMBOLS, LABELS, IST

FY = {"NSE:NIFTY50-INDEX": "NIFTY50", "NSE:NIFTYBANK-INDEX": "NIFTYBANK",
      "NSE:FINNIFTY-INDEX": "FINNIFTY", "NSE:MIDCPNIFTY-INDEX": "MIDCPNIFTY"}
_RANGE_CACHE: dict = {}


def _trailing_day_ranges(sym: str, n: int = 40) -> np.ndarray:
    """Trailing full-day range% distribution (the cross-day vol reference for HIGH_VOL)."""
    if sym in _RANGE_CACHE:
        return _RANGE_CACHE[sym]
    try:
        d = pd.read_parquet(f"data/historical/daily/NSE_{FY[sym]}_INDEX_daily.parquet")
        r = ((d["high"] - d["low"]) / d["open"] * 100).dropna().to_numpy()[-n:]
    except Exception:
        r = np.array([])
    _RANGE_CACHE[sym] = r
    return r


def _prev_close(sym: str, ref_date: str) -> float | None:
    """Prior captured trading day's last close (from the live mirrors — correct for live/
    recent, vs the stale daily parquet). Used for the opening-gap flag."""
    import glob, os
    from core.constants import LIVE_DIR
    try:
        days = sorted({os.path.basename(p)[:10]
                       for p in glob.glob(str(LIVE_DIR / "*_oi_snapshots.parquet"))
                       if os.path.getsize(p) > 800})
        priors = [d for d in days if d < ref_date]
        if not priors:
            return None
        cl = fc.build_series(sym, 5, priors[-1], None).get("close") or []
        return float(cl[-1]) if cl else None
    except Exception:
        return None


def read(sym: str, date=None, as_of: dt.datetime | None = None) -> dict:
    label = LABELS.get(sym, sym)
    ser = fc.build_series(sym, 5, date, as_of)
    if not ser.get("has_data"):
        return {"sym": sym, "label": label, "ok": False, "note": ser.get("note", "warming up")}
    df = pd.DataFrame({k: ser.get(k) for k in ("ts", "open", "high", "low", "close", "volume")})
    df = df.dropna(subset=["close"])
    if len(df) < 3:
        return {"sym": sym, "label": label, "ok": False, "note": "warming up"}
    spot = ser.get("spot", [None])[-1] or float(df["close"].iloc[-1])
    # cross-day vol percentile: today's range-so-far vs trailing full-day ranges
    trail = _trailing_day_ranges(sym)
    day_rng = (df["high"].max() - df["low"].min()) / df["open"].iloc[0] * 100
    dvp = float((trail <= day_rng).mean()) if len(trail) >= 10 else float("nan")
    ref = date or (as_of.date().isoformat() if as_of else dt.datetime.now(IST).date().isoformat())
    prev_c = _prev_close(sym, ref)

    f = FE.compute(df, day_vol_pctile=dvp, prev_close=prev_c)
    r = FE.classify(f)
    try:
        band = hf.forecast(sym, as_of=as_of, date=date) or {}
    except Exception:
        band = {}
    return {"sym": sym, "label": label, "ok": True, "spot": spot,
            "regime": r["regime"], "conf": r["confidence"], "action": r["action"],
            "structure": r.get("structure", ""), "carry": r["carry"],
            "flags": r.get("flags", []), "flag_notes": r.get("flag_notes", []),
            "phase": f.phase, "er": f.er, "iv_atm": ser.get("iv_atm", [None])[-1],
            "band_lo": band.get("lo"), "band_hi": band.get("hi"),
            "band_pct": band.get("exp_move_pct")}


def main():
    args = sys.argv[1:]
    date = args[0] if len(args) >= 1 else None
    as_of = None
    if date and len(args) >= 2:
        hh, mm = map(int, args[1].split(":"))
        as_of = dt.datetime.combine(dt.date.fromisoformat(date), dt.time(hh, mm), tzinfo=IST)
    now = as_of or dt.datetime.now(IST)
    live = date is None
    print("=" * 92)
    print(f"  INTRADAY READ — {'LIVE' if live else 'REPLAY'} {now:%Y-%m-%d %H:%M} IST"
          f"   (intraday-only; BTST-carry only fires after 15:00 on a strong close)")
    print("=" * 92)
    post3 = now.time() >= dt.time(15, 0)
    print(f"  {'index':11}{'spot':>10}  {'regime':10}{'conf':>5}  {'band (~70%)':>20}  action")
    carries = []
    for sym in INDEX_SYMBOLS:
        r = read(sym, date, as_of)
        if not r["ok"]:
            print(f"  {LABELS.get(sym, sym):11}{'':>10}  {'—':10}       {r['note']}")
            continue
        band = (f"{r['band_lo']:.0f}–{r['band_hi']:.0f}" if r["band_lo"] and r["band_hi"] else "—")
        flagstr = ("  ⚑ " + " ".join(r["flags"])) if r["flags"] else ""
        print(f"  {r['label']:11}{r['spot']:>10.1f}  {r['regime']:10}{r['conf']:>4}  {band:>20}  "
              f"{r['action'][:42]}{flagstr}")
        for note in r["flag_notes"]:
            print(f"  {'':11}{'':>10}       {'':10}   ↳ {note}")
        if r["carry"]:
            carries.append(r["label"])
    # the one intraday→BTST bridge
    print("\n  " + "-" * 88)
    if carries:
        print(f"  🌙 POST-3PM BTST-LONG CANDIDATE(S): {', '.join(carries)} — strong close forming.")
        print("     Long FUTURES now, exit next ~09:30. (validated close-strength overnight edge.)")
        print("     Size for a ~2% gap tail (NIFTY-only or hedge an OTM put).")
    elif post3:
        print("  after 15:00 — no strong close → NO BTST carry. Flat into the close.")
    else:
        print(f"  before 15:00 — intraday risk-map only. BTST-carry watch opens at 15:00.")
    print("  NOTE: no intraday directional arrow — it loses (58-75% wrong at cost). Trade the band.")


if __name__ == "__main__":
    main()
