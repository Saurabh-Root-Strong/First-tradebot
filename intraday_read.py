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
import regime_classifier as rc
from core.constants import INDEX_SYMBOLS, LABELS, IST

FY = {"NSE:NIFTY50-INDEX": "NIFTY50", "NSE:NIFTYBANK-INDEX": "NIFTYBANK",
      "NSE:FINNIFTY-INDEX": "FINNIFTY", "NSE:MIDCPNIFTY-INDEX": "MIDCPNIFTY"}
_RANGE_CACHE: dict = {}


def _trailing_day_ranges(sym: str, ref_date: str, n: int = 40) -> np.ndarray:
    """Trailing full-day range% distribution (the cross-day vol reference for HIGH_VOL).
    CAUSAL: uses only days STRICTLY BEFORE ref_date. Without the ref filter a replay of a
    past day would pull in later days' ranges = lookahead (live is unaffected — today is the
    last row, but replay/backtest via this path leaked)."""
    key = (sym, ref_date)
    if key in _RANGE_CACHE:
        return _RANGE_CACHE[key]
    try:
        d = pd.read_parquet(f"data/historical/daily/NSE_{FY[sym]}_INDEX_daily.parquet")
        d = d[pd.to_datetime(d["ts"]).dt.date.astype(str) < ref_date]      # past days only
        r = ((d["high"] - d["low"]) / d["open"] * 100).dropna().to_numpy()[-n:]
    except Exception:
        r = np.array([])
    _RANGE_CACHE[key] = r
    return r


_PREV_CACHE: dict = {}


def _prev_close(sym: str, ref_date: str) -> float | None:
    """Prior captured trading day's last close (from the live mirrors — correct for live/
    recent, vs the stale daily parquet). Used for the opening-gap flag. Cached: the prior
    day's close is static, so it never needs recomputing within a session."""
    import glob, os
    from core.constants import LIVE_DIR
    key = (sym, ref_date)
    if key in _PREV_CACHE:
        return _PREV_CACHE[key]
    try:
        days = sorted({os.path.basename(p)[:10]
                       for p in glob.glob(str(LIVE_DIR / "*_oi_snapshots.parquet"))
                       if os.path.getsize(p) > 800})
        priors = [d for d in days if d < ref_date]
        if not priors:
            return None
        cl = fc.build_series(sym, 5, priors[-1], None).get("close") or []
        pc = float(cl[-1]) if cl else None
        _PREV_CACHE[key] = pc
        return pc
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
    ref = date or (as_of.date().isoformat() if as_of else dt.datetime.now(IST).date().isoformat())
    # cross-day vol percentile: today's range-so-far vs trailing full-day ranges (PAST only)
    trail = _trailing_day_ranges(sym, ref)
    day_rng = (df["high"].max() - df["low"].min()) / df["open"].iloc[0] * 100
    dvp = float((trail <= day_rng).mean()) if len(trail) >= 10 else float("nan")
    prev_c = _prev_close(sym, ref)

    f = FE.compute(df, day_vol_pctile=dvp, prev_close=prev_c)
    r = FE.classify(f)
    # trend regime (Kaufman ER, causal, forming bar dropped) — used to widen the band
    # in a strong trend where the endpoint persists past the vol estimate (measured).
    mood = rc.classify_from_bars(ser, n=10)
    rw = rc.band_width_mult(mood)
    hz, H_eff = hf.horizon_band_factor(60, as_of)   # clip 60m → rest-of-session near close
    tw = hf.tod_width_mult(as_of, H_eff)   # close-hour widen — gated OFF on short clipped windows
    try:
        band = hf.forecast(sym, as_of=as_of, date=date) or {}
        # L4 per-index × regime width × time-of-day width × horizon clip, one rescale.
        bm = hf.band_multiplier(sym, H_eff)
        wf = bm * rw * tw * hz
        if wf != 1.0 and band.get("lo") is not None and band.get("hi") is not None:
            mid = (band["lo"] + band["hi"]) / 2.0
            half = (band["hi"] - band["lo"]) / 2.0 * wf
            band["lo"], band["hi"] = round(mid - half, 1), round(mid + half, 1)
            # keep the displayed ±% in lock-step with the (now rescaled) band
            if mid:
                band["exp_move_pct"] = round(half / mid * 100.0, 3)
    except Exception:
        band = {}
    # measured per-index coverage of THIS band at its (possibly clipped) horizon
    try:
        bcov = hf.band_coverage(sym, H_eff)
    except Exception:
        bcov = {"cover": None, "n": 0, "conf": "none"}
    return {"sym": sym, "label": label, "ok": True, "spot": spot,
            "regime": r["regime"], "conf": r["confidence"], "action": r["action"],
            "structure": r.get("structure", ""), "carry": r["carry"],
            "posture": r.get("posture", ""), "size": r.get("size"),
            "opt_buy": r.get("opt_buy", ""), "state": r.get("state", ""),
            "flags": r.get("flags", []), "flag_notes": r.get("flag_notes", []),
            "phase": f.phase, "er": f.er, "iv_atm": ser.get("iv_atm", [None])[-1],
            "band_lo": band.get("lo"), "band_hi": band.get("hi"),
            "band_pct": band.get("exp_move_pct"),
            "band_horizon": H_eff,
            "trend_regime": mood.short, "band_regime_mult": round(rw, 2),
            "band_cover": bcov.get("cover"), "band_n": bcov.get("n", 0),
            "band_conf": bcov.get("conf", "none")}


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
    print(f"  {'index':11}{'spot':>10}  {'regime':10}{'conf':>5}  {'next-60m band':>20}  action")
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
        # measured 60m coverage tag — trust the band per-index, not a flat ~70%
        if r.get("band_cover") is not None:
            _cm = {"ok": "✓", "soft": "~", "low": "⚠ low", "thin": "· thin"}.get(r["band_conf"], "")
            print(f"  {'':11}{'':>10}       {'':10}   ↳ 60m cover "
                  f"{r['band_cover']*100:.0f}% n{r['band_n']} {_cm}")
        # honest defensive POSTURE line — SIZE + trade/no-trade, never a direction
        szx = f"{r['size']:.1f}x" if r.get("size") is not None else "—"
        print(f"  {'':11}{'':>10}  → {r['posture']:11} size {szx:5} · {r['state']:14}"
              f" · option-buy: {r['opt_buy']}")
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
