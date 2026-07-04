"""
backtest_price_action_ticks_oos.py — OOS check of the price-action states on REAL live
tick mirrors (the 5 captured sessions), full LIVE band geometry.

backtest_price_action_60m.py measured 493 days of 5-min history and said: direction dead
at cost, band already ~68% where it matters, impulse-widen candidate fails the halves
test. This script re-runs the SAME state machine on the live tick capture (Mon–Fri last
week) as an out-of-sample fidelity check:
  • bars built from the actual tick stream the live scout sees (not vendor 5-min bars),
  • band = FULL live pipeline: _RANGE_M·sig1(1-min ticks)·√60 × L4 band_multiplier ×
    regime width × time-of-day widen — exactly what the dashboard displays,
  • n is TINY (~5 days × 4 idx × 18 times ≈ 360) → this VALIDATES or CONTRADICTS the
    493-day result; it cannot discover anything on its own.

    .venv\\Scripts\\python.exe backtest_price_action_ticks_oos.py
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

import hour_forecast as hf
import intraday_scout as scout
import regime_classifier as rc
from core.constants import IST
from core.mirror_io import read_mirror
from backtest_price_action_60m import _state, W, BUCKET, REG_MULT, STATES, SIGN, COST_BPS

DAYS = ["2026-06-29", "2026-06-30", "2026-07-01", "2026-07-02", "2026-07-03"]
SYM = {
    "NIFTY 50":     "NSE:NIFTY50-INDEX",
    "BANK NIFTY":   "NSE:NIFTYBANK-INDEX",
    "FIN NIFTY":    "NSE:FINNIFTY-INDEX",
    "MIDCAP NIFTY": "NSE:MIDCPNIFTY-INDEX",
}
FIRST_PRED = dt.time(10, 15)
LAST_PRED = dt.time(14, 30)
STEP_MIN = 15
H = 60


def _bars5(t: pd.DataFrame) -> pd.DataFrame:
    """Session 5-min OHLC from the tick stream (ltp only — index has no traded vol)."""
    s = t.set_index("ts")["ltp"].astype(float)
    o = s.resample("5min").first(); hi = s.resample("5min").max()
    lo = s.resample("5min").min();  c = s.resample("5min").last()
    b = pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c}).dropna()
    return b.reset_index().rename(columns={"index": "ts"})


def run() -> None:
    print("=" * 96)
    print("  PRICE-ACTION OOS — live tick mirrors, FULL live band geometry "
          "(base × L4 × regime × tod)")
    print(f"  days: {DAYS[0]} … {DAYS[-1]}   states from backtest_price_action_60m")
    print("=" * 96)

    cell = {s: {"sret": [], "end": [], "n_reg": {}} for s in STATES}
    rows = []
    for day in DAYS:
        df = read_mirror("ticks", day)
        if df is None:
            print(f"  {day}: no tick mirror — skipped"); continue
        d0 = dt.datetime.fromisoformat(day).date()
        so = pd.Timestamp(dt.datetime.combine(d0, dt.time(9, 15)), tz=IST)
        sc = pd.Timestamp(dt.datetime.combine(d0, dt.time(15, 30)), tz=IST)
        df = df[(df["ts"] >= so) & (df["ts"] <= sc)]
        for name, fy in SYM.items():
            t = df[df["symbol"] == fy].sort_values("ts")
            if len(t) < 600:
                continue
            bars = _bars5(t)
            if len(bars) < W + 8:
                continue
            moods = rc.classify_series(bars, n=10)["mood"].to_numpy()
            bar_end = bars["ts"] + pd.Timedelta(minutes=5)   # bar CLOSE time
            o = bars["open"].to_numpy(float); hh = bars["high"].to_numpy(float)
            ll = bars["low"].to_numpy(float); cc = bars["close"].to_numpy(float)
            ticks_s = t.set_index("ts")["ltp"].astype(float)
            one = ticks_s.resample("1min").last().dropna()

            pt = dt.datetime.combine(d0, FIRST_PRED)
            endt = dt.datetime.combine(d0, LAST_PRED)
            while pt <= endt:
                tp = pd.Timestamp(pt, tz=IST)
                # last CLOSED 5-min bar at tp (bar ts = window start; closed iff end<=tp)
                i = int((bar_end <= tp).sum()) - 1
                if i < W + 5:
                    pt += dt.timedelta(minutes=STEP_MIN); continue
                st, _ = _state(o, hh, ll, cc, i)
                if st is None:
                    pt += dt.timedelta(minutes=STEP_MIN); continue
                cur = ticks_s[ticks_s.index <= tp]
                if len(cur) < 100:
                    pt += dt.timedelta(minutes=STEP_MIN); continue
                spot = float(cur.iloc[-1])
                # sig1 exactly as hour_forecast._sigma1_pct: 1-min lasts, pct_change std
                r1 = one[one.index <= tp].pct_change().dropna() * 100.0
                if len(r1) < 8 or r1.std() == 0:
                    pt += dt.timedelta(minutes=STEP_MIN); continue
                sig1 = float(r1.std())
                bp = hf._RANGE_M * max(sig1, hf._SIG_FLOOR) * np.sqrt(60.0)
                reg = BUCKET.get(moods[i], "CHOP")
                wf = (hf.band_multiplier(name, H) * REG_MULT[reg]
                      * hf.tod_width_mult(tp.to_pydatetime(), H))
                bp_live = bp * wf
                fwd = ticks_s[(ticks_s.index > tp)
                              & (ticks_s.index <= tp + pd.Timedelta(minutes=H))]
                if len(fwd) < 50:
                    pt += dt.timedelta(minutes=STEP_MIN); continue
                end_px = float(fwd.iloc[-1])
                ret_bps = (end_px / spot - 1.0) * 1e4
                hit = abs(end_px - spot) <= spot * bp_live / 100.0
                cell[st]["end"].append(bool(hit))
                if SIGN[st]:
                    cell[st]["sret"].append(SIGN[st] * ret_bps)
                cell[st]["n_reg"][reg] = cell[st]["n_reg"].get(reg, 0) + 1
                rows.append((day, name, pt.strftime("%H:%M"), st, reg,
                             round(ret_bps, 1), "HIT" if hit else "MISS",
                             round(bp_live, 3)))
                pt += dt.timedelta(minutes=STEP_MIN)

    n_all = len(rows)
    hits = sum(1 for r in rows if r[6] == "HIT")
    print(f"\n  sample: {n_all} predictions over {len(DAYS)} sessions")
    print(f"  OVERALL live-band endpoint coverage: {100*hits/max(n_all,1):.1f}%  "
          f"(target 68% — the deployed product's OOS grade)")

    print(f"\n  per-state (OOS, thin n — direction of effect only)")
    print(f"  {'state':11}{'n':>6}{'cover':>8}{'dir mean bps':>14}{'dir win%':>10}")
    print("  " + "-" * 52)
    for s in STATES:
        e = cell[s]["end"]; r = cell[s]["sret"]
        cov = f"{100*np.mean(e):6.1f}%" if e else "   n/a"
        if SIGN[s] and len(r) >= 5:
            dm, dw = f"{np.mean(r):>+10.1f}", f"{100*np.mean(np.array(r)>0):>8.1f}%"
        else:
            dm, dw = f"{'—':>10}", f"{'—':>8}"
        print(f"  {s:11}{len(e):>6}{cov:>8}{dm:>14}{dw:>10}")

    print(f"\n  per-day coverage (stability across the week)")
    for day in DAYS:
        dr = [r for r in rows if r[0] == day]
        if not dr:
            continue
        h = sum(1 for r in dr if r[6] == "HIT")
        by_st = {}
        for r in dr:
            by_st[r[3]] = by_st.get(r[3], 0) + 1
        top = ", ".join(f"{k}:{v}" for k, v in sorted(by_st.items(), key=lambda x: -x[1]))
        print(f"    {day}  n={len(dr):>3}  cover {100*h/len(dr):5.1f}%   states: {top}")

    print("\n  READ: coverage ≈68% + states matching the 493-day table = the parquet result"
          "\n  STANDS on live ticks (include/exclude verdict unchanged). A state flipping"
          "\n  sign here on n<40 is NOISE, not a discovery.")


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    run()
