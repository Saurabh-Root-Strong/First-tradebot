"""
backtest_4day_trades.py — REPLAY the last 4 captured days (Mon–Thu) and answer two
questions the user actually asks looking at the board:

  1) REAL TRADES (1-hour scout): every time the scout emits TRADE CE/PE at tf=60, treat
     it as a live option trade — buy ATM at trigger, SL −35% / target +65% on the PREMIUM
     (the deployed _SLT[60]). Walk the option PATH tick-by-tick to the close and record
     which was TOUCHED FIRST: SL, TARGET, or neither (TIME-exit at horizon/close). Counts
     DISTINCT trades only (a new trade = transition NO-TRADE/opposite → TRADE dir; a held
     signal is ONE trade, not re-counted every bar — the "you're confirming the open trade"
     point).

  2) BAND ACCURACY (the risk-map product): endpoint-in-band rate at 15 / 30 / 60m, using
     the exact deployed band geometry (scan_index verify.band_hit). Direction-free.

Causal by construction — everything reads build_series/_read_mirror with ts<=t. No lookahead.

    .venv\\Scripts\\python.exe backtest_4day_trades.py
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

import intraday_scout as sc
from core.constants import INDEX_SYMBOLS, LABELS, IST
from core.mirror_io import read_mirror as _read_mirror

DAYS = ["2026-06-29", "2026-06-30", "2026-07-01", "2026-07-02"]
OPEN_VALID = dt.time(9, 45)      # scout calibration only valid from here
LAST_ENTRY = dt.time(15, 0)      # no new trade after this (no time to resolve on 60m)
CLOSE = dt.time(15, 30)
GRID = 5                         # minute grid for signal walk
SL_PCT, T_PCT = sc._SLT[60]      # 0.35 / 0.65 — the deployed 1-hour premium stop/target


def _grid(day: str, start: dt.time, end: dt.time):
    d = dt.date.fromisoformat(day)
    t = dt.datetime.combine(d, start, tzinfo=IST)
    e = dt.datetime.combine(d, end, tzinfo=IST)
    while t <= e:
        yield t
        t += dt.timedelta(minutes=GRID)


def _prem_path(sym, day, strike, side, t0, t_end) -> pd.DataFrame | None:
    """The option's premium PATH between entry and horizon (for first-touch SL/target)."""
    ch = _read_mirror("chain_snapshots", day, t_end, sym)
    if ch is None or not len(ch) or "ltp" not in ch.columns:
        return None
    import footprint_chart as fc
    ch, ok = fc._filter_expiry(ch, "weekly")
    if not ok or ch is None or not len(ch):
        return None
    sub = ch[(ch["side"] == side) & (ch["strike"] == strike)
             & (ch["ts"] > pd.Timestamp(t0)) & (ch["ts"] <= pd.Timestamp(t_end))]
    sub = sub[pd.notna(sub["ltp"]) & (sub["ltp"] > 0)].sort_values("ts")
    return sub if len(sub) else None


def run_trades():
    print("=" * 84)
    print("  1) REAL TRADES — 1-hour scout (tf=60), ATM option, SL −35% / target +65% on premium")
    print("=" * 84)
    rows = []
    for day in DAYS:
        for sym in INDEX_SYMBOLS:
            prev = None                 # last verdict-direction, to detect a NEW entry
            for t in _grid(day, OPEN_VALID, LAST_ENTRY):
                v = sc.scan_index(sym, 60, date=day, as_of=t,
                                  with_lifecycle=False, verdict_only=True)
                d = v.get("direction") or ""
                new_entry = d in ("CE", "PE") and d != prev
                prev = d
                if not new_entry:
                    continue
                # entry: ATM at t, premium at t
                spot0 = sc._spot_at(sym, day, t)
                atm = sc._atm(spot0, sym) if spot0 else None
                entry = sc._opt_premium(sym, day, t, atm, d) if atm else None
                if not entry:
                    continue
                t_end = dt.datetime.combine(dt.date.fromisoformat(day), CLOSE, tzinfo=IST)
                t_end = min(t_end, t + dt.timedelta(minutes=60))
                path = _prem_path(sym, day, atm, d, t, t_end)
                sl = entry * (1 - SL_PCT); tgt = entry * (1 + T_PCT)
                outcome, exitp = "TIME", entry
                if path is not None:
                    hit_sl = path[path["ltp"] <= sl]
                    hit_tg = path[path["ltp"] >= tgt]
                    ts_sl = hit_sl["ts"].min() if len(hit_sl) else pd.NaT
                    ts_tg = hit_tg["ts"].min() if len(hit_tg) else pd.NaT
                    if pd.notna(ts_sl) and (pd.isna(ts_tg) or ts_sl <= ts_tg):
                        outcome, exitp = "SL", sl
                    elif pd.notna(ts_tg):
                        outcome, exitp = "TARGET", tgt
                    else:
                        exitp = float(path.iloc[-1]["ltp"])
                pnl = (exitp / entry - 1.0) * 100.0
                rows.append({"day": day, "sym": LABELS.get(sym, sym), "t": t.strftime("%H:%M"),
                             "dir": d, "strike": atm, "entry": round(entry, 1),
                             "outcome": outcome, "exit": round(exitp, 1),
                             "pnl": round(pnl, 1), "net": round(pnl - sc._OPT_RT_COST, 1)})
    if not rows:
        print("  (no TRADE signals fired on the 1-hour timeframe across the 4 days)")
        return rows
    df = pd.DataFrame(rows)
    for _, r in df.iterrows():
        print(f"  {r['day']}  {r['sym']:11} {r['t']}  {r['dir']}  {r['strike']}  "
              f"entry ₹{r['entry']:>6}  → {r['outcome']:6} ₹{r['exit']:>6}  "
              f"gross {r['pnl']:+6.1f}%  net {r['net']:+6.1f}%")
    n = len(df)
    sl = int((df["outcome"] == "SL").sum())
    tg = int((df["outcome"] == "TARGET").sum())
    tm = int((df["outcome"] == "TIME").sum())
    print("  " + "-" * 80)
    _dow = {"2026-06-29": "Mon", "2026-06-30": "Tue", "2026-07-01": "Wed", "2026-07-02": "Thu"}
    print("  PER-DAY:")
    for day in DAYS:
        g = df[df["day"] == day]
        if not len(g):
            print(f"    {_dow[day]} {day}:  no 1h trades")
            continue
        print(f"    {_dow[day]} {day}:  {len(g):>2} trades  ·  TARGET "
              f"{int((g['outcome']=='TARGET').sum())}  SL {int((g['outcome']=='SL').sum())}  "
              f"TIME {int((g['outcome']=='TIME').sum())}  ·  net {g['net'].mean():+.1f}%  "
              f"win {100*(g['net']>0).mean():.0f}%")
    print("  " + "-" * 80)
    print(f"  TRADES {n}:  TARGET {tg} ({100*tg/n:.0f}%)  ·  SL {sl} ({100*sl/n:.0f}%)  ·  "
          f"TIME-exit {tm} ({100*tm/n:.0f}%)")
    print(f"  mean GROSS {df['pnl'].mean():+.1f}%   mean NET (−3% RT) {df['net'].mean():+.1f}%   "
          f"win(net>0) {100*(df['net']>0).mean():.0f}%")
    return rows


def run_band():
    print("\n" + "=" * 84)
    print("  2) BAND ACCURACY — endpoint inside the deployed band, by horizon (direction-free)")
    print("=" * 84)
    res = {15: [], 30: [], 60: []}
    by_day = {d: {15: [], 30: [], 60: []} for d in DAYS}
    by_idx = {LABELS.get(s, s): {15: [], 30: [], 60: []} for s in INDEX_SYMBOLS}
    for day in DAYS:
        for sym in INDEX_SYMBOLS:
            for t in _grid(day, OPEN_VALID, dt.time(15, 25)):
                for H in (15, 30, 60):
                    r = sc.scan_index(sym, 5, date=day, as_of=t, horizon_min=H,
                                      with_lifecycle=False)
                    v = r.get("verify")
                    if v and v.get("band_hit") is not None and r.get("pred_lo") is not None:
                        hit = bool(v["band_hit"])
                        res[H].append(hit)
                        by_day[day][H].append(hit)
                        by_idx[LABELS.get(sym, sym)][H].append(hit)

    def _c(x):
        return f"{100*np.mean(x):.1f}%" if x else "  n/a"

    print(f"\n  {'horizon':>10}{'N':>8}{'in-band':>10}   (target ~68%)")
    print("  " + "-" * 40)
    for H in (15, 30, 60):
        print(f"  {H:>8}m{len(res[H]):>8}{_c(res[H]):>10}")

    _dow = {"2026-06-29": "Mon", "2026-06-30": "Tue", "2026-07-01": "Wed", "2026-07-02": "Thu"}
    print(f"\n  PER-DAY (in-band %):   {'15m':>8}{'30m':>8}{'60m':>8}")
    for day in DAYS:
        print(f"    {_dow[day]} {day}: " + "".join(f"{_c(by_day[day][H]):>8}" for H in (15, 30, 60)))
    print(f"\n  PER-INDEX (in-band %): {'15m':>8}{'30m':>8}{'60m':>8}")
    for nm in by_idx:
        print(f"    {nm:13}     " + "".join(f"{_c(by_idx[nm][H]):>8}" for H in (15, 30, 60)))
    return res


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    run_trades()
    if "--trades-only" not in sys.argv:
        run_band()
