"""
backtest_band_horizon.py — RANGE-band coverage on the last N captured live days,
split by FORECAST HORIZON (15m vs 60m). The band is the desk's one trusted product;
this measures whether it actually covers at the claimed ~68% at EACH horizon.

The band is built in hour_forecast at 60m (calibrated mult hf._RANGE_M — 0.73 since
the 2026-07-02 recalibration; read the live value, don't trust this line) and then
(H/60)^_BAND_HURST-scaled to any horizon by intraday_scout._forward. That scaling
ASSUMES coverage is horizon-invariant. We test that assumption on real captured
ticks. NOTE: this grades the BASE band — the deployed band additionally carries the
L4 multiplier + regime/TOD widens (see the 2026-07-03 audit, finding M1).

Two honest coverage metrics per horizon:
  • endpoint   — price AT t+H inside [lo,hi]  (matches _verify "close-in-band" / the
                 claim; this is what the scout grades).
  • envelope   — price stayed inside [lo,hi] the WHOLE t→t+H window (stricter; the
                 band read as a no-breach range — a stop set at the wall survives).

Causal: forecast at t reads only ticks<=t; the answer key (t→t+H ticks) never feeds it.

    .venv\\Scripts\\python.exe backtest_band_horizon.py [--days 3] [--horizons 15,60]
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import math
import os

import numpy as np
import pandas as pd

from core.constants import IST, INDEX_SYMBOLS, LABELS, LIVE_DIR, DATA_DIR
from core.mirror_io import read_mirror
import footprint_chart as fcx           # 5-min series for the mood read (deployed band)
import hour_forecast as hf
import intraday_scout as scout          # deployed band exponent (_BAND_HURST) — keep in sync
import regime_classifier as rc          # regime width multiplier (deployed band)

SESSION_OPEN = dt.time(9, 15)
FIRST_PRED = dt.time(9, 45)          # need ~30m of ticks for sigma1
LAST_PRED = dt.time(15, 0)
STEP_MIN = 15                        # sample a prediction every 15 min


import re
_DAY_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_ticks\.parquet$")


def captured_days(n: int) -> list[str]:
    """Last n ISO days that have a non-trivial ticks mirror (strict YYYY-MM-DD only —
    ignore tmp_/scratch parquets that would otherwise glob in as a bogus day)."""
    days = []
    for p in glob.glob(str(LIVE_DIR / "*_ticks.parquet")):
        m = _DAY_RE.match(os.path.basename(p))
        if m and os.path.getsize(p) > 5000:
            days.append(m.group(1))
    return sorted(set(days))[-n:]


def pred_times(day: str) -> list[dt.datetime]:
    d = dt.date.fromisoformat(day)
    t = dt.datetime.combine(d, FIRST_PRED, tzinfo=IST)
    end = dt.datetime.combine(d, LAST_PRED, tzinfo=IST)
    out = []
    while t <= end:
        out.append(t)
        t += dt.timedelta(minutes=STEP_MIN)
    return out


def deployed_width_factor(sym: str, H_eff: int, as_of: dt.datetime, mood) -> float:
    """The three multipliers intraday_scout applies ON TOP of the base band, in the same
    order: L4 learned x regime-conditional x time-of-day. Kept here (not imported) so this
    file stays the independent measurement of what scan_index actually draws."""
    return (hf.band_multiplier(sym, H_eff)
            * rc.band_width_mult(mood)
            * hf.tod_width_mult(as_of, H_eff))


def grade(ticks: pd.DataFrame, t: dt.datetime, H: int, spot: float,
          band_pct_60: float, sym: str | None = None, mood=None) -> dict | None:
    """Grade one (t, horizon) band against actual forward ticks. None if unresolved.

    Two bands are graded from the same forward window:
      base     — hour_forecast's raw band, horizon-scaled. This is what the L4 loop
                 (calibration_engine) corrects against, so it must keep being reported.
      deployed — base x L4 x regime x tod = the band actually DRAWN on the board. The
                 UI used to print the BASE coverage next to the DEPLOYED band, which
                 overstated it by ~6pp (measured 79.4% vs 73.4% at 60m over 30 days).
                 Needs sym+mood; omitted -> deployed is None and only base is graded.
    """
    band_pct = band_pct_60 * (H / 60.0) ** scout._BAND_HURST   # deployed horizon scaling
    half = spot * band_pct / 100.0
    lo, hi = spot - half, spot + half
    t_end = t + dt.timedelta(minutes=H)
    fwd = ticks[(ticks["ts"] > pd.Timestamp(t)) & (ticks["ts"] <= pd.Timestamp(t_end))]
    if fwd.empty:
        return None
    # require ticks to actually reach ~t+H, else the horizon isn't resolved
    if fwd["ts"].max() < pd.Timestamp(t_end) - pd.Timedelta(minutes=2):
        return None
    end_px = float(fwd.iloc[-1]["ltp"])
    hi_px, lo_px = float(fwd["ltp"].max()), float(fwd["ltp"].min())
    out = {
        "endpoint": bool(lo <= end_px <= hi),
        "envelope": bool(hi_px <= hi and lo_px >= lo),
        "band_pct": band_pct,
        "width_pts": 2 * half,
        "deployed": None,
        "deployed_env": None,
    }
    if sym is not None and mood is not None:
        # session_horizon clips near the close exactly as scan_index does, so the
        # deployed band is reproduced for the window the board really showed.
        H_eff = hf.session_horizon(H, t)
        d_half = (spot * (band_pct_60 * (H_eff / 60.0) ** scout._BAND_HURST) / 100.0
                  * deployed_width_factor(sym, H_eff, t, mood))
        d_lo, d_hi = spot - d_half, spot + d_half
        out["deployed"] = bool(d_lo <= end_px <= d_hi)
        out["deployed_env"] = bool(hi_px <= d_hi and lo_px >= d_lo)
    return out


def run(n_days: int, horizons: list[int]) -> None:
    days = captured_days(n_days)
    if not days:
        print("no captured ticks mirrors found"); return
    print("=" * 90)
    print(f"  RANGE-BAND COVERAGE by HORIZON — last {len(days)} captured day(s): {', '.join(days)}")
    print(f"  band = hour_forecast 60m (mult {hf._RANGE_M}) sqrt(H/60)-scaled; "
          f"grid every {STEP_MIN}m {FIRST_PRED:%H:%M}-{LAST_PRED:%H:%M}")
    print("=" * 90)

    # accumulators: per horizon (all indices) and per (horizon, index)
    agg = {H: {"end": [], "env": [], "w": [], "dep": []} for H in horizons}
    per_idx = {H: {s: {"end": [], "env": [], "dep": []} for s in INDEX_SYMBOLS}
               for H in horizons}

    for day in days:
        # preload each index's full-day ticks once (answer key)
        tk = {}
        for s in INDEX_SYMBOLS:
            df = read_mirror("ticks", day, None, s)
            if df is not None and len(df):
                tk[s] = df[["ts", "ltp"]].copy()
        for s in INDEX_SYMBOLS:
            if s not in tk:
                continue
            for t in pred_times(day):
                try:
                    fc = hf.forecast(s, as_of=t, date=day)
                except Exception:
                    continue
                if not fc.get("has_data") or fc.get("exp_move_pct") is None:
                    continue
                spot, bp60 = fc["spot"], fc["exp_move_pct"]
                # mood drives the regime width multiplier; read off a FIXED 5-min series
                # (same pin as intraday_scout — the regime must not follow the chart tf).
                try:
                    mood = rc.classify_from_bars(fcx.build_series(s, 5, day, t),
                                                 n=scout._MOOD_WIN)
                except Exception:
                    mood = None
                for H in horizons:
                    g = grade(tk[s], t, H, spot, bp60, sym=s, mood=mood)
                    if g is None:
                        continue
                    agg[H]["end"].append(g["endpoint"])
                    agg[H]["env"].append(g["envelope"])
                    agg[H]["w"].append(g["band_pct"])
                    per_idx[H][s]["end"].append(g["endpoint"])
                    per_idx[H][s]["env"].append(g["envelope"])
                    if g["deployed"] is not None:
                        agg[H]["dep"].append(g["deployed"])
                        per_idx[H][s]["dep"].append(g["deployed"])

    def pct(x):
        return f"{100 * np.mean(x):5.1f}%" if x else "   n/a"

    print(f"\n  {'horizon':>8}  {'N':>5}  {'endpoint':>9}  {'DEPLOYED':>9}  {'envelope':>9}"
          f"  {'avg band ±%':>12}")
    print("  " + "-" * 68)
    for H in horizons:
        a = agg[H]
        wid = f"±{np.mean(a['w']):.3f}%" if a["w"] else "n/a"
        print(f"  {H:>6}m  {len(a['end']):>5}  {pct(a['end']):>9}  {pct(a['dep']):>9}"
              f"  {pct(a['env']):>9}  {wid:>12}")

    print(f"\n  per-index DEPLOYED endpoint coverage (the band actually drawn)")
    print(f"  {'index':12}" + "".join(f"{str(H)+'m':>16}" for H in horizons))
    for s in INDEX_SYMBOLS:
        cells = ""
        for H in horizons:
            e = per_idx[H][s]["dep"] or per_idx[H][s]["end"]
            cells += f"{pct(e)+' (n='+str(len(e))+')':>16}"
        print(f"  {LABELS.get(s, s):12}{cells}")

    print("\n  READ: target ~68% (band calibrated to 68% at 60m). endpoint = the claim")
    print("  (price at t+H in band). envelope = stricter no-breach (stop at wall survives).")
    print("  If 15m >> 68% the near band is too WIDE (over-covers); if << 68% too tight.")

    # ── persist the coverage ledger — hour_forecast reads it to surface an HONEST,
    # self-updating per-(index,horizon) coverage next to the band; a FUTURE auto re-level
    # of _RANGE_M should trigger ONLY once N per cell clears a safe threshold (not 3 days).
    import json
    led = {"updated": dt.datetime.now(IST).isoformat(), "days": days,
           "hurst": scout._BAND_HURST, "overall": {}, "by_index": {}}
    for H in horizons:
        a = agg[H]
        led["overall"][str(H)] = {"n": len(a["end"]),
                                  "endpoint": round(float(np.mean(a["end"])), 4) if a["end"] else None,
                                  "deployed": round(float(np.mean(a["dep"])), 4) if a["dep"] else None,
                                  "envelope": round(float(np.mean(a["env"])), 4) if a["env"] else None,
                                  "band_pct": round(float(np.mean(a["w"])), 4) if a["w"] else None}
    for s in INDEX_SYMBOLS:
        led["by_index"][s] = {}
        for H in horizons:
            e, d = per_idx[H][s]["end"], per_idx[H][s]["dep"]
            # `endpoint` = BASE band — calibration_engine corrects against it, so the key
            # keeps its meaning. `deployed` = the drawn band, what the UI must show.
            led["by_index"][s][str(H)] = {
                "n": len(e),
                "endpoint": round(float(np.mean(e)), 4) if e else None,
                "deployed": round(float(np.mean(d)), 4) if d else None}
    outp = DATA_DIR / "calibration"
    outp.mkdir(parents=True, exist_ok=True)
    (outp / "band_coverage.json").write_text(json.dumps(led, indent=2))
    print(f"\n  ledger → {outp / 'band_coverage.json'}")


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--horizons", default="15,60")
    a = ap.parse_args()
    run(a.days, [int(x) for x in a.horizons.split(",")])
