"""
backtest_band_horizon_hist.py — RANGE-band coverage over the FULL 2-yr 5-min history,
split by forecast horizon (30 / 60 / 120 / 240 min). Companion to backtest_band_horizon.py
(which is faithful to the live tick+chain pipeline but limited to the ~dozen captured days).

The band is symmetric around spot and built purely from realised vol — DIRECTION never
enters it — so coverage depends only on the vol estimate + horizon scaling. That lets us
reproduce the EXACT deployed band geometry on 2 years of 5-min index bars and measure its
real coverage on ~500 sessions instead of a dozen:

    band_pct_60 = _RANGE_M * max(sig1, _SIG_FLOOR) * sqrt(60)      # hour_forecast
    band_pct_H  = band_pct_60 * (H / 60) ** _BAND_HURST            # scout horizon scaling
    half        = spot * band_pct_H / 100

sig1 (1-min realised vol %) is estimated causally from the CURRENT day's 5-min bars up to t:
    sig1 ≈ std(5-min returns %) / sqrt(5)      (variance scales with time)
same intraday-reset, session-cumulative estimator the live model uses on ticks.

Causal: the forecast at t uses only bars with ts ≤ t; the forward window (t, t+H] is the
answer key and never feeds the vol estimate.

Two metrics per horizon (same as the live backtest):
  • endpoint — close AT t+H inside [lo,hi]   (the claim; what the scout grades)
  • envelope — every bar in (t,t+H] stayed inside [lo,hi]  (stricter no-breach / stop-survives)

Also reports the multiplier that WOULD hit the 68% target at each horizon
(recommended _RANGE_M recalibration = 68th pct of |move| in base-sig units).

    .venv\\Scripts\\python.exe backtest_band_horizon_hist.py [--horizons 30,60,120,240]
"""
from __future__ import annotations

import argparse
import datetime as dt

import numpy as np
import pandas as pd

# deployed constants — imported so this stays in lock-step with the live band
import hour_forecast as hf            # _RANGE_M, _SIG_FLOOR
import intraday_scout as scout        # _BAND_HURST

IDX = {
    "NIFTY 50":     "data/historical/5min/NSE_NIFTY50_INDEX_5min.parquet",
    "BANK NIFTY":   "data/historical/5min/NSE_NIFTYBANK_INDEX_5min.parquet",
    "FIN NIFTY":    "data/historical/5min/NSE_FINNIFTY_INDEX_5min.parquet",
    "MIDCAP NIFTY": "data/historical/5min/NSE_MIDCPNIFTY_INDEX_5min.parquet",
}

SESSION_OPEN = dt.time(9, 15)
SESSION_CLOSE = dt.time(15, 30)
FIRST_PRED = dt.time(9, 45)      # need ~30m of bars for sigma (matches live backtest)
LAST_PRED = dt.time(15, 0)
STEP_MIN = 15
MIN_BARS = 6                     # ≥6 five-min bars (30m) before a prediction


def _pred_times(day: dt.date) -> list[dt.datetime]:
    t = dt.datetime.combine(day, FIRST_PRED)
    end = dt.datetime.combine(day, LAST_PRED)
    out = []
    while t <= end:
        out.append(t)
        t += dt.timedelta(minutes=STEP_MIN)
    return out


def _band_pct_60(sig1: float) -> float:
    sig60 = max(sig1, hf._SIG_FLOOR) * np.sqrt(60.0)
    return hf._RANGE_M * sig60


def run(horizons: list[int]) -> None:
    print("=" * 96)
    print("  RANGE-BAND COVERAGE by HORIZON — FULL 5-min history (deployed band geometry)")
    print(f"  band = _RANGE_M {hf._RANGE_M} · sig1 · sqrt(60) · (H/60)^{scout._BAND_HURST}; "
          f"grid every {STEP_MIN}m {FIRST_PRED:%H:%M}-{LAST_PRED:%H:%M}")
    print("=" * 96)

    # per (index, horizon): endpoint hits, envelope hits, band width%, and |move|/base-sig ratio
    per = {name: {H: {"end": [], "env": [], "w": [], "ratio": []} for H in horizons}
           for name in IDX}
    span = {}

    for name, path in IDX.items():
        try:
            d = pd.read_parquet(path)
        except Exception as e:
            print(f"  {name}: cannot read ({e})"); continue
        d["ts"] = pd.to_datetime(d["ts"])
        d = d[(d["ts"].dt.time >= SESSION_OPEN) & (d["ts"].dt.time <= SESSION_CLOSE)]
        d["day"] = d["ts"].dt.date
        span[name] = (d["day"].min(), d["day"].max(), d["day"].nunique())

        for day, g in d.groupby("day", sort=True):
            g = g.sort_values("ts").reset_index(drop=True)
            if len(g) < MIN_BARS + 2:
                continue
            ts = g["ts"].to_numpy()
            close = g["close"].to_numpy(float)
            high = g["high"].to_numpy(float)
            low = g["low"].to_numpy(float)
            for t in _pred_times(day):
                tp = np.datetime64(t)
                upto = ts <= tp
                nb = int(upto.sum())
                if nb < MIN_BARS:
                    continue
                c_up = close[upto]
                r5 = np.diff(c_up) / c_up[:-1] * 100.0        # 5-min returns %
                if len(r5) < MIN_BARS - 1:
                    continue
                sd5 = float(np.std(r5, ddof=1)) if len(r5) >= 2 else 0.0
                sig1 = sd5 / np.sqrt(5.0)                       # 5-min vol -> 1-min vol
                spot = float(c_up[-1])
                bp60 = _band_pct_60(sig1)                       # band % at 60m (base sig unit)
                if not np.isfinite(bp60) or bp60 <= 0 or spot <= 0:
                    continue
                for H in horizons:
                    t_end = tp + np.timedelta64(H, "m")
                    fwd = (ts > tp) & (ts <= t_end)
                    if not fwd.any():
                        continue
                    # forward window must actually reach ~t+H (5-min granularity tolerance)
                    if ts[fwd].max() < t_end - np.timedelta64(5, "m"):
                        continue
                    bp_h = bp60 * (H / 60.0) ** scout._BAND_HURST
                    half = spot * bp_h / 100.0
                    lo, hi = spot - half, spot + half
                    end_px = float(close[fwd][-1])
                    hi_px = float(high[fwd].max())
                    lo_px = float(low[fwd].min())
                    cell = per[name][H]
                    cell["end"].append(bool(lo <= end_px <= hi))
                    cell["env"].append(bool(hi_px <= hi and lo_px >= lo))
                    cell["w"].append(bp_h)
                    # |move| in units of the m=1.0 sig band at this horizon. Coverage at
                    # multiplier m = P(ratio <= m), so the m that hits 68% = 68th pct of ratio.
                    base_unit = spot * (bp_h / hf._RANGE_M) / 100.0   # half-width if m were 1.0
                    cell["ratio"].append(abs(end_px - spot) / base_unit)

    def pct(x):
        return f"{100 * np.mean(x):5.1f}%" if x else "   n/a"

    # ── overall table (pooled across indices) ────────────────────────────────
    print("\n  POOLED (all 4 indices)")
    print(f"  {'horizon':>8}  {'N':>6}  {'endpoint':>9}  {'envelope':>9}  {'avg band ±%':>12}  "
          f"{'m→68%':>7}")
    print("  " + "-" * 66)
    for H in horizons:
        end = sum((per[n][H]["end"] for n in IDX), [])
        env = sum((per[n][H]["env"] for n in IDX), [])
        w = sum((per[n][H]["w"] for n in IDX), [])
        ratio = sum((per[n][H]["ratio"] for n in IDX), [])
        wid = f"±{np.mean(w):.3f}%" if w else "n/a"
        # multiplier that would put 68% of endpoints inside the band (68th pct of ratio)
        m68 = f"{np.percentile(ratio, 68):.2f}" if ratio else "n/a"
        print(f"  {H:>6}m  {len(end):>6}  {pct(end):>9}  {pct(env):>9}  {wid:>12}  {m68:>7}")

    # ── per-index endpoint coverage ──────────────────────────────────────────
    print("\n  PER-INDEX endpoint coverage")
    print(f"  {'index':13}" + "".join(f"{str(H)+'m':>17}" for H in horizons))
    for name in IDX:
        cells = ""
        for H in horizons:
            e = per[name][H]["end"]
            cells += f"{pct(e)+' (n='+str(len(e))+')':>17}"
        print(f"  {name:13}{cells}")

    # ── per-index envelope (no-breach) coverage ──────────────────────────────
    print("\n  PER-INDEX envelope (no-breach) coverage")
    print(f"  {'index':13}" + "".join(f"{str(H)+'m':>17}" for H in horizons))
    for name in IDX:
        cells = ""
        for H in horizons:
            e = per[name][H]["env"]
            cells += f"{pct(e)+' (n='+str(len(e))+')':>17}"
        print(f"  {name:13}{cells}")

    # ── recommended _RANGE_M per index to hit 68% endpoint at each horizon ────
    print("\n  RECOMMENDED band multiplier to hit 68% endpoint (current _RANGE_M ="
          f" {hf._RANGE_M})")
    print(f"  {'index':13}" + "".join(f"{str(H)+'m':>17}" for H in horizons))
    for name in IDX:
        cells = ""
        for H in horizons:
            r = per[name][H]["ratio"]
            cells += f"{(f'{np.percentile(r,68):.2f}' if r else 'n/a'):>17}"
        print(f"  {name:13}{cells}")

    print("\n  DATA SPAN")
    for name, (a, b, n) in span.items():
        print(f"  {name:13} {a} -> {b}  ({n} sessions)")

    print("\n  READ: target 68% endpoint. <68% = band too TIGHT (raise _RANGE_M);"
          " >68% = too wide.")
    print("  4hr (240m) partly extrapolates past the 60m calibration point and late-day"
          " preds don't resolve\n  (session ends 15:30), so its N is smaller and noisier.")


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", default="30,60,120,240")
    a = ap.parse_args()
    run([int(x) for x in a.horizons.split(",")])
