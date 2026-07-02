"""
backtest_band_closeclip.py — AUDIT the NEW near-close behaviour: does the session-CLIPPED
band (hour_forecast.session_horizon) with the time-of-day widen actually cover ~68% over the
REMAINING session, or did clip × tod combine into a mis-calibration?

The clip (commit d42beaf) was shipped WITHOUT measuring the clipped band's coverage, and the
tod ×1.15 factor was calibrated on UN-clipped 60m windows (14:00-14:30 predictions that
resolve at/before 15:30). For predictions after ~14:30 the 60m window would cross the close,
so it is now CLIPPED to time-to-close — a window the tod factor was never measured against.
This checks the deployed late-day geometry end to end:

    H_eff      = max(5, min(60, minutes_to_1530))                       # session_horizon
    band_pct   = _RANGE_M · regime_mult(mood) · tod_mult(t) · sig60 · (H_eff/60)^_BAND_HURST
    forward    = close at t+H_eff (clipped at 15:30) inside [spot ± half]

sig1 = cumulative session vol (the deployed estimator). Causal. Buckets late-day predictions
by H_eff so we see whether the clipped zone (H_eff<60, t>14:30) holds ~68% like the un-clipped
zone (H_eff=60, t in 14:00-14:30).

    .venv\\Scripts\\python.exe backtest_band_closeclip.py
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

import hour_forecast as hf
import regime_classifier as rc

IDX = {
    "NIFTY 50":     "data/historical/5min/NSE_NIFTY50_INDEX_5min.parquet",
    "BANK NIFTY":   "data/historical/5min/NSE_NIFTYBANK_INDEX_5min.parquet",
    "FIN NIFTY":    "data/historical/5min/NSE_FINNIFTY_INDEX_5min.parquet",
    "MIDCAP NIFTY": "data/historical/5min/NSE_MIDCPNIFTY_INDEX_5min.parquet",
}
SESSION_OPEN = dt.time(9, 15)
SESSION_CLOSE = dt.time(15, 30)
FIRST = dt.time(14, 0)          # the tod/clip zone only
LAST = dt.time(15, 25)
STEP = 5                         # finer grid to populate the clipped buckets
MIN_BARS = 6

# H_eff buckets (minutes)
BUCKETS = [("H=60 (t≤14:30, unclipped)", 60, 60),
           ("H=40-55 (clipped)", 40, 55),
           ("H=20-35 (clipped)", 20, 35),
           ("H=5-15  (clipped)", 5, 15)]


def _tod_mult(t: dt.time, h_eff: int = 60) -> float:
    return hf.tod_width_mult(
        dt.datetime.combine(dt.date(2000, 1, 1), t, tzinfo=hf.IST), h_eff)


def _h_eff(t: dt.time) -> int:
    to_close = (dt.datetime.combine(dt.date(2000, 1, 1), SESSION_CLOSE)
                - dt.datetime.combine(dt.date(2000, 1, 1), t)).total_seconds() / 60.0
    return int(max(5, min(60, to_close))) if to_close > 0 else 60


def _pred_times(day):
    t = dt.datetime.combine(day, FIRST); end = dt.datetime.combine(day, LAST)
    out = []
    while t <= end:
        out.append(t); t += dt.timedelta(minutes=STEP)
    return out


def run() -> None:
    print("=" * 92)
    print("  NEAR-CLOSE CLIPPED-BAND AUDIT — deployed geometry (clip × tod × regime), 2yr 5-min")
    print(f"  _RANGE_M={hf._RANGE_M} tod={hf._TOD_CLOSE_MULT}@≥{hf._TOD_CLOSE_START:%H:%M} "
          f"hurst={hf._BAND_HURST}")
    print("=" * 92)

    # coverage with the DEPLOYED band, and a counterfactual WITHOUT tod, per H_eff bucket
    dep = {b[0]: [] for b in BUCKETS}
    notod = {b[0]: [] for b in BUCKETS}
    per_idx = {n: {b[0]: [] for b in BUCKETS} for n in IDX}

    for name, path in IDX.items():
        d = pd.read_parquet(path)
        d["ts"] = pd.to_datetime(d["ts"])
        d = d[(d["ts"].dt.time >= SESSION_OPEN) & (d["ts"].dt.time <= SESSION_CLOSE)]
        d["day"] = d["ts"].dt.date
        for day, g in d.groupby("day", sort=True):
            g = g.sort_values("ts").reset_index(drop=True)
            if len(g) < MIN_BARS + 2:
                continue
            ts = g["ts"].to_numpy(); close = g["close"].to_numpy(float)
            moods = rc.classify_series(g, n=10)["mood"].to_numpy()
            for t in _pred_times(day):
                tp = np.datetime64(t); upto = ts <= tp
                nb = int(upto.sum())
                if nb < MIN_BARS:
                    continue
                c_up = close[upto]
                r5 = np.diff(c_up) / c_up[:-1] * 100.0
                if len(r5) < MIN_BARS - 1:
                    continue
                sig1 = float(np.std(r5, ddof=1)) / np.sqrt(5.0)
                sig60 = max(sig1, hf._SIG_FLOOR) * np.sqrt(60.0)
                spot = float(c_up[-1])
                H = _h_eff(t.time())
                t_end = np.datetime64(dt.datetime.combine(day, dt.time(15, 30))) \
                    if H < 60 else tp + np.timedelta64(60, "m")
                fwd = (ts > tp) & (ts <= t_end)
                if not fwd.any():
                    continue
                # require the window to actually reach ~t_end (within a bar)
                if ts[fwd].max() < t_end - np.timedelta64(5, "m"):
                    continue
                end_px = float(close[fwd][-1])
                mood = moods[nb - 1]
                rw = rc.band_width_mult(mood)
                hz = (H / 60.0) ** hf._BAND_HURST
                base = hf._RANGE_M * sig60 * hz
                half_dep = spot * (base * rw * _tod_mult(t.time(), H)) / 100.0
                half_not = spot * (base * rw) / 100.0
                hit_dep = spot - half_dep <= end_px <= spot + half_dep
                hit_not = spot - half_not <= end_px <= spot + half_not
                for label, lo, hi in BUCKETS:
                    if lo <= H <= hi:
                        dep[label].append(hit_dep); notod[label].append(hit_not)
                        per_idx[name][label].append(hit_dep)
                        break

    def pct(x):
        return f"{100*np.mean(x):5.1f}%" if x else "  n/a"

    print(f"\n  {'H_eff bucket':28}{'N':>8}{'deployed':>11}{'(no tod)':>10}")
    print("  " + "-" * 58)
    for label, lo, hi in BUCKETS:
        print(f"  {label:28}{len(dep[label]):>8}{pct(dep[label]):>11}{pct(notod[label]):>10}")

    print("\n  PER-INDEX (deployed)")
    print(f"  {'index':13}" + "".join(f"{b[0].split()[0]:>10}" for b in BUCKETS))
    for name in IDX:
        print(f"  {name:13}" + "".join(f"{pct(per_idx[name][b[0]]):>10}" for b in BUCKETS))

    print("\n  READ: every 'deployed' row should sit ~68%. If the clipped buckets (H<60) are")
    print("  <<68% the clip made the band too TIGHT into the close (raise the close band); if")
    print("  >>68% tod double-counts on the clipped window (drop tod when clipped). The (no tod)")
    print("  column isolates whether tod is helping or hurting each bucket.")


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    run()
