"""
backtest_band_scenarios.py — stress the 60m band across the conditions a flat daily
coverage number hides:

  1. TIME-OF-DAY. The live vol estimate (hour_forecast._sigma1_pct) is SESSION-CUMULATIVE
     — std of every 1-min return since the open. Intraday vol is U-shaped (high at open +
     close, low midday). A cumulative estimate therefore CARRIES the hot opening vol into
     the quiet midday (band too WIDE midday → over-covers) and LAGS the afternoon pickup
     (band too TIGHT near close → under-covers). We measure endpoint coverage by session
     bucket to see if the 68% average is actually ~68% at every hour, or a blend of over
     and under.

  2. SIGMA ESTIMATOR. Same test for a ROLLING last-60m window and an EWMA vs the deployed
     CUMULATIVE — does a recency-weighted vol track the U-shape better (flatter coverage
     across the day, still ~68% overall)?

  3. SIGMA FLOOR. hour_forecast._SIG_FLOOR clamps 1-min sigma at 0.02%. How often does it
     BIND, and is coverage distorted when it does (a floor that fires in genuinely quiet
     tape inflates the band → over-covers)?

Deployed band geometry, causal, on the full 5-min history:
    band_pct_60 = _RANGE_M * max(sig1, _SIG_FLOOR) * sqrt(60)
    sig1 (1-min vol %) = std(5-min returns %) / sqrt(5)

    .venv\\Scripts\\python.exe backtest_band_scenarios.py
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
FIRST_PRED = dt.time(9, 45)
LAST_PRED = dt.time(15, 0)
STEP_MIN = 15
MIN_BARS = 6
H = 60
ROLL = 12                 # rolling window = 12 five-min bars = last 60 min


def _tod_bucket(t: dt.time) -> str:
    if t < dt.time(10, 30):
        return "1 open  09:45-10:30"
    if t < dt.time(13, 0):
        return "2 mid   10:30-13:00"
    if t < dt.time(14, 0):
        return "3 pm    13:00-14:00"
    return "4 close 14:00-15:00"


BUCKETS = ["1 open  09:45-10:30", "2 mid   10:30-13:00",
           "3 pm    13:00-14:00", "4 close 14:00-15:00"]
ESTS = ["cumulative", "rolling60", "ewma"]


def _pred_times(day):
    t = dt.datetime.combine(day, FIRST_PRED)
    end = dt.datetime.combine(day, LAST_PRED)
    out = []
    while t <= end:
        out.append(t); t += dt.timedelta(minutes=STEP_MIN)
    return out


def _sigmas(r5: np.ndarray) -> dict:
    """1-min sigma % from 5-min returns, three ways (all causal, use r5 up to t)."""
    sd_cum = float(np.std(r5, ddof=1))
    sd_roll = float(np.std(r5[-ROLL:], ddof=1)) if len(r5) >= 2 else sd_cum
    # EWMA variance (RiskMetrics-style), lambda 0.94 per 5-min step
    lam = 0.94
    w = (1 - lam) * lam ** np.arange(len(r5))[::-1]
    m = np.average(r5, weights=w)
    var_ewma = np.average((r5 - m) ** 2, weights=w)
    sd_ewma = float(np.sqrt(var_ewma))
    return {"cumulative": sd_cum / np.sqrt(5.0),
            "rolling60":  sd_roll / np.sqrt(5.0),
            "ewma":       sd_ewma / np.sqrt(5.0)}


def run() -> None:
    print("=" * 96)
    print(f"  BAND SCENARIO AUDIT — H={H}m, deployed _RANGE_M={hf._RANGE_M}, "
          f"_SIG_FLOOR={hf._SIG_FLOOR}, full 5-min history")
    print("=" * 96)

    # coverage[est][bucket] = list of hits ; floor bind stats for the deployed estimator
    cov = {e: {b: [] for b in BUCKETS} for e in ESTS}
    cov_all = {e: [] for e in ESTS}
    floor_bind = []                 # (bound?, hit) for the deployed cumulative estimator
    per_idx = {n: {b: [] for b in BUCKETS} for n in IDX}   # cumulative, per index×bucket
    # close-hour (t>=14:00) coverage + |move|/base-unit ratio, split by trend regime,
    # to check the close miss is INDEPENDENT of the regime factor (no double-count)
    close_reg = {"BIG_TREND": {"hit": [], "ratio": []},
                 "not_big":   {"hit": [], "ratio": []}}

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
                spot = float(c_up[-1])
                t_end = tp + np.timedelta64(H, "m")
                fwd = (ts > tp) & (ts <= t_end)
                if not fwd.any() or ts[fwd].max() < t_end - np.timedelta64(5, "m"):
                    continue
                end_px = float(close[fwd][-1])
                bucket = _tod_bucket(t.time())
                sig = _sigmas(r5)
                for e in ESTS:
                    s = max(sig[e], hf._SIG_FLOOR)
                    half = spot * (hf._RANGE_M * s * np.sqrt(60.0)) / 100.0
                    hit = bool(spot - half <= end_px <= spot + half)
                    cov[e][bucket].append(hit); cov_all[e].append(hit)
                    if e == "cumulative":
                        per_idx[name][bucket].append(hit)
                # floor binding for the deployed (cumulative) estimator
                floor_bind.append((sig["cumulative"] < hf._SIG_FLOOR,
                                   cov["cumulative"][bucket][-1]))
                # close-hour × regime split (cumulative estimator)
                if t.time() >= dt.time(14, 0):
                    s = max(sig["cumulative"], hf._SIG_FLOOR)
                    half = spot * (hf._RANGE_M * s * np.sqrt(60.0)) / 100.0
                    hit = bool(spot - half <= end_px <= spot + half)
                    base_unit = spot * (s * np.sqrt(60.0)) / 100.0     # m=1.0 half-width
                    is_big = moods[nb - 1] in (rc.BIG_UP, rc.BIG_DOWN)
                    k = "BIG_TREND" if is_big else "not_big"
                    close_reg[k]["hit"].append(hit)
                    close_reg[k]["ratio"].append(abs(end_px - spot) / base_unit)

    def pct(x):
        return f"{100*np.mean(x):5.1f}%" if x else "  n/a"

    print("\n  OVERALL 60m coverage by sigma estimator (target 68%)")
    for e in ESTS:
        print(f"    {e:12} {pct(cov_all[e])}  (n={len(cov_all[e])})")

    print("\n  COVERAGE by TIME-OF-DAY  (target 68% in EVERY row = well-calibrated intraday)")
    print(f"    {'bucket':22}" + "".join(f"{e:>13}" for e in ESTS))
    for b in BUCKETS:
        row = "".join(f"{pct(cov[e][b]):>13}" for e in ESTS)
        print(f"    {b:22}{row}   n={len(cov['cumulative'][b])}")

    print("\n  PER-INDEX coverage by time-of-day (deployed cumulative) — is the close miss robust?")
    print(f"    {'index':13}" + "".join(f"{b.split()[0]:>9}" for b in BUCKETS))
    for name in IDX:
        print(f"    {name:13}" + "".join(f"{pct(per_idx[name][b]):>9}" for b in BUCKETS))

    bound = [h for (bd, h) in floor_bind if bd]
    free = [h for (bd, h) in floor_bind if not bd]
    rate = 100 * len(bound) / len(floor_bind) if floor_bind else 0
    print(f"\n  SIGMA FLOOR (_SIG_FLOOR={hf._SIG_FLOOR}) — deployed cumulative estimator")
    print(f"    binds on {rate:.1f}% of predictions")
    print(f"    coverage when floor BINDS : {pct(bound)}  (n={len(bound)})")
    print(f"    coverage when floor FREE  : {pct(free)}  (n={len(free)})")

    print("\n  CLOSE-HOUR (t>=14:00) coverage split by REGIME — is the miss INDEPENDENT of trend?")
    for k in ("not_big", "BIG_TREND"):
        h = close_reg[k]["hit"]; r = close_reg[k]["ratio"]
        m68 = f"{np.percentile(r,68):.2f}" if r else "n/a"
        print(f"    {k:10} coverage {pct(h)}  (n={len(h)})   m->68% = {m68}")
    print("    -> if 'not_big' (CHOP+SMALL) close coverage is ALSO <68%, the close widen is a")
    print("       SEPARATE time-of-day factor (composes with the regime factor, no double-count).")
    # verify the deployed ×1.15 close widen (regime factor also applies to the BIG slice)
    allr = np.array(close_reg["not_big"]["ratio"] + close_reg["BIG_TREND"]["ratio"])
    big_mask = np.array([False] * len(close_reg["not_big"]["ratio"])
                        + [True] * len(close_reg["BIG_TREND"]["ratio"]))
    thr = np.where(big_mask, hf._RANGE_M * 1.08 * 1.15, hf._RANGE_M * 1.15)  # regime×tod
    after = allr <= thr
    before = allr <= hf._RANGE_M
    print(f"    CLOSE coverage  before={100*before.mean():.1f}%  ->  after ×1.15 tod "
          f"(+×1.08 on BIG) = {100*after.mean():.1f}%   (target 68)")

    print("\n  READ: flat ~68% down the TIME-OF-DAY column = intraday-calibrated. A cumulative")
    print("  estimator that reads high at open + low midday would show open UNDER / midday OVER.")
    print("  If rolling/ewma flatten that spread while holding ~68% overall, the live")
    print("  _sigma1_pct (session-cumulative) should switch to a recency-weighted window.")


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    run()
