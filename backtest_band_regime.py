"""
backtest_band_regime.py — does the RANGE-band's coverage depend on the market REGIME?

The band width is realised-vol × horizon-scaling, centred on spot — DIRECTION-free. It
already breathes with vol (sig1 updates every bar). The open question this answers:
does the RATIO of the realised 60m endpoint move to that vol estimate differ by REGIME?

  • TREND  — moves PERSIST (Hurst>0.5): the endpoint travels FURTHER than vol predicts
             → a flat multiplier UNDER-covers in trend (band too tight there).
  • CHOP   — moves MEAN-REVERT (Hurst<0.5): the endpoint stays CLOSER
             → a flat multiplier OVER-covers in chop (band too wide there).

If that's true, a single _RANGE_M=0.73 is a BLEND that is wrong in both regimes even
though it's right on average. A per-regime multiplier then tightens coverage toward 68%
WITHIN each regime. If instead coverage is ~flat across regimes, regime-conditioning adds
complexity for nothing — and we DON'T build it (no unnecessary regimes).

Method (causal, same band geometry as the live scout/cockpit):
  regime at t = regime_classifier.classify_series(day bars<=t, n=ER_N).mood, collapsed
                to {BIG_TREND, SMALL_TREND, CHOP} (band is symmetric → drop up/down sign).
  band_pct_H  = _RANGE_M · max(sig1,_SIG_FLOOR) · sqrt(60) · (H/60)^_BAND_HURST
  endpoint    = close at t+H inside [spot±half]
  rec mult    = 68th pctile of |end-spot| / (m=1 half-width)  →  the m that hits 68% here

    .venv\\Scripts\\python.exe backtest_band_regime.py [--horizon 60] [--er-n 10]
"""
from __future__ import annotations

import argparse
import datetime as dt

import numpy as np
import pandas as pd

import hour_forecast as hf            # _RANGE_M, _SIG_FLOOR
import intraday_scout as scout        # _BAND_HURST
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

# collapse the 5 signed moods to 3 regime buckets (band is symmetric → sign irrelevant)
BUCKET = {
    rc.BIG_UP: "BIG_TREND", rc.BIG_DOWN: "BIG_TREND",
    rc.SMALL_UP: "SMALL_TREND", rc.SMALL_DOWN: "SMALL_TREND",
    rc.CHOP: "CHOP",
}
REGIMES = ["BIG_TREND", "SMALL_TREND", "CHOP"]


def _pred_times(day: dt.date) -> list[dt.datetime]:
    t = dt.datetime.combine(day, FIRST_PRED)
    end = dt.datetime.combine(day, LAST_PRED)
    out = []
    while t <= end:
        out.append(t)
        t += dt.timedelta(minutes=STEP_MIN)
    return out


def run(H: int, er_n: int) -> None:
    print("=" * 92)
    print("  RANGE-BAND COVERAGE by REGIME — full 5-min history (deployed band geometry)")
    print(f"  H={H}m  band=_RANGE_M {hf._RANGE_M}·sig1·sqrt(60)·(H/60)^{scout._BAND_HURST}  "
          f"regime=ER{er_n} (er_hi {rc.ER_HI}/er_lo {rc.ER_LO}/drift {rc.DRIFT_BIG})")
    print("=" * 92)

    # per (index, regime): endpoint hits + |move|/base-sig ratios
    per = {name: {r: {"end": [], "ratio": []} for r in REGIMES} for name in IDX}

    for name, path in IDX.items():
        try:
            d = pd.read_parquet(path)
        except Exception as e:
            print(f"  {name}: cannot read ({e})"); continue
        d["ts"] = pd.to_datetime(d["ts"])
        d = d[(d["ts"].dt.time >= SESSION_OPEN) & (d["ts"].dt.time <= SESSION_CLOSE)]
        d["day"] = d["ts"].dt.date

        for day, g in d.groupby("day", sort=True):
            g = g.sort_values("ts").reset_index(drop=True)
            if len(g) < MIN_BARS + 2:
                continue
            # causal per-bar regime for the whole day (intraday ER resets each session)
            moods = rc.classify_series(g, n=er_n)["mood"].to_numpy()
            ts = g["ts"].to_numpy()
            close = g["close"].to_numpy(float)
            for t in _pred_times(day):
                tp = np.datetime64(t)
                upto = ts <= tp
                nb = int(upto.sum())
                if nb < MIN_BARS:
                    continue
                c_up = close[upto]
                r5 = np.diff(c_up) / c_up[:-1] * 100.0
                if len(r5) < MIN_BARS - 1:
                    continue
                sd5 = float(np.std(r5, ddof=1))
                sig1 = sd5 / np.sqrt(5.0)
                spot = float(c_up[-1])
                sig60 = max(sig1, hf._SIG_FLOOR) * np.sqrt(60.0)
                bp60 = hf._RANGE_M * sig60
                if not np.isfinite(bp60) or bp60 <= 0 or spot <= 0:
                    continue
                bp_h = bp60 * (H / 60.0) ** scout._BAND_HURST
                # regime AS OF the last closed bar up to t (idx nb-1)
                reg = BUCKET.get(moods[nb - 1], "CHOP")
                t_end = tp + np.timedelta64(H, "m")
                fwd = (ts > tp) & (ts <= t_end)
                if not fwd.any() or ts[fwd].max() < t_end - np.timedelta64(5, "m"):
                    continue
                half = spot * bp_h / 100.0
                end_px = float(close[fwd][-1])
                base_unit = spot * (bp_h / hf._RANGE_M) / 100.0     # m=1.0 half-width
                cell = per[name][reg]
                cell["end"].append(bool(spot - half <= end_px <= spot + half))
                cell["ratio"].append(abs(end_px - spot) / base_unit)

    def pct(x):
        return f"{100*np.mean(x):5.1f}%" if x else "  n/a"

    # candidate regime width factor (multiplies the base band); CHOP=1.0 anchor
    CAND = {"BIG_TREND": 1.08, "SMALL_TREND": 0.96, "CHOP": 1.00}

    # ── pooled per-regime ────────────────────────────────────────────────────
    print(f"\n  POOLED (all 4 indices)   current _RANGE_M = {hf._RANGE_M}")
    print(f"  {'regime':13}{'N':>7}{'endpoint':>10}{'share':>8}{'m→68%':>8}"
          f"{'AFTER×'+str('?'):>8}{'cover→':>9}")
    print("  " + "-" * 65)
    tot = sum(len(per[n][r]["end"]) for n in IDX for r in REGIMES)
    after_all = []
    for r in REGIMES:
        end = sum((per[n][r]["end"] for n in IDX), [])
        ratio = sum((per[n][r]["ratio"] for n in IDX), [])
        share = f"{100*len(end)/tot:4.1f}%" if tot else "n/a"
        m68 = f"{np.percentile(ratio,68):.2f}" if ratio else "n/a"
        # coverage if the base band is scaled by CAND[r]: hit iff ratio <= 0.73*factor
        thr = hf._RANGE_M * CAND[r]
        after = [x <= thr for x in ratio]
        after_all += after
        print(f"  {r:13}{len(end):>7}{pct(end):>10}{share:>8}{m68:>8}"
              f"{CAND[r]:>8.2f}{pct(after):>9}")
    print("  " + "-" * 65)
    print(f"  {'POOLED after factor':33}{'':>8}{'':>8}{pct(after_all):>17}"
          f"   (target 68% overall, held)")

    # ── per-index per-regime endpoint coverage ───────────────────────────────
    print("\n  PER-INDEX endpoint coverage by regime")
    print(f"  {'index':13}" + "".join(f"{r:>16}" for r in REGIMES))
    for name in IDX:
        cells = ""
        for r in REGIMES:
            e = per[name][r]["end"]
            cells += f"{pct(e)+' (n='+str(len(e))+')':>16}"
        print(f"  {name:13}{cells}")

    # ── recommended per-regime multiplier ────────────────────────────────────
    print("\n  RECOMMENDED multiplier per regime to hit 68% endpoint (pooled + per-index)")
    print(f"  {'index':13}" + "".join(f"{r:>16}" for r in REGIMES))
    pooled = ""
    for r in REGIMES:
        ratio = sum((per[n][r]["ratio"] for n in IDX), [])
        pooled += f"{(f'{np.percentile(ratio,68):.2f}' if ratio else 'n/a'):>16}"
    print(f"  {'POOLED':13}{pooled}")
    for name in IDX:
        cells = ""
        for r in REGIMES:
            ratio = per[name][r]["ratio"]
            cells += f"{(f'{np.percentile(ratio,68):.2f}' if ratio else 'n/a'):>16}"
        print(f"  {name:13}{cells}")

    print("\n  READ: if the m→68% column SPREADS across regimes (e.g. BIG_TREND≫CHOP),"
          " a flat\n  multiplier is wrong in every regime → build a regime-conditional band."
          " If it's\n  ~flat near 0.73, the band already handles regime via vol → do NOT add"
          " regimes.")


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--er-n", type=int, default=10)
    a = ap.parse_args()
    run(a.horizon, a.er_n)
