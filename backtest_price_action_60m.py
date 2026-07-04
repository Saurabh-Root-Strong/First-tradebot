"""
backtest_price_action_60m.py — does CANDLE PRICE-ACTION at t improve the next-60m call?

The scout's validated product is the vol RANGE band (~68%); direction is a measured
coin-flip and the mood gate (ER) already widens the band ×1.08 in BIG_TREND. The open
question: does finer price-action STRUCTURE — consolidation (tight coil), breakout of
that coil, breakdown, directional impulse candles — carry information the ER mood does
NOT already capture, for either
  (a) DIRECTION of the next 60m (win% / mean bps vs cost), or
  (b) BAND WIDTH (does coverage differ by state → per-state multiplier m→68%)?

States at t (CAUSAL, last closed 5-min bars only, priority top→down):
  BREAK_UP   — last close > prior-60m high (excl last bar) AND prior 60m was compressed
  BREAK_DN   — mirror below prior-60m low
  COIL       — trailing 60m range < COMP_K × trailing 240m avg 60m range (tight, no break)
  IMPULSE_UP — last 3 bars net body ratio > +BODY_K (strong one-way candles)
  IMPULSE_DN — mirror
  DRIFT      — everything else

Forward over H=60m: signed return (state direction), endpoint inside deployed band
geometry (_RANGE_M·sig1·√60, ER-regime ×1.08 in BIG_TREND — i.e. what the live scout
shows), and m→68% per state. Halves split for stability. Costs: index futures round
trip ≈ 3 bps (BTST-audited); options wall ≈ 0.2R — any mean edge must clear that.

    .venv\\Scripts\\python.exe backtest_price_action_60m.py [--horizon 60] [--day 2026-07-03]
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
FIRST_PRED = dt.time(10, 15)   # need 60m of structure before classifying
LAST_PRED = dt.time(14, 30)    # full 60m forward window inside the session
STEP_MIN = 15
W = 12                         # 12×5m = 60m structure window
LOOK = 48                      # 48×5m = 240m for the compression baseline
COMP_K = 0.60                  # trailing 60m range < 60% of typical 60m range = tight
BODY_K = 0.60                  # |net body| / total range of last 3 bars
COST_BPS = 3.0                 # index futures round trip

BUCKET = {rc.BIG_UP: "BIG_TREND", rc.BIG_DOWN: "BIG_TREND",
          rc.SMALL_UP: "SMALL_TREND", rc.SMALL_DOWN: "SMALL_TREND",
          rc.CHOP: "CHOP"}
REG_MULT = {"BIG_TREND": 1.08, "SMALL_TREND": 1.00, "CHOP": 1.00}   # deployed
STATES = ["BREAK_UP", "BREAK_DN", "COIL", "IMPULSE_UP", "IMPULSE_DN", "DRIFT"]
SIGN = {"BREAK_UP": 1, "BREAK_DN": -1, "IMPULSE_UP": 1, "IMPULSE_DN": -1,
        "COIL": 0, "DRIFT": 0}


def _pred_times(day):
    t = dt.datetime.combine(day, FIRST_PRED)
    end = dt.datetime.combine(day, LAST_PRED)
    out = []
    while t <= end:
        out.append(t); t += dt.timedelta(minutes=STEP_MIN)
    return out


def _state(o, h, l, c, i):
    """Price-action state using bars [0..i] (i = last closed bar index). Causal."""
    if i + 1 < W + 6:
        return None, np.nan
    hi_w, lo_w = h[i - W + 1:i + 1].max(), l[i - W + 1:i + 1].min()
    rng_w = hi_w - lo_w
    # typical 60m range over the prior LOOK bars (rolling, ends BEFORE the current window)
    j0 = max(0, i - W + 1 - LOOK)
    base = []
    for j in range(j0, i - W + 1):
        base.append(h[j:j + W].max() - l[j:j + W].min())
    typ = float(np.median(base)) if base else np.nan
    compressed = bool(typ == typ and typ > 0 and rng_w < COMP_K * typ)
    # prior-60m extremes EXCLUDING the last bar → breakout test on the last close
    hi_prev = h[i - W:i].max()
    lo_prev = l[i - W:i].min()
    # was the window BEFORE this bar compressed? (breakout must come FROM a base)
    rng_prev = hi_prev - lo_prev
    comp_prev = bool(typ == typ and typ > 0 and rng_prev < COMP_K * typ)
    if comp_prev and c[i] > hi_prev:
        return "BREAK_UP", rng_w
    if comp_prev and c[i] < lo_prev:
        return "BREAK_DN", rng_w
    if compressed:
        return "COIL", rng_w
    body = float(np.sum(c[i - 2:i + 1] - o[i - 2:i + 1]))
    total = float(np.sum(h[i - 2:i + 1] - l[i - 2:i + 1]))
    if total > 0 and abs(body) / total > BODY_K:
        return ("IMPULSE_UP" if body > 0 else "IMPULSE_DN"), rng_w
    return "DRIFT", rng_w


def run(H: int, spot_day: str | None) -> None:
    print("=" * 96)
    print(f"  PRICE-ACTION states → next-{H}m DIRECTION + BAND — 5-min history, causal")
    print(f"  states: break-from-base / coil / 3-bar impulse / drift   cost={COST_BPS}bps fut RT")
    print("=" * 96)

    # cell[state] = dicts of lists: sret (signed bps for directional states),
    # ret (raw bps), end (band hit under deployed geometry), ratio (|move|/m1-unit), half tag
    cell = {s: {"sret": [], "ret": [], "end": [], "ratio": [], "h2": []} for s in STATES}
    spot_rows = []
    all_days = set()

    for name, path in IDX.items():
        try:
            d = pd.read_parquet(path)
        except Exception as e:
            print(f"  {name}: cannot read ({e})"); continue
        d["ts"] = pd.to_datetime(d["ts"])
        d = d[(d["ts"].dt.time >= SESSION_OPEN) & (d["ts"].dt.time <= SESSION_CLOSE)]
        d["day"] = d["ts"].dt.date
        days = sorted(d["day"].unique())
        mid = days[len(days) // 2] if days else None

        for day, g in d.groupby("day", sort=True):
            g = g.sort_values("ts").reset_index(drop=True)
            if len(g) < W + 14:
                continue
            all_days.add(day)
            moods = rc.classify_series(g, n=10)["mood"].to_numpy()
            ts = g["ts"].to_numpy()
            o = g["open"].to_numpy(float); h = g["high"].to_numpy(float)
            l = g["low"].to_numpy(float);  c = g["close"].to_numpy(float)
            for t in _pred_times(day):
                tp = np.datetime64(t)
                upto = ts <= tp
                nb = int(upto.sum())
                if nb < W + 6:
                    continue
                i = nb - 1
                st, _ = _state(o, h, l, c, i)
                if st is None:
                    continue
                spot = float(c[i])
                r5 = np.diff(c[:nb]) / c[:nb - 1] * 100.0
                sd5 = float(np.std(r5, ddof=1))
                sig1 = sd5 / np.sqrt(5.0)
                bp60 = hf._RANGE_M * max(sig1, hf._SIG_FLOOR) * np.sqrt(60.0)
                bp_h = bp60 * (H / 60.0) ** scout._BAND_HURST
                reg = BUCKET.get(moods[i], "CHOP")
                bp_live = bp_h * REG_MULT[reg]                     # what the scout shows
                t_end = tp + np.timedelta64(H, "m")
                fwd = (ts > tp) & (ts <= t_end)
                if not fwd.any() or ts[fwd].max() < t_end - np.timedelta64(5, "m"):
                    continue
                end_px = float(c[fwd][-1])
                ret_bps = (end_px / spot - 1.0) * 1e4
                half = spot * bp_live / 100.0
                base_unit = spot * (bp_h / hf._RANGE_M) / 100.0    # m=1 half-width
                cc = cell[st]
                cc["ret"].append(ret_bps)
                if SIGN[st]:
                    cc["sret"].append(SIGN[st] * ret_bps)
                cc["end"].append(bool(abs(end_px - spot) <= half))
                cc["ratio"].append(abs(end_px - spot) / base_unit)
                cc["h2"].append(bool(mid is not None and day >= mid))
                if spot_day and str(day) == spot_day:
                    spot_rows.append((name, t.strftime("%H:%M"), st, reg,
                                      round(ret_bps, 1),
                                      "HIT" if abs(end_px - spot) <= half else "MISS"))

    n_all = sum(len(cell[s]["ret"]) for s in STATES)
    print(f"\n  sample: {n_all} predictions, {len(all_days)} days, 4 indices, step {STEP_MIN}m")

    # ── A) DIRECTION — does the state's arrow predict the next 60m? ──────────────
    print(f"\n  A) DIRECTION test (signed next-{H}m return in the state's direction)")
    print(f"  {'state':11}{'n':>7}{'win%':>7}{'mean bps':>10}{'t':>7}{'net bps':>9}"
          f"{'h1 bps':>8}{'h2 bps':>8}")
    print("  " + "-" * 68)
    for s in STATES:
        r = np.array(cell[s]["sret"] if SIGN[s] else
                     [abs(x) for x in cell[s]["ret"]])
        if SIGN[s] == 0:
            # non-directional states: report |move| just for context, no direction test
            n = len(cell[s]["ret"])
            mu = np.mean(np.array(cell[s]["ret"])) if n else np.nan
            print(f"  {s:11}{n:>7}{'—':>7}{mu:>10.1f}{'—':>7}{'—':>9}"
                  f"{'—':>8}{'—':>8}   (no arrow; raw mean shown)")
            continue
        n = len(r)
        if n < 8:
            print(f"  {s:11}{n:>7}   thin"); continue
        mu, sd = float(np.mean(r)), float(np.std(r, ddof=1))
        tstat = mu / (sd / np.sqrt(n)) if sd > 0 else np.nan
        win = 100.0 * float(np.mean(r > 0))
        h2 = np.array(cell[s]["h2"])
        mu1 = float(np.mean(r[~h2])) if (~h2).any() else np.nan
        mu2 = float(np.mean(r[h2])) if h2.any() else np.nan
        print(f"  {s:11}{n:>7}{win:>6.1f}%{mu:>10.1f}{tstat:>7.2f}{mu - COST_BPS:>9.1f}"
              f"{mu1:>8.1f}{mu2:>8.1f}")

    # ── B) BAND — coverage per state under the LIVE geometry + m→68% ─────────────
    print(f"\n  B) BAND test (endpoint inside the deployed band — incl. BIG_TREND ×1.08)")
    print(f"  {'state':11}{'n':>7}{'cover':>8}{'m→68%':>8}   (target 68%; current m=0.73)")
    print("  " + "-" * 60)
    for s in STATES:
        e, ratio = cell[s]["end"], cell[s]["ratio"]
        if len(e) < 20:
            print(f"  {s:11}{len(e):>7}   thin"); continue
        m68 = np.percentile(ratio, 68)
        print(f"  {s:11}{len(e):>7}{100*np.mean(e):>7.1f}%{m68:>8.2f}")

    # ── C) July-3 style spot check ────────────────────────────────────────────────
    if spot_day:
        print(f"\n  C) SPOT CHECK {spot_day} (state @ t → next-{H}m signed bps, band verdict)")
        if not spot_rows:
            print(f"     no rows — {spot_day} not in the historical parquet yet "
                  f"(re-run download_historical.py)")
        for row in spot_rows:
            print(f"     {row[0]:13}{row[1]}  {row[2]:11}{row[3]:12}{row[4]:>8}  {row[5]}")

    print("\n  READ: a state earns a wire ONLY if (A) |t|>3 with both halves same sign and"
          "\n  net>0 after cost, or (B) m→68% spreads ≫/≪ 0.73 beyond what the ER mood mult"
          "\n  already applies. Otherwise price-action adds NOTHING over vol+ER → do not wire.")


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--day", type=str, default="2026-07-03")
    a = ap.parse_args()
    run(a.horizon, a.day)
