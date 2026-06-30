"""
backtest_levels.py — confirmed horizontal S/R: accuracy across ALL timeframe sets.

Formalises the desk method: a price is a real level when multiple swing pivots
cluster there, the level is confirmed on a chosen set of timeframes, and it persists
across >=2 days. Detected as-of t from prior bars only (lookahead-free), rebuilt
from captured ticks resampled to 5/15/30/60m across the prior sessions + today<=t.

Sweeps the TF-confirmation requirement so we can read which timeframe pairing turns
price best:
    5&15, 5&30, 15&30, 15&60, 30&60, 5&15&30, any-2-of-4
Only the LEVEL DEFINITION varies by config; the reaction/breakout logic is identical,
so the comparison is clean. Two claims, each a directional hit rate + day-block CI
at 15/30/60m forward:
  A. REVERSION — at a confirmed level, does price bounce/reject its way?
  B. BREAKOUT  — after a 15m close beyond a confirmed level, does it continue?

Both signed so >50% = the method was right; edge only where day-block CI clears 50.

    set LV_CACHE=...lv.parquet  &&  .venv\\Scripts\\python.exe backtest_levels.py
"""
from __future__ import annotations

import datetime
import glob
import math
import os
import sys

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from core.constants import IST, INDEX_SYMBOLS, LIVE_DIR
from core.mirror_io import read_mirror as _read

SEED = 7
PRIOR_DAYS = 5
PIVOT_K = 2
TOL_FRAC = 0.0012
PROX_FRAC = 0.0008
BRK_MARGIN = 0.0005
MIN_TOUCH = 3
MIN_DAYS = 2
BRK_TF = 15                    # breakout close-confirmation TF (fixed across configs)
TFS = [5, 15, 30, 60]
CONFIGS = {                    # name -> required TF set ("any2" = level on >=2 of the 4)
    "5&15": {5, 15}, "5&30": {5, 30}, "15&30": {15, 30},
    "15&60": {15, 60}, "30&60": {30, 60}, "5&15&30": {5, 15, 30}, "any2of4": "any2",
}
ENTRY_TIMES = ["09:45", "10:15", "10:45", "11:15", "11:45", "12:15", "12:45", "13:15", "13:45", "14:15"]
HORIZONS = (15, 30, 60)
CACHE = os.environ.get("LV_CACHE", "")


def _captured_days():
    out = set()
    for p in glob.glob(str(LIVE_DIR / "*_ticks.parquet")):
        if os.path.getsize(p) < 2000:
            continue
        name = os.path.basename(p).split("_")[0]
        try:
            datetime.date.fromisoformat(name)        # skip tmp_/partial/non-date files
        except ValueError:
            continue
        out.add(name)
    return sorted(out)


def _session_ticks(date, sym):
    t = _read("ticks", date, None, sym)
    if t is None or len(t) < 20:
        return None
    t = t[(t["ts"].dt.date == datetime.date.fromisoformat(date))
          & (t["ts"].dt.time >= datetime.time(9, 15))
          & (t["ts"].dt.time <= datetime.time(15, 30))]
    return t[["ts", "ltp"]].sort_values("ts") if len(t) else None


def _bars(ticks, tf):
    g = ticks.set_index("ts")["ltp"]
    return g.resample(f"{tf}min", origin="start_day").ohlc().dropna()


def _pivots(bars, tf):
    hi = bars["high"].to_numpy(float); lo = bars["low"].to_numpy(float)
    idx = bars.index; out = []; k = PIVOT_K
    for i in range(k, len(bars) - k):
        win_hi = hi[i - k:i + k + 1]; win_lo = lo[i - k:i + k + 1]
        if hi[i] == win_hi.max() and win_hi.argmax() == k:
            out.append((idx[i], float(hi[i]), tf))
        if lo[i] == win_lo.min() and win_lo.argmin() == k:
            out.append((idx[i], float(lo[i]), tf))
    return out


def _clusters(pivots, as_of, spot):
    pv = sorted(((ts, px, tf) for (ts, px, tf) in pivots if ts < as_of), key=lambda x: x[1])
    if not pv:
        return []
    tol = TOL_FRAC * spot
    out = [[pv[0]]]
    for p in pv[1:]:
        if p[1] - out[-1][-1][1] <= tol:
            out[-1].append(p)
        else:
            out.append([p])
    return out


def _levels(clusters, required):
    levels = []
    for c in clusters:
        if len(c) < MIN_TOUCH:
            continue
        tfs = set(x[2] for x in c)
        days = set(x[0].date() for x in c)
        if len(days) < MIN_DAYS:
            continue
        ok = (len(tfs) >= 2) if required == "any2" else required.issubset(tfs)
        if ok:
            levels.append(float(np.mean([x[1] for x in c])))
    return sorted(levels)


def harvest(days):
    rows = []
    for di, date in enumerate(days):
        if di < PRIOR_DAYS:
            continue
        priors = days[di - PRIOR_DAYS:di]
        d0 = datetime.date.fromisoformat(date)
        for sym in INDEX_SYMBOLS:
            today = _session_ticks(date, sym)
            if today is None:
                continue
            hist = [x for x in (_session_ticks(p, sym) for p in priors) if x is not None]
            if not hist:
                continue
            base = pd.concat(hist)
            for hhmm in ENTRY_TIMES:
                hh, mm = map(int, hhmm.split(":"))
                t = datetime.datetime.combine(d0, datetime.time(hh, mm), tzinfo=IST)
                tt = pd.Timestamp(t)
                upto = today[today["ts"] <= tt]
                if len(upto) < 6:
                    continue
                spot = float(upto.iloc[-1]["ltp"])
                ctx = pd.concat([base, upto])
                bars = {tf: _bars(ctx, tf) for tf in TFS}
                if len(bars[BRK_TF]) < 2 * PIVOT_K + 2:
                    continue
                pivots = []
                for tf in TFS:
                    pivots += _pivots(bars[tf], tf)
                clusters = _clusters(pivots, tt, spot)
                cb = bars[BRK_TF][bars[BRK_TF].index < tt]
                last_close = float(cb["close"].iloc[-1]) if len(cb) else spot
                s30 = today[today["ts"] <= pd.Timestamp(t - datetime.timedelta(minutes=30))]
                spot30 = float(s30.iloc[-1]["ltp"]) if len(s30) else spot
                drift = (spot / spot30 - 1.0) * 100.0 if spot30 else 0.0
                day_open = float(upto.iloc[0]["ltp"])
                net_sofar = (spot / day_open - 1.0) * 100.0 if day_open else 0.0
                day_dir = 1 if net_sofar > 0.15 else (-1 if net_sofar < -0.15 else 0)
                fwd = {}
                for H in HORIZONS:
                    sH = today[today["ts"] <= pd.Timestamp(t + datetime.timedelta(minutes=H))]
                    fwd[H] = float(sH.iloc[-1]["ltp"]) if len(sH) else None
                for cfg, req in CONFIGS.items():
                    levels = _levels(clusters, req)
                    if not levels:
                        continue
                    la = np.array(levels)
                    sup = la[la <= spot]; res = la[la >= spot]
                    nsup = float(sup.max()) if len(sup) else None
                    nres = float(res.min()) if len(res) else None
                    at_sup = nsup is not None and abs(spot - nsup) / spot < PROX_FRAC
                    at_res = nres is not None and abs(spot - nres) / spot < PROX_FRAC
                    rev_sign = 1 if at_sup else (-1 if at_res else 0)
                    up_brk = any((spot >= lv) and (lv < last_close) and
                                 last_close > lv * (1 + BRK_MARGIN) and
                                 abs(last_close - lv) / spot < 2 * TOL_FRAC for lv in levels)
                    dn_brk = any((spot <= lv) and (lv > last_close) and
                                 last_close < lv * (1 - BRK_MARGIN) and
                                 abs(last_close - lv) / spot < 2 * TOL_FRAC for lv in levels)
                    brk_sign = 1 if (up_brk and not dn_brk) else (-1 if (dn_brk and not up_brk) else 0)
                    rec = {"date": date, "sym": sym, "t": hhmm, "cfg": cfg, "drift": drift,
                           "n_levels": len(levels), "at_level": bool(rev_sign),
                           "is_brk": bool(brk_sign), "brk_dir": int(brk_sign), "day_dir": day_dir}
                    for H in HORIZONS:
                        r = ((fwd[H] / spot - 1.0) * 100.0) if (fwd[H] and fwd[H] != spot) else np.nan
                        rec[f"rev{H}"] = rev_sign * r if rev_sign else np.nan
                        rec[f"brk{H}"] = brk_sign * r if brk_sign else np.nan
                    rows.append(rec)
    return pd.DataFrame(rows)


def _wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * p, 100 * (c - h), 100 * (c + h)


def _dayblock(hits, days, rng, reps=2000):
    hits = np.asarray(hits, float); days = np.asarray(days)
    keep = ~np.isnan(hits); hits, days = hits[keep], days[keep]
    if len(hits) < 5:
        return float("nan"), float("nan"), float("nan"), 0
    u = np.unique(days); idx = {d: np.where(days == d)[0] for d in u}
    bs = [hits[np.concatenate([idx[d] for d in rng.choice(u, len(u), replace=True)])].mean()
          for _ in range(reps)]
    return 100 * hits.mean(), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5), len(hits)


def _row(sub, col, H, rng):
    s = sub[f"{col}{H}"]
    sgn = (s > 0).astype(float).where(s.notna())
    n = int(sgn.notna().sum())
    if n < 5:
        return f"{H}m n={n}(thin)"
    wm, wlo, whi = _wilson(int(np.nansum(sgn)), n)
    bm, blo, bhi, _ = _dayblock(sgn, sub.date.to_numpy(), rng)
    flag = "+" if blo > 50 else ("-" if bhi < 50 else "0")
    return f"{H}m {wm:3.0f}% db[{blo:3.0f},{bhi:3.0f}][{flag}] n={n}"


def report(df, rng):
    print("\nCONFIRMED-LEVEL ACCURACY — swept over timeframe-confirmation sets")
    print("=" * 92)
    days = sorted(df.date.unique())
    print(f"  rows={len(df)}  days={len(days)} ({days[0]}..{days[-1]})  "
          f"(db = day-block CI; edge only if it clears 50)\n")

    print(f"  {'config':9s} {'n@lvl':>5} | A) REVERSION (bounce/reject)            "
          f"| {'n@brk':>5} | B) BREAKOUT (continuation)")
    for cfg in CONFIGS:
        c = df[df.cfg == cfg]
        al = c[c.at_level]; bk = c[c.is_brk]
        rev = "  ".join(_row(al, "rev", H, rng) for H in HORIZONS)
        brk = "  ".join(_row(bk, "brk", H, rng) for H in HORIZONS)
        print(f"  {cfg:9s} {int(c.at_level.sum()):>5} | {rev}")
        print(f"  {'':9s} {'':>5} | {'':38s} {int(c.is_brk.sum()):>5} | {brk}")
        print()

    print("  " + "-" * 88)
    print("  CONFOUND CHECK — breakout continuation by direction (any2of4, the down-week test)")
    print("  if only DOWN-breaks continue -> it's trend not a level edge; UP too -> real breakout")
    bk = df[(df.cfg == "any2of4") & df.is_brk]
    for lbl, d in (("UP-break  ", bk[bk.brk_dir > 0]), ("DOWN-break", bk[bk.brk_dir < 0])):
        cells = "  ".join(_row(d, "brk", H, rng) for H in HORIZONS)
        print(f"    {lbl}  {cells}")
    print("  break-dir x DAY-so-far regime (is up-break failure a confound or a real asymmetry?)")
    for bd, bl in ((1, "UP-break"), (-1, "DOWN-break")):
        for dd, dl in ((1, "up-day"), (-1, "down-day")):
            d = bk[(bk.brk_dir == bd) & (bk.day_dir == dd)]
            print(f"    {bl:10s} on {dl:8s}  {_row(d, 'brk', 15, rng)}")

    print("=" * 92)
    print("READ: REVERSION>50 (db clears) = that TF-set turns price (fade works);")
    print("<50 = price breaks through (don't fade). BREAKOUT>50 = real breaks continue.")
    print("Pick the TF-set whose db CI clears 50 on the most horizons; else levels are")
    print("a RISK MAP + SL reference only, not a directional entry.")


def main():
    rng = np.random.default_rng(SEED)
    if CACHE and os.path.exists(CACHE):
        df = pd.read_parquet(CACHE); print(f"loaded cache {CACHE} rows={len(df)}")
    else:
        days = _captured_days()
        print(f"captured days: {days}")
        df = harvest(days)
        if len(df) == 0:
            print("no rows"); return
        if CACHE:
            df.to_parquet(CACHE); print(f"cached -> {CACHE}")
    report(df, rng)


if __name__ == "__main__":
    main()
