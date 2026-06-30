"""
backtest_mtf.py — does HIGHER-TIMEFRAME confirmation add edge to a fast trigger?

The hypothesis (trigger -> confirm -> context): a fast-TF scout trigger (5/10/15m)
should only be trusted when the HIGHER TF (15/30/60m) agrees — same trend, a real
breakout (not fading into the opposite consolidation). To give the higher TF real
structure (intraday 1h has only ~2 closed bars by midday) the context is built
from PRIOR captured sessions + today up to t, resampled to the higher TF. So S/R
and trend are measured on multi-session swing history, exactly the "yesterday /
day-back" data the desk would use.

Test: for every gated fast trigger, split CONFIRMED (fast dir agrees with higher-TF
trend) vs CONFLICT, and measure SIGNED forward spot return (ret * dir) + IC, with a
day-block bootstrap CI. If CONFIRMED >> CONFLICT with CI clearing 0, the MTF gate
is real and gets wired. If not, higher-TF agreement is just the same price
autocorrelation measured twice — no new edge — and we don't add the complexity.

Signed forward return (not option P&L) isolates SKILL from the cost wall: if there
is no skill lift, cost can only make it worse, so skill is the necessary test.

Lookahead-free: context bars use ts<=t (prior days + today partial); trigger uses
build_series(as_of=t); forward spot is the answer key only.

    .venv\\Scripts\\python.exe backtest_mtf.py
"""
from __future__ import annotations

import datetime
import glob
import os
import sys

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from core.constants import IST, INDEX_SYMBOLS, LIVE_DIR
from core.mirror_io import read_mirror as _read
from backtest_continuity import _spearman, _boot_ci
import footprint_chart as fc
import intraday_scout as scout

SEED = 7
GATE = 0.22
PRIOR_DAYS = 4
PAIRS = [(5, 30), (10, 30), (15, 60)]     # (fast trigger TF, higher confirm TF) minutes
ENTRY_TIMES = ["09:45", "10:15", "10:45", "11:15", "11:45", "12:15", "12:45", "13:15", "13:45"]
HORIZONS = (15, 30)
_SWING = 3                                 # bars each side for a higher-TF swing pivot
_NBACK = 4                                 # higher-TF trend lookback (bars)


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


def _bars(ticks, tf_min):
    """Resample tick ltp -> per-session OHLC bars (no overnight buckets)."""
    if ticks is None or len(ticks) == 0:
        return pd.DataFrame()
    g = ticks.set_index("ts")["ltp"]
    o = g.resample(f"{tf_min}min", origin="start_day").ohlc().dropna()
    return o


def _htf_state(bars):
    """Trend (+1/-1/0) + breakout dir on a higher-TF bar frame. ts<=t already."""
    if len(bars) < max(_NBACK + 1, 2 * _SWING + 2):
        return 0, 0
    c = bars["close"].to_numpy(float)
    # trend: close vs N bars ago, confirmed by EMA slope sign
    ema = pd.Series(c).ewm(span=5, adjust=False).mean().to_numpy()
    level = np.sign(c[-1] - c[-1 - _NBACK])
    slope = np.sign(ema[-1] - ema[-3])
    trend = int(level) if level == slope else 0
    # breakout: last close beyond the most recent swing hi/lo (excluding forming bar)
    hi = bars["high"].to_numpy(float); lo = bars["low"].to_numpy(float)
    sw_hi = np.nanmax(hi[-(2 * _SWING + 2):-1]); sw_lo = np.nanmin(lo[-(2 * _SWING + 2):-1])
    brk = 1 if c[-1] > sw_hi else (-1 if c[-1] < sw_lo else 0)
    return trend, brk


def harvest(days):
    rows = []
    for di, date in enumerate(days):
        if di < 1:
            continue
        priors = days[max(0, di - PRIOR_DAYS):di]
        d0 = datetime.date.fromisoformat(date)
        for sym in INDEX_SYMBOLS:
            today = _session_ticks(date, sym)
            if today is None:
                continue
            prior_ticks = [x for x in (_session_ticks(p, sym) for p in priors) if x is not None]
            prior_cat = pd.concat(prior_ticks) if prior_ticks else None
            for hhmm in ENTRY_TIMES:
                hh, mm = map(int, hhmm.split(":"))
                t = datetime.datetime.combine(d0, datetime.time(hh, mm), tzinfo=IST)
                upto = today[today["ts"] <= pd.Timestamp(t)]
                if len(upto) < 5:
                    continue
                spot = float(upto.iloc[-1]["ltp"])
                ctx = pd.concat([prior_cat, upto]) if prior_cat is not None else upto
                for (fast, htf) in PAIRS:
                    try:
                        ser = fc.build_series(sym, fast, date, t)
                        flow = scout._flow_signal(ser)[0]; div = scout._divergence_signal(ser)[0]
                        cross = scout._crossover_signal(ser)[0]
                        fut = scout._futures_signal(sym, fast, date, t)[0]
                    except Exception:
                        continue
                    strength = 0.40 * flow + 0.25 * div + 0.20 * fut + 0.15 * cross
                    if abs(strength) < GATE:
                        continue
                    direction = 1 if strength > 0 else -1
                    trend, brk = _htf_state(_bars(ctx, htf))
                    confirmed = (trend == direction) or (brk == direction)
                    conflict = (trend == -direction) or (brk == -direction)
                    fwd = {}
                    for H in HORIZONS:
                        sH = today[today["ts"] <= pd.Timestamp(t + datetime.timedelta(minutes=H))]
                        sH = float(sH.iloc[-1]["ltp"]) if len(sH) else None
                        fwd[H] = ((sH / spot - 1.0) * 100.0 * direction) if (sH and sH != spot) else np.nan
                    rows.append({"date": date, "sym": sym, "t": hhmm, "pair": f"{fast}->{htf}",
                                 "dir": direction, "strength": strength, "htf_trend": trend,
                                 "htf_brk": brk, "confirmed": confirmed, "conflict": conflict,
                                 **{f"sret{H}": fwd[H] for H in HORIZONS}})
    return pd.DataFrame(rows)


def _mean_ci(x, groups, rng, reps=1500):
    x = np.asarray(x, float); groups = np.asarray(groups)
    keep = ~np.isnan(x); x, groups = x[keep], groups[keep]
    if len(x) < 5:
        return np.nan, np.nan, np.nan, 0
    days = np.unique(groups); idx = {d: np.where(groups == d)[0] for d in days}
    ms = [x[np.concatenate([idx[d] for d in rng.choice(days, len(days), replace=True)])].mean()
          for _ in range(reps)]
    return float(x.mean()), float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5)), len(x)


def report(df, rng):
    print("\nMTF CONFIRMATION — does higher-TF agreement lift a fast trigger?")
    print("=" * 80)
    days = sorted(df.date.unique())
    print(f"  triggers={len(df)}  days={len(days)} ({days[0]}..{days[-1]})  "
          f"signed fwd return = ret*dir (>0 = trigger was right)")

    def grp(sub, label):
        print(f"\n  {label}  (n={len(sub)}, {100*sub.confirmed.mean():.0f}% confirmed)")
        for H in HORIZONS:
            for tag, s in (("ALL    ", sub),
                           ("CONFIRM", sub[sub.confirmed & ~sub.conflict]),
                           ("CONFLCT", sub[sub.conflict])):
                m, lo, hi, n = _mean_ci(s[f"sret{H}"], s.date.to_numpy(), rng)
                if n == 0:
                    print(f"     {H:>3}m {tag}  n/a"); continue
                win = 100 * (s[f"sret{H}"].dropna() > 0).mean()
                flag = "+" if lo > 0 else ("-" if hi < 0 else "0")
                print(f"     {H:>3}m {tag}  sret {m:+.3f}% [{lo:+.3f},{hi:+.3f}][{flag}]  win {win:3.0f}%  n={n}")

    for pair in df.pair.unique():
        grp(df[df.pair == pair], f"PAIR {pair}")

    print("\n  " + "-" * 60)
    print("  POOLED across all pairs (robustness — bigger n on the conflict veto)")
    for H in HORIZONS:
        for tag, s in (("ALL    ", df),
                       ("CONFIRM", df[df.confirmed & ~df.conflict]),
                       ("CONFLCT", df[df.conflict])):
            m, lo, hi, n = _mean_ci(s[f"sret{H}"], s.date.to_numpy(), rng)
            if n == 0:
                continue
            win = 100 * (s[f"sret{H}"].dropna() > 0).mean()
            flag = "+" if lo > 0 else ("-" if hi < 0 else "0")
            print(f"     {H:>3}m {tag}  sret {m:+.3f}% [{lo:+.3f},{hi:+.3f}][{flag}]  win {win:3.0f}%  n={n}")

    print("\n" + "=" * 80)
    print("READ: edge only if CONFIRM sret CI clears 0 AND CONFIRM > ALL > CONFLCT.")
    print("If CONFIRM ~ ALL ~ CONFLCT (all straddle 0), higher-TF agreement is the same")
    print("autocorrelation twice — no new edge — keep MTF as context display, don't gate.")


_CACHE = os.environ.get("MTF_CACHE", "")


def main():
    rng = np.random.default_rng(SEED)
    if _CACHE and os.path.exists(_CACHE):
        df = pd.read_parquet(_CACHE)
        print(f"loaded cache {_CACHE}  rows={len(df)}")
    else:
        days = _captured_days()
        if len(days) < 2:
            print("need >=2 captured days"); return
        print(f"captured days: {days}")
        df = harvest(days)
        if len(df) == 0:
            print("no triggers"); return
        if _CACHE:
            df.to_parquet(_CACHE); print(f"cached -> {_CACHE}")
    report(df, rng)


if __name__ == "__main__":
    main()
