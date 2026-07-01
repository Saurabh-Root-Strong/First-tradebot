"""
backtest_regime_edge.py — does CONDITIONING on a TREND regime rescue directional
(option-buy) entries? Tests the user's thesis: "in a trendy market, buying is good."

Logic: the UNDERLYING's trend-continuation is the CEILING for a trend option-buy. If the
index itself, gated to a strong trend, doesn't continue enough to beat even ~3bps FUTURES
cost with a >55% hit, then buying OPTIONS there is hopeless — options additionally bleed
theta and cross the ~3% premium round-trip wall, and need a BIGGER move to pay. So we
measure trend-gated forward continuation on 2yr 5min bars (a real sample), ATR-normalised.

At each 5min anchor: trend state from Kaufman ER(10) + ADX(14); direction = sign of the
10-bar drift (momentum). Enter WITH the trend, hold H minutes, exit at that bar's close
(same session). Report, per trend-strength tier × horizon:
    signed R  = direction · forward move / ATR      (continuation in ATR units)
    hit%      = share where the move continued (signed R > 0)
    net bps   = signed forward return in bps minus 3bps futures cost
A trend option-buy needs signed R comfortably positive AND hit >~55% AND the move to
clear the option breakeven (several × the futures cost). Day-block bootstrap CI.

    .venv\\Scripts\\python.exe backtest_regime_edge.py [--horizons 15,30,60]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

FY = {"NSE:NIFTY50-INDEX": "NIFTY50", "NSE:NIFTYBANK-INDEX": "NIFTYBANK",
      "NSE:FINNIFTY-INDEX": "FINNIFTY", "NSE:MIDCPNIFTY-INDEX": "MIDCPNIFTY"}
BAR_MIN = 5
# trend-strength tiers (ER threshold, ADX threshold) — loose → tight
TIERS = [("ER.4/ADX20", 0.40, 20), ("ER.5/ADX25", 0.50, 25), ("ER.6/ADX30", 0.60, 30)]
COST_BPS = 3.0            # futures round-trip (options need several × this)


def _er(c: np.ndarray, n: int = 10) -> np.ndarray:
    er = np.full(len(c), np.nan)
    ac = np.abs(np.diff(c))
    for i in range(n, len(c)):
        vol = ac[i - n:i].sum()
        er[i] = abs(c[i] - c[i - n]) / vol if vol > 0 else 0.0
    return er


def _atr(h, l, c, n: int = 14) -> np.ndarray:
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    atr = np.full(len(c), np.nan)
    if len(c) > n:
        atr[n] = tr[1:n + 1].mean()
        for i in range(n + 1, len(c)):
            atr[i] = (atr[i - 1] * (n - 1) + tr[i]) / n
    return atr


def _adx(h, l, c, n: int = 14) -> np.ndarray:
    up = np.diff(h, prepend=h[0]); dn = -np.diff(l, prepend=l[0])
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    def _wilder(x):
        out = np.full(len(x), np.nan);
        if len(x) <= n: return out
        out[n] = x[1:n + 1].sum()
        for i in range(n + 1, len(x)):
            out[i] = out[i - 1] - out[i - 1] / n + x[i]
        return out
    trn, pn, mn = _wilder(tr), _wilder(plus), _wilder(minus)
    with np.errstate(invalid="ignore", divide="ignore"):
        pdi = 100 * pn / trn; mdi = 100 * mn / trn
        dx = 100 * np.abs(pdi - mdi) / (pdi + mdi)
    adx = np.full(len(c), np.nan)
    idx = np.where(~np.isnan(dx))[0]
    if len(idx) > n:
        s = idx[0]; adx[s + n] = np.nanmean(dx[s:s + n])
        for i in range(s + n + 1, len(c)):
            if not np.isnan(dx[i]):
                adx[i] = (adx[i - 1] * (n - 1) + dx[i]) / n
    return adx


def _boot_ci(x, days, iters=1500, seed=7):
    if len(x) < 20: return (np.nan, np.nan)
    uniq = np.unique(days); rng = np.random.default_rng(seed)
    by = {d: x[days == d] for d in uniq}; m = np.empty(iters)
    for b in range(iters):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        m[b] = np.concatenate([by[d] for d in pick]).mean()
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def run(horizons):
    from core.constants import INDEX_SYMBOLS, LABELS
    print("=" * 98)
    print("  TREND-GATED MOMENTUM (option-buy CEILING) — enter WITH trend, hold H, 2yr 5min bars")
    print(f"  signed R = dir·fwd/ATR; hit = continued; net bps minus {COST_BPS}bps futures "
          "(options need several ×)")
    print("=" * 98)
    for sym in INDEX_SYMBOLS:
        try:
            df = pd.read_parquet(f"data/historical/5min/NSE_{FY[sym]}_INDEX_5min.parquet")
        except Exception as e:
            print(f"  {LABELS.get(sym, sym)}: no data ({e})"); continue
        df["ts"] = pd.to_datetime(df["ts"]); df["d"] = df["ts"].dt.date
        # per horizon × tier accumulators
        acc = {H: {t[0]: {"R": [], "bps": [], "day": []} for t in TIERS} for H in horizons}
        for di, (day, g) in enumerate(df.groupby("d")):
            g = g.sort_values("ts")
            c = g["close"].to_numpy(); h = g["high"].to_numpy(); l = g["low"].to_numpy()
            n = len(c)
            if n < 40: continue
            er = _er(c); adx = _adx(h, l, c); atr = _atr(h, l, c)
            drift = np.concatenate([np.full(10, np.nan), c[10:] - c[:-10]])
            for H in horizons:
                hb = max(1, H // BAR_MIN)
                for i in range(20, n - hb):
                    if np.isnan(er[i]) or np.isnan(adx[i]) or np.isnan(atr[i]) or atr[i] <= 0:
                        continue
                    if np.isnan(drift[i]) or drift[i] == 0:
                        continue
                    d_ = 1.0 if drift[i] > 0 else -1.0
                    fwd = c[i + hb] - c[i]
                    sR = d_ * fwd / atr[i]
                    sbps = d_ * (fwd / c[i]) * 1e4
                    for name, erth, adth in TIERS:
                        if er[i] >= erth and adx[i] >= adth:
                            a = acc[H][name]
                            a["R"].append(sR); a["bps"].append(sbps - COST_BPS); a["day"].append(di)
        print(f"\n  {LABELS.get(sym, sym)}")
        print(f"    {'H':>4} {'tier':>11} {'n':>6}  {'signed R':>9} {'hit%':>5}  "
              f"{'net bps':>8}  {'95% CI bps':>16}")
        for H in horizons:
            for name, _, _ in TIERS:
                a = acc[H][name]
                if len(a["R"]) < 20:
                    print(f"    {H:>4} {name:>11} {len(a['R']):>6}   (thin)"); continue
                R = np.array(a["R"]); bps = np.array(a["bps"]); dd = np.array(a["day"])
                ci = _boot_ci(bps, dd)
                print(f"    {H:>4} {name:>11} {len(R):>6}  {R.mean():>+8.3f} {100*(R>0).mean():>4.0f}%"
                      f"  {bps.mean():>+7.1f}  [{ci[0]:>+5.1f},{ci[1]:>+5.1f}]")
    print("\n  READ: signed R = continuation in ATR units. For a trend OPTION-buy to pay, need")
    print("  signed R clearly >0, hit >~55%, AND move to clear the ~3% option wall (several ×")
    print("  the 3bps futures line). If net bps CI straddles/≤0 even here, trend-buying is dead.")


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", default="15,30,60")
    a = ap.parse_args()
    run([int(x) for x in a.horizons.split(",")])
