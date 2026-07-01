"""
backtest_band_fade.py — does trading the RANGE-band EDGE (buy lower band / sell upper
band, mean-reversion fade) actually make money intraday, and which HORIZON is best?

The user's thesis: the 60m band is the trusted product, so fade its edges — long when
price hits the lower band, short at the upper, bet on reversion to mid. This tests it
on 2yr of 5min bars (a real sample, unlike the 3 captured tick days), across horizons,
net of FUTURES round-trip cost (the band edge is a directional index bet → futures ~3bps,
NOT the 3% option wall).

Band at anchor t0 (mirrors the deployed model): half-width% = K · σ_bar · (H/5)^0.40,
σ_bar = std of trailing L 5min returns, (H/5)^0.40 = the mean-reverting horizon scaling
(Hurst 0.40) fixed in intraday_scout. lower/upper = spot0·(1∓bw).

Fade rule: within the next H minutes (same session), the FIRST band touched is faded —
lower touch → LONG at lower, upper touch → SHORT at upper. Two exits:
  • TIME    — hold to t0+H, exit at that bar's close (pure "is a touch a reversion signal").
  • MANAGED — target = mid (spot0); stop = one more band-width beyond the entry; whichever
              hits first, else time-exit. (the trade you'd actually manage.)

Also reports the CHOP-gated subset (trailing Kaufman ER < 0.30) — fade theory says the
edge lives in range/chop and dies on trend days (where bands just break).

Honest stats: anchors sampled every 15m to cut overlap; day-block bootstrap CI on the
per-fade net bps (fades within a day are correlated — pooled iid CI would lie).

    .venv\\Scripts\\python.exe backtest_band_fade.py [--horizons 15,30,45,60,90,120] [--k 0.55]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

FY = {"NSE:NIFTY50-INDEX": "NIFTY50", "NSE:NIFTYBANK-INDEX": "NIFTYBANK",
      "NSE:FINNIFTY-INDEX": "FINNIFTY", "NSE:MIDCPNIFTY-INDEX": "MIDCPNIFTY"}
BAR_MIN = 5
L_VOL = 24            # trailing bars for σ (2h)
HURST = 0.40         # deployed horizon-scaling exponent
COST_BPS = 3.0       # futures round-trip
STEP = 3             # sample an anchor every 3 bars (15m) to cut overlap
ER_CHOP = 0.30       # trailing efficiency-ratio gate for "choppy"


def _er(close: np.ndarray, n: int = 10) -> np.ndarray:
    """Kaufman efficiency ratio over trailing n (causal, NaN until warm)."""
    er = np.full(len(close), np.nan)
    for i in range(n, len(close)):
        seg = close[i - n:i + 1]
        vol = np.abs(np.diff(seg)).sum()
        er[i] = abs(seg[-1] - seg[0]) / vol if vol > 0 else 0.0
    return er


def _boot_ci(x: np.ndarray, days: np.ndarray, iters: int = 2000, seed: int = 7):
    """Day-block bootstrap 95% CI of the mean (resample whole days)."""
    if len(x) < 20:
        return (np.nan, np.nan)
    uniq = np.unique(days)
    rng = np.random.default_rng(seed)
    means = np.empty(iters)
    by = {d: x[days == d] for d in uniq}
    for b in range(iters):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        means[b] = np.concatenate([by[d] for d in pick]).mean()
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def _fade_day(o, h, l, c, dates, day_id, K, H):
    """All fades for one session at horizon H. Yields (net_time, net_mgd, er, day_id)."""
    hb = max(1, H // BAR_MIN)
    ret = np.diff(c, prepend=c[0]) / np.where(c > 0, c, np.nan)
    er = _er(c)
    out = []
    n = len(c)
    for i in range(L_VOL, n - 1, STEP):
        sig = np.nanstd(ret[i - L_VOL:i]) * 100.0            # σ per-bar %
        if not (sig > 0):
            continue
        bw = K * sig * (hb ** HURST) / 100.0                 # band half-width (fraction)
        spot0 = c[i]
        lo, up = spot0 * (1 - bw), spot0 * (1 + bw)
        j_end = min(i + hb, n - 1)
        if j_end <= i:
            continue
        # first touch within (i, i+hb]
        side = None; entry = None; j_touch = None
        for j in range(i + 1, j_end + 1):
            if l[j] <= lo:
                side, entry, j_touch = "L", lo, j; break
            if h[j] >= up:
                side, entry, j_touch = "S", up, j; break
        if side is None:
            continue
        # TIME exit — close of horizon-end bar
        px_time = c[j_end]
        net_time = ((px_time / entry - 1) if side == "L" else (entry / px_time - 1)) * 1e4 - COST_BPS
        # MANAGED exit — target=mid, stop=one more bandwidth beyond entry
        mid = spot0
        stop = entry * (1 - bw) if side == "L" else entry * (1 + bw)
        px_mgd = c[j_end]
        for j in range(j_touch, j_end + 1):
            if side == "L":
                if l[j] <= stop: px_mgd = stop; break
                if h[j] >= mid:  px_mgd = mid;  break
            else:
                if h[j] >= stop: px_mgd = stop; break
                if l[j] <= mid:  px_mgd = mid;  break
        net_mgd = ((px_mgd / entry - 1) if side == "L" else (entry / px_mgd - 1)) * 1e4 - COST_BPS
        out.append((net_time, net_mgd, er[i], day_id))
    return out


def run(horizons, K):
    from core.constants import INDEX_SYMBOLS, LABELS
    print("=" * 96)
    print(f"  BAND-EDGE FADE — buy lower band / sell upper band, 2yr 5min bars, "
          f"net {COST_BPS}bps futures cost")
    print(f"  band = {K}·σ_bar·(H/5)^{HURST}; anchor every {STEP*BAR_MIN}m; day-block CI")
    print("=" * 96)
    for sym in INDEX_SYMBOLS:
        try:
            df = pd.read_parquet(f"data/historical/5min/NSE_{FY[sym]}_INDEX_5min.parquet")
        except Exception as e:
            print(f"  {LABELS.get(sym, sym)}: no data ({e})"); continue
        df["ts"] = pd.to_datetime(df["ts"]); df["d"] = df["ts"].dt.date
        rows = {H: [] for H in horizons}
        for di, (day, g) in enumerate(df.groupby("d")):
            g = g.sort_values("ts")
            if len(g) < L_VOL + 3:
                continue
            o, h, l, c = (g["open"].to_numpy(), g["high"].to_numpy(),
                          g["low"].to_numpy(), g["close"].to_numpy())
            for H in horizons:
                rows[H].extend(_fade_day(o, h, l, c, day, di, K, H))
        print(f"\n  {LABELS.get(sym, sym)}")
        print(f"    {'H':>4}  {'nFade':>6}  {'TIME net':>9} {'win%':>5} {'95% CI':>16}   "
              f"{'MGD net':>8} {'win%':>5}   {'CHOP(ER<.3) MGD':>16} {'n':>5}")
        for H in horizons:
            a = np.array(rows[H], dtype=float)
            if len(a) < 20:
                print(f"    {H:>4}  {len(a):>6}   (thin)"); continue
            nt, nm, er, days = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
            ci = _boot_ci(nt, days)
            chop = nm[er < ER_CHOP]
            cstr = f"{chop.mean():>+7.1f}bps" if len(chop) >= 20 else "   thin"
            print(f"    {H:>4}  {len(a):>6}  {nt.mean():>+7.1f}bp {100*(nt>0).mean():>4.0f}%"
                  f"  [{ci[0]:>+5.1f},{ci[1]:>+5.1f}]   {nm.mean():>+6.1f}bp {100*(nm>0).mean():>4.0f}%"
                  f"   {cstr:>16} {len(chop):>5}")
    print("\n  READ: TIME = hold to t0+H. MGD = target mid / stop 1 band beyond. Net of 3bps.")
    print("  +EV needs mean net > 0 with CI clearing 0. CHOP col = fade only when trailing")
    print("  ER<0.30 (range). If every H is <=0 net, band-edge fading has NO intraday edge.")


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", default="15,30,45,60,90,120")
    ap.add_argument("--k", type=float, default=0.55)
    a = ap.parse_args()
    run([int(x) for x in a.horizons.split(",")], a.k)
