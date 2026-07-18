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
COST_BPS = 3.0       # futures round-trip (flat; the HARD grade uses per-index below)
STEP = 3             # sample an anchor every 3 bars (15m) to cut overlap
ER_CHOP = 0.30       # trailing efficiency-ratio gate for "choppy"

# ── HARDENING (the reality haircut) — ESTIMATES, tune to your broker's fills ───────
# Per-index futures ROUND-TRIP cost bps: spread + STT(sell) + exchange/GST/stamp + impact.
# Liquid NIFTY/BANK tight; FIN/MIDCAP thinner futures → wider. Keyed by short name.
COST_IDX_BPS = {"NIFTY50": 3.0, "NIFTYBANK": 4.0, "FINNIFTY": 6.0, "MIDCPNIFTY": 7.0}
# Adverse STOP slippage bps: a stop is a market order into a move that just broke a band —
# it slips. Target(=mid) is a resting LIMIT → clean. Entry at the band edge = resting LIMIT
# → clean (but adverse-selection optimism remains; see note in _hard_oos).
SLIP_IDX_BPS = {"NIFTY50": 1.0, "NIFTYBANK": 2.0, "FINNIFTY": 3.0, "MIDCPNIFTY": 4.0}
ER_TREND = 0.45      # ER at/above this + trend agreeing WITH the move = BIG trend → veto fade


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


def _fade_day(o, h, l, c, dates, day_id, K, H, cost_bps=COST_BPS, slip_bps=0.0, hard=False):
    """All fades for one session at horizon H. Yields
    (net_time, net_mgd, er, day_id, reject) — reject=1 if the TOUCH bar closed with a
    rejection wick against the fade (upper touch closes in the LOWER half of its range,
    lower touch in the UPPER half); the candle disambiguates revert-vs-break at the edge.

    hard=True applies the reality haircut: cost_bps (per-index round-trip), slip_bps
    (adverse fill on STOP exits only), and a REGIME GATE that vetoes fading INTO a big
    trend (ER>=ER_TREND with the trend agreeing with the move — a knife, not a fade)."""
    hb = max(1, H // BAR_MIN)
    ret = np.diff(c, prepend=c[0]) / np.where(c > 0, c, np.nan)
    er = _er(c)
    slip = slip_bps / 1e4
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
        # REGIME GATE: don't fade INTO a big trend (ER high + trend agrees with the move)
        if hard and i >= 10 and er[i] >= ER_TREND:
            tsign = c[i] - c[i - 10]
            if (side == "S" and tsign > 0) or (side == "L" and tsign < 0):
                continue                                     # fading a strong trend = knife
        # rejection wick on the touch bar: close-location-in-range of bar j_touch
        rng_t = h[j_touch] - l[j_touch]
        clr = ((c[j_touch] - l[j_touch]) / rng_t) if rng_t > 0 else 0.5
        reject = 1.0 if ((side == "S" and clr < 0.5) or (side == "L" and clr > 0.5)) else 0.0
        # HONEST-FILL realism check: the rejection is only KNOWN at the touch bar's CLOSE, so a
        # rejection-selected trade cannot ALSO fill at the band edge (a resting limit fills on
        # EVERY touch, not selectively). The realizable fill is the touch-bar CLOSE (price has
        # already reverted inside). Isolates whether +REJECT was an unrealizable edge-fill +
        # hindsight-filter artifact. stop below recomputes off this entry.
        if _ENTRY_CLOSE:
            entry = c[j_touch]
        # TIME exit — close of horizon-end bar
        px_time = c[j_end]
        net_time = ((px_time / entry - 1) if side == "L" else (entry / px_time - 1)) * 1e4 - cost_bps
        # MANAGED exit — target=mid (resting LIMIT, clean); stop=one bandwidth beyond entry
        # (a STOP into a fresh break → adverse SLIPPAGE applied to that fill only).
        mid = spot0
        stop = entry * (1 - bw) if side == "L" else entry * (1 + bw)
        px_mgd = c[j_end]; hit_stop = False
        for j in range(j_touch, j_end + 1):
            if side == "L":
                if l[j] <= stop: px_mgd = stop; hit_stop = True; break
                if h[j] >= mid:  px_mgd = mid;  break
            else:
                if h[j] >= stop: px_mgd = stop; hit_stop = True; break
                if l[j] <= mid:  px_mgd = mid;  break
        if hit_stop and slip:                                # worsen the stop fill
            px_mgd = px_mgd * (1 - slip) if side == "L" else px_mgd * (1 + slip)
        net_mgd = ((px_mgd / entry - 1) if side == "L" else (entry / px_mgd - 1)) * 1e4 - cost_bps
        out.append((net_time, net_mgd, er[i], day_id, reject))
    return out


def _resample5(df5: pd.DataFrame, k: int) -> pd.DataFrame:
    """Aggregate 5min bars into k-chunks WITHIN each session (integer grouping, so bars
    align to the 09:15 open — pandas clock-resample would misbin the open into a 09:10 bin).
    OHLC = first-open / max-high / min-low / last-close; volume summed."""
    if k <= 1:
        return df5
    out = []
    for day, g in df5.groupby("d"):
        g = g.sort_values("ts").reset_index(drop=True)
        grp = g.index // k
        a = g.groupby(grp).agg(ts=("ts", "first"), open=("open", "first"),
                               high=("high", "max"), low=("low", "min"),
                               close=("close", "last"), volume=("volume", "sum"))
        a["d"] = day
        out.append(a)
    return pd.concat(out, ignore_index=True)


def _fade_recs(o, h, l, c, day_id, K, hb, l_vol):
    """Rejection-fade managed-net records for one session at hold=hb bars of THIS tf.
    Returns list of (net_mgd, day_id, reject). Same mechanics as _fade_day, hb given
    directly (tf-agnostic), σ window = l_vol bars of this tf."""
    ret = np.diff(c, prepend=c[0]) / np.where(c > 0, c, np.nan)
    out = []
    n = len(c)
    for i in range(l_vol, n - 1, max(1, hb)):        # step ~one hold to cut overlap
        sig = np.nanstd(ret[i - l_vol:i]) * 100.0
        if not (sig > 0):
            continue
        bw = K * sig * (hb ** HURST) / 100.0
        spot0 = c[i]
        lo, up = spot0 * (1 - bw), spot0 * (1 + bw)
        j_end = min(i + hb, n - 1)
        if j_end <= i:
            continue
        side = entry = j_touch = None
        for j in range(i + 1, j_end + 1):
            if l[j] <= lo:
                side, entry, j_touch = "L", lo, j; break
            if h[j] >= up:
                side, entry, j_touch = "S", up, j; break
        if side is None:
            continue
        rng_t = h[j_touch] - l[j_touch]
        clr = ((c[j_touch] - l[j_touch]) / rng_t) if rng_t > 0 else 0.5
        reject = 1.0 if ((side == "S" and clr < 0.5) or (side == "L" and clr > 0.5)) else 0.0
        mid = spot0
        stop = entry * (1 - bw) if side == "L" else entry * (1 + bw)
        px = c[j_end]
        for j in range(j_touch, j_end + 1):
            if side == "L":
                if l[j] <= stop: px = stop; break
                if h[j] >= mid:  px = mid;  break
            else:
                if h[j] >= stop: px = stop; break
                if l[j] <= mid:  px = mid;  break
        net = ((px / entry - 1) if side == "L" else (entry / px - 1)) * 1e4 - COST_BPS
        out.append((net, day_id, reject))
    return out


def sweep(tfs, K, hb):
    """Rejection-fade OOS (train 1st-half / test 2nd-half) across candle TFs — does a
    COARSER rejection candle filter better? Everything resampled from the 5min atom."""
    from core.constants import INDEX_SYMBOLS, LABELS
    print("=" * 100)
    print(f"  BAND-FADE +REJECT across candle TFs — hold={hb} bars, band {K}sigma, "
          f"net {COST_BPS}bps futures, OOS temporal split")
    print("=" * 100)
    print(f"  {'index':<11} " + "".join(f"{str(tf)+'m(test)':>15}" for tf in tfs))
    print(f"  {'(wallclock hold)':<11} " + "".join(f"{str(hb*tf)+'min':>15}" for tf in tfs))
    for sym in INDEX_SYMBOLS:
        try:
            df5 = pd.read_parquet(f"data/historical/5min/NSE_{FY[sym]}_INDEX_5min.parquet")
        except Exception:
            print(f"  {LABELS.get(sym, sym):<11} no data"); continue
        df5["ts"] = pd.to_datetime(df5["ts"]); df5["d"] = df5["ts"].dt.date
        cells = []
        for tf in tfs:
            k = tf // BAR_MIN
            dd = _resample5(df5, k)
            l_vol = max(8, 120 // tf)                 # ~2h-ish sigma window, floored
            recs = []
            for di, (day, g) in enumerate(dd.groupby("d")):
                g = g.sort_values("ts")
                if len(g) < l_vol + 3:
                    continue
                o, h, l, c = (g["open"].to_numpy(), g["high"].to_numpy(),
                              g["low"].to_numpy(), g["close"].to_numpy())
                for r in _fade_recs(o, h, l, c, day, K, hb, l_vol):
                    if r[2] == 1.0:                    # rejection fades only
                        recs.append((r[0], r[1]))
            if len(recs) < 40:
                cells.append("      thin"); continue
            a = np.array([x[0] for x in recs]); dts = np.array([x[1] for x in recs])
            days_sorted = np.array(sorted(set(dts)))
            cut = days_sorted[len(days_sorted) // 2]
            te = dts >= cut
            nte = a[te]; dte = dts[te]
            if len(nte) < 20:
                cells.append("      thin"); continue
            ci = _boot_ci(nte, dte)
            star = "*" if ci[0] > 0 else " "
            cells.append(f"{nte.mean():>+5.1f}[{ci[0]:>+4.1f},{ci[1]:>+4.1f}]{star}")
        print(f"  {LABELS.get(sym, sym):<11} " + "".join(f"{x:>15}" for x in cells))
    print("\n  READ: cell = OOS TEST (held-out 2nd-half) mean net bps [95% CI]; * = CI clears 0.")
    print("  Coarser TF = more significant rejection but fewer/slower fades. Liquid NIFTY/BANK")
    print("  trust 3bps; FIN/MIDCAP need a cost haircut. Best TF = highest net with * that is")
    print("  also tradeable on a LIQUID future.")


def hard_oos(K, H):
    """The REALITY grade: rejection-fade OOS (test = held-out 2nd half) at horizon H,
    NAIVE (flat 3bps, clean stops, no gate) vs HARD (per-index cost + stop slippage +
    regime gate). If the edge survives the haircut on a LIQUID index, it is deployable."""
    from core.constants import INDEX_SYMBOLS, LABELS
    print("=" * 100)
    print(f"  BAND-FADE +REJECT — REALITY HAIRCUT (H={H}, hold {H}min), OOS test-half net bps")
    print("=" * 100)
    print(f"    {'index':<11} {'cost/slip':>10}   {'NAIVE test':>20}   {'HARD test':>20}   {'nHard':>6}  survives?")
    for sym in INDEX_SYMBOLS:
        short = FY[sym]
        try:
            df = pd.read_parquet(f"data/historical/5min/NSE_{short}_INDEX_5min.parquet")
        except Exception:
            print(f"    {LABELS.get(sym, sym):<11} no data"); continue
        df["ts"] = pd.to_datetime(df["ts"]); df["d"] = df["ts"].dt.date
        cost = COST_IDX_BPS.get(short, 5.0); slip = SLIP_IDX_BPS.get(short, 3.0)

        def _collect(hard):
            recs = []
            cb = cost if hard else COST_BPS
            sl = slip if hard else 0.0
            for di, (day, g) in enumerate(df.groupby("d")):
                g = g.sort_values("ts")
                if len(g) < L_VOL + 3:
                    continue
                o, h, l, c = (g["open"].to_numpy(), g["high"].to_numpy(),
                              g["low"].to_numpy(), g["close"].to_numpy())
                for r in _fade_day(o, h, l, c, day, di, K, H, cost_bps=cb, slip_bps=sl, hard=hard):
                    if r[4] == 1.0:                          # rejection fades only
                        recs.append((r[1], day))            # net_mgd, date
            return recs

        def _oos(recs):
            if len(recs) < 40:
                return None
            a = np.array([x[0] for x in recs]); dts = np.array([x[1] for x in recs])
            cut = sorted(set(dts))[len(set(dts)) // 2]
            te = dts >= cut
            if te.sum() < 20:
                return None
            ci = _boot_ci(a[te], dts[te])
            return a[te].mean(), ci, int(te.sum())

        rn, rh = _oos(_collect(False)), _oos(_collect(True))
        if rn is None or rh is None:
            print(f"    {LABELS.get(sym, sym):<11} thin"); continue
        nstr = f"{rn[0]:>+5.1f}[{rn[1][0]:>+4.1f},{rn[1][1]:>+4.1f}]"
        hstr = f"{rh[0]:>+5.1f}[{rh[1][0]:>+4.1f},{rh[1][1]:>+4.1f}]"
        surv = "YES" if rh[1][0] > 0 else "no"
        print(f"    {LABELS.get(sym, sym):<11} {cost:>4.0f}/{slip:<4.0f}   {nstr:>20}   "
              f"{hstr:>20}   {rh[2]:>6}  {surv}")
    print("\n  READ: HARD applies per-index futures cost + adverse stop-slippage + a regime")
    print("  gate (no fading INTO a big trend). 'survives' = HARD test-CI still clears 0.")
    print("  Costs/slips are ESTIMATES (top of file) — the liquid NIFTY/BANK cells are the")
    print("  trustworthy verdict; FIN/MIDCAP carry the largest cost uncertainty.")
    print("  REMAINING OPTIMISM (not modelled): entry-limit ADVERSE SELECTION — you miss the")
    print("  fills on the touches that keep running (the good fades), keep the ones that come")
    print("  back to you. Real edge <= this. Only a live paper run measures it.")


def _liqsweep_day(o, h, l, c, day_id, N, hb, cost_bps, slip_bps, hard, band_gate=False, K=0.55):
    """LIQUIDITY-SWEEP fade for one session — the authentic price-action 'stop hunt':
    a bar pokes BEYOND the prior N-bar swing extreme but CLOSES back inside (failed
    breakout = stops grabbed) → fade the reversal. Short a swept high, long a swept low.
    target = swing-range MID (reversion magnet); stop = the sweep extreme + slip; hold hb
    bars else time-exit. Regime gate (hard): skip a sweep-short in a strong uptrend / a
    sweep-long in a strong downtrend. Yields (net, day_id)."""
    er = _er(c)
    ret = np.diff(c, prepend=c[0]) / np.where(c > 0, c, np.nan)
    slip = slip_bps / 1e4
    lvol = max(8, L_VOL)
    out = []
    n = len(c)
    for i in range(max(N, lvol), n - 1, STEP):
        swing_hi = h[i - N:i].max()
        swing_lo = l[i - N:i].min()
        mid = 0.5 * (swing_hi + swing_lo)
        side = entry = stop = None
        if h[i] > swing_hi and c[i] < swing_hi:              # upper sweep → fade SHORT
            side, entry, stop = "S", c[i], h[i]
        elif l[i] < swing_lo and c[i] > swing_lo:            # lower sweep → fade LONG
            side, entry, stop = "L", c[i], l[i]
        if side is None:
            continue
        # LOCATION FILTER: the sweep extreme must sit >= a band-width from the recent mean
        # (a STATISTICALLY significant level — the discretionary trader's "key level"), not
        # a micro-swing in normal chop. This is the confluence the naive sweep lacks.
        if band_gate:
            mref = c[i - lvol:i].mean()
            sig = np.nanstd(ret[i - lvol:i]) * 100.0
            band = K * sig * (hb ** HURST) / 100.0 if sig > 0 else np.inf
            ext = h[i] if side == "S" else l[i]
            if not (mref and band < np.inf and abs(ext / mref - 1.0) >= band):
                continue
        if hard and i >= 10 and er[i] >= ER_TREND:           # don't fade INTO a strong trend
            tsign = c[i] - c[i - 10]
            if (side == "S" and tsign > 0) or (side == "L" and tsign < 0):
                continue
        # target must be on the reversion side of entry, else skip (no edge to capture)
        if (side == "S" and mid >= entry) or (side == "L" and mid <= entry):
            continue
        j_end = min(i + hb, n - 1)
        px = c[j_end]; hit_stop = False
        for j in range(i + 1, j_end + 1):
            if side == "S":
                if h[j] >= stop: px = stop; hit_stop = True; break
                if l[j] <= mid:  px = mid;  break
            else:
                if l[j] <= stop: px = stop; hit_stop = True; break
                if h[j] >= mid:  px = mid;  break
        if hit_stop and slip:
            px = px * (1 + slip) if side == "S" else px * (1 - slip)
        net = ((entry / px - 1) if side == "S" else (px / entry - 1)) * 1e4 - cost_bps
        out.append((net, day_id))
    return out


_LIQ_BAND_GATE = False    # set by --liqband: gate the sweep to band-extreme locations
_ENTRY_CLOSE = False      # set by --honestfill: enter at touch-bar CLOSE, not the band edge


def liqsweep_grade(N, hb):
    """Grade the price-action LIQUIDITY-SWEEP fade (the user's 'stop hunt' pattern) HARD,
    per-index OOS test-half. This tests the AUTHENTIC PA pattern on 2yr, vs the band-fade."""
    from core.constants import INDEX_SYMBOLS, LABELS
    print("=" * 100)
    print(f"  LIQUIDITY-SWEEP FADE — poke prior {N}-bar swing, close back inside, fade the")
    print(f"  reversal (HARD costs+slip+gate). hold={hb} bars(5m). OOS test-half net bps")
    print("=" * 100)
    print(f"    {'index':<11} {'cost/slip':>10}   {'TRAIN test':>22}   {'OOS TEST':>22}   {'nTe':>5}  edge?")
    for sym in INDEX_SYMBOLS:
        short = FY[sym]
        try:
            df = pd.read_parquet(f"data/historical/5min/NSE_{short}_INDEX_5min.parquet")
        except Exception:
            print(f"    {LABELS.get(sym, sym):<11} no data"); continue
        df["ts"] = pd.to_datetime(df["ts"]); df["d"] = df["ts"].dt.date
        cost = COST_IDX_BPS.get(short, 5.0); slip = SLIP_IDX_BPS.get(short, 3.0)
        recs = []
        for di, (day, g) in enumerate(df.groupby("d")):
            g = g.sort_values("ts")
            if len(g) < N + 3:
                continue
            o, h, l, c = (g["open"].to_numpy(), g["high"].to_numpy(),
                          g["low"].to_numpy(), g["close"].to_numpy())
            recs.extend(_liqsweep_day(o, h, l, c, day, N, hb, cost, slip, True,
                                      band_gate=_LIQ_BAND_GATE))
        if len(recs) < 40:
            print(f"    {LABELS.get(sym, sym):<11} thin ({len(recs)})"); continue
        a = np.array([x[0] for x in recs]); dts = np.array([x[1] for x in recs])
        cut = sorted(set(dts))[len(set(dts)) // 2]
        tr, te = dts < cut, dts >= cut
        citr = _boot_ci(a[tr], dts[tr]); cite = _boot_ci(a[te], dts[te])
        edge = "YES" if (te.sum() >= 20 and cite[0] > 0) else "no"
        print(f"    {LABELS.get(sym, sym):<11} {cost:>4.0f}/{slip:<4.0f}   "
              f"{a[tr].mean():>+5.1f}[{citr[0]:>+4.1f},{citr[1]:>+4.1f}]      "
              f"{a[te].mean():>+5.1f}[{cite[0]:>+4.1f},{cite[1]:>+4.1f}]   {int(te.sum()):>5}  {edge}")
    print("\n  READ: the authentic PA stop-hunt fade, hardened. 'edge'=OOS test-CI clears 0.")
    print("  Compare to the band-fade (BANK ~+0.9bps HARD). OI/COI/premium confirmation is a")
    print("  FORWARD live-capture overlay — not in the 2yr history. This is the candle half.")


def expiry_grade(K, H):
    """APPLY the expiry data (the one options-derived feature computable on the full 2yr —
    OI/COI/premium exist only in the 34-day live capture, too thin to grade). Gamma-pinning
    theory: dealer hedging pins the index near expiry → mean-reversion (the fade) should be
    STRONGER in expiry week. Splits the HARD rejection-fade by monthly DTE: NEAR (<=4 trading-
    ish days) vs FAR. DTE weekday-regime drifted 2024-26, but the <=4 bucket is robust to a
    +-2d error (last-Tue and last-Thu both land in the final week)."""
    from core.constants import INDEX_SYMBOLS, LABELS
    from core.market_calendar import days_to_expiry
    print("=" * 100)
    print(f"  BAND-FADE +REJECT — EXPIRY-WEEK PINNING (HARD costs, H={H}), OOS test-half net bps")
    print("=" * 100)
    print(f"    {'index':<11}   {'NEAR-expiry (DTE<=4)':>24}   {'FAR (DTE>4)':>22}   pinning?")
    _dte_cache: dict = {}
    for sym in INDEX_SYMBOLS:
        short = FY[sym]
        try:
            df = pd.read_parquet(f"data/historical/5min/NSE_{short}_INDEX_5min.parquet")
        except Exception:
            print(f"    {LABELS.get(sym, sym):<11} no data"); continue
        df["ts"] = pd.to_datetime(df["ts"]); df["d"] = df["ts"].dt.date
        cost = COST_IDX_BPS.get(short, 5.0); slip = SLIP_IDX_BPS.get(short, 3.0)
        near, far = [], []
        for di, (day, g) in enumerate(df.groupby("d")):
            g = g.sort_values("ts")
            if len(g) < L_VOL + 3:
                continue
            if day not in _dte_cache:
                _dte_cache[day] = days_to_expiry(day, weekly=False)
            dte = _dte_cache[day]
            bucket = near if (0 <= dte <= 4) else far
            o, h, l, c = (g["open"].to_numpy(), g["high"].to_numpy(),
                          g["low"].to_numpy(), g["close"].to_numpy())
            for r in _fade_day(o, h, l, c, day, di, K, H, cost_bps=cost, slip_bps=slip, hard=True):
                if r[4] == 1.0:
                    bucket.append((r[1], day))

        def _oos(recs):
            if len(recs) < 30:
                return None
            a = np.array([x[0] for x in recs]); dts = np.array([x[1] for x in recs])
            cut = sorted(set(dts))[len(set(dts)) // 2]
            te = dts >= cut
            if te.sum() < 15:
                return None
            return a[te].mean(), _boot_ci(a[te], dts[te]), int(te.sum())

        rn, rf = _oos(near), _oos(far)
        def _fmt(r):
            return f"{r[0]:>+5.1f}[{r[1][0]:>+4.1f},{r[1][1]:>+4.1f}] n{r[2]}" if r else "thin"
        pin = "YES" if (rn and rf and rn[0] > rf[0] and rn[1][0] > 0) else "no"
        print(f"    {LABELS.get(sym, sym):<11}   {_fmt(rn):>24}   {_fmt(rf):>22}   {pin}")
    print("\n  READ: pinning=YES if NEAR-expiry net > FAR net AND near test-CI clears 0. If the")
    print("  edge concentrates near expiry, gate the live fade to expiry week (fewer, bigger).")
    print("  OI/COI/premium/volume fusion is NOT here — only 34 live days exist; that is a")
    print("  forward LIVE-CAPTURE grade (log the option context at each paper fade), not a")
    print("  backtest. Expiry is the only options feature the 2yr history can honestly carry.")


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
        print(f"    {'H':>4}  {'nFade':>6}  {'MGD net':>8} {'win%':>5} {'MGD 95%CI':>16}   "
              f"{'+REJECT net':>11} {'win%':>5} {'REJECT 95%CI':>16} {'nRej':>5}")
        for H in horizons:
            a = np.array(rows[H], dtype=float)
            if len(a) < 20:
                print(f"    {H:>4}  {len(a):>6}   (thin)"); continue
            nt, nm, er, days, rej = a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4]
            ci = _boot_ci(nm, days)                         # CI on the MANAGED net (the real trade)
            rmask = rej == 1.0
            rn, rdays = nm[rmask], days[rmask]
            rci = _boot_ci(rn, rdays)
            rstr = (f"[{rci[0]:>+5.1f},{rci[1]:>+5.1f}]" if len(rn) >= 20 else "        thin")
            rmean = f"{rn.mean():>+7.1f}bp" if len(rn) else "     n/a"
            rwin = f"{100*(rn>0).mean():>4.0f}%" if len(rn) else "  n/a"
            print(f"    {H:>4}  {len(a):>6}  {nm.mean():>+6.1f}bp {100*(nm>0).mean():>4.0f}%"
                  f"  [{ci[0]:>+5.1f},{ci[1]:>+5.1f}]   {rmean:>11} {rwin:>5} {rstr:>16} {len(rn):>5}")
    print("\n  READ: MGD = target mid / stop 1 band beyond, net 3bps futures. +REJECT = same")
    print("  fade but ONLY when the touch bar closed with a rejection wick (candle filter).")
    print("  +EV needs mean net > 0 with the 95% CI clearing 0. If REJECT doesn't lift MGD")
    print("  into a CI clearing 0, band-edge fading is breakeven — a RISK MAP, not a trade.")

    # ── OOS TEMPORAL SPLIT — the filter was chosen after seeing the data, so the only
    # honest test is whether the +REJECT edge holds on a HELD-OUT later slice. Train =
    # first half of dates, test = second half; report REJECT net+CI on each, per index. ─
    print("\n" + "=" * 96)
    print("  OOS TEMPORAL SPLIT — +REJECT fade, H=45, train(1st half dates) vs test(2nd half)")
    print("=" * 96)
    print(f"    {'index':<11} {'TRAIN net':>10} {'train 95%CI':>16} {'nTr':>5}   "
          f"{'TEST net':>9} {'test 95%CI':>16} {'nTe':>5}   verdict")
    H = 45
    for sym in INDEX_SYMBOLS:
        try:
            df = pd.read_parquet(f"data/historical/5min/NSE_{FY[sym]}_INDEX_5min.parquet")
        except Exception:
            continue
        df["ts"] = pd.to_datetime(df["ts"]); df["d"] = df["ts"].dt.date
        alldays = sorted(df["d"].unique())
        cut = alldays[len(alldays) // 2]
        recs = []
        for di, (day, g) in enumerate(df.groupby("d")):
            g = g.sort_values("ts")
            if len(g) < L_VOL + 3:
                continue
            o, h, l, c = (g["open"].to_numpy(), g["high"].to_numpy(),
                          g["low"].to_numpy(), g["close"].to_numpy())
            for r in _fade_day(o, h, l, c, day, di, K, H):
                recs.append((r[1], r[3], r[4], day))       # net_mgd, day_id, reject, date
        a = np.array([(x[0], x[1], x[2]) for x in recs], dtype=float)
        dts = np.array([x[3] for x in recs])
        if len(a) < 40:
            print(f"    {LABELS.get(sym, sym):<11} thin"); continue
        rmask = a[:, 2] == 1.0
        tr = rmask & (dts < cut); te = rmask & (dts >= cut)
        ntr, nte = a[tr, 0], a[te, 0]
        dtr, dte = a[tr, 1], a[te, 1]
        cit, cie = _boot_ci(ntr, dtr), _boot_ci(nte, dte)
        ok = "HOLDS" if (len(nte) >= 20 and cie[0] > 0) else "FAILS OOS"
        print(f"    {LABELS.get(sym, sym):<11} {ntr.mean():>+8.1f}bp "
              f"[{cit[0]:>+5.1f},{cit[1]:>+5.1f}] {len(ntr):>5}   "
              f"{nte.mean():>+7.1f}bp [{cie[0]:>+5.1f},{cie[1]:>+5.1f}] {len(nte):>5}   {ok}")
    print("\n  READ: an edge is only real if the TEST (held-out later) CI still clears 0. A")
    print("  filter that clears in-sample but FAILS OOS was curve-fit. 3bps assumes LIQUID")
    print("  futures — NIFTY/BANK realistic; FIN/MIDCAP thinner, real cost higher (haircut).")


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", default="15,30,45,60,90,120")
    ap.add_argument("--k", type=float, default=0.55)
    ap.add_argument("--sweep", action="store_true", help="rejection-fade OOS across candle TFs")
    ap.add_argument("--tfs", default="5,10,15,30,60")
    ap.add_argument("--hb", type=int, default=3, help="hold in bars of each tf")
    ap.add_argument("--hard", action="store_true", help="reality haircut: per-index cost+slip+gate")
    ap.add_argument("--hardh", type=int, default=45, help="horizon (min) for the --hard grade")
    ap.add_argument("--honestfill", action="store_true",
                    help="enter at touch-bar CLOSE not the band edge (realizable-fill audit)")
    ap.add_argument("--expiry", action="store_true", help="split HARD fade by expiry-week (DTE)")
    ap.add_argument("--liqsweep", action="store_true", help="price-action liquidity-sweep fade (HARD)")
    ap.add_argument("--liqband", action="store_true", help="gate liqsweep to band-extreme locations")
    ap.add_argument("--swingn", type=int, default=10, help="prior-swing lookback bars")
    ap.add_argument("--sweephold", type=int, default=9, help="liqsweep hold in 5m bars")
    a = ap.parse_args()
    _ENTRY_CLOSE = a.honestfill
    if a.liqsweep:
        _LIQ_BAND_GATE = a.liqband
        liqsweep_grade(a.swingn, a.sweephold)
    elif a.expiry:
        expiry_grade(a.k, a.hardh)
    elif a.hard:
        hard_oos(a.k, a.hardh)
    elif a.sweep:
        sweep([int(x) for x in a.tfs.split(",")], a.k, a.hb)
    else:
        run([int(x) for x in a.horizons.split(",")], a.k)
