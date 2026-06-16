"""
timeframe_delta.py — "What changed, per timeframe" multi-horizon change engine.

Answers the question directly: for each of 1 / 5 / 10 / 15 / 60-minute horizons,
what changed RIGHT NOW — and, the part that actually matters, how the horizons
AGREE or DIVERGE.

Data sources (both durable, see intraday_store / intraday_db):
  • Indices  — the lock-free 1-second tick ledger mirror
    (data/intraday/live/<date>_ticks.parquet) + OI snapshots for ΔOI/ΔPCR.
    NSE disseminates an index ~1/sec, so this is full-fidelity for an index.
  • Stocks   — the downloaded historical OHLCV (data/historical/5min/...), anchored
    at the last bar; horizons are limited to multiples of 5 min (no sub-minute),
    no OI. Useful for offline study, not a live read.

Every horizon is anchored to the SAME 'now', so the horizons are directly
comparable — the basis for the divergence/acceleration read.

Per-horizon metrics
  ret%      net close-to-close return over the trailing window
  range%    (max-min)/anchor — realized travel (volatility)
  path      |net move| / Σ|step moves|  ∈ [0,1]   (1 = clean trend, ~0 = chop)
  Δvol      traded volume in the window (cum_vol delta for ticks; bar-sum for stocks)
  ΔPCR/ΔOI  index only — change in put-call OI ratio and call/put OI over the window

Cross-horizon synthesis
  alignment    how many horizons share the sign  → ALIGNED vs MIXED
  divergence   short (1/5m) vs long (60m) sign clash → pullback/bounce call
  acceleration short-horizon per-min rate vs long-horizon → building / fading
  verdict      one-line regime read

  .venv\\Scripts\\python.exe timeframe_delta.py                 # all 4 indices, live
  .venv\\Scripts\\python.exe timeframe_delta.py --symbol NIFTY
  .venv\\Scripts\\python.exe timeframe_delta.py --hist RELIANCE # historical stock
"""
from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

from core.constants import IST, HIST_DIR, INDEX_SYMBOLS, LABELS, LABEL_TO_SYM   # noqa: E402
from core.mirror_io import read_mirror                                          # noqa: E402

HORIZONS = [1, 5, 10, 15, 60]          # minutes
_DEADBAND = 0.03                       # |ret%| below this counts as flat


# ── Loaders (thin wrappers over the shared mirror reader) ───────────────────────

def _load_index_ticks(sym: str, date: str | None = None,
                      as_of: datetime.datetime | None = None) -> pd.DataFrame | None:
    return read_mirror("ticks", date, as_of, sym)


def _load_index_oi(sym: str, date: str | None = None,
                   as_of: datetime.datetime | None = None) -> pd.DataFrame | None:
    return read_mirror("oi_snapshots", date, as_of, sym)


def _load_hist(label: str) -> tuple[str, pd.DataFrame] | None:
    safe = f"NSE_{label.upper()}_EQ"
    p = HIST_DIR / "5min" / f"{safe}_5min.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df["ts"] = pd.to_datetime(df["ts"])
    return label.upper(), df.sort_values("ts").reset_index(drop=True)


# ── Core per-horizon metric ─────────────────────────────────────────────────────

def _horizon(ts: np.ndarray, price: np.ndarray, vol_cum: np.ndarray | None,
             vol_bar: np.ndarray | None, T: int) -> dict | None:
    """Change over the trailing T-minute window ending at the latest sample."""
    now = ts[-1]
    anchor = now - np.timedelta64(T, "m")
    mask = ts >= anchor
    seg_p = price[mask]
    if seg_p.size < 2:
        return None
    p0, p1 = float(seg_p[0]), float(seg_p[-1])
    if p0 <= 0:
        return None
    ret = (p1 - p0) / p0 * 100
    rng = (float(seg_p.max()) - float(seg_p.min())) / p0 * 100
    # Path efficiency = |net| / Σ|step|, but measured on a fixed ~24-point
    # resample so it captures STRUCTURE (trend vs chop), not tick-level jitter
    # that scales with sampling frequency (1-sec ticks would always look choppy).
    k = min(seg_p.size, 24)
    coarse = seg_p[np.linspace(0, seg_p.size - 1, k).astype(int)]
    gross = float(np.abs(np.diff(coarse)).sum())
    path = (abs(p1 - p0) / gross) if gross > 0 else 0.0
    if vol_cum is not None:
        seg_cv = vol_cum[mask]
        vol = float(seg_cv[-1] - seg_cv[0])
    elif vol_bar is not None:
        vol = float(vol_bar[mask].sum())
    else:
        vol = float("nan")
    # actual covered span (window may be partial early in the session)
    span = (now - ts[mask][0]) / np.timedelta64(1, "m")
    return {"T": T, "ret": round(ret, 3), "range": round(rng, 3),
            "path": round(path, 2), "vol": vol, "span": round(float(span), 1),
            "rate": round(ret / max(float(span), 1e-9), 4)}


def _oi_delta(oi: pd.DataFrame, now_ts, T: int) -> dict:
    """ΔPCR and ΔOI between the snapshot nearest the window start and the latest."""
    anchor = now_ts - pd.Timedelta(minutes=T)
    before = oi[oi["ts"] <= anchor]
    a = before.iloc[-1] if len(before) else oi.iloc[0]
    b = oi.iloc[-1]
    def g(r, c):
        return float(r[c]) if c in r and pd.notna(r[c]) else 0.0
    return {"d_pcr": round(g(b, "pcr") - g(a, "pcr"), 3),
            "d_call": g(b, "total_call_oi") - g(a, "total_call_oi"),
            "d_put": g(b, "total_put_oi") - g(a, "total_put_oi")}


_Z_FLAT = 0.7        # |sigma-move| below this is indistinguishable from noise


def _sigma_1min(ts: np.ndarray, price: np.ndarray) -> float | None:
    """Session realized vol as std of 1-min % returns (the noise yardstick).
    Horizon vol scales ≈ σ₁·√T under a random walk — the normaliser for z-moves."""
    s = pd.Series(price, index=pd.DatetimeIndex(ts)).resample("60s").last().dropna()
    if len(s) < 5:
        return None
    r = s.pct_change().dropna() * 100
    sd = float(r.std())
    return sd if sd > 1e-9 else None


def _dirz(z: float) -> int:
    return 1 if z > _Z_FLAT else -1 if z < -_Z_FLAT else 0


def _incremental(ts: np.ndarray, price: np.ndarray, sigma1: float | None) -> dict | None:
    """Disjoint decomposition: the FRESH [now-5m, now] impulse vs the PRIOR
    [now-60m, now-5m] trend. Non-overlapping, so a sign clash is a TRUE reversal —
    not the subset-of-superset artifact nested cumulative windows produce."""
    if not sigma1:
        return None
    now = ts[-1]
    def seg(a: int, b: int):
        m = (ts >= now - np.timedelta64(b, "m")) & (ts <= now - np.timedelta64(a, "m"))
        return price[m]
    fresh, prior = seg(0, 5), seg(5, 60)
    if fresh.size < 2 or prior.size < 2:
        return None
    fr = (fresh[-1] - fresh[0]) / fresh[0] * 100
    pr = (prior[-1] - prior[0]) / prior[0] * 100
    return {"z_fresh": fr / (sigma1 * np.sqrt(5)),
            "z_prior": pr / (sigma1 * np.sqrt(55)), "fr": fr, "pr": pr}


def _oi_note(price_dir: int, d_put: float, d_call: float) -> tuple[int, str]:
    """Fuse positioning with price. Put-writing = support (bullish); call-writing =
    ceiling (bearish). Net OI bias confirming or contradicting price is the index's
    single highest-value cross-signal (it has no traded volume)."""
    # Floor: don't assert positioning off negligible flow. Both sides writing
    # heavily = range compression (premium sellers on both walls), not a bias.
    _FLOOR = 50_000              # 0.5L OI change
    net = d_put - d_call
    if max(abs(d_put), abs(d_call)) < _FLOOR or abs(net) < _FLOOR:
        both = abs(d_put) >= _FLOOR and abs(d_call) >= _FLOOR
        return 0, ("both walls writing — range compression" if both
                   else "no clear OI flow")
    bias = 1 if net > 0 else -1
    flow = (f"put-writing ΔP{d_put/1e5:+.0f}L" if abs(d_put) >= abs(d_call)
            else f"call-writing ΔC{d_call/1e5:+.0f}L")
    if price_dir == 0:
        return bias, f"positioning {'bullish' if bias > 0 else 'bearish'} ({flow})"
    if bias == price_dir:
        return bias, f"OI CONFIRMS ({flow})"
    return bias, f"OI DIVERGES — price {'up' if price_dir>0 else 'down'} but {flow}"


# ── Synthesis (the sharp part) ──────────────────────────────────────────────────

def _synthesize(rows: list[dict], incr: dict | None = None) -> dict:
    by_T = {r["T"]: r for r in rows}
    dirs = {r["T"]: _dirz(r.get("z", 0.0)) for r in rows}    # z-based, not raw ret
    signs = [d for d in dirs.values() if d != 0]
    up = sum(1 for d in signs if d > 0)
    dn = sum(1 for d in signs if d < 0)
    long_d = dirs.get(60, 0) or dirs.get(15, 0)

    # acceleration from a ROBUST short rate (5m, not noisy 1m) vs the long rate
    rate_s = by_T.get(5, by_T.get(1, {})).get("rate", 0.0)
    rate_l = by_T.get(60, by_T.get(15, {})).get("rate", 0.0)
    accel = ""
    if rate_s and rate_l and (rate_s > 0) == (rate_l > 0):
        accel = "accelerating" if abs(rate_s) > abs(rate_l) * 1.15 else \
                "fading" if abs(rate_s) < abs(rate_l) * 0.85 else "steady"

    chop = by_T.get(15, by_T.get(60, {})).get("path", 1.0)
    texture = "trending" if chop >= 0.6 else "choppy" if chop <= 0.3 else "mixed"

    # TRUE reversal from the disjoint decomposition (not nested windows)
    reversal = ""
    if incr:
        zf, zp = incr["z_fresh"], incr["z_prior"]
        if abs(zf) > _Z_FLAT and abs(zp) > _Z_FLAT and (zf > 0) != (zp > 0):
            reversal = (f"FRESH {abs(zf):.1f}σ {'up' if zf>0 else 'down'} impulse "
                        f"against a {abs(zp):.1f}σ {'up' if zp>0 else 'down'} prior trend")

    if reversal:
        verdict = reversal
    elif up and not dn:
        verdict = f"ALIGNED UP ({up} TF)"
    elif dn and not up:
        verdict = f"ALIGNED DOWN ({dn} TF)"
    else:
        verdict = "MIXED / TRANSITION"
    return {"verdict": verdict, "texture": texture, "up": up, "down": dn,
            "accel": accel, "dirs": dirs, "long_d": long_d}


# ── Public API ──────────────────────────────────────────────────────────────────

def analyze_index(sym: str, date: str | None = None,
                  as_of: datetime.datetime | None = None) -> dict:
    ticks = _load_index_ticks(sym, date, as_of)
    if ticks is None or len(ticks) < 3:
        return {"sym": sym, "has_data": False, "note": "no tick capture yet"}
    ts = ticks["ts"].to_numpy()
    price = ticks["ltp"].to_numpy(dtype="float64")
    cv = ticks["cum_vol"].to_numpy(dtype="float64")
    has_vol = float(np.nanmax(cv) - np.nanmin(cv)) > 0     # indices often have none
    oi = _load_index_oi(sym, date, as_of)
    now_ts = ticks["ts"].iloc[-1]
    sigma1 = _sigma_1min(ts, price)

    rows = []
    for T in HORIZONS:
        h = _horizon(ts, price, cv if has_vol else None, None, T)
        if h is None:
            continue
        h["z"] = round(h["ret"] / (sigma1 * np.sqrt(max(h["span"], 1e-9))), 2) if sigma1 else 0.0
        if oi is not None:
            h.update(_oi_delta(oi, now_ts, T))
        rows.append(h)
    incr = _incremental(ts, price, sigma1)
    syn = _synthesize(rows, incr)
    # OI fusion uses the longest horizon's net flow vs that horizon's price direction
    last = rows[-1] if rows else {}
    oi_bias, oi_msg = _oi_note(syn["dirs"].get(last.get("T"), 0),
                               last.get("d_put", 0), last.get("d_call", 0)) if oi is not None else (0, "")
    return {"sym": sym, "label": LABELS.get(sym, sym), "has_data": True,
            "now": str(now_ts), "spot": round(float(price[-1]), 2), "sigma1": sigma1,
            "rows": rows, "syn": syn, "oi_msg": oi_msg, "oi_bias": oi_bias,
            "has_vol": has_vol, "has_oi": oi is not None}


def analyze_hist(label: str) -> dict:
    loaded = _load_hist(label)
    if loaded is None:
        return {"sym": label, "has_data": False, "note": "no historical 5-min file"}
    lab, df = loaded
    ts = df["ts"].to_numpy()
    price = df["close"].to_numpy(dtype="float64")
    vol = df["volume"].to_numpy(dtype="float64")
    sigma1 = _sigma_1min(ts, price)
    rows = []
    for T in HORIZONS:
        if T % 5 != 0:                # 5-min bars can't resolve sub-5-min horizons
            continue
        h = _horizon(ts, price, None, vol, T)
        if h:
            h["z"] = round(h["ret"] / (sigma1 * np.sqrt(max(h["span"], 1e-9))), 2) if sigma1 else 0.0
            rows.append(h)
    incr = _incremental(ts, price, sigma1)
    return {"sym": lab, "label": lab, "has_data": bool(rows),
            "now": str(df["ts"].iloc[-1]), "spot": round(float(price[-1]), 2), "sigma1": sigma1,
            "rows": rows, "syn": _synthesize(rows, incr), "oi_msg": "", "oi_bias": 0,
            "has_vol": True, "has_oi": False}


# ── CLI rendering ───────────────────────────────────────────────────────────────

def _arrow(z: float) -> str:
    return "▲" if z > _Z_FLAT else "▼" if z < -_Z_FLAT else "·"


def _print_one(r: dict) -> None:
    if not r.get("has_data"):
        print(f"  {r.get('sym'):<26} — {r.get('note')}")
        return
    syn = r["syn"]
    sig = r.get("sigma1")
    sig_txt = f"σ₁ₘ {sig:.3f}%" if sig else "σ n/a"
    print(f"\n  {r['label']:<13} spot {r['spot']:>11,.2f}   {r['now'][11:19]}   {sig_txt}")
    hdr = ["TF", "dir", "ret%", "σ-move", "range%", "path"]
    if r["has_vol"]: hdr.append("Δvol")
    if r["has_oi"]:  hdr += ["ΔPCR", "ΔCallOI", "ΔPutOI"]
    print("    " + "  ".join(h.rjust(8) for h in hdr))
    for h in r["rows"]:
        cells = [f"{h['T']}m", _arrow(h.get("z", 0)), f"{h['ret']:+.2f}",
                 f"{h.get('z',0):+.1f}", f"{h['range']:.2f}", f"{h['path']:.2f}"]
        if r["has_vol"]:
            cells.append(f"{h['vol']/1e5:.1f}L" if not np.isnan(h["vol"]) else "—")
        if r["has_oi"]:
            cells += [f"{h.get('d_pcr',0):+.2f}",
                      f"{h.get('d_call',0)/1e5:+.1f}L", f"{h.get('d_put',0)/1e5:+.1f}L"]
        partial = "" if h["span"] >= h["T"] - 0.5 else f"  (only {h['span']:.0f}m)"
        print("    " + "  ".join(c.rjust(8) for c in cells) + partial)
    print(f"    └─ {syn['verdict']}  ·  texture: {syn['texture']}"
          + (f"  ·  {syn['accel']}" if syn['accel'] else ""))
    if r.get("oi_msg"):
        print(f"       positioning: {r['oi_msg']}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-horizon 'what changed per timeframe' read")
    ap.add_argument("--symbol", help="Single index (e.g. NIFTY, BANKNIFTY). Default: all 4")
    ap.add_argument("--hist", help="Historical stock by label (e.g. RELIANCE) from data/historical")
    ap.add_argument("--date", help="Replay an index day (YYYY-MM-DD) from its mirror")
    args = ap.parse_args()

    print("=" * 72)
    print("  WHAT CHANGED — PER TIMEFRAME  (1 / 5 / 10 / 15 / 60 min, anchored to now)")
    print("=" * 72)

    if args.hist:
        _print_one(analyze_hist(args.hist))
    elif args.symbol:
        sym = LABEL_TO_SYM.get(args.symbol.upper().replace(" ", ""), args.symbol)
        _print_one(analyze_index(sym, args.date))
    else:
        for sym in INDEX_SYMBOLS:
            _print_one(analyze_index(sym, args.date))
    print()


if __name__ == "__main__":
    main()
