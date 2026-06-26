"""
backtest_sr_break.py — does an OI-wall support/resistance BREAK actually work?

The structural levels: resistance = the strike with the most CALL OI (call writers
cap there), support = the strike with the most PUT OI (put writers floor there).
oi_snapshots stores them live as call_wall / put_wall (lookahead-safe — computed
from OI<=t). This harness tests the idea you described on the chart: "price breaks
support/resistance -> trend changed, an edge."

On a 10-MINUTE bar grid, per index, per day it detects a BREAK (a bar that closes
from INSIDE a wall to OUTSIDE it) and asks the only honest questions:
  CONTINUATION   after the break, did price keep going that way over +10/20/30m?
                 ("the break worked" = trend follows through)
  FALSE BREAK    or did it snap back inside the wall? (the wall held; break failed)
Plus the counter-hypothesis every level-trader must check:
  REJECTION      when price TOUCHED a wall but did NOT break, did it reverse off it?
                 (walls as barriers = mean-reversion, the opposite trade)

Lookahead-free: walls + the break are read from data with ts<=t; the forward price
is the answer key only. Reads the LOCAL archive. Few days = COUNTS, not a CI — this
tells you "how many times it worked" on the captured days, honestly, not a verdict.

    .venv\\Scripts\\python.exe backtest_sr_break.py
    .venv\\Scripts\\python.exe backtest_sr_break.py --days 2026-06-23,2026-06-24,2026-06-25
    .venv\\Scripts\\python.exe backtest_sr_break.py --buf 0.05 --tf 10
"""
from __future__ import annotations

import argparse
import datetime
import sys

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from core.constants import IST, INDEX_SYMBOLS, LIVE_DIR, LABELS
from core.mirror_io import read_mirror as _read

HORIZONS = (10, 20, 30)


def _captured_days() -> list[str]:
    out = set()
    for p in LIVE_DIR.glob("*_oi_snapshots.parquet"):
        if p.stat().st_size < 1024:
            continue
        stem = p.name.split("_")[0]
        try:
            datetime.date.fromisoformat(stem); out.add(stem)
        except ValueError:
            continue
    return sorted(out)


def _wall_at(oi: pd.DataFrame, t, col: str):
    s = oi[oi["ts"] <= pd.Timestamp(t)]
    if not len(s):
        return None
    v = s.iloc[-1][col]
    try:
        v = float(v)
        return v if (v == v and v > 0) else None
    except (TypeError, ValueError):
        return None


def _spot_at(px: pd.Series, t):
    s = px[px.index <= pd.Timestamp(t)]
    return float(s.iloc[-1]) if len(s) else None


def harvest(days: list[str], tf: int, buf: float) -> pd.DataFrame:
    """One row per 10-min bar with the wall context + break flags + forward moves."""
    rows = []
    for date in days:
        d0 = datetime.date.fromisoformat(date)
        for sym in INDEX_SYMBOLS:
            oi = _read("oi_snapshots", date, None, sym)
            ticks = _read("ticks", date, None, sym)
            if (oi is None or "call_wall" not in oi.columns or ticks is None
                    or len(ticks) < 20):
                continue
            px = ticks.set_index("ts")["ltp"].sort_index()
            bars = px.resample(f"{tf}min", label="right", closed="right").last().dropna()
            prev_close = None
            for t, close in bars.items():
                R = _wall_at(oi, t, "call_wall")     # resistance (max call OI)
                S = _wall_at(oi, t, "put_wall")      # support   (max put OI)
                close = float(close)
                rec = {"date": date, "sym": sym, "t": t.strftime("%H:%M"),
                       "close": close, "R": R, "S": S,
                       "brk_up": 0, "brk_dn": 0, "touch_R": 0, "touch_S": 0}
                bp = buf / 100.0
                if prev_close is not None and R:
                    # break UP through resistance: was at/below R, now clears it +buf
                    if prev_close <= R * (1 + bp) and close > R * (1 + bp):
                        rec["brk_up"] = 1
                    # touched R from below but did NOT break (rejection candidate)
                    elif close <= R and close >= R * (1 - bp) and prev_close < R:
                        rec["touch_R"] = 1
                if prev_close is not None and S:
                    if prev_close >= S * (1 - bp) and close < S * (1 - bp):
                        rec["brk_dn"] = 1
                    elif close >= S and close <= S * (1 + bp) and prev_close > S:
                        rec["touch_S"] = 1
                for H in HORIZONS:
                    s_end = _spot_at(px, t + datetime.timedelta(minutes=H))
                    rec[f"fwd{H}"] = ((s_end / close - 1.0) * 100.0
                                      if (s_end and s_end != close) else np.nan)
                rows.append(rec)
                prev_close = close
    return pd.DataFrame(rows)


def _rate(mask_hit, n):
    return (100.0 * mask_hit / n) if n else float("nan")


def evaluate(df: pd.DataFrame, tf: int) -> None:
    print(f"\nOI-WALL S/R BREAK — does it work? ({tf}m bars, COUNTS not CI)")
    print("=" * 74)
    if df.empty:
        print("  no data."); return
    days = sorted(df.date.unique())
    print(f"  bars={len(df)}  days={len(days)} ({days[0]}..{days[-1]})  indices={df.sym.nunique()}")
    nbu, nbd = int(df.brk_up.sum()), int(df.brk_dn.sum())
    ntr, nts = int(df.touch_R.sum()), int(df.touch_S.sum())
    print(f"  events: resistance-break {nbu}, support-break {nbd}  |  "
          f"touched-not-broken R {ntr}, S {nts}")

    # ── BREAK CONTINUATION: after a break, did price keep going that way? ─────────
    print("\n  1) BREAK CONTINUATION  (the 'break works / trend changed' claim)")
    for H in HORIZONS:
        up = df[(df.brk_up == 1)].dropna(subset=[f"fwd{H}"])
        dn = df[(df.brk_dn == 1)].dropna(subset=[f"fwd{H}"])
        # continuation = moved further in the break direction
        up_cont = int((up[f"fwd{H}"] > 0).sum())
        dn_cont = int((dn[f"fwd{H}"] < 0).sum())
        ncont = up_cont + dn_cont
        ntot = len(up) + len(dn)
        # signed-by-direction mean forward move (bps): + = break followed through
        signed = np.concatenate([up[f"fwd{H}"].to_numpy(float),
                                 -dn[f"fwd{H}"].to_numpy(float)]) if ntot else np.array([])
        mbps = float(np.nanmean(signed) * 100) if len(signed) else float("nan")
        print(f"     +{H:>2}m  continued {ncont}/{ntot}  ({_rate(ncont, ntot):4.0f}%)  "
              f"vs 50% null   follow-through {mbps:+.0f}bps")

    # ── FALSE BREAK: did it snap back INSIDE the wall by +H? ──────────────────────
    print("\n  2) FALSE BREAK  (closed beyond the wall, then reverted back inside)")
    for H in HORIZONS:
        up = df[(df.brk_up == 1) & df.R.notna()].dropna(subset=[f"fwd{H}"]).copy()
        dn = df[(df.brk_dn == 1) & df.S.notna()].dropna(subset=[f"fwd{H}"]).copy()
        up["end"] = up["close"] * (1 + up[f"fwd{H}"] / 100.0)
        dn["end"] = dn["close"] * (1 + dn[f"fwd{H}"] / 100.0)
        up_false = int((up["end"] < up["R"]).sum())     # back below resistance
        dn_false = int((dn["end"] > dn["S"]).sum())      # back above support
        nfalse = up_false + dn_false
        ntot = len(up) + len(dn)
        print(f"     +{H:>2}m  false {nfalse}/{ntot}  ({_rate(nfalse, ntot):4.0f}%)")

    # ── REJECTION (counter-trade): touched the wall, did it reverse off it? ───────
    print("\n  3) WALL REJECTION  (touched but did NOT break -> mean-reversion off it)")
    for H in HORIZONS:
        tr = df[(df.touch_R == 1)].dropna(subset=[f"fwd{H}"])   # at resistance -> expect DOWN
        ts = df[(df.touch_S == 1)].dropna(subset=[f"fwd{H}"])   # at support    -> expect UP
        rej = int((tr[f"fwd{H}"] < 0).sum()) + int((ts[f"fwd{H}"] > 0).sum())
        ntot = len(tr) + len(ts)
        print(f"     +{H:>2}m  reversed {rej}/{ntot}  ({_rate(rej, ntot):4.0f}%)  vs 50% null")

    # ── PER-DAY break tally so you see day-to-day, not just pooled ────────────────
    print("\n  PER-DAY break continuation (+%dm)" % HORIZONS[0])
    H = HORIZONS[0]
    for d, g in df.groupby("date"):
        up = g[(g.brk_up == 1)].dropna(subset=[f"fwd{H}"])
        dn = g[(g.brk_dn == 1)].dropna(subset=[f"fwd{H}"])
        ncont = int((up[f"fwd{H}"] > 0).sum()) + int((dn[f"fwd{H}"] < 0).sum())
        ntot = len(up) + len(dn)
        print(f"     {d}   {ncont}/{ntot} continued"
              + (f"  ({_rate(ncont, ntot):.0f}%)" if ntot else "   (no breaks)"))

    print("\n" + "=" * 74)
    print("READ: 3 days = COUNTS, not significance. 'Break works' needs continuation")
    print(">50% AND low false-break OOS; if REJECTION>50% instead, the wall is a fade")
    print("(mean-reversion), the OPPOSITE of a breakout trade. Walls are real structure")
    print("either way — draw them; only TRADE a break once continuation holds OOS.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", default="", help="comma list, default = all captured")
    ap.add_argument("--tf", type=int, default=10)
    ap.add_argument("--buf", type=float, default=0.03, help="break buffer beyond wall, pct")
    args = ap.parse_args()

    days = ([d.strip() for d in args.days.split(",") if d.strip()]
            if args.days else _captured_days())
    if not days:
        print("No captured days in", LIVE_DIR); return
    print(f"reading mirror: {LIVE_DIR}  days={days}  tf={args.tf}m  buf={args.buf}%")
    df = harvest(days, args.tf, args.buf)
    evaluate(df, args.tf)


if __name__ == "__main__":
    main()
