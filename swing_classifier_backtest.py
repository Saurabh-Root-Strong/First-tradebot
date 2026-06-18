"""
swing_classifier_backtest.py — does a "real" swing continue, or revert?

The live panel flips bias on every 20-40pt swing. The user's question: can we tell
a REAL regime change from noise — and only then commit? Before building any such
classifier into the UI, this answers the prerequisite empirically:

    Classify each intraday move by VOL-NORMALISED size (z = move / rolling σ) and
    PERSISTENCE (same-sign dwell across the window). Then measure the forward move
    in the SAME direction over the next K bars.

    If forward-in-direction RISES with |z|  → big confirmed moves trend → FOLLOW.
    If it stays flat / goes NEGATIVE        → they revert → use the signal to FADE
                                              / as a run-over GUARD, never to chase.

This is detection (nowcast), not prediction — a well-defined, measurable target,
unlike the six directional nulls already in the book. Reads only the index 5-min
parquets (price + σ), no live mirror, so it is fast and reproducible.

    .venv\\Scripts\\python.exe swing_classifier_backtest.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HIST = os.path.join("data", "historical", "5min")
INDEX_FILES = {
    "NIFTY":   "NSE_NIFTY50_INDEX_5min.parquet",
    "BANK":    "NSE_NIFTYBANK_INDEX_5min.parquet",
    "FIN":     "NSE_FINNIFTY_INDEX_5min.parquet",
    "MIDCAP":  "NSE_MIDCPNIFTY_INDEX_5min.parquet",
}

# Experiment knobs.
W_SWING = 3        # window defining "the recent move" = 3 bars = 15 min
K_FWD   = 6        # forward horizon = 6 bars = 30 min
VOL_LOOKBACK = 75  # ~1 session of 5-min bars for the rolling σ estimate
Z_BUCKETS = [(0.0, 1.0, "noise   <1σ"),
             (1.0, 2.0, "moderate 1-2σ"),
             (2.0, 3.0, "real     2-3σ"),
             (3.0, np.inf, "extreme  >3σ")]


def _load(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path).sort_values("ts").reset_index(drop=True)
    df["ts"] = pd.to_datetime(df["ts"])
    df["date"] = df["ts"].dt.date
    return df


def _measure(df: pd.DataFrame) -> pd.DataFrame:
    """Per-bar: swing size in σ, persistence, forward move in σ — intraday only."""
    c = df["close"].to_numpy(dtype=float)
    ret = np.diff(c, prepend=c[0])                      # 1-bar price change
    # rolling σ of 1-bar moves (price units), shifted so it uses only PAST bars
    sigma = pd.Series(ret).rolling(VOL_LOOKBACK).std().shift(1).to_numpy()

    n = len(c)
    move   = np.full(n, np.nan)   # close[t] - close[t-W]
    persist = np.full(n, np.nan)  # fraction of the W steps that share the move's sign
    fwd    = np.full(n, np.nan)   # close[t+K] - close[t]
    same_day_back = df["date"].to_numpy()

    for t in range(n):
        # backward window must stay within the same session (no overnight gap)
        if t - W_SWING >= 0 and same_day_back[t] == same_day_back[t - W_SWING]:
            mv = c[t] - c[t - W_SWING]
            move[t] = mv
            steps = np.sign(ret[t - W_SWING + 1: t + 1])
            persist[t] = np.mean(steps == np.sign(mv)) if mv != 0 else 0.0
        # forward window must also stay in-session
        if t + K_FWD < n and same_day_back[t] == same_day_back[t + K_FWD]:
            fwd[t] = c[t + K_FWD] - c[t]

    out = pd.DataFrame({
        "sigma_W": sigma * np.sqrt(W_SWING),   # σ scaled to the swing window
        "move": move, "persist": persist,
        "sigma_K": sigma * np.sqrt(K_FWD),     # σ scaled to the forward window
        "fwd": fwd,
    })
    out["z"] = out["move"] / out["sigma_W"]
    # forward move expressed in the SAME direction as the swing, in σ_K units
    out["fwd_dir_sigma"] = np.sign(out["move"]) * out["fwd"] / out["sigma_K"]
    return out.replace([np.inf, -np.inf], np.nan).dropna(subset=["z", "fwd_dir_sigma"])


def _bucket_report(m: pd.DataFrame, label: str, persist_min: float | None = None) -> None:
    d = m if persist_min is None else m[m["persist"] >= persist_min]
    tag = label + (f"  [persist≥{persist_min:.0%}]" if persist_min else "")
    print(f"\n  {tag}   (n={len(d):,})")
    print(f"    {'bucket':<14}{'n':>8}{'fwd σ (dir)':>13}{'continue%':>11}{'mean fwd pt':>13}")
    for lo, hi, name in Z_BUCKETS:
        b = d[(d["z"].abs() >= lo) & (d["z"].abs() < hi)]
        if len(b) < 30:
            print(f"    {name:<14}{len(b):>8}{'  (thin)':>13}")
            continue
        fwd_sig = b["fwd_dir_sigma"].mean()
        cont = (b["fwd_dir_sigma"] > 0).mean() * 100
        fwd_pt = (np.sign(b["move"]) * b["fwd"]).mean()
        print(f"    {name:<14}{len(b):>8}{fwd_sig:>+13.3f}{cont:>10.1f}%{fwd_pt:>+13.1f}")


def main() -> None:
    print("=" * 72)
    print("  SWING CLASSIFIER BACKTEST — does a confirmed swing continue or revert?")
    print(f"  swing={W_SWING}bars(15m)  fwd={K_FWD}bars(30m)  σ-lookback={VOL_LOOKBACK}bars")
    print("=" * 72)
    print("  read: fwd σ (dir) > 0 = move CONTINUES;  < 0 = move REVERTS (mean-reversion)")

    pooled = []
    for sym, fname in INDEX_FILES.items():
        path = os.path.join(HIST, fname)
        if not os.path.exists(path):
            print(f"\n  {sym}: missing {path}")
            continue
        m = _measure(_load(path))
        m["sym"] = sym
        pooled.append(m)
        _bucket_report(m, sym)

    if pooled:
        allm = pd.concat(pooled, ignore_index=True)
        print("\n" + "-" * 72)
        _bucket_report(allm, "POOLED (all indices)")
        _bucket_report(allm, "POOLED (all indices)", persist_min=0.99)  # monotone swings only


if __name__ == "__main__":
    main()
