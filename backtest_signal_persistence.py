"""
backtest_signal_persistence.py — does a directional signal that has PERSISTED (same side
for P bars) predict the next 60m better than a fresh one-bar signal, net of cost?

This is the powered version of the user's question: "you gave a trade at 1:00, then the
SAME side again at 1:20 — is the 1:20 a confirmation that makes the trade more real?"

The full scout arrow (OI crossover / ΔOI flow / futures basis) exists only in ~15 captured
days (n~73 trades) — far too thin to split by hold-duration. So we proxy the arrow with the
ER-trend DIRECTION (regime_classifier: the sign of the efficient drift) on the full 2yr
5-min history, and ask whether a direction that has held for longer forecasts a better
signed forward return. If persistence lifts the net-of-cost edge above zero, "only act on a
re-confirmed signal" is a real filter; if not, a repeated zero-edge signal is still zero edge.

Signal at bar i (causal): s_i = sign of the ER drift (0 = chop/no signal). Persistence run =
number of consecutive prior bars ending at i with the same non-zero sign. Forward = signed
60m return = (close[i+12]/close[i] - 1) * s_i  (>0 means the signal was RIGHT).

Cost: index FUTURES round-trip ≈ 3 bps (the only vehicle with a real intraday edge in this
project — naked options lose ~3% at the wall). net = gross - 3 bps. A tradeable directional
edge needs net mean > 0 with a day-block-bootstrap CI that excludes 0.

    .venv\\Scripts\\python.exe backtest_signal_persistence.py [--horizon 60]
"""
from __future__ import annotations

import argparse
import datetime as dt

import numpy as np
import pandas as pd

import regime_classifier as rc

IDX = {
    "NIFTY 50":     "data/historical/5min/NSE_NIFTY50_INDEX_5min.parquet",
    "BANK NIFTY":   "data/historical/5min/NSE_NIFTYBANK_INDEX_5min.parquet",
    "FIN NIFTY":    "data/historical/5min/NSE_FINNIFTY_INDEX_5min.parquet",
    "MIDCAP NIFTY": "data/historical/5min/NSE_MIDCPNIFTY_INDEX_5min.parquet",
}
SESSION_OPEN = dt.time(9, 15)
SESSION_CLOSE = dt.time(15, 30)
FIRST = dt.time(9, 45)
LAST = dt.time(15, 0)
STEP = 15
ER_N = 10
COST_BPS = 3.0                       # futures round-trip

# persistence buckets, in 5-min bars (1 bar = 5 min)
BUCKETS = [("fresh 1-2b (≤10m)", 1, 2),
           ("held 3-5b (15-25m)", 3, 5),
           ("held 6-11b (30-55m)", 6, 11),
           ("held 12+b (≥60m)", 12, 10_000)]


def _pred_times(day):
    t = dt.datetime.combine(day, FIRST); end = dt.datetime.combine(day, LAST)
    out = []
    while t <= end:
        out.append(t); t += dt.timedelta(minutes=STEP)
    return out


def _boot_ci(day_vals: dict, n_boot=2000, seed=7):
    """Day-block bootstrap mean CI: resample DAYS (keeps within-day overlap correlation)."""
    rng = np.random.default_rng(seed)
    days = list(day_vals.keys())
    if len(days) < 5:
        return (float("nan"), float("nan"))
    arrs = [np.array(day_vals[d]) for d in days]
    means = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(days), len(days))
        cat = np.concatenate([arrs[i] for i in pick])
        means[b] = cat.mean()
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def run(H: int) -> None:
    bars_fwd = H // 5
    print("=" * 94)
    print(f"  SIGNAL PERSISTENCE → net edge?  H={H}m  signal=ER{ER_N}-trend sign  "
          f"cost={COST_BPS}bps (futures)")
    print("=" * 94)

    # per bucket: signed gross bps, and per-day lists for the block bootstrap
    agg = {b[0]: {"gross": [], "byday": {}} for b in BUCKETS}

    for name, path in IDX.items():
        d = pd.read_parquet(path)
        d["ts"] = pd.to_datetime(d["ts"])
        d = d[(d["ts"].dt.time >= SESSION_OPEN) & (d["ts"].dt.time <= SESSION_CLOSE)]
        d["day"] = d["ts"].dt.date
        for day, g in d.groupby("day", sort=True):
            g = g.sort_values("ts").reset_index(drop=True)
            if len(g) < ER_N + bars_fwd + 2:
                continue
            ts = g["ts"].to_numpy(); close = g["close"].to_numpy(float)
            signs = np.sign(rc.classify_series(g, n=ER_N)["sign"].to_numpy())  # +1/-1/0
            # persistence run length ending at each bar (consecutive same non-zero sign)
            run_len = np.zeros(len(signs), int)
            for i in range(len(signs)):
                if signs[i] == 0:
                    run_len[i] = 0
                elif i > 0 and signs[i] == signs[i - 1]:
                    run_len[i] = run_len[i - 1] + 1
                else:
                    run_len[i] = 1
            for t in _pred_times(day):
                tp = np.datetime64(t); upto = ts <= tp
                nb = int(upto.sum())
                if nb < ER_N or nb - 1 + bars_fwd >= len(close):
                    continue
                i = nb - 1
                s = signs[i]
                if s == 0:                                   # chop = no directional signal
                    continue
                fwd = (close[i + bars_fwd] / close[i] - 1.0) * 100.0 * 100.0  # bps
                signed = fwd * s                             # >0 = signal correct
                rl = run_len[i]
                for label, lo, hi in BUCKETS:
                    if lo <= rl <= hi:
                        agg[label]["gross"].append(signed)
                        agg[label]["byday"].setdefault((name, day), []).append(signed)
                        break

    print(f"\n  {'persistence bucket':22}{'N':>7}{'gross bps':>11}{'hit%':>7}"
          f"{'net bps':>9}{'net CI (day-block 95%)':>26}")
    print("  " + "-" * 82)
    for label, lo, hi in BUCKETS:
        gross = np.array(agg[label]["gross"])
        if len(gross) == 0:
            print(f"  {label:22}{'0':>7}{'n/a':>11}"); continue
        g_mean = gross.mean()
        hit = 100.0 * (gross > 0).mean()
        net_mean = g_mean - COST_BPS
        # bootstrap CI on the NET mean (shift the per-day values by cost)
        byday_net = {k: [v - COST_BPS for v in vs] for k, vs in agg[label]["byday"].items()}
        lo_ci, hi_ci = _boot_ci(byday_net)
        flag = "  ✱ excl 0" if (lo_ci > 0 or hi_ci < 0) else ""
        print(f"  {label:22}{len(gross):>7}{g_mean:>11.2f}{hit:>7.1f}{net_mean:>9.2f}"
              f"   [{lo_ci:>6.2f}, {hi_ci:>6.2f}]{flag}")

    print("\n  READ: if 'net bps' RISES with persistence AND the longest bucket's CI EXCLUDES 0")
    print("  (net>0), a re-confirmed signal is a real tradeable filter. If net stays ≤0 / CI")
    print("  straddles 0 at every persistence, a repeated signal is a repeated coin flip —")
    print("  persistence does NOT rescue the arrow from the cost floor.")


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=60)
    a = ap.parse_args()
    run(a.horizon)
