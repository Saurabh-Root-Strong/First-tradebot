"""
backtest_shock.py — does a SHARP SHOCK (news footprint) lead to a tradeable reaction?

A sudden sharp move IS the market reacting to news, so we don't need a news feed — we
define a shock endogenously as an N-sigma move and test what follows. Two horizons:

  DAILY (the frontier): a >Nσ daily move — does it CONTINUE or REVERT next day? This is
    the overnight horizon where the intraday cost floor (~0.2R) does NOT bite: a next-day
    directional hold moves ~0.5-1%+, round-trip futures cost ~0.05% = negligible.
  INTRADAY: a >k·ATR 5m bar — continue or revert over the next 15/30m? (Expected to die
    at cost like every other intraday signal; included for completeness.)

Signed by shock direction so >50% / >0 = the shock CONTINUED. Bootstrap CI; net of a
realistic cost. Split up-shock vs down-shock (the short-asymmetry). Baseline = all-day
forward drift, so we see whether the shock ADDS information.

Data: data/historical/daily (4yr) + 5min (2yr), all 4 indices. Lookahead-free: shock
uses data <= d, forward = answer key.

    .venv\\Scripts\\python.exe backtest_shock.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SEED = 7
INDICES = ["NIFTY50", "NIFTYBANK", "FINNIFTY", "MIDCPNIFTY"]
DAILY_Z = (1.5, 2.0)
DAILY_COST = 0.05          # % round-trip, index futures overnight hold
INTRA_K = (2.0, 3.0)       # shock = |5m ret| > k * rolling std
INTRA_COST = 0.03


def _boot(x, rng, reps=2000):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 10:
        return None
    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(reps)]
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(x)


def _line(name, signed, rng, cost):
    r = _boot(signed, rng)
    if r is None:
        print(f"   {name:26s} n<10"); return
    m, lo, hi, n = r
    win = 100 * (np.asarray(signed)[~np.isnan(signed)] > 0).mean()
    net = np.asarray(signed, float) - cost
    nm, nlo, nhi, _ = _boot(net, rng)
    gf = "+" if lo > 0 else ("-" if hi < 0 else "0")
    nf = "+" if nlo > 0 else ("-" if nhi < 0 else "0")
    print(f"   {name:26s} GROSS {m:+.3f}% [{lo:+.3f},{hi:+.3f}][{gf}] win {win:3.0f}%  "
          f"NET(−{cost}%) {nm:+.3f}% [{nlo:+.3f},{nhi:+.3f}][{nf}]  n={n}")


def daily_test(rng):
    print("\nDAILY SHOCK -> next-day reaction  (the overnight frontier)")
    print("=" * 88)
    rows = []
    for idx in INDICES:
        d = pd.read_parquet(f"data/historical/daily/NSE_{idx}_INDEX_daily.parquet")
        d = d.sort_values("ts").reset_index(drop=True)
        c = d["close"].to_numpy(float); o = d["open"].to_numpy(float)
        ret = np.concatenate([[np.nan], np.diff(c) / c[:-1]]) * 100
        vol = pd.Series(ret).rolling(20).std().shift(1).to_numpy()      # causal trailing vol
        for i in range(21, len(d) - 1):
            if not (vol[i] > 0):
                continue
            z = ret[i] / vol[i]
            sdir = np.sign(ret[i])
            r1 = (c[i + 1] - c[i]) / c[i] * 100                          # close->next close
            gap = (o[i + 1] - c[i]) / c[i] * 100                          # overnight gap
            intr = (c[i + 1] - o[i + 1]) / o[i + 1] * 100                # next-day intraday
            rows.append({"idx": idx, "z": z, "sdir": sdir, "ret": ret[i],
                         "cont_close": r1 * sdir, "cont_gap": gap * sdir, "cont_intra": intr * sdir})
    df = pd.DataFrame(rows)
    print(f"  obs={len(df)}  indices={df.idx.nunique()}  (signed by shock dir: >0 = continued)")
    # baseline: all days next-day signed-by-prior-day drift already in cont_close at z~any
    for thr in DAILY_Z:
        s = df[df.z.abs() > thr]
        print(f"\n  |z| > {thr}  ({len(s)} shock days, {100*len(s)/len(df):.0f}% of days)")
        _line("close->next close", s["cont_close"], rng, DAILY_COST)
        _line("overnight gap", s["cont_gap"], rng, DAILY_COST)
        _line("next-day intraday", s["cont_intra"], rng, DAILY_COST)
        up = s[s.sdir > 0]; dn = s[s.sdir < 0]
        _line("  UP-shock close->close", up["cont_close"], rng, DAILY_COST)
        _line("  DOWN-shock close->close", dn["cont_close"], rng, DAILY_COST)
    # baseline drift (all days, unsigned next-day)
    base = df["cont_close"] * df["sdir"]      # undo sign -> raw next-day ret
    b = _boot(base.to_numpy(), rng)
    if b:
        print(f"\n  baseline raw next-day ret (all days): mean {b[0]:+.3f}% [{b[1]:+.3f},{b[2]:+.3f}] n={b[3]}")

    # ── ROBUSTNESS of the winning signal: overnight gap continuation, |z|>1.5 ──
    s = df[df.z.abs() > 1.5].copy()
    g = s["cont_gap"].to_numpy(float) - DAILY_COST     # net per-event
    print("\n  ROBUSTNESS — overnight gap continuation (net), |z|>1.5")
    print("  per INDEX:")
    for idx in INDICES:
        gi = (s[s.idx == idx]["cont_gap"].to_numpy(float) - DAILY_COST)
        r = _boot(gi, rng)
        if r:
            f = "+" if r[1] > 0 else ("-" if r[2] < 0 else "0")
            print(f"     {idx:11s} net {r[0]:+.3f}% [{r[1]:+.3f},{r[2]:+.3f}][{f}] win {100*(gi>0).mean():3.0f}% n={r[3]}")
    print("  per YEAR:")
    s["yr"] = pd.to_datetime(
        [pd.Timestamp(x) for x in s.index.map(lambda _: 0)]) if False else 0
    # recompute year from the source (need ts) — re-derive cheaply
    yrs = {}
    for idx in INDICES:
        d = pd.read_parquet(f"data/historical/daily/NSE_{idx}_INDEX_daily.parquet").sort_values("ts").reset_index(drop=True)
        d["ret"] = d["close"].pct_change() * 100
        d["vol"] = d["ret"].rolling(20).std().shift(1)
        d["yr"] = pd.to_datetime(d["ts"]).dt.year
        m = (d["ret"].abs() / d["vol"]) > 1.5
        for y, sub in d[m].groupby("yr"):
            yrs.setdefault(y, 0)
    for y in sorted(yrs):
        gi = []
        for idx in INDICES:
            d = pd.read_parquet(f"data/historical/daily/NSE_{idx}_INDEX_daily.parquet").sort_values("ts").reset_index(drop=True)
            c = d["close"].to_numpy(float); o = d["open"].to_numpy(float)
            ret = np.concatenate([[np.nan], np.diff(c) / c[:-1]]) * 100
            vol = pd.Series(ret).rolling(20).std().shift(1).to_numpy()
            yr = pd.to_datetime(d["ts"]).dt.year.to_numpy()
            for i in range(21, len(d) - 1):
                if vol[i] > 0 and abs(ret[i] / vol[i]) > 1.5 and yr[i] == y:
                    gi.append(np.sign(ret[i]) * (o[i + 1] - c[i]) / c[i] * 100 - DAILY_COST)
        gi = np.array(gi)
        if len(gi) >= 10:
            print(f"     {y}  net {gi.mean():+.3f}%  win {100*(gi>0).mean():3.0f}%  n={len(gi)}")
    # tail
    p5 = np.percentile(g, 5); worst = g.min(); big_adv = 100 * (g < -1.0).mean()
    print(f"  TAIL: mean {g.mean():+.3f}%  5th-pctile {p5:+.3f}%  worst {worst:+.3f}%  "
          f"P(loss>1%)={big_adv:.0f}%  (overnight gap-against risk)")


def intraday_test(rng):
    print("\n\nINTRADAY SHOCK -> next 15/30m  (expect cost-floor death)")
    print("=" * 88)
    rows = []
    for idx in INDICES:
        d = pd.read_parquet(f"data/historical/5min/NSE_{idx}_INDEX_5min.parquet")
        d = d.sort_values("ts").reset_index(drop=True)
        c = d["close"].to_numpy(float)
        ret = np.concatenate([[np.nan], np.diff(c) / c[:-1]]) * 100
        sd = pd.Series(ret).rolling(20).std().shift(1).to_numpy()
        for i in range(21, len(d) - 6):
            if not (sd[i] > 0):
                continue
            k = abs(ret[i]) / sd[i]
            sdir = np.sign(ret[i])
            f15 = (c[i + 3] - c[i]) / c[i] * 100 * sdir
            f30 = (c[i + 6] - c[i]) / c[i] * 100 * sdir
            rows.append({"idx": idx, "k": k, "f15": f15, "f30": f30})
    df = pd.DataFrame(rows)
    for thr in INTRA_K:
        s = df[df.k > thr]
        print(f"\n  |move| > {thr} sigma  ({len(s)} bars)")
        _line("fwd 15m (cont)", s["f15"], rng, INTRA_COST)
        _line("fwd 30m (cont)", s["f30"], rng, INTRA_COST)


def main():
    rng = np.random.default_rng(SEED)
    print("SHOCK-REACTION BACKTEST — sharp move = news footprint")
    daily_test(rng)
    intraday_test(rng)
    print("\n" + "=" * 88)
    print("READ: a horizon is tradeable only if NET CI clears 0. DAILY (overnight) is the")
    print("frontier — its cost is tiny vs the move. If close->next-close NET clears 0 on")
    print("shock days (esp. one direction), sharp-news reaction is a real overnight edge.")


if __name__ == "__main__":
    main()
