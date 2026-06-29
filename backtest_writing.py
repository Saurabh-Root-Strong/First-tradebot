"""
backtest_writing.py — does OPTION WRITING (short premium) actually pay intraday?

The scout's only validated product is the RANGE band (~68% close-in-band). A band
that holds = a strangle you wrote that expired worthless. So the band implies a
SHORT-VOL edge — but coverage is NOT P&L. The real question is the intraday
variance risk premium:

    is the premium you COLLECT (implied) > the move you PAY OUT (realised)?

This replays a short ATM straddle (and a band-edge strangle) entered at each
checkpoint, held H minutes, bought back at the same strikes on captured chain
premium. Writer P&L = credit - buyback, net of a round-trip cost. If the mean
clears 0 with a day-block CI, writing is a real edge and "write in sideways"
becomes the primary action — replacing buy-only. If not, writing bleeds like
buying and the band stays a DISPLAY-only risk map.

Conditioned on the scout's live |strength| (low = expect-sideways = write should
win MORE) so the filter is decidable in real time, not with hindsight.

Lookahead-free: entry uses chain<=t; buyback uses chain<=t+H; spot from ticks.

    .venv\\Scripts\\python.exe backtest_writing.py
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
import footprint_chart as fc
import intraday_scout as scout

SEED = 7
TF = 15
COST_PCT = 0.03          # round-trip bid-ask on premium traded (both legs), same scale as buy side
ENTRY_TIMES = ["09:45", "10:15", "10:45", "11:15", "11:45", "12:15", "12:45", "13:15"]
HORIZONS = (30, 60, 120)
GATE = 0.22              # scout's live trade gate; |strength|<GATE = "sideways" regime


def _chain_days():
    out = []
    for p in sorted(glob.glob(str(LIVE_DIR / "*_chain_snapshots.parquet"))):
        if os.path.getsize(p) < 50_000:        # skip empty / quota-truncated stubs
            continue
        out.append(os.path.basename(p).split("_")[0])
    return out


def _spot_at(ticks, t):
    s = ticks[ticks["ts"] <= pd.Timestamp(t)]
    return float(s.iloc[-1]["ltp"]) if len(s) else None


def _leg_asof(piv, ts, k):
    """Premium of strike k at-or-before ts from a (ts-index, strike-cols) pivot."""
    if k not in piv.columns:
        return np.nan
    v = piv[k].asof(pd.Timestamp(ts))
    return float(v) if pd.notna(v) else np.nan


def _boot_mean_ci(x, groups, reps, rng):
    """Day-block bootstrap CI of the mean (resample whole days)."""
    x = np.asarray(x, float)
    groups = np.asarray(groups)
    keep = ~np.isnan(x)
    x, groups = x[keep], groups[keep]
    if len(x) < 5:
        return np.nan, np.nan, np.nan
    days = np.unique(groups)
    idx_by_day = {d: np.where(groups == d)[0] for d in days}
    means = []
    for _ in range(reps):
        pick = rng.choice(days, size=len(days), replace=True)
        sel = np.concatenate([idx_by_day[d] for d in pick])
        means.append(x[sel].mean())
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def harvest(days):
    rows = []
    for date in days:
        d0 = datetime.date.fromisoformat(date)
        chain = _read("chain_snapshots", date, None, None)
        if chain is None or len(chain) == 0:
            continue
        for sym in INDEX_SYMBOLS:
            ticks = _read("ticks", date, None, sym)
            if ticks is None or len(ticks) < 20:
                continue
            cs = chain[chain["symbol"] == sym]
            if len(cs) == 0:
                continue
            ce = cs[cs["side"] == "CE"].pivot_table(index="ts", columns="strike",
                                                    values="ltp", aggfunc="last").sort_index()
            pe = cs[cs["side"] == "PE"].pivot_table(index="ts", columns="strike",
                                                    values="ltp", aggfunc="last").sort_index()
            strikes = np.array(sorted(set(ce.columns) & set(pe.columns)), float)
            if len(strikes) < 5:
                continue
            for hhmm in ENTRY_TIMES:
                hh, mm = map(int, hhmm.split(":"))
                t = datetime.datetime.combine(d0, datetime.time(hh, mm), tzinfo=IST)
                spot = _spot_at(ticks, t)
                if not spot:
                    continue
                k_atm = float(strikes[np.argmin(np.abs(strikes - spot))])
                # band edge (~1 strike step out) for the strangle
                step = float(np.median(np.diff(strikes))) if len(strikes) > 1 else 50.0
                k_ce = float(strikes[np.argmin(np.abs(strikes - (spot + step)))])
                k_pe = float(strikes[np.argmin(np.abs(strikes - (spot - step)))])
                ce0 = _leg_asof(ce, t, k_atm); pe0 = _leg_asof(pe, t, k_atm)
                ceo0 = _leg_asof(ce, t, k_ce); peo0 = _leg_asof(pe, t, k_pe)
                if not (ce0 == ce0 and pe0 == pe0 and ce0 > 0 and pe0 > 0):
                    continue
                strangle_ok = (ceo0 == ceo0 and peo0 == peo0 and ceo0 > 0 and peo0 > 0)
                # scout conviction at t (live-decidable regime filter)
                try:
                    ser = fc.build_series(sym, TF, date, t)
                    flow = scout._flow_signal(ser)[0]; div = scout._divergence_signal(ser)[0]
                    cross = scout._crossover_signal(ser)[0]; fut = scout._futures_signal(sym, TF, date, t)[0]
                    strength = abs(0.40 * flow + 0.25 * div + 0.20 * fut + 0.15 * cross)
                except Exception:
                    strength = np.nan
                rec = {"date": date, "sym": sym, "t": hhmm, "spot": spot,
                       "k_atm": k_atm, "credit_straddle": ce0 + pe0,
                       "credit_strangle": (ceo0 + peo0) if (ceo0 == ceo0 and peo0 == peo0) else np.nan,
                       "strength": strength}
                for H in HORIZONS:
                    tH = t + datetime.timedelta(minutes=H)
                    ce1 = _leg_asof(ce, tH, k_atm); pe1 = _leg_asof(pe, tH, k_atm)
                    buyback = (ce1 + pe1) if (ce1 == ce1 and pe1 == pe1) else np.nan
                    cred = ce0 + pe0
                    gross = cred - buyback
                    cost = COST_PCT * (cred + (buyback if buyback == buyback else 0.0))
                    rec[f"wr_gross{H}"] = (gross / cred * 100.0) if buyback == buyback else np.nan
                    rec[f"wr_net{H}"] = ((gross - cost) / cred * 100.0) if buyback == buyback else np.nan
                    sH = _spot_at(ticks, tH)
                    rec[f"realmove{H}"] = (abs(sH / spot - 1.0) * 100.0) if sH else np.nan
                    # OTM strangle at band edges (the band's actual instrument)
                    if strangle_ok:
                        ce1o = _leg_asof(ce, tH, k_ce); pe1o = _leg_asof(pe, tH, k_pe)
                        bbo = (ce1o + pe1o) if (ce1o == ce1o and pe1o == pe1o) else np.nan
                        credo = ceo0 + peo0
                        gro = credo - bbo
                        costo = COST_PCT * (credo + (bbo if bbo == bbo else 0.0))
                        rec[f"st_net{H}"] = ((gro - costo) / credo * 100.0) if bbo == bbo else np.nan
                        rec[f"st_gross{H}"] = (gro / credo * 100.0) if bbo == bbo else np.nan
                    else:
                        rec[f"st_net{H}"] = np.nan; rec[f"st_gross{H}"] = np.nan
                rows.append(rec)
    return pd.DataFrame(rows)


def report(df, rng):
    print("\nWRITING BACKTEST — short ATM straddle, MTM intraday, net of cost")
    print("=" * 78)
    days = sorted(df.date.unique())
    print(f"  rows={len(df)}  days={len(days)} ({days[0]}..{days[-1]})  indices={df.sym.nunique()}"
          f"  cost={COST_PCT*100:.0f}%/round-trip")

    def block(sub, label):
        print(f"\n  {label}  (n={len(sub)})")
        for H in HORIZONS:
            g = sub[f"wr_gross{H}"].to_numpy(float)
            n = sub[f"wr_net{H}"].to_numpy(float)
            gm, glo, ghi = _boot_mean_ci(g, sub.date.to_numpy(), 1500, rng)
            nm, nlo, nhi = _boot_mean_ci(n, sub.date.to_numpy(), 1500, rng)
            win = 100 * (n[~np.isnan(n)] > 0).mean() if (~np.isnan(n)).any() else float("nan")
            tag = "+" if nlo > 0 else ("-" if nhi < 0 else "0")
            print(f"     {H:>4}m  gross {gm:+5.1f}% [{glo:+5.1f},{ghi:+5.1f}]"
                  f"   net {nm:+5.1f}% [{nlo:+5.1f},{nhi:+5.1f}][{tag}]  win {win:4.0f}%")

    def block_st(sub, label):
        print(f"\n  {label}  (n={sub['st_net30'].notna().sum()})")
        for H in HORIZONS:
            g = sub[f"st_gross{H}"].to_numpy(float)
            n = sub[f"st_net{H}"].to_numpy(float)
            gm, glo, ghi = _boot_mean_ci(g, sub.date.to_numpy(), 1500, rng)
            nm, nlo, nhi = _boot_mean_ci(n, sub.date.to_numpy(), 1500, rng)
            win = 100 * (n[~np.isnan(n)] > 0).mean() if (~np.isnan(n)).any() else float("nan")
            tag = "+" if nlo > 0 else ("-" if nhi < 0 else "0")
            print(f"     {H:>4}m  gross {gm:+5.1f}% [{glo:+5.1f},{ghi:+5.1f}]"
                  f"   net {nm:+5.1f}% [{nlo:+5.1f},{nhi:+5.1f}][{tag}]  win {win:4.0f}%")

    block(df, "ATM STRADDLE — ALL checkpoints")
    side = df.dropna(subset=["strength"])
    block(side[side.strength < GATE], f"ATM STRADDLE — SIDEWAYS (scout |str|<{GATE}) — SHOULD win")
    block(side[side.strength >= GATE], f"ATM STRADDLE — CONVICTION (scout |str|>={GATE}) — SHOULD lose")
    print("\n  " + "-" * 60)
    block_st(df, "OTM STRANGLE (band edges) — ALL checkpoints")
    block_st(side[side.strength < GATE], f"OTM STRANGLE — SIDEWAYS (scout |str|<{GATE}) — SHOULD win")

    print("\n" + "=" * 78)
    print("READ: net CI clearing 0 (+) on SIDEWAYS = real intraday variance risk premium")
    print("-> 'write the straddle when scout says sideways' is the primary product.")
    print("CI straddling 0 = no writing edge either; band stays a DISPLAY risk map.")


def main():
    rng = np.random.default_rng(SEED)
    days = _chain_days()
    if not days:
        print("no chain days"); return
    print(f"chain days: {days}")
    df = harvest(days)
    if len(df) == 0:
        print("no rows harvested"); return
    report(df, rng)


if __name__ == "__main__":
    main()
