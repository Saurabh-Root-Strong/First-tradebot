"""
backtest_scout.py — does the SCOUT's own call actually pay? (the missing verdict)

The scout's strength = 0.40·flow + 0.25·div + 0.20·fut + 0.15·cross with a hard
gate (|str|>=0.22 AND >=3/4 agree) was hand-set, never validated. backtest_suggestion
grades a DIFFERENT model (hour_forecast); backtest_crossover grades ONE factor. This
harness replays the scout's EXACT live read — by calling intraday_scout.scan_index at
each checkpoint (parity by construction, lookahead-free: it reads mirrors with ts<=t)
— and scores it across every captured day with a day-block bootstrap CI.

It answers the only question that matters before you risk money:
  1. IC(strength, fwd_ret)               does the score rank the next move at all?
  2. TRADE sign-hit vs 50%               do the gated CE/PE calls beat a coin-flip?
  3. COST-AWARE hit                       same, but a call only "wins" if the index
                                          moved more than COST_BPS in its favour —
                                          a directional ✓ on a +0.03% move LOSES on
                                          the option after spread+theta. This is the
                                          honest, P&L-relevant number.
  4. RANGE close-in-band vs ~68%          the band at the scaled horizon.

Reads the LOCAL archive (data/intraday/live). Lookahead-free: the scout call uses
only ts<=t; the forward price is read AFTER t purely as the answer key, never fed
back. NOT wired to anything — this is the gate the scout must clear OOS before any
directional trade. Until IC-CI clears 0 / hit-CI clears 50% on OUT-of-sample days,
the directional call is decision-support, not a signal.

    .venv\\Scripts\\python.exe backtest_scout.py
    .venv\\Scripts\\python.exe backtest_scout.py --tf 15 --reps 3000
"""
from __future__ import annotations

import argparse
import datetime
import sys

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from core.constants import IST, INDEX_SYMBOLS, LIVE_DIR, DATA_DIR
from core.mirror_io import read_mirror as _read
from backtest_continuity import _spearman, _boot_ci
import intraday_scout as scout

SEED = 7
SAMPLE_TIMES = ["09:45", "10:00", "10:15", "10:30", "10:45", "11:00", "11:15",
                "11:30", "11:45", "12:00", "12:15", "12:30", "12:45", "13:00",
                "13:15", "13:30", "13:45", "14:00", "14:15", "14:30"]
HORIZONS = (5, 15, 30, 60)
# Cost hurdle for a directionally-correct call to actually PROFIT on the option.
# Index weekly ATM round-trip ~ spread + a few minutes theta. Conservative single
# number in index-% terms; the call must move the index at least this far its way.
COST_BPS = 6.0                                  # 0.06% of index
COST_OPT = scout._OPT_RT_COST                   # option round-trip % (single source)
LEDGER = DATA_DIR / "validation" / "scout_ledger.parquet"


def _captured_days() -> list[str]:
    out = set()
    for p in LIVE_DIR.glob("*_oi_snapshots.parquet"):
        if p.stat().st_size < 1024:                 # skip clobbered stubs
            continue
        stem = p.name.split("_")[0]
        try:
            datetime.date.fromisoformat(stem); out.add(stem)
        except ValueError:
            continue
    return sorted(out)


def _spot_at(ticks, t):
    s = ticks[ticks["ts"] <= pd.Timestamp(t)]
    return float(s.iloc[-1]["ltp"]) if len(s) else None


def harvest(days: list[str], tf: int) -> pd.DataFrame:
    rows = []
    for date in days:
        d0 = datetime.date.fromisoformat(date)
        for sym in INDEX_SYMBOLS:
            ticks = _read("ticks", date, None, sym)
            if ticks is None or len(ticks) < 20:
                continue
            for hhmm in SAMPLE_TIMES:
                hh, mm = map(int, hhmm.split(":"))
                t = datetime.datetime.combine(d0, datetime.time(hh, mm), tzinfo=IST)
                r = scout.scan_index(sym, tf, date=date, as_of=t, horizon_min=tf)
                if not r.get("has_data") or not r.get("spot"):
                    continue
                spot_t = r["spot"]
                is_trade = int(r["verdict"].startswith("TRADE"))
                atm, dir_ = r.get("atm"), r.get("direction")
                rec = {
                    "date": date, "sym": sym, "t": hhmm, "spot": spot_t,
                    "strength": r["strength"], "verdict": r["verdict"],
                    "direction": dir_, "pred_dir": r["pred_dir"],
                    "agree": r["agree"], "atm": atm,
                    "pred_lo": r.get("pred_lo"), "pred_hi": r.get("pred_hi"),
                    "is_trade": is_trade,
                }
                # price-structure veto (measured, not enforced live) — does skipping
                # arrows fired into S/R or no-breakout coils cut the option bleed?
                st = r.get("struct") or {}
                rec["struct_veto"] = int(bool(r.get("struct_veto")))
                rec["breakout"] = st.get("breakout") or ""
                rec["consolidating"] = int(bool(st.get("consolidating")))
                # day-trend veto (don't fight the trend) — measured, not enforced live
                rg = r.get("regime") or {}
                rec["trend_veto"] = int(bool(r.get("trend_veto")))
                rec["trend"] = rg.get("trend") or ""
                # range coverage at the scaled (horizon=tf) band
                s_h = _spot_at(ticks, t + datetime.timedelta(minutes=tf))
                lo, hi = r.get("pred_lo"), r.get("pred_hi")
                rec["close_in_band"] = (float(lo <= s_h <= hi)
                                        if (lo is not None and hi is not None and s_h)
                                        else np.nan)
                for H in HORIZONS:
                    s_end = _spot_at(ticks, t + datetime.timedelta(minutes=H))
                    rec[f"ret{H}"] = ((s_end / spot_t - 1.0) * 100.0
                                      if (s_end and s_end != spot_t) else np.nan)
                # OPTION P&L: buy the ATM CE/PE at t, exit at t+H, net of cost — the
                # honest grade (the actual trade, not the index tick).
                if is_trade and atm and dir_ in ("CE", "PE"):
                    entry = scout._opt_premium(sym, date, t, atm, dir_)
                    rec["opt_entry"] = entry
                    for H in HORIZONS:
                        x = scout._opt_premium(sym, date,
                                               t + datetime.timedelta(minutes=H), atm, dir_)
                        rec[f"opt_net{H}"] = (((x / entry - 1.0) * 100.0 - scout._OPT_RT_COST)
                                              if (entry and x) else np.nan)
                rows.append(rec)
    return pd.DataFrame(rows)


def _upsert(new: pd.DataFrame) -> pd.DataFrame:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    comb = (pd.concat([pd.read_parquet(LEDGER), new], ignore_index=True)
            if LEDGER.exists() else new)
    comb = comb.drop_duplicates(subset=["date", "sym", "t"], keep="last").reset_index(drop=True)
    comb.to_parquet(LEDGER, index=False)
    return comb


def evaluate(df: pd.DataFrame, reps: int, rng, tf: int) -> None:
    print("\nSCOUT SIGNAL — does the scout's own call pay? (day-block CV vs nulls)")
    print("=" * 74)
    if df.empty:
        print("  ledger empty — run a harvest first."); return
    days = sorted(df.date.unique())
    n_trade = int(df.is_trade.sum())
    print(f"  rows={len(df)}  days={len(days)} ({days[0]}..{days[-1]})  "
          f"indices={df.sym.nunique()}  TRADE rows={n_trade} ({100*n_trade/max(len(df),1):.0f}%)")

    # 1) IC(strength, fwd_ret) — does the score rank the move?
    print("\n  1) IC( strength , fwd_ret )   [EDGE only if CI clears 0]")
    for H in HORIZONS:
        sub = df.dropna(subset=[f"ret{H}"])
        if len(sub) < 10 or sub.date.nunique() < 2:
            print(f"     {H:>2}m  n={len(sub)} insufficient"); continue
        ic, lo, hi = _boot_ci(_spearman, sub["strength"].to_numpy(float),
                              sub[f"ret{H}"].to_numpy(float),
                              reps=reps, rng=rng, groups=sub["date"].to_numpy())
        v = "EDGE" if lo > 0 else ("anti" if hi < 0 else "—")
        print(f"     {H:>2}m  IC {ic:+.3f} [{lo:+.3f},{hi:+.3f}] {v}  (n={len(sub)})")

    # 2) TRADE sign-hit vs 50% (raw direction) + 3) cost-aware hit
    print("\n  2/3) GATED TRADE calls — raw dir-hit AND cost-aware hit (>%.2f%% move)"
          % (COST_BPS / 100.0))
    tr = df[df.is_trade == 1].copy()
    for H in HORIZONS:
        sub = tr.dropna(subset=[f"ret{H}"])
        if len(sub) < 5 or sub.date.nunique() < 2:
            print(f"     {H:>2}m  n={len(sub)} too few trades"); continue
        call = np.where(sub.direction == "CE", 1.0, -1.0)
        ret = sub[f"ret{H}"].to_numpy(float)
        raw = (call == np.sign(ret)).astype(float)
        # cost-aware: right direction AND move beyond the cost hurdle in its favour
        favour = call * ret                                    # +% when call was right
        net = (favour >= (COST_BPS / 100.0)).astype(float)
        hr, l1, h1 = _boot_ci(lambda a: a.mean(), raw, reps=reps, rng=rng,
                              groups=sub["date"].to_numpy())
        nr, l2, h2 = _boot_ci(lambda a: a.mean(), net, reps=reps, rng=rng,
                              groups=sub["date"].to_numpy())
        rv = "EDGE" if l1 > 0.5 else ("anti" if h1 < 0.5 else "—")
        print(f"     {H:>2}m  raw {100*hr:4.1f}% [{100*l1:4.1f},{100*h1:4.1f}] {rv}"
              f"   cost-aware {100*nr:4.1f}% [{100*l2:4.1f},{100*h2:4.1f}]  (n={len(sub)})")

    # 3b) OPTION P&L — the honest grade: actual ATM CE/PE bought at t, net of cost
    print("\n  3b) OPTION P&L (buy ATM CE/PE, exit at +H, net %.0f%% round-trip)"
          % COST_OPT)
    if "opt_net15" in df.columns:
        for H in HORIZONS:
            col = f"opt_net{H}"
            sub = df[(df.is_trade == 1)].dropna(subset=[col]) if col in df.columns else df.iloc[0:0]
            if len(sub) < 5 or sub.date.nunique() < 2:
                print(f"     {H:>2}m  n={len(sub)} too few"); continue
            net = sub[col].to_numpy(float)
            win = (net > 0).astype(float)
            wr, lw, hw = _boot_ci(lambda a: a.mean(), win, reps=reps, rng=rng,
                                  groups=sub["date"].to_numpy())
            mn, lm, hm = _boot_ci(lambda a: a.mean(), net, reps=reps, rng=rng,
                                  groups=sub["date"].to_numpy())
            wv = "EDGE" if lw > 0.5 else ("anti" if hw < 0.5 else "—")
            ev = "EDGE" if lm > 0 else ("bleed" if hm < 0 else "—")
            print(f"     {H:>2}m  win {100*wr:4.1f}% [{100*lw:4.1f},{100*hw:4.1f}] {wv}"
                  f"   mean net {mn:+5.1f}% [{lm:+5.1f},{hm:+5.1f}] {ev}  (n={len(sub)})")
    else:
        print("     no option premiums captured on these days")

    # 3c) PRICE-STRUCTURE VETO — does skipping the vetoed arrows cut the bleed?
    # Compares the option-P&L of trades the structure KEEPS (struct_veto=0) vs the ones
    # it VETOES (into S/R, or a no-breakout coil). The veto earns its place only if the
    # vetoed bucket bleeds MORE than the kept bucket — i.e. it removes losers.
    if "struct_veto" in df.columns and "opt_net15" in df.columns:
        print("\n  3c) STRUCTURE VETO effect on option P&L (kept vs vetoed trades)")
        for H in HORIZONS:
            col = f"opt_net{H}"
            if col not in df.columns:
                continue
            sub = df[df.is_trade == 1].dropna(subset=[col])
            if len(sub) < 6 or sub.date.nunique() < 2:
                print(f"     {H:>2}m  n={len(sub)} too few"); continue
            kept = sub[sub.struct_veto == 0]
            veto = sub[sub.struct_veto == 1]
            def _m(s):
                return (float(s[col].mean()), len(s)) if len(s) else (float("nan"), 0)
            mk, nk = _m(kept); mv, nv = _m(veto)
            if nk >= 5 and kept.date.nunique() >= 2:
                ck, lk, hk = _boot_ci(lambda a: a.mean(), kept[col].to_numpy(float),
                                      reps=reps, rng=rng, groups=kept["date"].to_numpy())
                kept_ci = f"[{lk:+.1f},{hk:+.1f}]"
            else:
                kept_ci = ""
            tag = "VETO HELPS" if (nv and mv < mk) else ("no help" if nv else "no vetoes")
            print(f"     {H:>2}m  kept {mk:+5.1f}% (n={nk}) {kept_ci}   "
                  f"vetoed {mv:+5.1f}% (n={nv})   {tag}")

    # 3d) TREND VETO — does NOT fighting the day trend cut the bleed? (the SL-audit fix)
    if "trend_veto" in df.columns and "opt_net15" in df.columns:
        print("\n  3d) TREND VETO effect on option P&L (kept vs trend-vetoed trades)")
        for H in HORIZONS:
            col = f"opt_net{H}"
            if col not in df.columns:
                continue
            sub = df[df.is_trade == 1].dropna(subset=[col])
            if len(sub) < 6 or sub.date.nunique() < 2:
                print(f"     {H:>2}m  n={len(sub)} too few"); continue
            kept = sub[sub.trend_veto == 0]
            veto = sub[sub.trend_veto == 1]
            mk = float(kept[col].mean()) if len(kept) else float("nan")
            mv = float(veto[col].mean()) if len(veto) else float("nan")
            if len(kept) >= 5 and kept.date.nunique() >= 2:
                ck, lk, hk = _boot_ci(lambda a: a.mean(), kept[col].to_numpy(float),
                                      reps=reps, rng=rng, groups=kept["date"].to_numpy())
                kept_ci = f"[{lk:+.1f},{hk:+.1f}]" + ("  EDGE" if lk > 0 else "")
            else:
                kept_ci = ""
            tag = ("VETO HELPS" if (len(veto) and mv < mk) else
                   ("no help" if len(veto) else "no vetoes"))
            print(f"     {H:>2}m  kept {mk:+5.1f}% (n={len(kept)}) {kept_ci}   "
                  f"vetoed {mv:+5.1f}% (n={len(veto)})   {tag}")

    # 4) RANGE band coverage at the scaled horizon
    cib = df.dropna(subset=["close_in_band"])
    if len(cib) >= 5 and cib.date.nunique() >= 2:
        cr, lo, hi = _boot_ci(lambda a: a.mean(), cib["close_in_band"].to_numpy(float),
                              reps=reps, rng=rng, groups=cib["date"].to_numpy())
        print(f"\n  4) RANGE {tf}m close-in-band {100*cr:.1f}% [{100*lo:.1f},{100*hi:.1f}]  "
              f"(target ~68%, n={len(cib)})")

    print("\n" + "=" * 74)
    print("READ: the GATED call earns a directional trade ONLY if its cost-aware hit-CI")
    print("clears 50% OUT of sample. Raw dir-hit ignores option spread+theta — a ✓ on a")
    print("flat move still loses money. Few days → wide CIs is honest. Until this clears,")
    print("the scout is decision-support; the range band is the only validated product.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", type=int, default=15)
    ap.add_argument("--reps", type=int, default=3000)
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    if args.eval_only:
        if not LEDGER.exists():
            print("No ledger yet — run without --eval-only first."); return
        evaluate(pd.read_parquet(LEDGER), args.reps, rng, args.tf); return

    days = _captured_days()
    if not days:
        print("No captured days in", LIVE_DIR); return
    print(f"reading mirror: {LIVE_DIR}  (tf={args.tf}m)")
    new = harvest(days, args.tf)
    print(f"harvested {len(new)} rows from {len(days)} days -> upserting {LEDGER.name}")
    comb = _upsert(new) if not new.empty else (pd.read_parquet(LEDGER) if LEDGER.exists() else new)
    evaluate(comb, args.reps, rng, args.tf)


if __name__ == "__main__":
    main()
