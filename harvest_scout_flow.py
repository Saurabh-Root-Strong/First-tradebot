"""
harvest_scout_flow.py — EOD harvest: persist each scout-ledger trade + its Phase-1 flow
context at trigger, so the FLOW-INVERSION finding (2026-07-17: flow-DISAGREE trades
outperformed flow-agree ~20x/trade, strong-agree worst — crowded-trade mechanism) keeps
measuring itself as days accumulate. SURVEILLANCE, not a signal: decision rule fixed in
advance — at n>=100 across >=3 months incl. a down-stretch, re-grade EX-jackpot-days; a
display chip only if it survives; auto-gating never.

ZERO live impact by construction: standalone process (no dashboard/state share), READ-ONLY
on the live mirrors, writes its OWN parquet (data/validation/scout_flow_harvest.parquet),
refuses to run before 15:35 IST, idempotent upsert. LAPTOP-ONLY — do not cron on the VM
(t3.micro is OOM-sensitive; this costs ~30s CPU).

    .venv\\Scripts\\python.exe harvest_scout_flow.py              # harvest today (post-close)
    .venv\\Scripts\\python.exe harvest_scout_flow.py --backfill   # all captured chain-days once
"""
from __future__ import annotations

import argparse
import datetime
import glob
import os
import sys

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from core.constants import IST, DATA_DIR
import tradeboard as tb
import intraday_scout as sc

OUT = DATA_DIR / "validation" / "scout_flow_harvest.parquet"
_EOD = datetime.time(15, 35)


def _chain_days():
    out = []
    for p in glob.glob(str(DATA_DIR / "intraday" / "live" / "*_chain_snapshots.parquet")):
        if os.path.getsize(p) > 1024:
            s = os.path.basename(p).split("_")[0]
            try:
                datetime.date.fromisoformat(s); out.append(s)
            except ValueError:
                pass
    return sorted(set(out))


def harvest_day(d: str) -> pd.DataFrame:
    dd = datetime.date.fromisoformat(d)
    as_of = datetime.datetime.combine(dd, _EOD, tzinfo=IST)
    L = tb.scout_pa_ledger(d, as_of, 15, 60)
    rows = []
    for r in L["closed"]:
        hh, mm = map(int, r["since"].split(":"))
        trig = datetime.datetime.combine(dd, datetime.time(hh, mm), tzinfo=IST)
        try:
            s = sc.scan_index(r["sym"], 15, date=d, as_of=trig, horizon_min=45)
        except Exception:
            s = {}
        st = float(s.get("strength") or 0.0) if s.get("has_data") else np.nan
        parts = s.get("parts") or {}
        want = 1 if r["side"] == "CE" else -1
        flow = ("agree" if (st == st and np.sign(st) == want and abs(st) > 0.02) else
                "disagree" if (st == st and np.sign(st) == -want and abs(st) > 0.02) else
                "neutral")
        rows.append({
            "date": d, "sym": r["sym"], "side": r["side"], "since": r["since"],
            "strike": r.get("strike"), "tag": r.get("tag"), "outcome": r["outcome"],
            "rmult": r.get("rmult"), "opt_rs": r.get("opt_rs"), "e_prem": r.get("e_prem"),
            "flow_strength": round(st, 3) if st == st else None, "flow_bucket": flow,
            "flow_strong_agree": bool(st == st and abs(st) >= 0.22 and np.sign(st) == want),
            "fut_part": round(float(parts.get("fut") or 0.0), 3),
        })
    return pd.DataFrame(rows)


def upsert(new: pd.DataFrame) -> pd.DataFrame:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    comb = (pd.concat([pd.read_parquet(OUT), new], ignore_index=True)
            if OUT.exists() else new)
    comb = comb.drop_duplicates(subset=["date", "sym", "since"], keep="last")
    comb = comb.sort_values(["date", "since"]).reset_index(drop=True)
    comb.to_parquet(OUT, index=False)
    return comb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", action="store_true", help="re-harvest ALL captured days")
    ap.add_argument("--force", action="store_true", help="skip the post-close guard for today")
    a = ap.parse_args()
    now = datetime.datetime.now(IST)
    today = now.date().isoformat()
    captured = _chain_days()
    if a.backfill:
        days = captured
    else:
        # SELF-HEALING default: harvest every captured day not yet in the parquet (the laptop
        # is a viewer — it may have been off at 15:40, or the VM sync late; any later run
        # catches up all missed days). PAST days are always safe (their market is closed);
        # TODAY only after 15:35 (or --force).
        done = set()
        if OUT.exists():
            done = set(pd.read_parquet(OUT, columns=["date"])["date"].unique())
        days = [d for d in captured if d not in done]
        if today in days and now.time() < _EOD and not a.force:
            days.remove(today)
            print(f"  (skipping {today} — before {_EOD} IST; it will be caught up later)")
        if not days:
            print("nothing new to harvest — all captured days already in the parquet")
            return
    frames = []
    for d in days:
        try:
            f = harvest_day(d)
            frames.append(f)
            print(f"  {d}: {len(f)} trades harvested")
        except Exception as e:
            print(f"  {d}: ERR {e}")
    new = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if new.empty:
        print("nothing to harvest"); return
    comb = upsert(new)
    # standing summary — the finding under surveillance
    g = comb.dropna(subset=["opt_rs"])
    print(f"\nharvest: {len(comb)} trades, {comb.date.nunique()} days -> {OUT.name}")
    for b in ("agree", "disagree", "neutral"):
        s = g[g.flow_bucket == b]
        if len(s):
            print(f"  flow-{b:<9} n={len(s):>3}  opt avg Rs{s.opt_rs.mean():>+7.0f}  "
                  f"idx avg {s.rmult.mean():+.3f}R  win {100*(s.opt_rs>0).mean():.0f}%")
    print("rule: re-grade EX-jackpot at n>=100 / >=3mo incl down-stretch; chip if it holds;")
    print("never auto-gate. Laptop-only — do not cron on the VM.")


if __name__ == "__main__":
    main()
