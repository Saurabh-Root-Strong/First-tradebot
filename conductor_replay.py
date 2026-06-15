"""
conductor_replay.py — lookahead-free validation of the Session Conductor across
every captured day.

What 7 days CAN honestly tell us (and what it cannot):
  • CAN  — stability (does the stance churn / whipsaw, or hold sensibly?),
           how often it correctly stands FLAT, and whether it usefully DIVERGES
           from the stale frozen opening call.
  • CANNOT — a forward-edge / expectancy claim. 7 days is statistically
           meaningless for that, and the cost study already showed the directional
           layer has no cost-survivable edge. Forward hit-rate is recorded only as
           a coarse coherence sanity check, with its sample size shown.

Each checkpoint reconstructs the conductor as-of that moment (no lookahead), with
position state tracked by THIS harness (not the live ledger). Forward return is the
underlying move over the next FWD_MIN, read from the same day's tick ledger.

  .venv\\Scripts\\python.exe conductor_replay.py
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

import session_conductor as SC
import timeframe_delta as TD

IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
LIVE_DIR = Path("data") / "intraday" / "live"
INDEX_SYMBOLS = SC.INDEX_SYMBOLS
LABELS = SC.LABELS

START = datetime.time(9, 45)
END   = datetime.time(15, 15)
STEP_MIN = 15
FWD_MIN  = 30
_STATE_DIR = {"LONG_DELTA": 1, "SHORT_DELTA": -1, "FLAT": 0}


def _days() -> list[str]:
    ds = sorted({p.name[:10] for p in LIVE_DIR.glob("*_ticks.parquet")})
    return ds


def _checkpoints(date: str) -> list[datetime.datetime]:
    d = datetime.date.fromisoformat(date)
    out, t = [], datetime.datetime.combine(d, START, tzinfo=IST)
    end = datetime.datetime.combine(d, END, tzinfo=IST)
    while t <= end:
        out.append(t)
        t += datetime.timedelta(minutes=STEP_MIN)
    return out


def _price_series(sym: str, date: str):
    df = TD._load_index_ticks(sym, date)
    if df is None or df.empty:
        return None
    return df[["ts", "ltp"]].reset_index(drop=True)


def _px_at(px: pd.DataFrame, t: datetime.datetime):
    sub = px[px["ts"] <= pd.Timestamp(t)]
    return float(sub["ltp"].iloc[-1]) if len(sub) else None


def replay_day_sym(date: str, sym: str) -> list[dict]:
    px = _price_series(sym, date)
    if px is None or len(px) < 20:
        return []
    recs, state = [], "FLAT"
    for t in _checkpoints(date):
        r = SC.conduct(sym, current_state=state, as_of=t, date=date)
        if not r.get("has_data"):
            continue
        # advance position state on an actionable transition
        if r["act_now"]:
            state = r["target_state"]
        p0 = _px_at(px, t)
        p1 = _px_at(px, t + datetime.timedelta(minutes=FWD_MIN))
        fwd = (p1 - p0) / p0 * 100 if (p0 and p1 and t.time() <= datetime.time(14, 45)) else None
        recs.append({"date": date, "sym": sym, "t": t.strftime("%H:%M"),
                     "dir": r["direction"], "state": state, "conv": r["conviction"],
                     "act": r["act_now"], "fused": r["fused"],
                     "open_dir": r.get("opening_dir"), "fwd": fwd})
    return recs


def _stab(recs: list[dict]) -> dict:
    """Stability over one (day,sym) acted-state path."""
    states = [_STATE_DIR[x["state"]] for x in recs]
    changes = sum(1 for a, b in zip(states, states[1:]) if a != b)
    flips = sum(1 for a, b in zip(states, states[1:]) if a * b < 0)   # LONG<->SHORT
    whip = sum(1 for a, b, c in zip(states, states[1:], states[2:])
               if a != 0 and a == c and b != a)                       # reversed within 2
    flat = sum(1 for x in recs if x["state"] == "FLAT")
    return {"n": len(recs), "changes": changes, "flips": flips, "whip": whip, "flat": flat}


def main() -> None:
    days = _days()
    print("=" * 78)
    print(f"  CONDUCTOR REPLAY VALIDATION  —  {len(days)} captured days  ({days[0]} … {days[-1]})")
    print("=" * 78)
    print(f"  checkpoints {START:%H:%M}-{END:%H:%M} every {STEP_MIN}m · forward {FWD_MIN}m · "
          f"lookahead-free as-of\n")

    all_recs: list[dict] = []
    print(f"  {'day':<11}{'index':<13}{'pts':>4}{'chg':>5}{'flip':>5}{'whip':>5}"
          f"{'flat%':>7}{'hit%':>7}{'avgFwd':>8}")
    print("  " + "-" * 74)
    for date in days:
        for sym in INDEX_SYMBOLS:
            recs = replay_day_sym(date, sym)
            if not recs:
                continue
            all_recs += recs
            s = _stab(recs)
            nf = [x for x in recs if x["dir"] != "FLAT" and x["fwd"] is not None]
            hit = (100 * np.mean([(x["dir"] == "LONG") == (x["fwd"] > 0) for x in nf])
                   if nf else float("nan"))
            avgfwd = (np.mean([(1 if x["dir"] == "LONG" else -1) * x["fwd"] for x in nf])
                      if nf else float("nan"))
            print(f"  {date:<11}{LABELS.get(sym, sym):<13}{s['n']:>4}{s['changes']:>5}"
                  f"{s['flips']:>5}{s['whip']:>5}{100*s['flat']/s['n']:>6.0f}%"
                  f"{hit:>6.0f}%{avgfwd:>+8.3f}")

    # ── Aggregate ────────────────────────────────────────────────────────────
    df = pd.DataFrame(all_recs)
    nf = df[(df["dir"] != "FLAT") & df["fwd"].notna()]
    by = df.groupby(["date", "sym"])
    flips = by.apply(lambda g: _stab(g.to_dict("records"))["flips"]).sum()
    whips = by.apply(lambda g: _stab(g.to_dict("records"))["whip"]).sum()
    n_paths = by.ngroups
    hit = 100 * ((nf["dir"] == "LONG") == (nf["fwd"] > 0)).mean()
    signed = ((nf["dir"].map({"LONG": 1, "SHORT": -1})) * nf["fwd"])

    # Divergence from the frozen opening call: does diverging help?
    dvg = nf[nf["open_dir"].notna()].copy()
    dvg["cdir"] = dvg["dir"].map({"LONG": "BULLISH", "SHORT": "BEARISH"})
    diverged = dvg[dvg["cdir"] != dvg["open_dir"]]
    agreed = dvg[dvg["cdir"] == dvg["open_dir"]]
    def _hit(g):
        return 100 * ((g["dir"] == "LONG") == (g["fwd"] > 0)).mean() if len(g) else float("nan")

    print("\n" + "=" * 78)
    print("  AGGREGATE")
    print("=" * 78)
    print(f"  checkpoints total      : {len(df):,}   over {n_paths} day×index paths")
    print(f"  FLAT (stood aside)     : {100*(df['dir']=='FLAT').mean():.0f}%")
    print(f"  directional flips      : {flips} total  ({flips/n_paths:.2f} per day×index path)")
    print(f"  whipsaws (reversed ≤2) : {whips} total  ({whips/n_paths:.2f} per path)  ← key stability tell")
    print(f"  acted (state change)   : {100*df['act'].mean():.0f}% of checkpoints")
    print("\n  — coarse forward coherence (NOT an edge claim; tiny sample) —")
    print(f"  directional checkpoints: {len(nf):,}")
    print(f"  forward hit-rate ({FWD_MIN}m) : {hit:.1f}%   (50% = coin flip)")
    print(f"  avg signed forward     : {signed.mean():+.3f}%   median {signed.median():+.3f}%")
    print(f"  when it AGREED w/ open : hit {_hit(agreed):.0f}%  (n={len(agreed)})")
    print(f"  when it DIVERGED       : hit {_hit(diverged):.0f}%  (n={len(diverged)})  "
          f"← does overriding the stale open help?")
    print()


if __name__ == "__main__":
    main()
