"""
backtest_crossover.py — is the intraday CE/PE OI crossover worth anything?

The charts show one structural signal everyone fixates on: total PUT OI vs total
CALL OI (oi_snapshots) — the "floor vs ceiling" lines that cross mid-session. This
harness ISOLATES that crossover (it is otherwise buried inside the playbook's f_oi
composite, never scored on its own) and answers three honest questions across every
captured day, lookahead-free, with a DAY-BLOCK bootstrap CI (whole days resampled —
intraday rows share one drift, so the day is the only honest sampling unit):

  1. STANDING TILT   IC( g , fwd_ret )    g = (putOI - callOI) / (putOI + callOI)
                     does a put-OI-dominant book (g>0, "floor") precede an up move?
  2. CROSSOVER EVENT fwd_ret right after g flips sign (PE overtakes CE = bullish
                     flip; the reverse = bearish flip) vs the day's base drift.
  3. SCOREBOARD      the replay workflow you described — at each 15-min checkpoint
                     a transparent structural call (sign of g ⊕ futures basis); per
                     day a W/L tally ("how many did I get right today"), then pooled
                     hit-rate vs the 50% coin-flip with a day-block CI.

Lookahead-free: features read only oi_snapshots / futures_quotes with ts<=t; the
forward label reads ticks AFTER t (that future read is the answer key, used only to
score — never fed back into the call). Reads the LOCAL archive (data/intraday/live),
the authoritative superset eod_sync pulls home from the VM each night.

NOT wired into any live decision. Re-run as days accumulate (or via eod_sync); the
verdict sharpens. Act on the crossover ONLY once it clears these nulls out-of-sample
— direction has been null in every index test so far (see hour_forecast).

    .venv\\Scripts\\python.exe backtest_crossover.py
    .venv\\Scripts\\python.exe backtest_crossover.py --eval-only
"""
from __future__ import annotations

import argparse
import datetime
import sys

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Reads the LOCAL archive (data/intraday/live) by default — eod_sync pulls the VM's
# office-hours capture home every night, so local is the authoritative, space-
# unlimited superset. Override with TRADEBOT_MIRROR_DIR only to read a specific dir.

from core.constants import IST, INDEX_SYMBOLS, LIVE_DIR, DATA_DIR
from core.mirror_io import read_mirror as _read
from backtest_continuity import _spearman, _boot_ci

SEED = 7
# Replay-style checkpoints: dense enough to mirror "pick any time", late enough
# that even the 60-min horizon resolves before the 15:30 close.
SAMPLE_TIMES = ["09:45", "10:00", "10:15", "10:30", "10:45", "11:00", "11:15",
                "11:30", "11:45", "12:00", "12:15", "12:30", "12:45", "13:00",
                "13:15", "13:30", "13:45", "14:00", "14:15", "14:30"]
HORIZONS = (15, 30, 60)          # minutes forward
LOOKBACK_MIN = 30                # window to detect a fresh g sign-flip (crossover)
DEADBAND = 0.02                  # |g| below this = no structural call (flat book)
W_BASIS = 0.20                   # futures-basis weight in the structural composite
LEDGER = DATA_DIR / "validation" / "crossover_ledger.parquet"


def _captured_days() -> list[str]:
    out = set()
    for p in LIVE_DIR.glob("*_oi_snapshots.parquet"):
        if p.stat().st_size < 1024:                 # skip clobbered stubs (~508B)
            continue
        stem = p.name.split("_")[0]
        try:
            datetime.date.fromisoformat(stem)
            out.add(stem)
        except ValueError:
            continue
    return sorted(out)


def _last_at(df: pd.DataFrame, t, col: str):
    """Latest value of `col` at or before t (lookahead-free), or None."""
    if df is None or col not in df.columns:
        return None
    s = df[df["ts"] <= pd.Timestamp(t)]
    if not len(s):
        return None
    v = s.iloc[-1][col]
    try:
        v = float(v)
        return None if v != v else v
    except (TypeError, ValueError):
        return None


def _g_at(oi: pd.DataFrame, t):
    """OI tilt g=(putOI-callOI)/(putOI+callOI) at instant t. None if no book."""
    c = _last_at(oi, t, "total_call_oi")
    p = _last_at(oi, t, "total_put_oi")
    if c is None or p is None or (c + p) <= 0:
        return None
    return (p - c) / (p + c)


def _spot_at(ticks: pd.DataFrame, t):
    s = ticks[ticks["ts"] <= pd.Timestamp(t)]
    return float(s.iloc[-1]["ltp"]) if len(s) else None


def harvest(days: list[str]) -> pd.DataFrame:
    rows = []
    for date in days:
        d0 = datetime.date.fromisoformat(date)
        for sym in INDEX_SYMBOLS:
            oi = _read("oi_snapshots", date, None, sym)
            ticks = _read("ticks", date, None, sym)
            fut = _read("futures_quotes", date, None, sym)
            if oi is None or "total_call_oi" not in oi.columns or ticks is None or len(ticks) < 20:
                continue
            for hhmm in SAMPLE_TIMES:
                hh, mm = map(int, hhmm.split(":"))
                t = datetime.datetime.combine(d0, datetime.time(hh, mm), tzinfo=IST)
                g = _g_at(oi, t)
                spot_t = _spot_at(ticks, t)
                if g is None or not spot_t:
                    continue
                g_prev = _g_at(oi, t - datetime.timedelta(minutes=LOOKBACK_MIN))
                # crossover = sign flip of g within the lookback window
                crossed = (g_prev is not None and np.sign(g) != np.sign(g_prev)
                           and abs(g) > DEADBAND and abs(g_prev) > DEADBAND)
                basis = _last_at(fut, t, "near_basis")
                # transparent structural composite (NOT fit): floor-tilt ⊕ basis sign
                comp = g + (W_BASIS * np.tanh((basis or 0.0) / max(abs(spot_t) * 1e-4, 1.0)))
                call = 1 if comp > DEADBAND else -1 if comp < -DEADBAND else 0
                base = {
                    "date": date, "sym": sym, "t": hhmm, "spot": spot_t,
                    "g": g, "g_prev": (g_prev if g_prev is not None else np.nan),
                    "crossed": int(bool(crossed)),
                    "flip_dir": int(np.sign(g)) if crossed else 0,
                    "basis": (basis if basis is not None else np.nan),
                    "comp": comp, "call": call,
                }
                for H in HORIZONS:
                    s_end = _spot_at(ticks, t + datetime.timedelta(minutes=H))
                    ret = (s_end / spot_t - 1.0) if (s_end and s_end != spot_t) else np.nan
                    base[f"ret{H}"] = ret
                rows.append(base)
    return pd.DataFrame(rows)


def _upsert(new: pd.DataFrame) -> pd.DataFrame:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    comb = pd.concat([pd.read_parquet(LEDGER), new], ignore_index=True) if LEDGER.exists() else new
    comb = comb.drop_duplicates(subset=["date", "sym", "t"], keep="last").reset_index(drop=True)
    comb.to_parquet(LEDGER, index=False)
    return comb


def evaluate(df: pd.DataFrame, reps: int, rng) -> None:
    print("\nCE/PE OI CROSSOVER — is it worth anything? (day-block CV vs nulls)")
    print("=" * 72)
    if df.empty:
        print("  ledger empty — run a harvest first."); return
    days = sorted(df.date.unique())
    print(f"  rows={len(df)}  days={len(days)} ({days[0]}..{days[-1]})  indices={df.sym.nunique()}")

    # ── 1. STANDING TILT: does g predict the forward move? ──────────────────────
    print("\n  1) STANDING TILT  IC( g , fwd_ret )   [EDGE only if CI clears 0]")
    for H in HORIZONS:
        sub = df.dropna(subset=[f"ret{H}"])
        if len(sub) < 10 or sub.date.nunique() < 2:
            print(f"     {H:>2}m   n={len(sub):<4} insufficient"); continue
        dts = sub["date"].to_numpy()
        ic, lo, hi = _boot_ci(_spearman, sub["g"].to_numpy(float), sub[f"ret{H}"].to_numpy(float),
                              reps=reps, rng=rng, groups=dts)
        v = "EDGE" if lo > 0 else ("anti" if hi < 0 else "—")
        print(f"     {H:>2}m   IC {ic:+.3f} [{lo:+.3f},{hi:+.3f}]  {v}  (n={len(sub)})")

    # ── 2. CROSSOVER EVENT: forward move right after g flips sign ────────────────
    print("\n  2) CROSSOVER EVENT  mean fwd_ret after a sign-flip vs base drift")
    ev = df[df.crossed == 1]
    print(f"     events={len(ev)} of {len(df)} checkpoints ({100*len(ev)/max(len(df),1):.0f}%)")
    for H in HORIZONS:
        e = ev.dropna(subset=[f"ret{H}"])
        b = df.dropna(subset=[f"ret{H}"])
        if len(e) < 5:
            print(f"     {H:>2}m   n={len(e):<3} too few events"); continue
        # signed by flip direction: bullish flip should yield +ret if the signal works
        signed = (np.sign(e["flip_dir"].to_numpy()) * e[f"ret{H}"].to_numpy(float)) * 1e4  # bps
        base_bps = float(b[f"ret{H}"].abs().mean() * 1e4)
        if e.date.nunique() >= 2:
            m, lo, hi = _boot_ci(lambda a: a.mean(), signed, reps=reps, rng=rng,
                                 groups=e["date"].to_numpy())
            v = "EDGE" if lo > 0 else ("anti" if hi < 0 else "—")
            print(f"     {H:>2}m   dir-signed {m:+6.1f}bps [{lo:+6.1f},{hi:+6.1f}] {v}  "
                  f"(base |move| {base_bps:.0f}bps, n={len(e)})")
        else:
            print(f"     {H:>2}m   {signed.mean():+6.1f}bps (1 day, no CI, n={len(e)})")

    # ── 3. SCOREBOARD: the replay workflow — call at t, did it hit? ──────────────
    print("\n  3) SCOREBOARD  structural call = sign(g ⊕ basis); hit if matches fwd move")
    H = 30
    sc = df[(df.call != 0)].dropna(subset=[f"ret{H}"]).copy()
    if len(sc) >= 5 and sc.date.nunique() >= 2:
        sc["hit"] = (np.sign(sc["call"]) == np.sign(sc[f"ret{H}"])).astype(float)
        print(f"     per-day (H={H}m):")
        for d, grp in sc.groupby("date"):
            w = int(grp.hit.sum()); n = len(grp)
            print(f"       {d}   {w:>2}/{n:<2}  {100*w/n:4.0f}%")
        hit = sc["hit"].to_numpy()
        hr, lo, hi = _boot_ci(lambda a: a.mean(), hit, reps=reps, rng=rng,
                              groups=sc["date"].to_numpy())
        v = "EDGE" if lo > 0.5 else ("anti" if hi < 0.5 else "—")
        print(f"     POOLED  {100*hr:.1f}% [{100*lo:.1f},{100*hi:.1f}]  {v}  "
              f"(n={len(sc)} calls vs 50% null)")
    else:
        print(f"     n={len(sc)} insufficient for a scoreboard")

    print("\n" + "=" * 72)
    print("READ: day-block 95% CI is wide with few days — that is honest, not a bug.")
    print("Crossover earns a place in the cockpit ONLY if (1) IC-CI clears 0, (2) the")
    print("event fwd-move CI clears 0, or (3) scoreboard hit-CI clears 50% — OUT of")
    print("sample. Until then it is positioning CONTEXT (a floor/ceiling read), not a call.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3000)
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    if args.eval_only:
        if not LEDGER.exists():
            print("No ledger yet — run without --eval-only first."); return
        evaluate(pd.read_parquet(LEDGER), args.reps, rng); return

    days = _captured_days()
    if not days:
        print("No captured oi_snapshot days in", LIVE_DIR); return
    print(f"reading mirror: {LIVE_DIR}")
    new = harvest(days)
    print(f"harvested {len(new)} rows from {len(days)} days -> upserting {LEDGER.name}")
    comb = _upsert(new) if not new.empty else (pd.read_parquet(LEDGER) if LEDGER.exists() else new)
    evaluate(comb, args.reps, rng)


if __name__ == "__main__":
    main()
