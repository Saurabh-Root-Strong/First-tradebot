"""
backtest_scout_trades.py — TRADE simulation of the LIVE scout, exit = first band touch.

Grades the engine EXACTLY as the live scout-alert ledger runs it: tf=60, horizon=60,
ONE stance per index, opened the moment the 60m board turns to a directional TRADE for an
index that is not already holding one. Then applies the user's exit rule:

    from just after entry, the trade CLOSES on the FIRST intrabar touch of EITHER 60m band
    edge (pred_lo / pred_hi). Favourable edge first = TARGET. Adverse edge first = SL.
    Neither edge inside 60m = TIMEOUT (marked win/loss by the sign of the final move).

For a CE (UP) call  : hi = target, lo = SL.
For a PE (DOWN) call : lo = target, hi = SL.

Two P&L views per trade:
  • INDEX  — signed favourable spot move % to the exit (the barrier itself).
  • OPTION — buy the ATM CE/PE at entry premium, sell at the exit instant, NET of a
             _OPT_RT_COST round-trip. THIS is the money grade (spread+theta included).

Lookahead-free: the scout call at t reads mirrors with ts<=t (with_lifecycle=False so it is
fast + I own the exit); ticks AFTER t are the answer key for the barrier only, never fed back.

    .venv\\Scripts\\python.exe backtest_scout_trades.py
    .venv\\Scripts\\python.exe backtest_scout_trades.py --tf 15 --step 5 --reps 5000
"""
from __future__ import annotations

import argparse
import datetime
import sys

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from core.constants import IST, INDEX_SYMBOLS, LIVE_DIR, NIFTY
from core.mirror_io import read_mirror as _read
from core import market_calendar as _mktcal
from backtest_continuity import _boot_ci
import intraday_scout as scout

SEED = 7
HORIZON = 60                       # minutes — the "60-minute basis"
FIRST_ENTRY = datetime.time(9, 35)
LAST_ENTRY = datetime.time(14, 30)  # keep a full ~60m runway before 15:30


def _target_week(n: int = 8) -> list[str]:
    """Most recent n captured sessions with BOTH a full tick mirror and a full chain mirror."""
    days = []
    for p in sorted(LIVE_DIR.glob("*_ticks.parquet")):
        stem = p.name.split("_")[0]
        try:
            datetime.date.fromisoformat(stem)
        except ValueError:
            continue
        if p.stat().st_size < 200_000:                       # skip sparse/dead-capture days
            continue
        ch = LIVE_DIR / f"{stem}_chain_snapshots.parquet"
        if not ch.exists() or ch.stat().st_size < 200_000:   # need option chain for P&L
            continue
        days.append(stem)
    return days[-n:]


def _dte(sym: str, day: str) -> int:
    """Calendar days-to-expiry of the arrow's option on `day` — the theta axis (near 0 the
    ATM premium decays fastest, theta ~ 1/sqrt(T)). Thin wrapper over the canonical
    market_calendar.days_to_expiry (weekly for NIFTY, monthly for the rest)."""
    return _mktcal.days_to_expiry(day, weekly=(sym == NIFTY))


def _dte_bkt(dte: int) -> str:
    if dte < 0:
        return "?"
    if dte == 0:
        return "0 EXPIRY"
    if dte <= 2:
        return "1-2 near"
    if dte <= 7:
        return "3-7 mid"
    return "8+ far"


def _grid(day: str, step_min: int) -> list[datetime.datetime]:
    d0 = datetime.date.fromisoformat(day)
    t = datetime.datetime.combine(d0, FIRST_ENTRY, tzinfo=IST)
    end = datetime.datetime.combine(d0, LAST_ENTRY, tzinfo=IST)
    out = []
    while t <= end:
        out.append(t)
        t += datetime.timedelta(minutes=step_min)
    return out


def _resolve_exit(ticks: pd.DataFrame, t, entry: float, lo: float, hi: float,
                  direction: str, max_hold: int = HORIZON) -> dict:
    """First-touch barrier exit on the tick path in (t, t+max_hold]. Barrier = the fixed
    60m band edges (the validated product); max_hold is a SEPARATE hard time cap = the
    user's 'close on band/SL/target ELSE timeout at max_hold' rule. Returns exit instant,
    exit spot, outcome (TARGET/SL/TIMEOUT_WIN/TIMEOUT_LOSS), minutes held."""
    t_end = t + datetime.timedelta(minutes=max_hold)
    w = ticks[(ticks["ts"] > pd.Timestamp(t)) & (ticks["ts"] <= pd.Timestamp(t_end))]
    if not len(w):
        return {}
    up = direction == "CE"
    target, sl = (hi, lo) if up else (lo, hi)
    hit_hi = w[w["ltp"] >= hi]
    hit_lo = w[w["ltp"] <= lo]
    t_hi = hit_hi["ts"].iloc[0] if len(hit_hi) else None
    t_lo = hit_lo["ts"].iloc[0] if len(hit_lo) else None
    tt = t_hi if target == hi else t_lo          # first touch of the TARGET edge
    ts_ = t_hi if sl == hi else t_lo             # first touch of the SL edge
    if tt is None and ts_ is None:               # neither edge in 60m → TIMEOUT
        last = w.iloc[-1]
        exit_spot = float(last["ltp"])
        fav = (exit_spot - entry) * (1 if up else -1)
        return {"exit_t": last["ts"], "exit_spot": exit_spot,
                "outcome": "TIMEOUT_WIN" if fav > 0 else "TIMEOUT_LOSS",
                "held": (last["ts"] - pd.Timestamp(t)).total_seconds() / 60.0}
    # whichever edge is touched first (target on ties = favourable to the call)
    if ts_ is None or (tt is not None and tt <= ts_):
        return {"exit_t": tt, "exit_spot": float(target), "outcome": "TARGET",
                "held": (tt - pd.Timestamp(t)).total_seconds() / 60.0}
    return {"exit_t": ts_, "exit_spot": float(sl), "outcome": "SL",
            "held": (ts_ - pd.Timestamp(t)).total_seconds() / 60.0}


# hold horizons for the THETA-DECAY CURVE (minutes). ≤120 = the user's 2-hour ceiling.
_HOLDS = (20, 30, 45, 60, 75, 90, 120)
# MAX-HOLD CAPS to sweep (minutes) — band/SL/target exit ELSE timeout at cap. 135 = the
# user's proposed 2h15m ceiling; 45/60/90/120 bracket it to find the least-bad cap.
_CAPS = (45, 60, 90, 120, 135)


def _exit_variants(sym, day, t, atm, d, e_prem) -> dict:
    """Option-net under alternative EXIT policies, same entry. Tests whether the theta
    bleed on the 56% timeouts is fixable by cutting earlier / options-native stops:
      • tsNN — hard TIME-STOP at NN minutes (exit regardless of band); NN in _HOLDS.
               The tsNN series across _HOLDS IS the theta-decay curve (buy-and-hold the
               ATM option NN minutes, net of cost) — answers "how long can I hold?".
      • pbar — PREMIUM barrier: TP +45% / SL -25% on the option, else time-stop 30m
               (options-native, walks the premium path at 5-min steps)
    All net of one _OPT_RT_COST round-trip. nan when premiums missing."""
    cost = scout._OPT_RT_COST
    out = {f"ts{m}": np.nan for m in _HOLDS}
    out["pbar"] = np.nan
    if not e_prem:
        return out
    for m in _HOLDS:
        x = scout._opt_premium(sym, day, t + datetime.timedelta(minutes=m), atm, d)
        if x:
            out[f"ts{m}"] = (x / e_prem - 1.0) * 100.0 - cost
    # premium barrier — first 5-min checkpoint that breaches TP/SL, else 30m time-stop
    tp, sl, last = 45.0, -25.0, np.nan
    for m in range(5, 31, 5):
        x = scout._opt_premium(sym, day, t + datetime.timedelta(minutes=m), atm, d)
        if not x:
            continue
        g = (x / e_prem - 1.0) * 100.0
        last = g
        if g >= tp or g <= sl:
            last = g
            break
    out["pbar"] = (last - cost) if last == last else np.nan
    return out


def simulate(days: list[str], tf: int, step: int) -> pd.DataFrame:
    rows = []
    for day in days:
        for sym in INDEX_SYMBOLS:
            ticks = _read("ticks", day, None, sym)
            if ticks is None or len(ticks) < 50:
                continue
            open_until = None            # one stance per index — re-enter only after exit
            for t in _grid(day, step):
                if open_until is not None and t <= open_until:
                    continue
                open_until = None
                r = scout.scan_index(sym, tf, date=day, as_of=t, horizon_min=HORIZON,
                                     with_lifecycle=False)
                if not r.get("has_data") or not r["verdict"].startswith("TRADE"):
                    continue
                d = r.get("direction")
                lo, hi, spot, atm = (r.get("pred_lo"), r.get("pred_hi"),
                                     r.get("spot"), r.get("atm"))
                if d not in ("CE", "PE") or lo is None or hi is None or not spot:
                    continue
                ex = _resolve_exit(ticks, t, spot, lo, hi, d)
                if not ex:
                    continue
                up = d == "CE"
                idx_ret = (ex["exit_spot"] / spot - 1.0) * 100.0 * (1 if up else -1)
                # Did the index actually GO anywhere in the first 60m? peak-to-trough
                # realized range as % of spot. <0.25% = range-bound/dead → the pure
                # theta-bleed case the user is worried about (hold a flat index → decay).
                w60 = ticks[(ticks["ts"] > pd.Timestamp(t)) &
                            (ticks["ts"] <= pd.Timestamp(t + datetime.timedelta(minutes=60)))]
                rng60 = round((w60["ltp"].max() - w60["ltp"].min()) / spot * 100, 3) \
                    if len(w60) else np.nan
                flat = bool(rng60 == rng60 and rng60 < 0.25)
                # CAUSAL 30m realized range — known AT t+30m (unlike `flat`, whose 60m window
                # peeks past a 30m decision). Feeds the mid-trade BLEED-EXIT policy test:
                # momentum stalls -> consolidation -> theta eats the premium.
                w30 = ticks[(ticks["ts"] > pd.Timestamp(t)) &
                            (ticks["ts"] <= pd.Timestamp(t + datetime.timedelta(minutes=30)))]
                rng30 = round((w30["ltp"].max() - w30["ltp"].min()) / spot * 100, 3) \
                    if len(w30) else np.nan
                # option P&L — the honest money grade (band-touch exit)
                e_prem = scout._opt_premium(sym, day, t, atm, d)
                x_prem = scout._opt_premium(sym, day, ex["exit_t"].to_pydatetime(), atm, d)
                opt_net = (((x_prem / e_prem - 1.0) * 100.0 - scout._OPT_RT_COST)
                           if (e_prem and x_prem) else np.nan)
                # ── EXIT-POLICY VARIANTS (the monetization test) — same entry, diff exit ──
                var = _exit_variants(sym, day, t, atm, d, e_prem)
                # MAX-HOLD SWEEP — the user's exact rule (band/SL/target exit ELSE hard
                # timeout at the cap), band fixed at 60m, cap swept. Same entry; each cap
                # just moves the timeout window. opt-net per cap = the money grade.
                caps = {}
                for cap in _CAPS:
                    cx = _resolve_exit(ticks, t, spot, lo, hi, d, max_hold=cap)
                    xp = (scout._opt_premium(sym, day, cx["exit_t"].to_pydatetime(), atm, d)
                          if cx else None)
                    caps[f"cap{cap}"] = (((xp / e_prem - 1.0) * 100.0 - scout._OPT_RT_COST)
                                         if (e_prem and xp) else np.nan)
                rg = r.get("regime") or {}
                dte = _dte(sym, day)
                rows.append({
                    "date": day, "sym": sym, "entry_t": t.strftime("%H:%M"),
                    "entry_hr": t.hour, "dte": dte, "dte_bkt": _dte_bkt(dte),
                    "dir": d, "strength": r.get("strength"),
                    "spot": round(spot, 2), "lo": lo, "hi": hi, "atm": atm,
                    "rng30": rng30,
                    "band_pct": round((hi - lo) / 2 / spot * 100, 3),
                    "outcome": ex["outcome"], "held_min": round(ex["held"], 1),
                    "idx_ret": round(idx_ret, 3), "rng60": rng60, "flat": flat,
                    "opt_entry": e_prem, "opt_net": opt_net, **var, **caps,
                    "mood": r.get("mood_full") or "", "trend": rg.get("trend") or "",
                })
                open_until = ex["exit_t"]      # block this index until the trade closes
    return pd.DataFrame(rows)


# ── reporting ─────────────────────────────────────────────────────────────────
def _pf(net: np.ndarray) -> float:
    g = net[net > 0].sum(); l = -net[net < 0].sum()
    return float(g / l) if l > 0 else float("inf")


def _max_consec_loss(net: np.ndarray) -> int:
    m = c = 0
    for x in net:
        c = c + 1 if x <= 0 else 0
        m = max(m, c)
    return m


def _block(df: pd.DataFrame, label: str, rng, reps: int) -> None:
    n = len(df)
    if not n:
        print(f"  {label:22s} n=0"); return
    oc = df["outcome"].value_counts().to_dict()
    tgt = oc.get("TARGET", 0); sl = oc.get("SL", 0)
    tw = oc.get("TIMEOUT_WIN", 0); tl = oc.get("TIMEOUT_LOSS", 0)
    wins = tgt + tw
    barrier = tgt / (tgt + sl) * 100 if (tgt + sl) else float("nan")
    on = df["opt_net"].dropna().to_numpy(float)
    opt_wr = (on > 0).mean() * 100 if len(on) else float("nan")
    opt_mean = on.mean() if len(on) else float("nan")
    shp = (on.mean() / on.std()) if (len(on) > 1 and on.std() > 0) else float("nan")
    print(f"  {label:22s} n={n:<3d} win {wins/n*100:4.0f}%  "
          f"TGT {tgt:<2d} SL {sl:<2d} TO {tw+tl:<2d}  barrier {barrier:4.0f}%  "
          f"opt: win {opt_wr:4.0f}% mean {opt_mean:+5.1f}% Sharpe {shp:+.2f}  "
          f"PF {_pf(on):.2f}")


def report(df: pd.DataFrame, rng, reps: int, tf: int, step: int, days: list[str]) -> None:
    print("\n" + "=" * 92)
    print("SCOUT TRADE SIM — live engine (tf=%dm, horizon=60m), exit = first band-edge touch"
          % tf)
    print("=" * 92)
    if df.empty:
        print("  no trades generated"); return
    n = len(df)
    print(f"  sessions={len(days)} ({days[0]}..{days[-1]})   indices={df.sym.nunique()}   "
          f"scan step={step}m   TRADES={n}  ({n/len(days):.1f}/day)")

    oc = df["outcome"].value_counts().to_dict()
    tgt = oc.get("TARGET", 0); sl = oc.get("SL", 0)
    tw = oc.get("TIMEOUT_WIN", 0); tl = oc.get("TIMEOUT_LOSS", 0)
    wins = tgt + tw
    print("\n  OUTCOMES (barrier accounting)")
    print(f"    TARGET hit  {tgt:3d} ({tgt/n*100:4.1f}%)      SL hit      {sl:3d} ({sl/n*100:4.1f}%)")
    print(f"    TIMEOUT win {tw:3d} ({tw/n*100:4.1f}%)      TIMEOUT loss {tl:3d} ({tl/n*100:4.1f}%)")
    print(f"    → overall WIN {wins}/{n} = {wins/n*100:.1f}%     "
          f"first-touch barrier (excl timeouts) {tgt}/{tgt+sl} = "
          f"{tgt/(tgt+sl)*100 if (tgt+sl) else float('nan'):.1f}%  (vs 50% null)")
    print(f"    band no-touch (RANGE held full 60m) = {(tw+tl)/n*100:.1f}%   "
          f"avg hold: TGT {df[df.outcome=='TARGET'].held_min.mean():.0f}m  "
          f"SL {df[df.outcome=='SL'].held_min.mean():.0f}m  all {df.held_min.mean():.0f}m")

    # day-block bootstrap CI — does it clear the null OOS?
    grp = df["date"].to_numpy()
    barr = (df["outcome"] == "TARGET").to_numpy(float)
    barr_mask = df["outcome"].isin(["TARGET", "SL"]).to_numpy()
    if barr_mask.sum() >= 5 and df[barr_mask].date.nunique() >= 2:
        bw, bl, bh = _boot_ci(lambda a: a.mean(), barr[barr_mask], reps=reps, rng=rng,
                              groups=grp[barr_mask])
        v = "EDGE" if bl > 0.5 else ("anti" if bh < 0.5 else "—")
        print(f"    barrier win-CI (day-block): {100*bw:.1f}% [{100*bl:.1f},{100*bh:.1f}] {v}")

    on_all = df.dropna(subset=["opt_net"])
    if len(on_all) >= 5 and on_all.date.nunique() >= 2:
        net = on_all["opt_net"].to_numpy(float)
        mw, ml, mh = _boot_ci(lambda a: a.mean(), net, reps=reps, rng=rng,
                              groups=on_all["date"].to_numpy())
        ev = "EDGE" if ml > 0 else ("bleed" if mh < 0 else "—")
        print("\n  OPTION P&L (buy ATM CE/PE at entry, sell at band-touch exit, net %.0f%% RT)"
              % scout._OPT_RT_COST)
        print(f"    trades w/ premium={len(net)}  win {100*(net>0).mean():.1f}%  "
              f"mean net {net.mean():+.1f}% [{ml:+.1f},{mh:+.1f}] {ev}")
        print(f"    median {np.median(net):+.1f}%   Sharpe/trade {net.mean()/net.std():+.2f}   "
              f"PF {_pf(net):.2f}   max-consec-loss {_max_consec_loss(net)}   "
              f"best {net.max():+.0f}% worst {net.min():+.0f}%")
        # daily aggregate Sharpe (equal size per trade)
        daily = on_all.groupby("date")["opt_net"].mean()
        if len(daily) > 1 and daily.std() > 0:
            print(f"    daily-mean net {daily.mean():+.1f}%  daily Sharpe {daily.mean()/daily.std():+.2f} "
                  f"(n_days={len(daily)})")

    # ── EXIT-POLICY SHOOTOUT — does any exit rescue the arrow to net-positive? ──────
    print("\n  EXIT-POLICY SHOOTOUT (same entries, mean option-net per exit rule)")
    pol = ([("band-touch", "opt_net")]
           + [(f"time-stop {m}m", f"ts{m}") for m in _HOLDS]
           + [("prem-barrier TP45/SL25", "pbar")])
    for name, col in pol:
        if col not in df.columns:
            continue
        v = df[col].dropna().to_numpy(float)
        if len(v) < 5:
            print(f"    {name:24s} n={len(v)} too few"); continue
        mw, ml, mh = _boot_ci(lambda a: a.mean(), v, reps=reps, rng=rng,
                              groups=df.dropna(subset=[col])["date"].to_numpy())
        shp = v.mean() / v.std() if v.std() > 0 else float("nan")
        ev = "EDGE" if ml > 0 else ("bleed" if mh < 0 else "flat")
        print(f"    {name:24s} mean {v.mean():+5.1f}% [{ml:+5.1f},{mh:+5.1f}] {ev:5s}  "
              f"win {100*(v>0).mean():4.0f}%  Sharpe {shp:+.2f}  PF {_pf(v):.2f}")

    print("\n  BY INDEX")
    for s in INDEX_SYMBOLS:
        _block(df[df.sym == s], s, rng, reps)
    print("\n  BY DIRECTION")
    for d in ("CE", "PE"):
        _block(df[df.dir == d], d, rng, reps)
    print("\n  BY SESSION")
    for day in days:
        _block(df[df.date == day], day, rng, reps)
    print("\n  BY ENTRY HOUR")
    for hr in sorted(df.entry_hr.unique()):
        _block(df[df.entry_hr == hr], f"{hr:02d}:xx", rng, reps)
    print("\n  BY MOOD (regime)")
    for m in sorted(x for x in df["mood"].unique() if x):
        _block(df[df["mood"] == m], m, rng, reps)

    # ── BY DTE — the THETA / expiry-proximity axis (mechanistic, not noise-mined) ──
    # theta ~ 1/sqrt(T): near expiry the ATM premium the arrow BUYS decays fastest (the
    # 'writers eat the premium' effect). Also the tiny-premium mirage — near-0 DTE gives
    # huge % swings on small ₹. NIFTY = weekly (DTE 0-4); BANK/FIN/MIDCAP = monthly (0-25).
    print("\n  BY DTE BUCKET (days-to-expiry — the theta axis)")
    for b in ("0 EXPIRY", "1-2 near", "3-7 mid", "8+ far", "?"):
        sub = df[df["dte_bkt"] == b]
        if len(sub):
            _block(sub, b, rng, reps)
    print("\n  BY DTE × INDEX  (expiry-proximity per index; weekly NIFTY vs monthly others)")
    for s in INDEX_SYMBOLS:
        for b in ("0 EXPIRY", "1-2 near", "3-7 mid", "8+ far"):
            sub = df[(df.sym == s) & (df.dte_bkt == b)]
            if len(sub):
                _block(sub, f"{s.split(':')[1][:10]:10s} {b}", rng, reps)

    # ── THETA-DECAY CURVE — buy ATM at entry, hold NN min, sell (net cost) ──────────
    # The tsNN columns ARE the decay curve. Split by whether the index actually moved:
    #   FLAT (peak-to-trough <0.25% in 60m) = the pure bleed case the user asked about.
    #   MOVED = the index went somewhere → is there a hold that banks it before decay?
    print("\n  THETA-DECAY / HOLD CURVE  (mean option-net %, buy ATM & hold NN min)")
    print(f"    {'hold':>6s} | {'ALL':>18s} | {'FLAT idx (<0.25%)':>20s} | {'MOVED idx':>18s}")
    sub_all, sub_flat, sub_mv = df, df[df["flat"]], df[~df["flat"].astype(bool)]

    def _cell(s, col):
        v = s[col].dropna().to_numpy(float) if col in s else np.empty(0)
        if len(v) < 3:
            return f"{'n<3':>18s}"
        return f"n={len(v):<3d} {v.mean():+5.1f}% W{100*(v>0).mean():3.0f}%"
    for m in _HOLDS:
        c = f"ts{m}"
        print(f"    {m:>4d}m  | {_cell(sub_all,c):>18s} | {_cell(sub_flat,c):>20s} | "
              f"{_cell(sub_mv,c):>18s}")
    print(f"    FLAT trades = {len(sub_flat)}/{len(df)} ({100*len(sub_flat)/max(len(df),1):.0f}%) — "
          "index went nowhere in 60m; each extra hold minute is pure premium bleed.")

    # ── MAX-HOLD CAP SWEEP — the user's rule: band/SL/target ELSE timeout at cap ────
    print("\n  MAX-HOLD CAP SWEEP  (band/SL/target exit ELSE hard timeout at cap; band=60m)")
    print(f"    {'cap':>6s} | {'ALL':>20s} | {'FLAT idx':>18s} | {'MOVED idx':>18s}   "
          "(mean option-net %, net 3% RT)")
    for cap in _CAPS:
        c = f"cap{cap}"
        tag = "  <- your 2h15m" if cap == 135 else ""
        print(f"    {cap:>4d}m  | {_cell(df,c):>20s} | {_cell(df[df['flat']],c):>18s} | "
              f"{_cell(df[~df['flat'].astype(bool)],c):>18s}{tag}")

    # ── BLEED-EXIT POLICY — the user's momentum→consolidation→theta-eat pattern ─────
    # CAUSAL rule, decided AT entry+30m with only past data: if the index's realized range
    # over the first 30m < THRESH (consolidation) AND the premium is underwater (gross<0 ⇔
    # theta already eating), EXIT at t+30m; otherwise hold to band/SL/target else 90m cap.
    # Baseline = plain cap90. Positive delta = the early bleed-exit SAVES money.
    print("\n  BLEED-EXIT POLICY  (at +30m: flat idx AND premium underwater -> exit early)")
    print(f"    {'thresh':>7s} | {'n_exited':>8s} | {'policy mean':>11s} | {'cap90 mean':>10s} | "
          f"{'delta':>6s}   (option-net %, same entries)")
    base = df["cap90"]
    for th in (0.10, 0.15, 0.20, 0.25):
        bleed = (df["rng30"] < th) & ((df["ts30"] + scout._OPT_RT_COST) < 0)
        pol = np.where(bleed, df["ts30"], df["cap90"])
        m = pd.Series(pol).dropna()
        print(f"    <{th:.2f}% | {int(bleed.sum()):>8d} | {np.nanmean(pol):>+10.1f}% | "
              f"{base.mean():>+9.1f}% | {np.nanmean(pol)-base.mean():>+5.1f}pp")

    # ── INDEX × ENTRY-HOUR MATRIX — the dynamic time-performance grid (option-net) ──
    print("\n  INDEX × ENTRY-HOUR  (mean band-touch option-net %, n) — the time matrix")
    hrs = sorted(df.entry_hr.unique())
    print("    " + "index      " + "".join(f"{h:02d}:xx        " for h in hrs))
    for s in INDEX_SYMBOLS:
        cells = []
        for h in hrs:
            v = df[(df.sym == s) & (df.entry_hr == h)]["opt_net"].dropna().to_numpy(float)
            cells.append(f"{v.mean():+4.0f}%/{len(v):<2d}    " if len(v) else f"{'·':>6s}     ")
        print(f"    {s:10s} " + "".join(cells))

    print("\n" + "=" * 92)
    print("READ: barrier win% and option mean-net CI vs the null (50% / 0%) are the verdict.")
    print("Symmetric ±band barrier → a directionless path hits each edge ~equally; the engine")
    print("earns a trade only if barrier-win clears 50% AND option mean-net clears 0 OOS.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", type=int, default=60, help="signal timeframe fed to scan (live=60)")
    ap.add_argument("--step", type=int, default=5, help="scan grid minutes")
    ap.add_argument("--days", type=int, default=8, help="most-recent N captured sessions")
    ap.add_argument("--reps", type=int, default=5000)
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    days = _target_week(args.days)
    if not days:
        print("no usable captured sessions found"); return
    print(f"simulating {len(days)} sessions: {days}")
    df = simulate(days, args.tf, args.step)
    if not df.empty:
        out = Path_out()
        df.to_parquet(out, index=False)
        print(f"  {len(df)} trades -> {out}")
    report(df, rng, args.reps, args.tf, args.step, days)


def Path_out():
    from core.constants import DATA_DIR
    p = DATA_DIR / "validation" / "scout_trade_sim.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


if __name__ == "__main__":
    main()
