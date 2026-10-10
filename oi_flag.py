"""oi_flag.py — "a sudden heavy OI pile-up just started on one side", scored against sessions at
the SAME STAGE of a contract's life: today so far, the earlier days of this expiry, and — where
this expiry has no comparable day yet — the same stage of the previous ones.

    build(side, T) = change in that side's total OI over the last 15 minutes, % of that side
    pool           = earlier 15-minute readings from same-stage sessions (see pool_days) + today
    z              = (build − mean(pool)) / std(pool)            FLAG: z >= 2.5 and build > 0

WHAT IS TRACKED: the NEAREST expiry only — the broker's callOi / putOi for the front contract
(NIFTY weekly, the others monthly). The next contract is not in the feed, so a ROLL shows here
only as its near-side half: positions leaving the front month, never arriving in the next. The
morning after an expiry the totals simply ARE the new contract (NIFTY ~1,700L against ~4,100L
the evening before; FIN NIFTY 0.3L against 38L).

WHY "STAGE" AND NOT ONE BASELINE. Normal OI activity is not one number. The median 15-minute
build is 1.15% on the FIRST day of a contract (filling from a small base), 0.44% MID-cycle,
1.53% in the LATE 1–2 days and 2.08% on EXPIRY day. Two simpler yardsticks were built first and
replayed, and both failed:
  * "the same clock time on the last 20 sessions" mixes expiries — flags the same instants as a
    cycle-aware score only about half the time;
  * "this expiry so far" for every day leaves a FIRST day silent until 12:45 (nothing to compare
    with) and fires 2.07 times per index per day in the LATE days (BANK NIFTY 7.5) because they
    are judged against the calm middle of their own cycle.
Stage-aware pools: FIRST 1.17, MID 0.90, LATE 0.56, EXPIRY 0.34 fires per index per day, and
every stage can fire from 10:15.

WHAT THIS FLAG IS, AND IS NOT. It reports that the activity is happening, on which side, at which
strike, and whether premium rose or fell with it. It is NOT a forecast, and the label says so.
THIS module's own fires replayed over 76 sessions (148 with outcome data):

    after an unusual CALL pile-up   next hour ×1.37 of usual, a big hour 29% [16–41]
    after an unusual PUT  pile-up   next hour ×1.27,          a big hour 25% [15–36]
    any moment after 10:15          next hour ×1.25,          a big hour 24%
    direction after either          −3.2 bps [−7.6, +1.2], up 45% — a coin

i.e. about an ordinary hour: the intervals sit on top of the any-moment figure. (An earlier cut
read "quieter than usual" — that was the LATE-day over-firing, whose false alarms were followed
by ×0.82 hours.) From the research cuts of the same idea: the piled strike holds as a
ceiling/floor 68.6% of the time against 66.2% for any strike at that distance; buyers-vs-writers
makes no difference; and OI lines up with the price move 5 minutes EARLIER (+0.33), not the one
after (+0.13) — it answers price, it does not announce it. The reading that does carry forward
is range_flag (price). This one is context: "this is what the book is doing".

GUARDS
  * rows stamped another date are dropped (day files can open with the previous evening's row)
  * NO READING ACROSS A CAPTURE GAP: both ends of the 15-minute window need a fresh snapshot;
    otherwise the catch-up after an outage would read as a burst that never happened
  * the OPENING FLOOD (windows starting before 10:00) is its own pool — scored against the whole
    day it was a "burst" every single morning
  * fewer than 30 readings in the pool → silent
  * weekend / holiday (mock, special) sessions never enter a pool
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

WINDOW_MIN = 15           # the pile-up window
GRID_MIN = 5              # readings every 5 minutes
Z_FLAG = 2.5              # the threshold the 63-session test used
MIN_POOL = 30             # readings needed before "normal for this expiry" means anything
HOLD_MIN = 30             # a flag stays on the board this long after it fires
MAIN_START = 10 * 60      # OI floods in after the bell EVERY day: scored against the whole day,
                          # the 09:20→09:35 window read as a "burst" on 91 of 119 first-cut fires.
                          # So the opening (before 10:00) and the main session are separate pools.
STAGE_DAYS_BACK = 6       # same-stage days of EARLIER expiries used on a FIRST / LATE / EXPIRY day
MID_MIN_DAYS = 2          # a MID day borrows the previous expiry's MID days until it has this many
_OPEN_MIN, _CLOSE_MIN = 9 * 60 + 15, 15 * 60 + 30
_SIDES = (("CALL", "ce", "total_call_oi", "atm_call_prem", +1.0),
          ("PUT", "pe", "total_put_oi", "atm_put_prem", -1.0))
_COLS = ["ts", "symbol", "spot", "atm", "total_call_oi", "total_put_oi", "atm_call_prem", "atm_put_prem"]
# measured follow-through, quoted verbatim by the label: side -> (next-hour ×usual, % big hour)
MEASURED = {"CALL": (1.37, 29), "PUT": (1.27, 25)}
BASELINE = (1.25, 24)


def builds(snap: "pd.DataFrame | None", day: "str | None" = None) -> "pd.DataFrame":
    """15-minute change in each side's total OI (% of that side) on a 5-minute grid for ONE
    session. Indexed by minute-of-day; columns ce, pe, plus the fields the label needs
    (spot, atm, premiums and their 15-minute-ago values). Rows are NaN — not zero — wherever
    either end of the window has no fresh snapshot."""
    if snap is None or not len(snap) or "total_call_oi" not in snap.columns:
        return pd.DataFrame()
    s = snap[(snap["total_call_oi"] > 0) & (snap["total_put_oi"] > 0)]
    if not len(s):
        return pd.DataFrame()
    d = dt.date.fromisoformat(day) if day else s["ts"].dt.date.max()
    mins = s["ts"].dt.hour * 60 + s["ts"].dt.minute
    s = s[(s["ts"].dt.date == d) & (mins >= _OPEN_MIN) & (mins <= _CLOSE_MIN)]
    if len(s) < 10:
        return pd.DataFrame()
    keep = [c for c in _COLS if c in s.columns and c not in ("ts", "symbol")]
    r = s.set_index("ts")[keep].resample(f"{GRID_MIN}min", label="right", closed="right").last()
    p = r.shift(WINDOW_MIN // GRID_MIN)          # NOT forward-filled: a gap stays a gap
    out = pd.DataFrame({
        "ce": (r["total_call_oi"] - p["total_call_oi"]) / p["total_call_oi"] * 100.0,
        "pe": (r["total_put_oi"] - p["total_put_oi"]) / p["total_put_oi"] * 100.0,
    })
    for c in ("spot", "atm", "atm_call_prem", "atm_put_prem"):
        if c in r.columns:
            out[c] = r[c]
            out[c + "_was"] = p[c]
    out.index = r.index.hour * 60 + r.index.minute
    return out


def _phase(minute) -> bool:
    """True = the main session, False = the opening flood. Decided by where the 15-minute window
    STARTS: the 10:00 reading covers 09:45–10:00 and is still opening flow (keyed on its end,
    95 of 192 fires landed at 10:00–10:10)."""
    return int(minute) - WINDOW_MIN >= MAIN_START


def score(today: "pd.Series", prior, minute: int) -> "dict | None":
    """Latest pile-up on one side at-or-before `minute` that is still inside its HOLD window.
    today = that side's builds for the session (minute-indexed). prior = earlier readings of the
    pool: a minute-indexed Series (so they can be split by session phase), or a bare array,
    which is taken as main-session readings. Each of today's readings is scored against prior +
    today's EARLIER readings OF THE SAME PHASE — the opening is never the yardstick for midday,
    nor midday for the opening."""
    t = today[today.index <= minute].dropna()
    if isinstance(prior, pd.Series):
        pv = prior.dropna()
        ph = np.array([_phase(m) for m in pv.index], dtype=bool)
        prior_by = {True: pv.to_numpy(dtype=float)[ph], False: pv.to_numpy(dtype=float)[~ph]}
    else:
        a = np.asarray(prior, dtype=float)
        prior_by = {True: a[np.isfinite(a)], False: np.empty(0)}
    hit = None
    vals = t.to_numpy()
    tph = np.array([_phase(m) for m in t.index], dtype=bool)
    for i, (m, x) in enumerate(zip(t.index, vals)):
        if minute - int(m) > HOLD_MIN or x <= 0:
            continue
        same = tph[:i] == tph[i]
        pool = np.concatenate([prior_by[bool(tph[i])], vals[:i][same]])
        if len(pool) < MIN_POOL:
            continue
        sd = float(pool.std())
        if sd <= 0:
            continue
        z = (float(x) - float(pool.mean())) / sd
        if z >= Z_FLAG:
            typ = float(np.median(np.abs(pool)))
            hit = {"minute": int(m), "build": float(x), "z": round(z, 1),
                   "ratio": round(float(x) / typ, 1) if typ > 0 else None,
                   "live": int(m) == int(t.index[-1]), "pool": int(len(pool))}
    return hit


def stage(day: str, weekly: bool) -> str:
    """Where a session sits in the life of the contract being tracked (the NEAREST expiry):
        EXPIRY  the contract's last day
        FIRST   the first session of a new contract (the previous session was an expiry)
        LATE    1–2 days to go
        MID     everything else
    OI is a different animal at each. Measured over 76 sessions, the normal 15-minute build is
    1.15% on a FIRST day (a new contract filling from a small base — FIN NIFTY opens at 0.3L
    against 7.7L mid-cycle), 0.44% MID, 1.53% LATE and 2.08% on EXPIRY day."""
    from core import market_calendar as mc
    dte = mc.days_to_expiry(day, weekly=weekly)
    if dte == 0:
        return "EXPIRY"
    prev = mc.prev_trading_day(dt.date.fromisoformat(day))
    if mc.days_to_expiry(prev.isoformat(), weekly=weekly) == 0:
        return "FIRST"
    return "LATE" if dte <= 2 else "MID"


def pool_days(captured: list, ref_date: str, weekly: bool) -> list:
    """Earlier sessions whose readings are the yardstick for ref_date: sessions at the SAME
    STAGE of a contract's life.

      MID     the earlier MID days of THIS expiry ("today and the days since the expiry started").
              Topped up from the previous expiry's MID days only while this one has fewer than
              MID_MIN_DAYS of its own, so day 2 of a contract is not left without a yardstick.
      FIRST / LATE / EXPIRY
              the same stage of the previous expiries (plus an earlier same-stage day of this
              one, when there is one).

    WHY NOT SIMPLY "this expiry so far" everywhere — that was the first cut, and replayed it
    failed at both ends: a FIRST day has no earlier days at all, so the flag sat silent until
    12:45; and a LATE day was judged against the calm middle of its own cycle, so it fired 2.07
    times per index per day (BANK NIFTY 7.5) against 0.65 mid-cycle — a calendar alarm, not an
    activity alarm."""
    from core import market_calendar as mc
    st = stage(ref_date, weekly)
    exp_of = lambda d: dt.date.fromisoformat(d) + dt.timedelta(days=mc.days_to_expiry(d, weekly=weekly))
    exp = exp_of(ref_date)
    prev = sorted(d for d in captured if d < ref_date)
    same = [d for d in prev if stage(d, weekly) == st]
    mine = [d for d in same if exp_of(d) == exp]
    others = [d for d in same if exp_of(d) != exp]
    if st == "MID":
        need = max(0, MID_MIN_DAYS - len(mine))
        return (others[-need:] if need else []) + mine
    return (others[-STAGE_DAYS_BACK:] + mine)


# ── I/O ────────────────────────────────────────────────────────────────────────
_DAY_BUILDS: dict = {}      # day -> {sym: builds frame} for FINISHED sessions (immutable)


def _load(day: str) -> "pd.DataFrame | None":
    """One finished session's oi_snapshots, only the needed columns (day_files.load: the parquet
    mirror on the laptop, the per-day DuckDB on the VM). Kept out of the shared past-day cache in
    core.mirror_io — same reason as range_flag._load_ticks."""
    import day_files
    return day_files.load("oi_snapshots", day, _COLS)


def _captured() -> list:
    """Captured TRADING sessions that have OI snapshots (mirror or per-day database)."""
    import day_files
    from core import market_calendar as mc
    return [d for d in day_files.days("oi_snapshots", 20_000)
            if mc.is_trading_day(dt.date.fromisoformat(d))]


def _day(day: str, sym: str) -> "pd.DataFrame":
    if day not in _DAY_BUILDS:
        t = _load(day)
        _DAY_BUILDS[day] = ({s: builds(g, day) for s, g in t.groupby("symbol")}
                            if t is not None and "symbol" in t.columns else {})
        while len(_DAY_BUILDS) > 120:
            _DAY_BUILDS.pop(next(iter(_DAY_BUILDS)))
    return _DAY_BUILDS[day].get(sym, pd.DataFrame())


def _top_strike(sym: str, ref_date: str, side: str, t_now, spot: float) -> "tuple | None":
    """(strike, OI added in the window) for the strike that took the most of the pile-up, among
    strikes within 3% of spot on the nearest expiry. Naming only — needs no baseline."""
    try:
        from core.mirror_io import read_mirror
        ch = read_mirror("chain_snapshots", ref_date, t_now, sym)
        if ch is None or not len(ch):
            return None
        if "expiry" in ch.columns:
            ex = sorted({int(e) for e in ch["expiry"].dropna().tolist() if int(e) > 0})
            if ex:
                ch = ch[ch["expiry"] == ex[0]]
        ch = ch[(ch["side"] == ("CE" if side == "CALL" else "PE"))
                & (ch["strike"] >= spot * 0.97) & (ch["strike"] <= spot * 1.03)]
        a = ch[ch["ts"] <= t_now]
        b = ch[ch["ts"] <= t_now - pd.Timedelta(minutes=WINDOW_MIN)]
        if not len(a) or not len(b):
            return None
        a = a[a["ts"] == a["ts"].max()].groupby("strike")["oi"].last()
        b = b[b["ts"] == b["ts"].max()].groupby("strike")["oi"].last()
        dlt = (a - b).dropna()
        dlt = dlt[dlt > 0]
        if not len(dlt):
            return None
        return int(dlt.idxmax()), float(dlt.max()), float(dlt.max() / dlt.sum())
    except Exception:
        return None


def read(sym: str, ref_date: str, as_of: "dt.datetime | None" = None, weekly: "bool | None" = None) -> list:
    """Pile-ups to show for one index right now: a list of 0–2 dicts (CALL and/or PUT)
        {'side','hm','build','z','ratio','live','strike','strike_add','strike_share','kind','pool'}
    kind = 'buyers' (that side's ATM premium rose beyond what the index move explains),
           'writers' (it fell), or None when the ATM strike changed inside the window."""
    try:
        from core.constants import IST, NIFTY, today_iso
        if weekly is None:
            weekly = (sym == NIFTY)
        if ref_date == today_iso():
            from core.mirror_io import read_mirror
            t = read_mirror("oi_snapshots", ref_date, as_of, sym)
        else:
            t = _load(ref_date)
            if t is not None:
                t = t[t["symbol"] == sym]
                if as_of is not None:
                    cut = pd.Timestamp(as_of)
                    t = t[t["ts"] <= (cut.tz_localize(IST) if cut.tzinfo is None else cut)]
        today = builds(t, ref_date)
        if not len(today):
            return []
        minute = int(today.index[-1])
        days = pool_days(_captured(), ref_date, weekly)
        out = []
        for name, col, _oi, prem, sgn in _SIDES:
            ps = [_day(d, sym)[col].dropna() for d in days if len(_day(d, sym))]
            prior = pd.concat(ps) if ps else pd.Series(dtype=float)     # minute-indexed -> phase-aware
            h = score(today[col], prior, minute)
            if not h:
                continue
            row = today.loc[h["minute"]]
            kind = None
            if all(k in row.index for k in ("atm", "atm_was", prem, prem + "_was", "spot", "spot_was")) \
                    and row["atm"] == row["atm_was"] and pd.notna(row[prem]) and pd.notna(row[prem + "_was"]):
                # premium move net of the index move (ATM delta ~ +-0.5): >0 = buyers paying up
                res = (row[prem] - row[prem + "_was"]) - sgn * 0.5 * (row["spot"] - row["spot_was"])
                kind = "buyers" if res > 0 else "writers"
            tn = pd.Timestamp(f"{ref_date} {h['minute'] // 60:02d}:{h['minute'] % 60:02d}:00", tz=IST)
            ts_ = _top_strike(sym, ref_date, name, tn, float(row["spot"])) if "spot" in row.index and pd.notna(row["spot"]) else None
            out.append({"side": name, "hm": f"{h['minute'] // 60:02d}:{h['minute'] % 60:02d}",
                        "build": round(h["build"], 2), "z": h["z"], "ratio": h["ratio"], "live": h["live"],
                        "strike": ts_[0] if ts_ else None, "strike_add": ts_[1] if ts_ else None,
                        "strike_share": round(ts_[2], 2) if ts_ else None, "kind": kind,
                        "pool": h["pool"], "pool_days": len(days),
                        "stage": stage(ref_date, weekly)})
        return out
    except Exception:
        return []
