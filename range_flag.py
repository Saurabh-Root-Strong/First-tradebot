"""range_flag.py — the BIG-MOVE-DAY size flag: "the index has moved N× what is normal BY NOW".

    mult(T) = today's range-so-far at clock time T
              ÷ the median range-so-far at the SAME clock time over the prior 20 sessions

WHY THIS AND NOT OPEN INTEREST. The question that prompted it (2026-10-09, NIFTY −371 then
+289 on consecutive days) was whether an abnormal OI pile-up warns of a big move. Measured on
63 captured sessions, three ways — at fixed checkpoints, as a burst at ANY time, and as a
lead-lag — it does not: an OI burst with price calm is followed by a slightly QUIETER hour
(×1.15 vs ×1.24 baseline), and OI churn lines up with the price move 15 minutes EARLIER and
in the same window (+0.16), not with the one after (+0.04). OI follows price. What does carry
forward is the price range itself.

WHAT IS MEASURED (2 years of 5-min bars, 4 indices, 1,932 index-days, 15-min grid):

    mult bucket    rest-of-day range    chance the rest of the day is itself big (>1.5× usual)
      < 0.75×           0.91×                    10%
    0.75–1.25×          1.13×                    19%
    1.25–1.5×           1.26×                    24%
     1.5–2×             1.39×                    35%
      2–2.5×            1.50×                    44%
    2.5–3×              1.82×                    56%
      > 3×              1.89×                    67%

Rank IC +0.33, and it does not wobble: +0.37 / +0.26 / +0.35 / +0.33 by half-year, +0.29..+0.37
on every index, +0.29..+0.36 at every clock time from 09:30 to 14:30. It also fires hours
before the cockpit's older HIGH_VOL rule (range-so-far vs the 85th percentile of FULL days):
by 10:00 it had flagged 114 index-days against 23.

WHAT IT IS NOT — and the reason it is rendered as a size chip with no arrow:
  * NOT a direction. On the first flag of the day, following the move made +1.3 bps (t 0.19),
    fading it −1.3, a breakout of the range +2.7 (t 0.43); thirteen scenario cuts, none
    reached t 2 and the two halves disagreed on every one.
  * NOT a reason to buy options. On flagged instants an ATM straddle still lost 1.28% an hour
    (CI −2.70..−0.23): implied volatility had already repriced (17.7 vs 14.3).
  * NOT a reason to touch the band. It is built from the day's own volatility and widens by
    itself; it finishes inside 68.3% of the time on flagged instants against 68.8% normally.

It is a RISK reading: widen the stop, cut the size, stop treating the band edge as a
quiet-day fade.

CAUSAL BY CONSTRUCTION. The baseline uses only sessions strictly BEFORE the day being read,
so a replay of a past day sees exactly what it would have seen then.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

LOOKBACK = 20            # prior sessions in the baseline — the window the 2-year test used
MIN_DAYS = 10            # fewer than this and "normal" is not a thing yet → no reading
FIRST_MINUTE = 9 * 60 + 45   # nothing before 09:45: the denominator is a few points of noise
ELEVATED, FLAG = 1.5, 2.0    # chip thresholds (1.5× = worth a glance, 2× = the tested flag)
_OPEN_MIN, _CLOSE_MIN = 9 * 60 + 15, 15 * 60 + 30

# (upper edge of the bucket, rest-of-day range ×usual, % chance the rest of the day is big)
_MEASURED = ((0.75, 0.91, 10), (1.25, 1.13, 19), (1.5, 1.26, 24), (2.0, 1.39, 35),
             (2.5, 1.50, 44), (3.0, 1.82, 56), (float("inf"), 1.89, 67))


def expectation(mult: float) -> tuple[float, int]:
    """(rest-of-day range as a multiple of usual, % chance the rest of the day is big) for a
    multiple — read straight off the measured table, never extrapolated past its last row."""
    for edge, rest, big in _MEASURED:
        if mult < edge:
            return rest, big
    return _MEASURED[-1][1], _MEASURED[-1][2]


def level(mult) -> str:
    """'' below 1.5×, 'elevated' from 1.5×, 'flag' from 2× (the threshold that was tested)."""
    if mult is None or mult != mult:
        return ""
    return "flag" if mult >= FLAG else "elevated" if mult >= ELEVATED else ""


LAST_FORECAST_MINUTE = 14 * 60 + 30   # the measured table stops at 14:30 — after that there is
                                      # too little day left for "rest of day" to mean anything
PRIOR_MAX_STALE = 10      # a prior session only counts at minute M if it has a reading within
                          # this many minutes of M — a capture that died at 10:44 must not go on
                          # supplying its 10:44 range as "normal for 13:00"
_EXCH = ("day_open", "day_high", "day_low")
_EMPTY = pd.Series(dtype=float)


def curve(ticks: "pd.DataFrame | None", day: "str | None" = None) -> "pd.Series":
    """Range-so-far (% of the session open) at each minute of ONE session, indexed by
    minute-of-day. Empty when there is too little to say anything.

    SOURCE, in order of preference:
      1. The EXCHANGE's own running day_high / day_low / day_open carried on every tick. It is
         the exact figure (intrabar extremes included) and — the reason it is preferred — it is
         RIGHT EVEN WHEN CAPTURE STARTED LATE: on 2026-08-12 the feed came up at 09:57; the
         tick prices from then on spanned 0.46% by 11:30 while the day had really ranged 0.74%.
         Checked over 320 index-days: never shrinks, the price never sits outside it, and it
         matches 5-min bars to 0.008pp (the tick-price route: 0.041pp).
      2. Fallback for frames without those columns: running max−min of 1-minute LAST prices.

    GUARDS: rows stamped another date are dropped (a day file can open with the previous
    evening's last quote); rows whose day_open differs from the session's settled value are
    dropped (the first seconds can still carry yesterday's fields)."""
    if ticks is None or not len(ticks) or "ltp" not in ticks.columns:
        return _EMPTY
    t = ticks[ticks["ltp"] > 0]
    if not len(t):
        return _EMPTY
    d = dt.date.fromisoformat(day) if day else t["ts"].dt.date.max()
    mins = t["ts"].dt.hour * 60 + t["ts"].dt.minute
    t = t[(t["ts"].dt.date == d) & (mins >= _OPEN_MIN) & (mins <= _CLOSE_MIN)]
    if len(t) < 30:
        return _EMPTY
    if all(c in t.columns for c in _EXCH):
        x = t[(t["day_open"] > 0) & (t["day_low"] > 0) & (t["day_high"] >= t["day_low"])]
        if len(x) >= 30:
            op = float(x["day_open"].iloc[-1])                       # the settled open
            x = x[(x["day_open"] - op).abs() <= op * 1e-4]
            if len(x) >= 30:
                r = ((x["day_high"] - x["day_low"]) / op * 100.0)
                r.index = x["ts"]
                s = r.resample("1min").last().ffill().dropna()
                s.index = s.index.hour * 60 + s.index.minute
                return s
    s = t.set_index("ts")["ltp"].resample("1min").last().ffill().dropna()
    if len(s) < 10:
        return _EMPTY
    out = (s.cummax() - s.cummin()) / float(s.iloc[0]) * 100.0
    out.index = s.index.hour * 60 + s.index.minute
    return out


def at(c: "pd.Series", minute: int, max_stale: "int | None" = None):
    """Value of a range-so-far curve at-or-before `minute`. None before its first point, or —
    when max_stale is given — if its nearest reading is older than that many minutes."""
    if c is None or not len(c):
        return None
    idx = c.index.to_numpy()
    i = int(np.searchsorted(idx, minute, side="right")) - 1
    if i < 0 or (max_stale is not None and minute - int(idx[i]) > max_stale):
        return None
    return float(c.iloc[i])


def multiple(today: "pd.Series", priors: list, minute: int):
    """today's range-so-far ÷ the median of the prior sessions' at the same minute.
    None when the clock is too early, today has no reading, or fewer than MIN_DAYS priors
    have a reading NEAR that minute (a session that stopped capturing earlier is skipped)."""
    if minute < FIRST_MINUTE:
        return None
    now = at(today, minute)
    base = [v for v in (at(c, minute, PRIOR_MAX_STALE) for c in priors[-LOOKBACK:])
            if v is not None and v > 0]
    if now is None or len(base) < MIN_DAYS:
        return None
    med = float(np.median(base))
    return now / med if med > 0 else None


_COLS = ["ts", "symbol", "ltp", *_EXCH]
_DAY_CURVES: dict = {}    # day -> {sym: curve} for FINISHED prior sessions (immutable)


def _load_ticks(day: str) -> "pd.DataFrame | None":
    """One day's ticks, ONLY the columns the curve needs, read straight off the parquet.

    Deliberately NOT core.mirror_io.read_mirror for past days: that reader keeps whole
    past-day frames in a shared 32-slot cache (right for the replay that re-reads one chain
    file hundreds of times). Twenty past tick files there is ~180 MB pinned on a 1 GB capture
    VM, and it evicts the frames the replay actually wants. Here each file is read once,
    reduced to four tiny curves, and dropped."""
    from core.constants import IST, LIVE_DIR
    p = LIVE_DIR / f"{day}_ticks.parquet"
    if not p.exists() or p.stat().st_size < 100:
        return None
    try:
        import pyarrow.parquet as pq
        have = set(pq.read_schema(p).names)
        df = pd.read_parquet(p, columns=[c for c in _COLS if c in have])
    except Exception:
        return None
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(IST)
    return df.sort_values("ts")


def _prior_days(ref_date: str) -> list:
    """Up to LOOKBACK captured TRADING sessions strictly before ref_date. Weekend / holiday
    files are excluded: mock and special sessions get captured too (2026-07-04, a Saturday,
    carries the previous day's open for its first minutes) and are not 'a normal day'."""
    from core import market_calendar as mc
    from core.constants import LIVE_DIR
    out = []
    for p in LIVE_DIR.glob("*_ticks.parquet"):
        d = p.name[:10]
        try:
            dd = dt.date.fromisoformat(d)
        except ValueError:
            continue
        if d < ref_date and p.stat().st_size > 200_000 and mc.is_trading_day(dd):
            out.append(d)
    return sorted(out)[-LOOKBACK:]


def _prior_curves(sym: str, ref_date: str) -> list:
    out = []
    for d in _prior_days(ref_date):
        if d not in _DAY_CURVES:
            t = _load_ticks(d)
            _DAY_CURVES[d] = ({s: curve(g, d) for s, g in t.groupby("symbol")}
                              if t is not None and "symbol" in t.columns else {})
            while len(_DAY_CURVES) > 120:                     # ~6 months of sessions; bounded
                _DAY_CURVES.pop(next(iter(_DAY_CURVES)))
        c = _DAY_CURVES[d].get(sym)
        if c is not None and len(c):
            out.append(c)
    return out


def read(sym: str, ref_date: str, as_of: "dt.datetime | None" = None) -> "dict | None":
    """The cockpit's reading for one index, or None when there is nothing honest to say:
        {'mult', 'level', 'rest', 'big', 'hm', 'n'}
    `rest`/`big` (the rest-of-day expectation) are None after 14:30 — the table they come from
    stops there. ref_date = the session being read; as_of caps its ticks (replay / ghost)."""
    try:
        from core.constants import today_iso
        if ref_date == today_iso():
            # today's growing file is already held (mtime-keyed) by the shared live cache
            from core.mirror_io import read_mirror
            t = read_mirror("ticks", ref_date, as_of, sym)
        else:
            t = _load_ticks(ref_date)
            if t is not None:
                t = t[t["symbol"] == sym]
                if as_of is not None:
                    cut = pd.Timestamp(as_of)
                    t = t[t["ts"] <= (cut.tz_localize(t["ts"].dt.tz) if cut.tzinfo is None else cut)]
        today = curve(t, ref_date)
        if not len(today):
            return None
        minute = int(today.index[-1])
        priors = _prior_curves(sym, ref_date)
        m = multiple(today, priors, minute)
        if m is None:
            return None
        rest, big = expectation(m) if minute <= LAST_FORECAST_MINUTE else (None, None)
        n = sum(1 for c in priors[-LOOKBACK:] if at(c, minute, PRIOR_MAX_STALE))
        return {"mult": round(m, 2), "level": level(m), "rest": rest, "big": big,
                "hm": f"{minute // 60:02d}:{minute % 60:02d}", "n": n}
    except Exception:
        return None
