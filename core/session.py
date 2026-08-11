"""session.py — the NSE cash-session window, and the clamp every bar/window builder owes it.

Capture runs WIDER than the session on both ends and both ends are poison:

  * PRE-OPEN (09:00-09:15). The call auction publishes INDICATIVE prices that never
    traded. Measured 2026-08-10 NIFTY: 504 pre-open ticks reaching 24722.5 against a
    true session high of 24618.9.
  * POST-CLOSE. Ticks kept arriving until 17:56 on the same day.
  * PREVIOUS DAY. A day's tick mirror can carry a straggler row stamped the evening
    BEFORE (2026-08-11's file opens with a 2026-08-10 17:56:31 tick) — which is how an
    "as of 09:28, last 60 minutes" lookback silently became an OVERNIGHT return.

Each consumer used to decide this for itself, and mostly decided not to: `atm_strikes`
clamped, `footprint_chart`'s bar builders did not (pre-open auction highs in the 60m
candle), and `intraday_tf` did not (the whole multi-timeframe footprint panel). Same
class of bug as the capture-role rule — one rule, several copies, silent disagreement.
One definition here; everything imports it.
"""
from __future__ import annotations

import datetime as dt

OPEN = dt.time(9, 15)
CLOSE = dt.time(15, 30)


def in_session(t: dt.time) -> bool:
    return OPEN <= t <= CLOSE


def session_only(df, col: str = "ts", day=None):
    """Rows inside 09:15-15:30. Pass `day` (date or ISO string) to ALSO drop rows
    stamped a different date — a per-day mirror is not guaranteed to hold only that
    day, and a single stale evening row from the day before is enough to turn a
    lookback window into an overnight move."""
    if df is None or not len(df) or col not in df.columns:
        return df
    ts = df[col]
    keep = (ts.dt.time >= OPEN) & (ts.dt.time <= CLOSE)
    if day is not None:
        d = dt.date.fromisoformat(day) if isinstance(day, str) else day
        keep &= (ts.dt.date == d)
    return df[keep]
