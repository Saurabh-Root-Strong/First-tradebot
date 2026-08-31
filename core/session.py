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

# ── THE CLOSING AUCTION SESSION (CAS) SPLIT THE END OF THE DAY IN TWO ────────────
# SEBI circular 2026-01-16, live on NSE from 2026-08-03. For F&O-eligible stocks —
# i.e. every heavyweight index constituent — CONTINUOUS trading now ends at 15:15.
# 15:15-15:30 is an auction (order entry only, no matching, window shuts at a RANDOM
# moment between 15:28 and 15:30), and the closing price is the single equilibrium
# price matched at ~15:28-15:35. The old rule (close = VWAP of the last 30 minutes)
# is gone.
#
# MEASURED IN OUR OWN MIRRORS: from 2026-08-03 the NIFTY tape carries exactly TWO
# distinct LTPs between 15:15 and 15:30 on EVERY session (225-480 before it) — a flat
# line from 15:15:00, then one step to the CAS print at ~15:28-15:29. That step
# averaged +12.8bps over the first 18 sessions and reached +197 points on day one.
#
# So 15:30 is still the SESSION boundary and the CAS print is the OFFICIAL close —
# CLOSE stays 15:30 and settlement still uses it. But the last continuously TRADED
# price is at CONTINUOUS_CLOSE, and any statistic that assumes prices were being
# discovered by trading (close-strength, range, sigma, bar builders, band width) owes
# itself that boundary instead: over 15:15-15:30 sigma collapses to ~0 and is then
# handed a gap, which is not volatility, it is a scheduled auction.
CONTINUOUS_CLOSE = dt.time(15, 15)

# Index derivatives keep trading through the auction and close at 15:40 — they are NOT
# auctioned. This is why the cash index and its futures diverge for the last 25 minutes
# of the day, and why a futures fill can never be assumed at the cash CAS print.
DERIV_CLOSE = dt.time(15, 40)


def in_session(t: dt.time) -> bool:
    return OPEN <= t <= CLOSE


def in_continuous(t: dt.time) -> bool:
    """Inside the CONTINUOUSLY-TRADED cash session (09:15-15:15 since CAS). Use this,
    not in_session, wherever a price is only meaningful if trading produced it."""
    return OPEN <= t <= CONTINUOUS_CLOSE


def continuous_only(df, col: str = "ts", day=None):
    """session_only(), but clamped at CONTINUOUS_CLOSE — drops the CAS auction window."""
    out = session_only(df, col=col, day=day)
    if out is None or not len(out) or col not in out.columns:
        return out
    return out[out[col].dt.time <= CONTINUOUS_CLOSE]


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
