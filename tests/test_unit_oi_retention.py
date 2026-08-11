"""OI snapshot cadence and retention must stay coupled.

`MAX_OI_SNAPSHOTS = 150` shipped with the comment "150 x 3 min = 7.5 hours — more than
full session" while the accept-throttle was actually 90s. 150 x 90s = 3.75h against a
6.25h session, so the in-memory series silently dropped the first ~2.5 hours of every
day — the morning, which carries the largest OI moves of the session (09:00-09:30 is
16.4% of total |dOI|, measured over 6 sessions). Two live readers saw a truncated day:
the dashboard OI panel and intraday_shock.

The comment was right about the arithmetic and wrong about the interval, which is the
signature of a constant that has to be re-reasoned by hand every time a nearby number
moves. It is derived now; this pins the relationship.
"""
from __future__ import annotations

import intraday_store as st

SESSION_HOURS = 6.25          # 09:15 -> 15:30


def test_retention_covers_a_full_session_at_the_configured_cadence():
    held_hours = st.MAX_OI_SNAPSHOTS * st.OI_SNAPSHOT_SEC / 3600
    assert held_hours >= SESSION_HOURS, (
        f"the in-memory OI series holds {held_hours:.2f}h at {st.OI_SNAPSHOT_SEC}s but the "
        f"session is {SESSION_HOURS}h — the morning will be silently evicted")


def test_the_old_broken_pairing_would_fail_this_test():
    """Guards the guard: if this assertion could not catch the original bug it is
    decoration. 150 snapshots at a 90s throttle held 3.75h of a 6.25h session."""
    assert 150 * 90 / 3600 < SESSION_HOURS


def test_store_default_throttle_is_the_shared_constant():
    """OIStore.add's default must BE the constant, not a copy of its value — the poller
    sleep and this throttle are two independent rate limits and they already disagreed
    once (poller 30s vs throttle 90s, so oi_snapshots was 90s data inside the fast
    window, computed from fetches that had already been paid for)."""
    import inspect
    sig = inspect.signature(st.OITimeSeriesStore.add)
    assert sig.parameters["min_interval_sec"].default == st.OI_SNAPSHOT_SEC


def test_deque_is_bounded_by_the_constant():
    s = st.OITimeSeriesStore(["NSE:NIFTY50-INDEX"])
    assert s._s["NSE:NIFTY50-INDEX"].maxlen == st.MAX_OI_SNAPSHOTS
