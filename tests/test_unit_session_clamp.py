"""The session window, and the footprint panel's refusal to invent a timeframe.

Regression for 2026-08-11: at 09:28 the live panel showed a "60m" cell of -0.368% that
was an OVERNIGHT return (the day's tick mirror opened with a 2026-08-10 17:56:31 row,
the only sample at or before the 08:28 anchor), a "15m" cell measured from the pre-open
call auction, and identical "optOI +851L / futOI +13.8L" on BOTH — because when
`_at_or_before` found nothing it fell back to `.iloc[0]`, so two windows collapsed onto
one measurement and then read as two timeframes agreeing.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from core import session

IST = "Asia/Kolkata"
DAY = "2026-08-11"


def _ts(*hhmm, day=DAY):
    return pd.Timestamp(f"{day} {hhmm[0]:02d}:{hhmm[1]:02d}", tz=IST)


# ── the window ───────────────────────────────────────────────────────────────────
def test_drops_pre_open_post_close_and_previous_day():
    df = pd.DataFrame({"ts": [
        _ts(17, 56, day="2026-08-10"),   # straggler from the evening before
        _ts(9, 0),                       # pre-open call auction
        _ts(9, 15),                      # first real print
        _ts(12, 0),
        _ts(15, 30),                     # the bell, inclusive
        _ts(17, 56),                     # post-close tape
    ]})
    out = session.session_only(df, "ts", DAY)
    assert list(out["ts"]) == [_ts(9, 15), _ts(12, 0), _ts(15, 30)]


def test_day_filter_is_opt_in():
    """Without `day` the clamp is time-of-day only — a previous-evening 17:56 row is
    dropped by the window anyway, but a previous-day 12:00 row is not. Callers that
    read a per-day mirror must pass the day."""
    df = pd.DataFrame({"ts": [_ts(12, 0, day="2026-08-10"), _ts(12, 0)]})
    assert len(session.session_only(df, "ts")) == 2
    assert len(session.session_only(df, "ts", DAY)) == 1


def test_boundaries_inclusive():
    assert session.in_session(dt.time(9, 15)) and session.in_session(dt.time(15, 30))
    assert not session.in_session(dt.time(9, 14, 59))
    assert not session.in_session(dt.time(15, 30, 1))


def test_empty_and_missing_column_pass_through():
    assert session.session_only(None) is None
    empty = pd.DataFrame({"ts": pd.to_datetime([])})
    assert len(session.session_only(empty)) == 0
    noc = pd.DataFrame({"other": [1]})
    assert len(session.session_only(noc, "ts")) == 1


# ── the panel ────────────────────────────────────────────────────────────────────
@pytest.fixture
def synthetic(monkeypatch):
    """Capture starting at 09:15, plus the two poison rows the real mirror carried."""
    import intraday_tf as itf
    sym = "NSE:NIFTY50-INDEX"
    mins = pd.date_range(_ts(9, 15), _ts(9, 28), freq="1min")
    ticks = pd.DataFrame({
        "ts": [_ts(17, 56, day="2026-08-10"), _ts(9, 0)] + list(mins),
        "symbol": sym,
        "ltp": [24700.0, 24650.0] + [24500.0 + i for i in range(len(mins))],
    })
    oi = pd.DataFrame({
        "ts": list(mins), "symbol": sym,
        "total_call_oi": [1_000_000 + i * 10_000 for i in range(len(mins))],
        "total_put_oi": [900_000 + i * 5_000 for i in range(len(mins))],
        "atm_iv": 12.0,
    })

    def fake_read(tbl, date=None, as_of=None, symbol=None):
        return {"ticks": ticks, "oi_snapshots": oi}.get(tbl)

    monkeypatch.setattr(itf, "_read", fake_read)
    return itf, sym


def test_window_longer_than_capture_is_pending_not_invented(synthetic):
    """13 minutes of capture cannot produce a 15m or 60m cell. Before the fix both
    existed, both carried the SAME optOI number, and the 60m price was an overnight."""
    itf, sym = synthetic
    r = itf.analyze(sym, DAY, dt.datetime(2026, 8, 11, 9, 28,
                                          tzinfo=dt.timezone(dt.timedelta(hours=5, minutes=30))))
    assert r["has_data"]
    live = {c["tf"] for c in r["cells"]}
    pend = {p["tf"] for p in r["pending"]}
    assert 15 in pend and 60 in pend, "a window longer than capture must not print a number"
    assert 15 not in live and 60 not in live


def test_no_two_cells_share_a_measurement(synthetic):
    """The panel's whole claim is multi-timeframe CONFIRMATION. Two cells reporting the
    identical OI delta is one measurement counted twice, and it inflated the stack's
    'N/N TFs agree' line."""
    itf, sym = synthetic
    r = itf.analyze(sym, DAY, dt.datetime(2026, 8, 11, 9, 28,
                                          tzinfo=dt.timezone(dt.timedelta(hours=5, minutes=30))))
    tots = [c["d_tot"] for c in r["cells"]]
    assert len(tots) == len(set(tots)), f"duplicate windows collapsed onto one number: {tots}"


def test_overnight_row_never_becomes_a_baseline(synthetic):
    """The 24700 row is stamped the previous evening and the 24650 row is the pre-open
    auction. No surviving cell may be measured against either."""
    itf, sym = synthetic
    r = itf.analyze(sym, DAY, dt.datetime(2026, 8, 11, 9, 28,
                                          tzinfo=dt.timezone(dt.timedelta(hours=5, minutes=30))))
    last = 24500.0 + 13
    for c in r["cells"]:
        implied_base = last / (1 + c["px"] / 100.0)
        assert 24499 <= implied_base <= 24515, (
            f"tf={c['tf']} priced off {implied_base:.0f}, outside the captured session")
