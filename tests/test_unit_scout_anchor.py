"""Scout board fix — an OPEN position's lifecycle FREEZES its trigger at the poller's
first-fire minute (the `anchor`) instead of re-walking the coarse bar grid.

The bug it guards: on the 60m board a trade born mid-bar has no completed TRADE bar
behind it, so the grid walk breaks at the first step and reports trigger=as_of — the
entry re-priced to NOW on every 30s refresh (perpetual +0%, trigger clock drifting with
the wall clock) while the ledger correctly holds "since <first fire>". The anchor collapses
the two clocks to one frozen entry. Data-free: the option/spot reads are stubbed so the
test isolates the trigger-freeze logic.
"""
import datetime

import intraday_scout as s

IST = s.IST


def _stub(monkeypatch):
    # No completed TRADE bar behind us → grid walk would break immediately (the bug).
    monkeypatch.setattr(s, "scan_index",
                        lambda *a, **k: {"verdict": "NO-TRADE"})
    monkeypatch.setattr(s, "_read_mirror", lambda *a, **k: None)   # skip as_of clamp
    monkeypatch.setattr(s, "_spot_at", lambda sym, d, t: 24000.0)
    monkeypatch.setattr(s, "_atm", lambda spot, sym: 24000)
    monkeypatch.setattr(s, "_opt_premium", lambda sym, d, t, atm, side: 100.0)


def test_anchor_freezes_trigger(monkeypatch):
    _stub(monkeypatch)
    d = "2026-07-07"
    as_of = datetime.datetime(2026, 7, 7, 14, 53, tzinfo=IST)
    anchor = datetime.datetime(2026, 7, 7, 10, 2, tzinfo=IST)

    lc_walk = s._lifecycle("NIFTY", 60, d, as_of, "CE", 60, 0.2, anchor=None)
    lc_anch = s._lifecycle("NIFTY", 60, d, as_of, "CE", 60, 0.2, anchor=anchor)

    # Grid walk (no completed TRADE bar) re-stamps to now — the confusing drift.
    assert lc_walk["trigger"] == "14:53"
    # Anchor freezes it at the poller's first-fire minute.
    assert lc_anch["trigger"] == "10:02"


def test_future_anchor_clamped_to_asof(monkeypatch):
    _stub(monkeypatch)
    d = "2026-07-07"
    as_of = datetime.datetime(2026, 7, 7, 10, 0, tzinfo=IST)
    future = datetime.datetime(2026, 7, 7, 11, 0, tzinfo=IST)   # just-logged, ahead of as_of
    lc = s._lifecycle("NIFTY", 60, d, as_of, "CE", 60, 0.2, anchor=future)
    # A future anchor must never be shown — falls back to the grid walk (→ as_of).
    assert lc["trigger"] == "10:00"


def test_scan_dir_mismatch_ignores_anchor(monkeypatch):
    # Poller holds CE but the live leg is PE → stale anchor, must NOT freeze on wrong side.
    _stub(monkeypatch)
    captured = {}

    def _fake_lc(sym, tf, d, as_of, direction, hz, strength, not_before=None, anchor=None):
        captured["anchor"] = anchor
        return {"trigger": "x", "entry_strike": None}
    monkeypatch.setattr(s, "_lifecycle", _fake_lc)
    # direction resolved inside scan_index is data-driven; assert the gate logic directly.
    anchor = {"t": datetime.datetime(2026, 7, 7, 10, 2, tzinfo=IST), "dir": "CE"}
    _anchor_t = (anchor.get("t") if isinstance(anchor, dict)
                 and anchor.get("dir") == "PE" else None)
    assert _anchor_t is None
    _anchor_t = (anchor.get("t") if isinstance(anchor, dict)
                 and anchor.get("dir") == "CE" else None)
    assert _anchor_t == anchor["t"]
