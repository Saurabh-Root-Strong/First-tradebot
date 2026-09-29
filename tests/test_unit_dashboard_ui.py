"""Unit — pure UI helpers extracted from dashboard.py into dashboard_ui.py.

Coverage the monolith never had: these were untestable while buried in dashboard (which
needs a broker token / Dash app to import). As leaf functions they test in isolation.
"""
from dashboard_ui import (_apply_trade_cap, _cap_muted, _fmt_contracts, _fmt_cr,
                          _scout_trade_status, _slug)


def _ep(sym, open_t, open_ts=None, **kw):
    return {"sym": sym, "open_t": open_t, "open_ts": open_ts or f"2026-09-29 {open_t}:00", **kw}


def test_trade_cap_off_is_identity():
    op, cl = [_ep("N", "10:00")], [_ep("N", "09:40"), _ep("N", "09:50")]
    k_op, k_cl, sk = _apply_trade_cap(op, cl, None)
    assert k_op == op and k_cl == cl and sk == []


def test_trade_cap_keeps_first_two_per_index_by_open_time():
    # closed list arrives NEWEST-first (as the ledger builds it); the cap must rank by time
    cl = [_ep("N", "13:00", outcome="SL"), _ep("N", "10:32", outcome="TARGET"),
          _ep("N", "10:17", outcome="FLIP"), _ep("B", "10:23", outcome="SL")]
    op = [_ep("N", "14:40")]                                   # the 4th NIFTY leg, still open
    k_op, k_cl, sk = _apply_trade_cap(op, cl, 2)
    assert [e["open_t"] for e in k_cl] == ["10:32", "10:17", "10:23"]   # order preserved
    assert k_op == []                                          # open 4th leg is NOT taken
    assert sorted((e["open_t"], e["trade_no"], e["cap_state"]) for e in sk) == [
        ("13:00", 3, "closed"), ("14:40", 4, "open")]
    assert all(e["sym"] == "N" for e in sk)                    # BANK untouched: indices independent


def test_trade_cap_same_minute_flip_ranks_by_seconds():
    # a FLIP closes leg 1 and leg 2 opens in the SAME minute — seconds decide the order
    cl = [_ep("M", "10:45", "2026-09-29 10:45:40"), _ep("M", "10:45", "2026-09-29 10:45:10"),
          _ep("M", "11:00", "2026-09-29 11:00:05")]
    _, k_cl, sk = _apply_trade_cap([], cl, 2)
    assert {e["open_ts"] for e in k_cl} == {"2026-09-29 10:45:10", "2026-09-29 10:45:40"}
    assert [e["open_t"] for e in sk] == ["11:00"]


def test_cap_muted():
    log = [{"kind": "NEW", "label": "NIFTY 50"}, {"kind": "SL", "label": "NIFTY 50"},
           {"kind": "NEW", "label": "NIFTY 50"}, {"kind": "NEW", "label": "BANK NIFTY"}]
    third = {"kind": "NEW", "label": "NIFTY 50"}
    assert _cap_muted([third], log + [third], 2) is True        # NIFTY's 3rd NEW → silent
    assert _cap_muted([third], log + [third], None) is False    # switch off → always beeps
    assert _cap_muted([log[2]], log, 2) is False                 # NIFTY's 2nd NEW still beeps
    bank_sl = {"kind": "SL", "label": "BANK NIFTY"}
    # a batch mixing a capped index with an in-cap one still beeps
    assert _cap_muted([third, bank_sl], log + [third, bank_sl], 2) is False
    assert _cap_muted([], log, 2) is False


def test_slug():
    assert _slug("NSE:NIFTY50-INDEX") == "NSE-NIFTY50-INDEX"
    assert _slug("NSE:NIFTYBANK-INDEX") == "NSE-NIFTYBANK-INDEX"


def test_fmt_cr():
    assert _fmt_cr(None) == "—"
    assert _fmt_cr("bad") == "—"
    assert _fmt_cr(500) == "+500"
    assert _fmt_cr(-500) == "-500"
    assert _fmt_cr(1500) == "+1.5K"
    assert _fmt_cr(12345) == "+12K"
    assert _fmt_cr(-2500) == "-2.5K"
    assert _fmt_cr(150000) == "+1.5L"


def test_fmt_contracts():
    assert _fmt_contracts(None) == "—"
    assert _fmt_contracts("bad") == "—"
    assert _fmt_contracts(500) == "+500"
    assert _fmt_contracts(-259253) == "-259K"
    assert _fmt_contracts(12000000) == "+12.0M"


def test_scout_trade_status():
    assert _scout_trade_status(None, 10, 6.5, 16.5, None) == "· no data"
    assert _scout_trade_status(10, None, 6.5, 16.5, None) == "· no data"
    # past target / past SL
    assert _scout_trade_status(10, 17, 6.5, 16.5, 17).startswith("🎯 target")
    assert _scout_trade_status(10, 6, 6.5, 16.5, 12).startswith("🛑 SL")
    # pullback: ran to +30% peak, now +10% -> gave back >=15pts
    assert _scout_trade_status(10, 11.0, 6.5, 16.5, 13.0).startswith("↩ pullback")
    # running / drawdown
    assert _scout_trade_status(10, 11.0, 6.5, 16.5, 11.0).startswith("▲")
    assert _scout_trade_status(10, 9.0, 6.5, 16.5, 10.0).startswith("▼")
