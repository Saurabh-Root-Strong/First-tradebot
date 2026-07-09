"""Unit — pure UI helpers extracted from dashboard.py into dashboard_ui.py.

Coverage the monolith never had: these were untestable while buried in dashboard (which
needs a broker token / Dash app to import). As leaf functions they test in isolation.
"""
from dashboard_ui import _fmt_contracts, _fmt_cr, _scout_trade_status, _slug


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
