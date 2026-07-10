"""Unit — btst_panel view model (rupee truth + scorecard verdict). Pure, no ledger IO."""
from btst_panel import BACKTEST_HI, BACKTEST_LO, net_rupees, per_index, summary


def test_net_rupees_uses_futures_lot_and_is_cost_inclusive():
    # NIFTY lot 65: 24005.85 x (39.8/1e4) x 65 = 6210.3 -> 6210
    assert net_rupees(24005.85, 39.8, "NIFTY") == 6210
    # MIDCAP lot 120 turns a modest bps loss into a big rupee loss (the whole point).
    # NOTE the panel computes rupees from RAW net_bps and only rounds for display, so the
    # ledger's -39.594 shows "-39.6 bps / Rs-6,951" while this rounded input gives -6,952.
    assert net_rupees(14629.20, -39.6, "MIDCPNIFTY") == -6952
    # the SAME bps loss costs far more on MIDCAP's 120-lot than on NIFTY's 65-lot
    assert abs(net_rupees(24000, -39.6, "NIFTY")) < abs(net_rupees(14629, -39.6, "MIDCPNIFTY"))


def test_net_rupees_none_safe():
    assert net_rupees(None, 10, "NIFTY") is None
    assert net_rupees(24000, None, "NIFTY") is None
    assert net_rupees(24000, 10, "BOGUS") is None
    assert net_rupees("bad", 10, "NIFTY") is None


def _row(sym, bps, rupee):
    return {"index": sym, "sym": sym, "net_bps": bps, "rupee": rupee}


def test_summary_tracking_verdict_against_backtest_band():
    below = summary([_row("NIFTY", 5.0, 100), _row("NIFTY", 5.0, 100)])
    assert below["tracking"] == "BELOW expectation" and below["mean_bps"] == 5.0
    inband = summary([_row("NIFTY", BACKTEST_LO, 1), _row("NIFTY", BACKTEST_HI, 1)])
    assert inband["tracking"] == "tracking"
    above = summary([_row("NIFTY", 30.0, 1)])
    assert above["tracking"] == "above expectation"


def test_summary_counts_and_gate():
    s = summary([_row("NIFTY", 10, 100), _row("NIFTY", -5, -50), _row("NIFTY", 20, 200)])
    assert s["n"] == 3 and s["wins"] == 2 and s["win_pct"] == 66.7
    assert s["total_rupees"] == 250 and s["worst_bps"] == -5.0
    assert s["gate_left"] == 22          # 25-night review gate


def test_summary_empty_is_not_a_pass():
    s = summary([])
    assert s["n"] == 0 and s["mean_bps"] is None and "no closed nights" in s["tracking"]


def test_per_index_sorted_worst_first():
    rows = [_row("NIFTY 50", 25.0, 500), _row("MIDCAP NIFTY", -30.0, -600),
            _row("MIDCAP NIFTY", -10.0, -200)]
    out = per_index(rows)
    assert out[0]["index"] == "MIDCAP NIFTY"      # worst leg surfaces first
    assert out[0]["n"] == 2 and out[0]["mean_bps"] == -20.0 and out[0]["rupee"] == -800
    assert out[-1]["index"] == "NIFTY 50"
