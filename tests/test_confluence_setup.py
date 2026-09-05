"""Offline tests for tradeboard.confluence_setup — the MTF level trade suggestion."""
import pytest

import tradeboard as tb

ATR = 100.0
SPOT = 23750.0


def test_fires_long_when_price_sits_on_a_both_frames_support():
    s = tb.confluence_setup([(23745.0, 3)], [(23752.0, 2)], SPOT, ATR)
    assert s["side"] == "LONG" and s["kind"] == "support"
    assert s["sl"] < s["ltf_level"] < s["entry"] < s["target"]


def test_fires_short_when_price_sits_under_a_both_frames_resistance():
    s = tb.confluence_setup([(23756.0, 2)], [(23749.0, 2)], SPOT, ATR)
    assert s["side"] == "SHORT" and s["kind"] == "resistance"
    assert s["target"] < s["entry"] < s["ltf_level"] < s["sl"]


def test_silent_when_the_htf_has_no_level_nearby():
    assert tb.confluence_setup([(23745.0, 3)], [(24500.0, 4)], SPOT, ATR) == {}


def test_silent_when_price_is_far_from_the_level():
    """Level exists on both frames but price is nowhere near it — context, not a setup."""
    assert tb.confluence_setup([(23000.0, 3)], [(23005.0, 3)], SPOT, ATR) == {}


def test_silent_when_squeezed_between_confluent_support_AND_resistance():
    """Two-sided = a coin flip inside a box. The backtest skipped these, so must this,
    or the grade shown next to the setup would not describe what fired."""
    lv_l = [(23745.0, 2), (23756.0, 2)]
    lv_h = [(23746.0, 2), (23755.0, 2)]
    assert tb.confluence_setup(lv_l, lv_h, SPOT, ATR) == {}


def test_risk_is_measured_from_ENTRY_not_from_the_level():
    s = tb.confluence_setup([(23745.0, 3)], [(23750.0, 2)], SPOT, ATR)
    assert s["risk_pts"] == pytest.approx(abs(SPOT - s["sl"]), abs=0.05)
    assert abs(s["target"] - s["entry"]) == pytest.approx(tb.CONF_RR * s["risk_pts"], abs=0.2)


def test_fixed_point_tolerance_is_honoured_when_asked():
    far = [(23700.0, 2)]                       # 45 pts from the 23745 LTF level
    assert tb.confluence_setup([(23745.0, 3)], far, SPOT, ATR, tol_pts=20) == {}
    assert tb.confluence_setup([(23745.0, 3)], far, SPOT, ATR, tol_pts=50) != {}


def test_degenerate_inputs_never_raise():
    for args in (([], [], SPOT, ATR), ([(1, 1)], [(1, 1)], 0, ATR),
                 ([(1, 1)], [(1, 1)], SPOT, 0), (None, None, SPOT, ATR)):
        assert tb.confluence_setup(*args) == {}


# ── the grade must stay attached to the suggestion ───────────────────────────────
def test_every_shipped_tf_pair_has_a_measured_grade():
    """The UI prints CONF_GRADE beside an armed setup. A pair with no grade would render
    a trade suggestion with no evidence next to it — the exact thing this must not do."""
    for pair in ("5m>15m", "10m>30m", "15m>60m"):
        g = tb.CONF_GRADE[pair]
        assert g["n"] > 100 and "verdict" in g
        # NOT asserting hi < 0 — 15m>60m measured [-5.74, +0.26], which touches zero, and
        # a test that demanded a negative upper bound would be forcing the data to agree
        # with the conclusion. What must hold is that NO pair is significantly POSITIVE:
        # nothing here may be presented as a winning setup.
        assert g["lo"] < 0, f"{pair} would be a positive edge — re-verify before shipping"
        assert g["bps"] < 0, f"{pair} mean must match the measured negative"
