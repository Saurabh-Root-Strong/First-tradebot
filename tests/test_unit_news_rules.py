"""Unit tests for news_events scoring — the deterministic rule-book classifier.

Pure text→(type, score) logic + macro-surprise scorer. Guards the routine-filing
suppressor (the SAST/insolvency-paperwork flood) and the material-event scores that
drive the news panel's |score|>=5 filter. No network.
"""
import news_events as ne


def test_material_events_score():
    assert ne.score_text("Board approves buyback of equity shares")[:2] == ("Buyback", 8)
    etype, sc, _ = ne.score_text(
        "NCLT admits company for Corporate Insolvency Resolution Process")
    assert etype == "Credit downgrade" and sc == -8


def test_routine_filings_suppressed_to_zero():
    # SAST takeover paperwork must NOT read as an Acquisition (the tape-flood guard)
    _t, sc, _ = ne.score_text(
        "Disclosure under SEBI Takeover Regulations — Welspun Group Master Trust "
        "has Submitted to the Exchange")
    assert sc == 0
    # procedural NCLT meeting scrutinizer report must NOT re-fire Credit downgrade
    _t2, sc2, _ = ne.score_text(
        "Voting results and the Consolidated Scrutinizer's Report on the NCLT "
        "convened Meeting")
    assert sc2 == 0


def test_unmatched_is_uncategorised_zero():
    _t, sc, _ = ne.score_text("Intimation of record date for dividend administrative note")
    assert sc == 0


def test_surprise_hot_inflation_is_bearish():
    assert ne.score_surprise("US CPI", 3.5, 3.2)["score"] < 0


def test_surprise_gdp_beat_is_bullish():
    assert ne.score_surprise("India GDP", 7.8, 7.0, higher_is_bullish=True)["score"] > 0


def test_surprise_no_surprise_is_neutral():
    assert ne.score_surprise("US CPI", 3.2, 3.2)["score"] == 0
