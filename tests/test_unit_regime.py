"""Unit tests for regime_classifier — the mood gate + band-width multiplier.

band_width_mult must ONLY ever widen (safe direction for a risk map); a regression
that tightens below 1.0 would understate risk. classify_from_bars must degrade to CHOP
on thin input, never raise. Pure. No network.
"""
import regime_classifier as rc


def test_band_widens_only_in_big_trend():
    assert rc.band_width_mult(rc.BIG_UP) == 1.08
    assert rc.band_width_mult(rc.BIG_DOWN) == 1.08
    assert rc.band_width_mult(rc.SMALL_UP) == 1.0
    assert rc.band_width_mult(rc.CHOP) == 1.0


def test_band_mult_unknown_is_unity():
    assert rc.band_width_mult("something-else") == 1.0
    assert rc.band_width_mult(None) == 1.0


def test_band_mult_never_tightens():
    # invariant: the band never narrows below base for ANY mood label
    for m in (rc.BIG_UP, rc.BIG_DOWN, rc.SMALL_UP, rc.SMALL_DOWN, rc.CHOP, "x"):
        assert rc.band_width_mult(m) >= 1.0


def test_classify_from_bars_thin_input_is_chop():
    assert rc.classify_from_bars({}).mood == rc.CHOP
    tiny = {k: [1.0, 2.0, 3.0] for k in ("open", "high", "low", "close")}
    assert rc.classify_from_bars(tiny).mood == rc.CHOP    # < n+2 bars → CHOP, no raise
