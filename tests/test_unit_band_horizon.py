"""Unit tests for the band-horizon + calibration math — the RANGE product's core.

session_horizon / eval_asof enforce the session-close cap (no phantom post-15:30
state, no band projecting into the overnight gap). _k is the coverage→sigma-multiple
used by the L4 calibration loop. All pure datetime / stats. No network.
"""
import datetime

import pytest

import hour_forecast as hf
import calibration_engine as ce

IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))


def _t(h, m):
    return datetime.datetime(2026, 7, 6, h, m, tzinfo=IST)


def test_session_horizon_unclipped_midday():
    assert hf.session_horizon(60, _t(13, 0)) == 60


def test_session_horizon_clipped_near_close():
    assert hf.session_horizon(60, _t(15, 0)) == 30      # 30m to close


def test_session_horizon_floor_5m():
    assert hf.session_horizon(60, _t(15, 29)) == 5       # 1m to close → floor 5


def test_session_horizon_postclose_unclipped():
    assert hf.session_horizon(60, _t(16, 0)) == 60       # no session left → pure projection


def test_eval_asof_caps_postclose():
    capped = hf.eval_asof(as_of=_t(16, 30))
    assert capped == datetime.datetime.combine(
        datetime.date(2026, 7, 6), datetime.time(15, 30), tzinfo=IST)


def test_eval_asof_intraday_unchanged():
    assert hf.eval_asof(as_of=_t(13, 0)) == _t(13, 0)


def test_horizon_band_factor_unity_at_60m():
    factor, h = hf.horizon_band_factor(60, _t(13, 0))
    assert h == 60 and factor == pytest.approx(1.0)


def test_k_known_sigma_multiples():
    assert ce._k(0.68) == pytest.approx(0.9945, abs=1e-3)
    assert ce._k(0.95) == pytest.approx(1.9600, abs=1e-3)


def test_k_monotone_and_clamped():
    assert ce._k(0.50) < ce._k(0.68) < ce._k(0.95)
    assert ce._k(0.0) == ce._k(0.05)      # clamped low
    assert ce._k(1.0) == ce._k(0.95)      # clamped high
