"""Unit tests for cost_model — the L4 transaction-cost math (gross R → net R).

Pure, offline, decision-critical: this is what turns a paper edge into a NET edge,
so a regression here would silently re-inflate dead strategies. No network / no token.
"""
import pytest

import cost_model as cm


def test_cost_r_basic():
    # entry 100, sl 68 → risk_frac 0.32 → cost = (80/1e4)/0.32
    assert cm.cost_r(100, 68, bps=80) == pytest.approx(0.008 / 0.32)


def test_cost_r_tighter_stop_costs_more():
    # cost-in-R scales inversely with the stop fraction (the whole thesis)
    wide = cm.cost_r(100, 50)     # 50% stop
    tight = cm.cost_r(100, 90)    # 10% stop
    assert tight > wide > 0


def test_cost_r_degenerate_inputs_return_zero():
    # never inflate an edge on bad input
    assert cm.cost_r(None, 50) == 0.0
    assert cm.cost_r("x", 50) == 0.0
    assert cm.cost_r(0, 50) == 0.0
    assert cm.cost_r(-5, 3) == 0.0
    assert cm.cost_r(100, 100) == 0.0   # zero risk_frac


def test_net_r_subtracts_cost():
    g = 0.50
    n = cm.net_r(g, 100, 68, bps=80)
    assert n is not None and n < g
    assert n == pytest.approx(round(g - cm.cost_r(100, 68, 80), 2))


def test_net_r_none_gross_is_none():
    assert cm.net_r(None, 100, 68) is None
