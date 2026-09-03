"""Offline unit tests for oi_moneyness — the put-side mirror is the one that matters."""
import pytest

import oi_moneyness as om


# ── the bug this module exists to prevent ────────────────────────────────────────
def test_put_above_spot_is_ITM_not_OTM():
    """A PUT above spot is IN the money. The naive 'above = OTM' rule inverts it."""
    assert om.moneyness(24300, 23873, "PE", delta=None, step=50) == "ITM"
    assert om.moneyness(24300, 23873, "CE", delta=None, step=50) == "OTM"


def test_put_below_spot_is_OTM_not_ITM():
    assert om.moneyness(23400, 23873, "PE", delta=None, step=50) == "OTM"
    assert om.moneyness(23400, 23873, "CE", delta=None, step=50) == "ITM"


def test_call_and_put_at_same_far_strike_get_opposite_labels():
    for k in (22000, 26000):
        ce = om.moneyness(k, 23873, "CE", None, 50)
        pe = om.moneyness(k, 23873, "PE", None, 50)
        assert {ce, pe} == {"ITM", "OTM"}, f"strike {k}: CE={ce} PE={pe}"


# ── delta path ───────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("delta,expect", [
    (0.95, "ITM"), (0.65, "ITM"), (0.50, "ATM"), (0.35, "ATM"), (0.20, "OTM"),
    (-0.95, "ITM"), (-0.50, "ATM"), (-0.10, "OTM"),      # PE deltas are negative
])
def test_delta_buckets_use_absolute_value(delta, expect):
    assert om.moneyness(23900, 23873, "PE", delta=delta, step=50) == expect


def test_zero_or_missing_delta_falls_back_to_strike_distance():
    """0.0 means BOTH 'no greeks' and 'worthless far OTM' — never guess between them."""
    assert om.moneyness(26000, 23873, "CE", delta=0.0, step=50) == "OTM"
    assert om.moneyness(26000, 23873, "CE", delta=None, step=50) == "OTM"
    assert om.moneyness(23880, 23873, "CE", delta=0.0, step=50) == "ATM"


# ── window ───────────────────────────────────────────────────────────────────────
def test_window_is_2n_plus_1_and_centred():
    ks = [23000 + 50 * i for i in range(60)]
    w = om.window_strikes(ks, 24000, n=11)
    assert len(w) == 23
    assert w == sorted(w)
    assert min(w) <= 24000 <= max(w)


def test_window_handles_thin_chain_without_padding():
    assert len(om.window_strikes([100, 200, 300], 200, n=11)) == 3


def test_window_empty_on_no_spot():
    assert om.window_strikes([100, 200], 0, n=11) == []


# ── snapshot / basket ────────────────────────────────────────────────────────────
def _rows(spot=23873, step=50, oi=1000):
    out = []
    for i in range(-11, 12):
        k = round((spot + i * step) / step) * step
        for side in ("CE", "PE"):
            out.append({"strike": k, "side": side, "oi": oi, "oich": 0.0,
                        "volume": 10, "delta": None})
    return out


def test_snapshot_totals_are_conserved_across_buckets():
    rows = _rows()
    s = om.bucket_snapshot(rows, 23873, step=50, use_delta=False)
    for side in ("CE", "PE"):
        assert sum(s[side][b]["n"] for b in om.BUCKETS) == 23


def test_basket_delta_ignores_vanished_legs_instead_of_zeroing_them():
    """A leg dropping out of the chain must not read as OI collapsing to zero."""
    rows = _rows()
    basket = om.pin_basket(rows, 23873, step=50, use_delta=False)
    gone = rows[0]["strike"]                       # a real strike, not the raw spot
    survivors = [r for r in rows if r["strike"] != gone]
    d = om.basket_delta(survivors, basket)
    assert d["missing"] == 2                       # the CE and PE at the dropped strike
    assert all(d[s][b]["oi_chg"] == 0.0 for s in ("CE", "PE") for b in om.BUCKETS)


def test_basket_delta_reports_real_change_at_fixed_strikes():
    rows = _rows(oi=1000)
    basket = om.pin_basket(rows, 23873, step=50, use_delta=False)
    moved = [dict(r, oi=1500) for r in rows]
    d = om.basket_delta(moved, basket)
    assert sum(d["CE"][b]["oi_chg"] for b in om.BUCKETS) == pytest.approx(500 * 23)


def test_pinned_labels_do_not_move_when_spot_moves():
    """The whole point: reclassification must not masquerade as flow."""
    rows = _rows()
    basket = om.pin_basket(rows, 23873, step=50, use_delta=False)
    before = dict(basket["labels"])
    om.basket_delta(rows, basket)                  # spot irrelevant here by construction
    assert basket["labels"] == before


# ── guards ───────────────────────────────────────────────────────────────────────
def test_bucket_pcr_returns_None_on_empty_call_side_not_zero():
    s = om.bucket_snapshot([{"strike": 100, "side": "PE", "oi": 5, "delta": None}],
                           100, step=50, use_delta=False)
    assert om.bucket_pcr(s, "ATM") is None


def test_escaped_detects_spot_leaving_the_window():
    s = om.bucket_snapshot(_rows(), 23873, step=50, use_delta=False)
    assert om.escaped(s, 23873) is False
    assert om.escaped(s, 30000) is True
