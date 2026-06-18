"""
cost_model.py — L4 transaction-cost model: gross R → net R, per trade.

The paper-trade ledger records GROSS outcomes (no brokerage, spread, or slippage).
For option BUYING the dominant cost is the bid-ask spread crossed twice, plus STT /
exchange txn / GST / stamp / brokerage on premium. Expressed per trade in R-multiple
terms so it is comparable across trades — exactly the structure cost_overlay.py uses:

    risk_frac = |entry - sl| / entry            # the stop as a fraction of ENTRY premium
    cost_R    = (round_trip_bps / 1e4) / risk_frac
    net_r     = gross_r - cost_R

Tight stops (small risk_frac) amplify cost-in-R — which is exactly where intraday
option scalps live, so cost is NOT a rounding error here. `round_trip_bps` is the
round-trip cost as bps of option premium; liquid index ATM/near-ATM ≈ 60-120 bps
(spread-dominated), OTM and the smaller indices (BANK/FIN/MIDCAP) wider. Configurable
and always SHOWN, never hidden, so position sizing is decided on NET, not gross.

This is a pure, injectable component (no ledger/UI coupling): the accounting layer
calls it; a future backtest uses the same model for live/backtest parity.
"""
from __future__ import annotations

from typing import Optional

# Round-trip cost in bps of option premium (spread + STT + exch txn + GST + stamp +
# brokerage). Conservative default for liquid index near-ATM options; tune per broker
# and per index (the thin indices trade wider). Exposed so the UI can show it.
DEFAULT_ROUND_TRIP_BPS = 80.0


def cost_r(entry, sl, bps: float = DEFAULT_ROUND_TRIP_BPS) -> float:
    """Transaction cost for one trade, in R. Needs entry premium + stop (premium).
    Returns 0.0 when inputs are missing/degenerate (never inflates an edge)."""
    try:
        entry = float(entry); sl = float(sl)
    except (TypeError, ValueError):
        return 0.0
    if entry <= 0:
        return 0.0
    risk_frac = abs(entry - sl) / entry
    if risk_frac <= 0:
        return 0.0
    return (bps / 1e4) / risk_frac


def net_r(gross_r, entry, sl, bps: float = DEFAULT_ROUND_TRIP_BPS) -> Optional[float]:
    """Net-of-cost R for one trade (None if gross R is undefined)."""
    if gross_r is None:
        return None
    return round(float(gross_r) - cost_r(entry, sl, bps), 2)
