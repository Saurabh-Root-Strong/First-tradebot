"""broker_costs.py — what a real index-option round trip costs at Zerodha, in rupees.

The ledger's `₹/lot` column is the RAW premium move times the lot size. It is the number the
exchange prints, not the number that reaches the account. Between them sit six charges, and on
a 1-lot index-option trade they are not a rounding error: the flat ₹20-per-order brokerage
alone is ₹40 a round trip, against a median winning trade of a few hundred rupees.

THE ONE THING TO UNDERSTAND: brokerage is per ORDER, not per lot. Everything else scales with
turnover. So the cost as a FRACTION of P&L collapses as size grows -- at 1 lot brokerage is
most of the bill, at 10 lots it is nearly free. A ledger that shows only the 1-lot figure will
overstate the drag for anyone trading size, and a ledger that ignores costs understates it for
everyone. Hence `lots` is an explicit argument and the breakdown is always available.

RATES (Zerodha, index options, F&O segment). These change with budgets and exchange circulars
-- VERIFIED 2026-08-05 against zerodha.com/charges, the STT support article and the
1-Apr-2026 revision bulletin (SOURCES at the bottom of this docstring). They are constants
HERE, in one place, so a change is a one-line edit and never a hunt through the dashboard:

    brokerage      Rs 20 per executed order, flat  (buy + sell = Rs 40)
    STT            0.15%  of SELL-side premium turnover only
    exchange txn   0.03553% of premium turnover, BOTH sides (NSE options)
    SEBI fees      0.0001% of premium turnover (Rs 10 per crore)
    GST            18% on (brokerage + exchange txn + SEBI fees)
    stamp duty     0.003% of BUY-side premium turnover only

STT HISTORY -- the biggest variable leg, and the easiest one to get stale:
    0.0625%  ->  0.10%   on 2024-10-01
    0.10%    ->  0.15%   on 2026-04-01  (Budget 2026-27)   <-- CURRENT
A stale 0.10% understates the STT leg by a third and the whole bill by roughly 8%.

THE EXPIRY TRAP, deliberately not modelled: an option SOLD pays 0.15% of the PREMIUM, but one
left to expire in-the-money and EXERCISED pays 0.15% of the INTRINSIC VALUE -- on a deep ITM
contract that is orders of magnitude larger. This ledger always closes by SELLING (band / SL /
flip / timeout / squared off at the bell), so premium-side STT is the right model here. If a
position is ever allowed to expire ITM, this function will understate the bill badly.

Deliberately NOT included: DP charges (equity delivery only, not F&O), auto-square-off
penalties, and the bid-ask spread. The spread is a real cost but it is not a CHARGE -- it is
already inside the entry/exit premiums the ledger records, so adding it here would double-count.

SOURCES (fetched 2026-08-05):
    https://zerodha.com/charges/
    https://support.zerodha.com/category/account-opening/resident-individual/ri-charges/
        articles/how-is-the-securities-transaction-tax-stt-calculated
    https://zerodha.com/marketintel/bulletin/445377/
        revision-in-stt-securities-transaction-tax-from-1st-april-2026
"""
from __future__ import annotations

# ── rate table — the only place these numbers live ────────────────────────────────
BROKERAGE_PER_ORDER = 20.0      # Rs, flat, per executed order
ORDERS_PER_ROUNDTRIP = 2        # one buy + one sell
STT_SELL = 0.0015               # 0.15%  of sell premium turnover, sell side only (1-Apr-2026)
EXCH_TXN = 0.0003553            # 0.03553% of premium turnover, both sides (NSE options)
SEBI_FEES = 0.000001            # 0.0001% of premium turnover
GST_RATE = 0.18                 # 18% on brokerage + exchange txn + SEBI
STAMP_BUY = 0.00003             # 0.003% of buy premium turnover (buy side only)


def roundtrip(entry_prem: float | None, exit_prem: float | None,
              lot_size: int | None, lots: int = 1) -> dict | None:
    """Full round-trip charge breakdown for a bought-then-sold option position.

    entry_prem / exit_prem are PER-UNIT premiums (what the ledger stores); lot_size is the
    index lot (NIFTY 65 / BANK 30 / FIN 60 / MIDCAP 120). Returns None when any input is
    missing so callers can render "—" rather than a confident zero.

    Direction note: this models BUY-then-SELL, which is every trade in the scout ledger (the
    arrow buys a naked CE/PE). For a short-first position the STT and stamp legs would swap
    sides; that is not a case this board produces, so it is not silently approximated here.
    """
    if entry_prem is None or exit_prem is None or not lot_size:
        return None
    try:
        entry_prem = float(entry_prem); exit_prem = float(exit_prem)
        qty = int(lot_size) * max(1, int(lots))
    except (TypeError, ValueError):
        return None
    if entry_prem < 0 or exit_prem < 0 or qty <= 0:
        return None

    buy_turnover = entry_prem * qty
    sell_turnover = exit_prem * qty
    turnover = buy_turnover + sell_turnover

    brokerage = BROKERAGE_PER_ORDER * ORDERS_PER_ROUNDTRIP
    stt = sell_turnover * STT_SELL
    exch = turnover * EXCH_TXN
    sebi = turnover * SEBI_FEES
    stamp = buy_turnover * STAMP_BUY
    gst = (brokerage + exch + sebi) * GST_RATE
    total = brokerage + stt + exch + sebi + stamp + gst

    return {
        "brokerage": round(brokerage, 2),
        "stt": round(stt, 2),
        "exchange": round(exch, 2),
        "sebi": round(sebi, 2),
        "stamp": round(stamp, 2),
        "gst": round(gst, 2),
        "total": round(total, 2),
        "buy_turnover": round(buy_turnover, 2),
        "sell_turnover": round(sell_turnover, 2),
        "qty": qty,
    }


def roundtrip_total(entry_prem, exit_prem, lot_size, lots: int = 1) -> float | None:
    """Just the rupee total (None if not computable)."""
    c = roundtrip(entry_prem, exit_prem, lot_size, lots)
    return c["total"] if c else None


def net_pnl(entry_prem, exit_prem, lot_size, lots: int = 1) -> tuple:
    """(gross_rs, cost_rs, net_rs) for a bought-then-sold option. (None, None, None) if the
    inputs cannot support the arithmetic."""
    c = roundtrip(entry_prem, exit_prem, lot_size, lots)
    if c is None:
        return (None, None, None)
    gross = (float(exit_prem) - float(entry_prem)) * c["qty"]
    return (round(gross, 2), c["total"], round(gross - c["total"], 2))


def explain(entry_prem, exit_prem, lot_size, lots: int = 1) -> str:
    """One-line human breakdown for a tooltip."""
    c = roundtrip(entry_prem, exit_prem, lot_size, lots)
    if c is None:
        return "costs unavailable (missing entry/exit premium or lot size)"
    return (f"brokerage ₹{c['brokerage']:.0f} · STT ₹{c['stt']:.0f} · "
            f"exchange ₹{c['exchange']:.2f} · SEBI ₹{c['sebi']:.2f} · "
            f"stamp ₹{c['stamp']:.2f} · GST ₹{c['gst']:.2f}  =  ₹{c['total']:.0f} "
            f"on {c['qty']} qty")
