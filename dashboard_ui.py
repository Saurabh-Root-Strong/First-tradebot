"""dashboard_ui.py — PURE, leaf UI helpers extracted from dashboard.py.

First slice of the dashboard monolith split (7.5k LOC): functions with NO Dash callback,
NO app dependency, NO module-global state — pure input -> string/dict. Living here they are
unit-testable in isolation and keep growing dashboard.py from importing itself for trivia.
dashboard.py imports these back under the same names, so all call sites are unchanged.

Keep this module a LEAF: stdlib only, no imports of dashboard/business modules.
"""
from __future__ import annotations


def _slug(sym: str) -> str:
    """Fyers symbol -> a Dash-element-id-safe slug (NSE:NIFTY50-INDEX -> NSE-NIFTY50-INDEX)."""
    return sym.replace(":", "-").replace(".", "-")


def _fmt_cr(v) -> str:
    """Format a Rs-Cr value compactly for narrow chips: 12345 -> +12K, -2500 -> -2.5K."""
    if v is None:
        return "—"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "—"
    sign = "+" if v >= 0 else ""
    a    = abs(v)
    if a >= 1_00_000:                    # >= 1 lakh Cr -> show in L
        return f"{sign if v >= 0 else '-'}{a/1_00_000:.1f}L"
    if a >= 10_000:                      # >= 10,000 Cr -> "12K"
        return f"{sign if v >= 0 else '-'}{a/1_000:.0f}K"
    if a >= 1_000:                       # >= 1,000 Cr  -> "1.2K"
        return f"{sign if v >= 0 else '-'}{a/1_000:.1f}K"
    return f"{v:+.0f}"


def _fmt_contracts(v) -> str:
    """Format FAO net contracts compactly: -259253 -> -259K, 12000000 -> +12M."""
    if v is None:
        return "—"
    try:
        v = int(v)
    except (TypeError, ValueError):
        return "—"
    a    = abs(v)
    sign = "+" if v >= 0 else "-"
    if a >= 1_000_000:
        return f"{sign}{a/1_000_000:.1f}M"
    if a >= 1_000:
        return f"{sign}{a/1_000:.0f}K"
    return f"{v:+,d}"


def _scout_trade_status(entry, now, sl, tgt, peak) -> str:
    """Live trajectory of an OPEN scout position on its option premium (NOT a close — the
    poller alone closes on SL/TARGET/FLIP). Shows WHY a position is still open:
      🎯/🛑  = already past target/SL (close pending on the next poll)
      ↩ pullback = ran up >=20% then gave back >=15pts of that gain
      ▲ / ▼  = running toward target / drawing toward SL, with the current premium move
    entry/sl/tgt are the alert-logged premium levels (SL −35%, target +65% of entry)."""
    if not entry or now is None:
        return "· no data"
    g = now / entry - 1.0                                  # current premium move
    if tgt and now >= tgt:
        return f"🎯 target {g:+.0%}"
    if sl and now <= sl:
        return f"🛑 SL {g:+.0%}"
    gp = (peak / entry - 1.0) if peak else g               # best since entry
    if gp >= 0.20 and (gp - g) >= 0.15:
        return f"↩ pullback {g:+.0%} (pk {gp:+.0%})"
    if g >= 0:
        return f"▲ {g:+.0%} → tgt +65%"
    return f"▼ {g:+.0%} → SL −35%"
