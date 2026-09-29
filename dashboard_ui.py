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


def _apply_trade_cap(opens: list, closed: list, cap) -> tuple:
    """The charts-page "N trades / index / day" discipline switch, as a pure filter over the
    day's scout episodes. Returns (opens_kept, closed_kept, skipped).

    Keeps the FIRST `cap` episodes per index by open time — the only CAUSAL way to pick:
    at the moment a trade opens you know how many you have already taken today, not which
    of the day's trades will turn out best. Measured over the last 30 sessions (2026-09-29):
    trades 3+ per index lost Rs67k live at a 30% win rate while the first two roughly broke
    even; capping at 2 beat 99.6% of random 2-trade picks. The poller still LOGS every
    episode — this filters what the board/ledger count, so the uncapped book stays gradeable.

    `cap` falsy -> no filter. Input lists keep their order; skipped rows are copies tagged
    with cap_state ("open"/"closed") and trade_no (1-based order of the day for that index).
    """
    if not cap:
        return list(opens), list(closed), []
    tagged = [(e, "open") for e in opens] + [(e, "closed") for e in closed]

    def _key(item):
        e = item[0]
        ts = e.get("open_ts")
        # open_ts carries seconds (a FLIP close + the next NEW can share a minute); open_t
        # is the HH:MM fallback for rows built without it.
        return (str(e.get("sym") or ""), str(ts) if ts is not None else "",
                str(e.get("open_t") or ""))

    kept, rank, skipped = set(), {}, []
    for e, state in sorted(tagged, key=_key):
        s = e.get("sym")
        k = rank.get(s, 0)
        rank[s] = k + 1
        if k < cap:
            kept.add(id(e))
        else:
            skipped.append({**e, "cap_state": state, "trade_no": k + 1})
    return ([e for e in opens if id(e) in kept], [e for e in closed if id(e) in kept],
            skipped)


def _cap_muted(new_events: list, day_log: list, cap) -> bool:
    """Should this batch of scout alerts stay SILENT under the N-trades cap?

    True only when EVERY alert in `new_events` belongs to an index that has already opened
    more than `cap` positions today (counted as NEW rows in `day_log`, which must include
    `new_events`). The poller holds one position per index, so once an index's (cap+1)-th
    NEW has fired, every later alert on it — that NEW, its SL/target/flip/timeout — belongs
    to a trade the cap skips. A batch mixing in any in-cap index still beeps."""
    if not cap or not new_events:
        return False
    n_new: dict = {}
    for a in day_log:
        if a.get("kind") == "NEW":
            n_new[a.get("label")] = n_new.get(a.get("label"), 0) + 1
    return all(n_new.get(a.get("label"), 0) > cap for a in new_events)


def _scout_trade_status(entry, now, sl, tgt, peak) -> str:
    """Live trajectory of an OPEN scout position on its option premium (NOT a close — the
    poller alone closes on SL/TARGET/FLIP). Shows WHY a position is still open:
      🎯/🛑  = already past target/SL (close pending on the next poll)
      ↩ pullback = ran up >=20% then gave back >=15pts of that gain
      ▲ / ▼  = running toward target / drawing toward SL, with the current premium move
    entry/sl/tgt are the alert-logged premium levels; the running ▲/▼ line quotes the
    bracket back from THOSE levels (never a hardcoded pair) so it cannot drift from the
    bracket the engine actually traded."""
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
        t = f" → tgt {tgt / entry - 1.0:+.0%}" if tgt else ""
        return f"▲ {g:+.0%}{t}"
    d = f" → SL {sl / entry - 1.0:+.0%}" if sl else ""
    return f"▼ {g:+.0%}{d}"
