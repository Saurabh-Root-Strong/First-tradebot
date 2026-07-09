"""dashboard_theme.py — visual theme constants + tooltip copy for the dashboard.

Second slice of the dashboard monolith split: pure DATA constants (per-index colours, base
backgrounds, the monospace style, and the long help-text tooltips) — no logic, no imports of
dashboard/business modules. dashboard.py imports these back under the same names so every
reference is unchanged. Keep this a LEAF (data only).

(The big _CSS block still lives in dashboard.py and is a separate slice — it belongs in
assets/*.css, which Dash auto-serves, retiring the index_string.replace hack.)
"""
from __future__ import annotations

# ── per-index palette ───────────────────────────────────────────────────────────
COLORS = {
    "NSE:NIFTY50-INDEX":    "#00d4ff",
    "NSE:NIFTYBANK-INDEX":  "#ff6b9d",
    "NSE:FINNIFTY-INDEX":   "#4ade80",
    "NSE:MIDCPNIFTY-INDEX": "#fbbf24",
}
FILLS = {
    "NSE:NIFTY50-INDEX":    "rgba(0,212,255,0.07)",
    "NSE:NIFTYBANK-INDEX":  "rgba(255,107,157,0.07)",
    "NSE:FINNIFTY-INDEX":   "rgba(74,222,128,0.07)",
    "NSE:MIDCPNIFTY-INDEX": "rgba(251,191,36,0.07)",
}

# ── base surfaces + type ────────────────────────────────────────────────────────
BG      = "#080d14"
BG_CARD = "#0f1623"
BG_SIDE = "#0a1020"
MONO    = {"fontFamily": "'Courier New', Courier, monospace"}

# ── tooltip help copy (hover text on the scout board / forecast) ─────────────────
_TIP_VERDICT = ("TRADE = anchor FLOW agrees AND >=1 INDEPENDENT family (cross/fut) "
                "agrees, |strength|>=0.22. NO-TRADE = not enough independent agreement. "
                "Direction is decision-support only — measured negative-EV in backtests; "
                "the RANGE band is the validated part, not the arrow.")
_TIP_STR = ("strength [-1,1] = weighted blend: flow 0.40 (delta-adjusted OI demand), "
            "div 0.25 (price vs OI), fut 0.20 (futures), cross 0.15 (OI-tilt flip). "
            "conf = |strength| rescaled — NOT a win probability.")
_TIP_AGREE = ("Independent families agreeing with the call: flow (anchor) + cross "
              "(OI-tilt flip) + fut (futures). div shares data with flow so it is "
              "confirmation only and never counted here.")
_TIP_RANGE = ("60-MIN volatility cone: spot ± ~1 std-dev of expected move over the "
              "next hour. MEASURED (ledger, n=236): price CLOSES inside ~77% of the "
              "time, but STAYS inside the whole hour only ~52% (it wicks out ~half "
              "the time, then closes back in). Destination band, not a fence. HOW "
              "FAR it can travel, not which way — use for stops / targets / sizing.")
_TIP_BAND = ("FORWARD prediction made RIGHT NOW for the NEXT timeframe — it has not "
             "happened yet. The arrow (UP/DOWN/RANGE) is the lean; the band is spot ± "
             "~1 std-dev of expected move over the coming TF (15m/etc cone, narrower "
             "than the 60m range above). It self-grades once the window elapses: "
             "⇒ pending → then HIT/MISS on direction and band ✓ (closed inside, ~77%) "
             "/ band ✗ (broke out, the big-move tail). HOW FAR, not which way.")
_TIP_TRIG = ("Trade lifecycle for the CURRENT unbroken TRADE run: the minute THIS run "
             "began, the INDEX level + ATM premium then, SL/target, live P&L, and the "
             "manage call (CLOSE / HOLD / BOOK). This time RESETS if the gate blinked "
             "NO-TRADE, and can shift while the newest bar is still forming. NOTE: the "
             "day-ledger's 'since' is the ORIGINAL open, held THROUGH NO-TRADE gaps, so "
             "it reads earlier. Same position, two clocks — this = current leg, "
             "ledger = first entry.")
