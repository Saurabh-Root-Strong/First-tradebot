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


# ── injected CSS string ── dashboard keeps the app.index_string.replace(…) inject;
# this is only the STRING (belongs in assets/*.css eventually; kept as a string for now).
_CSS = """
<style>
/* ── Global ──────────────────────────────────────────────────────── */
body { background:#030810 !important; overflow-x:hidden; }
* { box-sizing:border-box; }

/* ── Scrollbars ──────────────────────────────────────────────────── */
::-webkit-scrollbar { width:4px; height:4px; }
::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:#1a3050; border-radius:2px; }
::-webkit-scrollbar-thumb:hover { background:#2d5080; }

/* ── Keyframes ───────────────────────────────────────────────────── */
@keyframes live-pulse {
  0%,100%{ opacity:1; box-shadow:0 0 0 0 rgba(34,197,94,.5); }
  60%    { opacity:.7; box-shadow:0 0 0 7px rgba(34,197,94,0); }
}
@keyframes amber-pulse {
  0%,100%{ box-shadow:0 0 0 0 rgba(245,158,11,.5); }
  60%    { box-shadow:0 0 0 7px rgba(245,158,11,0); }
}
@keyframes red-pulse {
  0%,100%{ box-shadow:0 0 0 0 rgba(239,68,68,.6); }
  60%    { box-shadow:0 0 0 7px rgba(239,68,68,0); }
}
@keyframes glow-bull {
  0%,100%{ box-shadow:0 0 12px rgba(34,197,94,.1),0 2px 20px rgba(0,0,0,.5); }
  50%    { box-shadow:0 0 32px rgba(34,197,94,.4),0 2px 20px rgba(0,0,0,.5); }
}
@keyframes glow-bear {
  0%,100%{ box-shadow:0 0 12px rgba(239,68,68,.1),0 2px 20px rgba(0,0,0,.5); }
  50%    { box-shadow:0 0 32px rgba(239,68,68,.4),0 2px 20px rgba(0,0,0,.5); }
}
@keyframes bar-grow { from { width:0 !important; } }
@keyframes conf-fill { from { width:0 !important; } }
@keyframes slide-up {
  from { opacity:0; transform:translateY(12px); }
  to   { opacity:1; transform:translateY(0); }
}
@keyframes ticket-appear {
  from { opacity:0; transform:translateY(-8px) scale(.98); }
  to   { opacity:1; transform:translateY(0) scale(1); }
}
@keyframes num-flash {
  0%  { background:rgba(251,191,36,.15); border-radius:3px; }
  100%{ background:transparent; }
}

/* ── Live dots ───────────────────────────────────────────────────── */
.live-dot {
  width:8px; height:8px; border-radius:50%;
  background:#22c55e; display:inline-block; flex-shrink:0;
  animation:live-pulse 1.8s ease-out infinite;
}
.live-dot-amber { background:#f59e0b !important; animation:amber-pulse 1.8s ease-out infinite; }
.live-dot-dead  { background:#ef4444 !important; animation:red-pulse .9s ease-out infinite; }

/* ── Signal glow cards ───────────────────────────────────────────── */
.sig-bull { animation:glow-bull 2.8s ease-in-out infinite; }
.sig-bear { animation:glow-bear 2.8s ease-in-out infinite; }
.sig-card { animation:slide-up .35s cubic-bezier(.22,.68,0,1.2); }

/* ── Signal strength bar ─────────────────────────────────────────── */
.sbar-track {
  height:6px; border-radius:4px; overflow:hidden;
  background:rgba(255,255,255,.04); margin:9px 0 7px;
}
.sbar-fill {
  height:100%; border-radius:4px;
  animation:bar-grow .8s cubic-bezier(.22,.68,0,1.2);
  transition:width .6s ease;
}

/* ── Confidence bar ──────────────────────────────────────────────── */
.conf-track {
  height:4px; border-radius:3px; overflow:hidden;
  background:rgba(255,255,255,.05);
}
.conf-fill { height:100%; border-radius:3px; animation:conf-fill .9s ease-out; }

/* ── Timeframe pill badges ───────────────────────────────────────── */
.tf-pill {
  display:inline-block; padding:2px 7px; border-radius:20px;
  font-size:.55rem; font-family:'Courier New',monospace;
  font-weight:700; letter-spacing:.07em; margin:1px 2px;
  border:1px solid transparent; white-space:nowrap;
  transition:all .15s ease;
}
.tf-pill:hover { filter:brightness(1.3); }
.tf-bull { background:rgba(34,197,94,.14);  color:#4ade80; border-color:rgba(34,197,94,.3); }
.tf-bear { background:rgba(239,68,68,.14);  color:#f87171; border-color:rgba(239,68,68,.3); }
.tf-neut { background:rgba(148,163,184,.07);color:#475569; border-color:rgba(148,163,184,.13); }

/* ── Metric chips ────────────────────────────────────────────────── */
.metric-chip {
  display:inline-flex; align-items:center; gap:5px;
  padding:4px 10px; border-radius:5px; font-size:.62rem;
  background:rgba(255,255,255,.04);
  border:1px solid rgba(255,255,255,.07); margin:2px 3px;
  font-family:'Courier New',monospace;
  transition:background .15s ease;
}
.metric-chip:hover { background:rgba(255,255,255,.07); }

/* ── Section label ───────────────────────────────────────────────── */
.sec-label {
  font-size:.54rem; letter-spacing:.22em; color:#1e3a5f;
  font-weight:700; text-transform:uppercase;
}

/* ── Nav card hover ──────────────────────────────────────────────── */
[id^="nav-"] { transition:all .18s ease !important; }
[id^="nav-"]:hover {
  filter:brightness(1.45) !important;
  transform:translateX(5px) !important;
  box-shadow:3px 0 22px rgba(0,0,0,.5) !important;
}

/* ── Overview card depth + hover lift ───────────────────────────── */
.depth-card {
  box-shadow:0 4px 28px rgba(0,0,0,.5),0 1px 3px rgba(0,0,0,.6);
  transition:transform .2s ease,box-shadow .2s ease;
}
.depth-card:hover {
  transform:translateY(-2px);
  box-shadow:0 8px 36px rgba(0,0,0,.65),0 2px 6px rgba(0,0,0,.7);
}

/* ── Trade ticket entrance ───────────────────────────────────────── */
.trade-ticket { animation:ticket-appear .4s cubic-bezier(.22,.68,0,1.2); }

/* ── OC table row hover ──────────────────────────────────────────── */
.oc-row:hover td { background:rgba(255,255,255,.03) !important; }

/* ── Dash dropdown dark override ─────────────────────────────────── */
.Select-control,.Select-menu-outer { background:#0c1522 !important; border-color:#1e2d40 !important; }
.Select-option.is-focused { background:#1e2d40 !important; }
.Select-value-label,.Select-placeholder { color:#94a3b8 !important; }
.Select-option { color:#94a3b8 !important; }

/* ── Velocity progress bars ──────────────────────────────────────── */
.vbar-track {
  height:5px; border-radius:3px; overflow:hidden;
  background:rgba(255,255,255,.04);
}
.vbar-fill { height:100%; border-radius:3px; transition:width .7s ease; }

/* ── Verdict glow text ───────────────────────────────────────────── */
.verdict-bull { color:#22c55e !important; text-shadow:0 0 22px rgba(34,197,94,.55); }
.verdict-bear { color:#ef4444 !important; text-shadow:0 0 22px rgba(239,68,68,.55); }
.verdict-neut { color:#94a3b8 !important; }

/* ── Header shimmer gradient line ────────────────────────────────── */
.header-line {
  height:1px; margin-bottom:14px;
  background:linear-gradient(90deg,transparent,#00d4ff33,#ff6b9d33,#4ade8033,transparent);
}

/* ── Rec card hover lift ─────────────────────────────────────────── */
.rec-card { transition:all .2s ease; }
.rec-card:hover {
  transform:translateY(-3px);
  box-shadow:0 14px 32px rgba(0,0,0,.55) !important;
}

/* ── Index Prediction cards ──────────────────────────────────────── */
.pred-card {
  transition:transform .2s ease,box-shadow .2s ease;
  animation:slide-up .4s cubic-bezier(.22,.68,0,1.2);
}
.pred-card:hover {
  transform:translateY(-3px);
  box-shadow:0 14px 36px rgba(0,0,0,.65) !important;
}
.pred-meter-track {
  height:6px; border-radius:4px; overflow:hidden;
  background:rgba(255,255,255,.05); margin-bottom:12px;
}
.pred-meter-fill {
  height:100%; border-radius:4px;
  transition:width .9s cubic-bezier(.22,.68,0,1.2);
}
.pred-metric {
  flex:1; min-width:0; padding:5px 7px; border-radius:5px;
  background:rgba(255,255,255,.03);
  border:1px solid rgba(255,255,255,.055);
}
.pred-regime-badge {
  display:inline-block; padding:3px 8px; border-radius:4px;
  background:#060e1a; border:1px solid #1a2535;
  margin:0 3px 3px 0;
}
.pred-section-header {
  font-size:.46rem; letter-spacing:.16em;
  color:#1e3a5f; margin-bottom:5px; display:block;
}

/* ── Signal score badge ──────────────────────────────────────────── */
.score-badge {
  display:inline-block; padding:2px 8px; border-radius:12px;
  font-size:.58rem; font-weight:700; font-family:'Courier New',monospace;
  border:1px solid transparent;
}
.score-bull { background:rgba(34,197,94,.12); color:#4ade80; border-color:rgba(34,197,94,.25); }
.score-bear { background:rgba(239,68,68,.12); color:#f87171; border-color:rgba(239,68,68,.25); }
.score-neut { background:rgba(148,163,184,.07); color:#64748b; border-color:rgba(148,163,184,.15); }
</style>
"""
