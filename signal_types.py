"""
signal_types.py — the canonical signal vocabulary (one Direction language).

The codebase speaks four dialects for the SAME idea:
    CE / PE                     option trades (ledger)
    LONG / SHORT / FLAT         session conductor
    BULLISH / BEARISH / NEUTRAL footprint, playbook, regime, smart-money
    UP / DOWN / SIDEWAYS        DCM prediction engine
plus F&O ACTIONS (BUY CE / WRITE PE / BUY FUT …) whose bullish-or-bearish meaning is
non-obvious — WRITE PE is BULLISH, WRITE CE is BEARISH. That ambiguity caused real
confusion ("a WRITE PE looked like a bearish put trade").

This module is the single translation point. Every dialect → one canonical `Bias`,
and one place maps Bias / Action → its display (arrow · colour · label · stance hint).
Pure and dependency-free, so any layer (signals, engine, presentation) can share it.
"""
from __future__ import annotations

from enum import Enum


class Bias(str, Enum):
    BULL = "BULLISH"
    BEAR = "BEARISH"
    NEUTRAL = "NEUTRAL"


# Canonical palette — ONE green / red / grey (ends the #22c55e-vs-#4ade80 drift).
_CLR   = {Bias.BULL: "#22c55e", Bias.BEAR: "#ef4444", Bias.NEUTRAL: "#94a3b8"}
_ARROW = {Bias.BULL: "▲",       Bias.BEAR: "▼",       Bias.NEUTRAL: "•"}

# Every direction dialect token → canonical Bias.
_DIR_MAP = {
    "BULLISH": Bias.BULL, "BEARISH": Bias.BEAR, "NEUTRAL": Bias.NEUTRAL,
    "LONG": Bias.BULL,    "SHORT": Bias.BEAR,   "FLAT": Bias.NEUTRAL,
    "CE": Bias.BULL,      "PE": Bias.BEAR,
    "UP": Bias.BULL,      "DOWN": Bias.BEAR,    "SIDEWAYS": Bias.NEUTRAL,
    "BULL": Bias.BULL,    "BEAR": Bias.BEAR,
    "THIN": Bias.NEUTRAL,                       # low-conviction = no directional bias
}

# F&O action → the bias it EXPRESSES (the non-obvious mapping).
_ACTION_BIAS = {
    "BUY CE": Bias.BULL, "WRITE PE": Bias.BULL, "BUY FUT": Bias.BULL,
    "BUY PE": Bias.BEAR, "WRITE CE": Bias.BEAR, "SELL FUT": Bias.BEAR,
    "NO TRADE": Bias.NEUTRAL, "HOLD": Bias.NEUTRAL,
}
# Plain-English clarifier so a premium-sell never reads as the wrong direction.
_ACTION_HINT = {
    "WRITE PE": "sell puts · bullish", "WRITE CE": "sell calls · bearish",
    "BUY FUT": "long futures", "SELL FUT": "short futures",
}


def bias_of(raw) -> Bias:
    """Map any direction dialect token (or Bias) to the canonical Bias."""
    if isinstance(raw, Bias):
        return raw
    if raw is None:
        return Bias.NEUTRAL
    return _DIR_MAP.get(str(raw).strip().upper(), Bias.NEUTRAL)


def action_bias(action) -> Bias:
    """The directional bias an F&O action expresses (WRITE PE → BULLISH)."""
    if action is None:
        return Bias.NEUTRAL
    return _ACTION_BIAS.get(str(action).strip().upper(), Bias.NEUTRAL)


def action_hint(action) -> str:
    """Plain-English stance clarifier for WRITE/FUT actions ('' for plain BUY)."""
    if action is None:
        return ""
    return _ACTION_HINT.get(str(action).strip().upper(), "")


def color(raw) -> str:
    return _CLR[bias_of(raw)]


def arrow(raw) -> str:
    return _ARROW[bias_of(raw)]


def view(raw) -> dict:
    """One-stop view-model for a direction token: {bias, label, arrow, color}."""
    b = bias_of(raw)
    return {"bias": b.value, "label": b.value, "arrow": _ARROW[b], "color": _CLR[b]}
