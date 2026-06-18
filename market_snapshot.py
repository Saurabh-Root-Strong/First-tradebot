"""
market_snapshot.py — L2 aggregation: ONE coherent market state per evaluation instant.

Phase 1 of the layering refactor. Previously every Trade-Book panel re-derived the
same per-index primitives independently: the regime forecast was computed 3-4x per
render (radar, conductor, trade-book forecasts, per-trade risk badge), and the
playbook + timeframe-delta were recomputed inside the conductor AND again in their
own panels. That wasted compute and was the structural cause of the as_of mixing
bug — there was no single instant of truth, so panels could silently disagree.

MarketSnapshot computes each primitive ONCE for a given `as_of` and lazily caches
it, injecting the shared components into the conductor so the whole page reflects
exactly one market state. Lookahead-free: every read honours `as_of` (None = live).

    snap = MarketSnapshot(as_of)         # as_of=None → live 'now'
    snap.market                          # regime market_forecast (incl per_index)
    snap.forecast(sym)  snap.playbook(sym)  snap.footprint(sym)  snap.stance(sym)
    snap.stances()                       # {sym: conductor stance}

Parity contract (proven in tests): with a fixed as_of, every accessor returns a
value equal to calling the underlying module function directly — this is a pure
de-duplication layer, not a behaviour change.
"""
from __future__ import annotations

import datetime
from typing import Optional

import intraday_tf as _ITF
import opening_playbook as _OP
import regime_forecast as _RF
import session_conductor as _SC
import timeframe_delta as _TD
from core.constants import INDEX_SYMBOLS


class MarketSnapshot:
    """One render's worth of market state, evaluated at `as_of` (None = live)."""

    def __init__(self, as_of: Optional[datetime.datetime] = None) -> None:
        self.as_of = as_of
        self._market: Optional[dict] = None
        self._pb_all: Optional[dict] = None
        self._fp: dict = {}
        self._td: dict = {}
        self._stance: dict = {}

    # ── L2 aggregates (computed once) ───────────────────────────────────────────
    @property
    def market(self) -> dict:
        """Regime forecast for the whole market (its per_index == forecast_index)."""
        if self._market is None:
            self._market = _RF.market_forecast(self.as_of)
        return self._market

    @property
    def playbook_all(self) -> dict:
        """Opening playbook for all indices (its per_index == playbook_index)."""
        if self._pb_all is None:
            self._pb_all = _OP.playbook_all(self.as_of)
        return self._pb_all

    # ── per-index primitives (lazy, cached) ─────────────────────────────────────
    def forecast(self, sym: str) -> dict:
        return (self.market.get("per_index") or {}).get(sym, {})

    def playbook(self, sym: str) -> dict:
        return (self.playbook_all.get("per_index") or {}).get(sym, {})

    def footprint(self, sym: str) -> dict:
        if sym not in self._fp:
            self._fp[sym] = _ITF.analyze(sym, as_of=self.as_of)
        return self._fp[sym]

    def _delta(self, sym: str) -> dict:
        if sym not in self._td:
            self._td[sym] = _TD.analyze_index(sym, as_of=self.as_of)
        return self._td[sym]

    # ── L3 fused stance (reuses the primitives above — no recompute) ────────────
    def stance(self, sym: str) -> dict:
        if sym not in self._stance:
            self._stance[sym] = _SC.conduct(
                sym, as_of=self.as_of,
                pb=self.playbook(sym), rf=self.forecast(sym), td=self._delta(sym))
        return self._stance[sym]

    def stances(self) -> dict:
        return {s: self.stance(s) for s in INDEX_SYMBOLS}
