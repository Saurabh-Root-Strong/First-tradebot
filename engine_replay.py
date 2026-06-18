"""
engine_replay.py — Phase 4: event-driven replay of the headless engine.

Drives the SAME engine.py over a captured day by stepping a virtual clock and
feeding it the parquet mirrors at each tick (lookahead-free via as_of). This is the
execution-level backtest: it replays the open → track → resolve lifecycle (not just
the signal), producing a ledger scored net-of-cost via cost_model — the honest
counterpart to the live gross paper-ledger, and proof the engine is feed-agnostic.

    python engine_replay.py 2026-06-17

Fidelity notes (documented gaps, → Phase 4.1):
  * synthetic option symbols ("idx|strike|side") that round-trip within replay;
  * no captured expiry_data / option greeks (build_recommendation degrades, not crash);
  * approximate futures (mirror has near_ltp/basis, not ch/chp);
  * time-of-day phase GATES inside build_recommendation still read the wall clock,
    not the virtual clock (most of the session is conf_mult 1.0, so minor) — the
    real fix is to thread feed.now() into session_strategy.
Replay writes to a TEMP ledger db, never the live trades db.
"""
from __future__ import annotations

import datetime
import math
import sys
import tempfile
from pathlib import Path

from core.constants import IST, INDEX_SYMBOLS
from core.mirror_io import read_mirror as _read


class ReplayFeed:
    """Feed adapter backed by the captured mirrors at a virtual instant."""

    def __init__(self, date: str) -> None:
        self.date = date
        self._t: datetime.datetime | None = None

    def set_time(self, t: datetime.datetime) -> None:
        self._t = t

    def now(self) -> datetime.datetime:
        return self._t

    def spot(self, sym: str) -> float:
        df = _read("ticks", self.date, self._t, sym)
        return float(df.iloc[-1]["ltp"]) if df is not None and len(df) else 0.0

    def chain(self, sym: str):
        cs = _read("chain_snapshots", self.date, self._t, sym)
        if cs is None or cs.empty:
            return None
        cur = cs[cs["ts"] == cs["ts"].max()]
        spot = self.spot(sym) or 0.0
        k = max(spot * 0.006, 1.0)        # logistic width (~moneyness scale)
        has_g = "delta" in cur.columns    # captured greeks (Phase 4.1) vs old files

        def _num(v):
            try:
                v = float(v)
                return None if v != v else v   # NaN -> None
            except (TypeError, ValueError):
                return None

        sm: dict = {}
        for r in cur.itertuples():
            sp = float(r.strike); side = r.side
            md = _num(getattr(r, "delta", None)) if has_g else None
            iv = _num(getattr(r, "iv", None)) if has_g else None
            if md is None:
                # No captured greeks → moneyness-proxy delta (CE logistic; PE = CE-1).
                ce_d = 1.0 / (1.0 + math.exp((sp - spot) / k)) if spot else 0.5
                md = ce_d if side == "CE" else ce_d - 1.0
            sm.setdefault(sp, {})[side] = {
                "ltp": float(r.ltp or 0), "oi": float(r.oi or 0),
                "oich": float(r.oich or 0), "volume": float(r.volume or 0),
                "symbol": f"{sym}|{int(sp)}|{side}",
                "greeks": {"delta": md, "iv": (iv or 0)},
            }
        os_ = _read("oi_snapshots", self.date, self._t, sym)
        tot_c = tot_p = pcr = mp = 0.0
        if os_ is not None and len(os_):
            o = os_.iloc[-1]
            tot_c = float(o.get("total_call_oi") or 0); tot_p = float(o.get("total_put_oi") or 0)
            pcr = float(o.get("pcr") or 0); mp = float(o.get("max_pain") or 0)
        return {"sm": sm, "expiry_data": [], "tot_c": tot_c, "tot_p": tot_p, "pcr": pcr, "mp": mp}

    def futures(self, sym: str) -> list:
        f = _read("futures_quotes", self.date, self._t, sym)
        if f is None or not len(f):
            return []
        v = f.iloc[-1]
        return [{"ltp": float(v.get("near_ltp") or 0), "ch": 0, "chp": 0,
                 "basis": float(v.get("near_basis") or 0), "label": "near", "symbol": sym}]

    def quotes(self, option_syms) -> dict:
        out: dict = {}
        by: dict = {}
        for osym in option_syms:
            try:
                idx, sp, side = osym.split("|")
                by.setdefault(idx, []).append((osym, float(sp), side))
            except Exception:
                continue
        for idx, items in by.items():
            cs = _read("chain_snapshots", self.date, self._t, idx)
            if cs is None or cs.empty:
                continue
            cur = cs[cs["ts"] == cs["ts"].max()]
            for osym, sp, side in items:
                row = cur[(cur["strike"] == sp) & (cur["side"] == side)]
                if len(row):
                    out[osym] = float(row.iloc[-1]["ltp"] or 0)
        return out

    def eod_context(self, sym: str) -> dict:
        try:
            from daily_context_bridge import get_bridge
            return get_bridge().get_panel_data(sym) or {}
        except Exception:
            return {}

    def shock_against(self, index_sym, direction):
        return None   # intraday shock not reconstructed in replay


def run_replay(date: str, step_min: int = 2) -> dict:
    """Replay the engine over `date`, return net-of-cost stats. Uses a TEMP ledger."""
    import intraday_trades
    import engine

    orig_db = intraday_trades._DB
    tmp = Path(tempfile.mkdtemp()) / "replay_trades.db"
    intraday_trades._DB = tmp
    try:
        led = intraday_trades.TradeLedger()
        feed = ReplayFeed(date)
        d0 = datetime.date.fromisoformat(date)
        start = datetime.datetime.combine(d0, datetime.time(9, 16), tzinfo=IST)
        end   = datetime.datetime.combine(d0, datetime.time(15, 25), tzinfo=IST)
        t = start
        steps = opened = 0
        while t <= end:
            feed.set_time(t)
            engine.track_and_resolve(feed, led)              # resolve prior opens first
            opened += engine.eval_and_open(feed, led, INDEX_SYMBOLS)
            steps += 1
            t += datetime.timedelta(minutes=step_min)
        feed.set_time(datetime.datetime.combine(d0, datetime.time(15, 30), tzinfo=IST))
        engine.close_eod(feed, led)
        s = led.stats(date)
        s["_steps"] = steps; s["_opened"] = opened
        return s
    finally:
        intraday_trades._DB = orig_db


if __name__ == "__main__":
    date = sys.argv[1] if len(sys.argv) > 1 else datetime.datetime.now(IST).date().isoformat()
    step = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    print(f"replaying {date} (step {step}m) ...")
    s = run_replay(date, step)
    print("=" * 60)
    print(f"  steps={s.get('_steps')}  opened={s.get('_opened')}  resolved n={s.get('n', 0)}")
    if s.get("n"):
        print(f"  W/L {s['wins']}/{s['losses']}  hit {s.get('hit_rate')}%  "
              f"gross {s.get('avg_r')}R  net {s.get('net_avg_r')}R @{s.get('cost_bps')}bps")
    print("=" * 60)
