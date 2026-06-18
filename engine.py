"""
engine.py — headless intraday trading engine (Phase 3 of the layering refactor).

The open/track/resolve logic used to live as threads INSIDE dashboard.py, wired to
the UI's WebSocket store and Fyers fetch helpers — so it could not run or be
backtested without booting the UI. This module is that logic, lifted out and made
FEED-DRIVEN: it takes a `Feed` and never imports the dashboard.

  Feed (duck-typed):
    now() -> datetime
    spot(sym) -> float                         # 0 when unknown
    chain(sym) -> dict | None                  # {sm, expiry_data, tot_c, tot_p, pcr, mp}
    futures(sym) -> list
    quotes(option_syms) -> {option_sym: ltp}
    eod_context(sym) -> dict                   # last-night Daily-Context (get_panel_data)
    shock_against(index_sym, direction) -> dict | None

  LiveFeed   (built in dashboard.py) wraps the WS store + Fyers helpers — prod.
  ReplayFeed (engine_replay.py) reads the parquet mirrors at a virtual clock — backtest.

The engine functions are time-gate-free: the caller (live poller loop or replay
runner) decides WHEN to call them. Every ledger write is stamped with feed.now(),
so replay is lookahead-free and deterministic. Same code, live or replay = parity.
"""
from __future__ import annotations

from typing import Iterable

from trade_setup import build_recommendation
import intraday_oi_intel


def eval_and_open(feed, led, syms: Iterable[str], tf_key: str = "15min") -> int:
    """Evaluate every index and open tracked paper trades. Returns #opened."""
    opened = 0
    ts = feed.now()
    for sym in syms:
        try:
            spot = feed.spot(sym)
            if not spot:
                continue
            cv = feed.chain(sym)
            if not cv or not cv.get("sm"):
                continue
            rec = build_recommendation(
                sym=sym, tf_key=tf_key, spot=spot, strike_map=cv["sm"],
                expiry_data=cv.get("expiry_data", []), futures=feed.futures(sym),
                pcr=cv.get("pcr", 0), mp=cv.get("mp", 0),
                total_c_oi=cv.get("tot_c", 0), total_p_oi=cv.get("tot_p", 0),
                now=ts,     # virtual clock → phase gates correct in replay
            )
            oi_dyn = intraday_oi_intel.analyze_oi_dynamics(cv["sm"], spot)
            cont = intraday_oi_intel.analyze_continuity(cv["sm"], spot, feed.eod_context(sym))
            if led.maybe_open(sym, rec, oi_dyn=oi_dyn, cont=cont, ts=ts):
                opened += 1
        except Exception:
            pass   # one bad index never stalls the rest (matches old poller)
    return opened


def track_and_resolve(feed, led) -> int:
    """Apply latest option LTPs to open trades + the adaptive shock overlay."""
    ts = feed.now()
    opens = led.open_trades()
    syms = [t["option_sym"] for t in opens if t.get("option_sym")]
    touched = 0
    if syms:
        prices = feed.quotes(syms)
        if prices:
            touched = led.update_open_trades(prices, ts=ts)
    # Adaptive Layer-10: a shock now firing AGAINST an open trade tightens its stop
    # (never a forced exit — the next price tick still resolves it normally).
    for t in led.open_trades():        # re-read: some resolved above
        try:
            sh = feed.shock_against(t["index_sym"], t.get("direction"))
            if not sh:
                continue
            head = sh["signals"][0][1] if sh.get("signals") else "opposing market shock"
            note = f"⚠ REGIME SHIFT — {head}"
            last = t.get("last_ltp") or t.get("entry_ltp") or 0
            risk = t.get("risk") or 0
            new_sl = (last - 0.5 * risk) if (last and risk) else None
            led.flag_regime_shift(t["trade_id"], note, new_sl)
        except Exception:
            pass
    return touched


def close_eod(feed, led) -> int:
    """Mark every still-open trade to its last price at session end."""
    still = [t["option_sym"] for t in led.open_trades() if t.get("option_sym")]
    prices = feed.quotes(still) if still else {}
    return led.close_eod(prices, ts=feed.now())
