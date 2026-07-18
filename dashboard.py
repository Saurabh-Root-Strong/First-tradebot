r"""
dashboard.py  —  NSE Index Live Dashboard + Option Chain
Left pane: 4 live index cards  |  Click any → full option chain (right pane)
Run:   .venv\Scripts\python.exe dashboard.py
Open:  http://127.0.0.1:8050
"""

import base64, json, sys, threading, time, datetime, traceback
from pathlib import Path
from collections import deque

import requests
import pandas as pd
import dash
from dash import dcc, html, Input, Output, State, no_update, ALL
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dashboard_ui import _fmt_contracts, _fmt_cr, _scout_trade_status, _slug  # pure UI leaf helpers
from dashboard_theme import (BG, BG_CARD, BG_SIDE, COLORS, FILLS, MONO, _CSS,  # theme + CSS + tooltips
                             _TIP_AGREE, _TIP_BAND, _TIP_RANGE, _TIP_STR, _TIP_TRIG, _TIP_VERDICT)
import footprint_chart   # full-session OI/Volume/ATM-premium series for the popup chart
import hour_forecast     # next-60-min zone + directional lean (accumulating, unproven)
try:
    import intraday_read as _cockpit   # 3-regime + band + event flags + post-3pm BTST carry
    _COCKPIT_OK = True
except Exception:
    _COCKPIT_OK = False
from fyers_apiv3.FyersWebsocket import data_ws
from signals import run_full_analysis, fetch_ohlcv, _vwap
from trade_setup import build_recommendation, TF_PROFILES
from intraday_store import candle_store, oi_store, build_oi_snapshot, session_phase, session_strategy, record_tick
import intraday_oi_intel
import intraday_trades
import signal_types as sig   # canonical Direction/Bias vocabulary (one language)
import engine                # L4 headless trading engine (feed-driven open/track/resolve)
try:
    import intraday_shock
    _SHOCK_AVAILABLE = True
except Exception:
    _SHOCK_AVAILABLE = False
try:
    import regime_forecast
    _REGIME_AVAILABLE = True
except Exception:
    _REGIME_AVAILABLE = False
try:
    import opening_playbook
    _PLAYBOOK_AVAILABLE = True
except Exception:
    _PLAYBOOK_AVAILABLE = False
try:
    import session_conductor
    _CONDUCTOR_AVAILABLE = True
except Exception:
    _CONDUCTOR_AVAILABLE = False
try:
    import intraday_tf
    _ITF_AVAILABLE = True
except Exception:
    _ITF_AVAILABLE = False
try:
    import nse_oi
    _NSE_OI_AVAILABLE = True
except Exception:
    _NSE_OI_AVAILABLE = False
try:
    import news_events
    _NEWS_AVAILABLE = True
except Exception:
    _NEWS_AVAILABLE = False
try:
    import macro_radar
    _MACRO_AVAILABLE = True
except Exception:
    _MACRO_AVAILABLE = False
try:
    import trend_matrix
    _TREND_AVAILABLE = True
except Exception:
    _TREND_AVAILABLE = False
try:
    import smart_money
    _SM_AVAILABLE = True
except Exception:
    _SM_AVAILABLE = False
try:
    from market_snapshot import MarketSnapshot, get_snapshot  # L2: one shared state/tick
    _SNAPSHOT_OK = True
except Exception:
    _SNAPSHOT_OK = False
try:
    from dcm_prediction import get_dcm_reader as _get_dcm_reader
    _DCM_OK = True
except Exception:
    _DCM_OK = False
try:
    from daily_context_bridge import get_bridge as _get_ctx_bridge
    _CTX_BRIDGE_OK = True
    _get_ctx_bridge()  # pre-warm singleton: start background data load at startup
except Exception:
    _CTX_BRIDGE_OK = False

# ── Constants ──────────────────────────────────────────────────────────────────
from core.constants import IST, INDEX_SYMBOLS, LABELS   # single source of truth
from core.market_calendar import is_trading_day         # ghost-practice default
from tradebot.adapters.broker import token as _broker_token   # single broker-token source
from tradebot.adapters.broker import rest as _broker_rest     # single Fyers REST-fetch source
APP_ID     = _broker_token.APP_ID
TOKEN_FILE = _broker_token.TOKEN_FILE   # PROJECT_ROOT-anchored (was CWD-relative)
SEP        = "─" * 58

# ── Globals ────────────────────────────────────────────────────────────────────
_access_token = ""  # set in __main__; callbacks also read TOKEN_FILE directly as fallback
_lock   = threading.Lock()
_latest:  dict[str, dict]  = {}
_history: dict[str, deque] = {s: deque(maxlen=1800) for s in INDEX_SYMBOLS}
_ws     = None
_seen:  set[str] = set()

# Option-chain fetch + its quota-aware keyed cache now live in the broker adapter
# (tradebot.adapters.broker.rest) — fetch_option_chain below delegates there.


# ── Token validation ───────────────────────────────────────────────────────────
def _run_auth() -> None:
    """Launch the Fyers auth flow (browser login, or headless TOTP if FYERS_HEADLESS=1)
    and block until it finishes writing access_token.txt."""
    import os
    script = "fyers_auth_headless.py" if os.environ.get("FYERS_HEADLESS") == "1" else "fyers_auth.py"
    print(f"  AUTO   launching {script} — "
          + ("headless TOTP login…" if script.endswith("headless.py") else "log in via the browser…"))
    try:
        import subprocess
        subprocess.run([sys.executable, str(Path(__file__).parent / script)],
                       cwd=str(Path(__file__).parent))
    except Exception as exc:
        print(f"  AUTO   {script} failed: {exc}")


def _validate_token() -> str:
    # On a missing/expired token, auto-run the auth flow once and re-check, instead
    # of erroring out — so launching dashboard.py directly self-heals like supervise.py.
    # Validity + human summary come from the broker-token adapter (is_usable/describe),
    # the single source of truth — no inline JWT decode here.
    for attempt in (1, 2):
        raw = TOKEN_FILE.read_text(encoding="utf-8").strip() if TOKEN_FILE.exists() else ""
        if raw and _broker_token.token_remaining(raw) is None:
            # A token exists but our decoder can't parse it — fail OPEN: a real Fyers
            # token we simply can't read shouldn't block trading (REST 401s if truly bad).
            print("  Token  WARNING: could not parse — proceeding with stored token.")
            return raw
        if _broker_token.is_usable(raw):
            print("  Token  OK  —  " + _broker_token.describe(raw))
            return raw
        print("\n  Token  NOT USABLE  —  " + _broker_token.describe(raw))
        if attempt == 1:
            _run_auth()      # try once, then re-loop to re-validate
            continue
        print("  ERROR  still no valid token after auth.")
        print("  FIX    run  .venv\\Scripts\\python.exe fyers_auth.py  manually, then relaunch.\n")
        sys.exit(1)
    return TOKEN_FILE.read_text(encoding="utf-8").strip()


# ── Option chain API + cache ───────────────────────────────────────────────────
_expiry_to_epoch  = _broker_rest.expiry_to_epoch
fetch_option_chain = _broker_rest.fetch_option_chain


def _get_auth() -> str:
    """Always read token from file — avoids module-global timing issues with Dash threads."""
    return _broker_token.auth_header()


# ── Futures helpers ────────────────────────────────────────────────────────────
# Symbol-building + the cached REST fetch live in the broker adapter
# (tradebot.adapters.broker.rest). dashboard keeps only the storage side-effect.
_futures_symbols = _broker_rest.futures_symbols


def _idb_write_futures(index_sym: str, futures: list) -> None:
    """Non-blocking DB write for the futures strip. Spot from live tick cache."""
    try:
        from intraday_db import idb
        spot = float((_latest.get(index_sym) or {}).get("ltp") or 0.0)
        idb.write_futures(index_sym, futures, spot)
    except Exception:
        pass


def fetch_futures(index_sym: str) -> list[dict]:
    """Fetch near/next/far futures quotes (broker adapter) + persist the strip.
    DB write fires only on a fresh network fetch (not cache hits) via on_fresh."""
    return _broker_rest.fetch_futures(
        index_sym, on_fresh=lambda r: _idb_write_futures(index_sym, r))


def render_futures(futures: list[dict], spot: float) -> html.Div:
    """Render a compact futures strip card."""
    if not futures:
        return html.Div()

    MONO_S = {**MONO, "fontSize": "0.72rem"}

    rows = []
    for f in futures:
        ltp   = f["ltp"]
        ch    = f["ch"]
        chp   = f["chp"]
        basis = ltp - spot if (ltp and spot) else 0
        vol   = f["vol"]
        up    = ch >= 0
        clr   = "#22c55e" if up else "#ef4444"
        b_clr = "#22c55e" if basis >= 0 else "#ef4444"
        sign  = "+" if up else ""
        b_sign= "+" if basis >= 0 else ""

        rows.append(dbc.Row([
            dbc.Col(html.Span(f["label"],  style={"color":"#334155","fontSize":"0.6rem","letterSpacing":"0.1em"}), width=1),
            dbc.Col(html.Span(f["month"],  style={"color":"#475569","fontSize":"0.65rem"}), width=1),
            dbc.Col(html.Span(f"{ltp:,.2f}" if ltp else "—",
                              style={**MONO_S,"color":clr,"fontWeight":"700"}), width=2),
            dbc.Col(html.Span(f"{sign}{ch:,.2f}  ({sign}{chp:.2f}%)" if ltp else "—",
                              style={**MONO_S,"color":clr}), width=2),
            dbc.Col([
                html.Span("Basis  ", style={"color":"#334155","fontSize":"0.6rem"}),
                html.Span(f"{b_sign}{basis:,.2f}" if ltp else "—",
                          style={**MONO_S,"color":b_clr,"fontWeight":"600"}),
            ], width=2),
            dbc.Col([
                html.Span("H ", style={"color":"#334155","fontSize":"0.6rem"}),
                html.Span(f"{f['high']:,.2f}" if f["high"] else "—",
                          style={**MONO_S,"color":"#4ade80"}),
                html.Span("  L ", style={"color":"#334155","fontSize":"0.6rem"}),
                html.Span(f"{f['low']:,.2f}" if f["low"] else "—",
                          style={**MONO_S,"color":"#f87171"}),
            ], width=2),
            dbc.Col([
                html.Span("Vol  ", style={"color":"#334155","fontSize":"0.6rem"}),
                html.Span(_fmt_oi(vol), style={**MONO_S,"color":"#64748b"}),
            ], width=2),
        ], className="mb-1 align-items-center"))

    # Term structure insight
    if len(futures) >= 2 and futures[0]["ltp"] and futures[1]["ltp"]:
        roll = futures[1]["ltp"] - futures[0]["ltp"]
        structure = "CONTANGO" if roll > 0 else "BACKWARDATION"
        s_clr = "#4ade80" if roll > 0 else "#f87171"
        insight = html.Div([
            html.Span("Term Structure  ", style={"color":"#334155","fontSize":"0.6rem"}),
            html.Span(structure, style={"color":s_clr,"fontWeight":"700","fontSize":"0.68rem",**MONO}),
            html.Span(f"   Roll Spread  ", style={"color":"#334155","fontSize":"0.6rem"}),
            html.Span(f"{'+' if roll>0 else ''}{roll:,.2f} pts",
                      style={"color":s_clr,**MONO,"fontSize":"0.68rem"}),
        ], style={"marginTop":"6px"})
    else:
        insight = html.Div()

    return dbc.Card(dbc.CardBody([
        html.Div("FUTURES  —  NEAR / NEXT / FAR MONTH", style={
            "fontSize": "0.58rem", "letterSpacing": "0.15em",
            "color": "#1e3a5f", "marginBottom": "8px",
        }),
        *rows,
        insight,
    ]), style={
        "background": "#080f1c",
        "border": "1px solid #1a2535",
        "borderRadius": "8px",
        "marginBottom": "10px",
    })


# OI business-logic (max-pain, chain-prediction engine, OI formatters) lives in
# oi_analytics.py — pure, testable, shared with headless callers. (Refactor #6.)
from oi_analytics import (   # noqa: E402
    compute_max_pain, compute_prediction, _fmt_oi, _fmt_futoi,
)


def render_prediction(pred: dict, mp: float) -> html.Div:
    if not pred:
        return html.Div()

    trend   = pred["trend"]
    color   = pred["color"]
    score   = pred["score"]
    signals = pred["signals"]
    res     = pred["resistance"]
    sup     = pred["support"]

    bull_cnt = sum(1 for _, _, b in signals if b == "bull")
    bear_cnt = sum(1 for _, _, b in signals if b == "bear")
    neut_cnt = sum(1 for _, _, b in signals if b == "neut")
    total    = len(signals) or 1

    # Signal meter bar (shows bull/bear/neutral proportions)
    bull_pct = bull_cnt / total * 100
    bear_pct = bear_cnt / total * 100
    neut_pct = neut_cnt / total * 100

    meter_bar = html.Div([
        html.Div(style={"width": f"{bull_pct:.0f}%", "background": "#22c55e", "height": "4px", "display": "inline-block"}),
        html.Div(style={"width": f"{neut_pct:.0f}%", "background": "#334155", "height": "4px", "display": "inline-block"}),
        html.Div(style={"width": f"{bear_pct:.0f}%", "background": "#ef4444", "height": "4px", "display": "inline-block"}),
    ], style={"borderRadius": "2px", "overflow": "hidden", "marginBottom": "10px"})

    signal_rows = []
    for icon, text, bias in signals:
        clr = "#4ade80" if bias == "bull" else "#f87171" if bias == "bear" else "#334155"
        signal_rows.append(html.Div([
            html.Span(icon + "  "),
            html.Span(text, style={"color": clr}),
        ], style={"fontSize": "0.68rem", "marginBottom": "4px", **MONO}))

    return dbc.Card(dbc.CardBody([
        dbc.Row([
            dbc.Col([
                html.Div("PREDICTION", style={
                    "fontSize": "0.58rem", "letterSpacing": "0.2em",
                    "color": "#334155", "marginBottom": "4px",
                }),
                html.Div(trend, style={
                    "fontSize": "1.1rem", "fontWeight": "900",
                    "color": color, "letterSpacing": "0.06em",
                }),
                html.Div(f"Score  {score:+d} / {len(signals)}", style={
                    **MONO, "fontSize": "0.65rem", "color": "#475569", "marginTop": "2px",
                }),
            ], md=3),
            dbc.Col([
                html.Div([
                    html.Span("Support   ", style={"color": "#334155", "fontSize": "0.62rem"}),
                    html.Span(f"{sup:,.0f}", style={"color": "#4ade80", "fontWeight": "700", "fontSize": "0.8rem", **MONO}),
                    html.Span("    Resistance   ", style={"color": "#334155", "fontSize": "0.62rem"}),
                    html.Span(f"{res:,.0f}", style={"color": "#f87171", "fontWeight": "700", "fontSize": "0.8rem", **MONO}),
                    html.Span("    Max Pain   ", style={"color": "#334155", "fontSize": "0.62rem"}),
                    html.Span(f"{mp:,.0f}", style={"color": "#fbbf24", "fontWeight": "700", "fontSize": "0.8rem", **MONO}),
                ], style={"marginBottom": "8px"}),
                meter_bar,
            ], md=9),
        ], className="mb-2"),
        html.Div(signal_rows),
    ]), style={
        "background": "#080f1c",
        "border":     "1px solid #1a2535",
        "borderLeft": f"4px solid {color}",
        "borderRadius": "8px",
        "marginBottom": "10px",
    })


# ── WebSocket ──────────────────────────────────────────────────────────────────
def _onmessage(msg: dict) -> None:
    if "code" in msg:
        print(f"  [WS]   {msg.get('message', msg)}")
        return
    sym = msg.get("symbol")
    if sym not in INDEX_SYMBOLS:
        return

    # ── Extract all fields from Fyers SymbolUpdate ────────────────────────────
    ft  = msg.get("exch_feed_time", 0)
    ts  = datetime.datetime.fromtimestamp(ft, tz=IST) if ft else datetime.datetime.now(tz=IST)
    ltp = float(msg.get("ltp",        0) or 0)
    vol = float(msg.get("volume",     0) or 0)
    # Day OHLC, change — available in every SymbolUpdate packet
    day_open = float(msg.get("open_price",     0) or 0)
    day_high = float(msg.get("high_price",     0) or 0)
    day_low  = float(msg.get("low_price",      0) or 0)
    ch       = float(msg.get("ch",             0) or 0)
    chp      = float(msg.get("chp",            0) or 0)

    with _lock:
        _latest[sym] = msg
        _history[sym].append((ts, ltp))

    if ltp:
        # Single call: feeds candle builders AND persists raw tick to DuckDB
        record_tick(sym, ts, ltp, vol, day_open, day_high, day_low, ch, chp)
        global _last_tick_wall
        _last_tick_wall = time.time()

    if sym not in _seen:
        _seen.add(sym)
        print(f"  [WS]   First tick  {LABELS[sym]:<14}  LTP {ltp:>10,.2f}")
        if len(_seen) == len(INDEX_SYMBOLS):
            print("  [WS]   All 4 indices live ✓  — ticks persisting to DuckDB")

def _onopen():
    print("  [WS]   Connected — subscribing...")
    _ws.subscribe(symbols=INDEX_SYMBOLS, data_type="SymbolUpdate")
    _ws.keep_running()

def _onerror(m): print(f"  [WS]   ERROR: {m}")
def _onclose(m): print(f"  [WS]   Closed: {m}")

def _start_ws(token: str):
    global _ws
    _ws = data_ws.FyersDataSocket(
        access_token=token, log_path="", litemode=False,
        write_to_file=False, reconnect=True,
        on_connect=_onopen, on_close=_onclose,
        on_error=_onerror, on_message=_onmessage,
    )
    _ws.connect()


# ── WebSocket health heartbeat (read by supervise.py) ────────────────────────
# Baseline = process start, NOT 0. With 0 the heartbeat file reads "0" until the
# first tick, so a dashboard started PRE-OPEN aged past the supervisor's 120s
# launch grace and got restart-killed at 09:15:00 sharp (heartbeat_age ≈ 1e9 >
# WS_STALL_SEC) — a spurious restart in the open's first, most valuable seconds.
# Seeding wall-clock now means "no tick yet" only goes stale WS_STALL_SEC after
# start, which is exactly the stall semantics the supervisor wants.
_last_tick_wall = time.time()
HEARTBEAT_FILE = Path(__file__).parent / "data" / "ws_heartbeat.txt"


def _heartbeat_writer():
    """Write the wall-clock of the last received tick to a file every 10s, so the
    supervisor can detect a stalled WebSocket (no ticks during market hours) and
    restart the process before capture is lost."""
    HEARTBEAT_FILE.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            HEARTBEAT_FILE.write_text(f"{_last_tick_wall:.0f}")
        except Exception:
            pass
        time.sleep(10)


def _oi_background_poller():
    """
    Background thread: snapshot option chain for all 4 indices every 30 seconds.

    Runs independently of user interaction so OI history accumulates from 9:15
    whether or not the OC panel is open.  Uses the nearest expiry for each index.
    Skips outside market hours to avoid wasting API quota.

    30s (180→60→30): measured that Fyers option OI genuinely moves within ~25s, so
    these aren't duplicates. 30s ≈ 1/10 of the shortest (5-min) TF → ±10% anchor
    precision, matching the price side. NOT finer: <30s oversamples the 5-min frame
    (the noisiest, least-trusted signal) — the real footprint is the 15/60-min and
    day-level OI builds, indifferent to sub-30s. Export is decoupled (10s flush),
    API is trivial (8 calls/min). NSE futures OI stays 60s (publication-bound).
    """
    OPEN  = datetime.time(9, 14)
    CLOSE = datetime.time(15, 31)
    # Adaptive cadence: the Fyers option-chain REST endpoint has a daily call
    # budget (empirically ~exhausted by ~11:05 at a flat 30s × 4 indices, after
    # which capture silently died while WS ticks ran to close). The morning is
    # where the gap/overnight-unwinding read lives, so keep it fast there and
    # stretch the budget across the afternoon: 30s until 11:30, 90s after.
    POLL_FAST, POLL_SLOW = 30, 90
    SLOW_AFTER = datetime.time(11, 30)
    # Strikes: fetch/persist a WIDE band so last night's OI WALLS (often well OTM,
    # the put floor especially) are actually in the captured chain — at ±15 the
    # floor wall fell outside the map on ~90% of samples, blinding the continuity/
    # reconciliation read. One REST call regardless of strikecount, so this is free
    # on the quota; only the payload grows.
    CHAIN_STRIKES  = 25     # ± strikes Fyers returns around ATM
    CHAIN_PERSIST  = 60     # max legs persisted per index per snapshot
    _oc_ok = {s: None for s in INDEX_SYMBOLS}   # last fetch state, for transition logging

    while True:
        try:
            now_t = datetime.datetime.now(tz=IST).time()
            if OPEN <= now_t <= CLOSE:
                with _lock:
                    spots = {s: (_latest.get(s) or {}).get("ltp", 0) for s in INDEX_SYMBOLS}
                for sym in INDEX_SYMBOLS:
                    spot = spots.get(sym, 0)
                    if not spot:
                        continue
                    try:
                        data = fetch_option_chain(sym, n_strikes=CHAIN_STRIKES)
                        ok = data.get("s") == "ok"
                        # Log only on state TRANSITION so we measure exactly when /
                        # why chain capture dies (quota? auth? expiry roll) without
                        # spamming. This is the diagnostic for the ~11am cutoff.
                        if ok != _oc_ok.get(sym):
                            _oc_ok[sym] = ok
                            if not ok:
                                print(f"  [chain] {LABELS.get(sym, sym)} fetch FAILED @ "
                                      f"{datetime.datetime.now(tz=IST):%H:%M:%S} — "
                                      f"{str(data.get('message'))[:120]}", flush=True)
                            else:
                                print(f"  [chain] {LABELS.get(sym, sym)} recovered @ "
                                      f"{datetime.datetime.now(tz=IST):%H:%M:%S}", flush=True)
                        if not ok:
                            continue
                        d   = data.get("data", {})
                        raw = d.get("optionsChain", [])
                        if not raw:
                            continue
                        strike_map: dict = {}
                        for entry in raw:
                            sp = entry.get("strike_price", -1)
                            if sp <= 0:
                                continue
                            if sp not in strike_map:
                                strike_map[sp] = {}
                            ot = entry.get("option_type", "")
                            if ot in ("CE", "PE"):
                                strike_map[sp][ot] = entry
                        tot_c = d.get("callOi", 0)
                        tot_p = d.get("putOi",  0)
                        pcr   = tot_p / tot_c if tot_c else 0
                        # reuse compute_max_pain already defined in this module
                        mp_ch = [
                            {"strike_price": sp,
                             "call_options": {"oi": strike_map[sp].get("CE", {}).get("oi", 0)},
                             "put_options":  {"oi": strike_map[sp].get("PE", {}).get("oi", 0)}}
                            for sp in sorted(strike_map)
                        ]
                        mp   = compute_max_pain(mp_ch) if mp_ch else 0
                        snap = build_oi_snapshot(sym, spot, strike_map, tot_c, tot_p, pcr, mp)
                        if snap:
                            oi_store.add(snap)
                        # Persist per-strike legs near ATM so strike-level OI
                        # dynamics (writing vs unwinding) survive the session
                        # and reach the parquet mirrors for playbook/replay.
                        try:
                            from intraday_db import idb
                            near = sorted(strike_map, key=lambda sp: abs(sp - spot))[:CHAIN_PERSIST]
                            legs = []
                            for sp in near:
                                for side in ("CE", "PE"):
                                    e = strike_map[sp].get(side)
                                    if e:
                                        g = e.get("greeks") or {}
                                        legs.append((sp, side,
                                                     e.get("ltp"), e.get("ltpch"),
                                                     e.get("oi"), e.get("oich"),
                                                     e.get("volume"),
                                                     g.get("delta"), g.get("iv")))
                            idb.write_chain(sym, datetime.datetime.now(tz=IST), legs)
                        except Exception as exc:
                            print(f"  [chain] {LABELS.get(sym, sym)} persist error: "
                                  f"{str(exc)[:120]}", flush=True)
                    except Exception as exc:
                        # A RAISED exception (DNS/SSL/auth throw — not an s!="ok"
                        # payload) must hit the same transition log, or a persistent
                        # throw is a silent chain death with zero [chain] lines.
                        if _oc_ok.get(sym) is not False:
                            _oc_ok[sym] = False
                            print(f"  [chain] {LABELS.get(sym, sym)} fetch EXCEPTION @ "
                                  f"{datetime.datetime.now(tz=IST):%H:%M:%S} — "
                                  f"{type(exc).__name__}: {str(exc)[:120]}", flush=True)
        except Exception as exc:
            print(f"  [chain] poller cycle error: {type(exc).__name__}: {str(exc)[:120]}",
                  flush=True)
        time.sleep(POLL_FAST if datetime.datetime.now(tz=IST).time() < SLOW_AFTER else POLL_SLOW)


_fetch_quotes = _broker_rest.fetch_quotes   # /data/quotes lp map — broker adapter


class LiveFeed:
    """Feed adapter for the headless engine — wraps the live WS store + Fyers
    helpers. Parsing that used to live in the poller now lives here, so the engine
    consumes one normalized shape whether the data is live or replayed."""

    def now(self):
        return datetime.datetime.now(tz=IST)

    def spot(self, sym):
        with _lock:
            return (_latest.get(sym) or {}).get("ltp", 0)

    def chain(self, sym):
        oc = fetch_option_chain(sym)
        if oc.get("s") != "ok":
            return None
        d   = oc.get("data", {})
        raw = d.get("optionsChain", [])
        if not raw:
            return None
        sm: dict = {}
        for e in raw:
            sp = e.get("strike_price", -1)
            if sp > 0 and e.get("option_type") in ("CE", "PE"):
                sm.setdefault(sp, {})[e["option_type"]] = e
        tot_c = d.get("callOi", 0); tot_p = d.get("putOi", 0)
        mp_ch = [{"strike_price": sp,
                  "call_options": {"oi": sm[sp].get("CE", {}).get("oi", 0)},
                  "put_options":  {"oi": sm[sp].get("PE", {}).get("oi", 0)}}
                 for sp in sorted(sm)]
        return {"sm": sm, "expiry_data": d.get("expiryData", []),
                "tot_c": tot_c, "tot_p": tot_p,
                "pcr": (tot_p / tot_c if tot_c else 0),
                "mp": (compute_max_pain(mp_ch) if mp_ch else 0)}

    def futures(self, sym):
        return fetch_futures(sym)

    def quotes(self, option_syms):
        return _fetch_quotes(option_syms)

    def eod_context(self, sym):
        try:
            from daily_context_bridge import get_bridge
            return get_bridge().get_panel_data(sym) or {}
        except Exception:
            return {}

    def shock_against(self, index_sym, direction):
        if not _SHOCK_AVAILABLE:
            return None
        try:
            return intraday_shock.shock_against(index_sym, direction)
        except Exception:
            return None


def _trade_tracker_poller():
    """Drive engine.track_and_resolve every ~20s during market hours; mark-to-close
    once after 15:31. The lifecycle logic lives in engine.py (feed-driven, testable);
    this loop is just the live clock + gate."""
    led  = intraday_trades.get_ledger()
    feed = LiveFeed()
    eod_done_for: str = ""
    while True:
        try:
            now = datetime.datetime.now(tz=IST)
            if datetime.time(9, 14) <= now.time() <= datetime.time(15, 45):
                engine.track_and_resolve(feed, led)
            if now.time() >= datetime.time(15, 31) and eod_done_for != now.date().isoformat():
                engine.close_eod(feed, led)
                eod_done_for = now.date().isoformat()
        except Exception:
            pass
        time.sleep(20)


def _auto_signal_poller():
    """Drive engine.eval_and_open across all 4 indices every 60s during market hours
    (track-record coverage for the whole book, not just the viewed panel). Logic in
    engine.py; this loop is just the live clock + gate."""
    led  = intraday_trades.get_ledger()
    feed = LiveFeed()
    while True:
        try:
            now = datetime.datetime.now(tz=IST)
            if datetime.time(9, 16) <= now.time() <= datetime.time(15, 25):
                engine.eval_and_open(feed, led, INDEX_SYMBOLS, tf_key="15min")
        except Exception:
            pass
        time.sleep(60)


# ── Live news / event-impact panel ──────────────────────────────────────────────
def _render_news_panel(data: dict, tab: str = "ALL", tape: bool = False) -> html.Div:
    """Event-impact alerts: time · impact score (−10..+10) · ticker · headline.
    Driven by news_events.analyze_news(); colour = canonical bias green/red/grey.
    `tab` filters by trader-facing bucket; `tape`=True = full-day chronological
    view (taller scroll, shows the whole session's deduped event tape)."""
    if not data:
        return html.Div("news layer offline", style={"color": "#475569", "fontSize": "0.55rem"})
    alerts = data.get("alerts", [])
    if tab and tab != "ALL":
        alerts = [e for e in alerts if e.get("bucket") == tab]
    mb     = data.get("macro_bias", "NEUTRAL")
    mb_clr = sig.color(mb)
    bb     = data.get("by_bucket", {})
    cur    = data.get("date", "")
    header = html.Div([
        html.Span("📰 NEWS / EVENT IMPACT", style={
            "color": "#fbbf24", "fontWeight": "700", "fontSize": "0.62rem",
            "letterSpacing": "0.1em"}),
        html.Span(f"  macro ", style={"color": "#475569", "fontSize": "0.5rem"}),
        html.Span(mb, style={"color": mb_clr, "fontWeight": "700", "fontSize": "0.56rem"}),
        # live per-bucket tally (always full-day counts, regardless of active tab)
        html.Span(f"  ·  🟢{bb.get('BULLISH',0)} 🔴{bb.get('BEARISH',0)}",
                  style={"color": "#94a3b8", "fontSize": "0.5rem", "fontWeight": "700"}),
        html.Span(f"  ·  {cur}  ·  {data.get('as_of','')}",
                  style={"color": "#475569", "fontSize": "0.5rem"}),
    ], style={"marginBottom": "6px", "display": "flex", "alignItems": "center"})
    rows = []
    for e in alerts:                                  # scrollable container below
        sc  = e["score"]
        clr = sig.color(e["bias"])
        chip = {"MACRO": "#a78bfa", "SECTOR": "#38bdf8", "STOCK": "#94a3b8"}.get(e["scope"], "#94a3b8")
        try:
            tm = pd.Timestamp(e.get("ts")).strftime("%H:%M")
        except Exception:
            tm = "--:--"
        carry = bool(e.get("carry"))
        if carry:
            tm = f"◂{tm}"          # filed YESTERDAY post-close — reaction window TODAY
        # Severe-event discipline badge, GAP-CONDITIONED when the capture-time move
        # (chg = % vs prev close at news) is known. Entry-gap buckets are MEASURED
        # (backtest_news_short.py): barely-reacted carries the drift both sides;
        # ≥3% already moved = consumed (pos: −0.84% +1d t=−2.1; neg: bounces).
        # Tooltip language is deliberately SIMPLE — plain sentences, define F&O,
        # say what to do. The measured numbers back the advice but don't lead it.
        _FNO_DEF = ("(F&O stock = one of ~200 big companies where NSE allows "
                    "futures & options trading — so it CAN be shorted, and it is "
                    "large and liquid.) ")
        sev = e.get("severe")
        chg = float(e.get("chg") or 0.0)
        has_chg = abs(chg) > 1e-9
        sev_blk = None
        if sev in ("AVOID", "FUT"):
            if sev == "AVOID":
                label, clr_b = "⛔ AVOID/EXIT", "#f87171"
                tt = ("VERY BAD news (fraud / insolvency type) on a SMALL stock that "
                      "is NOT in F&O — no futures, no options, so there is NO way to "
                      "short it. What you can do: do NOT buy this dip, and if you "
                      "already hold it, think about exiting. In our data such stocks "
                      "fell about −0.9% the next day (2 of 3 fell).")
            else:
                label, clr_b = "⚠ F&O — careful", "#fbbf24"
                tt = ("VERY BAD news on an F&O stock. " + _FNO_DEF +
                      "You COULD short it via futures — BUT our data says don't rush: "
                      "big stocks usually get BOUGHT after bad news (dip buyers), and "
                      "shorting them lost money. Careful both ways.")
            if has_chg and chg >= -1:
                label = f"⛔ not fallen yet {chg:+.1f}%"
                tt += (" NOTE: the price has barely moved since this news — the fall "
                       "usually comes the NEXT day. This warning is at its most "
                       "useful right now.")
            elif has_chg and chg <= -3:
                label = f"⛔ already down {chg:+.1f}%"
                tt += (" NOTE: the price already fell 3%+ — the damage is done. "
                       "Panic-selling or shorting NOW is usually too late; such "
                       "stocks often bounce the next day.")
            elif has_chg:
                label = f"⛔ {chg:+.1f}% moved"
        elif sev in ("POS", "POS_FUT"):
            if sev == "POS":
                label, clr_b = "🚀 don't chase", "#4ade80"
                tt = ("GOOD news on a smaller stock (not in F&O — you can only buy "
                      "shares). The price jump usually happens IMMEDIATELY, at or "
                      "near the open. Buying AFTER the jump loses on average: e.g. "
                      "'large order win' stocks jump ~+0.5% on day 1, then give it "
                      "back (−1.6% by day 5). Only interesting if the price has NOT "
                      "moved up yet.")
            else:
                label, clr_b = "⚡ F&O — already in price", "#34d399"
                tt = ("GOOD news on an F&O stock. " + _FNO_DEF +
                      "For big stocks like this, good news goes into the price "
                      "within MINUTES. By the time you read it, it is already in the "
                      "price — buying now actually LOST money in our data (about "
                      "−0.4% the next day). Read it as information, not a buy signal.")
            if has_chg and chg <= 1 and sev == "POS":
                label = f"🟢 not moved yet {chg:+.1f}%"
                tt += (" NOTE: the price has moved less than 1% since this news — "
                       "the market may not have reacted yet. This is the ONE "
                       "situation where a small 1–3 day buy showed profit in our "
                       "data (+0.5% to +0.7%). Small edge, not a guarantee — keep "
                       "position size small.")
            elif has_chg and chg >= 3:
                label = f"🔴 already up {chg:+.1f}%"
                tt += (" NOTE: the price is ALREADY up 3%+ — the move is over. "
                       "Buying now usually loses (−0.8% next day on average; 3 of 4 "
                       "such stocks fell). No purpose purchasing now.")
            elif has_chg:
                label = f"🚀 {chg:+.1f}% moved"
                tt += (" NOTE: already up 1–3% — most of the move is done; buying "
                       "here showed no profit in our data.")
        if sev:
            sev_blk = html.Span(label, title=tt, style={
                "color": clr_b, "fontSize": "0.48rem", "fontWeight": "800",
                "width": "118px", "letterSpacing": "0.03em", "cursor": "help"})
        rows.append(html.Div([
            html.Span(tm, title=(
                "filed near/after the close of the LAST session (or on a weekend/"
                "holiday) — late news moves the NEXT session (desk rule), so its "
                "reaction window is the next trading day"
                if carry else ""),
                style={"color": "#fbbf24" if carry else "#475569",
                       "fontSize": "0.5rem", "width": "40px" if carry else "34px",
                       "marginRight": "4px",
                       "cursor": "help" if carry else "default", **MONO}),
            html.Span(f"{sc:+d}", style={
                "color": clr, "fontWeight": "800", "fontSize": "0.7rem",
                "width": "32px", "textAlign": "right", "marginRight": "8px",
                **MONO}),
            html.Span(e["scope"][:3], title=(
                {"STOCK": "STOCK — affects ONE stock (the ticker shown)",
                 "SECTOR": "SECTOR — affects a whole sector basket",
                 "MACRO": "MACRO — affects the WHOLE market (RBI/Fed/CPI-type news)"}
                .get(e["scope"], "")), style={
                "color": chip, "fontSize": "0.48rem", "fontWeight": "700",
                "width": "26px", "letterSpacing": "0.05em", "cursor": "help"}),
            html.Span((e["ticker"] or "—")[:11], style={
                "color": "#cbd5e1", "fontSize": "0.55rem", "fontWeight": "700",
                "width": "78px", **MONO}),
            # LTP at capture (the price you'd see when the news hit) — 0/absent on
            # rows captured before this field existed or on quote failure
            html.Span(f"₹{e['px']:,.1f}" if e.get("px") else "", title=(
                "stock price at the moment the news was captured"),
                style={"color": "#94a3b8", "fontSize": "0.5rem", "width": "58px",
                       "textAlign": "right", "marginRight": "6px", "cursor": "help",
                       **MONO}),
            html.Span(e["event_type"], style={
                "color": clr, "fontSize": "0.52rem", "width": "120px"}),
            # repeat count — same (ticker, event) re-filed N times; first shown, rest
            # collapsed (a new row appears only when the STORY changes)
            html.Span(f"×{e['n_rep']}" if e.get("n_rep", 1) > 1 else "", title=(
                f"re-filed {e.get('n_rep')} times this day — collapsed to first sighting"),
                style={"color": "#64748b", "fontSize": "0.48rem", "width": "26px",
                       "cursor": "help", **MONO}),
        ] + ([sev_blk] if sev_blk is not None else []) + [
            html.Span(e["headline"][:90], style={
                "color": "#64748b", "fontSize": "0.5rem", "flex": "1 1 auto",
                "overflow": "hidden", "textOverflow": "ellipsis", "whiteSpace": "nowrap"}),
        ], style={"display": "flex", "alignItems": "center", "gap": "2px",
                  "padding": "3px 0", "borderBottom": "1px solid #111d2e"}))
    if not rows:
        msg = (f"no {tab.lower()} events this day" if tab and tab != "ALL"
               else "no market-moving events this day (feeds quiet or IP-blocked)")
        rows = [html.Div(msg, style={"color": "#475569", "fontSize": "0.52rem", "padding": "4px 0"})]
    body = html.Div(rows, style={"maxHeight": "340px" if tape else "168px",
                                 "overflowY": "auto", "paddingRight": "4px"})
    return html.Div([header, body], style={
        "padding": "10px 12px", "borderRadius": "8px", "background": BG_CARD,
        "border": "1px solid #1e3a5f55", "borderLeft": "3px solid #fbbf24",
        "marginBottom": "12px"})


# ── Macro radar (global risk board -> India) ───────────────────────────────────
_MACRO_STATE: dict = {"rows": [], "tilt": 0.0, "lean": "—", "as_of": ""}


def _macro_radar_poller(every: int = 180):
    """Background: refresh the global macro board into _MACRO_STATE every `every`s.
    Network-bound (yfinance) — NEVER call macro_radar.compute() from a request path."""
    import time as _t
    while True:
        try:
            rows, tilt, lean = macro_radar.compute()
            _MACRO_STATE.update({"rows": rows, "tilt": tilt, "lean": lean,
                                 "as_of": pd.Timestamp.now(IST).strftime("%H:%M:%S")})
        except Exception:
            pass
        _t.sleep(max(60, every))


def _render_macro_radar(state: dict) -> html.Div:
    """Live global macro board -> India: per-factor %chg/z/impact + net risk tilt.
    CONTEXT/RISK only (public macro priced fast; gap pre-priced by GIFT Nifty)."""
    rows = state.get("rows") or []
    if not rows:
        return html.Div("macro radar warming up…",
                        style={"color": "#475569", "fontSize": "0.55rem"})
    tilt = state.get("tilt", 0.0)
    lean_clr = "#22c55e" if tilt > 1 else "#ef4444" if tilt < -1 else "#94a3b8"
    header = html.Div([
        html.Span("🌐 MACRO RADAR → INDIA", style={
            "color": "#38bdf8", "fontWeight": "700", "fontSize": "0.62rem",
            "letterSpacing": "0.1em"}),
        html.Span(f"  tilt {tilt:+.1f}  ", style={"color": "#475569", "fontSize": "0.5rem"}),
        html.Span(state.get("lean", "—"),
                  style={"color": lean_clr, "fontWeight": "700", "fontSize": "0.56rem"}),
        html.Span(f"  ·  {state.get('as_of','')}",
                  style={"color": "#475569", "fontSize": "0.5rem"}),
    ], style={"marginBottom": "6px", "display": "flex", "alignItems": "center"})
    body_rows = []
    for r in rows:
        if r.get("level") is None:
            continue
        up = r["impact"] == "UP"; dn = r["impact"] == "DOWN"
        iclr = "#22c55e" if up else "#ef4444" if dn else "#94a3b8"
        chg_clr = "#22c55e" if (r["chg"] or 0) >= 0 else "#ef4444"
        body_rows.append(html.Div([
            html.Span(r["factor"][:13], style={"color": "#cbd5e1", "fontSize": "0.55rem",
                      "width": "92px", "fontWeight": "700"}),
            html.Span(f"{r['chg']:+.2f}%", style={"color": chg_clr, "fontSize": "0.55rem",
                      "width": "56px", "textAlign": "right", **MONO}),
            html.Span(f"z{r['z']:+.1f}", style={"color": "#64748b", "fontSize": "0.5rem",
                      "width": "44px", "textAlign": "right", **MONO}),
            html.Span(("▲" if up else "▼" if dn else "•"), style={
                      "color": iclr, "fontSize": "0.6rem", "width": "20px", "textAlign": "center"}),
            html.Span("SPIKE" if r.get("spike") else "", style={"color": "#fbbf24",
                      "fontSize": "0.46rem", "fontWeight": "700"}),
        ], style={"display": "flex", "alignItems": "center", "gap": "4px",
                  "padding": "2px 0", "borderBottom": "1px solid #111d2e"}))
    body = html.Div(body_rows, style={"maxHeight": "168px", "overflowY": "auto",
                                      "paddingRight": "4px"})
    note = html.Div("context/risk — public macro is priced fast; not a front-run",
                    style={"color": "#475569", "fontSize": "0.46rem", "marginTop": "4px"})
    return html.Div([header, body, note], style={
        "padding": "10px 12px", "borderRadius": "8px", "background": BG_CARD,
        "border": "1px solid #1e3a5f55", "borderLeft": "3px solid #38bdf8",
        "marginBottom": "12px"})


# ── Sidebar card (always visible, clickable) ───────────────────────────────────
def _nav_card(sym: str) -> html.Div:
    slug, color = _slug(sym), COLORS[sym]
    return html.Div(id=f"nav-{slug}", n_clicks=0, children=[
        # Top row: label + live dot + OC hint
        html.Div([
            html.Div([
                html.Span(className="live-dot", style={"marginRight": "6px", "flexShrink": "0"}),
                html.Span(LABELS[sym], style={
                    "color": color, "fontSize": "0.6rem",
                    "letterSpacing": "0.15em", "fontWeight": "800",
                }),
            ], style={"display": "flex", "alignItems": "center"}),
            html.Span("OC ›", style={
                "color": f"{color}55", "fontSize": "0.52rem", "letterSpacing": "0.06em",
            }),
        ], style={"display": "flex", "justifyContent": "space-between",
                  "alignItems": "center", "marginBottom": "7px"}),
        # LTP — hero number
        html.Div(id=f"s-ltp-{slug}", children="—", style={
            **MONO, "fontSize": "1.5rem", "fontWeight": "900", "color": "#f1f5f9",
            "letterSpacing": "-0.02em", "lineHeight": "1",
        }),
        # Change line
        html.Div(id=f"s-chg-{slug}", children="", style={
            **MONO, "fontSize": "0.7rem", "marginTop": "4px",
        }),
        # Decorative gradient bar
        html.Div(style={
            "height": "2px", "marginTop": "9px", "borderRadius": "1px",
            "background": f"linear-gradient(90deg, {color}, {color}22, transparent)",
        }),
    ], style={
        "padding": "13px 15px", "marginBottom": "8px", "borderRadius": "10px",
        "border": f"1px solid {color}22", "borderLeft": f"3px solid {color}",
        "background": f"linear-gradient(135deg, #0c1522 0%, {color}0a 100%)",
        "cursor": "pointer",
    })


# ── Header nav chip (horizontal, clickable) — replaces the old left-pane tile ──
def _nav_chip_style(sym: str, selected: bool = False) -> dict:
    """Base style for a header index chip. Shared with toggle_view so the selected
    highlight stays in sync with the layout (the callback overwrites this style)."""
    c = COLORS[sym]
    return {
        "display": "flex", "alignItems": "center", "gap": "8px",
        "padding": "6px 12px", "borderRadius": "8px",
        "border": f"1px solid {c}{'aa' if selected else '33'}",
        "borderBottom": f"2px solid {c}",
        "background": f"{c}1f" if selected else BG_CARD,
        "cursor": "pointer", "transition": "background 0.15s",
        "whiteSpace": "nowrap",
    }


def _header_nav_card(sym: str) -> html.Div:
    """Compact horizontal index chip (label · LTP · change · OC) for the header bar.
    Keeps the same ids (nav-/s-ltp-/s-chg-) as the old tile so callbacks are unchanged."""
    slug, color = _slug(sym), COLORS[sym]
    return html.Div(id=f"nav-{slug}", n_clicks=0,
                    title=f"Open {LABELS[sym]} option chain", children=[
        html.Span(className="live-dot", style={"flexShrink": "0"}),
        html.Span(LABELS[sym], style={
            "color": color, "fontSize": "0.55rem",
            "letterSpacing": "0.12em", "fontWeight": "800"}),
        html.Span(id=f"s-ltp-{slug}", children="—", style={
            **MONO, "fontSize": "0.95rem", "fontWeight": "900",
            "color": "#f1f5f9", "lineHeight": "1", "letterSpacing": "-0.02em"}),
        html.Span(id=f"s-chg-{slug}", children="", style={**MONO, "fontSize": "0.6rem"}),
        html.Span("OC ›", style={
            "color": f"{color}66", "fontSize": "0.5rem", "letterSpacing": "0.05em"}),
    ], style=_nav_chip_style(sym))


def _header_action_chip(id_: str, icon: str, label: str, color: str,
                        extra_child=None) -> html.Div:
    """Compact clickable section button (Today's Trades / Live OI) for the header bar."""
    children = [
        html.Span(icon, style={"fontSize": "0.72rem"}),
        html.Span(label, style={"color": "#cbd5e1", "fontSize": "0.55rem",
                                "fontWeight": "700", "letterSpacing": "0.1em"}),
    ]
    if extra_child is not None:
        children.append(extra_child)
    return html.Div(id=id_, n_clicks=0, children=children, style={
        "display": "flex", "alignItems": "center", "gap": "7px",
        "padding": "6px 12px", "borderRadius": "8px",
        "border": f"1px solid {color}55", "borderBottom": f"2px solid {color}",
        "background": BG_CARD, "cursor": "pointer", "whiteSpace": "nowrap",
    })


# ── Overview panel components ──────────────────────────────────────────────────
def _overview_card(sym: str) -> dbc.Col:
    slug, color = _slug(sym), COLORS[sym]
    return dbc.Col(
        dbc.Card(dbc.CardBody([
            # Label + live dot
            html.Div([
                html.Span(LABELS[sym], style={
                    "color": color, "fontSize": "0.6rem",
                    "letterSpacing": "0.18em", "fontWeight": "800",
                }),
                html.Span(className="live-dot", style={"marginLeft": "8px"}),
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "8px"}),
            # LTP hero
            html.Div(id=f"ov-ltp-{slug}", style={
                **MONO, "fontSize": "2.2rem", "fontWeight": "900",
                "color": "#f1f5f9", "lineHeight": "1", "letterSpacing": "-0.03em",
            }),
            # Change line
            html.Div(id=f"ov-chg-{slug}", style={
                **MONO, "fontSize": "0.9rem", "marginTop": "5px", "fontWeight": "600",
            }),
            # Animated separator
            html.Div(style={
                "height": "1px",
                "background": f"linear-gradient(90deg, {color}55, {color}22, transparent)",
                "margin": "10px 0 9px",
            }),
            # OHLPC
            html.Div(id=f"ov-ohlpc-{slug}", style={
                **MONO, "fontSize": "0.65rem", "lineHeight": "1.95",
            }),
        ]), style={
            "background": f"linear-gradient(160deg, #0c1826 0%, {color}0e 100%)",
            "borderRadius": "12px",
            "border": f"1px solid {color}22",
            "borderTop": f"3px solid {color}",
        }, className="depth-card"),
        md=3, xs=6, className="mb-3 px-2",
    )


# ── Dash app ───────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    title="NSE Index Live",
    suppress_callback_exceptions=True,
)

app.index_string = app.index_string.replace("</head>", _CSS + "</head>")


_HELP_OPTIONS = {
    "intro": ("OPTIONS FLOW — five lenses on one timeframe. PRICE = WHAT happened · OI = WHO is "
              "defending where (the walls) · VOLUME = IF it is real · STRADDLE = HOW scared/calm "
              "the market is. Read together = WHY a move happened, not just that it did.  "
              "STRIKE PICKER: 'Totals' = aggregate CE/PE; or pick a strike — the ladder is FIXED at "
              "the 9:15 OPEN ±1000 (round-100 strikes, labelled by offset from open e.g. +300 / -700) "
              "— to drill into THAT strike's CE/PE OI, premium and write-vs-buy. The per-strike "
              "positioning panel is DELTA-ADJUSTED (Δprem − delta·Δindex = the index move removed), so "
              "buy vs write is the genuine aggressor read, not just price echoing. How to justify "
              "BUYING vs WRITING: OI building + delta-adjusted residual UP = aggressive BUYING; "
              "residual flat/DOWN = WRITING (the 90-95% case). Confirm with the futures panel — e.g. "
              "'call writing' but futures show long-buildup means the calls are being BOUGHT, not written."),
    "terms": [
        ("Price — candles + volume", "#e2e8f0",
         "WHAT: index OHLC per bar (green up/red down; long wick = rejected level), volume bars "
         "at the base.  WHY: the actual move + where buyers/sellers fight.  HOW: trend + rejections."),
        ("CE OI (red)", "#ef4444",
         "WHAT: total CALL open interest.  WHY: call writing = sellers capping = RESISTANCE/ceiling.  "
         "HOW: RISING = ceiling building (capped); FALLING = calls unwinding/short-cover = ceiling "
         "lifting (bullish fuel)."),
        ("PE OI (green)", "#22c55e",
         "WHAT: total PUT open interest.  WHY: put writing = sellers defending below = SUPPORT/floor.  "
         "HOW: RISING = floor building (bullish); FALLING = puts unwinding = floor cracking (bearish)."),
        ("ATM straddle + IV (amber / purple)", "#fbbf24",
         "WHAT: the AT-THE-MONEY call + put added together (the strike nearest the live INDEX, e.g. "
         "24,100 CE 36 + PE 46 = 82). It re-picks the ATM strike as the index moves, so it is always "
         "the CURRENT ATM-straddle cost (a small step when the index crosses a strike is normal). "
         "Dotted purple = ATM IV.  WHY: the straddle PRICE ≈ the market's EXPECTED MOVE — roughly how "
         "many points the index is expected to travel up OR down by expiry; buy it and you need a "
         "bigger move than that to profit.  IMPORTANT: it tells you HOW BIG a move may come — NOT up "
         "or down. Pure volatility, direction stripped out. For DIRECTION read the OI walls + "
         "positioning flow (options) and the futures basis / OI.  HOW: HIGH / RISING = a big move "
         "being priced (event / vol expanding) → option BUYERS favoured, breakout brewing; LOW / "
         "FALLING in a range = theta (time) decay bleeding premium → WRITERS win (sell straddle/"
         "strangle); a sudden mid-day SPIKE = vol expansion = a real move starting. Into expiry it "
         "bleeds toward zero (theta crush) — exactly what 06-23 did (82 → 24).  In short: straddle = "
         "'how big?', OI + positioning = 'which way?'."),
        ("Positioning flow (bottom)", "#a78bfa",
         "WHAT: per-bar ΔOI — calls UP, puts DOWN — red=call-write amber=call-buy green=put-write "
         "lime=put-buy hatched-grey=closing.  WHY: the 'what are they doing' read.  HOW: OI building + "
         "that leg's premium UP = BUYING; premium flat/DOWN = WRITING; OI falling = closing."),
    ],
    "combos": [
        "Price UP + CE OI DOWN = short-covering breakout (strong up).",
        "Price DOWN + PE OI DOWN = long-unwinding breakdown (strong down).",
        "Both OI rising in a range = two-sided writing → range tightens toward max-pain.",
    ],
    "caveat": ("Every contract has a buyer AND a writer — 'buy vs write' is the AGGRESSOR inferred from "
               "OI × premium, not certainty. OI turns only count on real volume. ATM premium also "
               "carries ~half the index move (delta), so labels lean with price on strong trend bars."),
}

_HELP_FUTURES = {
    "intro": ("FUTURES — the directional/positioning read. Price+volume per EXPIRY (near/next/far via "
              "the dropdown) · OI = the directional cash · basis = carry · rollover = positions moving "
              "to next month vs exiting."),
    "terms": [
        ("Price — candles + volume (selected expiry)", "#e2e8f0",
         "WHAT: OHLC + volume for the expiry you pick; the OTHER expiries show as dotted context lines.  "
         "WHY: see the near/next/far price ladder.  HOW: pick Near/Next/Far in the dropdown. Far is "
         "price-only (no volume in the feed)."),
        ("Futures OI (consolidated)", "#38bdf8",
         "WHAT: total futures OI across ALL expiries (near dominates ~80-90%).  WHY: directional smart "
         "money — options are often hedges, futures are the real bet.  HOW: RISING into a move = "
         "conviction. NOTE: same number whatever expiry you pick (not split by month)."),
        ("Positioning — ΔOI × price", "#22c55e",
         "WHAT: per-bar ΔOI coloured: green=long-buildup red=short-buildup teal=covering amber=unwinding "
         "(down bars = closing).  WHY: are big players pressing long or short?  HOW: OI↑+price↑=long "
         "buildup (bullish); OI↑+price↓=short buildup (bearish); OI↓+price↑=covering; OI↓+price↓=unwind."),
        ("Basis = futures − index (₹)", "#22c55e",
         "WHAT: the GAP between the FUTURES price and the LIVE INDEX price (cash). Green = futures "
         "above index (premium/contango); red = below (discount).  WHY: it isolates futures DEMAND, "
         "not just price direction.  HOW: bars RISING (premium widening) = futures bid up faster than "
         "the index = long demand GROWING (bullish); bars FALLING (premium shrinking) = longs "
         "unwinding / demand fading (bearish); flip to RED/discount = aggressive futures selling "
         "(strong bearish). Read WITH price: price-up + basis-up = real long buildup; price-up + "
         "basis-flat/down = hollow rally."),
        ("Rollover (bottom)", "#fbbf24",
         "WHAT: two lines — amber = roll spread = next-month PRICE minus near-month PRICE in ₹ (a price "
         "gap, NOT open interest; e.g. near 24,100, next 24,200 → 100); teal = % of futures VOLUME in "
         "the NEXT month. (Open interest is the separate blue 'Futures OI' panel above — contracts, not "
         "₹.)  WHY: futures EXPIRE, so to KEEP a position you must move it near→next ('roll'). This "
         "separates a ROLL from an EXIT.  HOW: both lines RISING = people are rolling, so a falling "
         "near-OI is just moving forward (NOT bearish); both flat while near-OI + total OI fall = a real "
         "EXIT (bearish).  Mostly only active in the last ~2-3 days before monthly expiry — quiet "
         "mid-month is normal."),
    ],
    "combos": [
        "Price UP + OI UP (green) + basis premium widening = longs pressing, strong real up.",
        "Price DOWN + OI UP (red) = fresh shorts building, strong down.",
        "Price DOWN + OI DOWN (long-unwind) = longs exiting — can bounce if just unwinding.",
        "Near fading + next-vol share rising + total OI flat = rollover, not a bearish exit.",
    ],
    "caveat": ("OI is consolidated (all expiries) — can't split near/next/far intraday; per-expiry OI "
               "needs the EOD bhavcopy. Far has no volume. Every contract has a long AND a short, so "
               "'long/short buildup' is the aggressor read, not certainty."),
}


def _charts_help(mode="options") -> "html.Details":
    """Mode-aware plain-English explainer for the Charts section. Rendered into its own
    row by a callback so opening it never disrupts the dropdown controls."""
    h = _HELP_FUTURES if mode == "futures" else _HELP_OPTIONS
    body = [html.Div(h["intro"], style={"color": "#cbd5e1", "fontSize": "0.58rem",
                     "lineHeight": "1.5", "marginBottom": "7px", "whiteSpace": "normal"})]
    for name, clr, txt in h["terms"]:
        body.append(html.Div([
            html.Span(name + " — ", style={"color": clr, "fontWeight": "700"}),
            html.Span(txt, style={"color": "#94a3b8"}),
        ], style={"fontSize": "0.56rem", "lineHeight": "1.5", "marginBottom": "5px",
                  "whiteSpace": "normal"}))
    body.append(html.Div("Read them together", style={"color": "#67e8f9", "fontWeight": "700",
                "fontSize": "0.56rem", "marginTop": "6px", "marginBottom": "3px"}))
    body += [html.Div("• " + c, style={"color": "#94a3b8", "fontSize": "0.56rem",
             "lineHeight": "1.5", "whiteSpace": "normal"}) for c in h["combos"]]
    body.append(html.Div("⚠ " + h["caveat"], style={"color": "#fbbf24", "fontSize": "0.56rem",
                "lineHeight": "1.45", "marginTop": "6px", "whiteSpace": "normal"}))
    return html.Div(body)   # rendered inside the help popup (dbc.Modal body)


def _captured_days() -> list:
    """All captured session days in the active mirror dir, NEWEST FIRST. This is the
    master date list — the header ◀▶ nav drives the cards AND the Charts off it."""
    from core.constants import LIVE_DIR, today_iso
    have = set()
    try:
        for p in LIVE_DIR.glob("*_oi_snapshots.parquet"):
            if p.stat().st_size > 800:
                have.add(p.name[:10])
    except Exception:
        pass
    # ALWAYS include the actual calendar today so the LIVE default is today from the first
    # load — even before this morning's capture has flushed a mirror (the >800B gate would
    # otherwise freeze the default to the last captured day, showing yesterday all session).
    have.add(today_iso())
    return sorted(have, reverse=True)


_DEFAULT_DAY = _captured_days()[0]


app.layout = dbc.Container([
    # ── Header (brand + status, then a horizontal index/nav strip) ──────────────
    html.Div([
        dbc.Row([
            dbc.Col(html.Div([
                html.Span("◆ ", style={
                    "color": "#00d4ff", "fontSize": "0.9rem", "marginRight": "8px"}),
                html.Span("NSE", style={
                    "fontSize": "0.72rem", "fontWeight": "900", "letterSpacing": "0.3em",
                    "color": "#00b4d8", "marginRight": "8px"}),
                html.Span("INDEX LIVE", style={
                    "fontSize": "0.82rem", "fontWeight": "700", "letterSpacing": "0.2em",
                    "color": "#e2e8f0"}),
                html.Span("  DASHBOARD", style={
                    "fontSize": "0.82rem", "fontWeight": "300", "letterSpacing": "0.2em",
                    "color": "#475569"}),
            ], style={"display": "flex", "alignItems": "center"})),
            dbc.Col(html.Div(id="status", style={
                "textAlign": "right", "fontSize": "0.68rem"}), width="auto"),
        ], className="align-items-center g-0"),
        # NSE holiday / weekend banner — set by callback from the viewed date so the
        # "warming up / no ticks" panels on a closed day read as intended, not broken.
        html.Div(id="holiday-banner"),
        # Horizontal index chips (clickable nav). Today's Trades now lives in the left pane.
        html.Div([
            *[_header_nav_card(sym) for sym in INDEX_SYMBOLS],
            html.Div(style={"flex": "1 1 auto"}),     # spacer pushes actions to the right
            _header_action_chip("nav-charts", "📈", "CHARTS", "#a78bfa"),
            _header_action_chip("nav-liveoi", "📡", "LIVE OI", "#40c4ff"),
            _header_action_chip("nav-tradeboard", "🎯", "TRADEBOARD", "#f472b6"),
            _header_action_chip("btst-btn", "🌙", "BTST", "#22c55e"),
            _header_action_chip(
                "nav-alerts", "🔔", "ALERTS", "#fbbf24",
                extra_child=html.Span(id="alert-badge", children="", style={
                    "fontSize": "0.55rem", "fontWeight": "800", "color": "#0a0f1a",
                    "background": "#fbbf24", "borderRadius": "9px", "padding": "0 6px",
                    "minWidth": "16px", "textAlign": "center", "display": "none"})),
        ], style={"display": "flex", "flexWrap": "wrap", "gap": "8px",
                  "alignItems": "center", "marginTop": "10px"}),
        # STALE-DATA banner (viewer only) — empty/zero-height until the mirror goes stale,
        # then a full-width red bar. A tiny header badge was missed in the wild.
        html.Div(id="mirror-banner", style={"marginTop": "8px"}),
        # Live news / event-impact ticker — always visible across all panels.
        # The date nav (◀ ▶) drives the news-date Store, which is the MASTER date for
        # the whole viewer: news panel, index cards (seed) AND the Charts section.
        dcc.Store(id="news-date"),
        dcc.Store(id="viewer-seed"),   # dummy output for the card-seed-on-date callback
        html.Div([
            html.Span("◀", id="news-prev", n_clicks=0, style={
                "color": "#67e8f9", "fontSize": "0.7rem", "fontWeight": "700",
                "cursor": "pointer", "padding": "0 8px", "userSelect": "none"}),
            html.Span(id="news-date-label", style={
                "color": "#cbd5e1", "fontSize": "0.56rem", "fontWeight": "700",
                "minWidth": "92px", "textAlign": "center", **MONO}),
            html.Span("▶", id="news-next", n_clicks=0, style={
                "color": "#67e8f9", "fontSize": "0.7rem", "fontWeight": "700",
                "cursor": "pointer", "padding": "0 8px", "userSelect": "none"}),
            html.Span("scroll dates ◂▸", style={
                "color": "#475569", "fontSize": "0.46rem", "marginLeft": "8px"}),
        ], style={"display": "flex", "alignItems": "center", "marginTop": "10px"}),
        # Bucket tabs — filter the news panel by trader-facing lens. Kept in the STATIC
        # layout (not inside news-panel) so the selection survives the 30s tick refresh.
        html.Div([
            dcc.RadioItems(
                id="news-tab", value="ALL", inline=True,
                options=[{"label": "ALL", "value": "ALL"},
                         {"label": "🟢 Bullish", "value": "BULLISH"},
                         {"label": "🔴 Bearish", "value": "BEARISH"}],
                className="news-tab",
                # color set ON the label — inherited color gets overridden to
                # near-invisible by the flex wrapper/theme (labels vanished)
                labelStyle={"display": "inline-block", "marginRight": "12px",
                            "cursor": "pointer", "fontSize": "0.6rem",
                            "fontWeight": "700", "color": "#cbd5e1"},
                inputStyle={"marginRight": "4px", "cursor": "pointer"},
                style={"color": "#cbd5e1"}),
            # Order: ⏱ full-day chronological tape (review a whole session start-to-
            # finish, deduped, uncapped) vs ⚡ top-impact glance (old default, top 25).
            dcc.RadioItems(
                id="news-sort", value="time", inline=True,
                options=[{"label": "⏱ latest first (full day)", "value": "time"},
                         {"label": "⚡ top impact", "value": "impact"}],
                className="news-tab",
                labelStyle={"display": "inline-block", "marginRight": "12px",
                            "cursor": "pointer", "fontSize": "0.6rem",
                            "fontWeight": "700", "color": "#cbd5e1"},
                inputStyle={"marginRight": "4px", "cursor": "pointer"},
                style={"color": "#cbd5e1", "marginLeft": "18px"}),
        ], style={"display": "flex", "alignItems": "center", "marginTop": "6px"}),
        # NEWS (left, flexible — headlines need width) + MACRO RADAR (right, a
        # narrow fixed factor table that wasted a full-width band on its own row).
        # flexWrap stacks them again on a narrow window; minWidth:0 lets the news
        # headlines ellipsis inside a flex child instead of forcing overflow.
        html.Div([
            html.Div(id="news-panel", style={"flex": "1 1 auto", "minWidth": "0"}),
            html.Div(id="macro-radar-panel",
                     style={"flex": "0 0 360px", "minWidth": "300px"}),
        ], style={"display": "flex", "gap": "12px", "alignItems": "flex-start",
                  "flexWrap": "wrap", "marginTop": "4px"}),
        # Cockpit REPLAY control — type a past time (HH:MM) to see the band/regime AS IT
        # STOOD at that instant on the selected date (causal, lookahead-free). Blank = live/
        # latest tick. Persistent (outside cockpit-panel) so the typed value survives the 30s
        # tick re-render. Combined with the ◀▶ date nav: pick the day, type the minute.
        html.Div([
            html.Span("cockpit replay →", style={
                "color": "#67e8f9", "fontSize": "0.52rem", "fontWeight": "700", **MONO}),
            dcc.Input(id="cockpit-asof", type="text", debounce=True, value="",
                      placeholder="live · type HH:MM", style={
                          "width": "120px", "marginLeft": "6px", "fontSize": "0.55rem",
                          "background": "#0b1220", "color": "#e2e8f0",
                          "border": "1px solid #334155", "borderRadius": "4px",
                          "padding": "1px 6px", **MONO}),
            html.Span("↵ band as it stood at that minute (past days too)", style={
                "color": "#475569", "fontSize": "0.46rem", "marginLeft": "8px"}),
        ], style={"display": "flex", "alignItems": "center", "marginTop": "8px"}),
        html.Div(id="cockpit-panel", style={"marginTop": "4px"}),
    ], style={"padding": "14px 16px 10px"}),
    html.Div(className="header-line"),

    dbc.Row([
        # ── Left pane: Intraday-TF footprint matrix + Today's Trades ─────────────
        dbc.Col([
            # One index selector drives BOTH the TF footprint AND that index's trades.
            html.Div([
                html.Span("📊 FOOTPRINT", style={"color": "#67e8f9", "fontWeight": "700",
                          "fontSize": "0.62rem", "letterSpacing": "0.1em"}),
                html.Span("  live monitor", style={"color": "#475569", "fontSize": "0.5rem"}),
                dcc.Dropdown(
                    id="itf-idx", clearable=False,
                    options=[{"label": LABELS[s], "value": s} for s in INDEX_SYMBOLS],
                    value="NSE:NIFTY50-INDEX",
                    style={"fontSize": "0.62rem", "marginTop": "5px", "color": "#0b1320"}),
            ], style={"marginBottom": "6px"}),
            # Footprint matrix for the SELECTED index (own-charts workflow; the
            # ALL-TRADES / smart-money footer was removed — charts replaced it).
            dcc.Loading(html.Div(id="itf-content"), type="circle", color="#67e8f9"),
        ], md=3, lg=3, style={
            "background": BG_SIDE, "padding": "14px 10px",
            "borderRight": "1px solid #111d2e", "minHeight": "calc(100vh - 120px)"}),

        # ── Main content ─────────────────────────────────────────────────────────
        dbc.Col([
            # OVERVIEW PANEL
            html.Div(id="overview-panel", children=[
                # Live index prices (4 cards)
                dbc.Row([_overview_card(s) for s in INDEX_SYMBOLS], className="gx-0 mb-2"),
                # Multi-timeframe trend ribbon (5m→weekly) — alignment vs divergence
                html.Div(id="trend-panel"),
                # ── INDEX PREDICTION — above chart so visible on first load ──────
                # This is tomorrow's directional forecast from the 24-signal engine.
                # Placed here (not after the chart) so analysts see it immediately.
                html.Div(id="context-panel"),
                # Intraday % change chart
                dbc.Card(dbc.CardBody([
                    html.Div("INTRADAY  %  CHANGE  FROM  PREVIOUS  CLOSE", style={
                        "fontSize": "0.62rem", "letterSpacing": "0.12em",
                        "color": "#1e3a5f", "marginBottom": "4px",
                    }),
                    dcc.Graph(id="ov-chart",
                              config={"displayModeBar": False},
                              style={"height": "220px"}),
                ]), style={"background": BG_CARD, "border": "1px solid #111d2e",
                           "borderRadius": "10px", "marginBottom": "12px"}),
                # ── Multi-timeframe candlestick chart (1m / 5m / 15m / 1h) ────────
                dbc.Card(dbc.CardBody([
                    dbc.Row([
                        dbc.Col(html.Div("CANDLES", style={
                            "fontSize": "0.62rem", "letterSpacing": "0.12em",
                            "color": "#1e3a5f", "fontWeight": "700", "paddingTop": "6px"}), md=2),
                        dbc.Col(dcc.Dropdown(
                            id="cndl-idx", clearable=False,
                            options=[{"label": LABELS[s], "value": s} for s in INDEX_SYMBOLS],
                            value="NSE:NIFTY50-INDEX", style={"fontSize": "0.72rem"}), md=4),
                        dbc.Col(dcc.Dropdown(
                            id="cndl-tf", clearable=False,
                            options=[{"label": "1 Sec", "value": "1S"}, {"label": "5 Sec", "value": "5S"},
                                     {"label": "15 Sec", "value": "15S"}, {"label": "30 Sec", "value": "30S"},
                                     {"label": "1 Min", "value": "1"}, {"label": "5 Min", "value": "5"},
                                     {"label": "15 Min", "value": "15"}, {"label": "1 Hour", "value": "60"},
                                     {"label": "Daily", "value": "D"}],
                            value="5", style={"fontSize": "0.72rem"}), md=3),
                    ], className="gx-1 mb-1"),
                    dcc.Graph(id="cndl-chart", config={"displayModeBar": False},
                              style={"height": "380px"}),
                ]), style={"background": BG_CARD, "border": "1px solid #111d2e",
                           "borderRadius": "10px", "marginBottom": "12px"}),
                # Trade signals panel
                html.Div(id="signal-panel"),
            ]),

            # OPTION CHAIN PANEL (hidden until index clicked)
            html.Div(id="oc-panel", style={"display": "none"}, children=[
                # Top bar: title + expiry + metrics + TIMEFRAME selector
                dbc.Row([
                    dbc.Col(html.Div(id="oc-title"), md=3),
                    dbc.Col(
                        dcc.Dropdown(id="expiry-dd",
                                     placeholder="Select expiry...",
                                     clearable=False,
                                     style={"fontSize": "0.75rem"}),
                        md=2,
                    ),
                    dbc.Col(
                        dcc.Dropdown(
                            id="tf-dd",
                            options=[
                                {"label": "5 Min  — Intraday Scalp",    "value": "5min"},
                                {"label": "15 Min — Intraday Swing",    "value": "15min"},
                                {"label": "1 Hour — BTST / Positional", "value": "60min"},
                                {"label": "Daily  — Swing Positional",  "value": "daily"},
                            ],
                            value="15min",
                            clearable=False,
                            style={"fontSize": "0.75rem"},
                            placeholder="Timeframe...",
                        ),
                        md=3,
                    ),
                    dbc.Col(html.Div(id="oc-metrics", style={
                        **MONO, "fontSize": "0.7rem", "textAlign": "right",
                        "paddingTop": "6px", "color": "#475569",
                    }), md=4),
                ], className="mb-2 align-items-center"),

                # ── TRADE RECOMMENDATION ──────────────────────────────────────
                html.Div(id="trade-rec"),

                # Velocity monitor (OI/IV/wall/PCR session history)
                html.Div(id="velocity-panel"),

                # Futures strip
                html.Div(id="futures-strip"),

                # Prediction section
                html.Div(id="oc-prediction"),

                # Component A: live per-strike OI-Dynamics map
                html.Div(id="oi-intel-panel"),

                # Component C: intraday paper-trade track record
                html.Div(id="track-record"),

                # Option chain table in a scrollable container
                dcc.Loading(
                    type="circle",
                    color="#00d4ff",
                    children=html.Div(id="oc-table",
                                      style={"overflowY": "auto",
                                             "maxHeight": "calc(100vh - 300px)"}),
                ),
            ]),

            # TRADE BOOK PANEL (hidden until 'Today's Trades' is clicked)
            html.Div(id="trade-book-panel", style={"display": "none"}),

            # LIVE OI PANEL (hidden until 'Live OI' is clicked)
            html.Div(id="live-oi-panel", style={"display": "none"}, children=[
                dbc.Row([
                    dbc.Col(html.Div("📡 LIVE OI — SESSION TIME-SERIES", style={
                        "color": "#40c4ff", "fontWeight": "700", "fontSize": "0.95rem",
                        "letterSpacing": "0.08em", "paddingTop": "6px"}), md=7),
                    dbc.Col(dcc.Dropdown(
                        id="liveoi-idx", clearable=False,
                        options=[{"label": LABELS[s], "value": s} for s in INDEX_SYMBOLS],
                        value="NSE:NIFTY50-INDEX", style={"fontSize": "0.75rem"}), md=4),
                ], className="mb-2 align-items-center"),
                html.Div(id="liveoi-content"),
            ]),

            # ALERTS PANEL (hidden until 🔔 clicked) — running log of NEW scout TRADE
            # triggers detected live (also fires a browser notification + beep). The
            # arrow is decision-support (negative-EV) — this surfaces the lean, it is
            # NOT a buy instruction. See the red note in the panel.
            html.Div(id="alerts-panel", style={"display": "none"}, children=[
                dbc.Row([
                    dbc.Col(html.Span("🔔 TRADE ALERTS", style={
                        "color": "#fbbf24", "fontWeight": "700", "fontSize": "0.9rem",
                        "letterSpacing": "0.06em", "paddingTop": "6px"}), md=3),
                    dbc.Col([
                        html.Span("hour ", style={"fontSize": "0.6rem", "color": "#64748b",
                                                  "fontWeight": "700"}),
                        dcc.Dropdown(
                            id="alert-hour", clearable=False, value="all",
                            options=[{"label": "all day", "value": "all"}] +
                                    [{"label": f"{h:02d}:00", "value": h} for h in range(9, 16)],
                            style={"width": "110px", "display": "inline-block",
                                   "fontSize": "0.62rem", "verticalAlign": "middle"}),
                    ], md=3, style={"paddingTop": "4px"}),
                    dbc.Col([
                        html.Button("Enable browser notifications", id="alert-perm-btn",
                                    n_clicks=0, style={
                                        "fontSize": "0.62rem", "fontWeight": "700",
                                        "color": "#fbbf24", "background": "#1e293b",
                                        "border": "1px solid #fbbf24", "borderRadius": "6px",
                                        "padding": "4px 10px", "cursor": "pointer"}),
                        html.Span(id="alert-perm-status", style={
                            "fontSize": "0.56rem", "color": "#64748b", "marginLeft": "8px"}),
                    ], md=6, style={"textAlign": "right", "paddingTop": "4px"}),
                ], className="mb-2 align-items-center"),
                html.Div(id="alerts-content"),
            ]),

            # CHARTS PANEL (hidden until 'Charts' is clicked) — full-session
            # price candle + OI + volume + premium for a chosen index & timeframe.
            html.Div(id="charts-panel", style={"display": "none"}, children=[
                dbc.Row([
                    dbc.Col(html.Span("📈 CHARTS", style={
                        "color": "#a78bfa", "fontWeight": "700", "fontSize": "0.9rem",
                        "letterSpacing": "0.06em", "paddingTop": "6px",
                        "display": "inline-block"}), md=2),
                    dbc.Col(dcc.Dropdown(
                        id="charts-mode", clearable=False,
                        options=[{"label": "⚙ Options flow", "value": "options"},
                                 {"label": "🛢 Futures", "value": "futures"}],
                        value="options", style={"fontSize": "0.72rem"}), md=2),
                    dbc.Col(dcc.Dropdown(
                        id="charts-leg", clearable=False,
                        options=[{"label": "Near expiry", "value": "near"},
                                 {"label": "Next expiry", "value": "next"},
                                 {"label": "Far expiry", "value": "far"}],
                        value="near", style={"fontSize": "0.72rem"}),
                        # only shown in Futures mode (toggled by _toggle_mode_cols)
                        id="charts-leg-col", md=2, style={"display": "none"}),
                    dbc.Col(dcc.Dropdown(
                        id="charts-expiry", clearable=False,
                        # Capture stores ONLY the nearest tradeable expiry per index
                        # (NIFTY weekly · others monthly — NSE killed non-NIFTY
                        # weeklies). "Monthly" was a phantom that blanked the chart, so
                        # there is one honest option; value stays "weekly" = the
                        # legacy/nearest bucket build_series serves.
                        options=[{"label": "Nearest expiry (NIFTY wk · others monthly)",
                                  "value": "weekly"}],
                        value="weekly", style={"fontSize": "0.72rem"}),
                        # only shown in Options mode (toggled by _toggle_mode_cols)
                        id="charts-expiry-col", md=2),
                    dbc.Col(dcc.Dropdown(
                        id="charts-strike", clearable=False,
                        options=[{"label": "Totals (CE/PE)", "value": "totals"}],
                        value="totals", style={"fontSize": "0.72rem"}),
                        # only shown in Options mode (populated by _fill_strikes)
                        id="charts-strike-col", md=2),
                    dbc.Col(dcc.Dropdown(
                        id="charts-idx", clearable=False,
                        options=[{"label": LABELS[s], "value": s} for s in INDEX_SYMBOLS],
                        value="NSE:NIFTY50-INDEX", style={"fontSize": "0.72rem"}), md=2),
                    dbc.Col(dcc.Dropdown(
                        id="charts-tf", clearable=False,
                        options=[{"label": "5 min", "value": 5},
                                 {"label": "15 min", "value": 15},
                                 {"label": "60 min", "value": 60}],
                        value=60, style={"fontSize": "0.72rem"}), md=2),
                    # Replay: pick ANY cutoff minute (truncate every chart at that
                    # time to study what the market did next). Cleared = full/live
                    # session — the as_of cutoff is leakage-safe (enforced at the
                    # data read). A searchable dropdown (NOT a native <input type=time>,
                    # whose picker always lists 00-23h and ignores min/max) so ONLY
                    # market minutes 09:15-15:30 are offered — no junk 17:00 / 22:00.
                    # Type "1130" to jump; clear (×) for live.
                    dbc.Col([
                        html.Div([
                            html.Span("⏱ Replay", style={
                                "fontSize": "0.62rem", "fontWeight": "700",
                                "color": "#a78bfa", "whiteSpace": "nowrap"}),
                            dcc.Dropdown(
                                id="charts-asof",
                                # "ghost" = PRACTICE mode: replay the last captured
                                # session pinned to TODAY'S wall clock (10:42 now →
                                # 10:42 on that day), auto-advancing — paper-trade a
                                # past day as if it were live. Verify/actual is hidden
                                # until 15:30 so the future can't be peeked.
                                options=[{"label": "👻 ghost live (practice)",
                                          "value": "ghost"}]
                                + [{"label": f"{h:02d}:{m:02d}",
                                    "value": f"{h:02d}:{m:02d}"}
                                   for h in range(9, 16) for m in range(60)
                                   if (9, 15) <= (h, m) <= (15, 30)],
                                # Default LIVE. The weekend ghost default is applied
                                # per PAGE LOAD by _ghost_boot (a static layout is
                                # built once at process start — baking the day check
                                # in here goes stale when a Saturday-started process
                                # survives into Monday and would boot ghost on a
                                # LIVE trading day).
                                value=None, clearable=True, placeholder="live",
                                style={"fontSize": "0.72rem", "minWidth": "110px"}),
                        ], style={"display": "flex", "alignItems": "center",
                                  "gap": "6px"}),
                    ], md=3),
                ], className="mb-2 align-items-center"),
                # SCOUT — multi-index TRADE/NO-TRADE scan at the chosen TF + replay
                # clock. Scans all 4 indices off the SAME series these charts plot;
                # says which index has the cleanest crossover/divergence/flow setup
                # right now (or under the Replay cutoff). Decision-support — direction
                # is null/contrarian in backtests, the range band is the honest part.
                # delay_show: the scout re-renders on the 30s setup-tick; without a delay the
                # spinner flashed over the panel every cycle (and the lifecycle walk-back makes
                # the callback slow), blanking what you're reading and breaking focus. Keep the
                # OLD content visible and only show the spinner if a load genuinely hangs (>8s).
                dcc.Loading(html.Div(id="charts-scout", style={"marginBottom": "10px"}),
                            type="circle", color="#34d399", delay_show=12000),
                # Mode-aware help in a POPUP — opens over the chart, closes via the X /
                # click-outside, so it never covers or pushes the chart down.
                dbc.Button("ℹ what is this · how to read", id="charts-help-btn",
                           color="link", size="sm",
                           style={"color": "#67e8f9", "fontSize": "0.62rem", "padding": "0 0 6px 0",
                                  "textDecoration": "none"}),
                dbc.Modal([
                    dbc.ModalHeader(dbc.ModalTitle("⚙ Options flow — what is this · how to read",
                                                   id="charts-help-title"), close_button=True),
                    dbc.ModalBody(html.Div(_charts_help("options"), id="charts-help-box")),
                ], id="charts-help-modal", is_open=False, size="lg", scrollable=True),
                # Open-positions popup — every stance the alert log is still holding
                # (since 09:35 post-warmup) + a live cross-check. Surfaces the sticky
                # holds that make the strip read as contradictory across indices.
                dbc.Modal([
                    dbc.ModalHeader(html.Div([
                        dbc.ModalTitle(
                            "📋 Scout day ledger — open + closed (since 09:35)"),
                        dcc.Input(
                            id="scout-search", type="search", autoComplete="off",
                            placeholder="🔍 search rows…", debounce=False,
                            style={"width": "190px", "fontSize": "0.72rem",
                                   "padding": "4px 8px", "borderRadius": "6px",
                                   "border": "1px solid #334155",
                                   "background": "#0b1220", "color": "#e2e8f0"}),
                    ], style={"display": "flex", "alignItems": "center",
                              "justifyContent": "space-between", "width": "100%",
                              "gap": "12px"}), close_button=True),
                    dbc.ModalBody(dcc.Loading(
                        html.Div(id="scout-openpos-body"),
                        type="circle", color="#67e8f9")),
                ], id="scout-openpos-modal", is_open=False, size="lg", scrollable=True),
                # BTST overnight paper ledger — the ONE validated edge, finally visible.
                dbc.Modal([
                    dbc.ModalHeader(dbc.ModalTitle(
                        "🌙 BTST overnight — close-strength paper ledger"),
                        close_button=True),
                    dbc.ModalBody(dcc.Loading(html.Div(id="btst-body"),
                                              type="circle", color="#22c55e")),
                ], id="btst-modal", is_open=False, size="lg", scrollable=True),
                dcc.Loading(dcc.Graph(
                    id="charts-graph",
                    config={
                        # zoom modebar (+ / − / autoscale / reset) + scroll-zoom,
                        # plus draw tools for support/resistance lines. 'drawline'
                        # = draw an S/R line (drag flat for a level); 'eraseshape'
                        # removes one. Shapes are editable (edits.shapePosition) so
                        # a level can be dragged after drawing.
                        "displayModeBar": True, "displaylogo": False,
                        "scrollZoom": True,
                        "modeBarButtonsToAdd": ["drawline", "drawrect", "eraseshape"],
                        "modeBarButtonsToRemove": ["lasso2d", "select2d",
                                                   "toggleSpikelines"],
                        "edits": {"shapePosition": True},
                    }),
                    type="circle", color="#a78bfa", delay_show=800),
                # Descriptive positioning map (Options mode only): where today's
                # live OI is REINFORCING vs ABANDONING last night's EOD positions,
                # per strike, anchored to the DCM EOD baseline. CONTEXT, not a call
                # — the directional edge is unproven (backtest_reconciliation null).
                html.Div(id="charts-recon", style={"marginTop": "8px"}),
            ]),
            # 🎯 TRADEBOARD — a proper routed PAGE (not a modal). Shows when sel-sym=='TRADEBOARD'
            # (toggle_view). Per index: 2x2 multi-TF candle grid (5/15/30/60m) + band overlay +
            # structure + full option chain. DATA for your own PA; no machine trade (fade edge
            # was a fill artifact). Index tabs switch symbol.
            html.Div(id="tradeboard-panel", style={"display": "none"}, children=[
                # live refresh — matched to the ~25-30s capture cadence (the VM writes a tick/
                # chain snapshot ~every 25s; the board's DECISIONS only change on 15m closes,
                # but spot/premium/tape are live fields and should track the feed). Callbacks
                # gate on sel-sym so the interval costs nothing on other pages.
                dcc.Interval(id="tb-refresh", interval=30_000),
                html.Div("🎯 TRADEBOARD — multi-TF price action · structure · option chain",
                         style={"color": "#f472b6", "fontWeight": "800", "fontSize": "0.9rem",
                                "letterSpacing": "0.08em", "marginBottom": "6px"}),
                # ── SCOUT — cross-index price-action scan, best setup first (default 15m→1h) ──
                html.Div([
                    html.Span("🔭 SCOUT — scan all indices · lower(entry) × higher(confirm)",
                              style={"color": "#22d3ee", "fontSize": "0.62rem", "fontWeight": "800",
                                     "letterSpacing": "0.1em"}),
                    dbc.Select(id="tb-scout-combo", value="15-60", style={"width": "170px",
                               "fontSize": "0.72rem", "marginLeft": "12px"}, options=[
                        {"label": "5m → 15m", "value": "5-15"},
                        {"label": "10m → 30m", "value": "10-30"},
                        {"label": "15m → 1h", "value": "15-60"}]),
                    html.Div(id="tb-ledger-badge", style={"marginLeft": "auto",
                             "fontSize": "0.62rem"}),
                    html.Div("📋 SCOUT LEDGER", id="tb-ledger-btn", n_clicks=0, style={
                        "cursor": "pointer", "marginLeft": "12px", "color": "#22d3ee",
                        "fontSize": "0.62rem", "fontWeight": "800", "border": "1px solid #22d3ee55",
                        "borderRadius": "6px", "padding": "4px 10px", "whiteSpace": "nowrap"}),
                ], style={"display": "flex", "alignItems": "center", "margin": "4px 0"}),
                dcc.Loading(html.Div(id="tb-scout"), type="circle", color="#22d3ee"),
                dbc.Modal([
                    dbc.ModalHeader(dbc.ModalTitle(
                        "📋 Scout day ledger — open + closed (PA level-trades)"),
                        close_button=True),
                    dbc.ModalBody(dcc.Loading(html.Div(id="tb-ledger"), type="circle",
                                              color="#22d3ee")),
                ], id="tb-ledger-modal", is_open=False, size="xl", scrollable=True),
                dbc.Tabs([dbc.Tab(label=LABELS.get(s, s), tab_id=s) for s in INDEX_SYMBOLS],
                         id="tb-idx-tabs", active_tab=INDEX_SYMBOLS[0]),
                html.Div(id="tb-intro"),
                dcc.Loading(type="circle", color="#f472b6", children=[
                    # 1D — the TOP of the stack, the ONE validated edge (overnight)
                    html.Div("DAILY · OVERNIGHT — the validated directional edge",
                             style={"color": "#22c55e", "fontSize": "0.6rem", "fontWeight": "800",
                                    "letterSpacing": "0.12em", "margin": "6px 0 2px"}),
                    dbc.Row([
                        dbc.Col(dcc.Graph(id="tb-fig-1d", config={"displayModeBar": False}),
                                xs=12, md=7),
                        dbc.Col(html.Div(id="tb-daily"), xs=12, md=5),
                    ], className="g-2"),
                    # ── MTF COMBO FOCUS — pick a master combo, see LTF entry + HTF confirm ──
                    html.Div([
                        html.Span("MTF COMBO FOCUS — lower(entry) × higher(confirm)",
                                  style={"color": "#a78bfa", "fontSize": "0.6rem",
                                         "fontWeight": "800", "letterSpacing": "0.12em"}),
                        dbc.Select(id="tb-combo", value="5-15", style={"width": "180px",
                                   "fontSize": "0.72rem", "marginLeft": "12px"}, options=[
                            {"label": "5m → 15m", "value": "5-15"},
                            {"label": "10m → 30m", "value": "10-30"},
                            {"label": "15m → 60m", "value": "15-60"}]),
                    ], style={"display": "flex", "alignItems": "center", "margin": "12px 0 4px"}),
                    html.Div(id="tb-combo-read"),
                    dbc.Row([
                        dbc.Col(dcc.Graph(id="tb-combo-ltf",
                                config={"displayModeBar": False, "scrollZoom": True}), xs=12, md=6),
                        dbc.Col(dcc.Graph(id="tb-combo-htf",
                                config={"displayModeBar": False, "scrollZoom": True}), xs=12, md=6),
                    ], className="g-2"),
                    html.Div("INTRADAY — all timeframes (CONTEXT for your PA) "
                             "· scroll-wheel zooms · drag pans back through history",
                             style={"color": "#64748b", "fontSize": "0.6rem", "fontWeight": "800",
                                    "letterSpacing": "0.12em", "margin": "10px 0 2px"}),
                    dbc.Row([
                        dbc.Col(dcc.Graph(id="tb-fig-5",
                                config={"displayModeBar": False, "scrollZoom": True}), xs=12, md=4),
                        dbc.Col(dcc.Graph(id="tb-fig-10",
                                config={"displayModeBar": False, "scrollZoom": True}), xs=12, md=4),
                        dbc.Col(dcc.Graph(id="tb-fig-15",
                                config={"displayModeBar": False, "scrollZoom": True}), xs=12, md=4),
                    ], className="g-2"),
                    dbc.Row([
                        dbc.Col(dcc.Graph(id="tb-fig-30",
                                config={"displayModeBar": False, "scrollZoom": True}), xs=12, md=6),
                        dbc.Col(dcc.Graph(id="tb-fig-60",
                                config={"displayModeBar": False, "scrollZoom": True}), xs=12, md=6),
                    ], className="g-2"),
                    html.Div(id="tb-chain"),
                ]),
            ]),
        ], md=9, lg=9, style={"padding": "12px 16px"}),
    ], className="gx-0"),

    # State stores
    dcc.Location(id="url", refresh=False),
    # sel-sym (open section) is derived from the URL by _route — restored on every
    # load/refresh — so it needs no client persistence.
    dcc.Store(id="sel-sym",    data=None),
    dcc.Store(id="sel-expiry", data=""),
    # Regime Radar checkpoint lives in a static Store: the dropdown that sets it
    # is rendered dynamically inside the Trade Book, and a callback may not use a
    # dynamically-created component as an Input before it exists in the DOM.
    dcc.Store(id="regime-asof", data="now"),
    # Scout alert state: last verdict + open/last-resolved trade per index (to detect a
    # NEW trigger and survive a NO-TRADE blink), the alert log, a fire counter (bumps →
    # clientside notification+beep), and a JS no-op sink. storage_type="local" so the
    # trade alert HISTORY survives a tab switch / page reload (was memory → wiped on
    # every nav); stale prior-day records are purged on the first tick of a new day.
    dcc.Store(id="scout-seen",       data={}, storage_type="local"),
    dcc.Store(id="scout-alerts",     data=[], storage_type="local"),
    dcc.Store(id="scout-alert-fire", data=0),
    dcc.Store(id="alert-noop",       data=""),

    # Click-to-popup footprint chart (OI · volume · ATM premium per timeframe)
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("", id="fp-modal-title"), close_button=True),
        dbc.ModalBody(dcc.Graph(id="fp-modal-graph", config={"displayModeBar": False})),
    ], id="fp-modal", is_open=False, size="xl", scrollable=True),

    # Intervals
    dcc.Interval(id="fast-tick",   interval=1000,  n_intervals=0),
    dcc.Interval(id="oc-tick",    interval=2000,  n_intervals=0),
    dcc.Interval(id="signal-tick",interval=60000, n_intervals=0),
    dcc.Interval(id="setup-tick", interval=30000, n_intervals=0),
    dcc.Interval(id="news-tick",  interval=60000, n_intervals=0),

], fluid=True, style={"background": BG, "minHeight": "100vh", "padding": "0"})


# ── Compact number formatters ─────────────────────────────────────────────────

# ── Sidebar prediction block ──────────────────────────────────────────────────

def _render_sidebar_pred() -> html.Div:
    """
    Compact always-visible sidebar block showing yesterday's signal verdicts
    per index: FII futures flow, EOD option chain PCR, FPI equity, VIX, breadth,
    futures carry, and HMM statistical regime.
    """
    if not _CTX_BRIDGE_OK:
        return html.Div()

    bridge = _get_ctx_bridge()

    if not bridge.is_available():
        return html.Div(
            html.Div("EOD data loading...", style={
                "color": "#475569", "fontSize": "0.5rem", **MONO,
                "textAlign": "center", "padding": "6px 0",
            }),
            style={"borderTop": "1px solid #1e3a5f", "paddingTop": "10px",
                   "marginTop": "6px"},
        )

    def _sig_row(tag: str, val: str, val_clr: str) -> html.Div:
        return html.Div([
            html.Span(tag, style={
                "color": "#334155", "fontSize": "0.44rem",
                "letterSpacing": "0.06em", "minWidth": "28px",
                "display": "inline-block",
            }),
            html.Span(val, style={
                "color": val_clr, "fontSize": "0.54rem",
                "fontWeight": "700", **MONO,
            }),
        ], style={"display": "flex", "alignItems": "center",
                  "gap": "4px", "marginBottom": "2px"})

    blocks = []
    for sym in INDEX_SYMBOLS:
        data    = bridge.get_panel_data(sym)
        common  = data.get("__common__", {})
        color   = COLORS[sym]
        label   = LABELS[sym]

        direction   = data.get("direction", "")
        raw_score   = data.get("composite_score", 0.0) or 0.0
        confidence  = data.get("confidence", "?")

        dir_sym = "↑" if direction == "UP" else "↓" if direction == "DOWN" else "↔"
        dir_clr = ("#22c55e" if direction == "UP" else
                   "#ef4444" if direction == "DOWN" else "#f59e0b")

        meter = max(0, min(100, int((raw_score + 20) / 40 * 100)))

        # ── FII Futures (B3): FAO net position + 5D flow ──────────────────────
        fii_net   = data.get("fii_fut_net")    # net contracts from fao_participant
        fii_5d    = data.get("fii_5d_cr", 0.0) or 0.0
        fii_today = data.get("fii_today_cr", 0.0) or 0.0
        fii_ref   = fii_net if fii_net is not None else fii_5d
        fii_clr   = ("#22c55e" if (fii_ref or 0) > 0 else
                     "#ef4444" if (fii_ref or 0) < 0 else "#475569")
        if fii_net is not None:
            # Show net OI position (contracts) + today's flow direction
            act    = "A" if fii_today >= 0 else "C"   # ADD / COV — abbreviated
            fii_val = f"{_fmt_contracts(fii_net)} ({act}{_fmt_cr(abs(fii_today))})"
        else:
            fii_val = f"{_fmt_cr(fii_5d)}Cr 5D" if fii_5d else "—"

        # ── EOD Option Chain (B6): PCR + OI change direction ──────────────────
        # feat_pcr = near-expiry PCR from prediction_log (what DCM used for signals)
        # eod_pcr  = all-expiry aggregate PCR from nightly_sync (different scope)
        # Use feat_pcr first to stay consistent with DCM's prediction calculation.
        pcr      = data.get("feat_pcr") or data.get("eod_pcr") or 0.0
        ce_chg   = data.get("eod_call_chg", 0) or 0
        pe_chg   = data.get("eod_put_chg",  0) or 0
        pcr_clr  = ("#22c55e" if pcr > 1.1 else
                    "#ef4444" if 0 < pcr < 0.85 else "#475569")
        if pcr:
            oi_dir = ""
            if abs(ce_chg) > 200_000 or abs(pe_chg) > 200_000:
                oi_dir = " P↑" if pe_chg > ce_chg else " C↑"
            pcr_val = f"{pcr:.2f}{oi_dir}"
        else:
            pcr_val = "—"

        # ── FPI Equity (B7): 5D + today's flow ────────────────────────────────
        fpi_5d    = data.get("fpi_equity_5d_cr", 0.0) or 0.0
        fpi_today = data.get("fpi_equity_today_cr", 0.0) or 0.0
        fpi_clr   = ("#22c55e" if fpi_5d > 1500 else
                     "#ef4444" if fpi_5d < -1500 else "#475569")
        fpi_val   = f"{_fmt_cr(fpi_5d)} 5D" if fpi_5d else "—"

        # ── India VIX (B5): level + 5D absolute change ────────────────────────
        vix     = common.get("india_vix")
        vix_5d  = common.get("india_vix_5d_chg", 0.0) or 0.0
        vix_clr = ("#ef4444" if (vix or 0) > 20 else
                   "#f59e0b" if (vix or 0) > 16 else "#22c55e")
        if vix:
            vdir    = "↑" if vix_5d > 0 else "↓"
            vix_val = f"{vix:.1f} {vdir}{abs(vix_5d):.1f}"
        else:
            vix_val = "—"

        # ── Market Breadth (B4): % advancing + heavy sector ───────────────────
        breadth = common.get("breadth_pct")
        heavy   = common.get("heavy_breadth_pct")
        brd_clr = ("#22c55e" if (breadth or 0) > 65 else
                   "#ef4444" if (breadth or 0) < 35 else "#475569")
        if breadth is not None:
            brd_val = f"{breadth:.0f}%"
            if heavy is not None:
                brd_val += f" H:{heavy:.0f}%"
        else:
            brd_val = "—"

        # ── Futures Carry (feat_carry = raw annualised %) ──────────────────────
        carry   = data.get("feat_carry")
        cry_clr = ("#22c55e" if (carry or 0) >= 4 else
                   "#ef4444" if (carry or 0) < 0 else "#475569")
        cry_val = f"{carry:+.1f}%A" if carry is not None else "—"

        # ── HMM Statistical Regime (B2): state + Hurst context ────────────────
        hmm     = data.get("hmm_state") or "?"
        hurst   = data.get("feat_hurst")
        entropy = data.get("feat_entropy")
        hmm_clr = ("#22c55e" if hmm == "Bull" else
                   "#ef4444" if hmm == "Bear" else "#f59e0b")
        if hurst is not None:
            h_lbl   = "T" if hurst > 0.58 else "R" if hurst < 0.42 else "N"
            hmm_val = f"{hmm} H:{hurst:.2f}{h_lbl}"
        else:
            hmm_val = hmm

        # ── Max pain distance ─────────────────────────────────────────────────
        mp_dist = data.get("max_pain_dist_pct")
        mp_val  = f"{mp_dist:+.1f}%" if mp_dist is not None else "—"
        mp_clr  = ("#22c55e" if (mp_dist or 0) < -0.5 else
                   "#ef4444" if (mp_dist or 0) > 0.5 else "#475569")

        blocks.append(html.Div([
            # Index name + direction arrow
            html.Div([
                html.Span(label, style={
                    "color": color, "fontSize": "0.52rem",
                    "fontWeight": "800", "letterSpacing": "0.06em",
                }),
                html.Span(f" {dir_sym}", style={
                    "color": dir_clr, "fontSize": "0.78rem", "fontWeight": "900",
                }),
            ], style={"display": "flex", "justifyContent": "space-between",
                      "alignItems": "center", "marginBottom": "3px"}),

            # Score label + confidence
            html.Div([
                html.Span("BULL " if raw_score >= 0 else "BEAR ", style={
                    "color": "#22c55e" if raw_score >= 0 else "#ef4444",
                    "fontSize": "0.44rem", "fontWeight": "700",
                    "letterSpacing": "0.06em", **MONO,
                }),
                html.Span(f"{raw_score:+.1f}", style={
                    "color": "#334155", "fontSize": "0.48rem", **MONO,
                }),
                html.Span(f"  {confidence}", title=(
                    "Conviction = signal agreement × strength, NOT a win probability."),
                    style={
                    "color": "#1e2d40", "fontSize": "0.42rem",
                }),
            ], style={"marginBottom": "4px"}),

            # Mini meter
            html.Div(html.Div(style={
                "width": f"{meter}%", "height": "100%",
                "background": "#22c55e" if meter > 50 else "#ef4444",
                "borderRadius": "2px",
            }), style={
                "height": "3px", "background": "rgba(255,255,255,.05)",
                "borderRadius": "2px", "marginBottom": "5px", "overflow": "hidden",
            }),

            # Signal breakdown rows
            _sig_row("FUT",   fii_val, fii_clr),
            _sig_row("OPT",   pcr_val, pcr_clr),
            _sig_row("FPI",   fpi_val, fpi_clr),
            _sig_row("VIX",   vix_val, vix_clr),
            _sig_row("BRD",   brd_val, brd_clr),
            _sig_row("CARRY", cry_val, cry_clr),
            _sig_row("HMM",   hmm_val, hmm_clr),
            _sig_row("MP",    mp_val,  mp_clr),
        ], style={
            "padding": "7px 8px 5px",
            "marginBottom": "6px",
            "borderRadius": "7px",
            "background": f"linear-gradient(135deg,#0a1020 0%,{color}07 100%)",
            "border": f"1px solid {color}22",
            "borderLeft": f"3px solid {dir_clr}",
        }))

    # Pull the pred_date for the header
    pred_date = bridge.get_panel_data(INDEX_SYMBOLS[0]).get("pred_date")
    date_str  = (pred_date.strftime("%d %b")
                 if pred_date and hasattr(pred_date, "strftime")
                 else "—")

    return html.Div([
        # Section header
        html.Div([
            html.Span("PREDICTION", style={
                "letterSpacing": "0.18em", "color": "#334155", "fontWeight": "700",
            }),
            html.Span(f"  {date_str}", style={
                "color": "#475569", "fontSize": "0.5rem",
            }),
        ], style={
            "fontSize": "0.5rem", "marginBottom": "8px",
            "borderTop": "1px solid #1e3a5f", "paddingTop": "12px",
        }),
        # Signal legend (column headers)
        html.Div([
            html.Span("TAG = signal category verdict (EOD)", style={
                "color": "#334155", "fontSize": "0.42rem", "letterSpacing": "0.02em",
            }),
        ], style={"marginBottom": "6px"}),
        *blocks,
    ])


# ── DCM Index Prediction replica ─────────────────────────────────────────────
# Source: dcm_prediction.py → market_data.duckdb (SAME DB as Daily_Cash_Market)
# Refreshes every 30 min; picks up new data automatically after DCM ingests.

def _rp_badge(text: str, color: str, bg: str) -> html.Span:
    """Direction / confidence badge."""
    return html.Span(text, style={
        "fontSize": "0.58rem", "fontWeight": "700", "color": color,
        "padding": "2px 9px", "borderRadius": "4px",
        "background": bg, "border": f"1px solid {color}55",
        "letterSpacing": "0.05em", "whiteSpace": "nowrap",
    })


def _rp_kv(label: str, value: str, val_clr: str = "#94a3b8") -> html.Div:
    """Metric tile (PCR / Carry / DTE / VIX)."""
    return html.Div([
        html.Span(label, style={
            "color": "#475569", "fontSize": "0.48rem",
            "letterSpacing": "0.08em", "display": "block", "marginBottom": "2px",
        }),
        html.Span(value, style={
            "color": val_clr, "fontSize": "0.64rem", "fontWeight": "700", **MONO,
        }),
    ], style={
        "flex": "1", "minWidth": "0", "padding": "5px 8px",
        "borderRadius": "5px",
        "background": "rgba(255,255,255,0.03)",
        "border": "1px solid rgba(255,255,255,0.055)",
    })


def _rp_regime(label: str, value: str, clr: str) -> html.Span:
    """Statistical regime badge (HURST / HMM / ENTROPY)."""
    return html.Span([
        html.Span(label + "  ", style={"color": "#475569", "fontSize": "0.44rem"}),
        html.Span(value, style={"color": clr, "fontSize": "0.58rem",
                                "fontWeight": "700", **MONO}),
    ], style={
        "display": "inline-block", "padding": "3px 8px", "borderRadius": "4px",
        "background": "#060e1a", "border": "1px solid #1a2535",
        "marginRight": "4px", "marginBottom": "3px",
    })


def _rp_card(sym: str, d: dict) -> dbc.Col:
    """One pure-replica prediction card. `d` is from dcm_prediction.get()."""
    color = COLORS[sym]
    label = LABELS[sym]

    direction  = d.get("direction", "")
    confidence = d.get("confidence", "?")
    raw_score  = d.get("composite_score", 0.0) or 0.0
    acc        = d.get("pred_acc_30d")
    pred_date  = d.get("pred_date")

    spot_close = d.get("spot_close")
    prev_close = d.get("prev_close")
    day_pct    = ((spot_close - prev_close) / prev_close * 100
                  if spot_close and prev_close and prev_close > 0 else 0.0)

    # Meter: composite_score -20..+20 → 0..100
    meter    = max(0, min(100, int((raw_score + 20) / 40 * 100)))
    bb_label = "BULL" if raw_score >= 0 else "BEAR"
    bb_clr   = "#22c55e" if raw_score >= 0 else "#ef4444"
    meter_bg = ("linear-gradient(90deg,#ef4444,#f59e0b)" if meter <= 50 else
                "linear-gradient(90deg,#f59e0b,#22c55e)")

    # Direction badge
    _DIR = {
        "UP":       ("↑ UP",       "#22c55e", "rgba(34,197,94,.14)"),
        "DOWN":     ("↓ DOWN",     "#ef4444", "rgba(239,68,68,.14)"),
        "SIDEWAYS": ("↔ SIDEWAYS", "#f59e0b", "rgba(245,158,11,.14)"),
    }
    dir_txt, dir_clr, dir_bg = _DIR.get(direction, ("—", "#64748b", "rgba(100,116,139,.1)"))

    # Confidence badge
    _CONF = {
        "HIGH":   ("#22c55e", "rgba(34,197,94,.1)"),
        "MEDIUM": ("#f59e0b", "rgba(245,158,11,.1)"),
        "LOW":    ("#64748b", "rgba(100,116,139,.1)"),
    }
    conf_clr, conf_bg = _CONF.get(confidence, ("#64748b", "rgba(100,116,139,.1)"))

    # PCR (from fno_bhavcopy at near expiry, same as DCM)
    pcr     = d.get("eod_pcr") or d.get("feat_pcr") or 0.0
    pcr_str = f"{pcr:.2f}" if pcr else "—"
    pcr_clr = "#22c55e" if pcr > 1.15 else "#ef4444" if 0 < pcr < 0.85 else "#94a3b8"

    # Carry (feat_carry = raw annualised %, from prediction_log)
    carry     = d.get("feat_carry")
    carry_str = f"{carry:.1f}% ann" if carry is not None else "—"
    carry_clr = "#22c55e" if (carry or 0) >= 4 else "#ef4444" if (carry or 0) < 0 else "#94a3b8"

    # DTE = (nearest_expiry − pred_date).days  [at prediction time, not today]
    dte_str = d.get("dte", None)
    dte_str = f"{dte_str}d" if dte_str is not None else "—"

    # VIX: feat_vix (raw level) + feat_vix_5d_chg (absolute pts change, not %)
    vix     = d.get("feat_vix") or d.get("india_vix")
    vix_5d  = d.get("feat_vix_5d_chg") or 0.0
    vix_str = f"{vix:.1f} ({vix_5d:+.1f}pt)" if vix else "—"
    vix_clr = "#ef4444" if (vix or 0) > 20 else "#f59e0b" if (vix or 0) > 16 else "#94a3b8"

    # FII: fao net contracts + COV/ADD delta (DCM exact logic)
    fii_net   = d.get("fii_fut_net")
    fii_delta = d.get("fii_net_change_1d") or 0
    fii_clr   = "#94a3b8"
    fii_main  = "—"
    fii_chg   = None
    if fii_net is not None:
        fii_clr   = "#22c55e" if fii_net >= 0 else "#ef4444"
        emoji     = "🐂" if fii_net > 80_000 else "🐻" if fii_net < -80_000 else "⚪"
        fii_main  = f"{emoji} {fii_net:+,}"
        if fii_delta:
            lbl   = "COV" if fii_delta > 0 else "ADD"
            dclr  = "#22c55e" if fii_delta > 0 else "#ef4444"
            fii_chg = html.Span(f" ({lbl} {abs(fii_delta):,})",
                                 style={"color": dclr, "fontSize": "0.58rem", **MONO})

    # Key levels (DCM exact field names)
    s_val  = d.get("top_put_strike")    # support  (max PE OI within band)
    mp_val = d.get("max_pain_price")    # max pain
    r_val  = d.get("top_call_strike")   # resistance (max CE OI within band)

    # Expected move
    range_lo = d.get("range_low")
    range_hi = d.get("range_high")
    exp_pts  = d.get("expected_move_pts")
    target   = d.get("target_close")
    tgt_pts  = (target - spot_close) if (target and spot_close) else None

    # Breakout scenarios (pre-computed by compute_breakout_scenarios)
    bk = d.get("breakout") or {}

    # Statistical regime
    hurst   = d.get("feat_hurst")
    entropy = d.get("feat_entropy")
    hmm     = d.get("hmm_state") or "?"
    h_lbl   = "TREND" if (hurst or 0) > 0.58 else "M-REV" if (hurst or 0) < 0.42 else "RND"
    e_lbl   = "ORD" if (entropy or 0) < 0.50 else "CHAOS" if (entropy or 0) > 0.72 else "MOD"
    hmm_clr = "#22c55e" if hmm == "Bull" else "#ef4444" if hmm == "Bear" else "#f59e0b"
    e_clr   = "#22c55e" if e_lbl == "ORD" else "#ef4444" if e_lbl == "CHAOS" else "#f59e0b"

    date_str = (pred_date.strftime("%d %b")
                if pred_date and hasattr(pred_date, "strftime")
                else str(pred_date or "—"))
    glow_cls = "sig-bull" if raw_score > 2.5 else "sig-bear" if raw_score < -2.5 else ""

    body = [
        # Index name + direction badge
        html.Div([
            html.Span(label, style={"color": color, "fontSize": "0.68rem",
                                     "fontWeight": "800", "letterSpacing": "0.1em"}),
            _rp_badge(dir_txt, dir_clr, dir_bg),
        ], style={"display": "flex", "justifyContent": "space-between",
                  "alignItems": "center", "marginBottom": "10px"}),

        # Spot close — hero price from prediction_log.spot_close
        html.Div(f"{spot_close:,.0f}" if spot_close else "—", style={
            **MONO, "fontSize": "2.0rem", "fontWeight": "900",
            "color": "#f1f5f9", "letterSpacing": "-0.02em",
            "lineHeight": "1", "marginBottom": "5px",
        }),

        # Day change % + confidence badge
        html.Div([
            html.Span(f"{'▲' if day_pct >= 0 else '▼'} {abs(day_pct):.2f}%",
                      style={"color": "#22c55e" if day_pct >= 0 else "#ef4444",
                             "fontSize": "0.72rem", "fontWeight": "600", **MONO}),
            html.Span(f"  {confidence} CONF", title=(
                "Model confidence bucket from the DCM prediction engine. Its next-day "
                "directional edge is weak (≈coin-flip in backtest) — read as context, "
                "not a win rate."),
                style={
                "color": conf_clr, "fontSize": "0.48rem",
                "padding": "2px 7px", "borderRadius": "3px",
                "background": conf_bg, "border": f"1px solid {conf_clr}44",
                "marginLeft": "8px",
            }),
        ], style={"marginBottom": "10px"}),

        # Bull/Bear score + gradient meter bar
        html.Div([
            html.Div([
                html.Span(f"{bb_label} {meter}/100",
                          style={"color": bb_clr, "fontSize": "0.58rem",
                                 "fontWeight": "700", **MONO}),
                html.Span(f" ({raw_score:+.1f})",
                          style={"color": "#475569", "fontSize": "0.56rem", **MONO}),
            ], style={"marginBottom": "5px"}),
            html.Div(html.Div(className="pred-meter-fill",
                              style={"width": f"{meter}%", "background": meter_bg}),
                     className="pred-meter-track"),
        ]),

        # Divider
        html.Div(style={"height": "1px",
                        "background": f"linear-gradient(90deg,{color}55,transparent)",
                        "marginBottom": "10px"}),

        # Metrics: PCR | Carry | DTE
        html.Div([_rp_kv("PCR", pcr_str, pcr_clr),
                  _rp_kv("Carry", carry_str, carry_clr),
                  _rp_kv("DTE", dte_str, "#64748b")],
                 style={"display": "flex", "gap": "5px", "marginBottom": "5px"}),

        # Metrics: VIX
        html.Div([_rp_kv("VIX", vix_str, vix_clr)],
                 style={"display": "flex", "gap": "5px", "marginBottom": "5px"}),

        # FII full row (net position + ADD/COV delta)
        html.Div([
            html.Span("FII  ", style={"color": "#475569", "fontSize": "0.48rem"}),
            html.Span(fii_main, style={"color": fii_clr, "fontSize": "0.64rem",
                                        "fontWeight": "700", **MONO}),
            fii_chg or html.Span(),
        ], style={"marginBottom": "12px", "padding": "5px 8px", "borderRadius": "5px",
                  "background": "rgba(255,255,255,0.03)",
                  "border": "1px solid rgba(255,255,255,0.055)"}),

        # Key levels: S | MP | R
        html.Div([
            html.Span("S  ", style={"color": "#475569", "fontSize": "0.52rem"}),
            html.Span(f"{s_val:,.0f}" if s_val else "—",
                      style={"color": "#22c55e", "fontWeight": "700",
                             "fontSize": "0.76rem", **MONO}),
            html.Span("  MP  ", style={"color": "#475569", "fontSize": "0.52rem"}),
            html.Span(f"{mp_val:,.0f}" if mp_val else "—",
                      style={"color": "#fbbf24", "fontWeight": "700",
                             "fontSize": "0.76rem", **MONO}),
            html.Span("  R  ", style={"color": "#475569", "fontSize": "0.52rem"}),
            html.Span(f"{r_val:,.0f}" if r_val else "—",
                      style={"color": "#ef4444", "fontWeight": "700",
                             "fontSize": "0.76rem", **MONO}),
        ], style={"marginBottom": "7px"}),

        # Expected move range + target
        html.Div([
            html.Span("Range  ", style={"color": "#475569", "fontSize": "0.5rem"}),
            html.Span(f"{range_lo:,.0f}–{range_hi:,.0f}"
                      if (range_lo and range_hi) else "—",
                      style={"color": "#94a3b8", "fontSize": "0.6rem", **MONO}),
            html.Span(f"  (±{exp_pts:.0f})" if exp_pts else "",
                      style={"color": "#475569", "fontSize": "0.56rem", **MONO}),
            html.Span(
                f"  Tgt {'▲' if (tgt_pts or 0) >= 0 else '▼'}{abs(tgt_pts):.0f}"
                if tgt_pts is not None else "",
                style={"color": "#fbbf24", "fontWeight": "600",
                       "fontSize": "0.62rem", **MONO}),
        ], style={"padding": "5px 8px", "borderRadius": "4px",
                  "background": "rgba(255,255,255,0.02)", "marginBottom": "10px"}),

        # ── Breakout scenarios ─────────────────────────────────────────────────
        # Consistent 2.4σ + DTE gravity for both directions. Three-tier downside.
        *([html.Div([
            html.Div("BREAKOUT SCENARIOS", style={
                "fontSize": "0.44rem", "letterSpacing": "0.14em",
                "color": "#1e3a5f", "marginBottom": "4px", "fontWeight": "700",
            }),
            # Upside row
            html.Div([
                html.Span("▲ IF >", style={"color": "#334155", "fontSize": "0.46rem"}),
                html.Span(f"{bk['u_trigger']:,}",
                          style={"color": "#22c55e", "fontWeight": "700",
                                 "fontSize": "0.58rem", **MONO}),
                html.Span("  →  ", style={"color": "#1e3a5f", "fontSize": "0.44rem"}),
                html.Span(f"{bk['u_trigger']:,}–{bk['u_corrected']:,}",
                          style={"color": "#4ade80", "fontWeight": "700",
                                 "fontSize": "0.6rem", **MONO}),
                html.Span(
                    f"  +{bk['u_ext_pts']} pts"
                    + ("  [SQUEEZE→{:,}]".format(bk['u_stat']) if bk.get("squeeze") else ""),
                    style={"color": "#22c55e", "fontSize": "0.5rem", **MONO}),
            ], style={"marginBottom": "3px"}),
            # Downside row — tier 1 + tier 2 + tier 3
            html.Div([
                html.Span("▼ IF <", style={"color": "#334155", "fontSize": "0.46rem"}),
                html.Span(f"{bk['d_trigger']:,}",
                          style={"color": "#ef4444", "fontWeight": "700",
                                 "fontSize": "0.58rem", **MONO}),
                html.Span("  →  ", style={"color": "#1e3a5f", "fontSize": "0.44rem"}),
                # Tier 1 (put wall) shown if present
                *(
                    [html.Span(f"wall {bk['d_tier1']:,}",
                               style={"color": "#f59e0b", "fontSize": "0.5rem",
                                      "fontWeight": "700", **MONO}),
                     html.Span(" / ", style={"color": "#1e3a5f", "fontSize": "0.44rem"})]
                    if bk.get("d_tier1") else []
                ),
                html.Span(f"{bk['d_tier2']:,}",
                          style={"color": "#f87171", "fontWeight": "700",
                                 "fontSize": "0.6rem", **MONO}),
                html.Span(f"  −{bk['d_ext_pts']} pts",
                          style={"color": "#ef4444", "fontSize": "0.5rem", **MONO}),
                html.Span(f"  [stat {bk['d_tier3']:,}]",
                          style={"color": "#334155", "fontSize": "0.46rem", **MONO}),
            ], style={"marginBottom": "3px"}),
            # Gravity note
            html.Div(
                f"MP {bk['mp']:,} · gravity {int(bk['gravity']*100)}% · {bk['dte']}DTE",
                style={"color": "#1e3a5f", "fontSize": "0.44rem", **MONO},
            ),
        ], style={
            "padding": "5px 8px", "borderRadius": "4px",
            "background": "rgba(0,0,0,0.25)", "border": "1px solid #111d2e",
            "marginBottom": "8px",
        })] if bk else []),

        # Statistical regime badges
        html.Div([
            html.Div("STATISTICAL REGIME", style={
                "fontSize": "0.46rem", "letterSpacing": "0.16em",
                "color": "#334155", "marginBottom": "5px",
            }),
            html.Div([
                _rp_regime("HURST",   f"{hurst:.3f} {h_lbl}" if hurst else "—", "#475569"),
                _rp_regime("HMM",     hmm,                                       hmm_clr),
                _rp_regime("ENTROPY", e_lbl,                                     e_clr),
            ]),
        ]),

        # Footer: accuracy + date
        html.Div([
            html.Span(f"30D acc {acc:.0f}%  " if acc is not None else "",
                      style={"color": ("#22c55e" if (acc or 0) >= 65 else
                                       "#f59e0b" if (acc or 0) >= 50 else "#ef4444"),
                             "fontSize": "0.5rem", **MONO}),
            html.Span(f"·  {date_str}", style={"color": "#1e2d40", "fontSize": "0.5rem"}),
        ], style={"marginTop": "8px", "paddingTop": "6px",
                  "borderTop": "1px solid #111d2e"}),
    ]

    return dbc.Col(
        html.Div(body, className=f"pred-card {glow_cls}", style={
            "padding": "14px 15px", "height": "100%",
            "background": f"linear-gradient(155deg,#0c1826 0%,{color}09 100%)",
            "border": f"1px solid {color}28",
            "borderTop": f"3px solid {dir_clr}",
            "borderRadius": "10px",
        }),
        md=3, xs=12, className="mb-3 px-2",
    )


# ── context-panel callback target ─────────────────────────────────────────────
def _render_context_panel() -> html.Div:
    """
    Pure replica of Daily_Cash_Market's Index Prediction page.
    Reads from market_data.duckdb directly (same DB, same data).
    30-minute cache; auto-picks up new data after DCM ingests.
    """
    if not _DCM_OK:
        return html.Div()

    reader = _get_dcm_reader()

    if not reader.is_available():
        return html.Div([
            html.Span("◉  ", style={"color": "#334155"}),
            html.Span(
                "Index Prediction: market_data.duckdb not reachable — "
                "start Daily_Cash_Market or wait for next retry",
                style={"color": "#475569", "fontSize": "0.62rem", **MONO},
            ),
        ], style={"padding": "14px 18px", "marginBottom": "12px",
                  "background": "#080f1c", "border": "1px solid #1a2535",
                  "borderRadius": "8px"})

    all_data = reader.get_all()

    # EOD date from first available record
    d0       = all_data.get(INDEX_SYMBOLS[0], {})
    pred_dt  = d0.get("pred_date")
    eod_str  = (pred_dt.strftime("%d %b %Y")
                if pred_dt and hasattr(pred_dt, "strftime") else str(pred_dt or "—"))

    return html.Div([
        # Section header — matches DCM page
        html.Div([
            html.Div([
                html.Span("Index Prediction  ", style={
                    "color": "#e2e8f0", "fontSize": "0.9rem", "fontWeight": "700",
                }),
                html.Span("—  Tomorrow's Directional Forecast", style={
                    "color": "#475569", "fontSize": "0.78rem", "fontWeight": "400",
                }),
            ], style={"marginBottom": "5px"}),
            html.Div(
                "24-signal quant engine: OI-Price Matrix · Carry · Max Pain · PCR · "
                "OI-Premium Matrix · Wyckoff Range · Price Mean-Reversion · "
                "FII Institutional · FII Options Delta · FII Flow · FII 5D Cumulative · "
                "FII OI Buildup · FII Position Change · Short Squeeze Setup · India VIX · "
                "Sector Breadth · Cyclical/Defensive Rotation · PE Valuation · "
                "Multi-Expiry PCR · Dual Max Pain · Gamma Wall · Hurst · HMM · Entropy",
                style={"fontSize": "0.48rem", "color": "#334155",
                       "letterSpacing": "0.01em", "lineHeight": "1.7",
                       "marginBottom": "4px"},
            ),
            html.Div(f"EOD: {eod_str}  ·  Source: Daily_Cash_Market (market_data.duckdb)",
                     style={"fontSize": "0.46rem", "color": "#1e3a5f"}),
        ], style={"marginBottom": "16px"}),

        # 4 prediction cards
        dbc.Row(
            [_rp_card(sym, all_data.get(sym, {})) for sym in INDEX_SYMBOLS],
            className="gx-2",
        ),
    ], style={
        "background": "#080f1c",
        "border":     "1px solid #111d2e",
        "borderTop":  "2px solid #1a2d42",
        "borderRadius": "10px",
        "padding": "18px 20px 10px",
        "marginBottom": "14px",
    })


# ── Velocity monitor renderer ────────────────────────────────────────────────
def _render_velocity_panel(sym: str) -> html.Div:
    vel = oi_store.velocity(sym)
    color = COLORS.get(sym, "#00d4ff")

    if not vel.get("has_data"):
        n = vel.get("snap_count", 0)
        body = html.Div([
            html.Div([
                html.Span(className="live-dot live-dot-amber", style={"marginRight": "8px"}),
                html.Span(f"Building session data — {n} / 3 snapshots",
                          style={"color": "#475569", "fontSize": "0.64rem", **MONO}),
            ], style={"display": "flex", "alignItems": "center", "padding": "4px 0"}),
        ])
    else:
        oi  = vel["oi"]
        iv  = vel["iv"]
        wal = vel["walls"]
        pcr = vel["pcr"]

        # OI flow dual bar
        c1 = oi.get("call_1hr") or 0
        p1 = oi.get("put_1hr") or 0
        total_flow = abs(c1) + abs(p1) or 1
        c_pct = abs(c1) / total_flow * 100
        p_pct = abs(p1) / total_flow * 100
        net_lbl = "PUT DOM" if p1 > c1 else "CALL DOM" if c1 > p1 else "BALANCED"
        net_clr = "#22c55e" if p1 > c1 else "#ef4444" if c1 > p1 else "#475569"

        def _ff(v):
            if not v: return "—"
            return f"{'+' if v > 0 else ''}{_fmt_oi(v)}"

        # IV
        regime = iv.get("regime", "stable")
        iv_now = iv.get("now") or 0
        iv_ch  = iv.get("change_1hr") or 0
        iv_clr = "#f59e0b" if regime == "expanding" else "#22c55e" if regime == "contracting" else "#64748b"
        iv_pct = min(iv_now / 30 * 100, 100)

        # PCR
        pcr_n  = pcr.get("now") or 0
        pcr_30 = pcr.get("30m_ago") or pcr_n
        pcr_ch = pcr.get("change_30m") or 0
        trend  = pcr.get("trend", "stable")
        pcr_clr = "#22c55e" if trend == "rising" else "#ef4444" if trend == "falling" else "#64748b"
        pcr_pct = min(pcr_n / 2.0 * 100, 100)

        # Walls
        cws = wal.get("call_shift_1hr") or 0
        pws = wal.get("put_shift_1hr") or 0
        cn  = wal.get("call_now") or 0
        pn  = wal.get("put_now") or 0

        def _warr(shift):
            if not shift: return "→"
            return f"↑ +{shift:.0f}" if shift > 0 else f"↓ {shift:.0f}"

        snaps = vel.get("snap_count", 0)
        body = html.Div([
            # ── OI Flow dual bar ──────────────────────────────────────────
            html.Div([
                html.Div([
                    html.Span("OI FLOW  1HR", style={"color": "#334155", "fontSize": "0.57rem",
                                                      "letterSpacing": "0.08em"}),
                    html.Span(net_lbl, style={"color": net_clr, "fontSize": "0.58rem",
                                              "fontWeight": "700", **MONO}),
                ], style={"display": "flex", "justifyContent": "space-between", "marginBottom": "4px"}),
                html.Div([
                    html.Div(style={"width": f"{c_pct:.1f}%", "height": "100%",
                                    "background": "#ef4444", "borderRadius": "3px 0 0 3px",
                                    "transition": "width .6s ease"}),
                    html.Div(style={"width": f"{p_pct:.1f}%", "height": "100%",
                                    "background": "#22c55e", "borderRadius": "0 3px 3px 0",
                                    "transition": "width .6s ease"}),
                ], style={"display": "flex", "height": "6px", "overflow": "hidden",
                          "background": "rgba(255,255,255,.04)", "borderRadius": "3px",
                          "marginBottom": "4px"}),
                html.Div([
                    html.Span(f"Call  {_ff(c1)}", style={"color": "#f87171", "fontSize": "0.6rem", **MONO}),
                    html.Span(f"Put  {_ff(p1)}", style={"color": "#4ade80", "fontSize": "0.6rem", **MONO}),
                ], style={"display": "flex", "justifyContent": "space-between"}),
            ], style={"marginBottom": "10px"}),

            # ── IV Regime bar ─────────────────────────────────────────────
            html.Div([
                html.Div([
                    html.Span("IV REGIME", style={"color": "#334155", "fontSize": "0.57rem",
                                                   "letterSpacing": "0.08em"}),
                    html.Span(regime.upper(), style={"color": iv_clr, "fontSize": "0.58rem",
                                                     "fontWeight": "700", **MONO}),
                ], style={"display": "flex", "justifyContent": "space-between", "marginBottom": "4px"}),
                html.Div(html.Div(style={"width": f"{iv_pct:.1f}%", "height": "100%",
                                         "background": iv_clr, "borderRadius": "3px",
                                         "transition": "width .6s ease"}),
                         className="vbar-track", style={"marginBottom": "4px"}),
                html.Span(f"{iv_now:.1f}%  ({'+' if iv_ch >= 0 else ''}{iv_ch:.1f}% / 1hr)",
                          style={"color": "#475569", "fontSize": "0.6rem", **MONO}),
            ], style={"marginBottom": "10px"}),

            # ── PCR Trend bar ─────────────────────────────────────────────
            html.Div([
                html.Div([
                    html.Span("PCR TREND  30M", style={"color": "#334155", "fontSize": "0.57rem",
                                                        "letterSpacing": "0.08em"}),
                    html.Span(trend.upper(), style={"color": pcr_clr, "fontSize": "0.58rem",
                                                    "fontWeight": "700", **MONO}),
                ], style={"display": "flex", "justifyContent": "space-between", "marginBottom": "4px"}),
                html.Div(html.Div(style={"width": f"{pcr_pct:.1f}%", "height": "100%",
                                         "background": pcr_clr, "borderRadius": "3px",
                                         "transition": "width .6s ease"}),
                         className="vbar-track", style={"marginBottom": "4px"}),
                html.Span(f"{pcr_30:.2f}  →  {pcr_n:.2f}  ({'+' if pcr_ch >= 0 else ''}{pcr_ch:.2f})",
                          style={"color": "#475569", "fontSize": "0.6rem", **MONO}),
            ], style={"marginBottom": "10px"}),

            # ── Wall shifts ───────────────────────────────────────────────
            html.Div([
                html.Span("WALLS  1HR", style={"color": "#334155", "fontSize": "0.57rem",
                                                "letterSpacing": "0.08em", "display": "block",
                                                "marginBottom": "4px"}),
                html.Div([
                    html.Span(f"Call  {cn:,.0f}  {_warr(cws)}",
                              style={"color": "#4ade80" if cws >= 0 else "#f87171",
                                     "fontSize": "0.62rem", **MONO, "marginRight": "16px"}),
                    html.Span(f"Put  {pn:,.0f}  {_warr(pws)}",
                              style={"color": "#4ade80" if pws >= 0 else "#f87171",
                                     "fontSize": "0.62rem", **MONO}),
                ]),
            ], style={"marginBottom": "6px"}),

            html.Div(f"{snaps} session snapshots", style={
                "color": "#1e2d40", "fontSize": "0.53rem", "marginTop": "2px", **MONO,
            }),
        ])

    return dbc.Card(dbc.CardBody([
        html.Div([
            html.Span("INTRADAY  VELOCITY", style={"fontSize": "0.56rem", "letterSpacing": "0.2em",
                                                    "color": "#1e3a5f", "fontWeight": "700"}),
            html.Span("  MONITOR", style={"fontSize": "0.52rem", "letterSpacing": "0.12em",
                                           "color": "#0f1e30"}),
        ], style={"marginBottom": "10px"}),
        body,
    ]), style={
        "background": "#080f1c",
        "border":     f"1px solid {color}22",
        "borderLeft": f"3px solid {color}",
        "borderRadius": "8px",
        "marginBottom": "10px",
    })


# ── Signal panel renderer ─────────────────────────────────────────────────────
def _render_signal_panel(results: dict, updated: str) -> html.Div:
    """
    4-index × 4-timeframe signal matrix + per-index verdict cards.
    Always shows all 4 index cards — degrades to 'building' state when data is unavailable.
    Adds session phase context and safeguards all key accesses.
    """
    phase_name, _phase_mult, phase_caution = session_phase()
    TF_KEYS  = [("5min","5M"), ("15min","15M"), ("60min","1H"), ("daily","D")]
    IDX_SHORT = {
        "NSE:NIFTY50-INDEX":    "N50",
        "NSE:NIFTYBANK-INDEX":  "BNK",
        "NSE:FINNIFTY-INDEX":   "FIN",
        "NSE:MIDCPNIFTY-INDEX": "MID",
    }

    def _s_cls(score, sig=""):
        if sig == "INSUFFICIENT DATA": return "tf-neut"
        if score > 0.5:  return "tf-bull"
        if score < -0.5: return "tf-bear"
        return "tf-neut"

    def _s_lbl(score, sig=""):
        if sig in ("INSUFFICIENT DATA", ""): return "—"
        if score > 0.5:  return "BUY"
        if score < -0.5: return "SELL"
        return "NEU"

    # ── 4 × 4 Signal Matrix ───────────────────────────────────────────────────
    TH_S = {"padding":"5px 12px","fontSize":"0.56rem","fontWeight":"800",
             "letterSpacing":"0.08em","textAlign":"center",
             "borderBottom":"2px solid #111d2e","background":"#060c14"}
    matrix_rows = [html.Tr([
        html.Th("", style={**TH_S,"textAlign":"left","color":"#334155","minWidth":"42px"}),
        *[html.Th(IDX_SHORT[s], style={**TH_S,"color":COLORS[s]}) for s in INDEX_SYMBOLS],
    ])]
    for tf_key, tf_lbl in TF_KEYS:
        cells = [html.Td(tf_lbl, style={
            "padding":"5px 8px","fontSize":"0.56rem","color":"#334155",
            "fontWeight":"700","letterSpacing":"0.08em",
            "borderRight":"1px solid #0d1a2a","background":"#060c14","whiteSpace":"nowrap",
        })]
        for sym in INDEX_SYMBOLS:
            t   = results.get(sym, {}).get("timeframes", {}).get(tf_key, {})
            s   = t.get("score", 0)
            sig = t.get("signal", "")
            lbl = _s_lbl(s, sig)
            cls = _s_cls(s, sig)
            cells.append(html.Td(
                html.Span(lbl, className=f"tf-pill {cls}",
                          style={"fontSize":"0.52rem","padding":"2px 7px","margin":"0"}),
                style={"textAlign":"center","padding":"4px 6px",
                       "borderBottom":"1px solid #0d1a2a","background":"rgba(0,0,0,.1)"},
            ))
        matrix_rows.append(html.Tr(cells))

    signal_matrix = dbc.Card(dbc.CardBody([
        html.Div([
            html.Span("SIGNAL MATRIX", style={"fontSize":"0.55rem","letterSpacing":"0.2em",
                                               "color":"#1e3a5f","fontWeight":"700"}),
            html.Span("  4 INDICES × 4 TIMEFRAMES",
                      style={"fontSize":"0.5rem","color":"#0f1e30"}),
        ], style={"marginBottom":"8px"}),
        html.Div(
            html.Table(matrix_rows, style={"width":"100%","borderCollapse":"collapse"}),
            style={"overflowX":"auto"},
        ),
    ]), style={"background":"#080f1c","border":"1px solid #111d2e",
               "borderRadius":"8px","marginBottom":"12px"})

    # ── 4 Index verdict cards (always all 4) ──────────────────────────────────
    def _sbar(ws):
        pct = max(0, min(abs(ws) / 4.0 * 100, 100))
        clr = ("#22c55e" if ws > 1.5 else "#4ade80" if ws > 0.3
               else "#ef4444" if ws < -1.5 else "#f87171" if ws < -0.3 else "#1e2d40")
        return html.Div(
            html.Div(className="sbar-fill", style={"width":f"{pct:.0f}%","background":clr}),
            className="sbar-track",
        )

    sig_cards, recs = [], []

    for sym in INDEX_SYMBOLS:
        r     = results.get(sym, {})
        color = COLORS[sym]
        label = LABELS[sym]
        tfs   = r.get("timeframes", {})

        # Always show a card — loading/error state when no real data
        if not tfs or "error" in r or all(t.get("signal","") == "INSUFFICIENT DATA"
                                          for t in tfs.values()):
            err = r.get("error", "")
            sig_cards.append(dbc.Col(
                html.Div([
                    html.Div([
                        html.Span(className="live-dot live-dot-amber",
                                  style={"marginRight":"5px"}),
                        html.Span(label, style={"color":color,"fontSize":"0.58rem",
                                                "fontWeight":"800","letterSpacing":"0.14em"}),
                    ], style={"display":"flex","alignItems":"center","marginBottom":"10px"}),
                    html.Div("BUILDING...", style={"color":"#334155","fontWeight":"700",
                                                    "fontSize":"0.88rem",**MONO}),
                    html.Div(err[:48] if err else "Collecting candle data...",
                             style={"color":"#1e2d40","fontSize":"0.55rem",
                                    **MONO,"marginTop":"6px"}),
                    html.Div(html.Div(className="sbar-fill",
                                      style={"width":"15%","background":"#1e2d40"}),
                             className="sbar-track"),
                ], className="sig-card", style={
                    "padding":"14px 16px","height":"100%","background":"#0c1522",
                    "borderRadius":"12px","border":f"1px solid {color}18",
                    "borderTop":f"2px solid {color}40",
                }),
                md=3, xs=6, className="mb-2 px-1",
            ))
            continue

        ov, _ov_clr = r.get("overall", ("NEUTRAL", "#94a3b8"))
        ws           = r.get("weighted_score", 0)
        bull_c       = sum(1 for t in tfs.values() if t.get("score", 0) > 0.5)
        bear_c       = sum(1 for t in tfs.values() if t.get("score", 0) < -0.5)
        glow         = ("sig-card sig-bull" if ws > 0.8 else
                        "sig-card sig-bear" if ws < -0.8 else "sig-card")
        score_cls    = "score-bull" if ws > 0.3 else "score-bear" if ws < -0.3 else "score-neut"
        vclass       = ("verdict-bull" if ws > 0.3 else
                        "verdict-bear" if ws < -0.3 else "verdict-neut")

        sig_cards.append(dbc.Col(
            html.Div([
                html.Div([
                    html.Div([
                        html.Span(className="live-dot", style={"marginRight":"5px"}),
                        html.Span(label, style={"color":color,"fontSize":"0.58rem",
                                                "fontWeight":"800","letterSpacing":"0.14em"}),
                    ], style={"display":"flex","alignItems":"center"}),
                    html.Span(f"{ws:+.1f}", className=f"score-badge {score_cls}"),
                ], style={"display":"flex","justifyContent":"space-between",
                          "alignItems":"center","marginBottom":"10px"}),

                html.Div(ov, className=vclass, style={
                    "fontWeight":"900","fontSize":"1.05rem",
                    **MONO,"letterSpacing":"0.04em","lineHeight":"1",
                }),

                _sbar(ws),

                html.Div([
                    html.Span(f"{bull_c}× bull",
                              style={"color":"#22c55e" if bull_c >= 3 else "#1e2d40",
                                     "fontSize":"0.54rem",**MONO,"marginRight":"8px"}),
                    html.Span(f"{bear_c}× bear",
                              style={"color":"#ef4444" if bear_c >= 3 else "#1e2d40",
                                     "fontSize":"0.54rem",**MONO}),
                ], style={"marginBottom":"4px"}),

                html.Div([
                    html.Span(
                        f"{lbl} {_s_lbl(tfs.get(k,{}).get('score',0), tfs.get(k,{}).get('signal',''))}",
                        className=f"tf-pill {_s_cls(tfs.get(k,{}).get('score',0), tfs.get(k,{}).get('signal',''))}",
                    )
                    for k, lbl in TF_KEYS
                ], style={"marginTop":"8px","lineHeight":"2.2"}),
            ], className=glow, style={
                "padding":"14px 16px","height":"100%",
                "background":f"linear-gradient(155deg, #0c1522 0%, {color}0d 100%)",
                "borderRadius":"12px","border":f"1px solid {color}20",
                "borderTop":f"2px solid {color}",
            }),
            md=3, xs=6, className="mb-2 px-1",
        ))
        if abs(ws) >= 0.8:
            recs.append((abs(ws), sym, r))

    # ── Opportunity rec cards ─────────────────────────────────────────────────
    recs.sort(key=lambda x: x[0], reverse=True)
    rec_cards = []
    for _, sym, r in recs[:3]:
        ws          = r["weighted_score"]
        ov, ov_clr  = r.get("overall", ("—", "#475569"))
        color       = COLORS[sym]
        label       = r["label"]
        tfs         = r["timeframes"]
        bull_tfs    = [k for k, t in tfs.items() if t.get("score", 0) > 0.5]
        bear_tfs    = [k for k, t in tfs.items() if t.get("score", 0) < -0.5]
        active_tfs  = bull_tfs if ws > 0 else bear_tfs
        # Guard: use .get("label", k) to avoid KeyError on incomplete TF dicts
        tf_lbls     = [tfs[k].get("label", k) for k in active_tfs if k in tfs]
        direction   = "BUY CALL (CE)" if ws > 0 else "BUY PUT (PE)"
        dir_clr     = "#4ade80" if ws > 0 else "#f87171"
        trade_type  = "INTRADAY" if len(active_tfs) <= 2 else "BTST / POSITIONAL"
        conf        = min(abs(ws) / 5 * 100, 95)
        # Guard: use max on values directly, not items, to avoid label KeyError
        best_tf_val = max(tfs.values(), key=lambda x: abs(x.get("score", 0)), default={})
        top_rsns    = best_tf_val.get("reasons", [])[:2]

        rec_cards.append(dbc.Col(
            html.Div([
                html.Div([
                    html.Span(label, style={"color":color,"fontWeight":"800",
                                            "fontSize":"0.68rem",**MONO}),
                    html.Span(f"  {conf:.0f}% conv",
                              title=("Conviction = signal agreement × strength, NOT a win probability."),
                              style={"color":"#475569","fontSize":"0.55rem",**MONO}),
                ], style={"marginBottom":"5px"}),
                html.Div(direction, style={
                    "color":dir_clr,"fontWeight":"900","fontSize":"0.9rem",
                    **MONO,"letterSpacing":"0.04em","marginBottom":"3px",
                }),
                html.Div(ov, style={"color":ov_clr,"fontSize":"0.62rem",
                                     "fontWeight":"600",**MONO,"marginBottom":"6px"}),
                html.Div(
                    html.Div(className="conf-fill",
                             style={"width":f"{conf:.0f}%",
                                    "background":f"linear-gradient(90deg,{color},{dir_clr})"}),
                    className="conf-track", style={"marginBottom":"6px"},
                ),
                html.Div([
                    html.Span(trade_type, style={"color":"#334155","fontSize":"0.54rem"}),
                    html.Span(f"  ·  {' + '.join(tf_lbls)}" if tf_lbls else "",
                              style={"color":"#1e3a5f","fontSize":"0.54rem"}),
                ], style={"marginBottom":"5px"}),
                *[html.Div([
                    html.Span("▸ ", style={"color":color,"fontSize":"0.55rem"}),
                    html.Span(txt, style={"color":"#334155","fontSize":"0.56rem",**MONO}),
                ], style={"overflow":"hidden","textOverflow":"ellipsis","whiteSpace":"nowrap"})
                  for _, txt in top_rsns],
                html.Div(f"← click {label} in sidebar for exact strike",
                         style={"color":"#1e2d40","fontSize":"0.5rem",
                                "borderTop":f"1px solid {color}18",
                                "marginTop":"7px","paddingTop":"4px"}),
            ], className="rec-card", style={
                "padding":"13px 15px","height":"100%",
                "background":f"linear-gradient(140deg, #0a1020 0%, {color}0a 100%)",
                "borderRadius":"10px","border":f"1px solid {color}22",
                "borderLeft":f"3px solid {color}",
                "boxShadow":"0 4px 20px rgba(0,0,0,.4)",
            }),
            md=4, className="mb-2 px-1",
        ))

    # ── Session phase chip ────────────────────────────────────────────────────
    PHASE_CLR = {"MORNING":"#22c55e","AFTERNOON":"#22c55e","OPENING":"#f59e0b",
                 "LUNCH":"#f59e0b","PRE-CLOSE":"#f59e0b","CLOSE":"#ef4444"}
    phase_clr      = PHASE_CLR.get(phase_name, "#64748b")
    phase_dot_cls  = ("live-dot" if phase_name in ("MORNING","AFTERNOON")
                      else "live-dot live-dot-amber" if phase_name in ("OPENING","LUNCH","PRE-CLOSE")
                      else "live-dot live-dot-dead")

    return html.Div([
        # Header
        dbc.Row([
            dbc.Col(html.Div([
                html.Span("TRADE SIGNALS", style={"letterSpacing":"0.22em","color":"#1e3a5f",
                                                    "fontWeight":"700","fontSize":"0.56rem"}),
                html.Span("  ·  4-INDEX  MULTI-TIMEFRAME",
                          style={"color":"#0f1e30","fontSize":"0.52rem"}),
            ])),
            dbc.Col(html.Div([
                html.Span([
                    html.Span(className=phase_dot_cls, style={"marginRight":"5px"}),
                    html.Span(phase_name, style={"color":phase_clr,"fontWeight":"700",
                                                  "fontSize":"0.58rem",**MONO}),
                ], style={"marginRight":"14px"}),
                html.Span(className="live-dot", style={"marginRight":"6px"}),
                html.Span(f"Updated {updated}",
                          style={"color":"#1e2d40","fontSize":"0.54rem",**MONO}),
            ], style={"textAlign":"right","display":"flex",
                      "alignItems":"center","justifyContent":"flex-end"})),
        ], className="mb-2 align-items-center"),

        # Phase caution message
        html.Div([
            html.Span(phase_caution, style={"color":"#f59e0b","fontSize":"0.58rem",**MONO}),
        ], style={"marginBottom":"10px","paddingLeft":"2px"}) if phase_caution else html.Div(),

        # 4×4 Signal matrix (primary data view)
        signal_matrix,

        # 4 index verdict cards (always all 4)
        dbc.Row(sig_cards, className="gx-2 mb-2"),

        html.Div("Algorithmic signals only — not financial advice — always use a stop-loss.",
                 style={"fontSize":"0.5rem","color":"#1e2d40",
                        "textAlign":"center","marginBottom":"12px"}),

        # Opportunity cards (only when strong signals exist)
        dbc.Row(rec_cards, className="gx-2") if rec_cards else html.Div(),
    ], style={"animation":"slide-up .35s ease-out"})


# ── 9-layer alignment strip (used inside trade ticket) ───────────────────────
def _render_layer_alignment(rec: dict) -> html.Div:
    """
    Renders a compact 9-chip row showing each layer's score + direction.
    Returns empty Div if rec has no layer_summary (neutral case).
    """
    ls = rec.get("layer_summary")
    if not ls:
        return html.Div()

    LAYERS = [
        ("tech",     "TECH"),
        ("oi",       "OI"),
        ("velocity", "VEL"),
        ("inst",     "INST"),
        ("futures",  "FUT"),
        ("iv",       "IV"),
        ("pcr",      "PCR"),
        ("mp",       "MP"),
        ("context",  "CTX"),
    ]

    chips = []
    for key, short in LAYERS:
        info  = ls.get(key, {})
        score = info.get("score", 0) or 0
        lbl   = info.get("label", short)
        clr   = "#22c55e" if score > 0.15 else "#ef4444" if score < -0.15 else "#475569"
        cls   = "tf-bull" if score > 0.15 else "tf-bear" if score < -0.15 else "tf-neut"
        chips.append(html.Div([
            html.Div(short, style={"fontSize":"0.5rem","color":"#334155",
                                    "letterSpacing":"0.06em","marginBottom":"2px",
                                    "textAlign":"center"}),
            html.Span(f"{score:+.1f}" if score != 0 else "0.0",
                      className=f"tf-pill {cls}",
                      style={"fontSize":"0.52rem","padding":"1px 6px","display":"block",
                             "textAlign":"center"},
                      title=lbl),
        ], style={"flex":"1","minWidth":"38px","textAlign":"center"}))

    agree = ls.get("agree", "")
    phase = ls.get("phase", "")
    n_aligned = int(agree.split("/")[0]) if "/" in agree else 0
    agree_pct = n_aligned / 9 * 100

    return html.Div([
        html.Div([
            html.Span("9-LAYER ALIGNMENT", style={"color":"#1e3a5f","fontSize":"0.53rem",
                                                    "letterSpacing":"0.14em","fontWeight":"700"}),
            html.Span(f"  {agree}", style={"color":"#475569","fontSize":"0.55rem",**MONO}),
            html.Span(f"  {phase}", style={"color":"#334155","fontSize":"0.52rem",**MONO}),
            # Alignment progress bar
            html.Div(html.Div(className="conf-fill",
                              style={"width":f"{agree_pct:.0f}%","background":"#00d4ff"}),
                     className="conf-track",
                     style={"display":"inline-block","width":"80px",
                            "verticalAlign":"middle","marginLeft":"10px"}),
        ], style={"marginBottom":"6px"}),
        html.Div(chips, style={"display":"flex","gap":"4px","flexWrap":"wrap"}),
    ])


# ── Trade recommendation card renderer ────────────────────────────────────────
def _render_trade_rec(rec: dict, sym: str) -> html.Div:
    if not rec:
        return html.Div()

    color = COLORS.get(sym, "#00d4ff")

    if rec.get("neutral"):
        return dbc.Card(dbc.CardBody([
            html.Div([
                html.Span(rec["tf_label"], style={"color":"#334155","fontSize":"0.6rem","marginRight":"8px"}),
                html.Span(rec["signal"], style={"color":rec["color"],"fontWeight":"700",
                                                 **MONO,"fontSize":"0.7rem"}),
            ], style={"marginBottom":"6px"}),
            html.Div("No trade setup — signal too weak or conflicting timeframes.",
                     style={"color":"#475569","fontSize":"0.65rem"}),
            html.Div(html.Div(style={"width":"18%","height":"100%","background":"#334155",
                                      "borderRadius":"3px"}),
                     className="conf-track", style={"marginTop":"10px"}),
        ]), style={"background":"#080f1c","border":f"1px solid {color}22",
                   "borderLeft":f"3px solid {color}","borderRadius":"8px","marginBottom":"10px"})

    dir_clr  = rec["dir_clr"]
    conf     = rec.get("confidence", 0)
    warn_clr = "#f59e0b" if "CAUTION" in rec.get("warning","") else "#22c55e"
    conf_clr = "#22c55e" if conf >= 70 else "#f59e0b" if conf >= 45 else "#ef4444"

    def _metric(label, value, val_clr="#94a3b8", big=False):
        return html.Div([
            html.Div(label, style={"color":"#334155","fontSize":"0.53rem",
                                    "letterSpacing":"0.08em","marginBottom":"1px"}),
            html.Div(value, style={"color":val_clr,
                                    "fontWeight":"700" if big else "600",
                                    "fontSize":"0.78rem" if big else "0.67rem",**MONO}),
        ], style={"marginBottom":"8px"})

    # ── Price zone visual bar ─────────────────────────────────────────────────
    sl = rec.get("sl", 0); entry_lo = rec.get("entry_lo", 0)
    entry_hi = rec.get("entry_hi", 0); t1 = rec.get("t1", 0)
    t2 = rec.get("t2", 0); ltp = rec.get("ltp", 0) or 1
    if sl and t2:
        z_min = min(sl, ltp) * 0.97
        z_max = max(t2, ltp) * 1.03
        z_rng = z_max - z_min or 1
        def _pp(v): return max(0.0, min((v - z_min) / z_rng * 100, 100))
        sl_p  = _pp(sl);   lo_p = _pp(entry_lo); hi_p = _pp(entry_hi)
        t1_p  = _pp(t1);   t2_p = _pp(t2)
        ez_w  = max(hi_p - lo_p, 1)
        price_zone = html.Div([
            html.Div("PRICE ZONE", style={"color":"#334155","fontSize":"0.54rem",
                                           "letterSpacing":"0.12em","marginBottom":"6px",
                                           "fontWeight":"700"}),
            html.Div(style={
                "position":"relative","height":"8px","borderRadius":"4px",
                "background":"linear-gradient(90deg,rgba(239,68,68,.2) 0%,rgba(251,191,36,.3) 35%,rgba(74,222,128,.35) 70%,rgba(34,197,94,.5) 100%)",
                "marginBottom":"6px",
            }, children=[
                html.Div(style={"position":"absolute","left":f"{sl_p:.1f}%",
                                 "top":"-3px","width":"2px","height":"14px",
                                 "background":"#ef4444","borderRadius":"1px"}),
                html.Div(style={"position":"absolute","left":f"{lo_p:.1f}%",
                                 "width":f"{ez_w:.1f}%","height":"8px",
                                 "background":"rgba(251,191,36,.65)","borderRadius":"2px"}),
                html.Div(style={"position":"absolute","left":f"{t1_p:.1f}%",
                                 "top":"-3px","width":"2px","height":"14px",
                                 "background":"#4ade80","borderRadius":"1px"}),
                html.Div(style={"position":"absolute","left":f"{t2_p:.1f}%",
                                 "top":"-3px","width":"2px","height":"14px",
                                 "background":"#22c55e","borderRadius":"1px"}),
            ]),
            html.Div([
                html.Span(f"SL {sl}", style={"color":"#f87171","fontSize":"0.55rem",**MONO}),
                html.Span(f"Entry {entry_lo}–{entry_hi}",
                          style={"color":"#fbbf24","fontSize":"0.55rem",**MONO,"margin":"0 10px"}),
                html.Span(f"T1 {t1}", style={"color":"#4ade80","fontSize":"0.55rem",**MONO,"marginRight":"10px"}),
                html.Span(f"T2 {t2}", style={"color":"#22c55e","fontSize":"0.55rem",**MONO}),
            ]),
        ], style={"marginBottom":"12px"})
    else:
        price_zone = html.Div()

    return dbc.Card([
        # ── Colored header band ───────────────────────────────────────────────
        html.Div([
            dbc.Row([
                dbc.Col([
                    html.Div(rec["tf_label"],
                             style={"color":"rgba(255,255,255,.4)","fontSize":"0.54rem",
                                    "letterSpacing":"0.12em"}),
                    html.Div([
                        html.Span(rec["signal"] + " ",
                                  style={"color":"white","fontWeight":"800","fontSize":"0.92rem",**MONO}),
                        html.Span(f"({rec['conviction']})",
                                  style={"color":"rgba(255,255,255,.45)","fontSize":"0.62rem"}),
                    ], style={"marginTop":"2px"}),
                ], md=5),
                dbc.Col([
                    html.Div("RECOMMENDATION",
                             style={"color":"rgba(255,255,255,.4)","fontSize":"0.52rem",
                                    "letterSpacing":"0.14em"}),
                    html.Div(f"BUY {rec['dir_label']}", style={
                        "color":"white","fontWeight":"900","fontSize":"0.95rem",
                        **MONO,"letterSpacing":"0.04em","marginTop":"2px",
                    }),
                ], md=4),
                dbc.Col([
                    html.Div("CONFIDENCE",
                             style={"color":"rgba(255,255,255,.4)","fontSize":"0.52rem",
                                    "letterSpacing":"0.1em","textAlign":"right"}),
                    html.Div(f"{conf:.0f}%", style={
                        "color":conf_clr,"fontWeight":"900","fontSize":"1.25rem",
                        **MONO,"textAlign":"right","marginTop":"2px",
                    }),
                    html.Div(html.Div(className="conf-fill",
                                      style={"width":f"{conf:.0f}%","background":conf_clr}),
                             className="conf-track", style={"marginTop":"5px"}),
                ], md=3),
            ], className="align-items-center"),
        ], style={
            "background": f"linear-gradient(135deg, {dir_clr}20 0%, {dir_clr}08 100%)",
            "borderBottom": f"1px solid {dir_clr}28",
            "padding": "14px 18px",
        }),

        # ── 9-Layer Alignment Panel ───────────────────────────────────────────
        html.Div(_render_layer_alignment(rec), style={
            "borderBottom":"1px solid #0d1a2a",
            "padding":"10px 18px",
            "background":"rgba(0,0,0,.25)",
        }),

        # ── Body ─────────────────────────────────────────────────────────────
        dbc.CardBody([
            price_zone,
            dbc.Row([
                # Trade details
                dbc.Col([
                    html.Div("TRADE DETAILS", style={"color":"#1e3a5f","fontSize":"0.54rem",
                             "letterSpacing":"0.14em","marginBottom":"8px","fontWeight":"700"}),
                    _metric("OPTION", f"{rec['strike']:,.0f} {rec['direction']}  ·  {rec['exp_date']}",
                            dir_clr, big=True),
                    _metric("ENTRY ZONE",  f"₹ {rec['entry_lo']} — {rec['entry_hi']}", "#fbbf24"),
                    _metric("STOP LOSS",   f"₹ {rec['sl']}  (Index ≈ {rec['spot_sl']:,.0f})", "#ef4444"),
                    _metric("TARGET 1",
                            f"₹ {rec['t1']}  +{int((rec['t1']/ltp-1)*100)}%  →  ₹{rec['profit_t1']:,.0f}/lot",
                            "#4ade80"),
                    _metric("TARGET 2",
                            f"₹ {rec['t2']}  +{int((rec['t2']/ltp-1)*100)}%  →  ₹{rec['profit_t2']:,.0f}/lot",
                            "#22c55e"),
                    _metric("RISK : REWARD",  f"1 : {rec['rr']}", "#fbbf24"),
                    _metric("MAX LOSS / LOT", f"₹ {rec['loss_lot']:,.0f}  ({rec['lot_size']} shares)", "#f87171"),
                    _metric("TRADE TYPE",     rec.get("trade_type","—"), "#475569"),
                ], md=5),

                # Greeks + OI
                dbc.Col([
                    html.Div("OPTION DETAILS", style={"color":"#1e3a5f","fontSize":"0.54rem",
                             "letterSpacing":"0.14em","marginBottom":"8px","fontWeight":"700"}),
                    _metric("LTP",    f"₹ {rec['ltp']:.2f}", "#e2e8f0", big=True),
                    _metric("IV",     f"{rec['iv']:.1f}%"),
                    _metric("DELTA",  f"{rec['delta']:.3f}"),
                    _metric("THETA",  f"{rec['theta']:.2f}  / day"),
                    _metric("VEGA",   f"{rec['vega']:.2f}"),
                    _metric("OI",     _fmt_oi(rec["oi"])),
                    _metric("VOLUME", _fmt_oi(rec["volume"])),
                    html.Div(rec["iv_context"],
                             style={"color":"#334155","fontSize":"0.6rem","marginTop":"4px"}),
                ], md=3),

                # Why this trade
                dbc.Col([
                    html.Div("WHY THIS TRADE?", style={"color":"#1e3a5f","fontSize":"0.54rem",
                             "letterSpacing":"0.14em","marginBottom":"8px","fontWeight":"700"}),
                    *[html.Div([
                        html.Span(
                            ("✓ " if (b=="bull" and rec.get("direction")=="CE")
                                  or (b=="bear" and rec.get("direction")=="PE")
                             else "· " if b=="neut"
                             else "✗ "),
                            style={"color": (
                                "#4ade80" if (b=="bull" and rec.get("direction")=="CE")
                                          or (b=="bear" and rec.get("direction")=="PE")
                                else "#f87171" if b != "neut"
                                else "#334155"),
                                   "fontWeight":"700","marginRight":"4px"}),
                        html.Span(t, style={"color":"#475569","fontSize":"0.62rem"}),
                    ], style={"marginBottom":"4px","display":"flex","alignItems":"flex-start"})
                      for b, t in rec["tech_reasons"][:4]],
                    *[html.Div([
                        html.Span("▸ ", style={"color":"#334155","marginRight":"3px"}),
                        html.Span(t, style={"color":"#334155","fontSize":"0.6rem"}),
                    ], style={"marginBottom":"3px"})
                      for b, t in rec["opt_signals"]],
                    html.Div(rec.get("fut_context",""),
                             style={"color":"#334155","fontSize":"0.6rem","marginTop":"4px"}),
                ], md=4),
            ]),

            html.Div([
                html.Span("⚠  ", style={"fontSize":"0.72rem","marginRight":"4px"}),
                html.Span(rec["warning"], style={"fontSize":"0.62rem",**MONO}),
            ], style={
                "color": warn_clr, "background": "#060d18",
                "padding": "8px 12px", "borderRadius": "5px",
                "marginTop": "8px", "border": f"1px solid {warn_clr}22",
            }) if rec.get("warning") else html.Div(),
        ]),
    ], className="trade-ticket", style={
        "background": "#080f1c",
        "border":     f"1px solid {color}33",
        "borderRadius": "8px",
        "overflow": "hidden",
        "marginBottom": "10px",
    })


# ── Callback 1: sidebar nav clicks → update selected symbol ────────────────────
_URL_SHORT = {"NSE:NIFTY50-INDEX": "nifty50", "NSE:NIFTYBANK-INDEX": "banknifty",
              "NSE:FINNIFTY-INDEX": "finnifty", "NSE:MIDCPNIFTY-INDEX": "midcpnifty"}
_SHORT_TO_SYM = {v: k for k, v in _URL_SHORT.items()}
_PATH_TO_SEL = {"/live-oi": "LIVEOI", "/charts": "CHARTS",
                "/alerts": "ALERTS", "/trades": "TRADES", "/tradeboard": "TRADEBOARD"}
_SEL_TO_PATH = {v: k for k, v in _PATH_TO_SEL.items()}


@app.callback(
    Output("url", "pathname"),
    [Input(f"nav-{_slug(s)}", "n_clicks") for s in INDEX_SYMBOLS],
    Input("nav-liveoi",    "n_clicks"),
    Input("nav-charts",    "n_clicks"),
    Input("nav-alerts",    "n_clicks"),
    Input("nav-tradeboard", "n_clicks"),
    State("url", "pathname"),
    prevent_initial_call=True,
)
def on_nav_click(*args):
    """Nav click → set the URL (the section's source of truth). Clicking the section
    you're already on toggles back to "/". The URL then drives sel-sym via _route, so
    there is NO sel-sym→url edge and thus no circular dependency."""
    from dash import callback_context as ctx
    from dash.exceptions import PreventUpdate
    *_, _liveoi_clicks, _charts_clicks, _alerts_clicks, _tb_clicks, cur = args
    if not ctx.triggered:
        raise PreventUpdate
    tid = ctx.triggered[0]["prop_id"].split(".")[0]
    cur = (cur or "/").rstrip("/") or "/"
    def toggle(path: str) -> str:
        return "/" if cur == path else path
    if tid == "nav-liveoi":
        return toggle("/live-oi")
    if tid == "nav-charts":
        return toggle("/charts")
    if tid == "nav-alerts":
        return toggle("/alerts")
    if tid == "nav-tradeboard":
        return toggle("/tradeboard")
    for sym in INDEX_SYMBOLS:
        if tid == f"nav-{_slug(sym)}":
            return toggle(f"/chain/{_URL_SHORT.get(sym, 'index')}")
    raise PreventUpdate


@app.callback(
    Output("sel-sym",    "data"),
    Output("sel-expiry", "data"),
    Input("url", "pathname"),
)
def _route(pathname):
    """URL → open section. Fires on every navigation AND on initial page load, so a
    direct hit / hard-refresh on /charts (or /live-oi, /alerts, /chain/...) restores
    the section instead of leaving section-gated callbacks stuck in PreventUpdate."""
    p = (pathname or "/").rstrip("/") or "/"
    if p in _PATH_TO_SEL:
        return _PATH_TO_SEL[p], ""
    if p.startswith("/chain/"):
        return _SHORT_TO_SYM.get(p.split("/chain/", 1)[1]), ""
    return None, ""


# ── Charts section: full-session price/OI/volume/premium for index + timeframe ──
@app.callback(Output("charts-leg-col", "style"), Output("charts-strike-col", "style"),
              Output("charts-expiry-col", "style"), Input("charts-mode", "value"))
def _toggle_mode_cols(mode):
    """Futures → show the near/next/far leg picker; Options → show expiry + strike pickers."""
    show, hide = {"display": "block"}, {"display": "none"}
    return (show, hide, hide) if mode == "futures" else (hide, show, show)


@app.callback(Output("charts-strike", "options"), Output("charts-strike", "value"),
              Input("charts-mode", "value"), Input("charts-idx", "value"),
              Input("news-date", "data"), Input("charts-expiry", "value"),
              Input("charts-asof", "value"),
              State("charts-strike", "value"))
def _fill_strikes(mode, sym, date, expiry, asof, cur):
    """Populate the option strike picker (Totals + open±1000) for the index/date/expiry/as-of."""
    opts = [{"label": "Totals (CE/PE)", "value": "totals"}]
    if mode == "options":
        if asof == "ghost":                    # practice: pin to the ghost clock/day
            date, asof = _ghost_ctx(date)
        asof_iso = f"{date}T{asof}:00+05:30" if (asof and asof != "full" and date) else None
        anchor, ks = footprint_chart.atm_strikes(sym or "NSE:NIFTY50-INDEX", date=date or None,
                                                 n=10, expiry=expiry or "weekly",
                                                 as_of=_parse_asof(asof_iso))
        for k in ks:
            off = k - (anchor or k)
            tag = "  • OPEN" if off == 0 else f"  ({off:+d})"
            opts.append({"label": f"{k}{tag}", "value": str(k)})
    vals = {o["value"] for o in opts}
    return opts, (cur if cur in vals else "totals")


@app.callback(Output("charts-help-modal", "is_open"),
              Input("charts-help-btn", "n_clicks"), prevent_initial_call=True)
def _open_help_modal(_n):
    """Open the help popup; the X / click-outside close it natively (dbc)."""
    return True


@app.callback(Output("scout-openpos-modal", "is_open"),
              Output("scout-openpos-body", "children"),
              Output("scout-search", "value"),
              Input("scout-openpos-btn", "n_clicks"),
              State("charts-asof", "value"), State("news-date", "data"),
              prevent_initial_call=True)
def _open_scout_openpos(_n, asof, date):
    """Populate + open the open-positions popup at the strip's current clock (respects an
    explicit Replay minute; else live now). Reconstructs held state fresh on each click."""
    today = datetime.datetime.now(IST).date().isoformat()
    if asof and asof not in ("full", "ghost"):
        day = date or today
        try:
            as_of = datetime.datetime.fromisoformat(f"{day}T{asof}:00+05:30")
        except Exception:
            day, as_of = today, datetime.datetime.now(IST)
    else:
        day, as_of = today, datetime.datetime.now(IST)
    return True, _scout_openpos_body(day, as_of), ""


def _btst_body():
    """🌙 BTST overnight paper ledger — the ONLY validated positive-expectancy signal here
    (close-strength clr>=0.66 → long index FUTURES at the close, exit next ~09:30). Read-only:
    the VM cron is the single writer. Rupee figures are on ONE futures lot, cost-inclusive."""
    import btst_panel as bp
    try:
        open_rows, stale_rows, closed_rows = bp.load()
    except Exception as exc:
        return html.Div(f"BTST ledger unavailable: {exc}",
                        style={"color": "#f87171", "fontSize": "0.8rem", "padding": "10px"})
    s = bp.summary(closed_rows)
    bits = []

    # ── LIVE FORMING PREVIEW — makes the 15:10-15:30 decision window actionable ──────
    # During the session, show each index's forming close-strength RIGHT NOW (leak-safe,
    # as-of this minute), so you see which futures to hold overnight IN your entry window,
    # before the ~15:28 emit cron logs them. Hidden after ~15:35 (the ledger is then the
    # record) and on non-trading days.
    from core.market_calendar import is_trading_day as _itd
    _now = datetime.datetime.now(IST)
    if _itd(_now.date()) and _now.time() <= datetime.time(15, 35):
        in_window = datetime.time(15, 10) <= _now.time() <= datetime.time(15, 30)
        pre = _now.time() < datetime.time(15, 0)
        try:
            fc = bp.forming_candidates(_now)
        except Exception:
            fc = []
        cand = [r for r in fc if r.get("candidate")]
        hdr_c = "#22c55e" if in_window else "#64748b" if pre else "#fbbf24"
        title = (f"🌙 FORMING NOW (as of {_now:%H:%M}) — YOUR ENTRY WINDOW IS OPEN"
                 if in_window else
                 f"🌙 BTST watch (as of {_now:%H:%M}) — decision window opens 15:10"
                 if pre else
                 f"🌙 FORMING (as of {_now:%H:%M}) — provisional, firms by 15:30")
        rows = []
        for r in fc:
            clr = r.get("clr")
            if clr is None:
                rows.append(html.Div(f"  {r['index']:<14} · {r.get('note') or 'no data'}",
                                     style={**MONO, "color": "#64748b", "fontSize": "0.66rem"}))
                continue
            hold = r["candidate"]
            rows.append(html.Div(
                f"  {r['index']:<14} clr {clr:.3f}   spot {r['spot']:,}   "
                + ("✓ HOLD OVERNIGHT (long futures)" if hold else "— weak, skip"),
                style={**MONO, "fontSize": "0.66rem", "fontWeight": "700" if hold else "400",
                       "color": "#34d399" if hold else "#64748b"}))
        bits.append(html.Div([
            html.Div(title, style={"color": hdr_c, "fontWeight": "800",
                                   "fontSize": "0.66rem", "marginBottom": "3px"}),
            *rows,
            html.Div(f"    → {len(cand)} candidate(s). Provisional — 97% of strong closes are "
                     "already set by 15:10; final at 15:30. Paper: nothing auto-executes.",
                     style={"color": "#64748b", "fontSize": "0.56rem", "marginTop": "2px"}),
        ], style={"marginBottom": "10px", "padding": "6px 8px",
                  "background": "#0c1f17" if in_window else "#0f172a",
                  "borderRadius": "6px", "border": f"1px solid {hdr_c}44"}))

    # VIEWER honesty — this box's ledger is a stale copy; the VM's cron owns the real one.
    if _ROLE_VIEWER:
        bits.append(html.Div(
            "⚠ viewer copy — the authoritative ledger lives on the capture VM (cron writes "
            "it). Numbers here can lag tonight's emit / this morning's reconcile.",
            style={"color": "#f59e0b", "fontSize": "0.6rem", "marginBottom": "6px"}))

    # STALE-OPEN = reconcile never ran. These are EXCLUDED from the P&L below, so a broken
    # job would otherwise silently flatter the record (it did once: a −39.6bps loss hidden).
    if stale_rows:
        bits.append(html.Div([
            html.Div(f"⚠ {len(stale_rows)} STALE-OPEN position(s) — the exit day passed but "
                     "reconcile never ran. They are EXCLUDED from the scorecard below, which "
                     "therefore OVERSTATES the edge.",
                     style={"color": "#ef4444", "fontWeight": "700"}),
            *[html.Div(f"    {r['signal_date']}  {r['index']}  clr {r['clr']}",
                       style={**MONO, "color": "#fca5a5"}) for r in stale_rows],
        ], style={"fontSize": "0.62rem", "marginBottom": "8px"}))

    trk_c = {"above expectation": "#22c55e", "tracking": "#22c55e"}.get(s["tracking"], "#f59e0b")
    if s["n"]:
        tot = s["total_rupees"]
        def _tile(label, value, color, sub=""):
            """One stat tile: big number, quiet label. The old summary was a single dense
            run-on line — the eye had nowhere to land and the key figure hid in the middle."""
            return html.Div([
                html.Div(value, style={"color": color, "fontSize": "1.02rem",
                                       "fontWeight": "800", "lineHeight": "1.15"}),
                html.Div(label, style={"color": "#64748b", "fontSize": "0.54rem",
                                       "textTransform": "uppercase",
                                       "letterSpacing": "0.06em", "marginTop": "2px"}),
                html.Div(sub, style={"color": "#475569", "fontSize": "0.52rem"}) if sub else "",
            ], style={"background": "#0f172a", "border": "1px solid #1e293b",
                      "borderRadius": "6px", "padding": "7px 11px", "minWidth": "88px"})

        bits.append(html.Div([
            _tile("nights", f"{s['n']}", "#e2e8f0", f"{s['gate_left']} to gate"),
            _tile("win rate", f"{s['win_pct']:.0f}%", "#e2e8f0", f"{s['wins']}/{s['n']}"),
            _tile("mean / night", f"{s['mean_bps']:+.1f} bps", trk_c,
                  f"expect +{bp.BACKTEST_LO:.0f}–{bp.BACKTEST_HI:.0f}"),
            _tile("total (1 lot)", f"₹{tot:+,}" if tot is not None else "—",
                  "#22c55e" if (tot or 0) >= 0 else "#f87171"),
            _tile("worst night", f"{s['worst_bps']:+.0f} bps", "#f87171"),
        ], style={"display": "flex", "gap": "8px", "flexWrap": "wrap",
                  "marginBottom": "8px"}))

        bits.append(html.Div(
            f"{'✓' if trk_c == '#22c55e' else '⚠'} {s['tracking'].upper()} vs the "
            f"+{bp.BACKTEST_LO:.0f}–{bp.BACKTEST_HI:.0f} bps backtest  ·  paper only, nothing "
            f"auto-executes  ·  real capital only after the {bp.REVIEW_GATE}-night gate",
            style={"color": trk_c, "fontSize": "0.62rem", "fontWeight": "700",
                   "marginBottom": "9px"}))

        # per-index, worst first — where MIDCAP's drag becomes impossible to miss
        rows = bp.per_index(closed_rows)
        bits.append(html.Div([
            html.Div("BY INDEX (worst first)", style={
                "color": "#64748b", "fontSize": "0.54rem", "textTransform": "uppercase",
                "letterSpacing": "0.06em", "marginBottom": "3px"}),
            html.Div([
                html.Span([
                    html.Span(f"{d['index']}  ", style={"color": "#94a3b8"}),
                    html.Span(f"{d['mean_bps']:+.1f}bps ", style={"fontWeight": "800"}),
                    html.Span(f"₹{d['rupee']:+,} ", style={"fontWeight": "700"}),
                    html.Span(f"({d['n']}n · {d['win_pct']:.0f}%)",
                              style={"color": "#64748b", "fontSize": "0.55rem"}),
                ], style={"color": "#22c55e" if d["mean_bps"] >= 0 else "#f87171",
                          "fontSize": "0.63rem", "background": "#0f172a",
                          "border": "1px solid #1e293b", "borderRadius": "5px",
                          "padding": "3px 8px", "marginRight": "6px",
                          "display": "inline-block", "marginBottom": "4px"})
                for d in rows]),
        ], style={"marginBottom": "8px"}))
    else:
        bits.append(html.Div("No closed BTST nights yet.",
                             style={"color": "#94a3b8", "fontSize": "0.75rem"}))

    def _tbl(title, color, rows, cols, cond=None, header_tips=None, tid=None):
        """Sortable, colour-coded DataTable — the same affordances the scout ledger has.
        `cols` is [(id, Name, format_or_None)]; a numeric format makes the column sort by
        VALUE, so clicking "Net bps" ranks -39.6 below +18.4 instead of sorting as text."""
        if not rows:
            return html.Div()
        from dash import dash_table
        return html.Div([
            html.Div(title, style={"color": color, "fontSize": "0.62rem",
                                   "fontWeight": "800", "margin": "8px 0 3px"}),
            dash_table.DataTable(
                id=tid, data=rows,
                columns=[({"name": n, "id": i, "type": "numeric", "format": f}
                          if f is not None else {"name": n, "id": i}) for i, n, f in cols],
                sort_action="native", sort_mode="single", page_action="none",
                style_as_list_view=True,
                style_table={"marginBottom": "4px", "overflowX": "auto"},
                style_header={"backgroundColor": "#1e293b", "color": "#e2e8f0",
                              "fontSize": "0.6rem", "textTransform": "uppercase",
                              "border": "none", "fontWeight": "700", "cursor": "pointer"},
                style_cell={"backgroundColor": "#0f172a", "color": "#e2e8f0",
                            "fontSize": "0.7rem", "border": "none",
                            "padding": "4px 9px", "textAlign": "left"},
                style_data_conditional=cond or [],
                tooltip_header=header_tips or {},
                tooltip_delay=150, tooltip_duration=None,
                css=[{"selector": ".dash-table-tooltip",
                      "rule": "background-color:#0f172a; color:#e2e8f0; "
                              "border:1px solid #334155; font-size:0.68rem; "
                              "max-width:280px; padding:6px 8px;"}],
            ),
        ])

    from dash.dash_table.Format import Format, Group, Scheme, Sign, Symbol
    _f_fut = Format(precision=2, scheme=Scheme.fixed).group(Group.yes).symbol(
        Symbol.yes).symbol_prefix("₹")
    _f_rs = (Format(precision=0, scheme=Scheme.fixed).group(Group.yes)
             .sign(Sign.positive).symbol(Symbol.yes).symbol_prefix("₹"))
    _f_bps = Format(precision=1, scheme=Scheme.fixed).sign(Sign.positive)
    _f_clr = Format(precision=3, scheme=Scheme.fixed)

    def _sign_cond(col):
        """Green when the number made money, red when it lost — the fastest read there is."""
        return [{"if": {"filter_query": f"{{{col}}} < 0", "column_id": col},
                 "color": "#f87171", "fontWeight": "700"},
                {"if": {"filter_query": f"{{{col}}} >= 0", "column_id": col},
                 "color": "#22c55e", "fontWeight": "700"}]

    # WHAT THE INSTRUMENT IS — the panel never said, and it is the #1 confusion: people look
    # for a strike and a CE/PE because the scout ledger has them. BTST has NEITHER.
    if open_rows:
        bits.append(html.Div([
            html.Span("📐 INSTRUMENT: index FUTURES — ", style={"fontWeight": "800"}),
            html.Span("NOT an option. There is no strike and no CE/PE. You BUY the near-month "
                      "futures contract itself — which is exactly why this works overnight: "
                      "futures have NO THETA, so nothing decays while you hold. An option "
                      "would bleed against you every hour."),
        ], style={"background": "#0c1f17", "border": "1px solid #22c55e44",
                  "borderRadius": "6px", "padding": "6px 10px", "marginTop": "6px",
                  "color": "#a7f3d0", "fontSize": "0.63rem", "lineHeight": "1.45"}))

    bits.append(_tbl(
        f"● OPEN OVERNIGHT — {len(open_rows)} position(s) held through the night",
        "#34d399", open_rows,
        [("contract", "Buy this contract", None), ("action", "Action", None),
         ("lot", "Qty (1 lot)", None), ("entry", "Index @ close", _f_fut),
         ("notional", "Exposure (1 lot)", _f_rs), ("clr", "clr", _f_clr),
         ("exits", "Exit at", None)],
        cond=[{"if": {"column_id": "exits"}, "color": "#fbbf24", "fontWeight": "700"},
              {"if": {"column_id": "contract"}, "color": "#e2e8f0", "fontWeight": "700"},
              {"if": {"column_id": "action"}, "color": "#22c55e", "fontWeight": "800"},
              {"if": {"column_id": "notional"}, "color": "#94a3b8"},
              {"if": {"filter_query": "{clr} >= 0.85", "column_id": "clr"},
               "color": "#22c55e", "fontWeight": "800"}],
        header_tips={
            "contract": "The actual instrument: the NEAR-MONTH index futures (all four indices "
                        "share the same last-Tuesday monthly expiry). No strike. No CE/PE.",
            "action": "Long-only. A WEAK close is never a short — the short leg fights the "
                      "overnight drift and adds tail risk.",
            "lot": "One lot = this many units. MIDCAP's 120 vs NIFTY's 65 means the SAME index "
                   "move costs you ~2× more on MIDCAP.",
            "entry": "The INDEX level at the close — the signal's reference price (clr is "
                     "computed from it). You execute the FUTURES, which trades at a small "
                     "basis to this; the overnight bps move tracks the index closely.",
            "notional": "Contract value you carry overnight = index × lot. Margin is only "
                        "~10–12% of this, but a gap moves the FULL exposure — this is the "
                        "number that decides whether a gap-down hurts or ruins.",
            "clr": "Close position in the day's range = (close−low)/(high−low). ≥0.66 fires "
                   "the signal; ≥0.85 is a very strong close.",
            "exits": "Exit is 09:30 the NEXT TRADING DAY (weekend / holiday aware).",
        }, tid="btst-open-table"))

    bits.append(_tbl(
        "○ CLOSED NIGHTS", "#94a3b8", closed_rows,
        [("index", "Index", None), ("triggered", "Triggered", None),
         ("exited", "Exited", None), ("clr", "clr", _f_clr),
         ("entry", "Entry", _f_fut), ("exit", "Exit", _f_fut),
         ("net_bps", "Net bps", _f_bps), ("rupee", "₹/lot (net)", _f_rs)],
        cond=_sign_cond("net_bps") + _sign_cond("rupee"),
        header_tips={
            "net_bps": "Overnight return in basis points, NET of the 3 bps round-trip. The "
                       "backtest expects +10–13 bps/night — that is the bar.",
            "rupee": "What ONE futures lot actually made/lost = entry × bps/1e4 × lot size.",
            "clr": "Close-strength that fired the signal (≥ 0.66).",
            "entry": "INDEX level at the close you entered on (the signal's reference). The "
                     "trade is on the near-month FUTURES — no strike, no CE/PE.",
            "exit": "INDEX level at the 09:30 exit next trading day.",
        }, tid="btst-closed-table"))
    bits.append(html.Div(
        "Rule LOCKED (no tuning): a close in the top third of the day's range (clr ≥ 0.66) → "
        "LONG index FUTURES at the close, exit next 09:30. Long-only — a weak close is NOT a "
        "short. NO STRIKE / NO CE-PE: this is a futures contract, not an option (that is why "
        "it survives overnight — no theta). Prices shown are the INDEX level (the signal's "
        "reference); you execute the near-month future, which trades at a small basis to it. "
        "₹ is on ONE lot, net of 3 bps round-trip. TIMES are FIXED BY THE RULE, not "
        "recorded per-trade: entry is the day's close (the emit job runs ~15:28 and prices the "
        "true ≤15:30 close); exit is 09:30 the next trading day. PAPER: nothing auto-executes; "
        "real capital only after the review gate AND a gap-tail plan (worst backtest night "
        "≈ −2%). This is the system's only validated positive-expectancy signal.",
        style={"color": "#64748b", "fontSize": "0.57rem", "lineHeight": "1.45",
               "marginTop": "10px"}))
    return html.Div(bits)


@app.callback(Output("btst-modal", "is_open"), Output("btst-body", "children"),
              Input("btst-btn", "n_clicks"), prevent_initial_call=True)
def _open_btst(_n):
    return True, _btst_body()


_VERDICT_COLOR = {"TRADE-FADE": "#f472b6", "RANGE-ONLY": "#fbbf24",
                  "NO-TRADE": "#64748b", "WARMING": "#64748b"}


def _tb_num(v, k=False):
    """Compact number: OI/vol in K/L; None → '—'."""
    if v is None:
        return "—"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    if k and abs(v) >= 1e5:
        return f"{v/1e5:.1f}L"
    if k and abs(v) >= 1e3:
        return f"{v/1e3:.0f}K"
    return f"{v:g}"


def _tb_mtf_strip(mtf):
    """Multi-TF price-action + structure grid (5/15/30/60m)."""
    _sc = {"BREAKOUT_UP": "#22c55e", "BREAKOUT_DOWN": "#f87171", "TREND_UP": "#4ade80",
           "TREND_DOWN": "#fb7185", "CONSOLIDATION": "#fbbf24", "RANGE": "#64748b",
           "n/a": "#475569"}
    cells = []
    for m in mtf:
        col = _sc.get(m["struct"], "#94a3b8")
        cells.append(html.Div([
            html.Div(f"{m['tf']}m", style={"color": "#64748b", "fontSize": "0.6rem"}),
            html.Div(m["struct"], style={"color": col, "fontWeight": "700",
                                         "fontSize": "0.66rem"}),
            html.Div(f"ER {m['er'] if m['er'] is not None else '—'} · {m['char']}",
                     style={"color": "#94a3b8", "fontSize": "0.6rem"}),
        ], style={"padding": "4px 10px", "borderRight": "1px solid #1e293b",
                  "minWidth": "120px"}))
    return html.Div(cells, style={"display": "flex", "flexWrap": "wrap",
                    "background": "#0a1220", "borderRadius": "8px", "marginBottom": "8px"})


def _tb_strike_table(strikes, chain):
    """Option chain: ATM+/-4 strikes, CE | STRIKE | PE, OI/COI/vol/IV/prem. Walls + max-pain
    highlighted. The RAW data — read it, it does not tell you a direction."""
    cw, pw, mp = chain.get("call_wall"), chain.get("put_wall"), chain.get("max_pain")
    th = lambda t: html.Th(t, style={"padding": "3px 6px", "color": "#64748b",
                                     "fontSize": "0.58rem", "fontWeight": "700"})
    header = html.Tr([th("CE OI"), th("COI"), th("Vol"), th("IV"), th("Prem"),
                      th("STRIKE"), th("Prem"), th("IV"), th("Vol"), th("COI"), th("PE OI")])
    body = []
    for s in strikes:
        k = s["strike"]
        tag = (" ●CW" if k == cw else "") + (" ●PW" if k == pw else "") + \
              (" ◆MP" if k == mp else "")
        kcol = "#f472b6" if s["atm"] else ("#40c4ff" if tag else "#cbd5e1")
        bg = "#141d2e" if s["atm"] else "transparent"

        def td(v, col="#94a3b8", kk=False, bold=False):
            return html.Td(_tb_num(v, kk), style={"padding": "2px 6px", "color": col,
                           "fontSize": "0.62rem", "textAlign": "center",
                           "fontWeight": "700" if bold else "400"})
        body.append(html.Tr([
            td(s["ce_oi"], "#93c5fd", True), td(s["ce_oich"],
               "#22c55e" if (s["ce_oich"] or 0) > 0 else "#f87171", True),
            td(s["ce_vol"], "#94a3b8", True), td(s["ce_iv"]), td(s["ce_prem"]),
            html.Td([f"{k}", html.Span(tag, style={"color": "#40c4ff", "fontSize": "0.52rem"})],
                    style={"padding": "2px 8px", "color": kcol, "fontWeight": "800",
                           "fontSize": "0.66rem", "textAlign": "center", "background": bg}),
            td(s["pe_prem"]), td(s["pe_iv"]), td(s["pe_vol"], "#94a3b8", True),
            td(s["pe_oich"], "#22c55e" if (s["pe_oich"] or 0) > 0 else "#f87171", True),
            td(s["pe_oi"], "#fca5a5", True),
        ], style={"background": bg}))
    legend = html.Div(f"●CW call-wall {cw or '—'} · ●PW put-wall {pw or '—'} · "
                      f"◆MP max-pain {mp or '—'} · PCR {chain.get('pcr') or '—'}",
                      style={"color": "#64748b", "fontSize": "0.6rem", "marginTop": "3px"})
    return html.Div([html.Table([html.Thead(header), html.Tbody(body)],
                    style={"width": "100%", "borderCollapse": "collapse"}), legend])


def _tradeboard_body():
    """🎯 The full fused PAGE. Read-only over tradeboard.build_board(): per index — mood/ER,
    the MULTI-TF price-action + structure grid, the VALIDATED band, the FULL option chain
    (OI/COI/vol/IV/premium + walls/max-pain/PCR + expiry), and the ONE honest suggestion
    (the band-fade, paper-first). No directional CE/PE arrow — that product is dead."""
    import tradeboard as tb
    now = datetime.datetime.now(IST)
    try:
        rows = tb.build_board(None, now)
    except Exception as exc:
        return html.Div(f"TradeBoard unavailable: {exc}",
                        style={"color": "#f87171", "fontSize": "0.8rem", "padding": "10px"})

    def _kv(label, val, color="#cbd5e1"):
        return html.Span([html.Span(f"{label} ", style={"color": "#64748b"}),
                          html.Span(str(val), style={"color": color, "fontWeight": "600"})],
                         style={"marginRight": "14px", "fontSize": "0.72rem"})

    def _sec(title):
        return html.Div(title, style={"color": "#64748b", "fontSize": "0.58rem",
                        "fontWeight": "800", "letterSpacing": "0.12em", "margin": "8px 0 3px"})

    cards = [html.Div(
        f"as of {now:%H:%M} IST · this board is DATA for YOUR read (multi-TF price action + "
        f"structure + full option chain) · NO validated machine trade: the band-fade edge was "
        f"a fill artifact (loses ~-6 to -11bps at real fills, audit 2026-07-16) · band = risk "
        f"map · overnight (BTST) is the only validated directional edge",
        style={"color": "#94a3b8", "fontSize": "0.7rem", "marginBottom": "12px",
               "fontStyle": "italic"})]

    for r in rows:
        vc = _VERDICT_COLOR.get(r["verdict"], "#64748b")
        head = [html.Span(r["label"], style={"fontWeight": "800", "fontSize": "1rem",
                                             "color": "#e2e8f0", "marginRight": "12px"})]
        if r["verdict"] == "WARMING":
            head.append(html.Span("WARMING — " + r.get("note", ""),
                                  style={"color": "#64748b", "fontSize": "0.72rem"}))
            cards.append(html.Div(head, style={"padding": "10px 12px", "marginBottom": "10px",
                        "background": BG_CARD, "borderRadius": "8px",
                        "borderLeft": "3px solid #64748b"}))
            continue
        head += [
            html.Span(f"{r['spot']:.2f}", style={"color": "#cbd5e1", "marginRight": "12px",
                     "fontSize": "0.95rem"}),
            _kv("mood", r["mood"], "#a78bfa"),
            _kv("ER", r["er"] if r["er"] is not None else "—"),
            _kv("expiry DTE", f"{r['expiry_dte']} ({'weekly' if r['weekly'] else 'monthly'})"),
            html.Span(r["verdict"], style={"marginLeft": "auto", "color": "#0a0f1a",
                     "background": vc, "borderRadius": "6px", "padding": "3px 12px",
                     "fontWeight": "800", "fontSize": "0.72rem"}),
        ]
        body = [html.Div(head, style={"display": "flex", "alignItems": "center",
                                      "flexWrap": "wrap", "marginBottom": "4px"})]
        # 1) PRICE ACTION + STRUCTURE (multi-TF)
        body.append(_sec("PRICE ACTION · STRUCTURE (multi-TF)"))
        body.append(_tb_mtf_strip(r.get("mtf", [])))
        # 2) BAND (validated risk product)
        cov = (f"{100*r['cover']:.0f}% ({r['cover_conf']})" if r.get("cover") is not None else "—")
        body.append(html.Div([_kv("band", f"[{r['band_lo']}, {r['band_hi']}]", "#40c4ff"),
                              _kv("cover", cov)], style={"marginBottom": "2px"}))
        # 3) OPTION CHAIN (OI/COI/vol/IV/premium + walls/max-pain/PCR)
        age = r.get("chain_age")
        stale = r.get("chain_stale")
        body.append(_sec(f"OPTION CHAIN — OI · COI · VOL · IV · PREMIUM   "
                         f"({'STALE ' + str(age) + 'm — context only' if stale else 'live ' + str(age) + 'm'})"))
        if stale:
            body.append(html.Div("chain data stale (capture dies ~11am) — greyed, not scored",
                                 style={"color": "#f87171", "fontSize": "0.64rem",
                                        "marginBottom": "4px"}))
        if r.get("strikes"):
            body.append(_tb_strike_table(r["strikes"], r.get("chain", {})))
        # 4) CONTEXT MARKER (NOT a trade — the fade edge was a fill artifact, audit 2026-07-16)
        s = r.get("setup")
        body.append(_sec("CONTEXT MARKER — no validated intraday trade"))
        if s:
            body.append(html.Div([
                html.Span("⚠ stretch+rejection at band edge (context, NOT a trade): ",
                          style={"color": "#fbbf24", "fontWeight": "700",
                                 "fontSize": "0.68rem"}),
                _kv("side", s["side"].replace("-fade", ""), "#cbd5e1"),
                _kv("edge~", s["entry"]), _kv("mean~", s["target"]),
                _kv("band", f"{s['band_pct']}%"), _kv("clr", s["clr"]), _kv("ER", s["er"]),
            ], style={"marginTop": "2px"}))
        else:
            body.append(html.Div("no marker — band is the product (risk map), not a direction",
                                 style={"color": "#94a3b8", "fontSize": "0.7rem"}))
        if r.get("note"):
            body.append(html.Div("· " + r["note"],
                                 style={"color": "#94a3b8", "fontSize": "0.68rem",
                                        "fontStyle": "italic", "marginTop": "3px"}))
        cards.append(html.Div(body, style={"padding": "14px 16px", "marginBottom": "14px",
                    "background": BG_CARD, "borderRadius": "10px",
                    "borderLeft": f"3px solid {vc}"}))
    return html.Div(cards)


_STRUCT_COLOR = {"BREAKOUT_UP": "#22c55e", "BREAKOUT_DOWN": "#ef4444", "TREND_UP": "#4ade80",
                 "TREND_DOWN": "#fb7185", "CONSOLIDATION": "#fbbf24", "RANGE": "#64748b",
                 "n/a": "#475569"}


def _tb_candle_fig(sym, tf, date, as_of, band_lo=None, band_hi=None):
    """One TF candlestick — CONTINUOUS recent bars (stitched multi-day, same series the ER is
    computed on), band overlay, structure+ER in the title. Drawing the continuous series (not
    a single day) means the chart is never empty pre-market / early session — the coarse TFs
    show recent sessions, matching the ER. rangebreaks hide weekend + overnight so contiguous."""
    import tradeboard as tb
    sf = tb._struct_full(sym, tf, date, as_of)          # 20-bar Kaufman ER + structure (BTST)
    struct, er = sf.get("struct", "n/a"), sf.get("er")
    # ER = Kaufman efficiency ratio over the last 20 CLOSED bars: |net travel| / Σ|bar moves|.
    # 1 = clean one-way trend, 0 = thrash-with-no-progress (chop). The trend-vs-chop meter.
    if er is None:
        er_txt = ""
    elif er < 0.30:
        er_txt = f" · ER {er} chop"
    elif er < 0.45:
        er_txt = f" · ER {er} weak-trend"
    else:
        er_txt = f" · ER {er} STRONG-trend"
    # LOAD a deep history so the user can scroll/drag BACK; DEFAULT-VIEW only the recent window.
    _load = {5: 750, 10: 500, 15: 400, 30: 220, 60: 160}.get(tf, 300)   # ~10-25 sessions
    _view = {5: 78, 10: 40, 15: 34, 30: 26, 60: 26}.get(tf, 40)          # initial visible bars
    cont = tb._bars_continuous(sym, tf, date, as_of, need=_load)
    fig = go.Figure()
    col = _STRUCT_COLOR.get(struct, "#94a3b8")
    if cont is None or len(cont) < 3:
        fig.add_annotation(text="no data yet", showarrow=False,
                           font=dict(color="#64748b", size=11))
        fig.update_layout(template="plotly_dark", height=250, paper_bgcolor=BG_CARD,
                          plot_bgcolor=BG_CARD, margin=dict(l=8, r=8, t=26, b=8),
                          xaxis=dict(visible=False), yaxis=dict(visible=False),
                          title=dict(text=f"{tf}m · {struct}{er_txt}", x=0.02,
                                     font=dict(size=11, color=col)))
        return fig
    fig.add_trace(go.Candlestick(
        x=cont["ts"], open=cont["open"], high=cont["high"], low=cont["low"],
        close=cont["close"], name="",
        increasing_line_color="#22c55e", decreasing_line_color="#ef4444",
        increasing_fillcolor="#22c55e", decreasing_fillcolor="#ef4444",
        line=dict(width=1), showlegend=False))
    # BAND overlay (validated risk zone — where price is likely to sit; NOT a signal)
    if band_lo and band_hi:
        fig.add_hrect(y0=band_lo, y1=band_hi, fillcolor="#40c4ff", opacity=0.07, line_width=0)
        for y in (band_lo, band_hi):
            fig.add_hline(y=y, line=dict(color="#40c4ff", width=0.8, dash="dot"))
    rb = [dict(bounds=["sat", "mon"]), dict(bounds=[15.6, 9.25], pattern="hour")]
    # initial view = the recent _view bars; the rest is loaded → drag/scroll BACK to see it
    x0 = cont["ts"].iloc[-min(_view, len(cont))]
    x1 = cont["ts"].iloc[-1] + pd.Timedelta(minutes=tf)
    fig.update_layout(
        template="plotly_dark", height=250, paper_bgcolor=BG_CARD, plot_bgcolor=BG_CARD,
        margin=dict(l=8, r=50, t=26, b=20), hovermode="x unified", showlegend=False,
        dragmode="pan",                                    # drag = scroll back through history
        title=dict(text=f"{tf}m · <b>{struct}</b>{er_txt}", x=0.02, font=dict(size=11, color=col)),
        xaxis=dict(rangeslider_visible=False, gridcolor="#0f1a2a", rangebreaks=rb,
                   range=[x0, x1], tickfont=dict(color="#475569", size=8)),
        yaxis=dict(side="right", gridcolor="#0f1a2a", tickfont=dict(color="#64748b", size=8),
                   tickformat=",.0f", autorange=True, fixedrange=False))
    return fig


def _tb_daily_fig(sym):
    """DAILY candle chart (last ~45 EOD bars) + 200-DMA. The 1D bar carries the validated
    overnight edge — the top of the MTF stack."""
    import tradeboard as tb
    d = tb._daily_read(sym)
    fig = go.Figure()
    from pathlib import Path
    from core.constants import DATA_DIR
    fn = sym.replace(":", "_").replace("-", "_")
    p = Path(DATA_DIR) / "historical" / "daily" / f"{fn}_daily.parquet"
    if not p.exists():
        fig.update_layout(template="plotly_dark", height=260, paper_bgcolor=BG_CARD,
                          plot_bgcolor=BG_CARD)
        return fig
    df = pd.read_parquet(p).sort_values("ts").tail(45)
    sma = pd.read_parquet(p).sort_values("ts")["close"].rolling(200).mean().tail(45)
    fig.add_trace(go.Candlestick(x=df["ts"], open=df["open"], high=df["high"], low=df["low"],
        close=df["close"], name="", increasing_line_color="#22c55e",
        decreasing_line_color="#ef4444", increasing_fillcolor="#22c55e",
        decreasing_fillcolor="#ef4444", line=dict(width=1), showlegend=False))
    fig.add_trace(go.Scatter(x=df["ts"], y=sma.values, mode="lines", name="200-DMA",
        line=dict(color="#a78bfa", width=1, dash="dot"), hoverinfo="skip"))
    struct = d.get("struct", "")
    col = "#22c55e" if "UP" in struct else "#ef4444" if "DOWN" in struct else "#fbbf24"
    fig.update_layout(template="plotly_dark", height=260, paper_bgcolor=BG_CARD,
        plot_bgcolor=BG_CARD, margin=dict(l=8, r=50, t=26, b=20), showlegend=False,
        title=dict(text=f"1D · <b>{struct}</b> · clr {d.get('clr')} · ER20 {d.get('er20')} · "
                   f"{d.get('regime','')}", x=0.02, font=dict(size=11, color=col)),
        xaxis=dict(rangeslider_visible=False, gridcolor="#0f1a2a",
                   rangebreaks=[dict(bounds=["sat", "mon"])],
                   tickfont=dict(color="#475569", size=8)),
        yaxis=dict(side="right", gridcolor="#0f1a2a", tickfont=dict(color="#64748b", size=8),
                   tickformat=",.0f"))
    return fig


def _tb_daily_panel(row):
    """The DAILY read = the top of the stack and the ONLY validated directional edge here:
    daily close-strength → overnight long (8.5yr-validated), regime-gated. Character shown
    for the user's own PA; the overnight lean is the machine's one real call."""
    d = row.get("daily") or {}
    if not d:
        return html.Div("daily EOD not available", style={"color": "#64748b",
                        "fontSize": "0.7rem"})
    strong = d.get("strong_close")
    lc = "#22c55e" if strong else "#64748b"

    def _kv(lab, val, c="#cbd5e1"):
        return html.Span([html.Span(f"{lab} ", style={"color": "#64748b"}),
                          html.Span(str(val), style={"color": c, "fontWeight": "600"})],
                         style={"marginRight": "13px", "fontSize": "0.72rem"})
    char = html.Div([
        _kv("close", d.get("close")), _kv("clr", d.get("clr"),
            "#22c55e" if (d.get("clr") or 0) >= 0.66 else "#cbd5e1"),
        _kv("body", d.get("body")), _kv("upper-wick", d.get("uwick")),
        _kv("lower-wick", d.get("lwick")), _kv("ER20", d.get("er20")),
        _kv("regime", d.get("regime"), "#a78bfa"),
    ], style={"marginBottom": "4px"})
    onv, win = d.get("onv"), d.get("onv_win")
    forming, actionable = d.get("forming"), d.get("actionable")
    src = ("TODAY forming" if forming else f"session {d.get('asof_date','')}")
    if strong and actionable and onv:
        lean = html.Div([
            html.Span("OVERNIGHT LEAN: ", style={"color": lc, "fontWeight": "800",
                      "fontSize": "0.74rem"}),
            html.Span("LONG into close → exit next open", style={"color": "#e2e8f0",
                      "fontWeight": "700", "fontSize": "0.72rem"}),
            html.Span(f"  (8.5yr-validated: strong-close pays ~+{onv}% overnight, win {win}%, "
                      f"{'size up' if d.get('regime') == 'bull' else 'smaller — bear'})",
                      style={"color": "#94a3b8", "fontSize": "0.66rem"}),
        ])
    elif strong and forming:
        lean = html.Div([
            html.Span("OVERNIGHT LEAN: ", style={"color": "#fbbf24", "fontWeight": "800",
                      "fontSize": "0.74rem"}),
            html.Span("forming strong (clr≥0.66) — FIRMS in the 15:10–15:30 window; not yet "
                      "actionable", style={"color": "#e2e8f0", "fontSize": "0.7rem"}),
        ])
    else:
        lean = html.Div("OVERNIGHT LEAN: none — daily close not strong (clr<0.66); the edge "
                        "fires only on a strong daily close", style={"color": "#94a3b8",
                        "fontSize": "0.7rem"})
    note = html.Div(f"daily = {src} · the ONE validated directional edge here (overnight); "
                    f"intraday TFs below are CONTEXT for your own PA · the tonight call firms "
                    f"at the 15:10–15:30 close", style={"color": "#64748b",
                    "fontSize": "0.58rem", "fontStyle": "italic", "marginTop": "3px"})
    return html.Div([char, lean, note], style={"background": "#0a1220",
                    "borderLeft": f"3px solid {lc}", "borderRadius": "8px",
                    "padding": "8px 12px", "margin": "6px 0"})


def _tb_chain_panel(row):
    """Below the candles: band + full option chain (OI/COI/vol/IV/premium + walls/max-pain/
    PCR) + expiry + the HONEST read (context marker, never a machine trade)."""
    def _kv(label, val, color="#cbd5e1"):
        return html.Span([html.Span(f"{label} ", style={"color": "#64748b"}),
                          html.Span(str(val), style={"color": color, "fontWeight": "600"})],
                         style={"marginRight": "14px", "fontSize": "0.72rem"})

    def _sec(t):
        return html.Div(t, style={"color": "#64748b", "fontSize": "0.58rem", "fontWeight": "800",
                        "letterSpacing": "0.12em", "margin": "10px 0 3px"})
    cov = (f"{100*row['cover']:.0f}% ({row['cover_conf']})" if row.get("cover") is not None else "—")
    age, stale = row.get("chain_age"), row.get("chain_stale")
    head = [html.Span(row["label"], style={"fontWeight": "800", "fontSize": "0.95rem",
                     "color": "#e2e8f0", "marginRight": "12px"}),
            html.Span(f"{row['spot']:.2f}", style={"color": "#cbd5e1", "marginRight": "12px"}),
            _kv("mood", row["mood"], "#a78bfa"), _kv("ER", row["er"] if row["er"] is not None else "—"),
            _kv("expiry DTE", f"{row['expiry_dte']} ({'weekly' if row['weekly'] else 'monthly'})"),
            _kv("band", f"[{row['band_lo']}, {row['band_hi']}]", "#40c4ff"), _kv("cover", cov)]
    out = [html.Div(head, style={"display": "flex", "flexWrap": "wrap", "alignItems": "center",
                    "marginTop": "6px"})]
    # ── PHASE 2 — MTF PRICE ACTION, the user's THREE MASTER COMBOS (5×15, 10×30, 15×60) ──
    # Each = a lower-TF entry confirmed by its higher-TF structure. CONTEXT for the user's own
    # PA read (intraday MTF has no validated MACHINE edge — every gate failed OOS this session).
    out.append(_sec("PHASE 2 — PRICE ACTION · MTF CONFIRMATION (3 combos · lower × higher)"))
    for cb in (row.get("combos") or []):
        sc = cb.get("color", "#94a3b8")
        locpct = f"{100*cb['loc']:.0f}%" if cb.get("loc") is not None else "—"
        out.append(html.Div([
            html.Div([
                html.Span(f"{cb.get('ltf_tf')}m→{cb.get('htf_tf')}m  ", style={"color": "#a78bfa",
                          "fontSize": "0.62rem", "fontWeight": "800"}),
                html.Span(f"LTF {cb.get('ltf_struct')}", style={"color": "#cbd5e1",
                          "fontSize": "0.64rem", "fontWeight": "700"}),
                html.Span(" conf.by ", style={"color": "#475569", "fontSize": "0.58rem"}),
                html.Span(f"HTF {cb.get('htf_struct')}", style={"color": "#cbd5e1",
                          "fontSize": "0.64rem", "fontWeight": "700"}),
                html.Span(f"  box {locpct}", style={"color": "#64748b", "fontSize": "0.58rem"}),
                html.Span(f"  {cb.get('tag','')}", style={"marginLeft": "6px", "color": "#0a0f1a",
                          "background": sc, "borderRadius": "5px", "padding": "1px 8px",
                          "fontWeight": "800", "fontSize": "0.62rem"}),
            ]),
            html.Div([
                html.Span("entry candle ", style={"color": "#64748b", "fontSize": "0.58rem"}),
                html.Span(f"{cb.get('ltf_tf')}m {cb.get('ltf_pattern','—')}",
                          style={"color": "#fbbf24", "fontSize": "0.62rem", "fontWeight": "700"}),
                html.Span(f"   · HTF confirm candle {cb.get('htf_tf')}m {cb.get('htf_pattern','—')}",
                          style={"color": "#64748b", "fontSize": "0.58rem"}),
            ], style={"marginTop": "1px"}),
            html.Div(cb.get("read", ""), style={"color": "#94a3b8", "fontSize": "0.64rem",
                     "marginTop": "2px"}),
        ], style={"background": "#0a1220", "borderLeft": f"3px solid {sc}",
                  "borderRadius": "6px", "padding": "6px 10px", "marginBottom": "4px"}))
    out.append(html.Div("PHASE 1 = the option data below (your Charts engine) · PHASE 2 = these "
                        "combos · YOU gate both with your PA — no machine auto-trade (fails OOS)",
                        style={"color": "#64748b", "fontSize": "0.56rem", "fontStyle": "italic",
                               "marginBottom": "4px"}))
    out.append(_sec(f"PHASE 1 — OPTION CHAIN · OI · COI · VOL · IV · PREMIUM  "
                    f"({'STALE ' + str(age) + 'm — context only' if stale else 'live ' + str(age) + 'm'})"))
    if stale:
        out.append(html.Div("chain stale (capture dies ~11am) — greyed, not scored",
                            style={"color": "#f87171", "fontSize": "0.64rem"}))
    if row.get("strikes"):
        out.append(_tb_strike_table(row["strikes"], row.get("chain", {})))
    out.append(_sec("READ — no validated machine trade"))
    s = row.get("setup")
    if s:
        out.append(html.Div([
            html.Span("⚠ stretch+rejection at band edge (CONTEXT for your own PA, NOT a trade): ",
                      style={"color": "#fbbf24", "fontWeight": "700", "fontSize": "0.68rem"}),
            _kv("side", s["side"].replace("-fade", ""), "#cbd5e1"),
            _kv("edge~", s["entry"]), _kv("mean~", s["target"]),
            _kv("band", f"{s['band_pct']}%"), _kv("clr", s["clr"]),
        ]))
    if row.get("note"):
        out.append(html.Div("· " + row["note"], style={"color": "#94a3b8", "fontSize": "0.66rem",
                            "fontStyle": "italic", "marginTop": "3px"}))
    return html.Div(out, style={"padding": "6px 4px"})


@app.callback(Output("tb-ledger-modal", "is_open"),
              Input("tb-ledger-btn", "n_clicks"), prevent_initial_call=True)
def _open_ledger(_n):
    return True


# outcome badges — SAME vocabulary as the Charts scout ledger (dashboard.py:5177)
_LED_BADGE = {"band ↑ upper": ("⚡ band ↑ upper", "#22c55e"),
              "band ↓ lower": ("⚡ band ↓ lower", "#22c55e"),
              "SL hit": ("🛑 SL hit", "#f87171"),
              "flipped · reversed out": ("↺ flipped · reversed out", "#f59e0b"),
              "squared off at the bell": ("🔔 squared off at the bell", "#94a3b8")}


def _tb_clock(vdate):
    """TradeBoard clock from the MASTER date store (news-date). Viewing a PAST day → that
    day's full-session replay (as_of 15:35); today → live now. Fixes the '0 closed' confusion
    when the date scroller points at yesterday (the board was hardcoded live-now)."""
    now = datetime.datetime.now(IST)
    today = now.date().isoformat()
    if vdate and vdate != today:
        d0 = datetime.date.fromisoformat(vdate)
        return vdate, datetime.datetime.combine(d0, datetime.time(15, 35), tzinfo=IST)
    return None, now


def _led_badge(outcome: str):
    """Badge lookup — 'timed out (Xm)' carries a combo-dependent minute count, so it is
    prefix-matched rather than keyed exactly."""
    if outcome.startswith("timed out"):
        return (f"⌛ {outcome}", "#a78bfa")
    return _LED_BADGE.get(outcome, (outcome, "#94a3b8"))


@app.callback(Output("tb-ledger", "children"), Output("tb-ledger-badge", "children"),
              Input("sel-sym", "data"), Input("tb-scout-combo", "value"),
              Input("news-date", "data"), Input("tb-refresh", "n_intervals"))
def _fill_ledger(sel, combo, vdate, _tick):
    """SCOUT day-ledger (Charts-parity) — today's PA level-trades, open + closed, with the same
    exit vocabulary as the Charts ledger: band touch / SL hit / flipped / timed out (90m) /
    squared off at the bell. Measures the method (2yr ~breakeven, MIDCAP marginally +) — NOT a
    machine fire; the naked CE/PE is negative-EV, the LEVELS are the trade."""
    from dash.exceptions import PreventUpdate
    if sel != "TRADEBOARD":
        raise PreventUpdate
    import tradeboard as tb
    vdt, now = _tb_clock(vdate)
    ltf_tf, htf_tf = (int(x) for x in (combo or "15-60").split("-"))
    try:
        L = tb.scout_pa_ledger(vdt, now, ltf_tf, htf_tf)
    except Exception as exc:
        return html.Div(f"ledger error: {exc}", style={"color": "#f87171"}), ""
    closed, openr = L["closed"], L["open"]
    badge = html.Span(f"📋 {len(openr)} open · {L['n_closed']} closed", style={
        "color": "#94a3b8", "fontWeight": "700"})

    def _tbl(rows, is_open):
        if not rows:
            return html.Div("none", style={"color": "#64748b", "fontSize": "0.66rem"})
        th = lambda t: html.Th(t, style={"padding": "3px 9px", "color": "#64748b",
                                         "fontSize": "0.56rem", "fontWeight": "800",
                                         "textAlign": "left"})
        cols = (["INDEX", "SIDE", "STRIKE", "TRIGGER", "ENTRY ₹", "NOW ₹", "₹ P&L", "%", "STATUS"]
                if is_open else
                ["INDEX", "SIDE", "STRIKE", "TRIGGER", "EXIT ⏱", "ENTRY ₹", "EXIT ₹", "₹ P&L",
                 "%", "OUTCOME"])
        body = []
        for r in rows:
            ors = r.get("opt_rs")
            pc = "#22c55e" if (ors or 0) > 0 else "#f87171" if (ors or 0) < 0 else "#94a3b8"

            def td(v, c="#cbd5e1", b=False):
                return html.Td("n/a" if v is None else str(v),
                               style={"padding": "3px 9px", "fontSize": "0.68rem",
                                      "color": c if v is not None else "#475569",
                                      "fontWeight": "700" if b else "400"})
            ep = r.get("e_prem"); xp = r.get("x_prem")
            pnl_rs = (f"{ors:+,}" if ors is not None else None)
            pnl_pc = (f"{r['opt_pct']:+}%" if r.get("opt_pct") is not None else None)
            if is_open:
                cells = [td(r["label"], "#e2e8f0", True),
                         td(r["side"], "#22c55e" if r["side"] == "CE" else "#f87171"),
                         td(r.get("strike")), td(r["since"]), td(ep, "#e2e8f0"), td(xp),
                         td(pnl_rs, pc, True), td(pnl_pc, pc), td("● open", "#94a3b8")]
            else:
                bt, bc = _led_badge(r["outcome"])
                _h = (r.get("held") or "→").split("→")
                trig_t, exit_t = (_h + ["", ""])[:2]
                cells = [td(r["label"], "#e2e8f0", True),
                         td(r["side"], "#22c55e" if r["side"] == "CE" else "#f87171"),
                         td(r.get("strike")), td(trig_t, "#e2e8f0"),
                         td(exit_t, "#fbbf24", True), td(ep, "#e2e8f0"), td(xp),
                         td(pnl_rs, pc, True), td(pnl_pc, pc), td(bt, bc, True)]
            body.append(html.Tr(cells, style={"borderTop": "1px solid #1e293b"}))
        return html.Table([html.Thead(html.Tr([th(c) for c in cols])), html.Tbody(body)],
                          style={"width": "100%", "borderCollapse": "collapse"})
    opt_rows = [r for r in closed if r.get("opt_rs") is not None]
    opt_total = sum(r["opt_rs"] for r in opt_rows)
    opt_wins = sum(1 for r in opt_rows if r["opt_rs"] > 0)
    opt_txt = (f"   ·   OPTION ₹ P&L: {opt_total:+,} ({opt_wins}/{len(opt_rows)} win, "
               f"chain-fresh rows)" if opt_rows else
               "   ·   OPTION premiums: n/a (chain not fresh — dies ~11am)")
    tc = "#22c55e" if opt_total > 0 else "#f87171" if opt_total < 0 else "#94a3b8"
    _sup = L.get("suppressed", 0)
    scoreboard = html.Div([
        html.Span(f"{len(openr)} open · {L['n_closed']} closed", style={"color": "#e2e8f0",
                  "fontWeight": "800", "fontSize": "0.78rem"}),
        html.Span(f"  · {_sup} suppressed (dead-tape / 3-strikes)" if _sup else "",
                  style={"color": "#f59e0b", "fontSize": "0.66rem"}),
        html.Span(opt_txt, style={"color": tc, "fontSize": "0.72rem", "fontWeight": "700"}),
        html.Span(f"   · index-level {L['avg_pct'] if L['avg_pct'] is not None else '—'}% avg / "
                  f"{L['avg_r'] if L['avg_r'] is not None else '—'}R",
                  style={"color": "#64748b", "fontSize": "0.66rem"})],
        style={"marginBottom": "8px"})
    body_div = html.Div([
        scoreboard,
        html.Div("● OPEN", style={"color": "#22c55e", "fontSize": "0.6rem", "fontWeight": "800",
                 "margin": "6px 0 2px"}), _tbl(openr, True),
        html.Div("○ CLOSED", style={"color": "#64748b", "fontSize": "0.6rem", "fontWeight": "800",
                 "margin": "10px 0 2px"}), _tbl(closed, False),
        html.Div("ENTRY ₹/NOW ₹/EXIT ₹ = the ATM CE/PE PREMIUM (1 lot) — from the captured chain, "
                 "LIVE ONLY and only while FRESH: the chain dies ~11am so afternoon rows show n/a "
                 "(no reliable premium) · exits: band touch / SL / flipped / 90m / bell · the "
                 "naked CE/PE is MEASURED NEGATIVE-EV (theta+spread bleed the option even when "
                 "the index level is ~breakeven) — this ledger SHOWS you that; the LEVELS are the "
                 "edge, not the option", style={"color": "#64748b", "fontSize": "0.56rem",
                 "fontStyle": "italic", "marginTop": "10px"}),
    ])
    return body_div, badge


@app.callback(Output("tb-scout", "children"),
              Input("sel-sym", "data"), Input("tb-scout-combo", "value"),
              Input("news-date", "data"), Input("tb-refresh", "n_intervals"))
def _fill_scout(sel, combo, vdate, _tick):
    """Cross-index SCOUT table — scan all indices on the chosen combo, best setup first.
    Discretionary-read scanner (which index has the cleanest MTF confluence), not a machine
    buy (intraday PA doesn't mechanize — audited)."""
    from dash.exceptions import PreventUpdate
    if sel != "TRADEBOARD":
        raise PreventUpdate
    import tradeboard as tb
    vdt, now = _tb_clock(vdate)
    ltf_tf, htf_tf = (int(x) for x in (combo or "15-60").split("-"))
    try:
        rows = tb.scout_scan(vdt, now, ltf_tf, htf_tf)
    except Exception as exc:
        return html.Div(f"scout error: {exc}", style={"color": "#f87171"})
    th = lambda t: html.Th(t, style={"padding": "3px 8px", "color": "#64748b",
                                     "fontSize": "0.56rem", "fontWeight": "800",
                                     "textAlign": "left"})
    header = html.Tr([th("INDEX"), th(f"{ltf_tf}m ENTRY (struct · candle)"),
                      th(f"{htf_tf}m CONFIRM (struct · candle)"), th("box"), th("VERDICT")])
    body = []
    for r in rows:
        sc = r.get("color", "#94a3b8")
        loc = f"{100*r['loc']:.0f}%" if r.get("loc") is not None else "—"

        def td(children, **kw):
            st = {"padding": "3px 8px", "fontSize": "0.66rem", **kw}
            return html.Td(children, style=st)
        body.append(html.Tr([
            td(r["label"], color="#e2e8f0", fontWeight="800"),
            td(f"{r.get('ltf_struct')} · {r.get('ltf_pattern')}", color="#cbd5e1"),
            td(f"{r.get('htf_struct')} · {r.get('htf_pattern')}", color="#cbd5e1"),
            td(loc, color="#64748b"),
            html.Td(html.Span(r.get("tag", ""), style={"color": "#0a0f1a", "background": sc,
                    "borderRadius": "5px", "padding": "2px 9px", "fontWeight": "800",
                    "fontSize": "0.62rem"}), style={"padding": "3px 8px"}),
        ], style={"borderTop": "1px solid #1e293b"}))
        # levels line — band / S-R / strike / entry-target-SL (context; naked arrow = neg-EV)
        lv = r.get("levels") or {}
        if lv:
            def _kv(lb, v, cl="#cbd5e1"):
                return html.Span([html.Span(f"{lb} ", style={"color": "#475569"}),
                                  html.Span(str(v), style={"color": cl, "fontWeight": "600"})],
                                 style={"marginRight": "12px", "fontSize": "0.6rem"})
            has_trade = lv.get("entry") is not None
            if r.get("tape_dead"):
                trade = [html.Span(f"🪫 DEAD TAPE (travel {r.get('tape_pct')}% < 0.45%) — stand "
                                   f"down: no new setups; flat days = the measured loss sink "
                                   f"(serial false-breaks + theta)",
                                   style={"color": "#f59e0b", "fontSize": "0.62rem",
                                          "fontWeight": "700"})]
            elif has_trade:
                trade = [_kv("strike", f"{lv.get('atm')} {lv.get('side')}", "#fbbf24"),
                         _kv("entry", lv.get("entry"), "#e2e8f0"),
                         _kv("target", lv.get("target"), "#22c55e"),
                         _kv("SL", lv.get("sl"), "#f87171"), _kv("R:R", lv.get("rr")),
                         html.Span(f"  tape {r.get('tape_pct')}%", style={"color": "#475569",
                                   "fontSize": "0.58rem"})]
            else:
                trade = [html.Span("no directional entry — wait for the break / trade the band",
                                   style={"color": "#64748b", "fontSize": "0.6rem"})]
            _st, _rt = lv.get("sup_touches") or 0, lv.get("res_touches") or 0
            _sup_txt = (f"{lv.get('support')} (×{_st})" if lv.get("support") else "None")
            _res_txt = (f"{lv.get('resistance')} (×{_rt})" if lv.get("resistance") else "None")
            _wall = ([html.Span(
                f"  ⚠ {lv.get('headroom_atr')} ATR to a {lv.get('warn_touches')}-touch "
                f"{lv.get('warn_tf')}m wall @{lv.get('warn_level')} — breaking it or buying "
                f"into it? your call",
                style={"color": "#fbbf24", "fontSize": "0.6rem", "fontWeight": "700"})]
                if lv.get("wall_warn") else [])
            _lsr = []
            if lv.get("sup_l") or lv.get("res_l"):
                _lsr = [_kv("15m S/R",
                        f"{lv.get('sup_l') or '—'}(×{lv.get('sup_l_t') or 0}) / "
                        f"{lv.get('res_l') or '—'}(×{lv.get('res_l_t') or 0})", "#a78bfa")]
            if lv.get("confluence"):
                _lsr.append(html.Span(f"◈ confluence @{', '.join(str(x) for x in lv['confluence'])}",
                            style={"color": "#22d3ee", "fontSize": "0.6rem", "fontWeight": "700",
                                   "marginRight": "10px"}))
            body.append(html.Tr([html.Td([
                _kv("band", f"[{lv.get('band_lo')}, {lv.get('band_hi')}]", "#40c4ff"),
                _kv("60m S/R", f"{_sup_txt} / {_res_txt}", "#4ade80"),
                *_lsr,
                html.Span(" | ", style={"color": "#334155"}), *trade, *_wall,
            ], colSpan=5, style={"padding": "1px 8px 5px 8px"})]))
    return html.Div([
        html.Table([html.Thead(header), html.Tbody(body)],
                   style={"width": "100%", "borderCollapse": "collapse"}),
        html.Div("ranked best-setup-first · band+S/R+levels = the PA edge (from past candle "
                 "closings) · the ATM CE/PE strike is the VEHICLE, flagged: buying the naked "
                 "arrow is MEASURED NEGATIVE-EV (−2 to −5%/trade) — trade the LEVELS with your "
                 "discipline, not the option; overnight = the only validated directional edge",
                 style={"color": "#64748b", "fontSize": "0.56rem", "fontStyle": "italic",
                        "marginTop": "4px"}),
    ], style={"background": "#0a1220", "borderRadius": "8px", "padding": "8px 12px",
              "margin": "4px 0 8px"})


@app.callback(
    Output("tb-combo-ltf", "figure"), Output("tb-combo-htf", "figure"),
    Output("tb-combo-read", "children"),
    Input("sel-sym", "data"), Input("tb-idx-tabs", "active_tab"), Input("tb-combo", "value"),
    Input("news-date", "data"), Input("tb-refresh", "n_intervals"))
def _fill_combo_focus(sel, active_tab, combo, vdate, _tick):
    """The MTF COMBO FOCUS panel — the dropdown-selected combo's LTF (entry) + HTF (confirm)
    charts + its structure/pattern/synthesis read. Context, not a machine trade."""
    from dash.exceptions import PreventUpdate
    if sel != "TRADEBOARD":
        raise PreventUpdate
    import tradeboard as tb
    vdt, now = _tb_clock(vdate)
    sym = active_tab or INDEX_SYMBOLS[0]
    ltf_tf, htf_tf = (int(x) for x in (combo or "5-15").split("-"))
    try:
        htf = tb._struct_full(sym, htf_tf, vdt, now, drop_forming=False)
        ltf = tb._struct_full(sym, ltf_tf, vdt, now)
        s = tb.synthesize(htf, ltf, ltf.get("last") or 0)
    except Exception as exc:
        e = go.Figure(); e.update_layout(template="plotly_dark", height=250,
            paper_bgcolor=BG_CARD, plot_bgcolor=BG_CARD)
        return e, e, html.Div(f"combo error: {exc}", style={"color": "#f87171"})
    fl = _tb_candle_fig(sym, ltf_tf, vdt, now)
    fh = _tb_candle_fig(sym, htf_tf, vdt, now)
    sc = s.get("color", "#94a3b8")
    locpct = f"{100*s['loc']:.0f}%" if s.get("loc") is not None else "—"
    read = html.Div([
        html.Div([
            html.Span(f"{ltf_tf}m (ENTRY) ", style={"color": "#fbbf24", "fontWeight": "800",
                      "fontSize": "0.72rem"}),
            html.Span(f"{ltf.get('struct')} · {ltf.get('pattern')}", style={"color": "#e2e8f0",
                      "fontWeight": "700", "fontSize": "0.72rem"}),
            html.Span("   confirmed by   ", style={"color": "#64748b", "fontSize": "0.62rem"}),
            html.Span(f"{htf_tf}m (CONFIRM) ", style={"color": "#40c4ff", "fontWeight": "800",
                      "fontSize": "0.72rem"}),
            html.Span(f"{htf.get('struct')} · {htf.get('pattern')}", style={"color": "#e2e8f0",
                      "fontWeight": "700", "fontSize": "0.72rem"}),
            html.Span(f"  price in HTF box {locpct}", style={"color": "#64748b",
                      "fontSize": "0.6rem"}),
            html.Span(f"  {s.get('tag','')}", style={"marginLeft": "8px", "color": "#0a0f1a",
                      "background": sc, "borderRadius": "5px", "padding": "2px 10px",
                      "fontWeight": "800", "fontSize": "0.66rem"}),
        ]),
        html.Div(s.get("read", ""), style={"color": "#cbd5e1", "fontSize": "0.7rem",
                 "marginTop": "3px"}),
        html.Div("HTF = confirmation · LTF = entry timing (the entry candle is the trigger) · "
                 "CONTEXT for your PA, not a machine trade", style={"color": "#64748b",
                 "fontSize": "0.56rem", "fontStyle": "italic", "marginTop": "2px"}),
    ], style={"background": "#0a1220", "borderLeft": f"3px solid {sc}", "borderRadius": "8px",
              "padding": "8px 12px", "marginBottom": "6px"})
    return fl, fh, read


@app.callback(
    Output("tb-fig-1d", "figure"), Output("tb-daily", "children"),
    Output("tb-fig-5", "figure"), Output("tb-fig-10", "figure"),
    Output("tb-fig-15", "figure"), Output("tb-fig-30", "figure"),
    Output("tb-fig-60", "figure"),
    Output("tb-intro", "children"), Output("tb-chain", "children"),
    Input("sel-sym", "data"), Input("tb-idx-tabs", "active_tab"),
    Input("news-date", "data"), Input("tb-refresh", "n_intervals"))
def _fill_tradeboard(sel, active_tab, vdate, _tick):
    """Fill the TradeBoard page when it becomes the active section (sel-sym=='TRADEBOARD')
    or the index tab changes. Skips when the page is not open (no wasted render)."""
    from dash.exceptions import PreventUpdate
    if sel != "TRADEBOARD":
        raise PreventUpdate
    import tradeboard as tb
    vdt, now = _tb_clock(vdate)
    sym = active_tab or INDEX_SYMBOLS[0]
    intro = html.Div(
        f"as of {now:%Y-%m-%d %H:%M} IST{' · REPLAY (past day, full session)' if vdt else ' · LIVE'} "
        f"· multi-TF price action + structure + full option chain for "
        f"YOUR read · NO machine trade (the fade edge was a fill artifact — loses at real "
        f"fills) · blue band = risk map (~where price sits) · overnight (BTST) = only "
        f"validated directional edge",
        style={"color": "#94a3b8", "fontSize": "0.7rem", "margin": "8px 0",
               "fontStyle": "italic"})
    try:
        row = tb.build_row(sym, vdt, now)
    except Exception as exc:
        e = go.Figure(); e.update_layout(template="plotly_dark", height=250,
            paper_bgcolor=BG_CARD, plot_bgcolor=BG_CARD)
        return e, html.Div(), e, e, e, e, intro, html.Div(f"row error: {exc}",
            style={"color": "#f87171", "fontSize": "0.75rem"})
    blo, bhi = row.get("band_lo"), row.get("band_hi")
    _blank = go.Figure(); _blank.update_layout(template="plotly_dark", height=250,
        paper_bgcolor=BG_CARD, plot_bgcolor=BG_CARD)

    def _safe(fn, *a, fig=True):
        try:
            return fn(*a)
        except Exception as exc:
            return _blank if fig else html.Div(f"panel error: {exc}",
                style={"color": "#f87171", "fontSize": "0.7rem"})
    dfig = _safe(_tb_daily_fig, sym)
    dpanel = _safe(_tb_daily_panel, row, fig=False)
    figs = [_safe(_tb_candle_fig, sym, tf, vdt, now, blo, bhi) for tf in (5, 10, 15, 30, 60)]
    chain = _safe(_tb_chain_panel, row, fig=False)
    return dfig, dpanel, figs[0], figs[1], figs[2], figs[3], figs[4], intro, chain


# One search box (modal top-right) filters BOTH ledger tables across every column, client
# side (no server round-trip). Empty query restores the full rows from the per-table Stores.
app.clientside_callback(
    """
    function(q, openData, closedData) {
        function filt(rows) {
            if (!rows) return [];
            if (!q) return rows;
            var s = q.toLowerCase();
            return rows.filter(function(r) {
                return Object.keys(r).some(function(k) {
                    var v = r[k];
                    return (v === null || v === undefined ? "" : String(v))
                        .toLowerCase().indexOf(s) !== -1;
                });
            });
        }
        return [filt(openData), filt(closedData)];
    }
    """,
    Output("scout-open-table", "data"),
    Output("scout-closed-table", "data"),
    Input("scout-search", "value"),
    State("scout-open-store", "data"),
    State("scout-closed-store", "data"),
    prevent_initial_call=True,
)


@app.callback(Output("charts-help-box", "children"), Output("charts-help-title", "children"),
              Input("charts-mode", "value"))
def _swap_charts_help(mode):
    """Show the help that matches the selected mode (options vs futures), and label
    the popup title with the mode so it's clear which help you're reading."""
    if mode == "futures":
        return _charts_help("futures"), "🛢 Futures — what is this · how to read"
    return _charts_help("options"), "⚙ Options flow — what is this · how to read"


@app.callback(
    Output("charts-graph", "figure"),
    Input("charts-mode",   "value"),
    Input("charts-leg",    "value"),
    Input("charts-strike", "value"),
    Input("charts-expiry", "value"),
    Input("charts-idx",    "value"),
    Input("charts-tf",     "value"),
    Input("charts-asof",   "value"),
    Input("news-date",     "data"),
    Input("sel-sym",       "data"),
    Input("setup-tick",    "n_intervals"),   # live refresh (uirevision keeps zoom/drawn lines)
)
def _update_charts(mode, leg, strike, expiry, sym, tf, asof, date, sel, _tick):
    """Redraw when mode/leg/strike/expiry/index/tf/as-of/date changes or section opens.
    as_of (replay): truncate every chart at the chosen time — leakage-safe (read_mirror)."""
    from dash.exceptions import PreventUpdate
    if sel != "CHARTS":
        raise PreventUpdate
    sym, tf, date = sym or "NSE:NIFTY50-INDEX", int(tf or 15), date or None
    expiry = expiry or "weekly"
    if asof == "ghost":                        # practice: pin to the ghost clock/day
        date, asof = _ghost_ctx(date)          # (else empty date → full day = future leak)
    asof_iso = f"{date}T{asof}:00+05:30" if (asof and asof != "full" and date) else None
    if mode == "futures":
        return _futures_fig(sym, tf, asof_value=asof_iso, date=date, leg=leg or "near")
    if strike and strike != "totals":
        try:
            return _strike_fig(sym, tf, int(strike), asof_value=asof_iso, date=date, expiry=expiry)
        except (ValueError, TypeError):
            pass
    return _footprint_fig(sym, tf, asof_value=asof_iso, date=date, expiry=expiry)


# ── Charts: overnight → now per-strike positioning map (DESCRIPTIVE) ───────────
# Anchors today's live per-strike OI change (oich = Δ vs prev EOD close) against
# last night's EOD per-strike OI baseline from DCM fno_bhavcopy, and labels each
# positioned strike REINFORCED (held) vs ABANDONED (covering/unwind). This is the
# view the user asked for. It is CONTEXT, never a buy/sell call — the directional
# edge is unproven (backtest_reconciliation.py CIs all straddle the null on the
# captured days). Reads the lock-free parquet mirrors at the chosen as_of, so it
# is leakage-safe and works for both live and replay.
_RECON_CLR = {"bull": "#22c55e", "bear": "#ef4444", "neut": "#94a3b8"}


def _charts_recon_panel(sym, date, as_of_dt):
    """Render the overnight→now reconciliation read for `sym` at `as_of_dt`, or a
    quiet warming-up note. Pure read (mirrors + DCM read-only) — no writes."""
    from core.mirror_io import read_mirror
    from core.constants import NSE_NAME
    import overnight_reconciliation as orc

    date = date or datetime.date.today().isoformat()

    def _num(v):
        try:
            return float(v) if pd.notna(v) else 0.0
        except Exception:
            return 0.0

    # ── live/replay strike_map: latest oich/ltpch/oi per (strike, side) ≤ as_of ──
    chain = read_mirror("chain_snapshots", date, as_of_dt, sym)
    if chain is None or "oich" not in chain.columns:
        return _recon_note("Positioning map warming up — option-chain capture not in yet.")
    last = chain.sort_values("ts").groupby(["strike", "side"]).last().reset_index()
    sm: dict = {}
    for _, r in last.iterrows():
        sm.setdefault(float(r["strike"]), {})[r["side"]] = {
            "oich": _num(r.get("oich")), "ltpch": _num(r.get("ltpch")),
            "oi": _num(r.get("oi")), "volume": _num(r.get("volume")),
            "delta": (None if pd.isna(r.get("delta")) else _num(r.get("delta")))
                     if "delta" in last.columns else None}

    ticks = read_mirror("ticks", date, as_of_dt, sym)
    if ticks is None or not len(ticks):
        return _recon_note("Positioning map warming up — no ticks yet.")
    spot   = _num(ticks["ltp"].iloc[-1])
    anchor = _num(ticks["day_open"].iloc[-1]) if "day_open" in ticks.columns else _num(ticks["ltp"].iloc[0])
    # open gap vs prev close (ch = Δ vs prev close) — gates the regime-shift flag
    prev_close = spot - _num(ticks["ch"].iloc[-1]) if "ch" in ticks.columns else 0.0
    gap = (anchor / prev_close - 1.0) if (anchor and prev_close) else 0.0
    # index points move vs prev close — same anchor as per-strike ltpch; drives the
    # delta-adjusted residual that splits each OI move writer- vs buyer-driven.
    index_chg = (spot - prev_close) if prev_close else 0.0

    # ── frozen EOD baseline from DCM (strictly before the selected day) ──────────
    try:
        import duckdb
        from eod_oi_range import DCM_DB
        if not DCM_DB.exists():
            return _recon_note("EOD baseline unavailable — Daily_Cash_Market DB not found.")
        prior = datetime.date.fromisoformat(date) - datetime.timedelta(days=1)
        con = duckdb.connect(str(DCM_DB), read_only=True)
        try:
            baseline = orc.eod_baseline(con, NSE_NAME[sym], as_of_date=prior)
        finally:
            con.close()
    except Exception:
        return _recon_note("EOD baseline read failed.")
    if not baseline:
        return _recon_note("No EOD positioning recorded for this index yet.")

    r = orc.analyze_reconciliation(sm, spot, baseline, gap=gap, vwap=anchor,
                                   index_chg=index_chg)
    if not r.has_data:
        return _recon_note(r.note or "No positioned strikes in range / no action yet.")

    lines = orc.summary_lines(r)
    head_bias, head_txt = lines[0]
    body = []
    for bias, txt in lines[1:]:
        body.append(html.Div(txt, style={
            **MONO, "fontSize": "0.66rem", "padding": "2px 0",
            "color": _RECON_CLR.get(bias, "#94a3b8")}))
    shift = (html.Span("  ⚡ REGIME SHIFT", style={"color": "#fbbf24", "fontWeight": "700"})
             if r.regime_shift else "")
    return html.Div([
        html.Div([
            html.Span("🧭 OVERNIGHT → NOW POSITIONING", style={
                "color": "#a78bfa", "fontWeight": "700", "fontSize": "0.62rem",
                "letterSpacing": "0.06em"}),
            html.Span("  ·  context, not a signal", style={
                "color": "#475569", "fontSize": "0.58rem"}),
            shift,
        ], style={"marginBottom": "4px"}),
        html.Div(head_txt, style={
            **MONO, "fontSize": "0.7rem", "fontWeight": "700", "paddingBottom": "3px",
            "color": _RECON_CLR.get(head_bias, "#94a3b8")}),
        html.Div(body),
        html.Div("EOD-anchored per-strike OI (oich = Δ vs prev close) vs last night's "
                 "DCM baseline. Reinforced = position held, abandoned = covering/unwind; "
                 "the covering/writing vs unwind/buying tag is the delta-adjusted "
                 "premium residual (writer- vs buyer-driven) and weights conviction. "
                 "Directional edge unproven — read as positioning structure, not a trade call.",
                 style={"color": "#475569", "fontSize": "0.55rem", "marginTop": "5px",
                        "fontStyle": "italic", "lineHeight": "1.3"}),
    ], style={"background": "#0a1020", "border": "1px solid #1e293b",
              "borderRadius": "6px", "padding": "8px 12px"})


def _recon_note(msg):
    return html.Div(msg, style={**MONO, "fontSize": "0.6rem", "color": "#475569",
                                "padding": "6px 12px", "fontStyle": "italic"})


# ── SCOUT panel: multi-index TRADE/NO-TRADE scan ─────────────────────────────────
# Hover-tooltip copy (native `title`) so the panel explains itself.

def _mem_open_age_min(op, today, as_of=None):
    """Minutes an open poller position has been held (from its 'trig' HH:MM). None if unknown.

    `as_of` is REQUIRED by any caller that is not on the wall clock. _scout_detect runs both
    live (as_of=now) AND over a historical grid in _backfill_scout_alerts, where measuring
    against datetime.now() makes every reconstructed position read as hours old -> the 90m cap
    fires instantly on the morning replay, TIMEOUTs the whole day and (with the re-arm lock)
    bolts every side shut. Same wall-clock-vs-as-of trap as the replay bug in 1911c85."""
    try:
        h, m = str(op.get("trig") or "").split(":")
        opened = datetime.datetime(*map(int, today.split("-")), int(h), int(m), tzinfo=IST)
        ref = as_of or datetime.datetime.now(IST)
        return max(0.0, (ref - opened).total_seconds() / 60.0)
    except Exception:
        return None


def _leg_age_min(trig, today):
    """Minutes since the live scan's current TRADE leg began ('HH:MM'). None if unknown."""
    try:
        h, m = str(trig or "").split(":")
        started = datetime.datetime(*map(int, today.split("-")), int(h), int(m), tzinfo=IST)
        return max(0.0, (datetime.datetime.now(IST) - started).total_seconds() / 60.0)
    except Exception:
        return None


def _ledger_mem(today, as_of):
    """Per-symbol trade memory built from the AUTHORITATIVE poller log (the same episodes the
    ledger popup and the open/closed badge read), shaped like the browser's `seen` state.

    WHY THIS EXISTS. The strip used to render its HOLDING line from `seen` -- the browser's
    OWN copy of the detector (persist=False). That is a SECOND ENGINE, and it drifts: on
    2026-07-14 the board said NIFTY "HOLDING 24100 PE since 12:13 ... held 81m/90m" while the
    poller log -- the thing the ledger, the badge and the P&L all read -- said the leg opened
    12:09 and was 86m old. Same position, two truths, and the one on screen was the wrong one.

    The drift is not cosmetic: the 90m CAP is measured off this timestamp. A 4-minute error
    means a 4-minute window in which the board shows HOLDING for a leg the ledger has already
    TIMED OUT -- exactly the badge-vs-popup contradiction, one layer down. One source of truth
    for "what is held": the poller's log."""
    op, cl = _scout_episodes(today, as_of=as_of)
    mem: dict = {}
    for e in op:
        mem.setdefault(e["sym"], {})["open"] = {
            "day": today, "trig": e.get("open_t"), "dir": e.get("dir"),
            "strike": e.get("strike"), "entry": e.get("entry"),
            "sl": e.get("sl"), "tgt": e.get("tgt"), "bb": bool(e.get("bb"))}
    for e in cl:                       # cl is newest-first -> first hit per sym is the latest
        m = mem.setdefault(e["sym"], {})
        if "open" in m or "last" in m:
            continue
        m["last"] = {"day": today, "dir": e.get("dir"), "strike": e.get("strike"),
                     "entry": e.get("entry"), "cur": e.get("exit"),
                     "outcome": e.get("outcome"), "closed_t": e.get("close_t"),
                     "trig": e.get("open_t"), "band_broke": bool(e.get("bb"))}
    return mem


def _scout_mem_block(mem, today):
    """Persistent trade memory for one index, drawn from the live alert brain
    (scout-seen). Two cases the stateless per-bar scan can't show on its own:
      • HOLDING — a trade is open but the gate is momentarily NO-TRADE on a forming bar
        (it would otherwise vanish mid-hold). Keep it on the board.
      • LAST    — the trade already RESOLVED (SL / target / flip). Instead of the row
        going silent, show what happened, so a trade you took never just disappears.
    Live-only overlay; ignored for replay / past days."""
    if not mem:
        return None
    op = mem.get("open")
    if op and op.get("day") == today:
        side = op.get("dir"); strike = op.get("strike")
        # The POLLER holds a position indefinitely (it only closes on SL/target/band/flip), so
        # without this it kept showing "📌 HOLDING since 10:33" FOUR HOURS into a 90-minute
        # rule — the board contradicting the ledger, which had already timed it out. The 90m
        # cap is the POLICY; every surface must honour it. Display-side: the raw poller stream
        # is untouched (the ledger reinterprets it the same way).
        _age = _mem_open_age_min(op, today)
        if _age is not None and _age >= _SCOUT_MAX_HOLD_MIN:
            seg = (f"⌛ TIMED OUT ({_SCOUT_MAX_HOLD_MIN}m) — {strike or ''} {side} opened "
                   f"{op.get('trig','?')}, held {_age:.0f}m with no SL/target/band/flip. "
                   f"Position is CLOSED per the max-hold rule (the poller still tracks it).")
            return html.Div(seg, style={**MONO, "fontSize": "0.6rem", "color": "#a78bfa",
                            "fontWeight": "700", "lineHeight": "1.4",
                            "background": "#171526", "borderRadius": "3px",
                            "padding": "2px 6px", "marginLeft": "120px", "marginTop": "2px"})
        _held = f"  ·  held {_age:.0f}m/{_SCOUT_MAX_HOLD_MIN}m" if _age is not None else ""
        seg = (f"📌 HOLDING {strike or ''} {side} since {op.get('trig','?')}{_held}"
               f"  ·  SL ₹{op.get('sl')}  T ₹{op.get('tgt')}"
               f"  ·  gate flickering NO-TRADE on the forming bar — position still live")
        if op.get("bb"):
            seg += "  ·  ⚠ band already broke (move > expected)"
        return html.Div(seg, style={**MONO, "fontSize": "0.6rem", "color": "#fbbf24",
                        "fontWeight": "700", "paddingLeft": "120px", "lineHeight": "1.4",
                        "background": "#15110a", "borderRadius": "3px",
                        "padding": "2px 6px", "marginLeft": "120px", "marginTop": "2px"})
    last = mem.get("last")
    if last and last.get("day") == today:
        oc = last.get("outcome")
        face = {"SL": ("🛑", "SL hit", "#ef4444"),
                "TARGET": ("🎯", "target booked", "#22c55e"),
                "FLIP": ("↔", "closed / flipped side", "#94a3b8")}.get(
                    oc, ("•", "closed", "#94a3b8"))
        emo, word, clr = face
        side = last.get("dir"); strike = last.get("strike")
        seg = (f"⟲ last trade: {strike or ''} {side} from {last.get('trig','?')} "
               f"→ {emo} {word} {last.get('closed_t','')}"
               f"  (entry ₹{last.get('entry')} → ₹{last.get('cur')})")
        if last.get("band_broke"):
            seg += "  ·  band broke"
        return html.Div(seg, style={**MONO, "fontSize": "0.6rem", "color": clr,
                        "fontWeight": "700", "paddingLeft": "120px", "lineHeight": "1.4",
                        "background": "#0a1422", "borderRadius": "3px",
                        "padding": "2px 6px", "marginLeft": "120px", "marginTop": "2px"})
    return None


def _scout_playbook(r):
    """Click-to-open scrollable trade playbook for one scout row, generated from THIS
    row's live numbers. Honest-quant content: the RANGE band is the validated edge
    (~77% close / ~52% path, measured), the CE/PE arrow is negative-EV context — so the
    primary plan is range-based, with an explicit walk-through of the band-break case
    (you bought the arrow, price broke the band against you → thesis dead → exit)."""
    lo, hi = r.get("range_lo"), r.get("range_hi")
    spot = r.get("spot")
    side = r.get("direction")                       # CE / PE / None
    lc = r.get("lifecycle") or {}
    strike = lc.get("entry_strike") or r.get("atm")
    entry_prem = lc.get("entry_prem"); sl_prem = lc.get("sl"); tgt_prem = lc.get("target")
    thin = r.get("thin")
    center = round((lo + hi) / 2, 1) if (lo is not None and hi is not None) else spot
    half = round((hi - lo) / 2, 1) if (lo is not None and hi is not None) else None
    bull = side == "CE"
    # the band edge that INVALIDATES a directional bet (CE dies on a lower break, PE on upper)
    inval_edge = lo if bull else hi
    fav_edge   = hi if bull else lo

    def line(txt, color="#cbd5e1", bold=False, pad=0, size="0.6rem"):
        return html.Div(txt, style={**MONO, "fontSize": size, "color": color,
                                    "fontWeight": "700" if bold else "400",
                                    "whiteSpace": "pre-wrap", "lineHeight": "1.45",
                                    "paddingLeft": f"{pad}px", "marginBottom": "2px"})

    def hdr(txt, color="#fbbf24"):
        return html.Div(txt, style={**MONO, "fontSize": "0.63rem", "color": color,
                                    "fontWeight": "800", "marginTop": "7px",
                                    "marginBottom": "3px", "letterSpacing": "0.03em"})

    op = r.get("opening") or {}
    phase = op.get("phase")

    body = []
    # ── 0. opening / data-warmup regime ──────────────────────────────────────────
    if op.get("warming") or phase in ("OPENING", "SETTLING"):
        body.append(hdr("⓪ OPENING / WARMUP — READ THIS FIRST", "#fbbf24"))
        if op.get("warming") and op.get("data_start"):
            ra = op.get("ready_at")
            body.append(line(
                f"DATA WARMUP: live feed running since {op['data_start']}; the scout needs "
                f"~{20}m of data before it trades"
                + (f", so it starts calling trades from ~{ra:%H:%M}." if ra else ".")
                + " If the feed dropped earlier and resumed, this clock re-anchored to the "
                "resume — not 09:15. No trade until then (signals on thin data are noise).",
                "#fbbf24"))
        gp = op.get("gap_pct"); gt = op.get("gap_type", "open")
        body.append(line(
            f"It is the opening window. Today is a {gt}"
            + (f" ({gp:+.2f}% vs prev close {op.get('prev_close')})." if gp is not None else ".")
            + (f" Opening range (first 15m) = [{op.get('or_lo')}, {op.get('or_hi')}]."
               if op.get('or_lo') is not None else "")))
        if phase == "OPENING":
            body.append(line(
                "RULE: NO trade yet. 09:15-09:35 is gap/settle noise — the engine "
                "SUPPRESSES the arrow here and the range/backtest edge is only validated "
                "from ~09:45. Let the opening range (OR) form first.", "#fbbf24"))
        else:
            body.append(line(
                "RULE: provisional. Edge validated from ~09:45 — half size, and only "
                "trade the OR break/fade, not a mid-range guess.", "#60a5fa"))
        ob = op.get("oi_build")
        if ob:
            def _L(x): return f"{x/1e5:+.0f}L" if x else "0"
            ce_w = ", ".join(str(s) for s, _ in (ob.get("ce_walls") or [])) or "—"
            pe_f = ", ".join(str(s) for s, _ in (ob.get("pe_floors") or [])) or "—"
            body.append(line(
                f"OPENING BOOK (OI vs prev close, first 20m):\n"
                f"   • CE OI {_L(ob['ce_oich'])}  (building at {ce_w}) ← ceilings/walls\n"
                f"   • PE OI {_L(ob['pe_oich'])}  (building at {pe_f}) ← floors/supports\n"
                f"   • Volume CE {ob['ce_vol']/1e5:.0f}L vs PE {ob['pe_vol']/1e5:.0f}L. "
                f"This is RAW positioning — the scout flow signal reads buy-vs-write; "
                f"the walls/floors are your near-term S/R levels.", "#cbd5e1"))
        body.append(line(
            "HOW TO PLAY THE OPEN (no validated gap edge — geometry only):\n"
            "   • Wait for the 15-min opening range (OR) to set (09:15-09:30).\n"
            "   • OR BREAK: a 5-min CLOSE above OR-high → lean long; below OR-low → "
            "lean short. Stop = the other side of the OR.\n"
            "   • OR FADE (gap into a level): sharp gap that stalls and re-enters the OR "
            "→ fade back toward prev close / VWAP. Stop = the gap extreme.\n"
            "   • Sharp gap = wider stops, smaller size. Range-bound open (tight OR) = "
            "wait for the break, don't pre-position.", "#94a3b8"))
    # ── 1. what this is ──────────────────────────────────────────────────────────
    body.append(hdr("① WHAT THIS ROW IS", "#34d399"))
    if side:
        body.append(line(
            f"The board leans {('UP / CALL' if bull else 'DOWN / PUT')} on {r['label']} "
            f"(structural flow + corroboration). This LEAN is decision-support ONLY — "
            f"backtested, buying the naked {side} off the arrow is NEGATIVE-EV "
            f"(wins ~14-23%, bleeds -2..-5%/trade). Do NOT trade the arrow as a signal."))
    else:
        body.append(line(
            f"NO clean directional setup on {r['label']} right now (families disagree). "
            f"That's fine — the tradeable product here is the RANGE, not a direction."))
    if lo is not None:
        body.append(line(
            f"The VALIDATED product is the 60-min range cone [{lo}, {hi}] "
            f"(center ~{center}, ±{half} pts). MEASURED: price CLOSES inside ~77% of the "
            f"time, but STAYS inside the whole hour only ~52% — it wicks out about half "
            f"the time then closes back in. Treat it as a ~1σ DESTINATION, not a wall."))
    else:
        body.append(line("Range cone still warming up (needs ~12 one-minute bars). "
                         "No trade until the band prints.", "#94a3b8"))

    # ── reading the live grade line (the "↪ live now · next 60m ..." strip row) ──
    plo, phi = r.get("pred_lo"), r.get("pred_hi")
    if plo is not None:
        hz = r.get("horizon", 60)
        pdir = r.get("pred_dir"); ptgt = r.get("pred_target")
        cov = r.get("band_cover"); bn = r.get("band_n", 0)
        v = r.get("verify") or {}
        body.append(hdr("📈 READING THE '↪ live now' LINE"))
        body.append(line(
            f"That grading row under the headline decodes token by token:\n"
            f"   • next {hz}m: {pdir or 'RANGE'}"
            + (f" → {ptgt}" if ptgt else "")
            + "  — the arrow's LEAN (+ target). CONTEXT ONLY, neg-EV — don't trade it.\n"
            f"   • band[{plo}, {phi}]  — the VALIDATED product: where price should sit "
            f"over the next {hz}m (~1σ cone, center ~{center}).\n"
            + (f"   • cover {cov*100:.0f}%  — MEASURED share of past days this exact "
               f"cell's band held (n{bn} samples). ✓ = healthy (~68%+), ~ = soft.\n"
               if cov is not None else "")
            + f"   • ⇒ actual (move%)  — what price ACTUALLY did once the {hz}m elapsed "
            "(the answer key; fills in after the fact, never feeds the call).\n"
            "   • HIT / MISS  = did the ARROW's direction land — IGNORE, it's a coin flip.\n"
            "   • band ✓ / ✗  = did price finish INSIDE the band. THIS is the score that "
            "matters — the band is the whole product."))
        if v.get("actual") is not None:
            bh = v.get("band_hit"); dh = v.get("dir_hit")
            body.append(line(
                f"THIS row, graded: price ended {v['actual']} ({v.get('move_pct', 0):+.2f}%) — "
                f"band {'HIT ✓ — stayed inside the cone' if bh else 'MISS ✗ — broke out'}; "
                f"arrow {'hit' if dh else 'missed'} (noise either way). "
                f"Read the band mark, not the arrow.", "#94a3b8"))
        else:
            body.append(line(
                f"THIS row is still LIVE — the {hz}m has not elapsed, so 'actual' and the "
                f"band ✓/✗ show PENDING; they self-grade once the hour completes.",
                "#94a3b8"))

    # ── 1. the trade I'd actually take ───────────────────────────────────────────
    if lo is not None:
        body.append(hdr("② THE TRADE (range-based — the honest edge)"))
        body.append(line(
            f"A) RANGE FADE (primary). When price pushes to a band EDGE, it reverts to "
            f"center ~77% of closes. So:\n"
            f"   • Near UPPER {hi}: fade DOWN — buy a PE / sell a CE-spread, target "
            f"center {center}.\n"
            f"   • Near LOWER {lo}: fade UP — buy a CE / sell a PE-spread, target "
            f"center {center}.\n"
            f"   • Mid-band (near {center}): NO trade — no edge in the middle, you just "
            f"pay theta."))
        if side:
            body.append(line(
                f"B) IF you take the {side} lean anyway (small size only — it's neg-EV): "
                f"enter only near the {('LOWER' if bull else 'UPPER')} edge "
                f"({inval_edge}) so the band is WITH you, never chase mid-band. Your hard "
                f"line in the sand is the OPPOSITE edge ({inval_edge})."))

    # ── 2. stop-loss ─────────────────────────────────────────────────────────────
    body.append(hdr("③ STOP-LOSS (two stops — whichever hits FIRST)"))
    body.append(line(
        "1) INDEX stop = a 5-MINUTE CLOSE beyond the band edge against you. Use a "
        "5-min CLOSE, NOT a single tick — price wicks past the band ~half the time "
        "(path 52%) and snaps back; only a confirmed close beyond it is a real break."))
    if entry_prem and sl_prem:
        body.append(line(
            f"2) PREMIUM stop = ₹{sl_prem} on the {strike} {side} "
            f"(entry ₹{entry_prem}, ~-{round((1-sl_prem/entry_prem)*100)}%). A ~1σ "
            f"adverse index move ≈ this premium stop, so the two usually fire together."))
    else:
        body.append(line(
            "2) PREMIUM stop = ~30-35% of the option premium you paid (the engine sets "
            "this on the ATM once you're in a live trade)."))

    # ── 3. target ────────────────────────────────────────────────────────────────
    if lo is not None:
        body.append(hdr("④ TARGET"))
        body.append(line(
            f"• Range fade: take profit at center {center} (first), runner to the "
            f"opposite edge.\n"
            f"• Directional: the band edge in your favor "
            f"({fav_edge if side else 'the far edge'})"
            + (f", premium ₹{tgt_prem}." if tgt_prem else ".")
            + " Book at the edge — past ~1σ the odds flip against you."))

    # ── 4. THE BAND-BREAK SCENARIO (the user's exact question) ───────────────────
    if side and inval_edge is not None:
        body.append(hdr("⑤ WHAT IF THE BAND BREAKS AGAINST YOU?", "#f87171"))
        brk = "LOWER" if bull else "UPPER"
        body.append(line(
            f"You bought the {strike} {side} (betting {('UP' if bull else 'DOWN')}), "
            f"but the index breaks the {brk} band ({inval_edge}) and CLOSES a 5-min "
            f"candle beyond it. Read it straight:\n"
            f"   • BOTH halves of the thesis just broke: direction inverted (you wanted "
            f"{('up' if bull else 'down')}) AND realized vol exceeded the ~1σ cone.\n"
            f"   • You are now in the ~tail where the move tends to CONTINUE, not "
            f"revert. Your {side} premium is already at/under the ₹"
            f"{sl_prem or 'stop'} stop (a 1σ adverse move guts an ATM option fast).\n"
            f"   • ACTION: EXIT NOW. Yes — stop-loss hits, close the position. Do NOT "
            f"average down, do NOT 'wait for it to come back'. A confirmed band break is "
            f"the definition of your thesis being wrong.\n"
            f"   • The alerts tab fires 🛑 STOP-LOSS and 📊 RANGE BREAK on exactly this — "
            f"that's your cue to be already out."))
        body.append(line(
            f"WICK vs BREAK: a single tick poking below {inval_edge} that snaps back "
            f"inside the SAME 5-min candle is just noise (the 52% path-breach) — hold. "
            f"A 5-min CLOSE beyond it is the real break — exit. That distinction is the "
            f"whole game.", "#94a3b8"))

    # ── 5. reality check ─────────────────────────────────────────────────────────
    body.append(hdr("⑥ SIZING & REALITY CHECK", "#94a3b8"))
    body.append(line(
        "• The arrow is neg-EV → the range fade is the edge, not the direction.\n"
        "• Coverage is ~77% CLOSE / ~52% PATH on n=236, 10 days (CI 72-82%) — small "
        "sample, per-day swings 62-100%. Don't size as if 77% is a floor.\n"
        "• Risk <=1% of capital per trade; the band gives you the geometry, not a "
        "guarantee." + ("\n• ⚠ THIN index — sparse OI, stale strike prices, every "
        "number here is less reliable. Halve size or skip." if thin else ""), "#94a3b8"))

    return html.Details([
        html.Summary("📖 how to trade this  ·  click", style={
            **MONO, "fontSize": "0.6rem", "color": "#fbbf24", "cursor": "pointer",
            "fontWeight": "700", "padding": "3px 8px", "background": "#1a1407",
            "border": "1px solid #7c5e10", "borderRadius": "4px",
            "display": "inline-block", "listStyle": "none", "userSelect": "none"}),
        html.Div(body, style={
            "maxHeight": "300px", "overflowY": "auto", "marginTop": "5px",
            "padding": "7px 10px", "background": "#060a12",
            "border": "1px solid #1e293b", "borderRadius": "5px"}),
    ], style={"marginLeft": "120px", "marginTop": "4px"})


def _ghost_help():
    """Click-to-open guide for ghost practice — how the mode works and how to use it."""
    _ln = lambda t, c="#94a3b8": html.Div(t, style={
        **MONO, "fontSize": "0.6rem", "color": c, "lineHeight": "1.55"})
    return html.Details([
        html.Summary("👻 how ghost practice works  ·  click", style={
            **MONO, "fontSize": "0.6rem", "color": "#a78bfa", "cursor": "pointer",
            "fontWeight": "700", "padding": "3px 8px", "background": "#0f0a1a",
            "border": "1px solid #4c1d95", "borderRadius": "4px",
            "display": "inline-block", "listStyle": "none", "userSelect": "none"}),
        html.Div([
            _ln("WHAT: a past captured session replayed on TODAY'S wall clock — at your "
                "real 13:29 you see that day's 13:29. A time machine for practice.", "#c4b5fd"),
            _ln("• Advances by itself every ~30s, exactly like a live feed. No clicking."),
            _ln("• The future is SEALED — actual/HIT/scoreboard are hidden until the real "
                "15:30, so you cannot peek. Write your calls down; the day grades you at "
                "the close."),
            _ln("• PICK A DAY: scroll the date strip (bottom bar ◀ ▶) to any captured "
                "session, keep the Replay picker on 👻. Untouched = the last session."),
            _ln("• You join the day at your CURRENT clock time — start at 14:00 and the "
                "morning is already history. Full-day practice = start at 09:15."),
            _ln("• ALERTS replay too: each alert fires at the minute it really fired that "
                "day (bell badge counts up with your clock)."),
            _ln("• The cockpit + left FOOTPRINT sidebar follow the ghost clock too "
                "(tagged 👻). Only the TOP TICKER stays live — ignore its weekend "
                "numbers.", "#fbbf24"),
            _ln("• Nothing is written anywhere — no cleanup, the next live session is "
                "untouched. Weekends/holidays open in ghost automatically; a trading "
                "day always boots LIVE."),
            _ln("• STUDY instead of practice? Pick an exact minute (e.g. 10:00) in the "
                "same Replay picker — that scrubs freely and grades instantly."),
        ], style={"maxHeight": "300px", "overflowY": "auto", "marginTop": "5px",
                  "padding": "7px 10px", "background": "#060a12",
                  "border": "1px solid #312e81", "borderRadius": "5px"}),
    ], style={"marginTop": "6px"})


def _scout_dte(sym, today):
    """Days-to-expiry for the theta-cliff badge — thin wrapper over the canonical
    market_calendar.days_to_expiry (weekly for NIFTY, monthly for the rest). The 9-session
    scenario map shows option-net −26% mean at 0-DTE (NIFTY-Tue −45%) → an AVOID flag."""
    from core import market_calendar as _mc
    from core.constants import NIFTY as _NIFTY
    day = today or datetime.datetime.now(IST).date()
    return _mc.days_to_expiry(day, weekly=(sym == _NIFTY))


def _session_over(live: bool) -> bool:
    """Is the cash session finished? The arrow is an INTRADAY product — it is flat overnight,
    always. Without this the board kept rendering a live TRADE leg with a running age long
    after 15:30 (BANK read 'leg since 15:15 ... held 435m' at 22:26), which looks exactly like
    a held position and contradicts the ledger's '0 open'. LIVE-only: replay/ghost drive their
    own clock and must not be judged against wall-time."""
    if not live:
        return False
    now = datetime.datetime.now(IST)
    from core.market_calendar import is_trading_day
    return is_trading_day(now.date()) and now.time() > datetime.time(15, 30)


def _scout_row(r, mem=None, today=None, live=True, practice=False, session_over=False):
    """One index row in the scout strip. `mem` = this index's persistent trade memory
    (scout-seen, live-only) so a held trade survives a NO-TRADE blink and a resolved
    trade shows its outcome instead of vanishing."""
    if not r.get("has_data"):
        return html.Div(f"{r['label']}: {r.get('note','warming up')}",
                        style={**MONO, "fontSize": "0.6rem", "color": "#475569",
                               "padding": "3px 10px"})
    trade = r["verdict"].startswith("TRADE")
    side  = r.get("direction")
    clr   = ("#34d399" if side == "CE" else "#f87171") if trade else "#64748b"
    bg    = "#0c1f17" if (trade and side == "CE") else "#1f0c0c" if (trade and side == "PE") else "#0a1020"
    # For an OPEN trade show the strike you actually HOLD (ATM at trigger), not the
    # drifting current ATM — they differ when the index moved since entry, which was
    # confusing (headline 14500 vs entry 14525). Fresh/no-trade rows show current ATM.
    lc = r.get("lifecycle")
    if lc and lc.get("entry_strike"):
        inst = f" · {lc['entry_strike']} {side} (this leg since {lc['trigger']})"
    else:
        inst = (f" · {r['instrument']}" if r.get("instrument") else "")
    # expiry label on EVERY strike-bearing row (holding OR trade), not just TRADE — a
    # trader must know weekly (0-3 DTE) vs monthly + the actual date to judge theta/liquidity.
    if inst and r.get("expiry"):
        _exp = r["expiry"] + (f" {r['expiry_date']}" if r.get("expiry_date") else "")
        inst += f" · {_exp} exp"
    if r.get("thin"):
        inst += "  ⚠ thin"
    # THETA-CLIFF badge — buying the arrow near expiry is the worst measured scenario
    # (9-session DTE map: option-net −26% @0-DTE, NIFTY-Tue weekly −45%; theta ~1/√T eats
    # the ATM premium). Flag it at the decision point on every strike-bearing row.
    theta_badge = None
    if inst and r.get("sym"):
        _dte = _scout_dte(r["sym"], today)
        if _dte == 0:
            # NIFTY was referenced BARE here and is not imported at module scope (the only
            # import is local, inside _scout_dte). This branch runs ONLY on 0-DTE, i.e. NIFTY's
            # weekly expiry -- every TUESDAY -- so the whole scout panel raised NameError and
            # died the moment NIFTY carried a TRADE verdict on an expiry day. Latent since
            # 5fcfeab, and it fired today (Tue 2026-07-14).
            from core.constants import NIFTY as _NIFTY_SYM
            _is_nifty = r["sym"] == _NIFTY_SYM
            _txt = ("⚠ EXPIRY DAY · theta cliff" +
                    (" (NIFTY-Tue: arrow ≈ −45% hist)" if _is_nifty
                     else " (arrow ≈ −26% hist)"))
            theta_badge = html.Span("  " + _txt,
                title="Days-to-expiry = 0. Buying the ATM arrow on expiry day is the worst "
                      "scenario in the 9-session scenario map: option-net −26% mean overall, "
                      "−45% for NIFTY on its weekly Tuesday. Theta (~1/√T) evaporates the "
                      "premium; the '+30% band' win here is the tiny-premium mirage. AVOID.",
                style={"color": "#f87171", "fontWeight": "700", "cursor": "help"})
        elif _dte == 1:
            theta_badge = html.Span("  ⚠ 1d to expiry · theta risk",
                title="1 day to expiry — accelerating theta decay on the ATM arrow "
                      "(option-net −8.5% mean in the scenario map). Size down / avoid.",
                style={"color": "#f59e0b", "fontWeight": "700", "cursor": "help"})
    # REVERSAL-ACCUMULATION flag — writers secretly closing shorts (OI↓ prem↑) + fresh
    # buying = a positioning flip that can precede a move (gamma-squeeze near expiry). Rare
    # by design (~1/36 day-cells); DISPLAY CONTEXT ONLY, never a trade trigger (cheap = the
    # −45% theta cliff; a false fire is a lottery-ticket loss). Live only — the footprint
    # reads the latest snapshot, so a replay would leak; skip it off-live.
    rev_badge = None
    if live and r.get("sym"):
        try:
            import smart_money as _sm
            _rev = _sm.reversal_signal(r["sym"], date=today,
                                       dte=_scout_dte(r["sym"], today))
        except Exception:
            _rev = None
        if _rev and _rev.get("fired"):
            _rc = "#34d399" if _rev.get("side") == "bullish" else "#f87171"
            rev_badge = html.Span("  ⚡ REVERSAL · " + _rev.get("note", ""),
                title="Smart-money REVERSAL ACCUMULATION: writers are net short-COVERING "
                      "(OI down, premium up) on one side — a positioning flip that can "
                      "precede a move (gamma-squeeze / pin-break near expiry). CONTEXT ONLY, "
                      "NOT a trade signal — this is unvalidated (grade via backtest_reversal "
                      "as expiry days accrue), and near expiry a false fire is the −45% "
                      "theta cliff. Reads the day-level footprint (net vs prev close).",
                style={"color": _rc, "fontWeight": "700", "cursor": "help"})
    _ul = {"textDecoration": "underline dotted", "cursor": "help"}
    metrics = [
        html.Span(f"str {r['strength']:+.2f}  ", title=_TIP_STR,
                  style={"color": "#94a3b8", **_ul}),
        html.Span(f"agree {r['agree']}/{r['active']}  ", title=_TIP_AGREE,
                  style={"color": "#94a3b8", **_ul}),
        html.Span(f"conf {r['confidence']}%", title=_TIP_STR, style={"color": "#94a3b8"}),
    ]
    if r.get("range_lo") is not None:
        metrics.append(html.Span(f"  range [{r['range_lo']}, {r['range_hi']}]",
                                 title=_TIP_RANGE, style={"color": "#94a3b8", **_ul}))
    head = html.Div([
        html.Span(r["label"], style={"fontWeight": "700", "minWidth": "120px",
                                     "display": "inline-block", "color": "#e2e8f0"}),
        html.Span(r["verdict"] + inst, title=_TIP_VERDICT,
                  style={"fontWeight": "700", "color": clr, "minWidth": "160px",
                         "display": "inline-block", "cursor": "help"}),
        html.Span(metrics),
        theta_badge if theta_badge is not None else "",
        rev_badge if rev_badge is not None else "",
    ], style={**MONO, "fontSize": "0.66rem"})
    # OPEN TRADE lifecycle: when it triggered, entry/SL/target, live P&L, manage call
    # (lc already fetched above for the held-strike headline)
    trade_blk = None
    if lc:
        mng = lc.get("manage", "HOLD")
        # MAX-HOLD OVERRIDE — the scan's lifecycle has no time cap, so a leg running since
        # 10:02 still read "HOLD" at 14:55 (~5h) under a 90-MINUTE rule, while the ledger had
        # already timed it out. A stale HOLD on a dead position is the most dangerous thing
        # this board can say. The cap is the policy: past it, the call is CLOSE.
        #
        # LIVE-ONLY. `today` here is the WALL-CLOCK date (see _charts_scout_panel), so on a
        # replay/ghost day the age would be measured against real `now` instead of the replay
        # clock: a leg 30m old at as_of 11:00 read as 259m and every replayed leg falsely
        # flipped to CLOSE, corrupting ghost practice. In replay the engine's own manage call
        # (computed AT as_of) is already correct — leave it alone.
        _lage = _leg_age_min(lc.get("trigger"), today) if (live and today) else None
        if session_over:
            # THE SESSION IS OVER. The scan's lifecycle is a rolling per-bar leg with no
            # end-of-day concept, so after 15:30 it happily keeps a TRADE leg "running" and
            # ages it into the night. The arrow is INTRADAY-ONLY — it is flat overnight, by
            # construction (an option held overnight bleeds theta; that is why BTST uses
            # futures instead). Say so, rather than showing what looks like an open position.
            mng = "SESSION CLOSED 15:30 · arrow is intraday-only — flat overnight"
        elif _lage is not None and _lage >= _SCOUT_MAX_HOLD_MIN and not mng.startswith("BOOK"):
            mng = f"CLOSE · max-hold {_SCOUT_MAX_HOLD_MIN}m (held {_lage:.0f}m)"
        is_close = mng.startswith("CLOSE") or mng.startswith("BOOK") or session_over
        mclr = ("#94a3b8" if session_over else                      # settled, not a live call
                "#ef4444" if mng.startswith("CLOSE") else
                "#22c55e" if mng.startswith("BOOK") else "#34d399")
        pnl = lc.get("pnl_pct")
        pnlclr = "#22c55e" if (pnl or 0) > 0 else "#ef4444" if (pnl or 0) < 0 else "#94a3b8"
        idx_seg = (f"@ index {lc.get('entry_spot')}→{lc.get('cur_spot')}  "
                   if lc.get("entry_spot") is not None else "")
        trade_blk = html.Div([
            html.Span(f"⏱ leg triggered {lc['trigger']} ", title=_TIP_TRIG,
                      style={"color": "#fbbf24", "fontWeight": "700", "cursor": "help"}),
            html.Span(idx_seg, style={"color": "#93c5fd"}),   # INDEX level at trigger → now
            html.Span(f"entry {lc.get('entry_strike')} {side} ₹{lc.get('entry_prem')} "
                      f"→ ₹{lc.get('cur_prem')} ", style={"color": "#cbd5e1"}),
            html.Span((f"({pnl:+.0f}%)  " if pnl is not None else ""), style={"color": pnlclr, "fontWeight": "700"}),
            html.Span(f"SL ₹{lc.get('sl')}  T ₹{lc.get('target')}   ", style={"color": "#64748b"}),
            html.Span(f"▸ {mng}", style={"color": mclr, "fontWeight": "700",
                                         "background": "#2a0a0a" if mng.startswith("CLOSE") else "transparent",
                                         "padding": "0 4px", "borderRadius": "3px"}),
        ], style={**MONO, "fontSize": "0.62rem", "paddingLeft": "120px", "lineHeight": "1.5"})
    # forward prediction over the selected horizon + (replay) grade
    pdir = r.get("pred_dir", "RANGE")
    pclr = {"UP": "#34d399", "DOWN": "#f87171"}.get(pdir, "#fbbf24")
    _ctx = "live now" if live else ("👻 ghost live" if practice else "replay")
    pred_txt = (f"↪ {_ctx} · next {r.get('horizon', r['tf'])}m: {pdir}"
                + (f" → {r['pred_target']}" if r.get("pred_target") else "")
                + (f"  band[{r['pred_lo']}, {r['pred_hi']}]" if r.get("pred_lo") else ""))
    pred_kids = [html.Span(pred_txt, title=_TIP_BAND,
                           style={"color": pclr, "fontWeight": "700", "cursor": "help"})]
    # HONEST measured band coverage for this (index, horizon) from the calibration ledger
    if r.get("band_cover") is not None:
        _cc = {"ok": "#22c55e", "soft": "#fbbf24", "low": "#f87171", "thin": "#64748b"}
        _cm = {"ok": "✓", "soft": "~", "low": "⚠ low", "thin": "· thin"}
        pred_kids.append(html.Span(
            f"  cover {r['band_cover']*100:.0f}% {_cm.get(r['band_conf'], '')} (n{r['band_n']})",
            title="measured endpoint coverage of this band at this horizon "
                  "(backtest_band_horizon.py ledger); trust the band only where this is high",
            style={"color": _cc.get(r["band_conf"], "#94a3b8"), "fontWeight": "700",
                   "cursor": "help"}))
    v = r.get("verify")
    if v:
        ok = v["dir_hit"]
        pred_kids.append(html.Span(
            f"   ⇒ actual {v['actual']} ({v['move_pct']:+.2f}%) "
            f"{'HIT ✓' if ok else 'MISS ✗'}  band {'✓' if v['band_hit'] else '✗'}",
            style={"color": "#22c55e" if ok else "#ef4444", "fontWeight": "700"}))
    else:
        pred_kids.append(html.Span(
            "   ⇒ pending — grades itself once the horizon elapses" if live else
            "   ⇒ hidden — grades unlock at the real 15:30" if practice else
            "   ⇒ pending (advance the replay clock to grade)",
            style={"color": "#475569"}))
    pred = html.Div(pred_kids, style={**MONO, "fontSize": "0.6rem",
                                      "paddingLeft": "120px", "lineHeight": "1.4"})
    why = html.Div(" · ".join(r["reasons"][:3]),
                   style={**MONO, "fontSize": "0.56rem", "color": "#64748b",
                          "paddingLeft": "120px", "lineHeight": "1.3"})
    # opening-phase banner (gap type + cool-off / provisional warning)
    op = r.get("opening")
    open_blk = None
    if op and op.get("note"):                 # OPENING/SETTLING phase OR a data-warmup
        oclr = "#fbbf24" if (op.get("warming") or op.get("phase") == "OPENING") else "#60a5fa"
        gp = op.get("gap_pct")
        seg = (f"  gap {gp:+.2f}%" if gp is not None else "")
        seg += (f"  ·  OR [{op['or_lo']}, {op['or_hi']}]"
                if op.get("or_lo") is not None else "")
        open_blk = html.Div(f"🕒 {op['note']}{seg}",
                            style={**MONO, "fontSize": "0.6rem", "color": oclr,
                                   "fontWeight": "700", "paddingLeft": "120px",
                                   "lineHeight": "1.4", "background": "#15110a"
                                   if (op.get("warming") or op.get("phase") == "OPENING") else "#0a1422",
                                   "borderRadius": "3px", "padding": "2px 6px",
                                   "marginLeft": "120px", "marginTop": "2px"})
    # Trade memory: only when the live scan ISN'T already showing a current trade for
    # this row (no lifecycle) — i.e. the gate has gone NO-TRADE but a position is still
    # open (blink) or recently resolved. Avoids double-rendering the active trade.
    mem_blk = _scout_mem_block(mem, today) if not lc else None
    kids = ([head] + ([open_blk] if open_blk else [])
            + ([trade_blk] if trade_blk else []) + ([mem_blk] if mem_blk else [])
            + [pred, why, _scout_playbook(r)])
    return html.Div(kids,
                    style={"background": bg, "border": f"1px solid {clr if trade else '#1e293b'}",
                           "borderRadius": "5px", "padding": "5px 10px", "marginBottom": "4px"})


# ── GHOST-LIVE practice mode ─────────────────────────────────────────────────────
# Replay a captured past session pinned to TODAY'S wall clock: at 10:42 on a closed
# Saturday the charts/scout show that day AS OF 10:42, advancing on the normal refresh
# tick — paper-trade practice that feels live. Purely a READ-side view: nothing is
# written (the alert poller / scout-seen / calibration all key off the real today),
# so there is nothing to clean up afterwards.
def _ghost_ctx(date):
    """(day, 'HH:MM') for ghost mode. Day = the chosen past day, else the most recent
    captured session before today. Clock = now clamped to [09:15, 15:30] — after the
    real 15:30 the whole day is visible and verify grades unlock (review phase)."""
    from core.constants import LIVE_DIR
    today = datetime.datetime.now(IST).date().isoformat()
    day = date if (date and date != today) else None
    if day is None:
        caps = sorted(p.name[:10] for p in LIVE_DIR.glob("*_ticks.parquet")
                      if p.name[:10] < today and p.stat().st_size > 200_000)
        day = caps[-1] if caps else today
    now = datetime.datetime.now(IST).time()
    t = max(datetime.time(9, 15), min(now, datetime.time(15, 30)))
    return day, f"{t.hour:02d}:{t.minute:02d}"


def _ghost_done():
    """True once the real wall clock passes 15:30 — practice over, grades unlock."""
    return datetime.datetime.now(IST).time() >= datetime.time(15, 30)


@app.callback(Output("charts-asof", "value"), Input("url", "pathname"),
              State("charts-asof", "value"))
def _ghost_boot(_path, cur):
    """Weekend/holiday page load → default the Replay picker to ghost practice.
    Evaluated PER LOAD (not baked into the static layout, which is built once at
    process start and would go stale across a weekend→Monday process lifetime).
    A trading day never touches the picker — live stays the default; an explicit
    user choice (any non-None value) is never overridden."""
    from dash import no_update
    if cur is None and not is_trading_day(datetime.datetime.now(IST)):
        return "ghost"
    return no_update


# Timeframe the ALERT poller runs at (NEW/FLIP/BAND detection + the forward band the
# RANGE-BREAK checks against). Set to 60 to match the charts-tf UI default (60m band =
# the validated RANGE product). The poller is a headless server loop and CANNOT read the
# per-browser charts-tf dropdown, so its tf lives here as the single source of truth —
# threaded into scan(), the "broke the {tf}m band" label, and the live-overlay gate.
# 60m structure flips far less than 15m (kills the whipsaw churn) and the wider band
# breaks less often (each RANGE BREAK more meaningful).
_ALERT_TF = 60

# Max-hold TIMEOUT (minutes): a scout position that never hit SL/target/band/flip is
# force-CLOSED in the ledger once it has been open this long — the poller itself holds a
# dangling position all day (2026-07-09 BANK CE opened 10:02, still open 15:56, ~6h of
# theta bleed). 90m is the chosen cap (user pref within the data-optimal zone): the max-hold
# sweep over 9 sessions (band/SL/target ELSE timeout at cap) put net P&L flat and best across
# 45-90m (90m = −7.0% ALL, the marginal best) and STRICTLY WORSE at 120/135m (FLAT-index
# danglers bleed −6.2%@45m → −11.7%@135m). The user's first-guess 2h15m was the worst swept
# cap; 90m gives movers a little more runway while still guillotining all-day rot.
# Ledger-only (no poller change).
_SCOUT_MAX_HOLD_MIN = 90

# ── THE BELL ───────────────────────────────────────────────────────────────────────────
# The arrow is an INTRADAY product. It is flat overnight, always — a long option held over
# the night bleeds theta (which is the whole reason the BTST edge buys FUTURES instead).
# Two consequences the engine never enforced:
#
# _SCOUT_EOD_T — every open leg is SQUARED OFF at the bell. The poller loop simply stopped
#   at 15:30 and left whatever was open dangling: it was never closed in the log, the
#   ledger went on calling it "open" for another hour (until its 90m cap passed), and the
#   board rendered a live, ageing leg deep into the night.
#
# _SCOUT_NO_NEW_AFTER — and therefore a leg opened in the last 20 minutes CANNOT reach its
#   SL or target; it is force-flat within minutes and pays the ~3% option round-trip for a
#   sliver of noise. MEASURED over 12 captured days: legs opened 15:10-15:30 returned
#   mean -50.8%, win 13% (n=47) versus -5.6%, win 37% for the rest of the day. Not a tuned
#   parameter — a structural one: do not open what the bell will immediately close.
_SCOUT_EOD_T = datetime.time(15, 30)
_SCOUT_NO_NEW_AFTER = datetime.time(15, 10)


def _scout_episodes(today: str, as_of=None):
    """Replay the persisted scout_alerts log into per-index EPISODES: each NEW → close
    (FLIP / SL / TARGET) pair, plus any still-open tail. Returns (open, closed): open is
    the held tail per index; closed is every resolved episode, newest-first. Realized P&L
    on a close = exit(cur) / entry − 1 (the arrow's option, entry→exit). Leak-safe — pure
    replay of what the poller already wrote."""
    import pandas as pd
    import intraday_scout as scout
    from core.mirror_io import read_mirror
    opens, closed = [], []
    # as_of caps the log so a REPLAY view can't see FLIP/SL/BAND closes that fire AFTER the
    # viewed minute (a future leak). None (the anchors caller) = full day, unchanged. This
    # also keeps the 90m TIMEOUT (below, cap_t <= as_of) consistent with the other closes.
    df = read_mirror("scout_alerts", today, as_of=as_of)
    if df is None or df.empty:
        return opens, closed

    def _v(x):
        return None if x is None or (isinstance(x, float) and pd.isna(x)) else x

    for sym, g in df.groupby("symbol"):                  # df is ts-ascending
        ep = None

        def _cap_t(e):
            return e["open_ts"] + pd.Timedelta(minutes=_SCOUT_MAX_HOLD_MIN)

        def _timeout_close(e):
            """Force-close `e` as TIMEOUT at min(open+cap, 15:30), premium looked up at that
            minute (leak-safe: the cap instant is <= as_of whenever this is invoked)."""
            eff_t = min(_cap_t(e), pd.Timestamp(f"{today} 15:30", tz=IST))
            exit_p = None
            try:
                exit_p = scout._opt_premium(sym, today, eff_t.to_pydatetime(),
                                            e.get("strike"), e.get("dir"))
            except Exception:
                exit_p = None
            entry = e.get("entry")
            pnl = round((exit_p / entry - 1.0) * 100.0, 1) if (exit_p and entry) else None
            return {**e, "close_t": eff_t.strftime("%H:%M"), "outcome": "TIMEOUT",
                    "exit": round(exit_p, 2) if exit_p else None, "pnl": pnl}

        for _, r in g.iterrows():
            kind = str(r.get("kind") or "")
            if kind == "NEW":
                if ep:                                   # defensive: prior NEW never closed
                    closed.append({**ep, "close_t": None, "outcome": "?",
                                   "exit": None, "pnl": None})
                strike = _v(r.get("strike"))
                ep = {"sym": sym, "label": _v(r.get("label")) or str(sym),
                      "dir": _v(r.get("side")),
                      "strike": int(strike) if strike is not None else None,
                      "entry": _v(r.get("entry")), "open_t": r["ts"].strftime("%H:%M"),
                      "open_ts": r["ts"],
                      "sl": _v(r.get("sl")), "tgt": _v(r.get("tgt")),
                      "bb": False, "band_dir": None}
            elif kind == "TIMEOUT" and ep:
                # NATIVE timeout row — the POLLER now enforces the cap at the source and logs
                # it. Close at the cap instant (not the poll minute, which trails it by up to
                # one 30s tick) so a native TIMEOUT and the ledger's own reinterpretation of a
                # legacy log produce the SAME close_t/exit. Older logs (pre-poller-cap) carry
                # no TIMEOUT row and are still handled by the two branches below.
                closed.append(_timeout_close(ep))
                ep = None
            elif kind == "EOD" and ep:
                # squared off at the bell (the engine's native row)
                exit_p, entry = _v(r.get("cur")), ep.get("entry")
                pnl = round((exit_p / entry - 1.0) * 100.0, 1) if (exit_p and entry) else None
                closed.append({**ep, "close_t": r["ts"].strftime("%H:%M"), "outcome": "EOD",
                               "exit": exit_p, "pnl": pnl})
                ep = None
            elif kind in ("BAND", "FLIP", "SL", "TARGET") and ep and r["ts"] > _cap_t(ep):
                # The poller's close fired AFTER the max-hold cap — under the 90m policy the
                # position was already dead at cap_t, so the ledger closes it as TIMEOUT at
                # the cap, NOT at the later trigger (2026-07-09 MIDCAP 12:16→15:20 "FLIP" was
                # really a 184-minute hold; the policy exit was 13:46). Keeps the scoreboard
                # consistent with the swept 90m rule instead of mixing hold windows. A FLIP's
                # subsequent NEW row still opens the next episode normally.
                closed.append(_timeout_close(ep))
                ep = None
            elif kind == "BAND" and ep:
                # A σ-band break RESOLVES the episode (range exceeded) → CLOSE it here,
                # like SL/TARGET/FLIP. Exit = the arrow's premium at the band-break minute;
                # the BAND row carries no cur, so look it up (leak-safe, ts = band time).
                ep["bb"] = True
                bd = _v(r.get("band_dir")) or ep.get("band_dir")
                ep["band_dir"] = bd
                exit_p = _v(r.get("cur"))
                if exit_p is None:
                    try:
                        exit_p = scout._opt_premium(sym, today, r["ts"],
                                                    ep.get("strike"), ep.get("dir"))
                    except Exception:
                        exit_p = None
                entry = ep.get("entry")
                pnl = round((exit_p / entry - 1.0) * 100.0, 1) if (exit_p and entry) else None
                closed.append({**ep, "close_t": r["ts"].strftime("%H:%M"),
                               "outcome": "BAND", "band_dir": bd,
                               "exit": round(exit_p, 2) if exit_p else None, "pnl": pnl})
                ep = None
            elif kind in ("FLIP", "SL", "TARGET") and ep:
                exit_p, entry = _v(r.get("cur")), ep.get("entry")
                pnl = round((exit_p / entry - 1.0) * 100.0, 1) if (exit_p and entry) else None
                closed.append({**ep, "close_t": r["ts"].strftime("%H:%M"),
                               "outcome": kind, "exit": exit_p, "pnl": pnl})
                ep = None
        if ep:
            # MAX-HOLD TIMEOUT (dangling tail) — a still-open position past the cap with NO
            # later close row is force-closed as TIMEOUT (leak-safe: only when as_of >= cap).
            # Without this a no-trigger position bleeds theta all day. Late closes (a poller
            # SL/FLIP AFTER the cap) are converted to TIMEOUT in the row loop above.
            _asof_ts = pd.Timestamp(as_of) if as_of is not None else None
            if _asof_ts is not None and _asof_ts.tzinfo is None:
                _asof_ts = _asof_ts.tz_localize(IST)   # defensive: match read_mirror
            # THE BELL closes it too, not just the 90m cap. A leg opened 15:29 has a cap of
            # 16:59, so the old test kept calling it OPEN for an hour and a half AFTER the
            # market shut. Nothing is held overnight; whichever comes first, cap or bell.
            _bell = pd.Timestamp(f"{today} 15:30", tz=IST)
            _dead_at = min(_cap_t(ep), _bell)
            if _asof_ts is not None and _asof_ts >= _dead_at:
                closed.append(_timeout_close(ep))
            else:
                opens.append(ep)
    # NEWEST FIRST, both tables. The episodes are built by grouping on SYMBOL, so without this
    # the open legs came out in symbol order (FIN 10:14, MIDCAP 10:05, NIFTY 10:33) — an order
    # that means nothing to a reader. The freshest position is the one you are still deciding
    # about, so it belongs at the top. `closed` was already newest-first by close time; `opens`
    # now matches, keyed on when the leg was OPENED.
    opens.sort(key=lambda e: e.get("open_t") or "", reverse=True)
    closed.sort(key=lambda e: e.get("close_t") or "", reverse=True)
    return opens, closed


_OUTCOME_BADGE = {"FLIP": ("↺ flipped · reversed out", "#f59e0b"),
                  "EOD": ("🔔 squared off at the bell", "#94a3b8"),
                  "SL": ("🛑 SL hit", "#f87171"),
                  "TARGET": ("🎯 target", "#22c55e"),
                  "TIMEOUT": (f"⌛ timed out ({_SCOUT_MAX_HOLD_MIN}m)", "#a78bfa"),
                  "?": ("? unresolved", "#94a3b8")}


def _scout_log_health(today: str):
    """Is the alert log actually being WRITTEN? The ledger replays that log, so a dead
    poller renders a calm "0 open" while real positions sit live on the board — which
    reads as "you have no position". That has happened (2026-07-13: market data captured
    through 15:30, alerts stopped at 14:23).

    Diagnose by comparing the alert log against a mirror the CAPTURER writes continuously
    (chain_snapshots). If capture is fresh but alerts are stale, the poller specifically is
    dead — not the feed, not the sync. Returns None when healthy."""
    from core.mirror_io import read_mirror
    try:
        now = datetime.datetime.now(IST)
        if not is_trading_day(now) or now.time() < datetime.time(9, 45):
            return None                       # pre-open / non-session: nothing to claim
        a = read_mirror("scout_alerts", today)
        c = read_mirror("chain_snapshots", today)
        if c is None or c.empty:
            return None                       # no capture reference → cannot diagnose
        cap_last = c["ts"].max()
        # the session clock the poller SHOULD have reached by now
        clock = min(now, datetime.datetime.combine(now.date(), datetime.time(15, 30)).replace(tzinfo=IST))
        alert_last = a["ts"].max() if (a is not None and not a.empty) else None
        cap_age = (clock - cap_last).total_seconds() / 60.0
        if cap_age > 15:
            return None                       # capture itself is behind → a different fault
        gap = ((clock - alert_last).total_seconds() / 60.0) if alert_last is not None \
            else (clock - datetime.datetime.combine(now.date(), datetime.time(9, 45)).replace(tzinfo=IST)).total_seconds() / 60.0
        if gap <= 20:                         # the poller only writes on EVENTS, so a quiet
            return None                       # stretch is normal — 20m of slack before alarm
        return {"alert_last": alert_last.strftime("%H:%M") if alert_last is not None else "never",
                "cap_last": cap_last.strftime("%H:%M"), "gap_min": int(gap)}
    except Exception:
        return None


def _btst_section(today: str):
    """🌙 BTST carries — a SEPARATE, walled-off section of the popup.

    Deliberately NOT merged into the scout episodes, and NOT counted in the scout summary.
    They are different animals and mixing them would corrupt both numbers:

        SCOUT : index OPTIONS, 60-90m, force-closed 15:30 — NEGATIVE-EV (the summary's whole
                job is to show the arrow LOSES; averaging a winning edge into it would hide
                that).
        BTST  : index FUTURES, held OVERNIGHT, exit ~09:30 — the one VALIDATED edge here
                (+10-13bps, Sharpe 2.4-4). Its P&L is bps on futures, not option premium %;
                the two units are not even commensurable.

    Also surfaces the silent failure that prompted this: the board can show BTST-CARRY while
    NOTHING is written to the ledger (the scheduled emit was disabled 07-07..07-13 and six
    days of the only validated edge went unrecorded)."""
    try:
        import btst_signal as bs
        led = bs._load_ledger()
    except Exception:
        return []
    if led is None or led.empty:
        return []
    led = led.copy()
    led["date"] = led["date"].astype(str)
    carried = led[led["status"] == "OPEN"]                       # held overnight, not yet exited
    closed_today = led[(led["status"] == "CLOSED") & (led["exit_date"].astype(str) == today)]
    if carried.empty and closed_today.empty:
        return []

    def _row(r, live: bool):
        px = r.get("exit_px")
        bps = r.get("net_bps")
        return html.Tr([
            html.Td(str(r["sym"]), style={"padding": "3px 8px", "fontWeight": "700"}),
            html.Td(str(r["date"]), style={"padding": "3px 8px", "color": "#94a3b8"}),
            html.Td(f"{r['clr']:.2f}", style={"padding": "3px 8px"}),
            html.Td(f"{r['entry_px']:,.1f}", style={"padding": "3px 8px"}),
            html.Td(f"{px:,.1f}" if px == px and px else "—", style={"padding": "3px 8px"}),
            html.Td(html.Span(f"{bps:+.1f}" if (bps == bps and bps is not None) else "—",
                              style={"color": ("#22c55e" if (bps or 0) >= 0 else "#f87171"),
                                     "fontWeight": "700"}), style={"padding": "3px 8px"}),
            html.Td(html.Span("🌙 OPEN OVERNIGHT" if live else "✓ closed",
                              style={"color": "#fbbf24" if live else "#94a3b8"}),
                    style={"padding": "3px 8px"}),
        ])

    hdr = html.Tr([html.Th(h, style={"padding": "3px 8px", "textAlign": "left",
                                     "color": "#94a3b8", "fontWeight": "600"})
                   for h in ("index", "entered", "clr", "entry", "exit", "net bps", "status")])
    rows = [_row(r, True) for _, r in carried.iterrows()] + \
           [_row(r, False) for _, r in closed_today.iterrows()]
    n_open = len(carried)
    return [
        html.Div([
            html.Span("🌙 BTST — carried OVERNIGHT ", style={"color": "#fbbf24",
                                                             "fontWeight": "700"}),
            html.Span(f"· {n_open} open · index FUTURES · exit ~09:30 next session",
                      style={"color": "#94a3b8"}),
        ], style={"fontSize": "0.62rem", "margin": "10px 0 3px"}),
        html.Table([html.Thead(hdr), html.Tbody(rows)],
                   style={"width": "100%", "fontSize": "0.62rem",
                          "borderCollapse": "collapse"}),
        html.Div("Separate from the scout P&L above — and deliberately so. SCOUT is the "
                 "intraday option ARROW (negative-EV; that summary exists to prove it "
                 "loses). BTST is the index-FUTURES overnight carry — the one validated "
                 "edge (+10–13bps). Mixing option-% into futures-bps would corrupt both. "
                 "PAPER only; nothing auto-executes.",
                 style={"color": "#64748b", "fontSize": "0.55rem", "lineHeight": "1.4",
                        "marginTop": "3px"}),
    ]


def _scout_openpos_body(today: str, as_of):
    """Popup body: the day's scout ledger — OPEN positions (with a live cross-check +
    unrealized P&L) on top, then CLOSED episodes (outcome + realized P&L), newest-first.
    All reconstructed from the persisted alert log (tf = _ALERT_TF). BTST carries are
    appended in their OWN section — never merged into the scout stats (see _btst_section)."""
    import intraday_scout as scout
    opens, closed = _scout_episodes(today, as_of=as_of)

    # ── the log must not be allowed to LOOK healthy when it is not ────────────────
    _sick = _scout_log_health(today) if as_of is None else None
    _warn = []
    if _sick:
        _warn = [html.Div(
            [html.B("🛑 ALERT LOG IS STALE — THIS LEDGER IS INCOMPLETE. "),
             f"The poller last wrote at {_sick['alert_last']}, but market data was captured "
             f"through {_sick['cap_last']} ({_sick['gap_min']}m gap). The capturer and the "
             f"sync are alive — the scout-alert writer specifically has stopped. Positions "
             f"opened after {_sick['alert_last']} are MISSING here, so an empty OPEN section "
             f"does NOT mean you have no position. Cross-check against the board."],
            style={"background": "#7f1d1d", "color": "#fecaca", "padding": "8px 10px",
                   "borderRadius": "6px", "fontSize": "0.78rem", "marginBottom": "8px",
                   "border": "1px solid #ef4444"})]

    _btst = _btst_section(today) if as_of is None else []
    if not opens and not closed:
        # BTST can exist with ZERO scout episodes — never hide the carries behind an
        # "alert log is empty" message.
        return html.Div(_warn + [html.Div(
            "No scout episodes yet — the alert log is empty.",
            style={"color": "#94a3b8", "fontSize": "0.8rem", "padding": "10px"})] + _btst)

    # ── OPEN section — live cross-check + unrealized P&L ─────────────────────────
    # Each open row needs a live premium read + a fresh verdict scan (heavy, IO-bound
    # mirror reads). Run the ≤4 rows CONCURRENTLY so the popup opens in ~one scan's time
    # instead of stacking 4 serially (was the ~1.5s "loading" lag).
    from concurrent.futures import ThreadPoolExecutor

    def _open_cross(o):
        sym, d, k = o["sym"], o.get("dir"), o.get("strike")
        cur = peak = None
        try:
            if k:
                st = scout._opt_stats(sym, today, as_of, k, d, since=o.get("open_ts"))
                if st:
                    cur, peak = st["now"], st["peak"]
        except Exception:
            cur = peak = None
        try:
            v = scout.scan_index(sym, _ALERT_TF, date=today, as_of=as_of,
                                 with_lifecycle=False, verdict_only=True)
            verdict, vdir = v.get("verdict"), v.get("direction")
        except Exception:
            verdict, vdir = None, None
        return cur, peak, verdict, vdir

    if opens:
        with ThreadPoolExecutor(max_workers=min(4, len(opens))) as _ex:
            _cross = list(_ex.map(_open_cross, opens))
    else:
        _cross = []

    # Both tables render as sortable/filterable DataTables sharing one style helper.
    # Numeric cols (Strike/Entry/Now/Exit/P&L) sort by VALUE not text; P&L is stored as
    # a fraction + percentage-formatted so its sort key is the true number.
    from dash import dash_table
    from dash.dash_table.Format import Format, Group, Scheme, Sign, Symbol
    from core.constants import LOT_SIZES
    _rupee = Format(precision=2, scheme=Scheme.fixed).symbol(Symbol.yes).symbol_prefix("₹")
    _pctf = Format(precision=1, scheme=Scheme.percentage).sign(Sign.positive)
    # ₹ P&L at ONE lot = (exit − entry) × index lot size. This is what actually hits the
    # account — and it is NOT the same across indices: MIDCAP's 120-lot makes every ₹1 of
    # premium move worth ₹120 (double NIFTY's 65), so an equal-% loss on MIDCAP is a far
    # bigger rupee hit. The % column flatters MIDCAP churn; the ₹ column tells the truth.
    _rupee0 = (Format(precision=0, scheme=Scheme.fixed).group(Group.yes)
               .sign(Sign.positive).symbol(Symbol.yes).symbol_prefix("₹"))

    def _pnl_rs(sym, entry, exit_p):
        lot = LOT_SIZES.get(sym)
        if not (lot and entry and exit_p):
            return None
        return round((exit_p - entry) * lot)
    _ledger_side_pnl_cond = [
        {"if": {"filter_query": "{side} = CE", "column_id": "side"},
         "color": "#34d399", "fontWeight": "700"},
        {"if": {"filter_query": "{side} = PE", "column_id": "side"},
         "color": "#f87171", "fontWeight": "700"},
        {"if": {"filter_query": "{pnl} < 0", "column_id": "pnl"},
         "color": "#f87171", "fontWeight": "700"},
        {"if": {"filter_query": "{pnl} >= 0", "column_id": "pnl"},
         "color": "#22c55e", "fontWeight": "700"},
    ]

    def _ledger_table(data, columns, extra_cond, tips=None, header_tips=None, tbl_id=None):
        return dash_table.DataTable(
            id=tbl_id, data=data, columns=columns,
            sort_action="native", sort_mode="single", page_action="none",
            style_as_list_view=True,
            style_table={"marginBottom": "6px", "overflowX": "auto"},
            style_header={"backgroundColor": "#1e293b", "color": "#e2e8f0",
                          "fontSize": "0.62rem", "textTransform": "uppercase",
                          "border": "none", "fontWeight": "700", "cursor": "pointer"},
            style_filter={"backgroundColor": "#0b1220", "color": "#e2e8f0",
                          "border": "none"},
            style_cell={"backgroundColor": "#0f172a", "color": "#e2e8f0",
                        "fontSize": "0.72rem", "border": "none", "padding": "4px 8px",
                        "textAlign": "left"},
            style_data_conditional=_ledger_side_pnl_cond + extra_cond,
            # hover help: per-state cell tooltips + per-column header tooltips
            tooltip_conditional=tips or [],
            tooltip_header=header_tips or {},
            tooltip_delay=150, tooltip_duration=None,
            css=[{"selector": ".dash-table-tooltip",
                  "rule": "background-color:#0f172a; color:#e2e8f0; "
                          "border:1px solid #334155; font-size:0.68rem; "
                          "max-width:260px; padding:6px 8px;"}],
        )

    def _tip(col, word, text):
        return {"if": {"column_id": col,
                       "filter_query": f'{{{col}}} contains "{word}"'},
                "value": text, "type": "markdown"}

    open_records = []
    for o, (cur, peak, verdict, vdir) in zip(opens, _cross):
        d, entry = o.get("dir"), o.get("entry")
        sl, tgt = o.get("sl"), o.get("tgt")
        cur = round(cur, 2) if cur else None
        pnl = round((cur / entry - 1.0) * 100.0, 1) if (cur and entry) else None
        if verdict is None:
            flag = "· no data"                       # scan failed / index has no data
        elif verdict == "NO-TRADE":
            flag = "⚠ stale"                          # arrow gone, position orphaned
        elif vdir and vdir != d:
            flag = "⚠ flipped"                        # board leans the OPPOSITE side
        elif vdir == d:
            flag = "✓ confirms"                       # board still on this side
        else:
            flag = "· unclear"                        # TRADE but no clear direction
        # Band that broke: band_dir 'above' = spot cleared the UPPER σ-band, 'below' = LOWER.
        bd = o.get("band_dir")
        band = ("↑ upper" if bd == "above" else "↓ lower" if bd == "below"
                else "broke" if o.get("bb") else "—")
        open_records.append({
            "index": o["label"], "side": d, "strike": o.get("strike"),
            "since": o.get("open_t"), "entry": entry, "now": cur,
            "pnl": (pnl / 100.0) if pnl is not None else None,
            "rupee": _pnl_rs(o["sym"], entry, cur),
            "band": band, "status": _scout_trade_status(entry, cur, sl, tgt, peak),
            "check": flag,
        })
    open_cols = [
        {"name": "Index", "id": "index"},
        {"name": "Side", "id": "side"},
        {"name": "Strike", "id": "strike", "type": "numeric"},
        {"name": "Since", "id": "since"},
        {"name": "Entry", "id": "entry", "type": "numeric", "format": _rupee},
        {"name": "Now", "id": "now", "type": "numeric", "format": _rupee},
        {"name": "P&L (unreal.)", "id": "pnl", "type": "numeric", "format": _pctf},
        {"name": "₹/lot (unreal.)", "id": "rupee", "type": "numeric", "format": _rupee0},
        {"name": "Band", "id": "band"},
        {"name": "Trade status", "id": "status"},
        {"name": "Live check", "id": "check"},
    ]
    open_extra_cond = [
        {"if": {"filter_query": "{rupee} < 0", "column_id": "rupee"},
         "color": "#f87171", "fontWeight": "700"},
        {"if": {"filter_query": "{rupee} >= 0", "column_id": "rupee"},
         "color": "#22c55e", "fontWeight": "700"},
        {"if": {"filter_query": '{check} contains "confirms"', "column_id": "check"},
         "color": "#22c55e", "fontWeight": "600"},
        {"if": {"filter_query": '{check} contains "stale"', "column_id": "check"},
         "color": "#f59e0b", "fontWeight": "600"},
        {"if": {"filter_query": '{check} contains "flipped"', "column_id": "check"},
         "color": "#f87171", "fontWeight": "600"},
        {"if": {"filter_query": '{check} contains "no data"', "column_id": "check"},
         "color": "#64748b", "fontWeight": "600"},
        {"if": {"filter_query": '{check} contains "unclear"', "column_id": "check"},
         "color": "#64748b", "fontWeight": "600"},
        {"if": {"filter_query": '{band} contains "broke"', "column_id": "band"},
         "color": "#f59e0b", "fontWeight": "600"},
        {"if": {"filter_query": '{band} contains "upper"', "column_id": "band"},
         "color": "#f59e0b", "fontWeight": "600"},
        {"if": {"filter_query": '{band} contains "lower"', "column_id": "band"},
         "color": "#f59e0b", "fontWeight": "600"},
        {"if": {"filter_query": '{status} contains "target"', "column_id": "status"},
         "color": "#22c55e", "fontWeight": "700"},
        {"if": {"filter_query": '{status} contains "SL"', "column_id": "status"},
         "color": "#f87171", "fontWeight": "700"},
        {"if": {"filter_query": '{status} contains "pullback"', "column_id": "status"},
         "color": "#f59e0b", "fontWeight": "600"},
        {"if": {"filter_query": '{status} contains "▲"', "column_id": "status"},
         "color": "#34d399", "fontWeight": "600"},
        {"if": {"filter_query": '{status} contains "▼"', "column_id": "status"},
         "color": "#fca5a5", "fontWeight": "600"},
    ]
    open_tips = [
        _tip("check", "confirms", "Board still on your side."),
        _tip("check", "stale", "Board went NO-TRADE. Held till it flips or hits SL / target."),
        _tip("check", "flipped", "Board now leans the other way."),
        _tip("check", "no data", "Scan failed / no data now."),
        _tip("check", "unclear", "Board shows a trade but no clear side."),
        _tip("band", "upper", "Index cleared the UPPER σ-range band (broke above)."),
        _tip("band", "lower", "Index broke the LOWER σ-range band (broke below)."),
        _tip("band", "broke", "Moved past the σ-range band since entry."),
        _tip("status", "target", "Premium reached the +65% target — close pending on the next poll."),
        _tip("status", "SL", "Premium hit the −35% stop — close pending on the next poll."),
        _tip("status", "pullback", "Ran up ≥20% then gave back ≥15pts of that gain (peak → now)."),
        _tip("status", "▲", "In profit, running toward the +65% target."),
        _tip("status", "▼", "Underwater, drawing toward the −35% stop."),
    ]
    open_header_tips = {
        "side": "CE = call (bullish lean) · PE = put (bearish lean)",
        "strike": "ATM strike at the moment the position opened.",
        "since": "When this position opened (HH:MM).",
        "entry": "Option premium paid at entry.",
        "now": "Live option premium.",
        "pnl": "Unrealised P&L = now / entry − 1.",
        "rupee": "Unrealised ₹ at ONE lot = (now − entry) × index lot size "
                 "(NIFTY 65 · BANK 30 · FIN 60 · MIDCAP 120). What actually moves in the "
                 "account — MIDCAP's 120-lot doubles NIFTY's rupee swing for the same %.",
        "band": "'—' = still inside the σ-range band. A break (↑ upper / ↓ lower) closes "
                "the episode → it moves to the CLOSED table below.",
        "status": "Live trajectory on the option premium vs SL (−35%) / target (+65%): "
                  "running ▲ / drawdown ▼ / pullback / hit. Poller closes on SL/target/flip.",
        "check": "Live board vs your held side — is the arrow still with you?",
    }
    open_tbl = _ledger_table(open_records, open_cols, open_extra_cond,
                             open_tips, open_header_tips, tbl_id="scout-open-table")

    # ── CLOSED section — same sortable/filterable ledger (shared _ledger_table) ──
    closed_records = []
    for e in closed:
        oc = e.get("outcome")
        if oc == "BAND":                              # directional: which σ-band broke
            bd = e.get("band_dir")
            badge = ("⚡ band ↑ upper" if bd == "above" else
                     "⚡ band ↓ lower" if bd == "below" else "⚡ band broke")
        else:
            badge, _bc = _OUTCOME_BADGE.get(oc, ("—", "#94a3b8"))
        pnl = e.get("pnl")
        closed_records.append({
            "index": e["label"], "side": e.get("dir"), "strike": e.get("strike"),
            "held": f"{e.get('open_t')}→{e.get('close_t') or '—'}",
            "entry": e.get("entry"), "exit": e.get("exit"),
            "pnl": (pnl / 100.0) if pnl is not None else None,
            "rupee": _pnl_rs(e["sym"], e.get("entry"), e.get("exit")),
            "outcome": badge,
        })
    closed_cols = [
        {"name": "Index", "id": "index"},
        {"name": "Side", "id": "side"},
        {"name": "Strike", "id": "strike", "type": "numeric"},
        {"name": "Held", "id": "held"},
        {"name": "Entry", "id": "entry", "type": "numeric", "format": _rupee},
        {"name": "Exit", "id": "exit", "type": "numeric", "format": _rupee},
        {"name": "P&L (real.)", "id": "pnl", "type": "numeric", "format": _pctf},
        {"name": "₹/lot (real.)", "id": "rupee", "type": "numeric", "format": _rupee0},
        {"name": "Outcome", "id": "outcome"},
    ]
    closed_extra_cond = [
        {"if": {"filter_query": "{rupee} < 0", "column_id": "rupee"},
         "color": "#f87171", "fontWeight": "700"},
        {"if": {"filter_query": "{rupee} >= 0", "column_id": "rupee"},
         "color": "#22c55e", "fontWeight": "700"},
        {"if": {"filter_query": '{outcome} contains "SL"', "column_id": "outcome"},
         "color": "#f87171", "fontWeight": "600"},
        {"if": {"filter_query": '{outcome} contains "target"', "column_id": "outcome"},
         "color": "#22c55e", "fontWeight": "600"},
        {"if": {"filter_query": '{outcome} contains "flipped"', "column_id": "outcome"},
         "color": "#f59e0b", "fontWeight": "600"},
        {"if": {"filter_query": '{outcome} contains "band"', "column_id": "outcome"},
         "color": "#f59e0b", "fontWeight": "700"},
        {"if": {"filter_query": '{outcome} contains "timed out"', "column_id": "outcome"},
         "color": "#a78bfa", "fontWeight": "700"},
        {"if": {"filter_query": '{outcome} contains "unresolved"', "column_id": "outcome"},
         "color": "#64748b", "fontWeight": "600"},
    ]
    closed_tips = [
        _tip("outcome", "target", "Premium hit the +65% target."),
        _tip("outcome", "flipped", "Arrow reversed to the other side → position exited "
             "(the dominant exit at 60m — SL / target rarely bind)."),
        _tip("outcome", "SL", "Premium hit the −35% stop."),
        _tip("outcome", "band", "Spot broke the σ-range band (↑ upper / ↓ lower) — range "
             "exceeded, episode closed at the arrow's premium then."),
        _tip("outcome", "timed out", f"Held {_SCOUT_MAX_HOLD_MIN}m without hitting SL / target "
             "/ band / flip → force-closed (max-hold cap). Beyond the band's 60m forecast "
             "window a no-touch position is just bleeding theta; 9-day sweep: >90m caps only "
             "deepen the loss."),
        _tip("outcome", "unresolved", "Defensive: a prior NEW that never recorded a close."),
    ]
    closed_header_tips = {
        "side": "CE = call (bullish lean) · PE = put (bearish lean)",
        "strike": "ATM strike at entry.",
        "held": "Open → close time span (HH:MM→HH:MM).",
        "entry": "Option premium at entry.",
        "exit": "Option premium at close.",
        "pnl": "Realised P&L = exit / entry − 1.",
        "rupee": "Realised ₹ at ONE lot = (exit − entry) × index lot size "
                 "(NIFTY 65 · BANK 30 · FIN 60 · MIDCAP 120). The real account move — "
                 "MIDCAP churn hits 120× per ₹, so its red trades dominate the day's total.",
        "outcome": "How the episode closed.",
    }
    closed_tbl = _ledger_table(closed_records, closed_cols, closed_extra_cond,
                               closed_tips, closed_header_tips, tbl_id="scout-closed-table")

    # ── day summary — realized win-rate + the ACCOUNT-TRUE ₹ total at 1 lot ───────
    rp = [e["pnl"] for e in closed if e.get("pnl") is not None]
    wins = sum(1 for x in rp if x > 0)
    avg = round(sum(rp) / len(rp), 1) if rp else None
    # Real ₹ at ONE lot per index — the number the % Σ hides. Summing %s across trades
    # that sit on different premiums AND different lot sizes is meaningless; ₹ is not.
    rs_by_idx, net_rs = {}, 0
    for e in closed:
        r = _pnl_rs(e.get("sym"), e.get("entry"), e.get("exit"))
        if r is None:
            continue
        net_rs += r
        rs_by_idx[e["label"]] = rs_by_idx.get(e["label"], 0) + r
    summ_c = "#22c55e" if (avg or 0) >= 0 else "#f87171"
    net_c = "#22c55e" if net_rs >= 0 else "#f87171"
    summary = html.Div([
        html.Div([
            html.Span(f"{len(opens)} open  ·  {len(closed)} closed", style={"color": "#e2e8f0",
                      "fontWeight": "700"}),
            html.Span(f"   realized: {wins}/{len(rp)} win", style={"color": "#94a3b8"})
            if rp else html.Span(""),
            html.Span(f"  ·  avg {avg:+.1f}%" if avg is not None else "",
                      style={"color": summ_c, "fontWeight": "700"}),
            # THE headline number: real rupees at 1 lot each. Not %-summed.
            html.Span(f"  ·  Σ ₹{net_rs:+,} (1 lot)" if rs_by_idx else "",
                      title="Realised ₹ if you traded exactly ONE lot of each — "
                            "Σ (exit − entry) × index lot size. The account-true total; "
                            "the arrow is negative-EV over large samples.",
                      style={"color": net_c, "fontWeight": "800", "cursor": "help",
                             "fontSize": "0.74rem"}),
        ]),
        # per-index ₹ breakdown, worst → best — makes the heavy-lot index's damage obvious
        html.Div([
            html.Span(f"{lbl} ₹{v:+,}   ",
                      style={"color": "#22c55e" if v >= 0 else "#f87171",
                             "fontWeight": "700", "marginRight": "4px"})
            for lbl, v in sorted(rs_by_idx.items(), key=lambda kv: kv[1])
        ], style={"fontSize": "0.62rem", "marginTop": "3px"}) if rs_by_idx else html.Span(""),
    ], style={"fontSize": "0.68rem", "marginBottom": "8px"})

    note = html.Div(
        "Hover any cell or header for what it means. Click a header to sort ⇕; use the "
        "search box (top-right) to filter both tables across all columns (e.g. NIFTY, PE, "
        "flipped, contradicts). One stance per index; OPEN holds through NO-TRADE blinks "
        "until it flips / hits SL / target / breaks the σ-band. CLOSED P&L is realized (entry→exit on the "
        f"arrow's option); ₹/lot = that move × the index lot size (the real account P&L — "
        "MIDCAP's 120-lot dominates the day). Alert-log state, "
        f"tf={_ALERT_TF}m. Decision-support only — the arrow is negative-EV; trade the "
        "range band, not the arrow.",
        style={"color": "#64748b", "fontSize": "0.58rem", "lineHeight": "1.4"})
    return html.Div(_warn + [
        # full unfiltered rows — the search box filters the tables FROM these
        dcc.Store(id="scout-open-store", data=open_records),
        dcc.Store(id="scout-closed-store", data=closed_records),
        summary,
        html.Div("● OPEN", style={"color": "#34d399", "fontSize": "0.6rem",
                 "fontWeight": "700", "marginBottom": "3px"}),
        open_tbl,
        html.Div("○ CLOSED", style={"color": "#94a3b8", "fontSize": "0.6rem",
                 "fontWeight": "700", "margin": "8px 0 3px"}),
        closed_tbl, note] + _btst)


def _charts_scout_panel(tf_min, date, as_of_dt, live=False, seen=None, practice=False):
    import intraday_scout as scout
    today = datetime.datetime.now(IST).date().isoformat()
    # Freeze each OPEN position's lifecycle at the poller's true first-fire minute (the
    # ledger's "since"), so the board stops re-stamping entry=now / +0% on the forming 60m
    # bar and reads the SAME entry the ledger does. Live only — the poller log is a live
    # artifact; replay/ghost keep the per-bar grid walk. Built once, passed into scan.
    anchors = None
    if live:
        try:
            # MUST pass as_of (live => now): with as_of=None the dangling-tail branch in
            # _scout_episodes can never fire, so a position PAST the 90m cap is still
            # reported open here and keeps anchoring the live board to a leg the ledger has
            # already timed out — the same badge-vs-popup divergence, one layer down.
            _op0, _ = _scout_episodes(today, as_of=datetime.datetime.now(IST))
            anchors = {}
            for e in _op0:
                try:
                    _h, _m = (e.get("open_t") or "").split(":")
                    anchors[e["sym"]] = {
                        "t": datetime.datetime(*map(int, date.split("-")),
                                               int(_h), int(_m), tzinfo=IST),
                        "dir": e.get("dir")}
                except Exception:
                    continue
        except Exception:
            anchors = None
    rows = scout.scan(int(tf_min or 15), date, as_of_dt, anchors=anchors)
    # PRACTICE: the verify line grades t→t+H against the FULL captured file — on a
    # past day that is the future. Blank it while the ghost session runs so the user
    # can't peek; after the real 15:30 the grades unlock and the day self-scores.
    if practice and not _ghost_done():
        for r in rows:
            r["verify"] = None
    # Overlay the live trade brain ONLY for the live 15m view (the alert detector that
    # fills scout-seen scans at 15m). On replay / a different TF the per-bar scan is the
    # honest source, so no overlay.
    seen = seen if (live and int(tf_min or _ALERT_TF) == _ALERT_TF) else {}
    when = ("LIVE" if live else
            f"👻 GHOST {date} @ {as_of_dt:%H:%M} — practice, future hidden" if
            (practice and as_of_dt and not _ghost_done()) else
            f"👻 GHOST {date} @ {as_of_dt:%H:%M} — session over, graded" if
            (practice and as_of_dt) else
            f"replay @ {as_of_dt:%H:%M}" if as_of_dt else f"{date} full day")
    _over = _session_over(live)
    _lmem = _ledger_mem(today, datetime.datetime.now(IST)) if live else {}
    n_trade = sum(1 for r in rows if r.get("has_data") and r["verdict"].startswith("TRADE"))
    hits = sum(1 for r in rows if r.get("verify") and r["verify"]["dir_hit"])
    graded = sum(1 for r in rows if r.get("verify"))
    _pend = ("  ·  live — calls grade themselves once the horizon elapses" if live else
             "  ·  practice — grades unlock at the real 15:30" if practice else
             "  ·  predictions pending (advance the replay clock to grade)")
    sb = (html.Span(f"  ·  scoreboard {hits}/{graded} hit",
                    style={"color": "#22c55e" if hits * 2 >= graded else "#f87171",
                           "fontSize": "0.62rem", "fontWeight": "700"})
          if graded else
          html.Span(_pend, style={"color": "#475569", "fontSize": "0.58rem"}))
    if live:
        # MUST pass as_of — without it the 90m TIMEOUT never fires and the badge disagrees
        # with the popup it opens (badge said "1 open · 6 closed", popup said "0 open · 7").
        # One source of truth for "what is open": _scout_episodes(today, as_of).
        _op, _cl = _scout_episodes(today, as_of=datetime.datetime.now(IST))
        n_open, n_closed = len(_op), len(_cl)
    else:
        n_open = n_closed = 0
    openbtn = (html.Button(
        f"📋 {n_open} open · {n_closed} closed",
        id="scout-openpos-btn", n_clicks=0, title="the day's scout ledger — open positions "
        "(with a live cross-check) + closed episodes (SL / target / flipped) with realized P&L",
        style={"marginLeft": "10px", "fontSize": "0.58rem", "color": "#67e8f9",
               "background": "#0b2530", "border": "1px solid #164e63",
               "borderRadius": "4px", "padding": "1px 8px", "cursor": "pointer"})
        if live else None)
    title = html.Div([
        html.Span(f"🎯 SCOUT — predict next {tf_min}m  ·  {when}  ·  ", title=(
            "GHOST PRACTICE: a past captured session replayed on today's wall clock — "
            "advances by itself, the future is hidden until the real 15:30, then the "
            "day grades your calls. Pick the day via the date strip (bottom bar). "
            "Nothing is written — the next live session is untouched."
            if practice else None), style={
            "color": "#34d399", "fontWeight": "700", "fontSize": "0.7rem",
            "letterSpacing": "0.05em", **({"cursor": "help"} if practice else {})}),
        # "N trades on the board" counted VERDICTS, while the ledger badge beside it counts
        # POSITIONS — two different questions rendered as one answer, so the strip could read
        # "1 trade on the board" next to "0 open" and look self-contradictory. Say which is
        # which, and after 15:30 say the session is done instead of implying something is live.
        html.Span("session closed 15:30 · flat overnight" if _over else
                  f"{n_trade} index verdict{'s' if n_trade != 1 else ''} = TRADE",
                  title="how many indices the scanner currently rates TRADE. This is NOT a "
                        "position count — the 📋 ledger beside it is the authority on what is "
                        "actually open.",
                  style={"color": "#94a3b8", "fontSize": "0.62rem", "cursor": "help"}),
        sb, openbtn,
    ], style={"marginBottom": "5px"})
    note = html.Div([
        html.Span("⛔ MEASURED OPTION P&L (backtest_scout, 8d, n=73): buying the ATM "
                  "CE/PE on the arrow WINS only ~14-23% and BLEEDS −2% to −5% per "
                  "trade (5m/30m loss CIs exclude 0 = significant loser). ",
                  style={"color": "#f87171", "fontWeight": "700"}),
        html.Span("Do NOT buy a naked option off the arrow — it is negative-"
                  "expectancy. The RANGE band (~70% in-band) is the ONLY validated "
                  "product; trade the band/levels, treat the arrow as context only.",
                  style={"color": "#94a3b8"}),
    ], style={"fontSize": "0.54rem", "marginTop": "5px", "lineHeight": "1.35",
              "background": "#1a0c0c", "border": "1px solid #7f1d1d",
              "borderRadius": "4px", "padding": "5px 8px"})
    return html.Div(
        # LIVE rows take their trade memory from the POLLER LOG (one source of truth with the
        # badge + ledger popup), not from the browser's private detector copy, which drifts by
        # minutes and was driving the 90m cap countdown off the wrong clock. Replay/ghost have
        # no poller log for their day, so they keep the per-bar engine's own memory.
        [title] + [_scout_row(r, mem=(_lmem if live else (seen or {})).get(r.get("sym")),
                              today=today, live=live, practice=practice, session_over=_over)
                   for r in rows] + [note]
        + ([_ghost_help()] if practice else []),
        style={"background": "#070d18", "border": "1px solid #1e293b",
               "borderRadius": "6px", "padding": "8px 12px"})


@app.callback(
    Output("charts-scout", "children"),
    Input("charts-tf",   "value"),
    Input("charts-asof", "value"),
    Input("news-date",   "data"),
    Input("sel-sym",     "data"),
    Input("setup-tick",  "n_intervals"),
    State("scout-seen",  "data"),
)
def _update_charts_scout(tf, asof, date, sel, _tick, seen):
    """Multi-index TRADE/NO-TRADE scan. LIVE (no Replay time, today): as_of=now so the
    trade lifecycle (trigger/entry/SL/target/manage/P&L) renders, and the 30s tick
    refreshes the board so a NEW trigger appears on its own. An explicit Replay time
    pins a past instant on the chosen day; a past date with no time = that full day."""
    from dash.exceptions import PreventUpdate
    if sel != "CHARTS":
        raise PreventUpdate
    today = datetime.datetime.now(IST).date().isoformat()
    live = False
    practice = (asof == "ghost")
    if practice:                                       # GHOST-LIVE practice session
        day, hhmm = _ghost_ctx(date)
        as_of_dt = datetime.datetime.fromisoformat(f"{day}T{hhmm}:00+05:30")
    elif asof and asof != "full":                     # explicit Replay minute
        day = date or today
        try:
            as_of_dt = datetime.datetime.fromisoformat(f"{day}T{asof}:00+05:30")
        except Exception:
            day, as_of_dt, live = today, datetime.datetime.now(IST), True
    elif date and date != today:                      # browsing a full PAST day
        day, as_of_dt = date, None
    else:                                              # LIVE now
        day, as_of_dt, live = today, datetime.datetime.now(IST), True
    try:
        return _charts_scout_panel(tf, day, as_of_dt, live=live, seen=seen,
                                   practice=practice)
    except Exception as exc:
        return _recon_note(f"Scout unavailable ({type(exc).__name__}: {exc}).")


# ── Scout alerts: detect lifecycle EVENTS (new / SL / target / exit / band) ──────
def _scout_alert_rec(now, pos, kind, cur=None, spot=None, band_dir=None):
    """Preformatted alert record (head/body/color) from an open-position dict, so the
    panel renderer AND the JS notification show identical wording (no JS format dup)."""
    label = pos["label"]; side = pos["dir"]; strike = pos.get("strike")
    entry = pos.get("entry"); sl = pos.get("sl"); tgt = pos.get("tgt")
    if kind == "NEW":
        act = "CALL BUY" if side == "CE" else "PUT BUY"
        _ek = pos.get("expiry"); _ed = pos.get("expiry_date")
        _exp = (f" · {_ek}{(' ' + _ed) if _ed else ''} exp" if _ek else "")
        head = f"{act} {strike or ''} {side}{_exp}"
        body = f"{label} @ index {pos.get('spot')} · entry ₹{entry}  SL ₹{sl}  T ₹{tgt}"
        color = "#34d399" if side == "CE" else "#f87171"
    elif kind == "SL":
        head = f"🛑 STOP-LOSS HIT · {label}"
        body = f"{strike or ''} {side} hit SL ₹{sl} (entry ₹{entry}, now ₹{cur})"
        color = "#ef4444"
    elif kind == "TARGET":
        head = f"🎯 TARGET HIT · {label}"
        body = f"{strike or ''} {side} booked target ₹{tgt} (entry ₹{entry}, now ₹{cur})"
        color = "#22c55e"
    elif kind == "FLIP":
        newdir = "CE" if side == "PE" else "PE"
        head = f"↺ REVERSED OUT · {label}"
        body = (f"{strike or ''} {side} exited ₹{cur} (entry ₹{entry}) — arrow flipped to "
                f"{newdir}; ⚠ whipsaw, NOT a new position on top (decision-support only)")
        color = "#f59e0b"
    elif kind == "TIMEOUT":
        head = f"⌛ TIMED OUT ({_SCOUT_MAX_HOLD_MIN}m) · {label}"
        body = (f"{strike or ''} {side} force-closed at ₹{cur} (entry ₹{entry}) — held the "
                f"{_SCOUT_MAX_HOLD_MIN}m max with no SL/target/band/flip. A flat index just "
                f"bleeds theta; the slot is now free to re-enter.")
        color = "#a78bfa"
    elif kind == "EOD":
        head = f"🔔 SQUARED OFF AT THE BELL · {label}"
        body = (f"{strike or ''} {side} force-closed at ₹{cur} (entry ₹{entry}) — the cash "
                f"session ended. The arrow is an INTRADAY product: it is flat overnight, "
                f"always. Holding a long option overnight bleeds theta (that is exactly why "
                f"BTST buys FUTURES instead).")
        color = "#94a3b8"
    else:                                                 # BAND
        head = f"📊 RANGE BREAK {band_dir} · {label}"
        body = (f"index {spot} broke the {_ALERT_TF}m band [{pos.get('bl')}, {pos.get('bh')}]"
                " — move bigger than expected")
        color = "#60a5fa"
    return {"t": now.strftime("%H:%M"), "d": now.date().isoformat(),
            "label": label, "kind": kind, "strike": strike, "side": side,
            "head": head, "body": body, "color": color, "thin": pos.get("thin", False),
            # raw fields for the canonical server-side log (intraday_db.scout_alerts)
            "entry": entry, "sl": sl, "tgt": tgt, "cur": cur,
            "spot": spot if spot is not None else pos.get("spot"),
            "band_dir": band_dir}


# Server-side per-index open/last position state. OWNED by the alert poller thread
# (the authoritative detector + writer). A single owner means: alerts are captured
# whether or not a browser tab is open, and there are NO duplicate rows from several
# tabs each running their own detection. The browser callback uses its own (browser)
# copy purely for the charts-overlay/beep and NEVER writes.
_SCOUT_ALERT_STATE: dict = {}


def _scout_detect(state, now, persist):
    """Core scout-alert detection over all 4 indices for one instant.

    Mutates `state` (per-symbol {open, last}) in place and returns the list of fired
    alert recs:
      • NEW    — a TRADE just opened (CALL/PUT buy lean)
      • SL     — the open trade's premium hit the stop
      • TARGET — the open trade booked the target
      • BAND   — index broke the forward range band snapshot at trigger (move > expected)
    Each index carries an OPEN-POSITION that PERSISTS across gate flicker (the verdict
    blinks TRADE/NO-TRADE on a forming bar) until it really exits — so NEW fires once
    (not on every blink) and SL/TARGET are checked gate-independently via the live
    premium (a stop hit during a NO-TRADE blink is still caught).

    persist=True → each fired event is written to the canonical scout_alerts store
    (intraday_db). Used by the server-side poller (authoritative writer) AND the
    browser callback (persist=False — UI/beep only; never writes, so multiple tabs
    can't create duplicate rows). Decision-support only (arrow negative-EV)."""
    import intraday_scout as scout
    today = now.date().isoformat()
    events: list = []
    try:
        rows = scout.scan(_ALERT_TF, today, now)
    except Exception:
        return events

    def _emit(sym, ev):
        events.append(ev)
        if persist:
            try:
                from intraday_db import idb
                idb.write_scout_alert(sym, now, ev)
            except Exception:
                pass

    for r in rows:
        if not r.get("has_data"):
            continue
        sym, v = r["sym"], r["verdict"]
        spot = r.get("spot")
        lc = r.get("lifecycle") or {}
        st = dict(state.get(sym) or {})
        pos = st.get("open")
        if pos and pos.get("day") != today:              # stale position from a prior day
            pos = None
        # ── RE-ARM GATE ────────────────────────────────────────────────────────────
        # A policy close (TIMEOUT/SL/TARGET/BAND) sets a LOCK on that side. Without it the
        # close below frees the slot and the NEW block re-opens the SAME side on the SAME
        # tick (the verdict is still TRADE — that is WHY it was open), which silently
        # DEFEATS both policies: the 90m cap merely restarts its clock, and a "terminal"
        # band break is undone instantly. The arrow would churn a fresh round-trip cost
        # every 90m forever. The lock clears only when the gate genuinely RESETS — the
        # verdict leaves TRADE, or it flips to the other side. A FLIP does NOT lock: the
        # direction really changed, and its NEW is the intended other leg.
        lock = st.get("lock") if st.get("lock_day") == today else None
        if lock and not (v.startswith("TRADE") and r.get("direction") == lock):
            lock = None                                  # gate reset → re-armed
        # ── manage an OPEN position (gate-independent, survives NO-TRADE blinks) ────
        if pos:
            cur = scout._opt_premium(sym, today, now, pos.get("strike"), pos["dir"])
            closed = outcome = None
            # ── MAX-HOLD TIMEOUT — the POLICY, enforced HERE at the source ─────────────
            # Without this the poller holds a no-trigger position FOREVER (it only closed on
            # SL/target/flip). Two consequences, one of them serious:
            #   1. dead positions sat on the board for hours (cosmetic), and
            #   2. the `continue` below SKIPS the "open a NEW position" block -- so a stuck
            #      position BLOCKED EVERY NEW SIGNAL for that index. 2026-07-13: MIDCAP
            #      opened 10:02 and never triggered, so it could not fire a NEW alert for 5
            #      HOURS. That is lost signal, not just clutter.
            # Checked FIRST: past the cap the position was already closed by policy, so a
            # later SL/target must not re-write history (the ledger converts late closes to
            # TIMEOUT for exactly this reason -- now the log carries it natively).
            _open_age = _mem_open_age_min(pos, today, as_of=now)
            if now.time() >= _SCOUT_EOD_T:
                # THE BELL — square off. Checked before the cap so a leg that reaches 15:30
                # is recorded as EOD (what actually happened) rather than mislabelled TIMEOUT.
                _emit(sym, _scout_alert_rec(now, pos, "EOD",
                                            cur=round(cur, 2) if cur is not None else None))
                closed, outcome = True, "EOD"
            elif _open_age is not None and _open_age >= _SCOUT_MAX_HOLD_MIN:
                _emit(sym, _scout_alert_rec(now, pos, "TIMEOUT",
                                            cur=round(cur, 2) if cur is not None else None))
                closed, outcome = True, "TIMEOUT"
            elif cur is not None and pos.get("sl") and cur <= pos["sl"]:
                _emit(sym, _scout_alert_rec(now, pos, "SL", cur=round(cur, 2)))
                closed, outcome = True, "SL"
            elif cur is not None and pos.get("tgt") and cur >= pos["tgt"]:
                _emit(sym, _scout_alert_rec(now, pos, "TARGET", cur=round(cur, 2)))
                closed, outcome = True, "TARGET"
            elif v.startswith("TRADE") and r.get("direction") != pos["dir"]:
                # arrow reversed — EMIT the flip so the log records the prior side exiting
                # (was silently overwritten → looked like two contradictory opens). One
                # coherent stance per index: PE reversed out THEN CE opens, not both held.
                _emit(sym, _scout_alert_rec(now, pos, "FLIP",
                                            cur=round(cur, 2) if cur is not None else None))
                closed, outcome = True, "FLIP"           # flipped to the other side
            elif (spot and pos.get("bl") and pos.get("bh")
                  and (spot > pos["bh"] or spot < pos["bl"])):
                # ── BAND BREAK IS TERMINAL ────────────────────────────────────────────
                # It used to only set pos["bb"]=True and KEEP HOLDING, while the ledger
                # (and the user's rule: "any band touch = the trade is closed") treated it
                # as a close. That split was the root of everything: the poller sat on a
                # position the ledger had already closed, and because a held position
                # `continue`s past the NEW block, it went DEAF. 2026-07-13: BANK broke its
                # band at 10:29 and could not fire another signal for 6+ HOURS — its 15:15
                # TRADE never reached the log. All four indices were blocked this way.
                # The band is the one VALIDATED product (~77% in-band); a break means the
                # move exceeded the forecast, so the thesis is spent. Close it.
                pos["bb"] = True
                _emit(sym, _scout_alert_rec(
                    now, pos, "BAND", spot=spot,
                    cur=round(cur, 2) if cur is not None else None,
                    band_dir=("above" if spot > pos["bh"] else "below")))
                closed, outcome = True, "BAND"
            if not closed:
                st["open"] = pos
                st["lock"], st["lock_day"] = lock, today
                state[sym] = st
                continue
            if outcome != "FLIP":
                lock = pos["dir"]        # policy close → no same-side re-entry until reset
            # position closed → record the RESOLVED episode so the charts board can
            # still show what happened (instead of the trade silently vanishing), then
            # free the slot to re-open.
            st["last"] = {"day": today, "dir": pos["dir"], "strike": pos.get("strike"),
                          "entry": pos.get("entry"), "sl": pos.get("sl"),
                          "tgt": pos.get("tgt"), "trig": pos.get("trig"),
                          "band_broke": bool(pos.get("bb")),
                          "outcome": outcome, "cur": round(cur, 2) if cur is not None else None,
                          "closed_t": now.strftime("%H:%M"), "label": pos["label"]}
            st["open"] = None
        # ── open a NEW position when a trade is live, none is open, and the side is
        #    RE-ARMED (see the lock above — a stopped/timed-out/band-broken side stays
        #    shut until the gate resets, instead of re-entering on the very next tick) ──
        if (v.startswith("TRADE") and r.get("direction") != lock
                and now.time() < _SCOUT_NO_NEW_AFTER):
            pos = {"day": today, "dir": r.get("direction"),
                   "strike": lc.get("entry_strike") or r.get("atm"),
                   "entry": lc.get("entry_prem"), "sl": lc.get("sl"), "tgt": lc.get("target"),
                   "bl": r.get("pred_lo"), "bh": r.get("pred_hi"), "bb": False,
                   "trig": now.strftime("%H:%M"), "expiry": r.get("expiry"),
                   "expiry_date": r.get("expiry_date"),
                   "label": r["label"], "thin": bool(r.get("thin")), "spot": spot}
            _emit(sym, _scout_alert_rec(now, pos, "NEW"))
            st["open"] = pos
            lock = None                                  # a fresh leg is armed by definition
        st["lock"], st["lock_day"] = lock, today
        state[sym] = st
    return events


def _rehydrate_scout_state(state, today):
    """Rebuild the per-symbol open-position state from the persisted scout_alerts log
    on poller startup. Without this, a mid-day restart resets _SCOUT_ALERT_STATE to
    empty → every still-open trade re-fires as a DUPLICATE 'NEW' and in-flight SL/
    TARGET tracking is lost (the morning of a trade silently re-opens). Replays the
    day's NEW/BAND/SL/TARGET rows per symbol and restores the last UNRESOLVED episode
    so detection resumes exactly where it left off. Band levels (pred_lo/hi) aren't
    stored, so a post-restart band break only re-fires once the trade re-opens — an
    accepted, conservative loss vs. spamming dup alerts."""
    import pandas as pd
    from core.mirror_io import read_mirror
    df = read_mirror("scout_alerts", today)
    if df is None or df.empty:
        return 0
    def _v(x):
        return None if x is None or (isinstance(x, float) and pd.isna(x)) else x
    for sym, g in df.groupby("symbol"):                  # df is ts-ascending
        st: dict = {}
        for _, r in g.iterrows():
            kind = str(r.get("kind") or "")
            if kind == "NEW":
                strike = _v(r.get("strike"))
                st["open"] = {
                    "day": today, "dir": _v(r.get("side")),
                    "strike": int(strike) if strike is not None else None,
                    "entry": _v(r.get("entry")), "sl": _v(r.get("sl")),
                    "tgt": _v(r.get("tgt")), "bl": None, "bh": None, "bb": False,
                    "trig": r["ts"].strftime("%H:%M"),
                    "label": _v(r.get("label")) or str(sym),
                    "thin": bool(r.get("thin")), "spot": _v(r.get("spot"))}
                st["lock"] = st["lock_day"] = None       # a leg that re-opened was armed
            elif kind in ("BAND", "TIMEOUT", "EOD", "SL", "TARGET", "FLIP") and st.get("open"):
                # ALL FIVE are terminal — the same close policy the poller enforces. BAND used
                # to only set bb=True and KEEP the position, and TIMEOUT was not handled at
                # all, so a restart RESURRECTED a position the log had already closed: the
                # poller then sat on a dead leg and (because a held position skips the NEW
                # block) that index went DEAF for the rest of the day. A container restart is
                # routine — deploy_vm.bat does one — so this path had to match the engine.
                side = _v(r.get("side")) or st["open"].get("dir")
                st["open"] = None                        # episode closed → free the slot
                if kind != "FLIP":                       # re-entry lock survives the restart
                    st["lock"], st["lock_day"] = side, today
        # MAX-HOLD, applied HERE too. A LEGACY log (written before the poller enforced the
        # cap) carries no TIMEOUT row, so the loop above leaves the position open while the
        # ledger — which converts a dangling tail at cap_t — already closed it. A restart
        # would then resurrect a leg the ledger calls dead and go deaf on that index. The cap
        # is the policy: engine, ledger and rehydrator must all apply it or they disagree.
        op = st.get("open")
        if op:
            age = _mem_open_age_min(op, today)       # live: rehydrate only runs at startup
            if age is not None and age >= _SCOUT_MAX_HOLD_MIN:
                st["open"] = None
                st["lock"], st["lock_day"] = op.get("dir"), today
        if st.get("open") or st.get("lock"):
            # keep the row even with NO open position: it may carry only the LOCK, and
            # dropping it would re-arm a stopped-out side for free across a restart.
            state[str(sym)] = st
    return sum(1 for v in state.values() if v.get("open"))


def _backfill_scout_alerts(today, upto, write_before=None, step_sec=300):
    """Reconstruct EVERY missed scout alert for the day by replaying the as_of-safe
    detector over the already-captured market data (candles/OI/futures/chain mirrors),
    from 09:15 up to `upto` on a coarse grid. The scout is replay-DETERMINISTIC, so a
    replay produces exactly the alerts that WOULD have fired — making the alert log
    COMPLETE and *derivable from the captured data*, independent of when the capturer
    process actually started (a mid-session feature launch, a crash, or a redeploy no
    longer leaves a hole in the morning). Writes via the normal idb path
    (ON CONFLICT DO NOTHING); any same-minute overlap with rows the live loop already
    wrote collapses in the panel's minute-level dedup. A held position only emits its
    NEW once (at its true open time) because the rebuilt state persists across the
    walk, so the afternoon's live rows are not re-created. Returns (n_events, state)
    so the poller continues live from the reconstructed open-position state.

    `write_before` (the earliest timestamp the live log already holds): rows at/after
    it are NOT written (the live loop already logged that span) — detection still runs
    to keep the rebuilt STATE correct up to `now`, but persist is off, so a position
    that straddles the boundary is not double-listed with two slightly different times.
    Bounded: ONE pass, 5-min grid (~75 scans/full day, ~2-3 min, GC between steps)."""
    state: dict = {}
    d0 = datetime.datetime.fromisoformat(today).date()
    ts = datetime.datetime.combine(d0, datetime.time(9, 15), IST)
    cap = datetime.datetime.combine(d0, datetime.time(15, 30), IST)
    upto = min(upto, cap)
    step = datetime.timedelta(seconds=step_sec)
    n = 0
    while ts <= upto:
        try:
            persist = write_before is None or ts < write_before
            ev = _scout_detect(state, ts, persist=persist)
            if persist:
                n += len(ev)
        except Exception:
            pass
        ts += step
    return n, state


# Health of the authoritative writer. A dead poller must be VISIBLE: the ledger otherwise
# renders a calm "0 open" while live positions sit on the board unrecorded.
_SCOUT_POLLER_HEALTH = {"last_ok": None, "fails": 0, "last_err": None, "last_err_t": None}


def _scout_alert_poller():
    """AUTHORITATIVE scout-alert detector — a server-side background thread (started
    only by the live capturer, not in VIEWER mode). Scans all 4 indices every 30s
    during 09:15-15:30 and writes every NEW/SL/TARGET/BAND to the canonical
    scout_alerts store, WHETHER OR NOT a browser is open. This is the system of
    record; the browser just reads it. Owns _SCOUT_ALERT_STATE so position tracking
    (and thus SL/TARGET/dedup) is single-sourced. Decision-support only.

    BOOT RECOVERY — the log is always made whole from 09:15:
      • if the persisted log already reaches back near the open (a normal pre-open
        boot), just REHYDRATE the open-position state (instant); else
      • REPLAY the captured market data to reconstruct every alert that fired before
        this process existed (mid-day cold start / late feature launch), so the
        morning is never missing again."""
    import time as _time
    today = datetime.datetime.now(IST).date().isoformat()
    try:
        from core.mirror_io import read_mirror
        now = datetime.datetime.now(IST)
        if not is_trading_day(now):
            raise RuntimeError("non-trading day — no session to backfill")
        df = read_mirror("scout_alerts", today)
        existing_min = (df["ts"].min() if (df is not None and not df.empty) else None)
        # "morning covered" = the log already holds an alert from before 11:00 → the
        # capturer was clearly running in the morning, so just rehydrate (cheap). A
        # later/empty earliest means a cold mid-day start → reconstruct the gap. (No
        # alert can fire before the 09:45 open-gate, and quiet mornings are rare, so
        # this avoids a needless full-day re-walk on every restart.)
        have_open = existing_min is not None and existing_min.time() <= datetime.time(11, 0)
        if (not have_open) and now.time() >= datetime.time(9, 20):
            t0 = _time.time()
            # write only the missing span [09:15, existing_min); detection still walks
            # to `now` to rebuild current state without double-listing live rows.
            n, st = _backfill_scout_alerts(today, now, write_before=existing_min)
            _SCOUT_ALERT_STATE.clear()
            _SCOUT_ALERT_STATE.update(st)
            opens = sum(1 for v in st.values() if v.get("open"))
            print(f"  Scout-alert BACKFILL — reconstructed {n} alert(s) from 09:15 "
                  f"over captured data in {_time.time()-t0:.0f}s ({opens} still open); "
                  f"morning log now complete")
        else:
            k = _rehydrate_scout_state(_SCOUT_ALERT_STATE, today)
            print(f"  Scout-alert state rehydrated from today's log "
                  f"({k} open position(s)) — log already covers the open, restart-safe")
    except Exception as e:
        print(f"  Scout-alert boot recovery skipped: {e}")
    fails = 0
    while True:
        try:
            now = datetime.datetime.now(IST)
            # trading-day gate: a weekend/holiday process otherwise scans the dead
            # static quote feed every 30s with persist=True — one feed glitch away
            # from writing junk alerts into the canonical log on a non-session day.
            # Upper bound runs a couple of minutes PAST the bell on purpose: the EOD square-off
            # lives inside _scout_detect, and with a 30s tick a gate of "<= 15:30:00" can skip
            # straight from 15:29:58 to 15:30:28 and never call it — leaving the open leg
            # dangling exactly as before. The extra ticks open nothing (_SCOUT_NO_NEW_AFTER)
            # and simply flatten what is open.
            if (is_trading_day(now)
                    and datetime.time(9, 15) <= now.time() <= datetime.time(15, 32)):
                _scout_detect(_SCOUT_ALERT_STATE, now, persist=True)
                _SCOUT_POLLER_HEALTH.update(last_ok=now, fails=0, last_err=None)
                fails = 0
        except Exception as e:
            # NEVER swallow this silently. This thread is the SYSTEM OF RECORD: if it
            # stops writing, the ledger silently shows "0 open" while real positions are
            # live on the board — which reads as "you have no position". It has already
            # happened (2026-07-13: capture ran to 15:30, alerts stopped at 14:23, and a
            # bare `except: pass` meant nothing surfaced it).
            fails += 1
            _SCOUT_POLLER_HEALTH.update(fails=fails, last_err=f"{type(e).__name__}: {e}",
                                        last_err_t=datetime.datetime.now(IST))
            if fails in (1, 2, 5, 10) or fails % 20 == 0:
                print(f"  ⚠ SCOUT-ALERT POLLER FAILING ({fails}x) — the alert log is NOT "
                      f"being written: {type(e).__name__}: {e}")
                traceback.print_exc()
        _time.sleep(30)


# Server ROLE, set once in __main__ (_resolve_role). Callbacks read it at runtime to
# pick the authoritative alert source: capturer = local browser detector (fresh data),
# viewer = the synced scout_alerts log (avoids phantom re-strike notifications).
_ROLE_VIEWER = False
# Per-day seed for viewer notifications: adopt the day's already-logged alerts silently
# on first sight, then fire only on a genuinely NEW synced row. Per-process (resets on
# restart → re-seeds silently, correct); day-keyed so a new session doesn't misfire.
_VIEWER_ALERT_SEED = {"day": None, "n": 0}


@app.callback(
    Output("scout-seen",       "data"),
    Output("scout-alerts",     "data"),
    Output("scout-alert-fire", "data"),
    Input("setup-tick", "n_intervals"),
    State("scout-seen",       "data"),
    State("scout-alerts",     "data"),
    State("scout-alert-fire", "data"),
)
def _detect_scout_alerts(_tick, seen, alerts, fire):
    """Browser-side mirror of the scout-alert detector — UI ONLY (never writes; the
    server-side _scout_alert_poller is the sole authoritative writer). Maintains the
    browser's scout-seen so the CHARTS board overlay shows HOLDING / last-trade, and
    bumps the fire counter → clientside notification + beep. The ALERTS panel itself
    reads the canonical log, not this list. Decision-support only (arrow negative-EV)."""
    from dash.exceptions import PreventUpdate
    now = datetime.datetime.now(IST)
    if not (is_trading_day(now)
            and datetime.time(9, 15) <= now.time() <= datetime.time(15, 30)):
        raise PreventUpdate
    seen = dict(seen or {})
    today = now.date().isoformat()
    events = _scout_detect(seen, now, persist=False)   # persist=False → no dup rows
    # purge any prior-day alerts on the first tick of a new session (local store persists
    # across days) so the panel only ever shows TODAY's trade history.
    log = [a for a in (alerts or []) if a.get("d") == today]

    if _ROLE_VIEWER:
        # VIEWER: the VM is the sole authoritative detector. Notify off newly-SYNCED
        # scout_alerts rows (what the ledger shows), NOT the local re-scan above — a
        # viewer re-scan runs on lagged mirrors with its OWN state, re-strikes the ATM
        # independently, and pops PHANTOM trades that exist in no log (the 56700-PE
        # confusion). The local scan still refreshed `seen` for the board overlay only.
        recs = _alerts_from_mirror(today) or []
        s = _VIEWER_ALERT_SEED
        prev_n = s["n"] if s["day"] == today else None
        s["day"] = today
        if prev_n is None:                         # first sight of the day → adopt history
            s["n"] = len(recs)                     # silently (no notification burst on load)
            return seen, recs[:50], no_update
        if len(recs) > prev_n:                      # a genuinely new authoritative alert
            s["n"] = len(recs)
            return seen, recs[:50], int(fire or 0) + 1
        if len(recs) != prev_n or recs[:50] != log:  # shrank/deduped → resync list, no fire
            s["n"] = len(recs)
            return seen, recs[:50], no_update
        return seen, no_update, no_update

    # CAPTURER (incl. anyone viewing the VM URL): the browser detector runs on the same
    # host's FRESH data + mirrors the poller, so its local events are authoritative here.
    if events:
        return seen, (events + log)[:50], int(fire or 0) + 1
    if log != list(alerts or []):                # purged stale rows → write back
        return seen, log, no_update
    return seen, no_update, no_update          # persist state; no re-render churn


@app.callback(
    Output("alert-badge", "children"),
    Output("alert-badge", "style"),
    Input("setup-tick",   "n_intervals"),
    Input("scout-alerts", "data"),
    State("charts-asof",  "value"),
    State("news-date",    "data"),
)
def _alert_badge(_n, alerts, asof, date):
    # Count the CANONICAL deduped log for today — the SAME source the panel shows, so
    # the badge and the list always agree (the browser localStorage list was dup-
    # inflated). Fall back to the browser list only very early before the mirror exists.
    today = datetime.datetime.now(IST).date().isoformat()
    if asof == "ghost":                    # practice: badge counts the ghost day's
        day, hhmm = _ghost_ctx(date)       # alerts up to the ghost clock — SAME day
        recs = _alerts_from_mirror(day)    # resolution as the panel, so they agree
        n = len([a for a in recs if str(a.get("t", ""))[:5] <= hhmm]
                if not _ghost_done() else recs)
    else:
        n = len(_alerts_from_mirror(today))
    if not n:
        n = len([a for a in (alerts or []) if a.get("d", today) == today])
    base = {"fontSize": "0.55rem", "fontWeight": "800", "color": "#0a0f1a",
            "background": "#fbbf24", "borderRadius": "9px", "padding": "0 6px",
            "minWidth": "16px", "textAlign": "center"}
    if not n:
        return "", {**base, "display": "none"}
    return str(n), {**base, "display": "inline-block"}


_ALERT_KIND_COLOR = {"SL": "#ef4444", "TARGET": "#22c55e",
                     "BAND": "#60a5fa", "NEW": "#34d399", "FLIP": "#f59e0b"}


def _alerts_from_mirror(date):
    """Read the canonical server-side alert log (intraday_db.scout_alerts mirror) for
    a day → list of UI alert recs (newest first). This is the authoritative record:
    survives a browser cache-clear, is the SAME on every device, captures alerts that
    fired while only the VM was watching, and is archived per-day for evening review.
    Returns [] if the mirror is missing (e.g. a day before this log existed).

    DEDUPED on two levels so the panel shows the TRUE distinct-alert count:

    1) POSITION-AWARE (NEW): a NEW is only real if the prior position on that exact
       (symbol, strike, side) actually CLOSED (an SL/TARGET row for the symbol). The
       scout holds ONE position per index until it exits, so a second NEW for a still-
       open (strike, side) is a re-fire — the old per-tab browser detector and pre-
       rehydrate restarts both re-emitted open trades as fresh NEWs. Those are dropped;
       a genuine re-entry (after an SL/TARGET) and a flip to the other side survive.
    2) EVENT-WINDOW (all kinds): same (symbol, kind, strike, band_dir) within
       _DEDUP_WIN s = one row (earliest kept) — collapses writer-timing twins (legacy
       multi-tab; 5-min backfill grid time vs the live 30s wall-clock for a position
       straddling the boundary). Genuinely separate events (a 2nd break of the same
       wall 30 min later) survive."""
    from core.mirror_io import read_mirror
    df = read_mirror("scout_alerts", date)
    if df is None or df.empty:
        return []
    _DEDUP_WIN = 240                                 # seconds: same event if within 4 min
    out, last, open_pos = [], {}, {}                 # base→ts; symbol→current open (strike,side)
    for _, r in df.iterrows():                       # df is ts-ascending → keep earliest
        kind = str(r.get("kind") or "")
        ts = r["ts"]
        sym = str(r.get("symbol") or "")
        # ── position bookkeeping: the scout holds ONE position per index. A NEW that
        # repeats the CURRENT open (strike,side) is a re-fire (restart / multi-tab);
        # any genuinely different NEW (new strike, or a flip to the other side)
        # REPLACES it and is kept. SL/TARGET frees the slot. ─────────────────────────
        if kind == "NEW":
            key = (r.get("strike"), r.get("side"))
            if open_pos.get(sym) == key:
                continue                             # still-open position re-emitted → dup
            open_pos[sym] = key
        elif kind in ("SL", "TARGET"):
            open_pos.pop(sym, None)                  # the open position closed → free it
        # ── timing-twin collapse (writer skew) ───────────────────────────────────────
        base = (sym, kind, r.get("strike"), r.get("band_dir"))
        prev = last.get(base)
        if prev is not None and (ts - prev).total_seconds() < _DEDUP_WIN:
            continue                                 # same event, just a timing twin
        last[base] = ts
        out.append({
            "t": ts.strftime("%H:%M"), "d": date, "kind": kind,
            "head": r.get("head") or "", "body": r.get("body") or "",
            "color": _ALERT_KIND_COLOR.get(kind, "#e2e8f0"),
            "thin": bool(r.get("thin")),
        })
    out.reverse()   # newest first (mirror is ts-ascending)
    return out


@app.callback(
    Output("alerts-content", "children"),
    Input("setup-tick",   "n_intervals"),
    Input("scout-alerts", "data"),
    Input("sel-sym",      "data"),
    Input("news-date",    "data"),
    Input("alert-hour",   "value"),
    State("charts-asof",  "value"),
)
def _render_alerts(_tick, alerts, sel, date, hour, asof):
    # Re-reads the canonical log on the 30s setup-tick (same cadence as the badge), so a
    # newly fired alert appears at the TOP of the list automatically — the panel tracks
    # the system-of-record on a timer, NOT the browser's parallel detection (which could
    # leave the list lagging the badge). Also re-renders on section/date/hour change.
    from dash.exceptions import PreventUpdate
    if sel != "ALERTS":
        raise PreventUpdate
    today = datetime.datetime.now(IST).date().isoformat()
    date = date or today
    # GHOST practice: replay the ghost day's ARCHIVED alerts up to the ghost clock —
    # each alert "fires" in the panel at the minute it fired that day; the afternoon
    # stays hidden (future). The canonical log is read-only here, nothing is written.
    ghost_hhmm = None
    if asof == "ghost":
        date, ghost_hhmm = _ghost_ctx(date)
    # Source of truth = the canonical server-side log (archived, device-independent).
    # Fall back to the per-browser localStorage list only for TODAY when the mirror is
    # not yet written (very start of session / pure replay box).
    recs = _alerts_from_mirror(date)
    if ghost_hhmm and not _ghost_done():
        recs = [a for a in recs if str(a.get("t", ""))[:5] <= ghost_hhmm]
    if not recs and date == today:
        recs = [a for a in (alerts or []) if a.get("d", today) == today]

    note = html.Div([
        html.Span("⛔ Decision-support only. ", style={"color": "#f87171", "fontWeight": "700"}),
        html.Span("The CE/PE arrow is measured negative-EV — these are structural "
                  "leans, NOT buy instructions. Trade the range band / levels.",
                  style={"color": "#94a3b8"}),
    ], style={"fontSize": "0.56rem", "marginBottom": "8px", "background": "#1a0c0c",
              "border": "1px solid #7f1d1d", "borderRadius": "4px", "padding": "5px 8px"})

    if not recs:
        msg = ("No triggers yet today — the board is scanned every 30s during market hours."
               if date == today else f"No scout alerts logged on {date}.")
        return html.Div([note, html.Div(
            msg, style={"color": "#475569", "fontSize": "0.7rem", "padding": "10px"})])

    # ── hour filter: show only the selected trading hour's alerts (tally over it) ────
    def _hr(a):
        try:
            return int(str(a.get("t", ""))[:2])
        except ValueError:
            return -1
    if hour not in (None, "all"):
        shown = [a for a in recs if _hr(a) == int(hour)]
        scope = f"{int(hour):02d}:00–{int(hour):02d}:59"
    else:
        shown = recs
        scope = "all day"

    # ── tally header: how many SL / TARGET / BAND / NEW in the shown scope ───────────
    tally = {}
    for a in shown:
        tally[a.get("kind", "")] = tally.get(a.get("kind", ""), 0) + 1
    chips = []
    for kind, glyph in (("NEW", "▶ opened"), ("TARGET", "🎯 target"),
                        ("SL", "🛑 stop"), ("FLIP", "↺ flipped"), ("BAND", "📊 band")):
        n = tally.get(kind, 0)
        chips.append(html.Span(
            f"{glyph}: {n}",
            style={"color": _ALERT_KIND_COLOR.get(kind, "#94a3b8") if n else "#475569",
                   "fontWeight": "700", "marginRight": "12px"}))
    summary = html.Div(
        [html.Span(f"{date} · {scope}  ", style={"color": "#64748b", "fontWeight": "700"})] + chips,
        style={**MONO, "fontSize": "0.62rem", "marginBottom": "8px", "padding": "5px 8px",
               "background": "#0b1220", "border": "1px solid #1e293b", "borderRadius": "4px"})

    if not shown:
        return html.Div([note, summary, html.Div(
            f"No alerts in {scope}.",
            style={"color": "#475569", "fontSize": "0.7rem", "padding": "10px"})])

    rows = []
    for a in shown:
        bits = [
            html.Span(f"{a['t']}  ", style={"color": "#fbbf24", "fontWeight": "700"}),
            html.Span(a.get("head", ""), style={"color": a.get("color", "#e2e8f0"),
                                                "fontWeight": "700"}),
            html.Span("  " + a.get("body", ""), style={"color": "#cbd5e1"}),
        ]
        if a.get("thin"):
            bits.append(html.Span("  ⚠ thin", style={"color": "#f59e0b"}))
        rows.append(html.Div(bits, style={**MONO, "fontSize": "0.66rem", "padding": "5px 8px",
                                          "borderBottom": "1px solid #1e293b"}))
    return html.Div([note, summary] + rows)


# Clientside: on a new alert, browser notification + a short beep.
app.clientside_callback(
    """
    function(fireN, alerts){
        if(!fireN || !alerts || !alerts.length){ return ''; }
        var a = alerts[0];
        try {
            if ('Notification' in window && Notification.permission === 'granted'){
                new Notification(a.head + '  ' + a.t, { body: a.body });
            }
        } catch(e){}
        try {
            var AC = window.AudioContext || window.webkitAudioContext;
            var ctx = new AC();
            var o = ctx.createOscillator(), g = ctx.createGain();
            o.connect(g); g.connect(ctx.destination);
            o.type = 'sine'; o.frequency.value = 880; g.gain.value = 0.12;
            o.start(); o.stop(ctx.currentTime + 0.18);
        } catch(e){}
        return '';
    }
    """,
    Output("alert-noop", "data"),
    Input("scout-alert-fire", "data"),
    State("scout-alerts", "data"),
)

# Clientside: request notification permission on the button click (a user gesture,
# which browsers require to show the permission prompt).
app.clientside_callback(
    """
    function(n){
        if(!n){ return ''; }
        if(!('Notification' in window)){ return 'This browser has no notifications.'; }
        Notification.requestPermission();
        return 'Allow notifications in the browser prompt — then alerts pop even off-tab.';
    }
    """,
    Output("alert-perm-status", "children"),
    Input("alert-perm-btn", "n_clicks"),
    prevent_initial_call=True,
)


@app.callback(
    Output("charts-recon", "children"),
    Input("charts-mode", "value"),
    Input("charts-idx",  "value"),
    Input("charts-asof", "value"),
    Input("news-date",   "data"),
    Input("sel-sym",     "data"),
)
def _update_charts_recon(mode, sym, asof, date, sel):
    """Descriptive overnight→now positioning map under the chart (Options mode only)."""
    from dash.exceptions import PreventUpdate
    if sel != "CHARTS":
        raise PreventUpdate
    if mode == "futures":
        return ""        # positioning map is an options-chain read
    sym, date = sym or "NSE:NIFTY50-INDEX", date or None
    if asof == "ghost":                        # practice: pin to the ghost clock/day
        date, asof = _ghost_ctx(date)
    as_of_dt = None
    if asof and asof != "full" and date:
        try:
            as_of_dt = datetime.datetime.fromisoformat(f"{date}T{asof}:00+05:30")
        except Exception:
            as_of_dt = None
    try:
        return _charts_recon_panel(sym, date, as_of_dt)
    except Exception as exc:
        return _recon_note(f"Positioning map unavailable ({type(exc).__name__}).")


@app.callback(
    Output("holiday-banner", "children"),
    Output("holiday-banner", "style"),
    Input("news-date", "data"),
)
def _update_holiday_banner(date):
    """Show an NSE-holiday / weekend banner for the VIEWED date, so a closed day's
    empty panels read as 'market closed' rather than 'broken / warming up'."""
    from core.market_calendar import holiday_name
    d = date or datetime.datetime.now(tz=IST).date().isoformat()
    try:
        dd = datetime.date.fromisoformat(str(d)[:10])
    except Exception:
        return "", {"display": "none"}
    hn = holiday_name(dd)
    if hn:
        msg, clr, bg, bd = (f"🟠  NSE HOLIDAY — {hn} · markets closed · {dd:%d %b %Y}",
                            "#fbbf24", "#2a1f06", "#854d0e")
    elif dd.weekday() >= 5:
        msg, clr, bg, bd = (f"⚪  Weekend — markets closed · {dd:%d %b %Y}",
                            "#94a3b8", "#0f172a", "#1e293b")
    else:
        return "", {"display": "none"}
    return msg, {"display": "block", "marginTop": "8px", "padding": "5px 12px",
                 "background": bg, "border": f"1px solid {bd}", "borderRadius": "6px",
                 "color": clr, "fontSize": "0.66rem", "fontWeight": "700",
                 "letterSpacing": "0.04em", **MONO}


# ── Callback 2: toggle panels + highlight selected nav card ───────────────────
@app.callback(
    Output("overview-panel",  "style"),
    Output("oc-panel",        "style"),
    Output("trade-book-panel", "style"),
    Output("live-oi-panel",   "style"),
    Output("charts-panel",    "style"),
    Output("alerts-panel",    "style"),
    Output("tradeboard-panel", "style"),
    Output("oc-title",        "children"),
    *[Output(f"nav-{_slug(s)}", "style") for s in INDEX_SYMBOLS],
    Input("sel-sym", "data"),
)
def toggle_view(sym):
    is_trades = (sym == "TRADES")
    is_liveoi = (sym == "LIVEOI")
    is_charts = (sym == "CHARTS")
    is_alerts = (sym == "ALERTS")
    is_tboard = (sym == "TRADEBOARD")
    is_index  = bool(sym) and not is_trades and not is_liveoi and not is_charts \
        and not is_alerts and not is_tboard

    ov_style = {"display": "block"} if not sym else {"display": "none"}
    oc_style = {"display": "block"} if is_index else {"display": "none"}
    tb_style = {"display": "block"} if is_trades else {"display": "none"}
    lo_style = {"display": "block"} if is_liveoi else {"display": "none"}
    ch_style = {"display": "block"} if is_charts else {"display": "none"}
    al_style = {"display": "block"} if is_alerts else {"display": "none"}
    tbd_style = {"display": "block"} if is_tboard else {"display": "none"}

    if is_index:
        color = COLORS[sym]
        title = html.Div([
            html.Span(LABELS[sym], style={
                "color": color, "fontWeight": "700",
                "fontSize": "0.9rem", "letterSpacing": "0.1em",
            }),
            html.Span("  OPTION CHAIN", style={"color": "#334155", "fontSize": "0.75rem"}),
        ])
    else:
        title = ""

    nav_styles = [_nav_chip_style(s, selected=(s == sym)) for s in INDEX_SYMBOLS]
    return (ov_style, oc_style, tb_style, lo_style, ch_style, al_style, tbd_style,
            title, *nav_styles)


# ── Callback 3: sidebar live prices + status ───────────────────────────────────
@app.callback(
    [Output(f"s-ltp-{_slug(s)}", "children") for s in INDEX_SYMBOLS] +
    [Output(f"s-chg-{_slug(s)}", "children") for s in INDEX_SYMBOLS] +
    [Output(f"s-chg-{_slug(s)}", "style")    for s in INDEX_SYMBOLS] +
    [Output("status", "children")],
    Input("fast-tick", "n_intervals"),
)
def update_sidebar(_):
    with _lock:
        latest = {s: dict(t) for s, t in _latest.items()}
    now = datetime.datetime.now(tz=IST).strftime("%H:%M:%S IST")
    ltps, chgs, stys = [], [], []
    for sym in INDEX_SYMBOLS:
        t = latest.get(sym)
        if not t:
            ltps.append("—"); chgs.append(""); stys.append(MONO); continue
        ltp = t.get("ltp", 0); ch = t.get("ch", 0); chp = t.get("chp", 0)
        up  = ch >= 0; clr = "#22c55e" if up else "#ef4444"
        ltps.append(f"{ltp:,.2f}")
        chgs.append(f"{'▲' if up else '▼'} {'+' if up else ''}{ch:,.2f} ({'+' if up else ''}{chp:.2f}%)")
        stys.append({**MONO, "fontSize": "0.68rem", "color": clr})
    n = len(latest)
    dot_c = "#22c55e" if n == 4 else "#f59e0b" if n else "#ef4444"
    lbl   = "LIVE" if n == 4 else f"PARTIAL {n}/4" if n else "CONNECTING..."
    # Write-health badge — a systematic DB insert failure (schema drift, the chain_snapshots
    # bug class) OR a frozen parquet export (mirror stops advancing) is counted in idb but was
    # otherwise only in the log. Show it RED on the header so silent data loss is impossible to
    # miss. 0 errors → no badge (no clutter).
    try:
        from intraday_db import idb as _idb
        _ierr = _idb.insert_error_count() + _idb.export_error_count()
    except Exception:
        _ierr = 0
    werr = (html.Span(f"  ⚠ WRITE ERR {_ierr}",
                      style={"color": "#ef4444", "fontWeight": "700"})
            if _ierr else "")
    # Degradation badge — swallowed computation failures (core.obs.warn_once) surface here on
    # the HEADER, not only stderr (the chain_snapshots lesson: buried logs = invisible). AMBER,
    # not red: a defaulted field is softer than write data-loss, but it may be feeding a wrong
    # number into a live score, so it must be seen. Count = distinct failing sites. 0 → hidden.
    try:
        from core.obs import warn_counts as _wc
        _degr = len(_wc())
    except Exception:
        _degr = 0
    degr = (html.Span(f"  ⚠ DEGRADED {_degr}",
                      title="Silently-swallowed computation failures (a defaulted field may "
                            "be feeding a wrong number into a score). See stderr/logs for the "
                            "file:line contexts.",
                      style={"color": "#f59e0b", "fontWeight": "700", "cursor": "help"})
            if _degr else "")
    # Capture-freshness badge — presence-based LIVE/PARTIAL/CONNECTING can't see a FROZEN
    # feed: a stale _latest (VM capture died, or in viewer mode the sync stopped) still reads
    # "● LIVE". During a live session only, if the freshest tick has aged past the tolerance
    # (sync 60s + mirror-export lag → a healthy viewer can trail ~2min; 240s avoids false
    # alarms, catches a real death within 4min), surface it RED so a silent gap is impossible
    # to miss. Outside the session / on a past day it stays hidden (no clutter after close).
    # On a VIEWER the label must NOT say "CAPTURE" -- the VM is usually capturing perfectly and
    # it is the LOCAL SYNC that died. Blaming capture sends you to fix the wrong machine (it did,
    # 2026-07-13). _viewer_mirror_health() tells the two apart; the full-width banner carries the
    # detail, this badge is just the always-visible marker.
    stale = ""
    try:
        _h = _viewer_mirror_health()
        if _h and _h[0] != "OK":
            _txt = ("  ⛔ SYNC DEAD — SCREEN STALE" if _h[0] == "SYNC_DEAD"
                    else "  ⛔ VM CAPTURE DOWN — SCREEN STALE")
            stale = html.Span(_txt, style={"color": "#ef4444", "fontWeight": "800"})
        elif _h is None:                      # CAPTURER: keep the original in-memory feed check
            from core.market_calendar import is_trading_day
            _nowdt = datetime.datetime.now(tz=IST)
            if (is_trading_day(_nowdt.date())
                    and datetime.time(9, 15) <= _nowdt.time() <= datetime.time(15, 31)):
                _fts = [float((t or {}).get("exch_feed_time") or 0) for t in latest.values()]
                _newest = max(_fts) if _fts else 0.0
                _age = (_nowdt.timestamp() - _newest) if _newest else 1e9
                if _age > 240:
                    stale = html.Span(f"  ⚠ CAPTURE STALE {int(_age)}s",
                                      style={"color": "#ef4444", "fontWeight": "700"})
    except Exception:
        stale = ""
    status = html.Span([html.Span("● ", style={"color": dot_c}),
                        html.Span(f"{lbl}  ·  {now}", style={"color": "#334155"}),
                        werr, degr, stale])
    return ltps + chgs + stys + [status]


# ── Callback 4: overview cards + chart ────────────────────────────────────────
_OV_OUTPUTS = (
    [Output(f"ov-ltp-{_slug(s)}",   "children") for s in INDEX_SYMBOLS] +
    [Output(f"ov-ltp-{_slug(s)}",   "style")    for s in INDEX_SYMBOLS] +
    [Output(f"ov-chg-{_slug(s)}",   "children") for s in INDEX_SYMBOLS] +
    [Output(f"ov-chg-{_slug(s)}",   "style")    for s in INDEX_SYMBOLS] +
    [Output(f"ov-ohlpc-{_slug(s)}", "children") for s in INDEX_SYMBOLS] +
    [Output("ov-chart", "figure")]
)

@app.callback(_OV_OUTPUTS, Input("fast-tick", "n_intervals"),
              State("sel-sym", "data"))
def update_overview(_, sel):
    if sel:
        return [no_update] * len(_OV_OUTPUTS)
    with _lock:
        latest  = {s: dict(t) for s, t in _latest.items()}
        history = {s: list(h) for s, h in _history.items()}
    ltps, ltp_s, chgs, chg_s, ohlpcs = [], [], [], [], []
    for sym in INDEX_SYMBOLS:
        t = latest.get(sym)
        if not t:
            ltps.append("—"); ltp_s.append({**MONO,"fontSize":"2rem","fontWeight":"bold","color":"#f1f5f9","lineHeight":"1"})
            chgs.append("—"); chg_s.append({**MONO,"fontSize":"0.88rem","marginTop":"5px","color":"#4a5568"})
            ohlpcs.append(""); continue
        ltp=t.get("ltp",0); ch=t.get("ch",0); chp=t.get("chp",0)
        o=t.get("open_price",0); h=t.get("high_price",0); l=t.get("low_price",0); pc=t.get("prev_close_price",0)
        up=ch>=0; clr="#22c55e" if up else "#ef4444"; s="+" if up else ""
        ltps.append(f"{ltp:,.2f}")
        ltp_s.append({**MONO,"fontSize":"2.2rem","fontWeight":"900","color":clr,
                       "lineHeight":"1","letterSpacing":"-0.03em"})
        chgs.append(f"{'▲' if up else '▼'}  {s}{ch:,.2f}   {s}{chp:.2f}%")
        chg_s.append({**MONO,"fontSize":"0.88rem","marginTop":"5px","color":clr,"fontWeight":"600"})
        _lbl = {"color":"#1e3a5f","fontSize":"0.62rem"}
        # Visual H/L range bar: show where current LTP sits between day's High and Low
        if h and l and h != l:
            _rng_pct = max(0, min((ltp - l) / (h - l) * 100, 100))
            _rng_bar = html.Div([
                html.Div(style={"width":f"{_rng_pct:.1f}%","height":"100%",
                                "background":clr,"borderRadius":"2px",
                                "transition":"width .5s ease"}),
            ], style={"height":"4px","borderRadius":"2px","marginBottom":"8px",
                      "background":"rgba(255,255,255,.05)","overflow":"hidden"})
        else:
            _rng_bar = html.Div()
        ohlpcs.append(html.Div([
            _rng_bar,
            html.Div([html.Span("O  ",style=_lbl), html.Span(f"{o:>10,.2f}",style={"color":"#64748b"})]),
            html.Div([html.Span("H  ",style=_lbl), html.Span(f"{h:>10,.2f}",style={"color":"#4ade80","fontWeight":"600"})]),
            html.Div([html.Span("L  ",style=_lbl), html.Span(f"{l:>10,.2f}",style={"color":"#f87171","fontWeight":"600"})]),
            html.Div([html.Span("PC ",style=_lbl), html.Span(f"{pc:>10,.2f}",style={"color":"#94a3b8"})]),
        ]))
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor=BG_CARD, plot_bgcolor=BG_CARD,
        font={"color":"#475569","family":"Courier New","size":11},
        margin={"l":48,"r":16,"t":12,"b":36},
        legend={"orientation":"h","x":0,"y":1.08,"bgcolor":"rgba(0,0,0,0)","font":{"size":10,"color":"#94a3b8"}},
        xaxis={"gridcolor":"#131e2e","linecolor":"#1e2d40","tickformat":"%H:%M","tickfont":{"color":"#334155"},"showgrid":True,"zeroline":False},
        yaxis={"gridcolor":"#131e2e","linecolor":"#1e2d40","tickfont":{"color":"#334155"},"showgrid":True,
               "zeroline":True,"zerolinecolor":"#1e3a5f","zerolinewidth":1.5,"ticksuffix":"%"},
        hovermode="x unified",
        hoverlabel={"bgcolor":"#0f1623","bordercolor":"#1e2d40","font":{"color":"#e2e8f0","family":"Courier New"}},
    )
    for sym in INDEX_SYMBOLS:
        pts=history.get(sym,[]); t=latest.get(sym,{}); pc=t.get("prev_close_price",0)
        if not pts or pc==0: continue
        fig.add_trace(go.Scatter(
            x=[p[0] for p in pts], y=[round((p[1]-pc)/pc*100,4) for p in pts],
            name=LABELS[sym], mode="lines",
            line={"color":COLORS[sym],"width":1.8},
            fill="tozeroy", fillcolor=FILLS[sym],
            hovertemplate=f"<b>{LABELS[sym]}</b>  %{{y:+.3f}}%<extra></extra>",
        ))
    return ltps + ltp_s + chgs + chg_s + ohlpcs + [fig]


# ── Callback 5: load expiry dropdown when index selected ──────────────────────
@app.callback(
    Output("expiry-dd", "options"),
    Output("expiry-dd", "value"),
    Input("sel-sym", "data"),
    prevent_initial_call=True,
)
def load_expiries(sym):
    if not sym:
        return [], None
    data = fetch_option_chain(sym)
    if data.get("s") != "ok":
        return [], None
    # expiryData: [{date: "30-06-2026", expiry: "1782813600", expiry_flag: "M"}, ...]
    exp_list = data.get("data", {}).get("expiryData", [])
    opts = [{"label": e["date"], "value": e["expiry"]} for e in exp_list]
    return opts, (opts[0]["value"] if opts else None)


# ── Callback 6: render option chain table ─────────────────────────────────────
@app.callback(
    Output("oc-table",      "children"),
    Output("oc-metrics",    "children"),
    Output("oc-prediction", "children"),
    Output("futures-strip", "children"),
    Input("oc-tick",        "n_intervals"),
    Input("expiry-dd",      "value"),
    State("sel-sym",        "data"),
    prevent_initial_call=True,
)
def update_oc(_, expiry, sym):
    if not sym:
        return "", "", "", ""

    with _lock:
        spot = (_latest.get(sym) or {}).get("ltp", 0)

    data = fetch_option_chain(sym, expiry or "")
    if data.get("s") != "ok":
        return html.Div(f"API error: {data.get('message','unknown')}",
                        style={"color": "#ef4444", "padding": "20px", **MONO}), "", "", ""

    d   = data.get("data", {})
    raw = d.get("optionsChain", [])
    if not raw:
        return html.Div("No option chain data returned", style={"color":"#4a5568","padding":"20px",**MONO}), "", "", ""

    # ── Parse flat array → dict[strike] = {CE: entry, PE: entry} ──────────────
    strike_map: dict[int, dict] = {}
    for entry in raw:
        sp = entry.get("strike_price", -1)
        if sp <= 0:
            continue
        if sp not in strike_map:
            strike_map[sp] = {}
        ot = entry.get("option_type", "")
        if ot in ("CE", "PE"):
            strike_map[sp][ot] = entry

    if not strike_map:
        return html.Div("No strike data", style={"color":"#4a5568","padding":"20px",**MONO}), "", "", ""

    strikes = sorted(strike_map.keys())

    # ATM = nearest strike to spot
    atm = min(strikes, key=lambda x: abs(x - spot)) if spot else strikes[len(strikes)//2]

    total_c_oi = d.get("callOi", 0)
    total_p_oi = d.get("putOi",  0)
    pcr        = total_p_oi / total_c_oi if total_c_oi else 0

    # Max pain using parsed data
    mp_chain = [{"strike_price": sp,
                 "call_options": {"oi": strike_map[sp].get("CE", {}).get("oi", 0)},
                 "put_options":  {"oi": strike_map[sp].get("PE", {}).get("oi", 0)}}
                for sp in strikes]
    mp = compute_max_pain(mp_chain)

    max_c_oi = max((strike_map[sp].get("CE", {}).get("oi", 0) or 0 for sp in strikes), default=1) or 1
    max_p_oi = max((strike_map[sp].get("PE", {}).get("oi", 0) or 0 for sp in strikes), default=1) or 1

    # ── Helpers ──────────────────────────────────────────────────────────────────
    def _f(v, dec=0):
        if v is None: return "—"
        return f"{v:,.{dec}f}"

    def _oi(v):
        if not v: return "—"
        if v >= 1_000_000: return f"{v/1_000_000:.2f}M"
        if v >= 1_000:     return f"{v/1_000:.1f}K"
        return str(v)

    def _oi_td(v, max_v, side):
        pct  = min(int((v or 0) / max_v * 100), 100) if max_v else 0
        grad = (f"linear-gradient(to left,  rgba(239,68,68,0.35) {pct}%, transparent {pct}%)"
                if side == "call" else
                f"linear-gradient(to right, rgba(74,222,128,0.35) {pct}%, transparent {pct}%)")
        return html.Td(_oi(v), style={
            **MONO, "textAlign": "right" if side=="call" else "left",
            "padding": "3px 8px", "fontSize": "0.68rem",
            "background": grad, "color": "#94a3b8",
        })

    def _oich_td(v):
        if not v:
            return html.Td("—", style={"color":"#1e2d40","fontSize":"0.65rem","padding":"3px 6px","textAlign":"center"})
        clr = "#22c55e" if v > 0 else "#ef4444"
        return html.Td(_oi(v), style={**MONO,"color":clr,"fontSize":"0.65rem","padding":"3px 6px","textAlign":"center"})

    # ── Table header ─────────────────────────────────────────────────────────────
    TH = {"padding":"6px 8px","fontSize":"0.58rem","letterSpacing":"0.08em",
          "fontWeight":"600","borderBottom":"2px solid #111d2e",
          "background":"#060c14","whiteSpace":"nowrap"}

    header = html.Tr([
        html.Th("OI CHG", style={**TH,"textAlign":"center","color":"#7f1d1d"}),
        html.Th("OI",     style={**TH,"textAlign":"right", "color":"#7f1d1d"}),
        html.Th("VOLUME", style={**TH,"textAlign":"right", "color":"#475569"}),
        html.Th("IV %",   style={**TH,"textAlign":"right", "color":"#475569"}),
        html.Th("DELTA",  style={**TH,"textAlign":"right", "color":"#475569"}),
        html.Th("LTP",    style={**TH,"textAlign":"right", "color":"#ef4444"}),
        html.Th("STRIKE", style={**TH,"textAlign":"center","color":"#94a3b8","minWidth":"90px"}),
        html.Th("LTP",    style={**TH,"textAlign":"left",  "color":"#4ade80"}),
        html.Th("DELTA",  style={**TH,"textAlign":"left",  "color":"#475569"}),
        html.Th("IV %",   style={**TH,"textAlign":"left",  "color":"#475569"}),
        html.Th("VOLUME", style={**TH,"textAlign":"left",  "color":"#475569"}),
        html.Th("OI",     style={**TH,"textAlign":"left",  "color":"#14532d"}),
        html.Th("OI CHG", style={**TH,"textAlign":"center","color":"#14532d"}),
    ])

    rows = [header]
    for sp in strikes:
        ce = strike_map[sp].get("CE", {})
        pe = strike_map[sp].get("PE", {})

        is_atm   = (sp == atm)
        itm_call = (sp < spot)

        if is_atm:
            row_bg = "rgba(251,191,36,0.10)"; sp_bg = "rgba(251,191,36,0.25)"; sp_clr = "#fbbf24"
        elif itm_call:
            row_bg = "rgba(239,68,68,0.04)";  sp_bg = "transparent"; sp_clr = "#64748b"
        else:
            row_bg = "transparent";            sp_bg = "transparent"; sp_clr = "#475569"

        TD = {"padding":"3px 8px","fontSize":"0.7rem","background":row_bg}

        # CE fields
        c_ltp   = ce.get("ltp");   c_oi  = ce.get("oi") or 0
        c_vol   = ce.get("volume");c_oic = ce.get("oich")
        c_iv    = (ce.get("greeks") or {}).get("iv")
        c_delta = (ce.get("greeks") or {}).get("delta")
        # PE fields
        p_ltp   = pe.get("ltp");   p_oi  = pe.get("oi") or 0
        p_vol   = pe.get("volume");p_oic = pe.get("oich")
        p_iv    = (pe.get("greeks") or {}).get("iv")
        p_delta = (pe.get("greeks") or {}).get("delta")

        rows.append(html.Tr([
            _oich_td(c_oic),
            _oi_td(c_oi, max_c_oi, "call"),
            html.Td(_oi(c_vol),   style={**TD,**MONO,"textAlign":"right","color":"#64748b"}),
            html.Td(_f(c_iv,1),   style={**TD,**MONO,"textAlign":"right","color":"#64748b"}),
            html.Td(_f(c_delta,2),style={**TD,**MONO,"textAlign":"right","color":"#64748b"}),
            html.Td(_f(c_ltp,2),  style={**TD,**MONO,"textAlign":"right","color":"#f87171","fontWeight":"600"}),
            # Strike
            html.Td(html.Div([
                f"{sp:,.0f}",
                html.Span(" ◆", style={"color":"#fbbf24","fontSize":"0.5rem"}) if is_atm else "",
            ]), style={**MONO,"textAlign":"center","padding":"3px 10px",
                        "fontWeight":"700" if is_atm else "400",
                        "color":sp_clr,"background":sp_bg,"fontSize":"0.72rem","whiteSpace":"nowrap"}),
            html.Td(_f(p_ltp,2),  style={**TD,**MONO,"color":"#86efac","fontWeight":"600"}),
            html.Td(_f(p_delta,2),style={**TD,**MONO,"color":"#64748b"}),
            html.Td(_f(p_iv,1),   style={**TD,**MONO,"color":"#64748b"}),
            html.Td(_oi(p_vol),   style={**TD,**MONO,"color":"#64748b"}),
            _oi_td(p_oi, max_p_oi, "put"),
            _oich_td(p_oic),
        ]))

    table = html.Table(rows, style={"width":"100%","borderCollapse":"collapse","fontSize":"0.72rem"})

    pcr_c   = "#22c55e" if pcr >= 1 else "#ef4444"
    metrics = html.Div([
        html.Span(f"Spot  {spot:,.2f}  ",  style={"color":"#94a3b8"}),
        html.Span(f"ATM  {atm:,.0f}  ",   style={"color":"#fbbf24"}),
        html.Span("PCR  ",                 style={"color":"#94a3b8"}),
        html.Span(f"{pcr:.2f}  ",          style={"color":pcr_c,"fontWeight":"700"}),
        html.Span(f"Max Pain  {mp:,.0f}",  style={"color":"#94a3b8"}),
        html.Span("    "),
        html.Span(f"Call OI  {_oi(total_c_oi)}  ", style={"color":"#f87171"}),
        html.Span(f"Put OI  {_oi(total_p_oi)}",     style={"color":"#86efac"}),
    ])

    pred       = compute_prediction(strike_map, spot, pcr, mp)
    prediction = render_prediction(pred, mp)

    futures     = fetch_futures(sym)
    fut_strip   = render_futures(futures, spot)

    return table, metrics, prediction, fut_strip


# ── Callback: trade recommendation + velocity panel ───────────────────────────
_OII_CLR = {"bull": "#4ade80", "bear": "#f87171", "neut": "#94a3b8"}
_OII_HEAD = {"fontSize": "0.6rem", "letterSpacing": "1px", "color": "#64748b"}
_OII_BOX  = {"padding": "8px 10px", "background": "rgba(255,255,255,0.02)",
             "borderRadius": "6px", "marginTop": "8px",
             "border": "1px solid rgba(255,255,255,0.05)"}


def _render_continuity(c) -> "html.Div":
    """Component B — overnight→morning continuity block."""
    lines = intraday_oi_intel.continuity_lines(c)
    if not c.has_data:
        return html.Div([
            html.Div("🌙→☀️ OVERNIGHT CONTINUITY", style={**_OII_HEAD, "marginBottom": "4px"}),
            html.Div(lines[0][1], style={"fontSize": "0.64rem", "color": "#64748b", **MONO}),
        ], style=_OII_BOX)
    hdr_clr = _OII_CLR["bull"] if c.score > 0 else _OII_CLR["bear"] if c.score < 0 else _OII_CLR["neut"]
    rows = [html.Div(text, style={"fontSize": "0.66rem", "color": _OII_CLR.get(bias, "#94a3b8"),
                                  "padding": "2px 0", **MONO})
            for bias, text in lines[1:]]
    return html.Div([
        html.Div([
            html.Span("🌙→☀️ OVERNIGHT CONTINUITY   ", style=_OII_HEAD),
            html.Span(c.verdict, style={"fontSize": "0.62rem", "fontWeight": "700", "color": hdr_clr}),
        ], style={"marginBottom": "4px"}),
        html.Div(rows),
    ], style=_OII_BOX)


def _render_oi_intel(d, cont=None) -> "html.Div":
    """Per-strike OI-Dynamics map (Component A) + overnight continuity (Component B)."""
    blocks = []
    if cont is not None:
        blocks.append(_render_continuity(cont))

    lines = intraday_oi_intel.summary_lines(d)
    if not d.has_data:
        blocks.append(html.Div([
            html.Div("⚡ LIVE OI DYNAMICS", style={**_OII_HEAD, "marginBottom": "4px"}),
            html.Div(lines[0][1], style={"fontSize": "0.65rem", "color": "#64748b", **MONO}),
        ], style=_OII_BOX))
        return html.Div(blocks)

    hdr_clr = _OII_CLR["bull"] if d.net_bias_score > 0 else _OII_CLR["bear"] if d.net_bias_score < 0 else _OII_CLR["neut"]
    chips = []
    if d.ceiling_strike is not None:
        chips.append(html.Span(f"⛔ Ceiling {d.ceiling_strike:,.0f}",
                               style={"color": "#f87171", "marginRight": "12px", "fontSize": "0.62rem"}))
    if d.floor_strike is not None:
        chips.append(html.Span(f"🛡 Floor {d.floor_strike:,.0f}",
                               style={"color": "#4ade80", "fontSize": "0.62rem"}))
    rows = [html.Div(text, style={"fontSize": "0.66rem", "color": _OII_CLR.get(bias, "#94a3b8"),
                                  "padding": "2px 0", **MONO})
            for bias, text in lines[1:]]
    blocks.append(html.Div([
        html.Div([
            html.Span("⚡ LIVE OI DYNAMICS   ", style=_OII_HEAD),
            html.Span(f"{d.verdict}  ({d.net_bias_score:+.1f})",
                      style={"fontSize": "0.62rem", "fontWeight": "700", "color": hdr_clr}),
        ], style={"marginBottom": "4px"}),
        html.Div(chips, style={"marginBottom": "5px"}) if chips else html.Div(),
        html.Div(rows),
    ], style=_OII_BOX))
    return html.Div(blocks)


def _render_track_record() -> "html.Div":
    """Component C — today's intraday paper-trade track record + honest stats."""
    led    = intraday_trades.get_ledger()
    today  = datetime.datetime.now(tz=IST).date().isoformat()
    s      = led.stats(today)
    recent = led.recent(12, today)
    if not recent:
        return html.Div([
            html.Div("📒 TRACK RECORD (today)", style={**_OII_HEAD, "marginBottom": "4px"}),
            html.Div("No signals fired yet today.", style={"fontSize": "0.64rem", "color": "#64748b", **MONO}),
        ], style=_OII_BOX)

    hit = s.get("hit_rate")
    avg_r = s.get("avg_r")
    hit_clr = "#4ade80" if (hit or 0) >= 50 else "#f87171"
    stat_line = []
    if s.get("n"):
        stat_line.append(html.Span(f"{s['wins']}W / {s['losses']}L  ", style={"color": "#94a3b8"}))
        if hit is not None:
            stat_line.append(html.Span(f"{hit:.0f}% hit  ", style={"color": hit_clr, "fontWeight": "700"}))
        if avg_r is not None:
            stat_line.append(html.Span(f"avg {avg_r:+.2f}R", style={"color": "#4ade80" if avg_r >= 0 else "#f87171"}))
        net = s.get("net_avg_r"); nhit = s.get("net_hit_rate"); bps = s.get("cost_bps")
        if net is not None:
            stat_line.append(html.Span(f"  ·  net {net:+.2f}R", style={
                "color": "#4ade80" if net >= 0 else "#f87171", "fontWeight": "700"}))
            stat_line.append(html.Span(
                (f" ({nhit:.0f}% @{bps:.0f}bps)" if nhit is not None else f" @{bps:.0f}bps"),
                style={"color": "#64748b"}))

    _ST = {"OPEN": "#fbbf24", "T1": "#4ade80", "T2": "#22c55e", "SL": "#f87171", "EOD": "#94a3b8"}
    rows = []
    for t in recent:
        oc = t.get("outcome") or "OPEN"
        st = t.get("status") or "OPEN"
        r  = t.get("r_multiple")
        rtxt = f"{r:+.2f}R" if r is not None else "live"
        clr = _ST.get(st, "#94a3b8")
        idx = (t.get("index_sym") or "").replace("NSE:", "").replace("-INDEX", "")
        tm  = (t.get("opened_ts") or "")[11:16]
        rows.append(html.Div([
            html.Span(f"{tm} ", style={"color": "#475569"}),
            html.Span(f"{idx} {t.get('direction')} {t.get('strike'):,.0f}  ", style={"color": "#cbd5e1"}),
            html.Span(f"{st} ", style={"color": clr, "fontWeight": "700"}),
            html.Span(rtxt, style={"color": "#4ade80" if (r or 0) > 0 else "#f87171" if (r or 0) < 0 else "#fbbf24"}),
        ], style={"fontSize": "0.64rem", "padding": "1px 0", **MONO}))

    return html.Div([
        html.Div([
            html.Span("📒 TRACK RECORD (today)   ", style=_OII_HEAD),
            html.Span(stat_line),
        ], style={"marginBottom": "5px"}),
        html.Div(rows),
    ], style=_OII_BOX)


@app.callback(
    Output("track-record", "children"),
    Input("setup-tick",    "n_intervals"),
    State("sel-sym",       "data"),
)
def update_track_record(_, sel):
    if not sel:
        return html.Div()
    return _render_track_record()


_IDX_ABBR = {"NSE:NIFTY50-INDEX": "NIFTY", "NSE:NIFTYBANK-INDEX": "BANK",
             "NSE:FINNIFTY-INDEX": "FIN", "NSE:MIDCPNIFTY-INDEX": "MIDCP"}
_TRADE_ST_CLR = {"OPEN": "#fbbf24", "T1": "#4ade80", "T2": "#22c55e", "SL": "#f87171",
                 "EOD": "#94a3b8", "FLIP": "#a78bfa"}


_REGIME_CLR = {"BULLISH": "#4ade80", "BEARISH": "#f87171", "NEUTRAL": "#94a3b8"}
_STAGE_CLR  = {"IMMINENT": "#ef4444", "BUILDING": "#f59e0b", "STABLE": "#22c55e"}


def _panel_help(what: str, read: list[str], caveat: str = "") -> "html.Details":
    """A click-to-open, scrollable 'what is this / how to read / why' explainer
    embedded in a panel heading. Keeps the surface clean; the full plain-English
    guide is one click away (and survives on mobile — it's a real expander, not a
    hover tooltip)."""
    body = [html.Div(what, style={"color": "#cbd5e1", "fontSize": "0.56rem",
                                  "lineHeight": "1.5", "marginBottom": "5px", "whiteSpace": "pre-line"})]
    body += [html.Div("• " + r, style={"color": "#94a3b8", "fontSize": "0.54rem",
                                       "lineHeight": "1.5", "whiteSpace": "normal"}) for r in read]
    if caveat:
        body.append(html.Div("⚠ " + caveat, style={"color": "#fbbf24", "fontSize": "0.54rem",
                    "lineHeight": "1.45", "marginTop": "5px", "whiteSpace": "normal"}))
    return html.Details([
        html.Summary("ℹ what is this · how to read", style={
            "color": "#67e8f9", "fontSize": "0.54rem", "cursor": "pointer", "marginLeft": "8px"}),
        html.Div(body, style={"maxHeight": "230px", "overflowY": "auto", "marginTop": "5px",
                 "padding": "7px 9px", "background": "#0a0f1a", "border": "1px solid #1e2d40",
                 "borderRadius": "4px", "maxWidth": "640px"}),
    ], style={"display": "inline-block", "verticalAlign": "middle"})


_PB_ACTION_CLR = {"BUY CE": "#4ade80", "BUY PE": "#f87171", "WRITE PE": "#fbbf24",
                  "WRITE CE": "#fbbf24", "BUY FUT": "#60a5fa", "SELL FUT": "#60a5fa",
                  "NO TRADE": "#64748b"}
_PB_TONE_CLR = {"bull": "#4ade80", "bear": "#f87171", "flat": "#475569"}
# Direction tag + the "WRITE PE is bullish" stance clarifier now come from
# signal_types (one canonical source — see sig.arrow/color/action_hint).


# ── Session Conductor panel — the unified, evolving stance per index ────────────
# (LONG/SHORT/FLAT arrow + colour now from signal_types — one canonical source)


# ── Intraday-TF footprint matrix (left pane) ────────────────────────────────────
_ITF_TAG_CLR = {"LONG BUILDUP": "#22c55e", "SHORT COVER": "#86efac",
                "SHORT BUILDUP": "#ef4444", "LONG UNWIND": "#fca5a5", "BALANCED": "#64748b"}
# OI-bias colour (BULLISH/BEARISH/NEUTRAL) now from signal_types.color()


def _parse_asof(asof_value):
    """Regime-Radar dropdown value → as_of datetime (None = live 'now'). One parser
    so every Trade-Book panel reconstructs the SAME instant — no mixing a past
    review mark with live (future-relative) conductor/footprint/forecast data."""
    if asof_value and asof_value != "now":
        try:
            dt = datetime.datetime.fromisoformat(asof_value)
            # read_mirror compares against tz-aware IST ts; a naive as_of would raise
            # "Cannot compare tz-naive and tz-aware". Localise naive inputs to IST.
            return dt.replace(tzinfo=IST) if dt.tzinfo is None else dt
        except Exception:
            return None
    return None

# ── Hover-tooltip copy for the footprint (native browser `title` on each element) ──
_TF_TAG_DESC = {
    "LONG BUILDUP":  "price UP + option OI BUILDING -> fresh longs adding (sustainable up-move)",
    "SHORT COVER":   "price UP + OI FALLING -> shorts buying back to exit (up, but not fresh demand - can fade)",
    "SHORT BUILDUP": "price DOWN + OI BUILDING -> fresh shorts adding (sustainable down-move)",
    "LONG UNWIND":   "price DOWN + OI FALLING -> longs exiting (down from position-closing, not fresh selling)",
    "BALANCED":      "no confirmed regime - the price move and/or the OI build is below the noise threshold",
}
_GRADE_DESC = {
    "aligned":   "ALIGNED (star) - a higher timeframe AGREES and an independent flow (OI positioning or "
                 "futures) CONFIRMS it, and the move persisted. The only grade meant to be traded.",
    "confirmed": "confirmed (check) - persisted with ONE corroboration (higher-TF or flow), not both. "
                 "A solid single-frame read.",
    "tentative": "tentative (~) - a significant move that is not yet corroborated or did not persist. "
                 "Watch only; do not act.",
}
_STACK_TIP = ("STACK = multi-timeframe consensus direction (longer timeframes weighted heavier). "
              "'N/M TFs agree' = how many directional frames point this way. "
              "'star K tradeable' = frames that are fully stack-confirmed (safe to act on). "
              "'no confirmed entry' = nothing is corroborated yet -> stand aside.")
_BIAS_TIP = ("OI BIAS = net option POSITIONING across frames (put-writing vs call-writing). "
             "BULLISH = put-writers dominant (a floor is being built); BEARISH = call-writers dominant "
             "(a ceiling). This is positioning, separate from the price-regime tag on each line.")


def _tf_tooltip(c: dict) -> str:
    """Per-frame hover text: what the regime tag means + WHY it is showing now,
    derived from the live z-score / OI build / confirmation grade."""
    z = c.get("z", 0.0); tag = c["tag"]; grade = c.get("grade", "balanced")
    out = [f"{c['tf']}-MINUTE FRAME", "",
           f"{tag} - {_TF_TAG_DESC.get(tag, '')}", "", "WHY THIS LABEL NOW:"]
    zsig = ">=1 sigma -> significant" if abs(z) >= 1 else "<1 sigma -> within noise (NOT a real move)"
    out.append(f"- price moved {z:+.1f} sigma  ({zsig})")
    out.append("- OI build is significant" if c.get("oi_build")
               else "- OI build is below the significance bar")
    if tag == "BALANCED":
        out.append("- => BALANCED until BOTH a >=1 sigma move AND a material OI build line up")
    else:
        out.append("- " + _GRADE_DESC.get(grade, ""))
        if c.get("confirms"):
            out.append("- confirmed by: " + " - ".join(c["confirms"]))
    return "\n".join(out)


def _crosshair(fig: "go.Figure") -> "go.Figure":
    """TradingView-style hover crosshair for the stacked-panel charts. A dotted
    VERTICAL guide spikes across EVERY panel (spikemode='across' on the shared
    x-axis) so a price bar lines up with its OI / premium / flow bars at the same
    instant; a dotted HORIZONTAL guide runs to the y-axis. Free-following
    (spikesnap='cursor'), always drawn (spikedistance=-1).

    The value TAGS at the cursor (time on x, price on y) are drawn by
    assets/crosshair_price.js — plotly has no native cursor-value axis label. That
    JS reads the x-axis range back in UTC (the tz-correct wall-clock, since the
    chart's naive IST timestamps are stored as UTC epoch-ms); do NOT switch it to
    local time or the time tag drifts +05:30."""
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor",
                     spikedash="dot", spikecolor="#7dd3fc", spikethickness=1)
    fig.update_yaxes(showspikes=True, spikemode="across", spikesnap="cursor",
                     spikedash="dot", spikecolor="#7dd3fc", spikethickness=1)
    # Hand-drawn support/resistance lines (modebar 'drawline'): amber dashed, the
    # classic S/R look. dragmode stays 'zoom' so +/- and box-zoom work by default;
    # the user clicks the draw tool to switch. Shapes are editable/movable (config
    # edits.shapePosition) so a level can be nudged after drawing.
    fig.update_layout(hovermode="closest", spikedistance=-1, hoverdistance=100,
                      newshape=dict(line=dict(color="#fbbf24", width=1.5, dash="dash"),
                                    opacity=0.9))
    return fig


def _footprint_fig(sym, tf_min: int, asof_value=None, date=None, expiry="weekly") -> "go.Figure":
    """4-panel popup chart for one index/timeframe — full session, tf-minute bars:
      1. Price  — candlestick + close line, with volume merged at the base.
      2. Option OI — CE (ceiling) vs PE (floor): writing build-up vs unwinding.
      3. ATM-straddle premium + ATM IV (decay vs vol-expansion).
      4. Positioning flow — per-bar ΔOI coloured BUY vs WRITE (per-leg premium):
         the "what are they doing" read. Calls plotted up, puts down.
    The clicked lookback window is shaded across every panel."""
    d = footprint_chart.build_series(sym, int(tf_min), date=date,
                                     as_of=_parse_asof(asof_value), expiry=expiry)
    if not d.get("has_data"):
        fig = go.Figure()
        fig.add_annotation(text=d.get("note", "no data"), showarrow=False,
                           font=dict(color="#64748b", size=13))
        fig.update_layout(template="plotly_dark", height=300,
                          paper_bgcolor=BG_CARD, plot_bgcolor=BG_CARD,
                          xaxis=dict(visible=False), yaxis=dict(visible=False))
        return fig
    ts = d["ts"]
    label = LABELS.get(sym, sym)
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.045,
        row_heights=[0.34, 0.22, 0.20, 0.24],
        specs=[[{"secondary_y": True}], [{}], [{"secondary_y": True}], [{}]],
        subplot_titles=(
            f"{label} price — {tf_min}m candles + volume", "Option OI — CE vs PE (lakh)",
            "ATM straddle premium (₹) + ATM IV (%)",
            "Positioning flow — ΔOI/bar · calls↑ puts↓ · red=call-write amber=call-buy "
            "green=put-write lime=put-buy · hatched grey=closing"))
    # 1 — price: candlestick + close line, with volume merged at the base (secondary axis).
    fig.add_trace(go.Candlestick(
        x=ts, open=d["open"], high=d["high"], low=d["low"], close=d["close"],
        name="price", increasing_line_color="#22c55e", decreasing_line_color="#ef4444",
        increasing_fillcolor="#22c55e", decreasing_fillcolor="#ef4444",
        line=dict(width=1), showlegend=False), row=1, col=1, secondary_y=False)
    # close line + volume are CONTEXT only — skip their hover so they don't hijack
    # the readout with a single close value (the candle below the cursor). The
    # candlestick OHLC stays the price tooltip, so it matches the crosshair height.
    fig.add_trace(go.Scatter(x=ts, y=d["close"], mode="lines", name="close",
                             line=dict(color="#67e8f9", width=1, dash="dot"),
                             connectgaps=True, opacity=0.7, hoverinfo="skip"),
                  row=1, col=1, secondary_y=False)
    fig.add_trace(go.Bar(x=ts, y=d["volume"], name="volume",
                         marker_color="rgba(34,211,238,0.45)", showlegend=False,
                         hoverinfo="skip"),
                  row=1, col=1, secondary_y=True)
    _maxv = max([v for v in d["volume"] if v] or [1])
    fig.update_yaxes(range=[0, _maxv * 4.5], row=1, col=1, secondary_y=True,
                     showticklabels=False, showgrid=False)
    fig.update_yaxes(tickformat=",.0f", row=1, col=1, secondary_y=False)   # 24,100 not 24k
    # OI walls as auto-drawn S/R lines on the price panel: resistance = max-call-OI
    # strike, support = max-put-OI strike, + max pain. STRUCTURE, not a signal — the
    # break did NOT predict continuation on captured days (backtest_sr_break.py); read
    # them as where dealer size sits, draw your own lines with the modebar to mark levels.
    try:
        _oiw = read_mirror("oi_snapshots", date, _parse_asof(asof_value), sym)
        if _oiw is not None and len(_oiw) and "call_wall" in _oiw.columns:
            _lw = _oiw.sort_values("ts").iloc[-1]
            for _lvl, _clr, _lab in (
                    (_lw.get("call_wall"), "#f87171", "R · call wall"),
                    (_lw.get("put_wall"),  "#4ade80", "S · put wall"),
                    (_lw.get("max_pain"),  "#a78bfa", "max pain")):
                try:
                    _lvl = float(_lvl)
                except (TypeError, ValueError):
                    continue
                if _lvl > 0:
                    fig.add_hline(y=_lvl, row=1, col=1, secondary_y=False,
                                  line=dict(color=_clr, width=1, dash="dot"),
                                  annotation_text=_lab, annotation_position="top left",
                                  annotation_font=dict(size=8, color=_clr))
    except Exception:
        pass
    # 2 — OI CE vs PE.
    fig.add_trace(go.Scatter(x=ts, y=d["oi_ce"], mode="lines", name="CE OI (ceiling)",
                             line=dict(color="#ef4444", width=1.6)), row=2, col=1)
    fig.add_trace(go.Scatter(x=ts, y=d["oi_pe"], mode="lines", name="PE OI (floor)",
                             line=dict(color="#22c55e", width=1.6)), row=2, col=1)
    # 3 — premium (left axis) + ATM IV (right axis): decay vs vol-expansion.
    fig.add_trace(go.Scatter(x=ts, y=d["premium"], mode="lines", name="ATM straddle",
                             line=dict(color="#fbbf24", width=2),
                             connectgaps=True), row=3, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=ts, y=d.get("iv_atm"), mode="lines", name="ATM IV %",
                             line=dict(color="#a78bfa", width=1.3, dash="dot"),
                             connectgaps=True), row=3, col=1, secondary_y=True)
    # 4 — positioning flow: per-bar ΔOI, colour = buy vs write (from per-leg premium).
    # Closing (cover/unwind) = one muted grey + a hatch pattern, so it is unmistakable
    # from the solid build bars (no amber-vs-grey confusion).
    _CLOSE = "#64748b"
    _CE = {"write": "#ef4444", "buy": "#f59e0b", "cover": _CLOSE, "unwind": _CLOSE, "flat": "rgba(0,0,0,0)"}
    _PE = {"write": "#22c55e", "buy": "#84cc16", "cover": _CLOSE, "unwind": _CLOSE, "flat": "rgba(0,0,0,0)"}
    _is_close = {"cover", "unwind"}
    ce_y = [abs(v) if v is not None else 0 for v in d["d_oi_ce"]]      # calls plotted up
    pe_y = [-abs(v) if v is not None else 0 for v in d["d_oi_pe"]]     # puts plotted down
    ce_c = [_CE.get(a, _CLOSE) for a in d["ce_act"]]
    pe_c = [_PE.get(a, _CLOSE) for a in d["pe_act"]]
    ce_pat = ["/" if a in _is_close else "" for a in d["ce_act"]]       # hatch = closing
    pe_pat = ["/" if a in _is_close else "" for a in d["pe_act"]]
    ce_t = [f"CALL {a} · ΔOI {v}L" for a, v in zip(d["ce_act"], d["d_oi_ce"])]
    pe_t = [f"PUT {a} · ΔOI {v}L" for a, v in zip(d["pe_act"], d["d_oi_pe"])]
    fig.add_trace(go.Bar(x=ts, y=ce_y, name="calls", showlegend=False,
                         marker=dict(color=ce_c, pattern=dict(shape=ce_pat, solidity=0.45,
                                     fgcolor="#0a0f1a")),
                         hovertext=ce_t, hoverinfo="x+text"), row=4, col=1)
    fig.add_trace(go.Bar(x=ts, y=pe_y, name="puts", showlegend=False,
                         marker=dict(color=pe_c, pattern=dict(shape=pe_pat, solidity=0.45,
                                     fgcolor="#0a0f1a")),
                         hovertext=pe_t, hoverinfo="x+text"), row=4, col=1)
    fig.add_hline(y=0, line_width=0.8, line_color="#475569", row=4, col=1)
    # Shade the clicked lookback window across every panel.
    if d.get("last_ts"):
        x0 = d["last_ts"] - datetime.timedelta(minutes=int(tf_min))
        for rr in (1, 2, 3, 4):
            fig.add_vrect(x0=x0, x1=d["last_ts"], fillcolor="#67e8f9",
                          opacity=0.08, line_width=0, row=rr, col=1)
    fig.update_yaxes(title_text="IV %", row=3, col=1, secondary_y=True,
                     showgrid=False, color="#a78bfa")
    fig.update_layout(template="plotly_dark", height=900,
                      margin=dict(l=58, r=46, t=46, b=28),
                      paper_bgcolor=BG_CARD, plot_bgcolor=BG_CARD,
                      barmode="overlay", bargap=0.15, showlegend=True,
                      legend=dict(orientation="h", y=1.05, x=0, font=dict(size=9)),
                      font=dict(size=10))
    fig.update_xaxes(rangeslider_visible=False)   # candlestick adds one by default
    for a in fig["layout"]["annotations"]:        # subplot titles
        a["font"] = dict(size=10.5, color="#94a3b8")
    # keep user zoom + hand-drawn S/R lines across figure refreshes; reset only
    # when the index/timeframe changes (old levels don't apply to a new chart)
    fig.update_layout(uirevision=f"{sym}-{tf_min}")
    return _crosshair(fig)


def _strike_fig(sym, tf_min: int, strike: int, asof_value=None, date=None, expiry="weekly") -> "go.Figure":
    """4-panel single-strike option chart — full session, tf-minute bars:
      1. Index price candles + volume, with a dashed line at the strike.
      2. This strike's CE OI vs PE OI (writing/unwinding AT this strike).
      3. This strike's CE vs PE premium (LTP).
      4. This strike's positioning — per-bar ΔOI, CE up / PE down, buy vs write."""
    d = footprint_chart.build_strike_series(sym, int(tf_min), int(strike),
                                            date=date, as_of=_parse_asof(asof_value), expiry=expiry)
    if not d.get("has_data"):
        fig = go.Figure()
        fig.add_annotation(text=d.get("note", "no data"), showarrow=False,
                           font=dict(color="#64748b", size=13))
        fig.update_layout(template="plotly_dark", height=300, paper_bgcolor=BG_CARD,
                          plot_bgcolor=BG_CARD, xaxis=dict(visible=False), yaxis=dict(visible=False))
        return fig
    ts = d["ts"]
    label = LABELS.get(sym, sym)
    k = d["strike"]
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.045,
        row_heights=[0.32, 0.22, 0.22, 0.24],
        specs=[[{"secondary_y": True}], [{}], [{}], [{}]],
        subplot_titles=(f"{label} price — {tf_min}m + volume  ·  dashed line = {k} strike",
                        f"OI at {k} — CE (ceiling) vs PE (floor) (lakh)",
                        f"Premium at {k} — CE vs PE (₹)",
                        f"Positioning at {k} — ΔOI/bar (delta-adjusted: true buy vs write) · CE↑ PE↓ "
                        "· red=call-write amber=call-buy green=put-write lime=put-buy · hatched=closing"))
    # 1 — index price + volume + strike line.
    fig.add_trace(go.Candlestick(
        x=ts, open=d["open"], high=d["high"], low=d["low"], close=d["close"], name="price",
        increasing_line_color="#22c55e", decreasing_line_color="#ef4444",
        increasing_fillcolor="#22c55e", decreasing_fillcolor="#ef4444",
        line=dict(width=1), showlegend=False), row=1, col=1, secondary_y=False)
    _vol = [(a or 0) + (b or 0) for a, b in zip(d["ce_vol"], d["pe_vol"])]
    fig.add_trace(go.Bar(x=ts, y=_vol, name="volume", marker_color="rgba(34,211,238,0.4)",
                         showlegend=False, hoverinfo="skip"), row=1, col=1, secondary_y=True)
    _mv = max([v for v in _vol if v] or [1])
    fig.update_yaxes(range=[0, _mv * 4.5], row=1, col=1, secondary_y=True,
                     showticklabels=False, showgrid=False)
    fig.update_yaxes(tickformat=",.0f", row=1, col=1, secondary_y=False)
    fig.add_hline(y=k, line_width=1, line_dash="dash", line_color="#a78bfa", row=1, col=1)
    # 2 — OI at strike.
    fig.add_trace(go.Scatter(x=ts, y=d["ce_oi"], mode="lines", name="CE OI",
                             line=dict(color="#ef4444", width=1.6)), row=2, col=1)
    fig.add_trace(go.Scatter(x=ts, y=d["pe_oi"], mode="lines", name="PE OI",
                             line=dict(color="#22c55e", width=1.6)), row=2, col=1)
    # 3 — premium at strike.
    fig.add_trace(go.Scatter(x=ts, y=d["ce_prem"], mode="lines", name="CE ₹",
                             line=dict(color="#ef4444", width=1.4)), row=3, col=1)
    fig.add_trace(go.Scatter(x=ts, y=d["pe_prem"], mode="lines", name="PE ₹",
                             line=dict(color="#22c55e", width=1.4)), row=3, col=1)
    # 4 — positioning at strike.
    _CLOSE = "#64748b"
    _CE = {"write": "#ef4444", "buy": "#f59e0b", "cover": _CLOSE, "unwind": _CLOSE, "flat": "rgba(0,0,0,0)"}
    _PE = {"write": "#22c55e", "buy": "#84cc16", "cover": _CLOSE, "unwind": _CLOSE, "flat": "rgba(0,0,0,0)"}
    _isc = {"cover", "unwind"}
    ce_y = [abs(v) if v is not None else 0 for v in d["ce_doi"]]
    pe_y = [-abs(v) if v is not None else 0 for v in d["pe_doi"]]
    fig.add_trace(go.Bar(x=ts, y=ce_y, name="CE", showlegend=False,
                         marker=dict(color=[_CE.get(a, _CLOSE) for a in d["ce_act"]],
                                     pattern=dict(shape=["/" if a in _isc else "" for a in d["ce_act"]],
                                                  solidity=0.45, fgcolor="#0a0f1a")),
                         hovertext=[f"CE {a}" for a in d["ce_act"]], hoverinfo="x+text"), row=4, col=1)
    fig.add_trace(go.Bar(x=ts, y=pe_y, name="PE", showlegend=False,
                         marker=dict(color=[_PE.get(a, _CLOSE) for a in d["pe_act"]],
                                     pattern=dict(shape=["/" if a in _isc else "" for a in d["pe_act"]],
                                                  solidity=0.45, fgcolor="#0a0f1a")),
                         hovertext=[f"PE {a}" for a in d["pe_act"]], hoverinfo="x+text"), row=4, col=1)
    fig.add_hline(y=0, line_width=0.8, line_color="#475569", row=4, col=1)
    if d.get("last_ts"):
        x0 = d["last_ts"] - datetime.timedelta(minutes=int(tf_min))
        for rr in (1, 2, 3, 4):
            fig.add_vrect(x0=x0, x1=d["last_ts"], fillcolor="#67e8f9",
                          opacity=0.08, line_width=0, row=rr, col=1)
    fig.update_layout(template="plotly_dark", height=900, margin=dict(l=58, r=18, t=46, b=28),
                      paper_bgcolor=BG_CARD, plot_bgcolor=BG_CARD, barmode="overlay", bargap=0.15,
                      showlegend=True, legend=dict(orientation="h", y=1.05, x=0, font=dict(size=9)),
                      font=dict(size=10))
    fig.update_xaxes(rangeslider_visible=False)
    for a in fig["layout"]["annotations"]:
        a["font"] = dict(size=10.5, color="#94a3b8")
    # keep user zoom + hand-drawn S/R lines across figure refreshes; reset only
    # when the index/timeframe changes (old levels don't apply to a new chart)
    fig.update_layout(uirevision=f"{sym}-{tf_min}")
    return _crosshair(fig)


def _futures_fig(sym, tf_min: int, asof_value=None, date=None, leg="near") -> "go.Figure":
    """5-panel futures chart for the chosen expiry leg (near/next/far) — full session:
      1. Selected-leg candles + volume, with near/next/far context lines.
      2. Futures OI (lakh contracts, consolidated — NSE oi-spurts).
      3. Futures positioning — per-bar ΔOI × price (long/short buildup vs cover/unwind).
      4. Basis (futures − index): rising = long demand (bullish); falling/discount = bearish.
      5. Rollover — roll spread (next − near ₹) + next-month volume share (%)."""
    d = footprint_chart.build_futures_series(sym, int(tf_min), date=date,
                                             as_of=_parse_asof(asof_value), leg=leg)
    if not d.get("has_data"):
        fig = go.Figure()
        fig.add_annotation(text=d.get("note", "no data"), showarrow=False,
                           font=dict(color="#64748b", size=13))
        fig.update_layout(template="plotly_dark", height=300, paper_bgcolor=BG_CARD,
                          plot_bgcolor=BG_CARD, xaxis=dict(visible=False), yaxis=dict(visible=False))
        return fig
    ts = d["ts"]
    label = LABELS.get(sym, sym)
    lg = d.get("leg", "near").upper()
    oi_note = "" if d.get("has_oi") else "  (no OI this day)"
    vol_note = "" if d.get("has_vol") else "  (no volume for this leg)"
    fig = make_subplots(
        rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        row_heights=[0.32, 0.15, 0.20, 0.15, 0.18],
        specs=[[{"secondary_y": True}], [{}], [{}], [{}], [{"secondary_y": True}]],
        subplot_titles=(f"{label} FUT — {lg} expiry — {tf_min}m candles + volume{vol_note}",
                        f"Futures OI (lakh contracts, all expiries){oi_note}",
                        "Futures positioning — ΔOI/bar · green=long-buildup red=short-buildup "
                        "· teal=covering amber=unwinding (down=closing)",
                        "Basis = futures − index (₹)  ·  rising=long demand (bullish) · "
                        "falling / red discount=demand fading (bearish)",
                        "Rollover — roll spread next−near (₹, amber) + next-month volume share (%, teal)"))
    # 1 — selected-leg candles + volume, with the OTHER legs as faint context lines.
    fig.add_trace(go.Candlestick(
        x=ts, open=d["open"], high=d["high"], low=d["low"], close=d["close"], name=f"{lg} FUT",
        increasing_line_color="#22c55e", decreasing_line_color="#ef4444",
        increasing_fillcolor="#22c55e", decreasing_fillcolor="#ef4444",
        line=dict(width=1), showlegend=False), row=1, col=1, secondary_y=False)
    _legclr = {"near": "#67e8f9", "next": "#a78bfa", "far": "#fb923c"}
    for _lg in ("near", "next", "far"):
        if _lg == d.get("leg") or not any(v is not None for v in d.get(_lg, [])):
            continue
        fig.add_trace(go.Scatter(x=ts, y=d[_lg], mode="lines", name=f"{_lg} mth",
                                 line=dict(color=_legclr[_lg], width=1, dash="dot"), opacity=0.55,
                                 hoverinfo="skip"), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Bar(x=ts, y=d["volume"], name="volume",
                         marker_color="rgba(34,211,238,0.45)", showlegend=False,
                         hoverinfo="skip"), row=1, col=1, secondary_y=True)
    _maxv = max([v for v in d["volume"] if v] or [1])
    fig.update_yaxes(range=[0, _maxv * 4.5], row=1, col=1, secondary_y=True,
                     showticklabels=False, showgrid=False)
    fig.update_yaxes(tickformat=",.0f", row=1, col=1, secondary_y=False)   # 24,100 not 24k
    # 2 — futures OI level.
    fig.add_trace(go.Scatter(x=ts, y=d.get("oi"), mode="lines", name="futures OI",
                             line=dict(color="#38bdf8", width=1.8), connectgaps=True,
                             showlegend=False), row=2, col=1)
    # 3 — positioning flow.
    _FC = {"long": "#22c55e", "short": "#ef4444", "cover": "#5eead4", "unwind": "#f59e0b",
           "flat": "rgba(0,0,0,0)"}
    fa = d.get("fut_act", [])
    fy = [v if v is not None else 0 for v in d.get("d_oi", [])]
    fc_clr = [_FC.get(a, "rgba(0,0,0,0)") for a in fa]
    ft = [f"{a} · ΔOI {v}L" for a, v in zip(fa, d.get("d_oi", []))]
    fig.add_trace(go.Bar(x=ts, y=fy, marker_color=fc_clr, name="ΔOI", showlegend=False,
                         hovertext=ft, hoverinfo="x+text"), row=3, col=1)
    fig.add_hline(y=0, line_width=0.8, line_color="#475569", row=3, col=1)
    # 4 — basis.
    b_clr = ["#22c55e" if (v is not None and v >= 0) else "#ef4444" for v in d["basis"]]
    fig.add_trace(go.Bar(x=ts, y=d["basis"], name="basis", marker_color=b_clr,
                         showlegend=False), row=4, col=1)
    fig.add_hline(y=0, line_width=0.8, line_color="#475569", row=4, col=1)
    # 5 — rollover: roll spread (₹) + next-month volume share (%).
    fig.add_trace(go.Scatter(x=ts, y=d["roll"], mode="lines", name="roll spread ₹",
                             line=dict(color="#fbbf24", width=1.6), showlegend=False),
                  row=5, col=1, secondary_y=False)
    if d.get("has_roll"):
        fig.add_trace(go.Scatter(x=ts, y=d["roll_share"], mode="lines", name="next vol %",
                                 line=dict(color="#5eead4", width=1.2, dash="dot"), showlegend=False),
                      row=5, col=1, secondary_y=True)
        fig.update_yaxes(range=[0, 100], row=5, col=1, secondary_y=True, showgrid=False, color="#5eead4")
    if d.get("last_ts"):
        x0 = d["last_ts"] - datetime.timedelta(minutes=int(tf_min))
        for rr in (1, 2, 3, 4, 5):
            fig.add_vrect(x0=x0, x1=d["last_ts"], fillcolor="#67e8f9",
                          opacity=0.08, line_width=0, row=rr, col=1)
    fig.update_layout(template="plotly_dark", height=980,
                      margin=dict(l=58, r=46, t=46, b=28),
                      paper_bgcolor=BG_CARD, plot_bgcolor=BG_CARD, bargap=0.15,
                      showlegend=True, legend=dict(orientation="h", y=1.04, x=0, font=dict(size=9)),
                      font=dict(size=10))
    fig.update_xaxes(rangeslider_visible=False)
    for a in fig["layout"]["annotations"]:
        a["font"] = dict(size=10.5, color="#94a3b8")
    # keep user zoom + hand-drawn S/R lines across figure refreshes; reset only
    # when the index/timeframe changes (old levels don't apply to a new chart)
    fig.update_layout(uirevision=f"{sym}-{tf_min}")
    return _crosshair(fig)


def _render_intraday_tf(sym, asof_value=None, snap=None, clickable=False,
                        date=None) -> "html.Div":
    """Per-timeframe OI·Price·Volume matrix + divergence flags for one index.
    Shows whether each 5/10/15/60-min frame is fresh buildup or positions CLOSING,
    so a rally on closing (distribution) or hidden call-writing is visible early.
    asof_value (Trade-Book replay) reconstructs the read at a past instant; `snap`
    is the shared MarketSnapshot so the footprint isn't recomputed per panel.
    `date` = replay a PAST day's mirror (ghost practice) instead of today's."""
    if not _ITF_AVAILABLE:
        return html.Div()
    try:
        r = (snap.footprint(sym) if snap is not None else
             intraday_tf.analyze(sym, date=date, as_of=_parse_asof(asof_value)))
    except Exception:
        return html.Div("—", style={"color": "#475569", "fontSize": "0.55rem"})
    if not r.get("has_data"):
        return html.Div(r.get("note", "warming up"),
                        style={"color": "#475569", "fontSize": "0.55rem", **MONO})

    rows = []
    for c in r["cells"]:
        pu = c.get("pu", 0); z = c.get("z", 0.0)
        # colour by SIGNIFICANCE (>=1σ), not raw sign — a sub-σ wiggle stays grey
        pcl = "#22c55e" if pu > 0 else "#ef4444" if pu < 0 else "#94a3b8"
        parr = "▲" if c["px"] > 0 else "▼" if c["px"] < 0 else "·"
        bld = c["oi_build"]
        boi = ("↑ build" if bld > 0 else "↓ close" if bld < 0 else "· flat")
        bclr = "#4ade80" if bld > 0 else "#f87171" if bld < 0 else "#64748b"
        tclr = _ITF_TAG_CLR.get(c["tag"], "#64748b")
        is_dir = c["tag"] != "BALANCED"
        grade = c.get("grade", "balanced")
        # grade badge: ★ aligned (stack-confirmed, tradeable) · ✓ confirmed (solid 1-TF)
        # · ~ tentative (significant but uncorroborated — watch only)
        badge = {"aligned": " ★", "confirmed": " ✓"}.get(grade, "")
        pre   = "~" if (is_dir and grade == "tentative") else ""
        tag_txt = f"{pre}{c['tag']}{badge}" if is_dir else c["tag"]
        tag_style = {"color": tclr, "fontSize": "0.55rem", "fontWeight": "700",
                     "marginLeft": "auto"}
        if grade == "tentative":
            tag_style["opacity"] = 0.4
        elif grade == "confirmed":
            tag_style["opacity"] = 0.8
        ovol = "" if c["ovol"] is None else f" vol {c['ovol']:.0f}L"
        farr = ("fut▲" if (c["fpx"] or 0) > 0.03 else "fut▼" if (c["fpx"] or 0) < -0.03 else "fut·")
        foi = c.get("foi"); fb = c.get("foi_build", 0)
        foi_txt = "" if foi is None else f" OI{'↑' if fb>0 else '↓' if fb<0 else '·'}{foi:+.0f}L"
        fclr = "#4ade80" if fb > 0 else "#f87171" if fb < 0 else "#475569"
        # confirm trail — which higher TFs / flows back an aligned or confirmed call
        trail = []
        if grade in ("aligned", "confirmed") and c.get("confirms"):
            tclr2 = "#4ade80" if grade == "aligned" else "#64748b"
            trail = [html.Div(("★ stack: " if grade == "aligned" else "✓ ")
                              + " · ".join(c["confirms"]),
                              style={"color": tclr2, "fontSize": "0.46rem", "opacity": 0.85})]
        row_kw: dict = {}
        if clickable:
            row_kw["id"] = {"type": "fp-tf", "sym": sym, "tf": c["tf"]}
            row_kw["n_clicks"] = 0
        rows.append(html.Div([
            html.Div([
                html.Span(f"{c['tf']}m", style={"color": "#cbd5e1", "fontWeight": "700",
                          "fontSize": "0.58rem", "minWidth": "26px", "display": "inline-block"}),
                html.Span(f"{parr}{c['px']:+.2f}%", style={"color": pcl, "fontSize": "0.58rem"}),
                html.Span(f" {z:+.1f}σ", style={"color": "#475569", "fontSize": "0.5rem"}),
                html.Span("  📈" if clickable else "", style={"fontSize": "0.5rem", "opacity": 0.6}),
                html.Span(f"  {tag_txt}", style=tag_style),
            ], style={"display": "flex", "alignItems": "center"}),
            html.Div([
                html.Span(f"optOI {boi} {c['d_tot']:+.0f}L", style={"color": bclr, "fontSize": "0.52rem"}),
                html.Span([html.Span(f"{farr}", style={"color": "#64748b"}),
                           html.Span(foi_txt, style={"color": fclr}),
                           html.Span(ovol, style={"color": "#475569"})],
                          style={"fontSize": "0.5rem"}),
            ], style={"display": "flex", "justifyContent": "space-between"}),
            *trail,
        ], title=(_tf_tooltip(c) + ("  ·  click → OI/volume/premium chart" if clickable else "")),
           style={"padding": "4px 0", "borderBottom": "1px solid #111d2e",
                  "cursor": "pointer" if clickable else "help"}, **row_kw))

    # Greyed placeholders for timeframes that can't be computed yet (their lookback
    # predates capture-start). Shows the slot is warming up + when it unlocks, rather
    # than vanishing silently — e.g. the 1h frame before ~60 min of ticks exist.
    for pc in r.get("pending", []):
        rows.append(html.Div([
            html.Span(f"{pc['tf']}m", style={"color": "#475569", "fontWeight": "700",
                      "fontSize": "0.58rem", "minWidth": "26px", "display": "inline-block"}),
            html.Span(f"needs {pc['tf']}m of capture · ~{pc['eta']}", style={
                      "color": "#334155", "fontSize": "0.52rem", "fontStyle": "italic"}),
        ], style={"display": "flex", "alignItems": "center",
                  "padding": "4px 0", "borderBottom": "1px solid #111d2e", "opacity": 0.7}))

    flag_divs = [html.Div(("⚠ " if t == "warn" else "✓ ") + m, style={
        "color": "#fb923c" if t == "warn" else "#4ade80", "fontSize": "0.52rem",
        "lineHeight": "1.35", "marginBottom": "3px", "whiteSpace": "normal",
        "background": "#1a1407" if t == "warn" else "#0a1f12",
        "border": f"1px solid {'#7c2d12' if t == 'warn' else '#14532d'}",
        "borderRadius": "3px", "padding": "3px 5px"}) for t, m in r.get("flags", [])]

    stk = r.get("stack", {}) or {}
    sdir = stk.get("dir", 0)
    sclr = "#22c55e" if sdir > 0 else "#ef4444" if sdir < 0 else "#64748b"
    sarr = "▲" if sdir > 0 else "▼" if sdir < 0 else "•"
    sword = "BULLISH" if sdir > 0 else "BEARISH" if sdir < 0 else "NO TREND"
    n_aligned = len(stk.get("aligned", []))
    return html.Div([
        html.Div([
            html.Span(f"OI bias ", style={"color": "#64748b", "fontSize": "0.52rem"}),
            html.Span(r["bias"], style={"color": sig.color(r["bias"]),
                      "fontWeight": "700", "fontSize": "0.56rem"}),
            html.Span(_fmt_futoi(r.get("fut_oi_chg")),
                      style={"color": "#67e8f9", "fontSize": "0.5rem"}),
            html.Span(f"  {r['now']}", style={"color": "#475569", "fontSize": "0.5rem",
                      "marginLeft": "auto"}),
        ], title=_BIAS_TIP,
           style={"display": "flex", "alignItems": "center", "marginBottom": "2px", "cursor": "help"}),
        # Multi-TF stack consensus — the stable cross-timeframe direction. ★N = how
        # many frames are stack-confirmed (tradeable), not lone-TF noise.
        html.Div([
            html.Span("stack ", style={"color": "#64748b", "fontSize": "0.52rem"}),
            html.Span(f"{sarr} {sword}", style={"color": sclr, "fontWeight": "700",
                      "fontSize": "0.56rem"}),
            html.Span(f"  {stk.get('agree', 0)}/{stk.get('total', 0)} TFs agree"
                      + (f"  ·  ★{n_aligned} tradeable" if n_aligned else "  ·  no confirmed entry"),
                      style={"color": "#475569", "fontSize": "0.5rem"}),
        ], title=_STACK_TIP,
           style={"display": "flex", "alignItems": "center", "marginBottom": "4px", "cursor": "help"}),
        *flag_divs,
        html.Div(rows),
        html.Div("option OI·price·volume per TF · futures: price/vol/basis "
                 "(intraday futures OI not in the Fyers feed)",
                 style={"color": "#334155", "fontSize": "0.46rem", "marginTop": "5px"}),
    ], style={**MONO})


# ── Multi-timeframe trend ribbon (5m → weekly) ──────────────────────────────────
_TREND_CLR = {1: "#22c55e", -1: "#ef4444", 0: "#64748b"}


def _trend_cell(key: str, t: dict) -> "html.Div":
    clr = _TREND_CLR.get(t.get("dir", 0), "#64748b")
    return html.Div([
        html.Div(key, style={"color": "#475569", "fontSize": "0.46rem"}),
        html.Div(t.get("label", "—"), style={"color": clr, "fontSize": "0.55rem", "fontWeight": "700"}),
    ], style={"textAlign": "center", "minWidth": "64px", "padding": "3px 4px",
              "borderRadius": "4px", "background": f"{clr}14", "border": f"1px solid {clr}33"})


def _render_trend_matrix() -> "html.Div":
    """Per-index trend across 5m/15m/1h/daily/weekly + alignment verdict. CONTEXT,
    not a signal — tells you whether an intraday move rides or fights the bigger trend."""
    if not _TREND_AVAILABLE:
        return html.Div()
    head = html.Div([
        html.Span("📈 MULTI-TF TREND", style={"color": "#a78bfa", "fontWeight": "700",
                  "fontSize": "0.68rem", "letterSpacing": "0.06em"}),
        html.Span("  5m → weekly · ride the aligned, scalp the counter-trend",
                  style={"color": "#64748b", "fontSize": "0.54rem"}),
    ], style={"marginBottom": "6px"})
    rows = []
    for sym in INDEX_SYMBOLS:
        try:
            r = trend_matrix.trend_index(sym, fetch_ohlcv)
        except Exception:
            continue
        rows.append(html.Div([
            html.Div(LABELS.get(sym, sym), style={"color": COLORS.get(sym, "#a78bfa"),
                     "fontWeight": "700", "fontSize": "0.6rem", "minWidth": "92px"}),
            html.Div([_trend_cell(k, t) for k, t in r["ribbon"]],
                     style={"display": "flex", "gap": "5px", "flexWrap": "wrap"}),
            html.Div(r["verdict"], style={"color": r["vclr"], "fontSize": "0.55rem",
                     "fontWeight": "600", "marginLeft": "auto", "textAlign": "right"}),
        ], style={"display": "flex", "alignItems": "center", "gap": "10px",
                  "padding": "5px 0", "borderBottom": "1px solid #111d2e"}))
    return html.Div([head, *rows], style={
        "marginBottom": "12px", "padding": "10px 12px", "borderRadius": "10px",
        "background": "#0c1018", "border": "1px solid #241f3a", **MONO})


@app.callback(
    Output("itf-content", "children"),
    Input("itf-idx",    "value"),
    Input("setup-tick", "n_intervals"),
)
def update_intraday_tf(sym, _):
    # Route through the canonical snapshot so the sidebar footprint shares the
    # Trade Book's one-instant computation (live, ~1-tick TTL) instead of a
    # redundant intraday_tf.analyze recompute.
    # Weekend/holiday: the live snapshot is the dead static-quote feed — follow the
    # ghost-practice clock instead so the sidebar matches the CHARTS replay.
    now = datetime.datetime.now(IST)
    if not is_trading_day(now):
        day, hhmm = _ghost_ctx(None)
        if day != now.date().isoformat():
            return _render_intraday_tf(sym or "NSE:NIFTY50-INDEX",
                                       f"{day}T{hhmm}:00+05:30", date=day)
    snap = get_snapshot(None) if _SNAPSHOT_OK else None
    return _render_intraday_tf(sym or "NSE:NIFTY50-INDEX", None, snap)


@app.callback(
    Output("trend-panel", "children"),
    Input("signal-tick", "n_intervals"),
)
def update_trend_matrix(_):
    return _render_trend_matrix()


def _liveoi_db_fallback(sym: str):
    """Latest persisted OI session from DuckDB (for review when in-memory is empty)."""
    try:
        from intraday_db import idb
        from types import SimpleNamespace
        cols = ("ts,spot,pcr,total_call_oi,total_put_oi,atm_iv,call_wall,put_wall,max_pain,"
                "atm_call_oi,atm_put_oi,near_call_oi,near_put_oi,atm_call_iv,atm_put_iv")
        for d in reversed(idb.list_sessions() or []):
            df = idb.query(f"SELECT {cols} FROM oi_snapshots WHERE symbol='{sym}' ORDER BY ts", date=d)
            if df is not None and not df.empty:
                out = [SimpleNamespace(
                    ts=pd.to_datetime(r["ts"]), spot=r["spot"], pcr=r["pcr"],
                    total_call_oi=r["total_call_oi"], total_put_oi=r["total_put_oi"],
                    atm_iv=r["atm_iv"], call_wall=r["call_wall"], put_wall=r["put_wall"],
                    max_pain=r["max_pain"], atm_call_oi=r["atm_call_oi"], atm_put_oi=r["atm_put_oi"],
                    near_call_oi=r["near_call_oi"], near_put_oi=r["near_put_oi"],
                    atm_call_iv=r["atm_call_iv"], atm_put_iv=r["atm_put_iv"])
                    for _, r in df.iterrows()]
                return out, str(d)[:10]
    except Exception:
        pass
    return [], None


def _render_liveoi(sym: str) -> "html.Div":
    """Live OI session time-series: total OI buildup, PCR, walls/max-pain — all vs spot."""
    series = oi_store.series(sym)
    src_note = "live"
    if not series:
        series, _sd = _liveoi_db_fallback(sym)
        src_note = f"last session {_sd}" if series else "live"
    cd = COLORS.get(sym, "#40c4ff")
    if not series:
        return html.Div(
            "No OI snapshots yet — they fill every ~90s during market hours "
            "(market closed / engine warming up).",
            style={"color": "#475569", "fontSize": "0.78rem", "padding": "20px 4px", **MONO})
    ts = [s.ts for s in series]
    last = series[-1]

    # ── EOD support context (last night's DCM levels) ─────────────────────────
    eod = {}
    try:
        from daily_context_bridge import get_bridge
        eod = get_bridge().get_panel_data(sym) or {}
    except Exception:
        eod = {}
    eod_cw = eod.get("top_call_strike"); eod_pw = eod.get("top_put_strike")
    eod_mp = eod.get("max_pain_price");  eod_pc = eod.get("prev_close")

    def metric(lab, val, c="#cbd5e1"):
        return html.Div([
            html.Div(lab, style={"color": "#475569", "fontSize": "0.55rem", "letterSpacing": "0.08em"}),
            html.Div(val, style={"color": c, "fontSize": "0.82rem", "fontWeight": "700", **MONO}),
        ], style={"marginRight": "22px", "marginBottom": "4px"})

    _n = lambda v: float(v) if v is not None and not (isinstance(v, float) and v != v) else 0.0
    pcr_c = "#22c55e" if _n(last.pcr) >= 1 else "#ef4444"
    strip = html.Div([
        metric("SPOT", f"{_n(last.spot):,.1f}", cd),
        metric("CALL OI", _fmt_oi(_n(last.total_call_oi)), "#ef4444"),
        metric("PUT OI", _fmt_oi(_n(last.total_put_oi)), "#22c55e"),
        metric("PCR", f"{_n(last.pcr):.2f}", pcr_c),
        metric("ATM IV", f"{_n(last.atm_iv):.1f}%"),
        metric("CALL WALL", f"{_n(last.call_wall):,.0f}", "#ef4444"),
        metric("PUT WALL", f"{_n(last.put_wall):,.0f}", "#22c55e"),
        metric("MAX PAIN", f"{_n(last.max_pain):,.0f}", "#a78bfa"),
    ], style={"display": "flex", "flexWrap": "wrap", "padding": "8px 4px",
              "marginBottom": "6px", "borderBottom": "1px solid #111d2e"})

    def lay(title, h=260, y2=False):
        d = dict(plot_bgcolor=BG, paper_bgcolor=BG, height=h, margin=dict(l=12, r=52, t=22, b=24),
                 title=dict(text=title, font=dict(color="#64748b", size=11), x=0.01),
                 xaxis=dict(gridcolor="#0f1a2a", tickfont=dict(color="#475569", size=9), tickformat="%H:%M"),
                 yaxis=dict(gridcolor="#0f1a2a", tickfont=dict(color="#64748b", size=9)),
                 legend=dict(orientation="h", y=1.14, x=0, font=dict(size=9, color="#94a3b8")),
                 hovermode="x unified")
        if y2:
            d["yaxis2"] = dict(overlaying="y", side="right", showgrid=False, tickfont=dict(color="#fbbf24", size=9))
        return d

    f1 = go.Figure()
    f1.add_trace(go.Scatter(x=ts, y=[s.total_call_oi for s in series], name="Call OI", line=dict(color="#ef4444", width=1.5)))
    f1.add_trace(go.Scatter(x=ts, y=[s.total_put_oi for s in series], name="Put OI", line=dict(color="#22c55e", width=1.5)))
    f1.add_trace(go.Scatter(x=ts, y=[s.spot for s in series], name="Spot", yaxis="y2", line=dict(color="#fbbf24", width=2)))
    f1.update_layout(**lay("Total OI buildup vs Spot", 270, y2=True))

    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=ts, y=[s.pcr for s in series], name="PCR", line=dict(color="#40c4ff", width=1.5),
                            fill="tozeroy", fillcolor="rgba(64,196,255,0.08)"))
    f2.update_layout(**lay("PCR (put / call OI)", 190))
    f2.add_hline(y=1.0, line=dict(color="#475569", width=1, dash="dash"))

    f3 = go.Figure()
    f3.add_trace(go.Scatter(x=ts, y=[s.spot for s in series], name="Spot", line=dict(color="#fbbf24", width=2)))
    f3.add_trace(go.Scatter(x=ts, y=[s.call_wall for s in series], name="Call wall", line=dict(color="#ef4444", width=1, dash="dot")))
    f3.add_trace(go.Scatter(x=ts, y=[s.put_wall for s in series], name="Put wall", line=dict(color="#22c55e", width=1, dash="dot")))
    f3.add_trace(go.Scatter(x=ts, y=[s.max_pain for s in series], name="Max pain", line=dict(color="#a78bfa", width=1, dash="dash")))
    f3.update_layout(**lay("Walls & Max Pain vs Spot  (— live · ‑‑ EOD support)", 250))
    # EOD support overlay: last night's DCM levels as horizontal reference lines
    def _eod_line(fig, y, color, label):
        if y:
            fig.add_hline(y=float(y), line=dict(color=color, width=1, dash="dashdot"),
                          annotation_text=f"EOD {label}", annotation_position="right",
                          annotation_font=dict(size=8, color=color))
    _eod_line(f3, eod_cw, "#ef4444", "call wall")
    _eod_line(f3, eod_pw, "#22c55e", "put wall")
    _eod_line(f3, eod_mp, "#a78bfa", "max pain")
    _eod_line(f3, eod_pc, "#64748b", "prev close")

    eod_bits = []
    if eod_pc: eod_bits.append(f"prev close {eod_pc:,.0f}")
    if eod_cw: eod_bits.append(f"call wall {eod_cw:,.0f}")
    if eod_pw: eod_bits.append(f"put wall {eod_pw:,.0f}")
    if eod_mp: eod_bits.append(f"max pain {eod_mp:,.0f}")
    eod_cap = (html.Div("🌙 EOD support (last night): " + "  ·  ".join(eod_bits),
                        style={"color": "#52708f", "fontSize": "0.6rem", "marginBottom": "6px", **MONO})
               if eod_bits else html.Div())

    g = lambda fig: dcc.Graph(figure=fig, config={"displayModeBar": False})
    return html.Div([
        strip,
        eod_cap,
        html.Div(f"{len(series)} snapshots · {ts[0]:%H:%M}–{ts[-1]:%H:%M} IST · {src_note}",
                 style={"color": "#334155", "fontSize": "0.55rem", "marginBottom": "4px", **MONO}),
        g(f1), g(f2), g(f3),
    ], style={"background": BG_CARD, "border": "1px solid #111d2e",
              "borderRadius": "10px", "padding": "14px 16px"})


@app.callback(
    Output("fp-modal",       "is_open"),
    Output("fp-modal-graph", "figure"),
    Output("fp-modal-title", "children"),
    Input({"type": "fp-tf", "sym": ALL, "tf": ALL}, "n_clicks"),
    State("regime-asof", "data"),
    prevent_initial_call=True,
)
def open_footprint_modal(clicks, asof_value):
    # A footprint TF row was clicked -> pop the OI/volume/premium chart for that
    # (index, timeframe). triggered_id carries which row; ignore the initial/no-op
    # fire where every n_clicks is still 0/None.
    if not clicks or not any(clicks):
        return no_update, no_update, no_update
    trig = dash.callback_context.triggered
    pid = trig[0]["prop_id"].rsplit(".n_clicks", 1)[0] if trig else ""
    if not pid:
        return no_update, no_update, no_update
    try:
        ident = json.loads(pid)
        sym, tf = ident["sym"], int(ident["tf"])
    except Exception:
        return no_update, no_update, no_update
    # Thread the replay DATE too, not just the time: if asof_value carries a past
    # day (e.g. regime-asof ever wired to replay), build_series must read THAT day's
    # mirror — passing time-only with date=None would read today's file under a past
    # as_of (empty, or a cross-day leak). None when live ("now").
    _aod = _parse_asof(asof_value)
    fig = _footprint_fig(sym, tf, asof_value,
                         date=(_aod.date().isoformat() if _aod else None))
    title = f"{LABELS.get(sym, sym)} · {tf}m footprint — ATM premium · OI · volume"
    return True, fig, title


@app.callback(
    Output("liveoi-content", "children"),
    Input("liveoi-idx",   "value"),
    Input("setup-tick",   "n_intervals"),
    Input("sel-sym",      "data"),
)
def update_liveoi(sym, _, sel):
    if sel != "LIVEOI":
        return no_update
    return _render_liveoi(sym or "NSE:NIFTY50-INDEX")


@app.callback(
    Output("trade-rec",      "children"),
    Output("velocity-panel", "children"),
    Output("oi-intel-panel", "children"),
    Input("tf-dd",           "value"),
    Input("setup-tick",      "n_intervals"),
    Input("expiry-dd",       "value"),
    State("sel-sym",         "data"),
    prevent_initial_call=True,
)
def update_trade_rec(tf_key, _, expiry, sym):
    if not sym or not tf_key:
        return html.Div(), html.Div(), html.Div()

    with _lock:
        spot = (_latest.get(sym) or {}).get("ltp", 0)

    # Fetch option chain with selected expiry
    oc_data = fetch_option_chain(sym, expiry or "")
    if oc_data.get("s") != "ok":
        return (html.Div("Option chain unavailable",
                         style={"color": "#475569", "fontSize": "0.65rem", **MONO}),
                html.Div(), html.Div())

    d   = oc_data.get("data", {})
    raw = d.get("optionsChain", [])
    expiry_data = d.get("expiryData", [])

    # Build strike_map
    strike_map: dict[int, dict] = {}
    for entry in raw:
        sp = entry.get("strike_price", -1)
        if sp <= 0: continue
        if sp not in strike_map: strike_map[sp] = {}
        ot = entry.get("option_type", "")
        if ot in ("CE", "PE"):
            strike_map[sp][ot] = entry

    tot_c_oi = d.get("callOi", 0)
    tot_p_oi = d.get("putOi",  0)
    pcr      = tot_p_oi / tot_c_oi if tot_c_oi else 0

    # Max pain
    mp_chain = [{"strike_price": sp,
                 "call_options": {"oi": strike_map[sp].get("CE", {}).get("oi", 0)},
                 "put_options":  {"oi": strike_map[sp].get("PE", {}).get("oi", 0)}}
                for sp in sorted(strike_map)]
    mp = compute_max_pain(mp_chain) if mp_chain else 0

    # Snapshot into OI store (background poller also does this, but capture here
    # for the active index at higher frequency when OC panel is open)
    snap = build_oi_snapshot(sym, spot, strike_map, tot_c_oi, tot_p_oi, pcr, mp)
    if snap:
        oi_store.add(snap)

    # Futures
    futures = fetch_futures(sym)

    # Build recommendation
    rec = build_recommendation(
        sym=sym, tf_key=tf_key, spot=spot,
        strike_map=strike_map, expiry_data=expiry_data,
        futures=futures, pcr=pcr, mp=mp,
        total_c_oi=tot_c_oi, total_p_oi=tot_p_oi,
    )

    # Component A: per-strike live OI-Dynamics map
    oi_dyn = intraday_oi_intel.analyze_oi_dynamics(strike_map, spot)

    # Component B: overnight->morning continuity vs last night's DCM walls
    _levels = {}
    try:
        from daily_context_bridge import get_bridge
        _levels = get_bridge().get_panel_data(sym) or {}
    except Exception:
        _levels = {}
    cont = intraday_oi_intel.analyze_continuity(strike_map, spot, _levels)

    # Component C: log a tracked paper trade when the engine fires an actionable setup
    try:
        intraday_trades.get_ledger().maybe_open(sym, rec, oi_dyn=oi_dyn, cont=cont)
    except Exception:
        pass

    return _render_trade_rec(rec, sym), _render_velocity_panel(sym), _render_oi_intel(oi_dyn, cont)


def _prediction_dropdown(is_open: bool = True) -> "html.Div":
    """Index Prediction as a collapsible, scrollable dropdown (shared by overview + Trade Book)."""
    try:
        _pred = _render_context_panel()
    except Exception:
        _pred = html.Div()
    return html.Details([
        html.Summary("📈  INDEX PREDICTION — Tomorrow's Directional Forecast",
                     style={"cursor": "pointer", "color": "#40c4ff", "fontSize": "0.78rem",
                            "fontWeight": "700", "letterSpacing": "0.05em", "padding": "8px 4px"}),
        html.Div(_pred, style={"maxHeight": "440px", "overflowY": "auto", "padding": "8px 4px"}),
    ], open=is_open, style={"border": "1px solid #14243a", "borderRadius": "8px",
                            "background": "#0b1320", "marginBottom": "12px", "padding": "0 6px"})


# ── Callback: daily context panel (60-second refresh, same interval) ──────────
@app.callback(
    Output("context-panel", "children"),
    Input("signal-tick",    "n_intervals"),
    State("sel-sym",        "data"),
)
def update_context_panel(_, sel):
    if sel:   # overview hidden while OC panel is active
        return no_update
    return _prediction_dropdown(is_open=True)


def _stored_candles(sym: str, resolution: str, limit: int = 200) -> "pd.DataFrame":
    """Read built candles from our own DuckDB `candles` table (for 1-sec, which Fyers can't serve)."""
    try:
        from intraday_db import idb
        sql = (f"SELECT ts, open, high, low, close, volume FROM candles "
               f"WHERE symbol = '{sym}' AND resolution = '{resolution}' "
               f"ORDER BY ts DESC LIMIT {int(limit)}")
        df = idb.query(sql)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.sort_values("ts").reset_index(drop=True)
        df["ts"] = pd.to_datetime(df["ts"])
        return df
    except Exception:
        return pd.DataFrame()


def _candle_fig(sym: str, res: str) -> "go.Figure":
    """Multi-timeframe OHLC candlestick chart. 1-sec from our store; rest from Fyers + live forming bar."""
    fig = go.Figure()
    if res == "1S":
        df = _stored_candles(sym, "1sec", 200)
        empty_msg = "1-sec candles build from your tick store during market hours — none stored yet"
    else:
        days = {"5S": 1, "15S": 1, "30S": 1, "1": 2, "5": 4, "15": 8, "60": 20, "D": 90}.get(res, 4)
        try:
            df = fetch_ohlcv(sym, res, days)
        except Exception:
            df = pd.DataFrame()
        empty_msg = "No candle data yet (market closed / warming up)"
    if df is None or df.empty:
        fig.update_layout(plot_bgcolor=BG, paper_bgcolor=BG, height=380,
                          margin=dict(l=40, r=16, t=8, b=28),
                          annotations=[dict(text=empty_msg, showarrow=False,
                                            font=dict(color="#475569", size=12))])
        return fig
    df = df.tail(160).copy()
    fig.add_trace(go.Candlestick(
        x=df["ts"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        increasing_line_color="#22c55e", decreasing_line_color="#ef4444",
        increasing_fillcolor="#22c55e", decreasing_fillcolor="#ef4444",
        line=dict(width=1), name="", showlegend=False))
    if res != "D":
        try:
            fig.add_trace(go.Scatter(x=df["ts"], y=_vwap(df), mode="lines",
                                     line=dict(color="#fbbf24", width=1.1), name="VWAP", showlegend=False))
        except Exception:
            pass
        # EOD support overlay (last night's DCM levels) — dotted reference lines
        try:
            from daily_context_bridge import get_bridge
            _eod = get_bridge().get_panel_data(sym) or {}
            for _y, _c, _lbl in [(_eod.get("prev_close"), "#64748b", "prev close"),
                                 (_eod.get("max_pain_price"), "#a78bfa", "max pain"),
                                 (_eod.get("top_call_strike"), "#ef4444", "call wall"),
                                 (_eod.get("top_put_strike"), "#22c55e", "put wall")]:
                if _y:
                    fig.add_hline(y=float(_y), line=dict(color=_c, width=1, dash="dot"),
                                  annotation_text=_lbl, annotation_position="left",
                                  annotation_font=dict(size=8, color=_c))
        except Exception:
            pass
    rb = [dict(bounds=["sat", "mon"])]
    if res != "D":
        rb.append(dict(bounds=[15.6, 9.25], pattern="hour"))   # hide overnight
    fig.update_layout(
        plot_bgcolor=BG, paper_bgcolor=BG, height=380,
        margin=dict(l=10, r=52, t=8, b=28),
        xaxis=dict(rangeslider_visible=False, gridcolor="#0f1a2a",
                   tickfont=dict(color="#475569", size=9), rangebreaks=rb),
        yaxis=dict(side="right", gridcolor="#0f1a2a", tickfont=dict(color="#64748b", size=9)),
        hovermode="x unified",
    )
    return fig


@app.callback(
    Output("cndl-chart", "figure"),
    Input("cndl-idx",   "value"),
    Input("cndl-tf",    "value"),
    Input("setup-tick", "n_intervals"),
    State("sel-sym",    "data"),
)
def update_candle_chart(sym, res, _, sel):
    if sel:   # overview hidden while a panel is active
        return no_update
    return _candle_fig(sym or "NSE:NIFTY50-INDEX", res or "5")


# ── Callback: trade signals (60-second refresh) ────────────────────────────────
@app.callback(
    Output("signal-panel", "children"),
    Input("signal-tick",   "n_intervals"),
    State("sel-sym",       "data"),
)
def update_signals(_, sel):
    if sel:   # option chain is active — skip signal update
        return no_update
    now = datetime.datetime.now(tz=IST).strftime("%H:%M:%S IST")
    try:
        results = run_full_analysis(INDEX_SYMBOLS)
    except Exception as e:
        return html.Div(f"Signal engine error: {e}",
                        style={"color": "#ef4444", "fontSize": "0.7rem", **MONO})
    return _render_signal_panel(results, now)


_NEWS_NAV_ON  = {"color": "#67e8f9", "fontSize": "0.7rem", "fontWeight": "700",
                 "cursor": "pointer", "padding": "0 8px", "userSelect": "none"}
_NEWS_NAV_OFF = {**_NEWS_NAV_ON, "color": "#334155", "cursor": "default", "opacity": 0.5}


@app.callback(
    Output("news-date", "data"),
    Output("news-date-label", "children"),
    Output("news-prev", "style"),
    Output("news-next", "style"),
    Input("news-prev", "n_clicks"),
    Input("news-next", "n_clicks"),
    Input("news-tick", "n_intervals"),
    State("news-date", "data"),
)
def _news_nav(_p, _n, _tick, cur):
    """◀ older / ▶ newer across captured days. Tick only refreshes the date list
    (so a brand-new day appears) without moving the user off their chosen date.
    Buttons grey out at the ends (oldest / newest=today)."""
    dates = _captured_days()                         # newest first — all captured days
    if not dates and _NEWS_AVAILABLE:
        dates = news_events.available_dates()
    if not dates:
        return None, "—", _NEWS_NAV_OFF, _NEWS_NAV_OFF
    cur = cur if cur in dates else dates[0]
    idx = dates.index(cur)
    trig = dash.callback_context.triggered
    src  = trig[0]["prop_id"].split(".")[0] if trig else ""
    if src == "news-prev":
        idx = min(idx + 1, len(dates) - 1)           # older
    elif src == "news-next":
        idx = max(idx - 1, 0)                         # newer
    new = dates[idx]
    tag = " (today)" if idx == 0 else ""
    prev_style = _NEWS_NAV_OFF if idx >= len(dates) - 1 else _NEWS_NAV_ON   # at oldest
    next_style = _NEWS_NAV_OFF if idx <= 0 else _NEWS_NAV_ON                # at newest
    return new, f"{new}{tag}", prev_style, next_style


@app.callback(Output("viewer-seed", "data"), Input("news-date", "data"))
def _seed_cards_on_date(date):
    """VIEWER: re-seed the index cards from the MASTER date (header nav) so the cards
    show that day's close — keeping cards, Charts and news on one date."""
    import os as _os
    if _os.environ.get("DASH_VIEWER") == "1" and date:
        _viewer_seed_latest(date)
    return date


@app.callback(Output("news-panel", "children"),
              Input("news-date", "data"), Input("news-tick", "n_intervals"),
              Input("news-tab", "value"), Input("news-sort", "value"))
def _update_news_panel(date, _n, tab, order):
    if not _NEWS_AVAILABLE:
        return no_update
    try:
        order = order or "time"
        return _render_news_panel(news_events.analyze_news(date=date, order=order),
                                  tab or "ALL", tape=(order == "time"))
    except Exception:
        return no_update


@app.callback(Output("macro-radar-panel", "children"), Input("news-tick", "n_intervals"))
def _update_macro_radar(_n):
    if not _MACRO_AVAILABLE:
        return no_update
    try:
        return _render_macro_radar(_MACRO_STATE)     # reads the poller cache, never fetches
    except Exception:
        return no_update


# ── Intraday Regime Cockpit (3-regime + band + event flags + post-3pm BTST carry) ──
_REG_COLOR = {"OPENING": "#64748b", "NORMAL": "#38bdf8", "HIGH_VOL": "#f59e0b"}
_POSTURE_COLOR = {"TRADE-BAND": "#38bdf8", "SIZE-DOWN": "#f59e0b",
                  "STAND-ASIDE": "#f87171", "BTST-CARRY": "#a78bfa", "WAIT": "#64748b"}
_CK_HEAD = {"color": "#67e8f9", "fontSize": "0.72rem", "fontWeight": "700", **MONO}
_CK_CHIP = {"background": "#1e293b", "color": "#fbbf24", "fontSize": "0.58rem",
            "padding": "1px 5px", "borderRadius": "4px", "marginLeft": "5px", **MONO}


def _parse_asof_hhmm(date, asof_str):
    """Parse an 'HH:MM' (or 'HHMM'/'HH.MM') replay time against the selected date into an
    IST datetime. Returns None on blank/garbage → live/latest. The date is the news-date
    (today if blank). Causal: read()/hour_forecast use only ts ≤ as_of, so this reconstructs
    the band exactly as it stood at that minute — for today OR any past captured day."""
    import datetime as _dt
    if not asof_str or not str(asof_str).strip():
        return None
    s = str(asof_str).strip().replace(".", ":")
    try:
        if ":" in s:
            hh, mm = s.split(":")[:2]
        elif s.isdigit() and len(s) in (3, 4):      # 930 / 1230
            hh, mm = s[:-2], s[-2:]
        else:
            return None
        hh, mm = int(hh), int(mm)
        if not (0 <= hh <= 23 and 0 <= mm <= 59):
            return None
    except Exception:
        return None
    d = date or _dt.datetime.now(IST).date().isoformat()
    return _dt.datetime.combine(_dt.date.fromisoformat(d), _dt.time(hh, mm), tzinfo=IST)


def _render_cockpit(date, asof_str=""):
    import datetime as _dt
    today = _dt.datetime.now(IST).date().isoformat()
    date_arg = None if (not date or date == today) else date
    as_of = _parse_asof_hhmm(date, asof_str)
    live = date_arg is None and as_of is None
    # Weekend/holiday "live" = the dead static-quote feed → junk bands (±3%+) and
    # phantom gap flags that LOOK tradeable. Follow the ghost-practice clock instead
    # (same day/instant the CHARTS section replays) so the whole page tells one story.
    ghost = False
    if live and not is_trading_day(_dt.datetime.now(IST)):
        _gd, _ghh = _ghost_ctx(None)
        if _gd != today:
            date_arg, live, ghost = _gd, False, True
            as_of = _dt.datetime.fromisoformat(f"{_gd}T{_ghh}:00+05:30")
    rows, carries = [], []
    for sym in INDEX_SYMBOLS:
        try:
            r = _cockpit.read(sym, date_arg, as_of)
        except Exception:
            continue
        if not r.get("ok"):
            rows.append(html.Div(f"  {LABELS.get(sym, sym):11} {r.get('note', '—')}",
                        style={"color": "#475569", "fontSize": "0.62rem", **MONO}))
            continue
        col = _REG_COLOR.get(r["regime"], "#94a3b8")
        band = (f"{r['band_lo']:.0f}–{r['band_hi']:.0f}"
                if r.get("band_lo") and r.get("band_hi") else "—")
        chips = [html.Span(fl, style=_CK_CHIP) for fl in r.get("flags", [])]
        # trend-regime badge: WHY the band width was adjusted (widen in a strong trend)
        _trm = r.get("band_regime_mult", 1.0)
        trend_span = html.Span(
            f" [{r.get('trend_regime', '')}" + (f" ×{_trm:g}]" if _trm != 1.0 else "]"),
            style={"color": "#f59e0b" if _trm > 1.0 else "#22d3ee" if _trm < 1.0 else "#475569",
                   "fontSize": "0.52rem"},
            title="trend regime (Kaufman efficiency, causal). The 60m band WIDENS ×1.08 in a "
                  "strong (BIG) trend — a symmetric vol band misses the trend's drift so it "
                  "under-covers there (measured on 2yr history, robust). Base width otherwise.")
        # measured 60m coverage tag: how often price actually lands in THIS index's band
        cov, cconf = r.get("band_cover"), r.get("band_conf", "none")
        _cc = {"ok": "#22c55e", "soft": "#eab308", "low": "#ef4444", "thin": "#64748b"}
        cov_span = (html.Span(
            f" {cov*100:.0f}% {'✓' if cconf=='ok' else '~' if cconf=='soft' else '⚠' if cconf=='low' else '·'}",
            style={"color": _cc.get(cconf, "#64748b"), "fontSize": "0.55rem"},
            title=f"measured: price lands in this 60m band {cov*100:.0f}% of the time "
                  f"(n={r.get('band_n', 0)}). Trust the band where this is high.")
            if cov is not None else html.Span(""))
        rows.append(html.Div([
            html.Span(f"{r['label']:11}", style={"color": "#e2e8f0", "fontWeight": "700"}),
            html.Span(f"{r['spot']:>9.1f}  ", style={"color": "#94a3b8"}),
            html.Span(f"{r['regime']:9}", style={"color": col, "fontWeight": "700"}),
            html.Span(f" {r['conf']:>2}%  ", style={"color": "#64748b"}),
            html.Span(f"{r.get('band_horizon', 60)}m band ",
                      style={"color": "#475569", "fontSize": "0.55rem"},
                      title="forward band horizon. Clipped to the time left in the session near "
                            "the close (a '60m' band at 15:18 would run past 15:30 into the "
                            "overnight gap — not intraday)."),
            html.Span(f"{band:>16}", style={"color": "#38bdf8"}),
            trend_span,
            cov_span,
            html.Span("  " + r["action"][:44], style={"color": "#94a3b8", "fontSize": "0.6rem"}),
            *chips,
        ], style={"fontSize": "0.64rem", "padding": "2px 0", "whiteSpace": "nowrap", **MONO}))
        # honest defensive POSTURE sub-line — SIZE + trade/no-trade, never a direction
        pcol = _POSTURE_COLOR.get(r.get("posture", ""), "#94a3b8")
        szx = f"{r['size']:.1f}x" if r.get("size") is not None else "—"
        rows.append(html.Div([
            html.Span(f"{'':13}→ ", style={"color": "#334155"}),
            html.Span(f"{r.get('posture', '—'):11}", style={"color": pcol, "fontWeight": "700"}),
            html.Span(f" size {szx:5} · ", style={"color": "#64748b"}),
            html.Span(f"{r.get('state', '—'):14}", style={"color": "#94a3b8"}),
            html.Span(f" · option-buy: {r.get('opt_buy', '')}", style={"color": "#64748b"}),
        ], style={"fontSize": "0.58rem", "padding": "0 0 3px 0", "whiteSpace": "nowrap", **MONO}))
        if r.get("carry"):
            carries.append(r["label"])
    def _hh(txt):   # section heading
        return html.Div(txt, style={"color": "#67e8f9", "fontSize": "0.62rem",
                        "fontWeight": "700", "padding": "5px 0 1px 0", **MONO})
    def _hl(txt, color="#94a3b8"):   # plain line
        return html.Div(txt, style={"color": color, "fontSize": "0.6rem",
                        "padding": "1px 0 1px 8px", **MONO})
    def _hm(txt, color="#38bdf8"):   # mono sample from the screen
        return html.Div(txt, style={"color": color, "fontSize": "0.6rem", "fontWeight": "700",
                        "padding": "2px 6px", "margin": "1px 0", "background": "#0b1220",
                        "borderLeft": "2px solid #334155", **MONO})
    help_box = html.Details([
        html.Summary("ⓘ how to read", style={"color": "#67e8f9", "fontSize": "0.58rem",
                     "cursor": "pointer", **MONO}),
        html.Div([
            _hl("A live map of where each index goes in the next 60 minutes — and whether "
                "to trade. It NEVER says buy or sell a direction (that loses money). It "
                "gives you a range and one order per index. Two lines each:", "#cbd5e1"),
            _hh("LINE 1 — the state (where price is + the range)"),
            _hm("NIFTY 50  23987  NORMAL 62%  60m band 23982–24030  74% ✓"),
            _hl("• 23987 = price now    • NORMAL = calm mood (62% sure)"),
            _hl("• 23982–24030 = where it likely sits in 1 hour (your range)"),
            _hl("• 74% ✓ = that band was right 74% of the time. GREEN ✓ = trust it. "
                "RED ⚠ (~50%) = shaky, rough guide only. · = too few days."),
            _hh("LINE 2 — your order (the → line; this is the boss)"),
            _hm("→ STAND-ASIDE  size 0.0x · choppy · option-buy: NO"),
            _hl("• the WORD = your order    • size = how big (1.0x full, 0.5x half, 0.0x none)"),
            _hl("• option-buy: NO = never buy calls/puts intraday (coin flip + 3% cost = you lose)"),
            _hh("THE ORDERS (just read the → word)"),
            _hl("• TRADE-BAND  = buy at the LOW edge, sell at the HIGH edge, nothing in the middle"),
            _hl("• STAND-ASIDE = choppy — do nothing"),
            _hl("• SIZE-DOWN   = wild day — half size"),
            _hl("• WAIT        = too early, range not formed yet"),
            _hl("• BTST-CARRY  = after 3 PM + strong close — buy the FUTURE, hold overnight, "
                "sell tomorrow ~9:30 (the one proven trade)"),
            _hh("EXAMPLE — band 23982–24030, order says TRADE-BAND"),
            _hl("• price near 23982 (low edge)  → BUY,  target middle ~24006, stop below 23982",
                "#4ade80"),
            _hl("• price near 24030 (high edge) → SELL, target middle ~24006, stop above 24030",
                "#f87171"),
            _hl("• price in the middle (~24006) → do NOTHING (no edge there)"),
            _hl("If the → word is STAND-ASIDE, or trust% is red ⚠ → SIT OUT, no trade.",
                "#f59e0b"),
        ], style={"padding": "5px 8px 7px 8px", "marginTop": "3px", "maxWidth": "660px",
                  "background": "#0b1220", "border": "1px solid #1e293b", "borderRadius": "6px"}),
    ], style={"display": "inline-block", "marginLeft": "10px", "verticalAlign": "middle"})
    if live:
        _tag = "live"
    elif ghost:
        _tag = f"👻 ghost {as_of:%Y-%m-%d} @ {as_of:%H:%M} — practice clock"
    elif as_of is not None:
        _tag = f"replay {as_of:%Y-%m-%d %H:%M} (band as it stood then)"
    else:
        _tag = f"replay {date}"
    head = html.Div([
        html.Span("🧭 INTRADAY REGIME COCKPIT   ", style=_CK_HEAD),
        html.Span(_tag,
                  style={"color": "#22c55e" if live else "#f59e0b", "fontSize": "0.6rem", **MONO}),
        help_box,
    ])
    tail = []
    _now_t = as_of.time() if as_of is not None else _dt.datetime.now(IST).time()
    post3 = _now_t >= _dt.time(15, 0)
    if carries:
        tail.append(html.Div(f"🌙 POST-3PM BTST-LONG: {', '.join(carries)} — strong close; long "
                             f"FUTURES, exit next ~09:30 (size for gap tail)",
                             style={"color": "#a78bfa", "fontSize": "0.62rem", "fontWeight": "700",
                                    "marginTop": "3px", **MONO}))
    elif post3:
        tail.append(html.Div("after 15:00 — no strong close → no BTST carry; flat into close",
                             style={"color": "#64748b", "fontSize": "0.58rem", **MONO}))
    tail.append(html.Div("no intraday directional arrow (loses at cost) — trade the band; "
                         "BTST only on a post-3pm strong close",
                         style={"color": "#475569", "fontSize": "0.55rem", "marginTop": "2px", **MONO}))
    return html.Div([head, html.Div(rows, style={"marginTop": "3px"})] + tail,
                    style={"padding": "8px 12px", "background": "#0b1220",
                           "border": "1px solid #1e293b", "borderRadius": "6px"})


@app.callback(Output("cockpit-panel", "children"),
              Input("news-date", "data"), Input("setup-tick", "n_intervals"),
              Input("cockpit-asof", "value"))
def _update_cockpit(date, _n, asof_str):
    if not _COCKPIT_OK:
        return no_update
    try:
        return _render_cockpit(date, asof_str)
    except Exception:
        return no_update


def _viewer_seed_latest(date=None) -> None:
    """VIEWER mode: fill _latest from the last SESSION tick (≤15:30) of each index in
    the mirror, mapped to the WS field names the overview/sidebar cards expect, so the
    cards show the day's last/close price instead of blanks (no live WS in viewer mode)."""
    from core.mirror_io import read_mirror
    for sym in INDEX_SYMBOLS:
        try:
            df = read_mirror("ticks", date, symbol=sym)
            if df is None or df.empty:
                continue
            sess = df[df["ts"].dt.time <= datetime.time(15, 30)]
            row = (sess if len(sess) else df).iloc[-1]
            ltp = float(row.get("ltp", 0) or 0)
            ch  = float(row.get("ch", 0) or 0)
            with _lock:
                _latest[sym] = {
                    "ltp": ltp, "ch": ch, "chp": float(row.get("chp", 0) or 0),
                    "open_price": float(row.get("day_open", 0) or 0),
                    "high_price": float(row.get("day_high", 0) or 0),
                    "low_price":  float(row.get("day_low", 0) or 0),
                    "prev_close_price": ltp - ch,   # not in tick feed; derive
                    # last tick's exchange time → lets the header freshness badge detect a
                    # stale feed in viewer mode (sync stopped, or VM capture died upstream).
                    "exch_feed_time": row["ts"].timestamp(),
                }
        except Exception:
            continue


# A viewer showing a stale mirror is the WORST failure mode in this system: the screen looks
# alive and is quietly lying. 2026-07-13: the laptop slept an hour, the dashboard restarted but
# the sync watcher did NOT, and the viewer served 11:19 data at 12:27 with no usable warning.
# The old "⚠ CAPTURE STALE" badge fired, but it is (a) a tiny header span and (b) MISLEADING —
# it blames CAPTURE when the VM was perfectly healthy and the LOCAL SYNC was the dead thing.
# Those two failures need OPPOSITE fixes, so we diagnose them apart:
#   SYNC_DEAD — the mirror FILE is not being rewritten (sync_from_vm is not running) -> restart
#               the sync HERE. The VM is probably fine.
#   VM_DEAD   — the file IS being rewritten every ~60s but its NEWEST TICK is old -> the VM
#               stopped producing. Fix the VM (token/capture), not the laptop.
_MIRROR_STALE_SEC = 240      # >4 missed 60s sync cycles = the data on screen is not live
_SYNC_DEAD_SEC = 180         # the mirror file should be republished every ~60s


def _viewer_mirror_health():
    """(state, msg) for the viewer's mirror freshness, or None when the check does not apply
    (capturer, non-trading day, outside market hours — a quiet feed is CORRECT then, and a
    monitor that cries wolf gets ignored). Cheap: reuses the exch_feed_time the seed loop
    already maintains, plus one os.stat — no parquet re-parse."""
    if not _ROLE_VIEWER:
        return None
    now = datetime.datetime.now(IST)
    try:
        from core.market_calendar import is_trading_day
        if not (is_trading_day(now.date())
                and datetime.time(9, 15) <= now.time() <= datetime.time(15, 31)):
            return None
        from core.constants import LIVE_DIR
        f = LIVE_DIR / f"{now.date().isoformat()}_ticks.parquet"
        if not f.exists():
            return ("SYNC_DEAD", "No mirror for today — nothing has been synced from the VM yet.")
        file_age = now.timestamp() - f.stat().st_mtime
        with _lock:
            fts = [float((t or {}).get("exch_feed_time") or 0) for t in _latest.values()]
        newest = max(fts) if fts else 0.0
        content_age = (now.timestamp() - newest) if newest else 1e9
        if content_age <= _MIRROR_STALE_SEC:
            return ("OK", "")
        mins = content_age / 60.0
        if file_age > _SYNC_DEAD_SEC:
            return ("SYNC_DEAD",
                    f"Your screen is {mins:.0f} MIN STALE. The local sync is NOT running — the "
                    f"mirror file has not been updated in {file_age/60:.0f} min. The VM is likely "
                    f"capturing fine; it is THIS laptop that stopped fetching. "
                    f"FIX: run dev.bat (or: python sync_from_vm.py --watch 60).")
        return ("VM_DEAD",
                f"Your screen is {mins:.0f} MIN STALE. The sync IS running (mirror rewritten "
                f"{file_age:.0f}s ago) but the VM has stopped producing ticks — capture is down "
                f"UPSTREAM. FIX: python check_vm_capture.py --fix")
    except Exception:
        return None


@app.callback(Output("mirror-banner", "children"), Input("fast-tick", "n_intervals"))
def _mirror_banner(_):
    """Full-width RED banner when the viewer's data is stale. Empty (zero height) when healthy,
    so it costs nothing visually until it matters."""
    h = _viewer_mirror_health()
    if not h or h[0] == "OK":
        return ""
    state, msg = h
    return html.Div([
        html.Span("⛔ STALE DATA — DO NOT TRADE OFF THIS SCREEN  ",
                  style={"fontWeight": "900", "letterSpacing": "0.04em"}),
        html.Span(msg, style={"fontWeight": "600"}),
    ], style={
        "background": "#7f1d1d", "color": "#fee2e2", "padding": "8px 14px",
        "borderRadius": "6px", "marginBottom": "8px", "fontSize": "0.72rem",
        "border": "1px solid #ef4444", "animation": "red-pulse 1.6s ease-out infinite",
    })


def _viewer_seed_loop() -> None:
    """Re-seed every 15s from the latest captured day (picks up new mirror data)."""
    import glob
    import time as _t
    from core.constants import LIVE_DIR
    while True:
        try:
            days = sorted(os.path.basename(p)[:10]
                          for p in glob.glob(str(LIVE_DIR / "*_ticks.parquet")))
            _viewer_seed_latest(days[-1] if days else None)
        except Exception:
            pass
        _t.sleep(15)


def _pid_alive(pid: int) -> bool:
    """Portable 'is this PID still running?' — Windows via OpenProcess+exit-code, POSIX
    via signal-0. Used to tell a live capture lock from a stale one."""
    import os
    if not pid or pid <= 0:
        return False
    try:
        if os.name == "nt":
            import ctypes
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            h = ctypes.windll.kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not h:
                return False
            code = ctypes.c_ulong()
            ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
            ctypes.windll.kernel32.CloseHandle(h)
            return code.value == 259                     # STILL_ACTIVE
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _resolve_role() -> str:
    """Capturer vs viewer — SAFE BY DEFAULT. This box CAPTURES (holds the Fyers WS,
    runs the pollers, WRITES the mirrors) only when it is the DESIGNATED capturer —
    a `.capture_host` marker beside the mirror dir — or capture is explicitly forced
    (TRADEBOT_CAPTURE=1). Anything else, including a bare `python dashboard.py`, runs
    as a read-only VIEWER, so a laptop can NEVER accidentally clobber the VM's synced
    mirrors. DASH_VIEWER=1 always forces viewer. Returns 'viewer' | 'capturer'."""
    import os
    from core.constants import LIVE_DIR
    if os.environ.get("DASH_VIEWER") == "1":
        return "viewer"
    forced = os.environ.get("TRADEBOT_CAPTURE") == "1"
    marked = (LIVE_DIR.parent / ".capture_host").exists()
    return "capturer" if (forced or marked) else "viewer"


def _acquire_capture_lock():
    """Enforce ONE capturer per box (the single-writer invariant, not by convention).
    Atomically create LIVE_DIR/.capture.lock with our PID (O_CREAT|O_EXCL). If it is
    already held by a LIVE pid → REFUSE loudly and exit (two capturers double-write /
    clobber the mirrors). A stale lock (dead pid) is taken over. Returns the lock Path."""
    import os, sys
    from core.constants import LIVE_DIR
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    lock = LIVE_DIR / ".capture.lock"
    for _ in range(2):
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return lock
        except FileExistsError:
            try:
                held = int((lock.read_text() or "0").strip() or "0")
            except Exception:
                held = 0
            if held and held != os.getpid() and _pid_alive(held):
                print(SEP)
                print(f"  REFUSING to capture — another capturer (PID {held}) already "
                      f"holds {lock.name}.")
                print("  Two capturers would double-write and clobber the mirrors.")
                print("  Run the read-only viewer instead:")
                print("      DASH_VIEWER=1 python dashboard.py")
                print(SEP)
                sys.exit(1)
            try:                                         # stale lock → reclaim, retry once
                lock.unlink()
            except Exception:
                pass
    print("  WARN: could not acquire capture lock; proceeding.", file=sys.stderr)
    return lock


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os, webbrowser
    # ROLE IS SAFE BY DEFAULT (see _resolve_role): a bare launch on a non-capture box
    # runs as a read-only VIEWER — no WebSocket, no pollers, no mirror writes — so it
    # can never CLOBBER the parquet mirrors synced from the VM (sync_from_vm.py). The
    # capturer role (VM, 24/7) is opted into via a `.capture_host` marker or
    # TRADEBOT_CAPTURE=1; a second capturer on the same box is refused by a PID lock.
    role = _resolve_role()
    VIEWER = role == "viewer"
    _ROLE_VIEWER = VIEWER          # expose to callbacks: viewer notifies off the synced log
    _explicit_viewer = os.environ.get("DASH_VIEWER") == "1"

    print(SEP)
    print("  NSE INDEX " + ("VIEWER (read-only, synced mirrors)" if VIEWER
                            else "LIVE DASHBOARD  +  OPTION CHAIN  [CAPTURER]"))
    print(SEP)

    if VIEWER:
        print("  VIEWER MODE — capture disabled. Reading data/intraday/live/ mirrors.")
        if not _explicit_viewer:
            print("  (default-safe: no .capture_host marker → viewer. To CAPTURE on this "
                  "box, create data/intraday/.capture_host or set TRADEBOT_CAPTURE=1.)")
        print("  Refresh them from the VM with:  python sync_from_vm.py  (or --watch).")
        _viewer_seed_latest()   # immediate seed so cards aren't blank on first paint
        # (the per-date re-seed is driven by the master news-date via _seed_cards_on_date)
        # Macro radar is network-only + in-memory (_MACRO_STATE, no mirror write) → safe
        # to run in VIEWER too, so the local viewer shows a LIVE global risk board. (The
        # NEWS poller stays OFF here — it writes the news_events mirror and would clobber
        # the copy synced from the VM; news is read from that synced mirror instead.)
        if _MACRO_AVAILABLE:
            threading.Thread(target=lambda: _macro_radar_poller(180),
                             daemon=True, name="macro-radar").start()
            print("  Macro radar poller started (viewer) — 180s intervals")
    else:
        # single-writer guard — refuse if another capturer already owns this box's mirrors
        _acquire_capture_lock()
        print(f"  CAPTURER — mirror lock held (PID {os.getpid()}). Writing mirrors.")
        raw_token     = _validate_token()
        _access_token = raw_token

        threading.Thread(
            target=_start_ws, args=(f"{APP_ID}:{raw_token}",),
            daemon=True, name="ws",
        ).start()
        print("  WebSocket started — connecting...")

        threading.Thread(
            target=_oi_background_poller,
            daemon=True, name="oi-poller",
        ).start()
        print("  OI snapshot poller started — 30s intervals")

        # NSE intraday futures-OI poller (Fyers doesn't serve it). May 403 from a
        # datacenter IP — degrades gracefully (the panel just omits futures OI).
        # 60s poll + timestamp-dedupe → catches each NSE refresh without storing dups.
        if _NSE_OI_AVAILABLE:
            threading.Thread(
                target=lambda: nse_oi.poll_loop(60),
                daemon=True, name="nse-oi",
            ).start()
            print("  NSE futures-OI poller started — 60s intervals")

        threading.Thread(
            target=_trade_tracker_poller,
            daemon=True, name="trade-tracker",
        ).start()
        print("  Trade tracker started — paper-trade outcomes every 20s")

        threading.Thread(
            target=_auto_signal_poller,
            daemon=True, name="auto-signal",
        ).start()
        print("  Auto signal eval started — all 4 indices every 60s")

        # News / event-impact poller — NSE filings + RBI releases, scored to −10..+10.
        # All-day window (filings land pre-open/post-close); may 403 from a datacenter IP.
        if _NEWS_AVAILABLE:
            threading.Thread(
                target=lambda: news_events.poll_loop(60),
                daemon=True, name="news",
            ).start()
            print("  News/event poller started — 60s intervals")

        # Macro radar — global risk board (crude/USD-INR/DXY/yields/US-futures/metals/VIX)
        # via yfinance into _MACRO_STATE; the panel reads the cache (never fetches in-request).
        if _MACRO_AVAILABLE:
            threading.Thread(
                target=lambda: _macro_radar_poller(180),
                daemon=True, name="macro-radar",
            ).start()
            print("  Macro radar poller started — 180s intervals")

        threading.Thread(
            target=_heartbeat_writer,
            daemon=True, name="heartbeat",
        ).start()

        # AUTHORITATIVE scout-alert detector — logs every NEW/SL/TARGET/BAND to the
        # canonical scout_alerts store whether or not a browser is open (so the evening
        # review is complete) and is the SOLE writer (no duplicate rows from multiple
        # tabs). Only the capturer runs it — never in VIEWER mode (would clobber the
        # synced mirrors via _export_parquet).
        threading.Thread(
            target=_scout_alert_poller,
            daemon=True, name="scout-alerts",
        ).start()
        print("  Scout-alert poller started — 30s, logs NEW/SL/TARGET/BAND")

    print(f"  Open  →  http://127.0.0.1:8050")
    print(SEP)

    # Auto-open the dashboard in the default browser once the server is up.
    # Suppressed via TRADEBOT_NO_BROWSER=1 (the supervisor sets it on auto-restarts
    # so a crash/WS-stall recovery doesn't spawn a fresh tab each time).
    if not os.environ.get("TRADEBOT_NO_BROWSER"):
        threading.Timer(1.5, lambda: webbrowser.open("http://127.0.0.1:8050")).start()

    # Host/port are env-overridable for containerised/cloud deploys. Defaults stay
    # 127.0.0.1:8050 for local use; the cloud image sets DASH_HOST=0.0.0.0 so the
    # Caddy reverse proxy (TLS + password) can reach it over the internal network.
    app.run(debug=False,
            host=os.environ.get("DASH_HOST", "127.0.0.1"),
            port=int(os.environ.get("DASH_PORT", "8050")))
