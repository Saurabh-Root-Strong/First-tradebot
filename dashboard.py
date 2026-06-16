"""
dashboard.py  —  NSE Index Live Dashboard + Option Chain
Left pane: 4 live index cards  |  Click any → full option chain (right pane)
Run:   .venv\Scripts\python.exe dashboard.py
Open:  http://127.0.0.1:8050
"""

import base64, json, sys, threading, time, datetime
from pathlib import Path
from collections import deque

import requests
import pandas as pd
import dash
from dash import dcc, html, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from fyers_apiv3.FyersWebsocket import data_ws
from signals import run_full_analysis, recommend_option, fetch_ohlcv, _vwap
from trade_setup import build_recommendation, TF_PROFILES
from intraday_store import candle_store, oi_store, build_oi_snapshot, session_phase, session_strategy, record_tick
import intraday_oi_intel
import intraday_trades
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
APP_ID     = "WVDZUTO6HL-100"
TOKEN_FILE = Path("access_token.txt")
IST        = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
OC_URL     = "https://api-t1.fyers.in/data/options-chain-v3"
SEP        = "─" * 58

INDEX_SYMBOLS = [
    "NSE:NIFTY50-INDEX",
    "NSE:NIFTYBANK-INDEX",
    "NSE:FINNIFTY-INDEX",
    "NSE:MIDCPNIFTY-INDEX",
]
LABELS = {
    "NSE:NIFTY50-INDEX":    "NIFTY 50",
    "NSE:NIFTYBANK-INDEX":  "BANK NIFTY",
    "NSE:FINNIFTY-INDEX":   "FIN NIFTY",
    "NSE:MIDCPNIFTY-INDEX": "MIDCAP NIFTY",
}
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

# ── Globals ────────────────────────────────────────────────────────────────────
_access_token = ""  # set in __main__; callbacks also read TOKEN_FILE directly as fallback
_lock   = threading.Lock()
_latest:  dict[str, dict]  = {}
_history: dict[str, deque] = {s: deque(maxlen=1800) for s in INDEX_SYMBOLS}
_ws     = None
_seen:  set[str] = set()

_oc_lock  = threading.Lock()
_oc_cache: dict = {"sym": None, "expiry": "", "data": None, "ts": 0.0}


# ── Token validation ───────────────────────────────────────────────────────────
def _validate_token() -> str:
    if not TOKEN_FILE.exists():
        print("\n  ERROR  access_token.txt not found.")
        print("  FIX    .venv\\Scripts\\python.exe fyers_auth.py\n")
        sys.exit(1)
    raw = TOKEN_FILE.read_text(encoding="utf-8").strip()
    try:
        payload = raw.split(".")[1]
        payload += "=" * (4 - len(payload) % 4)
        claims    = json.loads(base64.urlsafe_b64decode(payload))
        remaining = claims.get("exp", 0) - time.time()
        if remaining <= 0:
            exp_dt = datetime.datetime.fromtimestamp(claims["exp"], tz=IST)
            print(f"\n  ERROR  Token EXPIRED  ({exp_dt:%Y-%m-%d %H:%M IST})")
            print("  FIX    .venv\\Scripts\\python.exe fyers_auth.py\n")
            sys.exit(1)
        h, m = int(remaining // 3600), int((remaining % 3600) // 60)
        print(f"  Token  OK  —  fy_id: {claims.get('fy_id','?')}  expires in {h}h {m}m")
    except SystemExit:
        raise
    except Exception as e:
        print(f"  Token  WARNING: {e}")
    return raw


# ── Option chain API + cache ───────────────────────────────────────────────────
def _expiry_to_epoch(expiry_val: str) -> str:
    """expiry_val is already the epoch string returned by Fyers expiryData."""
    return expiry_val or ""


def _get_auth() -> str:
    """Always read token from file — avoids module-global timing issues with Dash threads."""
    raw = TOKEN_FILE.read_text(encoding="utf-8").strip() if TOKEN_FILE.exists() else ""
    return f"{APP_ID}:{raw}"


def fetch_option_chain(sym: str, expiry: str = "", n_strikes: int = 15) -> dict:
    with _oc_lock:
        c = _oc_cache
        if c["sym"] == sym and c["expiry"] == expiry and c["ts"] and time.time() - c["ts"] < 2:
            return c["data"]
    try:
        resp = requests.get(
            OC_URL,
            headers={
                "Authorization": _get_auth(),
                "Content-Type":  "application/json",
                "version":       "3",
            },
            params={
                "symbol":      sym,
                "strikecount": n_strikes,
                "timestamp":   _expiry_to_epoch(expiry) if expiry else "",
                "greeks":      "1",
            },
            timeout=10,
        )
        try:
            data = resp.json()
        except Exception:
            data = {"s": "error", "message": f"HTTP {resp.status_code}: {resp.text[:300]}"}
    except Exception as e:
        data = {"s": "error", "message": str(e)}
    with _oc_lock:
        _oc_cache.update({"sym": sym, "expiry": expiry, "data": data, "ts": time.time()})
    return data


# ── Futures helpers ────────────────────────────────────────────────────────────
_MONTH_ABB = {1:"JAN",2:"FEB",3:"MAR",4:"APR",5:"MAY",6:"JUN",
              7:"JUL",8:"AUG",9:"SEP",10:"OCT",11:"NOV",12:"DEC"}
_FUT_UNDERLYING = {
    "NSE:NIFTY50-INDEX":    "NIFTY",
    "NSE:NIFTYBANK-INDEX":  "BANKNIFTY",
    "NSE:FINNIFTY-INDEX":   "FINNIFTY",
    "NSE:MIDCPNIFTY-INDEX": "MIDCPNIFTY",
}
_FUT_LABELS = ["NEAR", "NEXT", "FAR "]


def _futures_symbols(index_sym: str) -> list[dict]:
    """Build near/next/far futures symbols for the given index."""
    ul = _FUT_UNDERLYING.get(index_sym, "")
    if not ul:
        return []
    now = datetime.datetime.now(tz=IST)
    out = []
    for i in range(3):
        mo = (now.month - 1 + i) % 12 + 1
        yr = now.year + (now.month - 1 + i) // 12
        sym = f"NSE:{ul}{str(yr)[2:]}{_MONTH_ABB[mo]}FUT"
        out.append({"symbol": sym, "label": _FUT_LABELS[i],
                    "month": f"{_MONTH_ABB[mo]} {yr}"})
    return out


_fut_cache: dict = {}
_fut_lock  = threading.Lock()


def _idb_write_futures(index_sym: str, futures: list) -> None:
    """Non-blocking DB write for the futures strip. Spot from live tick cache."""
    try:
        from intraday_db import idb
        spot = float((_latest.get(index_sym) or {}).get("ltp") or 0.0)
        idb.write_futures(index_sym, futures, spot)
    except Exception:
        pass


def fetch_futures(index_sym: str) -> list[dict]:
    """Fetch near/next/far futures quotes. Cached 2s."""
    with _fut_lock:
        c = _fut_cache.get(index_sym)
        if c and time.time() - c["ts"] < 2:
            return c["data"]

    sym_info = _futures_symbols(index_sym)
    if not sym_info:
        return []
    try:
        resp = requests.get(
            "https://api-t1.fyers.in/data/quotes",
            headers={"Authorization": _get_auth(),
                     "Content-Type": "application/json", "version": "3"},
            params={"symbols": ",".join(s["symbol"] for s in sym_info)},
            timeout=10,
        )
        raw = resp.json()
    except Exception:
        return []

    if raw.get("s") != "ok":
        return []

    q_map = {item["n"]: item.get("v", {}) for item in raw.get("d", [])}
    result = []
    for info in sym_info:
        v = q_map.get(info["symbol"], {})
        result.append({
            **info,
            "ltp":   v.get("lp",               0) or 0,
            "ch":    v.get("ch",                0) or 0,
            "chp":   v.get("chp",               0) or 0,
            "high":  v.get("high_price",        0) or 0,
            "low":   v.get("low_price",         0) or 0,
            "open":  v.get("open_price",        0) or 0,
            "prev":  v.get("prev_close_price",  0) or 0,
            "vol":   v.get("volume",            0) or 0,
            "bid":   v.get("bid",               0) or 0,
            "ask":   v.get("ask",               0) or 0,
            "atp":   v.get("atp",               0) or 0,
        })

    with _fut_lock:
        _fut_cache[index_sym] = {"data": result, "ts": time.time()}
    _idb_write_futures(index_sym, result)
    return result


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


def compute_max_pain(chain: list) -> float:
    # Max pain = price minimising total ITM intrinsic loss for option WRITERS.
    # CE writer loses max(0, test - strike): ITM when test > strike.
    # PE writer loses max(0, strike - test): ITM when test < strike.
    strikes = [r["strike_price"] for r in chain]
    best, mp = float("inf"), strikes[0]
    for test in strikes:
        loss = sum(
            ((r.get("call_options") or {}).get("oi") or 0) * max(0, test - r["strike_price"]) +
            ((r.get("put_options")  or {}).get("oi") or 0) * max(0, r["strike_price"] - test)
            for r in chain
        )
        if loss < best:
            best, mp = loss, test
    return mp


def _fmt_oi(v) -> str:
    """Format an OI/volume number. Handles negative values (OI unwinding)."""
    if not v: return "—"
    sign = "-" if v < 0 else ""
    a = abs(v)
    if a >= 1_000_000: return f"{sign}{a / 1_000_000:.2f}M"
    if a >= 1_000:     return f"{sign}{a / 1_000:.1f}K"
    return f"{sign}{int(a)}"


def compute_prediction(strike_map: dict, spot: float, pcr: float, mp: float) -> dict:
    """
    Multi-signal option chain prediction engine.
    Signals: PCR, Max Pain gravity, OI walls, OI change direction, IV skew.
    Returns verdict, score, signals, support, resistance.
    """
    if not strike_map or not spot:
        return {}

    strikes = sorted(strike_map.keys())

    # Max OI strikes (resistance = highest call OI, support = highest put OI)
    max_c_sp = max(strikes, key=lambda sp: (strike_map[sp].get("CE", {}).get("oi") or 0))
    max_p_sp = max(strikes, key=lambda sp: (strike_map[sp].get("PE", {}).get("oi") or 0))
    max_c_oi = strike_map[max_c_sp].get("CE", {}).get("oi") or 0
    max_p_oi = strike_map[max_p_sp].get("PE", {}).get("oi") or 0

    # OI change totals (positive = fresh writing, negative = unwinding)
    tot_c_oich = sum((strike_map[sp].get("CE", {}).get("oich") or 0) for sp in strikes)
    tot_p_oich = sum((strike_map[sp].get("PE", {}).get("oich") or 0) for sp in strikes)

    # Average IV (call vs put skew)
    c_ivs = [(strike_map[sp].get("CE", {}).get("greeks") or {}).get("iv") or 0 for sp in strikes]
    p_ivs = [(strike_map[sp].get("PE", {}).get("greeks") or {}).get("iv") or 0 for sp in strikes]
    avg_c_iv = sum(c_ivs) / len(c_ivs) if c_ivs else 0
    avg_p_iv = sum(p_ivs) / len(p_ivs) if p_ivs else 0

    score    = 0
    signals  = []

    # ── Signal 1: PCR ──────────────────────────────────────────────────────────
    if pcr > 1.3:
        score += 2
        signals.append(("🟢", f"PCR {pcr:.2f} — Heavy put writing → strong floor, bullish",         "bull"))
    elif pcr > 1.0:
        score += 1
        signals.append(("🟢", f"PCR {pcr:.2f} — More puts than calls → mild bullish bias",           "bull"))
    elif pcr > 0.85:
        signals.append(("⚪", f"PCR {pcr:.2f} — Balanced OI, no clear directional edge",            "neut"))
    elif pcr > 0.7:
        score -= 1
        signals.append(("🔴", f"PCR {pcr:.2f} — Call writing dominating → mild resistance",         "bear"))
    else:
        score -= 2
        signals.append(("🔴", f"PCR {pcr:.2f} — Heavy call writing → strong ceiling, bearish",      "bear"))

    # ── Signal 2: Max Pain gravity ─────────────────────────────────────────────
    mp_diff = (spot - mp) / mp * 100 if mp else 0
    if mp_diff > 1.5:
        score -= 1
        signals.append(("🔴", f"Spot {mp_diff:+.1f}% above Max Pain {mp:,.0f} → gravitational pull down",  "bear"))
    elif mp_diff < -1.5:
        score += 1
        signals.append(("🟢", f"Spot {mp_diff:+.1f}% below Max Pain {mp:,.0f} → gravitational pull up",    "bull"))
    else:
        signals.append(("⚪", f"Spot near Max Pain {mp:,.0f} ({mp_diff:+.1f}%) → range-bound expiry likely", "neut"))

    # ── Signal 3: Call wall (resistance) proximity ─────────────────────────────
    res_pct = (max_c_sp - spot) / spot * 100 if spot else 0
    if 0 < res_pct < 0.5:
        score -= 1
        signals.append(("🔴", f"Approaching Call wall {max_c_sp:,.0f} (OI {_fmt_oi(max_c_oi)}) → strong resistance just above", "bear"))
    elif res_pct <= 0:
        score -= 1
        signals.append(("🔴", f"Spot above Call wall {max_c_sp:,.0f} → shorts covering, volatile", "bear"))
    else:
        signals.append(("⚪", f"Resistance @ {max_c_sp:,.0f} (+{res_pct:.1f}%), Call OI {_fmt_oi(max_c_oi)}", "neut"))

    # ── Signal 4: Put wall (support) proximity ─────────────────────────────────
    sup_pct = (spot - max_p_sp) / spot * 100 if spot else 0
    if 0 < sup_pct < 0.5:
        score += 1
        signals.append(("🟢", f"Near Put wall {max_p_sp:,.0f} (OI {_fmt_oi(max_p_oi)}) → strong floor just below", "bull"))
    elif sup_pct <= 0:
        score -= 1
        signals.append(("🔴", f"Spot below Put wall {max_p_sp:,.0f} → puts in pain, breakdown risk", "bear"))
    else:
        signals.append(("⚪", f"Support @ {max_p_sp:,.0f} (-{sup_pct:.1f}%), Put OI {_fmt_oi(max_p_oi)}", "neut"))

    # ── Signal 5: OI change direction (fresh build vs unwinding) ──────────────
    if tot_p_oich > 0 and tot_c_oich > 0:
        ratio = tot_p_oich / tot_c_oich if tot_c_oich else 1
        if ratio > 1.3:
            score += 1
            signals.append(("🟢", f"Put OI growing {ratio:.1f}x faster than Call OI → fresh put writing, bullish", "bull"))
        elif ratio < 0.7:
            score -= 1
            signals.append(("🔴", f"Call OI growing {1/ratio:.1f}x faster than Put OI → fresh call writing, bearish", "bear"))
        else:
            signals.append(("⚪", "OI buildup balanced across calls and puts", "neut"))
    elif tot_c_oich < 0 and tot_p_oich > 0:
        score += 1
        signals.append(("🟢", "Call OI unwinding + Put OI adding → bullish shift", "bull"))
    elif tot_p_oich < 0 and tot_c_oich > 0:
        score -= 1
        signals.append(("🔴", "Put OI unwinding + Call OI adding → bearish shift", "bear"))

    # ── Signal 6: IV skew ──────────────────────────────────────────────────────
    if avg_p_iv > 0 and avg_c_iv > 0:
        iv_skew = avg_p_iv - avg_c_iv
        if iv_skew > 2:
            score -= 1
            signals.append(("🔴", f"Put IV {avg_p_iv:.1f}% > Call IV {avg_c_iv:.1f}% → fear premium in puts, bearish skew", "bear"))
        elif iv_skew < -2:
            score += 1
            signals.append(("🟢", f"Call IV {avg_c_iv:.1f}% > Put IV {avg_p_iv:.1f}% → demand for upside calls, bullish skew", "bull"))
        else:
            signals.append(("⚪", f"IV balanced — Call {avg_c_iv:.1f}%  Put {avg_p_iv:.1f}%", "neut"))

    # ── Verdict ────────────────────────────────────────────────────────────────
    if   score >= 4: trend, clr, meter = "STRONG BULLISH",  "#22c55e", 5
    elif score >= 2: trend, clr, meter = "BULLISH",         "#4ade80", 4
    elif score >= 1: trend, clr, meter = "MILDLY BULLISH",  "#86efac", 3
    elif score <= -4:trend, clr, meter = "STRONG BEARISH",  "#ef4444", -5
    elif score <= -2:trend, clr, meter = "BEARISH",         "#f87171", -4
    elif score <= -1:trend, clr, meter = "MILDLY BEARISH",  "#fca5a5", -3
    else:            trend, clr, meter = "NEUTRAL",          "#94a3b8", 0

    return {
        "trend": trend, "color": clr, "score": score,
        "signals": signals,
        "resistance": max_c_sp, "support": max_p_sp,
    }


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
_last_tick_wall = 0.0
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
    Background thread: snapshot option chain for all 4 indices every 3 minutes.

    Runs independently of user interaction so OI history accumulates from 9:15
    whether or not the OC panel is open.  Uses the nearest expiry for each index.
    Skips outside market hours to avoid wasting API quota.
    """
    OPEN  = datetime.time(9, 14)
    CLOSE = datetime.time(15, 31)
    POLL  = 180  # seconds between snapshots

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
                        data = fetch_option_chain(sym)
                        if data.get("s") != "ok":
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
                            near = sorted(strike_map, key=lambda sp: abs(sp - spot))[:17]
                            legs = []
                            for sp in near:
                                for side in ("CE", "PE"):
                                    e = strike_map[sp].get(side)
                                    if e:
                                        legs.append((sp, side,
                                                     e.get("ltp"), e.get("ltpch"),
                                                     e.get("oi"), e.get("oich"),
                                                     e.get("volume")))
                            idb.write_chain(sym, datetime.datetime.now(tz=IST), legs)
                        except Exception:
                            pass
                    except Exception:
                        pass
        except Exception:
            pass
        time.sleep(POLL)


def _fetch_quotes(symbols: list[str]) -> dict:
    """Fetch last price (lp) for a list of Fyers symbols via /data/quotes."""
    if not symbols:
        return {}
    out: dict = {}
    try:
        resp = requests.get(
            "https://api-t1.fyers.in/data/quotes",
            headers={"Authorization": _get_auth(), "version": "3"},
            params={"symbols": ",".join(symbols)},
            timeout=8,
        )
        for item in (resp.json().get("d") or []):
            n = item.get("n")
            lp = (item.get("v") or {}).get("lp")
            if n and lp is not None:
                out[n] = float(lp)
    except Exception:
        pass
    return out


def _trade_tracker_poller():
    """
    Component C: follow each open paper trade's option LTP to resolution.

    Every ~20s during market hours: fetch quotes for all open trades' option
    symbols, update MFE/MAE and resolve on SL/T1/T2. After 15:31 IST, close any
    still-open trades at last price (mark-to-close).
    """
    led = intraday_trades.get_ledger()
    eod_done_for: str = ""
    while True:
        try:
            now = datetime.datetime.now(tz=IST)
            opens = led.open_trades()
            syms  = [t["option_sym"] for t in opens if t.get("option_sym")]
            if syms and datetime.time(9, 14) <= now.time() <= datetime.time(15, 45):
                prices = _fetch_quotes(syms)
                if prices:
                    led.update_open_trades(prices)
                # Adaptive Layer-10 overlay: if a sudden shock now fires AGAINST an
                # open trade, flag it and tighten the stop to lock risk (no forced
                # exit — the next price tick still resolves it via the normal path).
                if _SHOCK_AVAILABLE:
                    for t in led.open_trades():        # re-read: some may have resolved above
                        try:
                            sh = intraday_shock.shock_against(t["index_sym"], t.get("direction"))
                            if not sh:
                                continue
                            head = sh["signals"][0][1] if sh.get("signals") else "opposing market shock"
                            note = f"⚠ REGIME SHIFT — {head}"
                            last = t.get("last_ltp") or t.get("entry_ltp") or 0
                            risk = t.get("risk") or 0
                            # Lock half the remaining risk: raise SL toward last price.
                            new_sl = (last - 0.5 * risk) if (last and risk) else None
                            led.flag_regime_shift(t["trade_id"], note, new_sl)
                        except Exception:
                            pass
            # End-of-session mark-to-close (once per day)
            if now.time() >= datetime.time(15, 31) and eod_done_for != now.date().isoformat():
                still = [t["option_sym"] for t in led.open_trades() if t.get("option_sym")]
                led.close_eod(_fetch_quotes(still) if still else {})
                eod_done_for = now.date().isoformat()
        except Exception:
            pass
        time.sleep(20)


def _auto_signal_poller():
    """
    Autonomous signal eval for ALL 4 indices (Component C — track-record coverage).

    The UI callback only logs trades for the index you're viewing; this evaluates
    every index every 60s during market hours and opens tracked trades, so the
    Today's-Trades section reflects the whole book, not just the active panel.
    Uses the 15-min intraday profile (same as the UI default).
    """
    led = intraday_trades.get_ledger()
    while True:
        try:
            now = datetime.datetime.now(tz=IST)
            if datetime.time(9, 16) <= now.time() <= datetime.time(15, 25):
                for sym in INDEX_SYMBOLS:
                    try:
                        with _lock:
                            spot = (_latest.get(sym) or {}).get("ltp", 0)
                        if not spot:
                            continue
                        oc = fetch_option_chain(sym)
                        if oc.get("s") != "ok":
                            continue
                        d   = oc.get("data", {})
                        raw = d.get("optionsChain", [])
                        if not raw:
                            continue
                        expiry_data = d.get("expiryData", [])
                        sm: dict = {}
                        for e in raw:
                            sp = e.get("strike_price", -1)
                            if sp > 0 and e.get("option_type") in ("CE", "PE"):
                                sm.setdefault(sp, {})[e["option_type"]] = e
                        tot_c = d.get("callOi", 0); tot_p = d.get("putOi", 0)
                        pcr   = tot_p / tot_c if tot_c else 0
                        mp_ch = [{"strike_price": sp,
                                  "call_options": {"oi": sm[sp].get("CE", {}).get("oi", 0)},
                                  "put_options":  {"oi": sm[sp].get("PE", {}).get("oi", 0)}}
                                 for sp in sorted(sm)]
                        mp  = compute_max_pain(mp_ch) if mp_ch else 0
                        rec = build_recommendation(
                            sym=sym, tf_key="15min", spot=spot, strike_map=sm,
                            expiry_data=expiry_data, futures=fetch_futures(sym),
                            pcr=pcr, mp=mp, total_c_oi=tot_c, total_p_oi=tot_p,
                        )
                        # A/B intelligence for the trade's edge reason
                        oi_dyn = intraday_oi_intel.analyze_oi_dynamics(sm, spot)
                        _lv = {}
                        try:
                            from daily_context_bridge import get_bridge
                            _lv = get_bridge().get_panel_data(sym) or {}
                        except Exception:
                            _lv = {}
                        cont = intraday_oi_intel.analyze_continuity(sm, spot, _lv)
                        led.maybe_open(sym, rec, oi_dyn=oi_dyn, cont=cont)
                    except Exception:
                        pass
        except Exception:
            pass
        time.sleep(60)


# ── Style constants ────────────────────────────────────────────────────────────
BG      = "#080d14"
BG_CARD = "#0f1623"
BG_SIDE = "#0a1020"
MONO    = {"fontFamily": "'Courier New', Courier, monospace"}

def _slug(sym: str) -> str:
    return sym.replace(":", "-").replace(".", "-")


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
app.index_string = app.index_string.replace("</head>", _CSS + "</head>")

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
        # Horizontal index chips (clickable nav). Today's Trades now lives in the left pane.
        html.Div([
            *[_header_nav_card(sym) for sym in INDEX_SYMBOLS],
            html.Div(style={"flex": "1 1 auto"}),     # spacer pushes actions to the right
            _header_action_chip("nav-liveoi", "📡", "LIVE OI", "#40c4ff"),
        ], style={"display": "flex", "flexWrap": "wrap", "gap": "8px",
                  "alignItems": "center", "marginTop": "10px"}),
    ], style={"padding": "14px 16px 10px"}),
    html.Div(className="header-line"),

    dbc.Row([
        # ── Left pane: Intraday-TF footprint matrix + Today's Trades ─────────────
        dbc.Col([
            html.Div([
                html.Span("📊 INTRADAY TF", style={"color": "#67e8f9", "fontWeight": "700",
                          "fontSize": "0.62rem", "letterSpacing": "0.1em"}),
                dcc.Dropdown(
                    id="itf-idx", clearable=False,
                    options=[{"label": LABELS[s], "value": s} for s in INDEX_SYMBOLS],
                    value="NSE:NIFTY50-INDEX",
                    style={"fontSize": "0.62rem", "marginTop": "5px", "color": "#0b1320"}),
            ], style={"marginBottom": "6px"}),
            dcc.Loading(html.Div(id="itf-content"), type="circle", color="#67e8f9"),
            # Today's Trades — moved here from the header; opens the Trade Book.
            html.Div(id="nav-tradebook", n_clicks=0, children=[
                html.Div("📒 TODAY'S TRADES", style={
                    "letterSpacing": "0.1em", "color": "#cbd5e1", "fontWeight": "700",
                    "fontSize": "0.6rem", "marginBottom": "4px"}),
                html.Div(id="sidebar-trades"),
                html.Div("click to open ▸", style={
                    "color": "#475569", "fontSize": "0.5rem", "marginTop": "4px"}),
            ], style={"marginTop": "14px", "padding": "10px 12px", "borderRadius": "8px",
                      "border": "1px solid #1e3a5f55", "borderLeft": "3px solid #fbbf24",
                      "background": BG_CARD, "cursor": "pointer"}),
        ], md=3, lg=3, style={
            "background": BG_SIDE, "padding": "14px 10px",
            "borderRight": "1px solid #111d2e", "minHeight": "calc(100vh - 120px)"}),

        # ── Main content ─────────────────────────────────────────────────────────
        dbc.Col([
            # OVERVIEW PANEL
            html.Div(id="overview-panel", children=[
                # Live index prices (4 cards)
                dbc.Row([_overview_card(s) for s in INDEX_SYMBOLS], className="gx-0 mb-2"),
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
        ], md=9, lg=9, style={"padding": "12px 16px"}),
    ], className="gx-0"),

    # State stores
    dcc.Location(id="url", refresh=False),
    dcc.Store(id="sel-sym",    data=None),
    dcc.Store(id="sel-expiry", data=""),
    # Regime Radar checkpoint lives in a static Store: the dropdown that sets it
    # is rendered dynamically inside the Trade Book, and a callback may not use a
    # dynamically-created component as an Input before it exists in the DOM.
    dcc.Store(id="regime-asof", data="now"),

    # Intervals
    dcc.Interval(id="fast-tick",   interval=1000,  n_intervals=0),
    dcc.Interval(id="oc-tick",    interval=2000,  n_intervals=0),
    dcc.Interval(id="signal-tick",interval=60000, n_intervals=0),
    dcc.Interval(id="setup-tick", interval=30000, n_intervals=0),

], fluid=True, style={"background": BG, "minHeight": "100vh", "padding": "0"})


# ── Compact number formatters ─────────────────────────────────────────────────

def _fmt_cr(v) -> str:
    """Format a ₹Cr value compactly for narrow chips: 12345 → +12.3K, -2500 → -2.5K."""
    if v is None:
        return "—"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "—"
    sign = "+" if v >= 0 else ""
    a    = abs(v)
    if a >= 1_00_000:                    # ≥ 1 lakh Cr → show in L
        return f"{sign if v >= 0 else '-'}{a/1_00_000:.1f}L"
    if a >= 10_000:                      # ≥ 10,000 Cr → "12.3K"
        return f"{sign if v >= 0 else '-'}{a/1_000:.0f}K"
    if a >= 1_000:                       # ≥ 1,000 Cr  → "1.2K"
        return f"{sign if v >= 0 else '-'}{a/1_000:.1f}K"
    return f"{v:+.0f}"


def _fmt_contracts(v) -> str:
    """Format FAO net contracts compactly: -259253 → -259K, 12000000 → +12M."""
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
                html.Span(f"  {confidence}", style={
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
            html.Span(f"  {confidence} CONF", style={
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
                    html.Span(f"  {conf:.0f}% conf",
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
@app.callback(
    Output("sel-sym",    "data"),
    Output("sel-expiry", "data"),
    [Input(f"nav-{_slug(s)}", "n_clicks") for s in INDEX_SYMBOLS],
    Input("nav-tradebook", "n_clicks"),
    Input("nav-liveoi",    "n_clicks"),
    State("sel-sym", "data"),
    prevent_initial_call=True,
)
def on_nav_click(*args):
    *_, _tradebook_clicks, _liveoi_clicks, current = args
    from dash import callback_context as ctx
    if not ctx.triggered:
        return current, ""
    tid = ctx.triggered[0]["prop_id"].split(".")[0]
    if tid == "nav-tradebook":
        return (None, "") if current == "TRADES" else ("TRADES", "")
    if tid == "nav-liveoi":
        return (None, "") if current == "LIVEOI" else ("LIVEOI", "")
    for sym in INDEX_SYMBOLS:
        if tid == f"nav-{_slug(sym)}":
            return (None, "") if current == sym else (sym, "")
    return current, ""


_URL_SHORT = {"NSE:NIFTY50-INDEX": "nifty50", "NSE:NIFTYBANK-INDEX": "banknifty",
              "NSE:FINNIFTY-INDEX": "finnifty", "NSE:MIDCPNIFTY-INDEX": "midcpnifty"}


@app.callback(Output("url", "pathname"), Input("sel-sym", "data"))
def _sync_url(sym):
    """Reflect the active section in the address bar so the current page is visible."""
    if sym == "TRADES":
        return "/trades"
    if sym == "LIVEOI":
        return "/live-oi"
    if sym in INDEX_SYMBOLS:
        return f"/chain/{_URL_SHORT.get(sym, 'index')}"
    return "/"


# ── Callback 2: toggle panels + highlight selected nav card ───────────────────
@app.callback(
    Output("overview-panel",  "style"),
    Output("oc-panel",        "style"),
    Output("trade-book-panel", "style"),
    Output("live-oi-panel",   "style"),
    Output("oc-title",        "children"),
    *[Output(f"nav-{_slug(s)}", "style") for s in INDEX_SYMBOLS],
    Input("sel-sym", "data"),
)
def toggle_view(sym):
    is_trades = (sym == "TRADES")
    is_liveoi = (sym == "LIVEOI")
    is_index  = bool(sym) and not is_trades and not is_liveoi

    ov_style = {"display": "block"} if not sym else {"display": "none"}
    oc_style = {"display": "block"} if is_index else {"display": "none"}
    tb_style = {"display": "block"} if is_trades else {"display": "none"}
    lo_style = {"display": "block"} if is_liveoi else {"display": "none"}

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
    return (ov_style, oc_style, tb_style, lo_style, title, *nav_styles)


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
    status = html.Span([html.Span("● ", style={"color": dot_c}),
                        html.Span(f"{lbl}  ·  {now}", style={"color": "#334155"})])
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


def _render_sidebar_trades() -> "html.Div":
    """Compact stat line shown inside the clickable 'Today's Trades' nav button."""
    led   = intraday_trades.get_ledger()
    today = datetime.datetime.now(tz=IST).date().isoformat()
    s     = led.stats(today)
    n_open = len(led.open_trades())
    if not s.get("n") and n_open == 0:
        return html.Div("no trades yet", style={"color": "#475569", "fontSize": "0.54rem", **MONO})
    parts = []
    if s.get("n"):
        hit = s.get("hit_rate"); avgr = s.get("avg_r")
        parts.append(html.Span(f"{s['wins']}W/{s['losses']}L ",
                               style={"color": "#94a3b8", "fontSize": "0.55rem"}))
        if hit is not None:
            parts.append(html.Span(f"{hit:.0f}% ",
                         style={"color": "#4ade80" if hit >= 50 else "#f87171",
                                "fontWeight": "700", "fontSize": "0.55rem"}))
        if avgr is not None:
            parts.append(html.Span(f"{avgr:+.1f}R ",
                         style={"color": "#4ade80" if avgr >= 0 else "#f87171", "fontSize": "0.55rem"}))
    if n_open:
        parts.append(html.Span(f"· {n_open} open", style={"color": "#fbbf24", "fontSize": "0.55rem"}))
    return html.Div(parts, style=MONO)


def _trade_levels_bar(t) -> "html.Div":
    """SL → entry → current → T1 → T2 progress bar for a single trade."""
    sl = t.get("sl"); t1 = t.get("t1"); t2 = t.get("t2")
    entry = t.get("entry_ltp"); r = t.get("r_multiple")
    cur = t.get("exit_ltp") if (t.get("status") or "OPEN") != "OPEN" else t.get("last_ltp")
    if not (sl and t2 and t2 > sl):
        return html.Div()
    span = t2 - sl
    def pct(x):
        return 0 if x is None else max(2, min(98, (x - sl) / span * 100))
    cur_clr = "#4ade80" if (r or 0) > 0 else "#f87171" if (r or 0) < 0 else "#fbbf24"
    return html.Div([
        html.Div(style={"position": "absolute", "left": f"{pct(entry)}%", "top": "-1px",
                        "width": "2px", "height": "9px", "background": "#cbd5e1"}),
        html.Div(style={"position": "absolute", "left": f"{pct(t1)}%", "top": "-1px",
                        "width": "2px", "height": "9px", "background": "#22c55e"}),
        html.Div(style={"position": "absolute", "left": f"{pct(cur)}%", "top": "-3px",
                        "width": "9px", "height": "9px", "borderRadius": "50%",
                        "background": cur_clr, "transform": "translateX(-4px)",
                        "boxShadow": f"0 0 5px {cur_clr}"}),
    ], style={"position": "relative", "height": "6px", "borderRadius": "3px",
              "background": "linear-gradient(90deg,#7f1d1d 0%,#3f3f46 42%,#14532d 100%)",
              "margin": "11px 3px 7px 3px"})


_REGIME_CLR = {"BULLISH": "#4ade80", "BEARISH": "#f87171", "NEUTRAL": "#94a3b8"}
_STAGE_CLR  = {"IMMINENT": "#ef4444", "BUILDING": "#f59e0b", "STABLE": "#22c55e"}


def _regime_risk_badge(t, fc=None):
    """Amber/red forward-looking badge when a regime flip is building against this trade."""
    if not _REGIME_AVAILABLE:
        return None
    try:
        risk = regime_forecast.trade_regime_risk(t, fc)
    except Exception:
        risk = None
    if not risk:
        return None
    clr = _STAGE_CLR.get(risk["stage"], "#f59e0b")
    return html.Div(risk["msg"][:170], style={
        "color": clr, "fontSize": "0.54rem", "marginTop": "3px", "fontWeight": "700",
        "lineHeight": "1.35", "whiteSpace": "normal", "background": "#1a1206",
        "border": f"1px solid {clr}66", "borderRadius": "3px", "padding": "3px 5px"})


def _render_regime_radar(asof_value=None) -> "html.Div":
    """
    Forward-looking Regime Radar with a 30-min time-machine dropdown.
    Selecting a checkpoint reconstructs the market regime + change-forecast as it
    stood at that moment (read from the persisted snapshot mirrors).
    """
    if not _REGIME_AVAILABLE:
        return html.Div()
    marks = regime_forecast.checkpoint_times()
    opts = [{"label": "● Now (live)", "value": "now"}] + \
           [{"label": t.strftime("%H:%M"), "value": t.isoformat()} for t in marks]
    val = asof_value or "now"
    as_of = None
    if val and val != "now":
        try:
            as_of = datetime.datetime.fromisoformat(val)
        except Exception:
            as_of = None

    m = regime_forecast.market_forecast(as_of)
    stage = m.get("stage", "—") if m.get("has_data") else "—"
    sclr  = _STAGE_CLR.get(stage, "#64748b")

    head = html.Div([
        html.Span("🛰 REGIME RADAR", style={"color": "#67e8f9", "fontWeight": "700",
                  "fontSize": "0.7rem", "letterSpacing": "0.06em"}),
        html.Span("  forecast as of", style={"color": "#64748b", "fontSize": "0.55rem"}),
        dcc.Dropdown(id="regime-checkpoint", options=opts, value=val, clearable=False,
                     style={"width": "130px", "fontSize": "0.62rem", "color": "#0b1320"}),
    ], style={"display": "flex", "alignItems": "center", "gap": "8px", "marginBottom": "6px"})

    if not m.get("has_data"):
        body = html.Div("Regime data warming up — need ~12 min of snapshots.",
                        style={"color": "#475569", "fontSize": "0.6rem"})
    else:
        nxt = m.get("next_dir") or "—"
        market_line = html.Div([
            html.Span(f"MARKET {m['regime']}", style={
                "color": _REGIME_CLR.get(m["regime"], "#94a3b8"), "fontWeight": "700",
                "fontSize": "0.66rem"}),
            html.Span(f"  · {stage}", style={"color": sclr, "fontWeight": "700", "fontSize": "0.62rem"}),
            html.Span(f"  next {nxt} in {m['eta']} ({m['confidence']}%)  · {m['coherence']}"
                      if stage != "STABLE" else f"  holding · {m['coherence']}",
                      style={"color": "#94a3b8", "fontSize": "0.58rem"}),
        ], style={"marginBottom": "4px"})
        rows = []
        for s in regime_forecast.INDEX_SYMBOLS:
            f = (m.get("per_index") or {}).get(s, {})
            if not f.get("has_data"):
                continue
            st = f["stage"]
            rows.append(html.Div([
                html.Span(LABELS.get(s, s), style={"color": COLORS.get(s, "#94a3b8"),
                          "fontWeight": "700", "fontSize": "0.58rem", "minWidth": "78px",
                          "display": "inline-block"}),
                html.Span(f["regime"], style={"color": _REGIME_CLR.get(f["regime"], "#94a3b8"),
                          "fontSize": "0.58rem", "fontWeight": "600"}),
                html.Span(f"  {st}" + (f" → {f['next_dir']} {f['eta']}" if st != "STABLE" and f["next_dir"] else ""),
                          style={"color": _STAGE_CLR.get(st, "#64748b"), "fontSize": "0.56rem"}),
            ], style={"marginBottom": "1px"}))
        body = html.Div([market_line, *rows])

    return html.Div([head, body], style={
        "marginBottom": "10px", "padding": "8px 10px", "borderRadius": "4px",
        "background": "#0a1622", "border": f"1px solid {sclr}33", **MONO})


_PB_ACTION_CLR = {"BUY CE": "#4ade80", "BUY PE": "#f87171", "WRITE PE": "#fbbf24",
                  "WRITE CE": "#fbbf24", "BUY FUT": "#60a5fa", "SELL FUT": "#60a5fa",
                  "NO TRADE": "#64748b"}
_PB_TONE_CLR = {"bull": "#4ade80", "bear": "#f87171", "flat": "#475569"}
# Direction tag + stance clarifier — so a WRITE PE (a bullish premium-sell) never
# reads as a bearish "put" trade at a glance.
_PB_DIR_TAG = {"BULLISH": ("▲ BULLISH", "#4ade80"),
               "BEARISH": ("▼ BEARISH", "#f87171"),
               "NEUTRAL": ("● NEUTRAL", "#94a3b8")}
_PB_STANCE_HINT = {"WRITE PE": "sell puts · bullish", "WRITE CE": "sell calls · bearish",
                   "BUY FUT": "long futures", "SELL FUT": "short futures"}


def _render_opening_playbook(asof_value=None) -> "html.Div":
    """Opening Playbook — the first-20-min F&O morning call, one card per index."""
    if not _PLAYBOOK_AVAILABLE:
        return html.Div()
    as_of = None
    if asof_value and asof_value != "now":
        try:
            as_of = datetime.datetime.fromisoformat(asof_value)
        except Exception:
            as_of = None
    try:
        pb = opening_playbook.playbook_all(as_of)
    except Exception:
        return html.Div()

    head = html.Div([
        html.Span("⚡ OPENING PLAYBOOK", style={"color": "#fbbf24", "fontWeight": "700",
                  "fontSize": "0.7rem", "letterSpacing": "0.06em"}),
        html.Span("  first-20-min F&O read · OI · premium · basis · EOD memory",
                  style={"color": "#64748b", "fontSize": "0.55rem"}),
        html.Span(f"  · {pb.get('coherence', '')}", style={"color": "#94a3b8", "fontSize": "0.55rem"}),
    ], style={"marginBottom": "6px"})

    cards = []
    for sym in INDEX_SYMBOLS:
        p = (pb.get("per_index") or {}).get(sym) or {}
        cd = COLORS.get(sym, "#40c4ff")
        if not p.get("has_data"):
            body = html.Div(p.get("note") or "no data", style={"color": "#475569", "fontSize": "0.55rem"})
            cards.append(dbc.Col([html.Div(LABELS.get(sym, sym), style={
                "color": cd, "fontWeight": "700", "fontSize": "0.6rem"}), body],
                md=3, style={"padding": "0 8px"}))
            continue
        act = p["action"]; aclr = _PB_ACTION_CLR.get(act, "#94a3b8")
        strike_txt = f" {p['strike']:,}" if p.get("strike") else ""
        flip = p.get("flip") or {}
        dirn = p.get("direction", "NEUTRAL")
        dtag, dclr = _PB_DIR_TAG.get(dirn, ("● NEUTRAL", "#94a3b8"))
        hint = _PB_STANCE_HINT.get(act, "")
        cards.append(dbc.Col([
            # Headline (always visible): label · conviction · direction stance
            html.Div([
                html.Span(LABELS.get(sym, sym), style={"color": cd, "fontWeight": "700",
                          "fontSize": "0.6rem", "letterSpacing": "0.06em"}),
                html.Span(f" {p['conviction']}%", style={"color": "#94a3b8", "fontSize": "0.55rem"}),
                html.Span(dtag, style={"color": dclr, "fontWeight": "700",
                          "fontSize": "0.52rem", "marginLeft": "auto"}),
            ], style={"display": "flex", "alignItems": "center"}),
            # Action + strike, with a plain-English stance clarifier (WRITE/FUT)
            html.Div([
                html.Span(f"{act}{strike_txt}", style={"color": aclr, "fontWeight": "700",
                          "fontSize": "0.82rem"}),
                html.Span(f"  · {hint}", style={"color": dclr, "fontSize": "0.5rem"}) if hint else None,
            ], style={"margin": "2px 0"}),
            html.Div(flip.get("msg", ""), style={"color": "#fb923c", "fontSize": "0.52rem",
                     "fontWeight": "700", "background": "#27160a", "border": "1px solid #7c2d12",
                     "borderRadius": "3px", "padding": "2px 4px", "marginBottom": "3px"})
                if flip.get("flipped") else None,
            # Invalidation stays visible — the one risk number you must see.
            html.Div(f"wrong {'below' if dirn == 'BULLISH' else 'above'} "
                     f"{p['invalidation']:,.0f}", style={"color": "#f87171",
                     "fontSize": "0.52rem", "marginTop": "2px", "fontWeight": "600"})
                if p.get("invalidation") and dirn != "NEUTRAL" else None,
            # Verbose rationale folded into a collapsible, scrollable disclosure.
            html.Details([
                html.Summary("why & factors", style={"color": "#64748b",
                             "fontSize": "0.52rem", "cursor": "pointer", "marginTop": "3px"}),
                html.Div([
                    html.Div(p.get("why", ""), style={"color": "#94a3b8", "fontSize": "0.54rem",
                             "lineHeight": "1.35", "marginBottom": "3px", "whiteSpace": "normal"}),
                    *[html.Div(m, style={"color": _PB_TONE_CLR.get(tone, "#64748b"),
                              "fontSize": "0.52rem", "lineHeight": "1.4", "whiteSpace": "normal"})
                      for tone, m in (p.get("factors") or [])],
                    html.Div(p.get("margin_note", ""), style={"color": "#a16207",
                             "fontSize": "0.5rem", "marginTop": "2px"}) if p.get("margin_note") else None,
                ], style={"maxHeight": "150px", "overflowY": "auto", "marginTop": "3px",
                          "paddingRight": "4px"}),
            ]),
        ], md=3, style={"padding": "0 8px", "borderLeft": f"2px solid {cd}33"}))

    return html.Div([head, dbc.Row(cards, className="gx-0")], style={
        "marginBottom": "10px", "padding": "8px 10px", "borderRadius": "4px",
        "background": "#0d1420", "border": "1px solid #2a2410", **MONO})


# ── Session Conductor panel — the unified, evolving stance per index ────────────
_COND_DIR_CLR   = {"LONG": "#4ade80", "SHORT": "#f87171", "FLAT": "#94a3b8"}
_COND_DIR_ARROW = {"LONG": "▲", "SHORT": "▼", "FLAT": "•"}


def _cond_drv_clr(v: float) -> str:
    return "#4ade80" if v > 0.05 else "#f87171" if v < -0.05 else "#475569"


def _render_conductor() -> "html.Div":
    """One fused, evolving stance per index — resolves the stale-opening-vs-live
    dissonance into a single directive. Decision-support (does not auto-execute)."""
    if not _CONDUCTOR_AVAILABLE:
        return html.Div()
    try:
        per = session_conductor.conduct_all()
    except Exception:
        return html.Div()
    head = html.Div([
        html.Span("🎛 SESSION CONDUCTOR", style={"color": "#a78bfa", "fontWeight": "700",
                  "fontSize": "0.7rem", "letterSpacing": "0.06em"}),
        html.Span("  fused stance — opening thesis (decaying) ⊕ regime ⊕ momentum ⊕ OI"
                  "  ·  decision-support", style={"color": "#64748b", "fontSize": "0.55rem"}),
    ], style={"marginBottom": "6px"})

    cards = []
    for sym in INDEX_SYMBOLS:
        r = per.get(sym) or {}
        cd = COLORS.get(sym, "#a78bfa")
        if not r.get("has_data"):
            cards.append(dbc.Col([
                html.Div(LABELS.get(sym, sym), style={"color": cd, "fontWeight": "700", "fontSize": "0.6rem"}),
                html.Div(r.get("note", "warming up"), style={"color": "#475569", "fontSize": "0.52rem"})],
                md=3, style={"padding": "0 8px"}))
            continue
        dirn = r["direction"]; dclr = _COND_DIR_CLR.get(dirn, "#94a3b8")
        inst = r["instrument"]; aclr = _PB_ACTION_CLR.get(inst["action"], "#94a3b8")
        strike = f" {inst['strike']:,}" if inst.get("strike") else ""
        cards.append(dbc.Col([
            html.Div([
                html.Span(LABELS.get(sym, sym), style={"color": cd, "fontWeight": "700",
                          "fontSize": "0.6rem", "letterSpacing": "0.06em"}),
                html.Span(f"  {_COND_DIR_ARROW[dirn]} {dirn}", style={"color": dclr,
                          "fontWeight": "700", "fontSize": "0.56rem"}),
                html.Span(f"{r['conviction']}%", style={"color": "#94a3b8", "fontSize": "0.54rem",
                          "marginLeft": "auto"}),
            ], style={"display": "flex", "alignItems": "center"}),
            html.Div(f"{inst['action']}{strike}", style={"color": aclr, "fontWeight": "700",
                     "fontSize": "0.8rem", "margin": "2px 0"}),
            html.Div(f"{r['transition']}" + ("" if r["act_now"] else " · wait"),
                     style={"color": "#cbd5e1" if r["act_now"] else "#64748b", "fontSize": "0.5rem",
                            "fontWeight": "700" if r["act_now"] else "400", "marginBottom": "2px"}),
            html.Div(f"opened {r.get('opening_dir','?')} ({r.get('opening_action','—')}) → {dirn}"
                     if r.get("opening_action") else "",
                     style={"color": "#64748b", "fontSize": "0.5rem", "marginBottom": "2px"}),
            html.Details([
                html.Summary("drivers & why", style={"color": "#64748b", "fontSize": "0.5rem", "cursor": "pointer"}),
                html.Div([html.Div(inst.get("why", ""), style={"color": "#94a3b8",
                          "fontSize": "0.5rem", "marginBottom": "2px", "whiteSpace": "normal"})] +
                    [html.Div([
                        html.Span(("▲" if v > 0.05 else "▼" if v < -0.05 else "·") + " ",
                                  style={"color": _cond_drv_clr(v)}),
                        html.Span(f"{name} ", style={"color": "#94a3b8"}),
                        html.Span(f"{v:+.2f} ", style={"color": _cond_drv_clr(v), "fontWeight": "600"}),
                        html.Span(str(detail)[:44], style={"color": "#475569"}),
                    ], style={"fontSize": "0.5rem", "lineHeight": "1.5", "whiteSpace": "normal"})
                     for name, v, detail in r.get("drivers", [])],
                    style={"maxHeight": "120px", "overflowY": "auto", "marginTop": "2px"}),
            ]),
            html.Div(f"wrong {'below' if dirn == 'LONG' else 'above'} {r['invalidation']:,.0f}"
                     if r.get("invalidation") and dirn != "FLAT" else "",
                     style={"color": "#f87171", "fontSize": "0.5rem", "marginTop": "2px"}),
            html.Div(f"⏳ regime → {r['regime_eta']}" if r.get("regime_eta") else "",
                     style={"color": "#fb923c", "fontSize": "0.5rem"}),
        ], md=3, style={"padding": "0 8px", "borderLeft": f"2px solid {dclr}55"}))

    return html.Div([head, dbc.Row(cards, className="gx-0")], style={
        "marginBottom": "10px", "padding": "9px 11px", "borderRadius": "4px",
        "background": "#0c0e18", "border": "1px solid #2a2440", **MONO})


# ── Intraday-TF footprint matrix (left pane) ────────────────────────────────────
_ITF_TAG_CLR = {"LONG BUILDUP": "#22c55e", "SHORT COVER": "#86efac",
                "SHORT BUILDUP": "#ef4444", "LONG UNWIND": "#fca5a5", "BALANCED": "#64748b"}
_ITF_BIAS_CLR = {"BULLISH": "#22c55e", "BEARISH": "#ef4444", "NEUTRAL": "#94a3b8"}


def _render_intraday_tf(sym) -> "html.Div":
    """Per-timeframe OI·Price·Volume matrix + divergence flags for one index.
    Shows whether each 5/10/15/60-min frame is fresh buildup or positions CLOSING,
    so a rally on closing (distribution) or hidden call-writing is visible early."""
    if not _ITF_AVAILABLE:
        return html.Div()
    try:
        r = intraday_tf.analyze(sym)
    except Exception:
        return html.Div("—", style={"color": "#475569", "fontSize": "0.55rem"})
    if not r.get("has_data"):
        return html.Div(r.get("note", "warming up"),
                        style={"color": "#475569", "fontSize": "0.55rem", **MONO})

    rows = []
    for c in r["cells"]:
        up = c["px"] > 0.03; dn = c["px"] < -0.03
        pcl = "#22c55e" if up else "#ef4444" if dn else "#94a3b8"
        parr = "▲" if up else "▼" if dn else "·"
        bld = c["oi_build"]
        boi = ("↑ build" if bld > 0 else "↓ close" if bld < 0 else "· flat")
        bclr = "#4ade80" if bld > 0 else "#f87171" if bld < 0 else "#64748b"
        tclr = _ITF_TAG_CLR.get(c["tag"], "#64748b")
        ovol = "" if c["ovol"] is None else f"  vol {c['ovol']:.0f}L"
        farr = ("  fut▲" if (c["fpx"] or 0) > 0.03 else "  fut▼" if (c["fpx"] or 0) < -0.03 else "")
        rows.append(html.Div([
            html.Div([
                html.Span(f"{c['tf']}m", style={"color": "#cbd5e1", "fontWeight": "700",
                          "fontSize": "0.58rem", "minWidth": "26px", "display": "inline-block"}),
                html.Span(f"{parr}{c['px']:+.2f}%", style={"color": pcl, "fontSize": "0.58rem"}),
                html.Span(f"  {c['tag']}", style={"color": tclr, "fontSize": "0.55rem",
                          "fontWeight": "700", "marginLeft": "auto"}),
            ], style={"display": "flex", "alignItems": "center"}),
            html.Div([
                html.Span(f"OI {boi} {c['d_tot']:+.0f}L", style={"color": bclr, "fontSize": "0.52rem"}),
                html.Span(f"{ovol}{farr}", style={"color": "#475569", "fontSize": "0.52rem"}),
            ], style={"display": "flex", "justifyContent": "space-between"}),
        ], style={"padding": "4px 0", "borderBottom": "1px solid #111d2e"}))

    flag_divs = [html.Div(("⚠ " if t == "warn" else "✓ ") + m, style={
        "color": "#fb923c" if t == "warn" else "#4ade80", "fontSize": "0.52rem",
        "lineHeight": "1.35", "marginBottom": "3px", "whiteSpace": "normal",
        "background": "#1a1407" if t == "warn" else "#0a1f12",
        "border": f"1px solid {'#7c2d12' if t == 'warn' else '#14532d'}",
        "borderRadius": "3px", "padding": "3px 5px"}) for t, m in r.get("flags", [])]

    return html.Div([
        html.Div([
            html.Span(f"OI bias ", style={"color": "#64748b", "fontSize": "0.52rem"}),
            html.Span(r["bias"], style={"color": _ITF_BIAS_CLR.get(r["bias"], "#94a3b8"),
                      "fontWeight": "700", "fontSize": "0.56rem"}),
            html.Span(f"  {r['now']}", style={"color": "#475569", "fontSize": "0.5rem",
                      "marginLeft": "auto"}),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "4px"}),
        *flag_divs,
        html.Div(rows),
        html.Div("option OI·price·volume per TF · futures: price/vol/basis "
                 "(intraday futures OI not in the Fyers feed)",
                 style={"color": "#334155", "fontSize": "0.46rem", "marginTop": "5px"}),
    ], style={**MONO})


def _trade_card(t, fc=None) -> "html.Div":
    st = t.get("status") or "OPEN"
    r  = t.get("r_multiple")
    clr = _TRADE_ST_CLR.get(st, "#94a3b8")
    dirn = t.get("direction"); strike = t.get("strike") or 0
    entry = t.get("entry_ltp"); sl = t.get("sl"); t1 = t.get("t1"); t2 = t.get("t2")
    cur = t.get("exit_ltp") if st != "OPEN" else t.get("last_ltp")
    mfe = t.get("mfe"); mae = t.get("mae")
    tm = (t.get("opened_ts") or "")[11:16]
    dir_clr = "#4ade80" if dirn == "CE" else "#f87171"
    rtxt = f"{r:+.2f}R" if r is not None else "live"
    r_clr = "#4ade80" if (r or 0) > 0 else "#f87171" if (r or 0) < 0 else "#fbbf24"
    lvl = lambda lab, v, c: html.Span([html.Span(lab, style={"color": "#475569"}),
                                       html.Span(f"{v:.1f}", style={"color": c, "fontWeight": "600"})],
                                      style={"fontSize": "0.55rem", "marginRight": "6px"})
    return html.Div([
        html.Div([
            html.Span(f"{tm} ", style={"color": "#475569", "fontSize": "0.58rem"}),
            html.Span(f"{dirn} {strike:,.0f} ", style={"color": dir_clr, "fontWeight": "700", "fontSize": "0.74rem"}),
            html.Span(st, style={"color": clr, "fontWeight": "700", "fontSize": "0.62rem"}),
            html.Span(f"  {rtxt}", style={"color": r_clr, "fontSize": "0.66rem", "fontWeight": "700"}),
            html.Span("  T1✓ BE", style={"color": "#22c55e", "fontSize": "0.55rem", "fontWeight": "700"})
                if (st == "OPEN" and t.get("t1_booked")) else None,
            html.Span(f"  {t.get('conviction') or ''}", style={"color": "#475569", "fontSize": "0.52rem"}),
        ]),
        _trade_levels_bar(t),
        html.Div([
            lvl("SL ", sl, "#f87171"), lvl("E ", entry, "#94a3b8"),
            lvl("● ", cur, "#e2e8f0"), lvl("T1 ", t1, "#22c55e"), lvl("T2 ", t2, "#16a34a"),
        ]),
        html.Div(f"peak {mfe:.1f} / low {mae:.1f}" if mfe and mae else "",
                 style={"color": "#334155", "fontSize": "0.5rem", "marginTop": "2px"}),
        # Adaptive Layer-10 regime-shift banner (set when a shock fired against this trade).
        html.Div((t.get("regime_flag") or "")[:160],
                 style={"color": "#fb923c", "fontSize": "0.54rem", "marginTop": "3px",
                        "fontWeight": "700", "lineHeight": "1.35", "whiteSpace": "normal",
                        "background": "#27160a", "border": "1px solid #7c2d12",
                        "borderRadius": "3px", "padding": "3px 5px"}) if t.get("regime_flag") else None,
        # Forward-looking regime-RISK forecast (only when a flip is building against an OPEN trade).
        _regime_risk_badge(t, fc) if st == "OPEN" else None,
        html.Div((t.get("reason") or "")[:180],
                 style={"color": "#52708f", "fontSize": "0.52rem", "marginTop": "3px",
                        "lineHeight": "1.4", "whiteSpace": "normal"}),
    ], style={"padding": "9px 0", "borderBottom": "1px solid #111d2e", **MONO})


def _render_trade_book(asof_value=None) -> "html.Div":
    """Full Trade Book cockpit — trades by index with levels, progress, and reason."""
    led   = intraday_trades.get_ledger()
    today = datetime.datetime.now(tz=IST).date().isoformat()
    s     = led.stats(today)
    rows  = led.recent(200, today)
    with _lock:
        spots = {x: dict(_latest.get(x) or {}) for x in INDEX_SYMBOLS}

    # Live (now) regime forecasts — computed once per render, reused by every card.
    forecasts = {}
    if _REGIME_AVAILABLE:
        for x in INDEX_SYMBOLS:
            try:
                forecasts[x] = regime_forecast.forecast_index(x)
            except Exception:
                forecasts[x] = {}
    regime_radar = _render_regime_radar(asof_value)

    hdr_stats = []
    if s.get("n"):
        hit = s.get("hit_rate"); avgr = s.get("avg_r")
        hdr_stats = [
            html.Span(f"{s['n']} trades   ", style={"color": "#94a3b8"}),
            html.Span(f"{s['wins']}W / {s['losses']}L   ", style={"color": "#cbd5e1"}),
            html.Span(f"{hit:.0f}% hit   " if hit is not None else "",
                      style={"color": "#4ade80" if (hit or 0) >= 50 else "#f87171", "fontWeight": "700"}),
            html.Span(f"avg {avgr:+.2f}R" if avgr is not None else "",
                      style={"color": "#4ade80" if (avgr or 0) >= 0 else "#f87171"}),
        ]
    header = html.Div([
        html.Span("📒 TRADE BOOK — TODAY   ", style={
            "color": "#fbbf24", "fontWeight": "700", "fontSize": "0.95rem", "letterSpacing": "0.08em"}),
        html.Span(hdr_stats, style={"fontSize": "0.72rem"}),
    ], style={"marginBottom": "10px"})

    # ── Session-strategy banner: what the engine is doing RIGHT NOW by clock ──
    strat = session_strategy()
    gate_open = strat["allow_new_entry"]
    gate_clr  = "#22c55e" if gate_open else "#f87171"
    strat_banner = html.Div([
        html.Span(f"⏱ {strat['phase']}", style={
            "color": "#e2e8f0", "fontWeight": "700", "fontSize": "0.7rem", "letterSpacing": "0.06em"}),
        html.Span("  ENTRIES " + ("OPEN" if gate_open else "BLOCKED"), style={
            "color": gate_clr, "fontWeight": "700", "fontSize": "0.62rem"}),
        html.Span(f"  · min-conf {strat['min_conf']}%  · stop ×{strat['stop_mult']:.2f}"
                  f"  · target ×{strat['target_mult']:.2f}", style={
            "color": "#64748b", "fontSize": "0.58rem"}),
        html.Div(strat["bias"], style={
            "color": "#94a3b8", "fontSize": "0.6rem", "marginTop": "2px", "fontStyle": "italic"}),
    ], style={"marginBottom": "10px", "padding": "6px 9px", "borderRadius": "4px",
              "background": "#0c1626", "border": f"1px solid {'#15351f' if gate_open else '#3a1418'}",
              **MONO})

    pred_dropdown = _prediction_dropdown(is_open=False)
    conductor = _render_conductor()
    playbook = _render_opening_playbook(asof_value)

    if not rows:
        return html.Div([header, strat_banner, conductor, playbook, regime_radar, pred_dropdown, html.Div(
            "No trades yet today — signals log here as the engine fires across all 4 indices.",
            style={"color": "#475569", "fontSize": "0.75rem", **MONO})])

    cols = []
    for sym in INDEX_SYMBOLS:
        idx_rows = [t for t in rows if t.get("index_sym") == sym]
        cd = COLORS[sym]
        sp = spots.get(sym, {})
        spot_v = sp.get("ltp", 0); chp = sp.get("chp", 0)
        spot_clr = "#22c55e" if chp >= 0 else "#ef4444"
        cards = [_trade_card(t, forecasts.get(sym)) for t in idx_rows] or [
            html.Div("—", style={"color": "#334155", "fontSize": "0.7rem", "padding": "6px 0"})]
        cols.append(dbc.Col([
            html.Div([
                html.Span(LABELS[sym], style={"color": cd, "fontWeight": "700", "fontSize": "0.68rem",
                                              "letterSpacing": "0.08em"}),
                html.Span(f"  {spot_v:,.0f} ", style={"color": "#cbd5e1", "fontSize": "0.6rem"}) if spot_v else "",
                html.Span(f"{chp:+.2f}%" if spot_v else "", style={"color": spot_clr, "fontSize": "0.56rem"}),
            ], style={"borderBottom": f"2px solid {cd}55", "paddingBottom": "5px", "marginBottom": "4px"}),
            html.Div(cards),
        ], md=3, style={"padding": "0 9px"}))

    return html.Div([header, strat_banner, conductor, playbook, regime_radar, pred_dropdown,
                     dbc.Row(cols, className="gx-0")],
                    style={"background": BG_CARD, "border": "1px solid #111d2e",
                           "borderRadius": "10px", "padding": "16px 18px"})


@app.callback(
    Output("sidebar-trades", "children"),
    Input("setup-tick",      "n_intervals"),
)
def update_sidebar_trades(_):
    return _render_sidebar_trades()


@app.callback(
    Output("itf-content", "children"),
    Input("itf-idx",    "value"),
    Input("setup-tick", "n_intervals"),
)
def update_intraday_tf(sym, _):
    return _render_intraday_tf(sym or "NSE:NIFTY50-INDEX")


@app.callback(
    Output("trade-book-panel", "children"),
    Input("setup-tick",        "n_intervals"),
    Input("sel-sym",           "data"),
    Input("regime-asof",       "data"),
)
def update_trade_book(_, sel, asof_value):
    if sel != "TRADES":
        return no_update
    # asof_value mirrors the Regime Radar 30-min dropdown ("now" or an ISO mark)
    # via the regime-asof Store, so the selection persists across the 30s tick.
    return _render_trade_book(asof_value)


@app.callback(
    Output("regime-asof",      "data"),
    Input("regime-checkpoint", "value"),
    State("regime-asof",       "data"),
    prevent_initial_call=True,
)
def set_regime_asof(value, current):
    # The dropdown is re-created on every Trade Book render, which re-fires this
    # callback with the same value — pass no_update then, or it would loop.
    if not value or value == current:
        return no_update
    return value


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


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(SEP)
    print("  NSE INDEX LIVE DASHBOARD  +  OPTION CHAIN")
    print(SEP)

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
    print("  OI snapshot poller started — 3-min intervals")

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

    threading.Thread(
        target=_heartbeat_writer,
        daemon=True, name="heartbeat",
    ).start()

    print(f"  Open  →  http://127.0.0.1:8050")
    print(SEP)

    # Auto-open the dashboard in the default browser once the server is up.
    # Suppressed via TRADEBOT_NO_BROWSER=1 (the supervisor sets it on auto-restarts
    # so a crash/WS-stall recovery doesn't spawn a fresh tab each time).
    import os, webbrowser
    if not os.environ.get("TRADEBOT_NO_BROWSER"):
        threading.Timer(1.5, lambda: webbrowser.open("http://127.0.0.1:8050")).start()

    # Host/port are env-overridable for containerised/cloud deploys. Defaults stay
    # 127.0.0.1:8050 for local use; the cloud image sets DASH_HOST=0.0.0.0 so the
    # Caddy reverse proxy (TLS + password) can reach it over the internal network.
    app.run(debug=False,
            host=os.environ.get("DASH_HOST", "127.0.0.1"),
            port=int(os.environ.get("DASH_PORT", "8050")))
