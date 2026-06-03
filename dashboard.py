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
from signals import run_full_analysis, recommend_option
from trade_setup import build_recommendation, TF_PROFILES
from intraday_store import candle_store, oi_store, build_oi_snapshot
try:
    from daily_context_bridge import get_bridge as _get_ctx_bridge
    _CTX_BRIDGE_OK = True
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
    strikes = [r["strike_price"] for r in chain]
    best, mp = float("inf"), strikes[0]
    for test in strikes:
        loss = sum(
            ((r.get("call_options") or {}).get("oi") or 0) * max(0, r["strike_price"] - test) +
            ((r.get("put_options")  or {}).get("oi") or 0) * max(0, test - r["strike_price"])
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
    ft  = msg.get("exch_feed_time", 0)
    ts  = datetime.datetime.fromtimestamp(ft, tz=IST) if ft else datetime.datetime.now(tz=IST)
    ltp = float(msg.get("ltp", 0))
    vol = float(msg.get("volume", 0))
    with _lock:
        _latest[sym] = msg
        _history[sym].append((ts, ltp))
    # feed into candle builder — every tick, real time
    if ltp:
        candle_store.on_tick(sym, ltp, vol, ts)
    if sym not in _seen:
        _seen.add(sym)
        print(f"  [WS]   First tick  {LABELS[sym]:<14}  LTP {ltp:>10,.2f}")
        if len(_seen) == len(INDEX_SYMBOLS):
            print("  [WS]   All 4 indices live ✓")

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
                    except Exception:
                        pass
        except Exception:
            pass
        time.sleep(POLL)


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
        # Top row: label + click hint
        html.Div([
            html.Span(LABELS[sym], style={
                "color": color, "fontSize": "0.6rem",
                "letterSpacing": "0.15em", "fontWeight": "700",
            }),
            html.Span(" ⛓ CHAIN", style={
                "color": "#1e3a5f", "fontSize": "0.52rem",
                "float": "right", "letterSpacing": "0.05em",
            }),
        ], style={"marginBottom": "4px"}),
        html.Div(id=f"s-ltp-{slug}", children="—", style={
            **MONO, "fontSize": "1.2rem", "fontWeight": "bold", "color": "#e2e8f0",
        }),
        html.Div(id=f"s-chg-{slug}", children="", style={
            **MONO, "fontSize": "0.68rem", "marginTop": "2px",
        }),
        # Bottom click prompt
        html.Div("▶  Click for Option Chain + Futures", style={
            "fontSize": "0.52rem", "color": "#1e3a5f",
            "marginTop": "6px", "letterSpacing": "0.04em",
        }),
    ], style={
        "padding": "12px 14px", "marginBottom": "8px", "borderRadius": "8px",
        "border": f"1px solid {color}33", "borderLeft": f"3px solid {color}",
        "background": BG_CARD, "cursor": "pointer", "transition": "all 0.15s",
    })


# ── Overview panel components ──────────────────────────────────────────────────
def _overview_card(sym: str) -> dbc.Col:
    slug, color = _slug(sym), COLORS[sym]
    return dbc.Col(
        dbc.Card(dbc.CardBody([
            html.Div(LABELS[sym], style={
                "color": color, "fontSize": "0.68rem",
                "letterSpacing": "0.18em", "fontWeight": "700", "marginBottom": "8px",
            }),
            html.Div(id=f"ov-ltp-{slug}", style={
                **MONO, "fontSize": "2rem", "fontWeight": "bold",
                "color": "#f1f5f9", "lineHeight": "1",
            }),
            html.Div(id=f"ov-chg-{slug}", style={
                **MONO, "fontSize": "0.88rem", "marginTop": "5px",
            }),
            html.Hr(style={"borderColor": "#1e2d40", "margin": "10px 0 8px"}),
            html.Div(id=f"ov-ohlpc-{slug}", style={
                **MONO, "fontSize": "0.67rem", "color": "#4a5568", "lineHeight": "2",
            }),
        ]), style={
            "background": BG_CARD, "borderRadius": "10px",
            "border": "1px solid #1a2535", "borderTop": f"3px solid {color}",
        }),
        md=3, xs=6, className="mb-3 px-2",
    )


# ── Dash app ───────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    title="NSE Index Live",
    suppress_callback_exceptions=True,
)

# Inject hover CSS for nav cards
app.index_string = app.index_string.replace(
    "</head>",
    """<style>
[id^="nav-"]:hover {
    filter: brightness(1.25);
    border-color: rgba(255,255,255,0.15) !important;
    transform: translateX(3px);
}
[id^="nav-"] { transition: all 0.18s ease !important; }
</style></head>"""
)

app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col(html.Div("NSE  INDEX  LIVE  DASHBOARD", style={
            "fontSize": "0.95rem", "fontWeight": "800",
            "letterSpacing": "0.2em", "color": "#e2e8f0",
        }), width=8),
        dbc.Col(html.Div(id="status", style={
            "textAlign": "right", "fontSize": "0.72rem", "marginTop": "4px",
        }), width=4),
    ], className="mt-3 mb-3 align-items-center"),

    dbc.Row([
        # ── Left sidebar ───────────────────────────────────────────────────────
        dbc.Col([
            html.Div("INDICES", style={
                "fontSize": "0.58rem", "letterSpacing": "0.2em",
                "color": "#1e3a5f", "marginBottom": "10px",
            }),
            *[_nav_card(sym) for sym in INDEX_SYMBOLS],
            html.Div("👆 Click any index above",
                     style={"fontSize": "0.58rem", "color": "#334155",
                            "textAlign": "center", "marginTop": "16px",
                            "letterSpacing": "0.05em"}),
        ], md=2, style={
            "background": BG_SIDE, "padding": "16px 10px",
            "borderRight": "1px solid #111d2e",
            "minHeight": "calc(100vh - 70px)",
        }),

        # ── Main content ────────────────────────────────────────────────────────
        dbc.Col([
            # OVERVIEW PANEL
            html.Div(id="overview-panel", children=[
                dbc.Row([_overview_card(s) for s in INDEX_SYMBOLS], className="gx-0 mb-2"),
                dbc.Card(dbc.CardBody([
                    html.Div("INTRADAY  %  CHANGE  FROM  PREVIOUS  CLOSE", style={
                        "fontSize": "0.62rem", "letterSpacing": "0.12em",
                        "color": "#1e3a5f", "marginBottom": "4px",
                    }),
                    dcc.Graph(id="ov-chart",
                              config={"displayModeBar": False},
                              style={"height": "260px"}),
                ]), style={"background": BG_CARD, "border": "1px solid #111d2e",
                           "borderRadius": "10px", "marginBottom": "12px"}),
                # Daily EOD context from Daily_Cash_Market bridge
                html.Div(id="context-panel"),
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

                # Option chain table in a scrollable container
                dcc.Loading(
                    type="circle",
                    color="#00d4ff",
                    children=html.Div(id="oc-table",
                                      style={"overflowY": "auto",
                                             "maxHeight": "calc(100vh - 300px)"}),
                ),
            ]),
        ], md=10, style={"padding": "12px 16px"}),
    ], className="gx-0"),

    # State stores
    dcc.Store(id="sel-sym",    data=None),
    dcc.Store(id="sel-expiry", data=""),

    # Intervals
    dcc.Interval(id="fast-tick",   interval=1000,  n_intervals=0),
    dcc.Interval(id="oc-tick",    interval=2000,  n_intervals=0),
    dcc.Interval(id="signal-tick",interval=60000, n_intervals=0),
    dcc.Interval(id="setup-tick", interval=30000, n_intervals=0),

], fluid=True, style={"background": BG, "minHeight": "100vh", "padding": "0"})


# ── Daily context panel renderer ─────────────────────────────────────────────
def _render_context_panel() -> html.Div:
    """
    Compact 4-column card showing yesterday's EOD structural setup for all
    4 indices. Uses Daily_Cash_Market bridge (Layer 9 data).
    Renders a 'building' state when bridge is loading or DB is unavailable.
    """
    if not _CTX_BRIDGE_OK:
        return html.Div()

    bridge = _get_ctx_bridge()
    if not bridge.is_available():
        return dbc.Card(dbc.CardBody(
            html.Div(
                "Daily Context (Layer 9): Daily_Cash_Market DB not reachable "
                "— run daily ingestion after 6:30 PM",
                style={"color": "#334155", "fontSize": "0.62rem", **MONO},
            )
        ), style={"background": "#080f1c", "border": "1px solid #1a2535",
                  "borderRadius": "8px", "marginBottom": "10px"})

    cols = []
    for sym in INDEX_SYMBOLS:
        data   = bridge.get_panel_data(sym)
        color  = COLORS[sym]
        label  = LABELS[sym]
        common = data.get("__common__", {})

        direction  = data.get("direction", "?")
        confidence = data.get("confidence", "?")
        hmm        = data.get("hmm_state", "?")
        acc        = data.get("pred_acc_30d")
        fii_5d     = data.get("fii_5d_cr")
        vix        = common.get("india_vix")
        breadth    = common.get("breadth_pct")
        pred_date  = data.get("pred_date")
        rng_lo     = data.get("range_low")
        rng_hi     = data.get("range_high")
        target     = data.get("target_close")
        score, _   = bridge.score_index(sym)

        # Direction colour
        dir_clr = ("#22c55e" if direction == "UP"
                   else "#ef4444" if direction == "DOWN"
                   else "#f59e0b")
        score_clr = "#22c55e" if score > 0.5 else "#ef4444" if score < -0.5 else "#94a3b8"

        def _kv(k, v, vc="#475569"):
            return html.Div([
                html.Span(k + "  ", style={"color": "#334155", "fontSize": "0.55rem"}),
                html.Span(str(v),   style={"color": vc, "fontSize": "0.62rem", **MONO}),
            ], style={"marginBottom": "2px"})

        date_str = (pred_date.strftime("%d %b") if pred_date and
                    hasattr(pred_date, "strftime") else str(pred_date or "—"))

        col_children = [
            html.Div(label, style={"color": color, "fontSize": "0.58rem",
                                    "letterSpacing": "0.12em", "fontWeight": "700",
                                    "marginBottom": "6px"}),
            html.Div([
                html.Span(direction or "—",
                          style={"color": dir_clr, "fontWeight": "800",
                                 "fontSize": "0.85rem", **MONO}),
                html.Span(f"  [{confidence}]" if confidence else "",
                          style={"color": "#475569", "fontSize": "0.58rem"}),
            ], style={"marginBottom": "4px"}),
            _kv("REGIME",  hmm or "—",         "#94a3b8"),
            _kv("FII 5D",  f"{fii_5d:+,.0f}Cr" if fii_5d is not None else "—",
                "#22c55e" if (fii_5d or 0) > 0 else "#ef4444"),
            _kv("VIX",     f"{vix:.1f}%" if vix else "—",
                "#f59e0b" if vix and vix > 18 else "#94a3b8"),
            _kv("BREADTH", f"{breadth:.0f}%" if breadth else "—",
                "#22c55e" if (breadth or 0) > 60 else "#ef4444" if (breadth or 0) < 40 else "#94a3b8"),
        ]

        if rng_lo and rng_hi:
            col_children.append(_kv(
                "PRED RNG",
                f"{rng_lo:,.0f} – {rng_hi:,.0f}",
                "#fbbf24",
            ))

        col_children.append(html.Div([
            html.Span("L9 SCORE  ", style={"color": "#334155", "fontSize": "0.55rem"}),
            html.Span(f"{score:+.2f}", style={"color": score_clr, "fontWeight": "700",
                                               "fontSize": "0.65rem", **MONO}),
            html.Span(f"  ({date_str})", style={"color": "#1e2d40", "fontSize": "0.52rem"}),
        ], style={"marginTop": "4px", "borderTop": "1px solid #111d2e", "paddingTop": "4px"}))

        if acc is not None:
            col_children.append(html.Div(
                f"30D accuracy: {acc:.0f}%",
                style={"color": "#1e2d40", "fontSize": "0.52rem", **MONO},
            ))

        cols.append(dbc.Col(
            html.Div(col_children, style={
                "padding": "10px 12px",
                "background": "#0a1020",
                "borderRadius": "6px",
                "borderLeft": f"3px solid {color}",
                "border": f"1px solid {color}22",
            }),
            md=3, xs=6, className="mb-2 px-1",
        ))

    return dbc.Card(dbc.CardBody([
        dbc.Row([
            dbc.Col(html.Div("DAILY  CONTEXT  —  LAYER 9  (EOD  STRUCTURAL  SETUP)", style={
                "fontSize": "0.58rem", "letterSpacing": "0.14em",
                "color": "#1e3a5f", "fontWeight": "600",
            })),
            dbc.Col(html.Div(
                "Source: Daily_Cash_Market · 24-signal engine · refreshes post-market",
                style={"fontSize": "0.52rem", "color": "#1e2d40",
                       "textAlign": "right"},
            )),
        ], className="mb-2 align-items-center"),
        dbc.Row(cols, className="gx-2"),
    ]), style={
        "background": "#080f1c",
        "border":     "1px solid #111d2e",
        "borderTop":  "2px solid #1e3a5f",
        "borderRadius": "8px",
        "marginBottom": "12px",
    })


# ── Velocity monitor renderer ────────────────────────────────────────────────
def _render_velocity_panel(sym: str) -> html.Div:
    """
    Compact session-memory card: OI flow, wall shift, IV regime, PCR slope.
    Shows 'building' state when < 3 snapshots (first ~9 min of session).
    """
    vel = oi_store.velocity(sym)
    color = COLORS.get(sym, "#00d4ff")

    def _row(label: str, value: str, clr: str = "#475569"):
        return dbc.Row([
            dbc.Col(html.Span(label, style={"color": "#334155", "fontSize": "0.58rem",
                                             "letterSpacing": "0.08em"}), width=4),
            dbc.Col(html.Span(value, style={"color": clr, "fontWeight": "600",
                                             "fontSize": "0.68rem", **MONO}), width=8),
        ], className="mb-1")

    if not vel.get("has_data"):
        n = vel.get("snap_count", 0)
        body = html.Div(
            f"Building session history — {n}/3 snapshots collected...",
            style={"color": "#334155", "fontSize": "0.65rem", "padding": "4px 0", **MONO},
        )
    else:
        oi   = vel["oi"]
        iv   = vel["iv"]
        wal  = vel["walls"]
        pcr  = vel["pcr"]

        def _oi_str(v):
            if v is None or v == 0: return "—"
            # _fmt_oi already includes the "-" sign for negative values.
            # Prepend "+" only for positives so display is "+2.3M / -1.1M".
            return f"{'+' if v > 0 else ''}{_fmt_oi(v)}"

        def _oi_clr(v):
            if v is None: return "#475569"
            return "#22c55e" if v > 0 else "#ef4444" if v < 0 else "#475569"

        # OI flow
        c1 = oi.get("call_1hr"); p1 = oi.get("put_1hr")
        oi_clr = "#22c55e" if (p1 or 0) > (c1 or 0) else "#ef4444" if (c1 or 0) > (p1 or 0) else "#475569"
        oi_str = f"Call {_oi_str(c1)}  │  Put {_oi_str(p1)}"

        # wall shift
        cws = wal.get("call_shift_1hr"); pws = wal.get("put_shift_1hr")
        cn  = wal.get("call_now", 0);   pn  = wal.get("put_now", 0)
        cwa = wal.get("call_1hr_ago") or cn; pwa = wal.get("put_1hr_ago") or pn
        def _wall_str(old, now, shift):
            if shift is None: return f"{now:,.0f}"
            arrow = "↑" if shift > 0 else "↓" if shift < 0 else "→"
            return f"{old:,.0f} → {now:,.0f}  {arrow}{abs(shift):.0f}"
        def _wall_clr(shift):
            if shift is None: return "#475569"
            return "#22c55e" if shift > 0 else "#ef4444" if shift < 0 else "#475569"

        # IV regime
        regime = iv.get("regime", "stable")
        iv_now = iv.get("now", 0)
        iv_ch  = iv.get("change_1hr") or 0
        iv_clr = "#f59e0b" if regime == "expanding" else "#22c55e" if regime == "contracting" else "#475569"
        iv_str = f"{iv_now:.1f}%  {'+' if iv_ch>=0 else ''}{iv_ch:.1f}% (1hr)  [{regime.upper()}]"

        # PCR trend
        pcr_n  = pcr.get("now", 0); pcr_30 = pcr.get("30m_ago") or pcr_n
        pcr_ch = pcr.get("change_30m") or 0
        trend  = pcr.get("trend", "stable")
        pcr_clr = "#22c55e" if trend == "rising" else "#ef4444" if trend == "falling" else "#475569"
        pcr_str = f"{pcr_30:.2f} → {pcr_n:.2f}  ({'+' if pcr_ch>=0 else ''}{pcr_ch:.2f}/30m)  {trend.upper()}"

        snaps = vel.get("snap_count", 0)
        body = html.Div([
            _row("OI FLOW  1hr",   oi_str,                    oi_clr),
            _row("CALL WALL  1hr", _wall_str(cwa, cn, cws),   _wall_clr(cws)),
            _row("PUT WALL  1hr",  _wall_str(pwa, pn, pws),   _wall_clr(pws)),
            _row("IV REGIME",      iv_str,                    iv_clr),
            _row("PCR TREND  30m", pcr_str,                   pcr_clr),
            html.Div(f"{snaps} snapshots in session history",
                     style={"color": "#1e2d40", "fontSize": "0.55rem",
                            "marginTop": "6px", **MONO}),
        ])

    return dbc.Card(dbc.CardBody([
        html.Div("INTRADAY  VELOCITY  MONITOR", style={
            "fontSize": "0.58rem", "letterSpacing": "0.18em",
            "color": "#1e3a5f", "marginBottom": "8px",
        }),
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
    MONO_XS = {**MONO, "fontSize": "0.65rem"}

    def _sig_cell(t: dict) -> html.Td:
        sig = t.get("signal", "—")
        clr = t.get("color", "#475569")
        con = t.get("confidence", 0)
        return html.Td([
            html.Div(sig,  style={"color": clr, "fontWeight": "700", **MONO_XS}),
            html.Div(f"{con:.0f}%", style={"color": "#334155", "fontSize": "0.58rem", **MONO}),
        ], style={"padding": "5px 8px", "textAlign": "center"})

    TH = {"padding": "5px 8px", "fontSize": "0.58rem", "letterSpacing": "0.1em",
          "color": "#1e3a5f", "fontWeight": "600", "background": "#060c14",
          "textAlign": "center", "whiteSpace": "nowrap"}

    header = html.Tr([
        html.Th("INDEX",   style={**TH, "textAlign": "left"}),
        html.Th("5 MIN",   style=TH),
        html.Th("15 MIN",  style=TH),
        html.Th("1 HOUR",  style=TH),
        html.Th("DAILY",   style=TH),
        html.Th("OVERALL", style={**TH, "color": "#475569"}),
    ])

    rows = [header]
    recs = []  # collect recommendations

    for sym in INDEX_SYMBOLS:
        r = results.get(sym, {})
        if not r or "timeframes" not in r:
            continue
        tfs   = r["timeframes"]
        ov, ov_clr = r.get("overall", ("—", "#475569"))
        ws    = r.get("weighted_score", 0)
        label = r.get("label", sym)
        color = COLORS[sym]

        rows.append(html.Tr([
            html.Td(html.Span(label, style={"color": color, "fontWeight": "700",
                                             "fontSize": "0.65rem", **MONO}),
                    style={"padding": "5px 10px"}),
            _sig_cell(tfs.get("5min",  {})),
            _sig_cell(tfs.get("15min", {})),
            _sig_cell(tfs.get("60min", {})),
            _sig_cell(tfs.get("daily", {})),
            html.Td([
                html.Div(ov, style={"color": ov_clr, "fontWeight": "800",
                                     "fontSize": "0.68rem", **MONO}),
                html.Div(f"score {ws:+.1f}", style={"color": "#334155",
                                                      "fontSize": "0.58rem", **MONO}),
            ], style={"padding": "5px 8px", "textAlign": "center"}),
        ]))

        # Collect for recommendations if signal is meaningful
        if abs(ws) >= 0.8:
            recs.append((abs(ws), sym, r))

    grid = html.Table(rows, style={"width": "100%", "borderCollapse": "collapse"})

    # Top recommendations (sorted by conviction)
    recs.sort(key=lambda x: x[0], reverse=True)
    rec_cards = []
    for _, sym, r in recs[:3]:
        ws  = r["weighted_score"]
        ov, ov_clr = r.get("overall", ("—", "#475569"))
        color = COLORS[sym]
        label = r["label"]
        tfs   = r["timeframes"]

        # Best timeframe details
        bull_tfs = [k for k, t in tfs.items() if t.get("score", 0) > 0.5]
        bear_tfs = [k for k, t in tfs.items() if t.get("score", 0) < -0.5]
        active_tfs = bull_tfs if ws > 0 else bear_tfs
        tf_labels  = [tfs[k]["label"] for k in active_tfs]

        direction = "CE (CALL)" if ws > 0 else "PE (PUT)"
        dir_clr   = "#4ade80" if ws > 0 else "#f87171"

        # Top reasons from most aligned timeframe
        best_tf = max(tfs.items(), key=lambda x: abs(x[1].get("score", 0)))[1]
        top_rsns = best_tf.get("reasons", [])[:3]

        trade_hint = "INTRADAY" if len(active_tfs) <= 2 else "BTST / POSITIONAL"

        rec_cards.append(dbc.Col(
            dbc.Card(dbc.CardBody([
                html.Div([
                    html.Span(label, style={"color": color, "fontWeight": "800",
                                            "fontSize": "0.72rem", **MONO}),
                    html.Span(f"  {ov}", style={"color": ov_clr, "fontSize": "0.65rem", **MONO}),
                ], style={"marginBottom": "4px"}),
                html.Div([
                    html.Span("TRADE: BUY ", style={"color": "#94a3b8", "fontSize": "0.62rem"}),
                    html.Span(direction, style={"color": dir_clr, "fontWeight": "700",
                                                "fontSize": "0.72rem", **MONO}),
                ], style={"marginBottom": "4px"}),
                html.Div(f"Type: {trade_hint}  |  TFs aligned: {' + '.join(tf_labels)}",
                         style={"color": "#334155", "fontSize": "0.6rem", "marginBottom": "6px"}),
                html.Div([
                    html.Div([html.Span(icon + "  ", style={"fontSize": "0.65rem"}),
                              html.Span(txt, style={"color": "#4a5568"})],
                             style={"fontSize": "0.62rem", **MONO, "marginBottom": "2px"})
                    for icon, txt in [("🟢" if b == "bull" else "🔴" if b == "bear" else "⚪", t)
                                      for b, t in top_rsns]
                ]),
                html.Div(f"Confidence: {min(abs(ws)/5*100, 95):.0f}%  |  "
                         f"Click {label} in left pane for exact strike",
                         style={"color": "#1e3a5f", "fontSize": "0.6rem",
                                "marginTop": "6px", "borderTop": "1px solid #111d2e",
                                "paddingTop": "4px"}),
            ]), style={
                "background": "#080f1c",
                "border":    f"1px solid {color}33",
                "borderTop": f"2px solid {color}",
                "borderRadius": "8px",
            }),
            md=4, className="mb-2 px-1",
        ))

    return html.Div([
        # Section header
        dbc.Row([
            dbc.Col(html.Div("TRADE SIGNALS  —  MULTI-TIMEFRAME ANALYSIS", style={
                "fontSize": "0.6rem", "letterSpacing": "0.15em",
                "color": "#1e3a5f", "fontWeight": "600",
            })),
            dbc.Col(html.Div(f"Updated {updated}", style={
                "fontSize": "0.58rem", "color": "#1e2d40",
                "textAlign": "right",
            })),
        ], className="mb-2 align-items-center"),

        # Signal grid
        dbc.Card(dbc.CardBody(grid),
                 style={"background": "#0a1020", "border": "1px solid #111d2e",
                        "borderRadius": "8px", "marginBottom": "10px"}),

        # Disclaimer
        html.Div("* Algorithmic signals only. Not financial advice. Always use stop-loss.",
                 style={"fontSize": "0.55rem", "color": "#1e2d40",
                        "textAlign": "center", "marginBottom": "8px"}),

        # Recommendation cards
        dbc.Row(rec_cards, className="gx-2") if rec_cards else html.Div(),
    ])


# ── Trade recommendation card renderer ────────────────────────────────────────
def _render_trade_rec(rec: dict, sym: str) -> html.Div:
    if not rec:
        return html.Div()

    color = COLORS.get(sym, "#00d4ff")
    MONO_S = {**MONO, "fontSize": "0.7rem"}

    if rec.get("neutral"):
        return dbc.Card(dbc.CardBody([
            html.Div(rec["tf_label"], style={"color":"#334155","fontSize":"0.6rem","marginBottom":"4px"}),
            html.Div(rec["signal"], style={"color":rec["color"],"fontWeight":"700",**MONO_S}),
            html.Div("No trade recommended — signal too weak or mixed.",
                     style={"color":"#475569","fontSize":"0.65rem","marginTop":"4px"}),
        ]), style={"background":"#080f1c","border":f"1px solid {color}22",
                   "borderLeft":f"3px solid {color}","borderRadius":"8px","marginBottom":"10px"})

    dir_clr  = rec["dir_clr"]
    warn_clr = "#f59e0b" if "CAUTION" in rec.get("warning","") else "#22c55e"

    def _row(label, value, val_clr="#e2e8f0"):
        return dbc.Row([
            dbc.Col(html.Span(label, style={"color":"#334155","fontSize":"0.6rem"}), width=4),
            dbc.Col(html.Span(value, style={"color":val_clr,"fontWeight":"600",**MONO_S}), width=8),
        ], className="mb-1")

    # Slot label for the option
    slot_sym = f"NSE:{LABELS.get(sym,'?').replace(' ','')}{rec['exp_date'][:6].replace('-','').replace(' ','')}{rec['strike']}{rec['direction']}"

    return dbc.Card(dbc.CardBody([
        # Header
        dbc.Row([
            dbc.Col([
                html.Div(rec["tf_label"], style={"color":"#334155","fontSize":"0.58rem","letterSpacing":"0.1em"}),
                html.Div([
                    html.Span(rec["signal"]+"  ", style={"color":rec["color"],"fontWeight":"800","fontSize":"0.9rem",**MONO}),
                    html.Span(f"Confidence {rec['confidence']:.0f}%  ({rec['conviction']})",
                              style={"color":"#475569","fontSize":"0.65rem"}),
                ]),
            ], md=5),
            dbc.Col([
                html.Div([
                    html.Span("RECOMMENDATION:  ", style={"color":"#334155","fontSize":"0.62rem"}),
                    html.Span(f"BUY {rec['dir_label']}", style={"color":dir_clr,"fontWeight":"800",
                              "fontSize":"0.85rem",**MONO}),
                ]),
                html.Div(f"Trade type: {rec['trade_type']}",
                         style={"color":"#334155","fontSize":"0.6rem"}),
            ], md=7),
        ], className="mb-2"),

        html.Hr(style={"borderColor":"#111d2e","margin":"6px 0"}),

        dbc.Row([
            # Left: trade details
            dbc.Col([
                html.Div("TRADE DETAILS", style={"color":"#1e3a5f","fontSize":"0.58rem",
                         "letterSpacing":"0.12em","marginBottom":"6px"}),
                _row("Option",  f"{rec['strike']:,.0f} {rec['direction']}  ·  Expiry {rec['exp_date']}", dir_clr),
                _row("Entry",   f"₹ {rec['entry_lo']} – {rec['entry_hi']}", "#e2e8f0"),
                _row("Stop Loss",f"₹ {rec['sl']}  ({int(TF_PROFILES[list(TF_PROFILES)[0]['sl_pct']*100 if False else 0] if False else rec['sl']/(rec['ltp'] or 1)*100-100):.0f}%)" if False
                                  else f"₹ {rec['sl']}  (Index SL ≈ {rec['spot_sl']:,.0f})", "#ef4444"),
                _row("Target 1", f"₹ {rec['t1']}  (+{int((rec['t1']/rec['ltp']-1)*100)}%)"
                                  f"  →  ₹{rec['profit_t1']:,.0f}/lot", "#4ade80"),
                _row("Target 2", f"₹ {rec['t2']}  (+{int((rec['t2']/rec['ltp']-1)*100)}%)"
                                  f"  →  ₹{rec['profit_t2']:,.0f}/lot", "#22c55e"),
                _row("R : R",    f"1 : {rec['rr']}", "#fbbf24"),
                _row("Max Loss",  f"₹ {rec['loss_lot']:,.0f} / lot  ({rec['lot_size']} shares)", "#f87171"),
            ], md=5),

            # Middle: Greeks + OI
            dbc.Col([
                html.Div("OPTION DETAILS", style={"color":"#1e3a5f","fontSize":"0.58rem",
                         "letterSpacing":"0.12em","marginBottom":"6px"}),
                _row("LTP",    f"₹ {rec['ltp']:.2f}"),
                _row("IV",     f"{rec['iv']:.1f}%"),
                _row("Delta",  f"{rec['delta']:.3f}"),
                _row("Theta",  f"{rec['theta']:.2f}  (daily decay)"),
                _row("Vega",   f"{rec['vega']:.2f}"),
                _row("OI",     _fmt_oi(rec["oi"])),
                _row("Volume", _fmt_oi(rec["volume"])),
                html.Div(rec["iv_context"], style={"color":"#475569","fontSize":"0.6rem","marginTop":"4px"}),
            ], md=3),

            # Right: why this trade
            dbc.Col([
                html.Div("WHY THIS TRADE?", style={"color":"#1e3a5f","fontSize":"0.58rem",
                         "letterSpacing":"0.12em","marginBottom":"6px"}),
                html.Div([
                    html.Div([
                        html.Span(("✓ " if b == "bull" and rec.get("direction")=="CE"
                                   else "✓ " if b == "bear" and rec.get("direction")=="PE"
                                   else "✗ " if b == "bear" else "– "),
                                  style={"color":"#4ade80" if b in ("bull","neut") else "#f87171"}),
                        html.Span(t, style={"color":"#475569","fontSize":"0.62rem"}),
                    ], style={"marginBottom":"3px"})
                    for b, t in rec["tech_reasons"][:4]
                ]),
                html.Div([
                    html.Div([
                        html.Span("▸ ", style={"color":"#334155"}),
                        html.Span(t, style={"color":"#334155","fontSize":"0.6rem"}),
                    ], style={"marginBottom":"3px"})
                    for b, t in rec["opt_signals"]
                ]),
                html.Div(rec.get("fut_context",""), style={"color":"#334155","fontSize":"0.6rem","marginTop":"4px"}),
            ], md=4),
        ]),

        # Warning bar
        html.Div(f"⚠ {rec['warning']}", style={
            "color": warn_clr, "fontSize": "0.62rem",
            "background": "#0a1525", "padding": "6px 10px",
            "borderRadius": "4px", "marginTop": "8px",
            **MONO,
        }) if rec.get("warning") else html.Div(),

    ]), style={
        "background": "#080f1c",
        "border":     f"1px solid {color}33",
        "borderTop":  f"3px solid {dir_clr}",
        "borderRadius": "8px",
        "marginBottom": "10px",
    })


# ── Callback 1: sidebar nav clicks → update selected symbol ────────────────────
@app.callback(
    Output("sel-sym",    "data"),
    Output("sel-expiry", "data"),
    [Input(f"nav-{_slug(s)}", "n_clicks") for s in INDEX_SYMBOLS],
    State("sel-sym", "data"),
    prevent_initial_call=True,
)
def on_nav_click(*args):
    *_, current = args
    from dash import callback_context as ctx
    if not ctx.triggered:
        return current, ""
    tid = ctx.triggered[0]["prop_id"].split(".")[0]
    for sym in INDEX_SYMBOLS:
        if tid == f"nav-{_slug(sym)}":
            return (None, "") if current == sym else (sym, "")
    return current, ""


# ── Callback 2: toggle panels + highlight selected nav card ───────────────────
@app.callback(
    Output("overview-panel", "style"),
    Output("oc-panel",       "style"),
    Output("oc-title",       "children"),
    *[Output(f"nav-{_slug(s)}", "style") for s in INDEX_SYMBOLS],
    Input("sel-sym", "data"),
)
def toggle_view(sym):
    ov_style = {"display": "none"} if sym else {"display": "block"}
    oc_style = {"display": "block"} if sym else {"display": "none"}

    if sym:
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

    nav_styles = []
    for s in INDEX_SYMBOLS:
        c = COLORS[s]
        selected = (s == sym)
        nav_styles.append({
            "padding": "12px 14px", "marginBottom": "8px", "borderRadius": "8px",
            "border": f"1px solid {c}{'88' if selected else '33'}",
            "borderLeft": f"3px solid {c}",
            "background": f"{c}18" if selected else BG_CARD,
            "cursor": "pointer", "transition": "background 0.15s",
        })
    return (ov_style, oc_style, title, *nav_styles)


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
        ltp_s.append({**MONO,"fontSize":"2rem","fontWeight":"bold","color":clr,"lineHeight":"1"})
        chgs.append(f"{'▲' if up else '▼'}  {s}{ch:,.2f}   {s}{chp:.2f}%")
        chg_s.append({**MONO,"fontSize":"0.88rem","marginTop":"5px","color":clr})
        ohlpcs.append(html.Div([html.Div(f"O   {o:>11,.2f}"),html.Div(f"H   {h:>11,.2f}"),
                                html.Div(f"L   {l:>11,.2f}"),html.Div(f"PC  {pc:>11,.2f}")]))
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
@app.callback(
    Output("trade-rec",      "children"),
    Output("velocity-panel", "children"),
    Input("tf-dd",           "value"),
    Input("setup-tick",      "n_intervals"),
    Input("expiry-dd",       "value"),
    State("sel-sym",         "data"),
    prevent_initial_call=True,
)
def update_trade_rec(tf_key, _, expiry, sym):
    if not sym or not tf_key:
        return html.Div(), html.Div()

    with _lock:
        spot = (_latest.get(sym) or {}).get("ltp", 0)

    # Fetch option chain with selected expiry
    oc_data = fetch_option_chain(sym, expiry or "")
    if oc_data.get("s") != "ok":
        return (html.Div("Option chain unavailable",
                         style={"color": "#475569", "fontSize": "0.65rem", **MONO}),
                html.Div())

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

    return _render_trade_rec(rec, sym), _render_velocity_panel(sym)


# ── Callback: daily context panel (60-second refresh, same interval) ──────────
@app.callback(
    Output("context-panel", "children"),
    Input("signal-tick",    "n_intervals"),
    State("sel-sym",        "data"),
)
def update_context_panel(_, sel):
    if sel:   # overview hidden while OC panel is active
        return no_update
    return _render_context_panel()


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

    print(f"  Open  →  http://127.0.0.1:8050")
    print(SEP)

    app.run(debug=False, host="127.0.0.1", port=8050)
