"""Fyers REST data fetch — option chain, futures, quotes — behind one boundary.

Owns the rate/quota handling that used to live as module globals inside
dashboard.py. The option-chain endpoint is quota-limited ("request limit
reached"); the poller AND every UI callback called it independently, so a
single-slot cache collapsed to ~zero hits across 4 rotating indices → each
caller hit Fyers fresh → burst → limit tripped mid-morning. A keyed TTL cache
lets all callers share one fetch per (sym, expiry) per window; errors are cached
briefly so a hit limit backs off instead of being re-hammered.

Pure data adapter: imports only core + token + stdlib/requests. No Dash, no DB,
no dashboard — returns the raw normalized shapes the callers already expect.
"""
from __future__ import annotations

import datetime
import threading
import time

import requests

from core.constants import IST

from .token import auth_header

OC_URL     = "https://api-t1.fyers.in/data/options-chain-v3"
QUOTES_URL = "https://api-t1.fyers.in/data/quotes"


# ── Option chain + cache ───────────────────────────────────────────────────────
_oc_lock    = threading.Lock()
_oc_cache: dict = {}          # (sym, expiry) -> {"data":.., "ts":.., "ttl":..}
_OC_TTL     = 25.0            # seconds to reuse a good chain (matches 30s poll)
_OC_ERR_TTL = 8.0            # back-off after a failed/limited fetch


def expiry_to_epoch(expiry_val: str) -> str:
    """expiry_val is already the epoch string returned by Fyers expiryData."""
    return expiry_val or ""


def fetch_option_chain(sym: str, expiry: str = "", n_strikes: int = 15) -> dict:
    key = (sym, expiry)
    now = time.time()
    with _oc_lock:
        e = _oc_cache.get(key)
        if e and now - e["ts"] < e["ttl"]:
            return e["data"]
    try:
        resp = requests.get(
            OC_URL,
            headers={
                "Authorization": auth_header(),
                "Content-Type":  "application/json",
                "version":       "3",
            },
            params={
                "symbol":      sym,
                "strikecount": n_strikes,
                "timestamp":   expiry_to_epoch(expiry) if expiry else "",
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
    ttl = _OC_TTL if data.get("s") == "ok" else _OC_ERR_TTL
    with _oc_lock:
        _oc_cache[key] = {"data": data, "ts": time.time(), "ttl": ttl}
    return data


# ── Futures ────────────────────────────────────────────────────────────────────
_MONTH_ABB = {1:"JAN",2:"FEB",3:"MAR",4:"APR",5:"MAY",6:"JUN",
              7:"JUL",8:"AUG",9:"SEP",10:"OCT",11:"NOV",12:"DEC"}
_FUT_UNDERLYING = {
    "NSE:NIFTY50-INDEX":    "NIFTY",
    "NSE:NIFTYBANK-INDEX":  "BANKNIFTY",
    "NSE:FINNIFTY-INDEX":   "FINNIFTY",
    "NSE:MIDCPNIFTY-INDEX": "MIDCPNIFTY",
}
_FUT_LABELS = ["NEAR", "NEXT", "FAR "]

_fut_lock  = threading.Lock()
_fut_cache: dict = {}


def futures_symbols(index_sym: str) -> list[dict]:
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


def fetch_futures(index_sym: str, on_fresh=None) -> list[dict]:
    """Fetch near/next/far futures quotes (no DB write of its own). Cached 2s.

    on_fresh(result): optional callback fired only on a real network fetch (not on
    a cache hit) — lets a caller persist the strip without the adapter knowing storage.
    """
    with _fut_lock:
        c = _fut_cache.get(index_sym)
        if c and time.time() - c["ts"] < 2:
            return c["data"]

    sym_info = futures_symbols(index_sym)
    if not sym_info:
        return []
    try:
        resp = requests.get(
            QUOTES_URL,
            headers={"Authorization": auth_header(),
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
    if on_fresh is not None:
        on_fresh(result)
    return result


def fetch_quotes(symbols: list[str]) -> dict:
    """Fetch last price (lp) for a list of Fyers symbols via /data/quotes."""
    if not symbols:
        return {}
    out: dict = {}
    try:
        resp = requests.get(
            QUOTES_URL,
            headers={"Authorization": auth_header(), "version": "3"},
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
