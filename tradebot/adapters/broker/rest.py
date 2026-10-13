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

import collections
import datetime
import json
import os
import threading
import time

import requests

from core.constants import DATA_DIR, IST, NIFTY

from .token import auth_header

OC_URL     = "https://api-t1.fyers.in/data/options-chain-v3"
QUOTES_URL = "https://api-t1.fyers.in/data/quotes"


# ── Data-API budget gate ───────────────────────────────────────────────────────
# From 2026-10-09 Fyers caps the DATA APIs (quotes, option chain, depth) on the
# Standard plan at 5/sec, 50/min, 5,000/day — down from 10 / 200 / 100,000. One
# dashboard process was measured at ~4,240 calls/day before open-trade quotes and
# the browser, and tripped "request limit reached" on a restart burst (2026-10-13
# 09:37). So every Data call now passes one in-process gate that (a) smooths the
# per-second burst, (b) refuses rather than sends once the minute or day budget
# is spent, and (c) keeps the count, so a breach is a logged number instead of a
# silent hole in the capture. The count survives a restart (state file) — the
# quota is per day, not per process.
#
# Soft limits sit just inside Fyers' own so our clock and theirs can disagree a
# little. A Prime account overrides via env.
DATA_PER_SEC = max(1, int(os.environ.get("FYERS_DATA_PER_SEC", "5")))
DATA_PER_MIN = max(1, int(os.environ.get("FYERS_DATA_PER_MIN", "50")))
DATA_PER_DAY = max(1, int(os.environ.get("FYERS_DATA_PER_DAY", "5000")))
_SEC_SOFT = max(1, DATA_PER_SEC - 1)
_MIN_SOFT = max(1, DATA_PER_MIN - 5)
_DAY_SOFT = max(1, DATA_PER_DAY - 100)
_GATE_WAIT_SEC = 2.0          # longest a caller waits for a per-second slot

# What the budget is FOR. If it runs short, the feeds below are the last to go: the
# NIFTY chain (the only weekly, the BTST instrument), the futures strip (the CAS-tail
# fill price) and open-trade quotes. Everything else — the three monthly-expiry
# indices, far expiries opened in the panel — is refused once what is left only
# covers these to the close. Without this the day's last hour, the part that matters
# most, was simply whatever happened to be left.
_ESSENTIAL_PER_MIN = 6        # NIFTY chain 2 + futures batch 1 + open-trade quotes 3
_ESSENTIAL_TAIL    = 60       # 15:31-15:45: CAS-tail futures + quotes to the EOD mark
_CAPTURE_FROM = datetime.time(9, 14)      # the chain poller's window (dashboard)
_CAPTURE_TO   = datetime.time(15, 31)
_LIVE_FROM    = datetime.time(9, 0)
_LIVE_TO      = datetime.time(15, 45)
_IDLE_TTL     = 900.0         # market shut: the chain is not moving, do not re-buy it

# A limit rejection from Fyers means their count disagrees with ours (another machine,
# another script, their clock). Their older rule blocks the account for the rest of the
# day after repeated per-minute breaches, so a rejection silences EVERY Data call for a
# while — losing a minute of capture is cheap next to losing the session.
_COOL_STEPS = (60, 120, 300, 600)
_COOL_FORGET_SEC = 1800

_STATE_FILE = DATA_DIR / "fyers_data_calls.json"
_gate_lock  = threading.Lock()
_io_lock    = threading.Lock()
_recent: collections.deque = collections.deque()   # send times (monotonic), last 60s
_day: dict = {}               # today's counters — see _fresh_day
_said: dict = {}              # log throttles: key -> last wall time / milestone
_cool = {"until": 0.0, "level": 0, "last": 0.0}    # monotonic


def _market_live(now: datetime.datetime | None = None) -> bool:
    now = now or datetime.datetime.now(tz=IST)
    return now.weekday() < 5 and _LIVE_FROM <= now.time() <= _LIVE_TO


def _reserve(now: datetime.datetime | None = None) -> int:
    """Data calls the essential feeds still need between now and the end of today."""
    now = now or datetime.datetime.now(tz=IST)
    t = now.time()
    if now.weekday() >= 5 or t > _LIVE_TO:
        return 0
    start = max(t, _CAPTURE_FROM)
    mins = max(0, (_CAPTURE_TO.hour * 60 + _CAPTURE_TO.minute) - (start.hour * 60 + start.minute))
    return mins * _ESSENTIAL_PER_MIN + _ESSENTIAL_TAIL


def _fresh_day(date: str) -> dict:
    return {"date": date, "used": 0, "by_kind": {}, "denied": 0, "rejected": 0, "history": 0}


def _roll() -> None:
    """Start today's counters (caller holds _gate_lock). Reload the state file on the
    first call of a process so a mid-session restart forgets neither the morning's
    count nor the last minute's sends — a restart burst is what tripped the limit."""
    today = datetime.datetime.now(tz=IST).date().isoformat()
    if _day.get("date") == today:
        return
    fresh = _fresh_day(today)
    if not _day:
        try:
            saved = json.loads(_STATE_FILE.read_text(encoding="utf-8"))
            if saved.get("date") == today:
                fresh.update({k: saved[k] for k in fresh if k in saved})
                wall, mono = time.time(), time.monotonic()
                for w in saved.get("recent", []):
                    if 0 <= wall - w < 60:
                        _recent.append(mono - (wall - w))
        except Exception:
            pass
    _day.clear()
    _day.update(fresh)
    _said.clear()


def _snapshot(every: float = 5.0) -> str | None:
    """The state file's next contents, if one is due (caller holds _gate_lock)."""
    wall = time.time()
    if wall - _said.get("saved", 0) < every:
        return None
    _said["saved"] = wall
    mono = time.monotonic()
    return json.dumps({**_day, "recent": [round(wall - (mono - t), 2) for t in _recent]})


def _write(payload: str | None) -> None:
    """Write the state file OUTSIDE the gate lock — a stalled disk (the VM swaps) must
    not hold every Data call behind it. A write already in flight is simply skipped."""
    if not payload or not _io_lock.acquire(blocking=False):
        return
    try:
        tmp = _STATE_FILE.with_suffix(".json.tmp")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, _STATE_FILE)
    except Exception:
        pass
    finally:
        _io_lock.release()


def _say(key: str, msg: str, every: float = 60.0) -> None:
    now = time.time()
    if now - _said.get(key, 0) >= every:
        _said[key] = now
        print(f"  [fyers] {msg}", flush=True)


def _acquire(kind: str, essential: bool = True) -> bool:
    """Take one Data-API slot. False = do not send.

    Refused when the day or minute budget is spent, during a cool-down after Fyers
    rejected a call, or — for a non-essential call — when what is left today only
    covers the essential feeds to the close. A full second just waits (≤2s)."""
    deadline = time.monotonic() + _GATE_WAIT_SEC
    while True:
        with _gate_lock:
            now = time.monotonic()
            _roll()
            while _recent and now - _recent[0] > 60:
                _recent.popleft()
            slot = sum(1 for t in _recent if now - t < 1.0) < _SEC_SOFT
            left = _DAY_SOFT - _day["used"]
            why = ("day" if left <= 0 else
                   "cool-down" if now < _cool["until"] else
                   "reserve" if (not essential and left <= _reserve()) else
                   "minute" if len(_recent) >= _MIN_SOFT else
                   "second" if (not slot and now > deadline) else "")
            took = None
            if why:
                took = False
                _day["denied"] += 1
                _say(f"deny-{why}",
                     f"Data call ({kind}) NOT sent — {why}: "
                     f"{_day['used']}/{DATA_PER_DAY} today, {len(_recent)}/{DATA_PER_MIN} "
                     f"in the last minute, {_day['denied']} refused today"
                     + (" — the rest is held for NIFTY chain, futures and open-trade quotes"
                        if why == "reserve" else ""))
            elif slot:
                took = True
                _recent.append(now)
                _day["used"] += 1
                _day["by_kind"][kind] = _day["by_kind"].get(kind, 0) + 1
                pct = 100 * _day["used"] // DATA_PER_DAY
                for mark in (90, 80, 50):
                    if pct >= mark:
                        if _said.get("mark", 0) < mark:
                            _said["mark"] = mark
                            print(f"  [fyers] Data-API budget {mark}% used @ "
                                  f"{datetime.datetime.now(tz=IST):%H:%M:%S} — "
                                  f"{_day['used']}/{DATA_PER_DAY} {_day['by_kind']}", flush=True)
                        break
            payload = _snapshot() if took is not None else None
        if took is not None:
            _write(payload)
            return took
        time.sleep(0.1)


def _note_reject(kind: str, message, status: int | None = None) -> None:
    """Count a Fyers-side limit rejection and start a cool-down. The futures/quotes
    paths used to swallow these; a 429 with no JSON body counts too."""
    if status != 429 and "limit" not in str(message).lower():
        return
    with _gate_lock:
        _roll()
        now = time.monotonic()
        _day["rejected"] += 1
        n, used = _day["rejected"], _day["used"]
        if now >= _cool["until"]:              # calls already in flight do not escalate
            if now - _cool["last"] > _COOL_FORGET_SEC:
                _cool["level"] = 0
            secs = _COOL_STEPS[min(_cool["level"], len(_COOL_STEPS) - 1)]
            _cool.update(until=now + secs, level=_cool["level"] + 1, last=now)
        wait = max(0, _cool["until"] - now)
    _say("reject", f"Fyers REJECTED a {kind} call — {str(message)[:80]} (our count "
                   f"{used}/{DATA_PER_DAY}, {n} rejections today) — all Data calls "
                   f"paused {wait:.0f}s", every=0)


def _error_ttl(message, base: float) -> float:
    """How long to stop retrying after a failed fetch. A dead token cannot heal on its
    own, so it is not worth a call every 8s from every open panel until it is renewed."""
    m = str(message).lower()
    return 60.0 if ("token" in m or "authenticate" in m) else base


def note_history_call() -> None:
    """Count a /data/history call. Counted apart from the Data budget and never gated:
    Fyers' article does not say which bucket history falls in, so this is the number
    that will tell us if a block ever arrives with the Data count still low."""
    with _gate_lock:
        _roll()
        _day["history"] += 1


def data_calls_remaining() -> int:
    with _gate_lock:
        _roll()
        return _DAY_SOFT - _day["used"]


def data_call_stats() -> dict:
    with _gate_lock:
        _roll()
        now = time.monotonic()
        return {**_day, "by_kind": dict(_day["by_kind"]), "cap": DATA_PER_DAY,
                "last_minute": sum(1 for t in _recent if now - t <= 60)}


# ── Option chain + cache ───────────────────────────────────────────────────────
_oc_lock    = threading.Lock()
_oc_cache: dict = {}          # (sym, expiry) -> {"data":.., "ts":.., "ttl":.., "n":..}
_oc_nearest: dict = {}        # sym -> epoch of the nearest expiry (what expiry="" returns)
_oc_flight: dict = {}         # (sym, expiry) -> Lock held by the one caller fetching it
_OC_TTL     = 25.0            # seconds to reuse a good chain (matches 30s poll)
_OC_ERR_TTL = 8.0            # back-off after a failed/limited fetch
_OC_MIN_STRIKES = 25          # ± strikes; the capture poller's width (dashboard CHAIN_STRIKES)


def expiry_to_epoch(expiry_val: str) -> str:
    """expiry_val is already the epoch string returned by Fyers expiryData."""
    return expiry_val or ""


def fetch_option_chain(sym: str, expiry: str = "", n_strikes: int = 15,
                       max_age: float | None = None) -> dict:
    """One chain per (sym, expiry) per window, shared by the poller and every callback.

    max_age: a caller that can live with an older good chain (the 60s signal eval)
    passes it to ride on the capture poller's fetch instead of spending its own call.

    Two cache rules that each used to cost calls or data:
      * the nearest expiry asked for BY EPOCH (the option-chain panel's default) is the
        same chain as expiry="" — it shares that entry instead of fetching it twice;
      * every fetch asks for at least the capture width, and an entry only satisfies a
        request for the same or fewer strikes. The key has no strike count, so a ±15
        fetch landing just before the ±25 capture poll was being handed to it — 162 of
        2,833 snapshots on 2026-10-09 were persisted narrow. Strike count is free on the
        quota (one call either way), so a wide fetch is one the poller can reuse.
    """
    want = n_strikes
    n_strikes = max(n_strikes, _OC_MIN_STRIKES)
    if expiry and _oc_nearest.get(sym) == expiry:
        expiry = ""
    key = (sym, expiry)

    def cached():
        with _oc_lock:
            e = _oc_cache.get(key)
            if not e:
                return None
            good = e["data"].get("s") == "ok"
            ttl = e["ttl"]
            if good:
                ttl = max(_OC_TTL if _market_live() else _IDLE_TTL, max_age or 0)
            if time.time() - e["ts"] < ttl and (not good or e["n"] >= n_strikes):
                return e["data"]
            return None

    hit = cached()
    if hit is not None:
        return _trim_chain(hit, want)
    with _oc_lock:
        flight = _oc_flight.setdefault(key, threading.Lock())
    with flight:                  # one fetch per key; the callers queued behind share it
        hit = cached()
        if hit is not None:
            return _trim_chain(hit, want)
        if not _acquire("chain", essential=(sym == NIFTY and not expiry)):
            data = {"s": "error", "message": "local rate gate — Fyers Data-API budget "
                                             "spent or held for the essential feeds"}
        else:
            status = None
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
                status = resp.status_code
                try:
                    data = resp.json()
                except Exception:
                    data = {"s": "error", "message": f"HTTP {resp.status_code}: {resp.text[:300]}"}
            except Exception as e:
                data = {"s": "error", "message": str(e)}
            if not isinstance(data, dict):
                data = {"s": "error", "message": f"unexpected response: {str(data)[:120]}"}
            if data.get("s") != "ok":
                _note_reject("chain", data.get("message"), status)
        ok = data.get("s") == "ok"
        if ok and not expiry:
            try:
                _oc_nearest[sym] = str(data["data"]["expiryData"][0]["expiry"])
            except Exception:
                pass
        with _oc_lock:
            _oc_cache[key] = {
                "data": data, "ts": time.time(), "n": n_strikes,
                "ttl": _OC_TTL if ok else _error_ttl(data.get("message"), _OC_ERR_TTL)}
        return _trim_chain(data, want)


def _trim_chain(data: dict, n_strikes: int) -> dict:
    """Hand a caller the ±n_strikes it asked for out of a wider cached chain.

    Fyers centres the chain on ATM, so the 2n+1 strikes around ATM of a wide response
    ARE the narrow response (checked live 2026-10-13 on all four indices: ±15 == centre
    of ±25). Callers keep the shape they always had; only the fetch is shared. Totals
    (callOi/putOi) are expiry-wide either way. A full-width response is centred by Fyers,
    so its middle strike is used as is; a lopsided one (strikes run out on one side) is
    re-centred on the underlying's own price."""
    try:
        chain = data["data"]["optionsChain"]
    except Exception:
        return data
    strikes = sorted({e.get("strike_price", -1) for e in chain} - {-1})
    if len(strikes) <= 2 * n_strikes + 1:
        return data
    mid = len(strikes) // 2
    spot = next((e.get("ltp") for e in chain if e.get("strike_price", -1) == -1), 0) or 0
    if spot and len(strikes) != 2 * _OC_MIN_STRIKES + 1:
        mid = min(range(len(strikes)), key=lambda i: abs(strikes[i] - spot))
        mid = max(n_strikes, min(mid, len(strikes) - n_strikes - 1))   # keep 2n+1 in range
    keep = set(strikes[mid - n_strikes: mid + n_strikes + 1])
    keep.add(-1)                                   # the underlying's own row
    return {**data, "data": {**data["data"], "optionsChain": [
        e for e in chain if e.get("strike_price", -1) in keep]}}


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
_FUT_TTL     = 30.0           # was 2s: the 2s option-chain tick turned that into 30 calls/min
_FUT_ERR_TTL = 8.0            # after a failed fetch, do not retry on every tick
_fut_fail_until = 0.0         # wall time before which a failed fetch is not retried
_fut_flight = threading.Lock()


def futures_symbols(index_sym: str) -> list[dict]:
    """Build near/next/far futures symbols for the given index, anchored to the REAL
    NSE expiry calendar (last Tuesday, holiday-adjusted — core.market_calendar,
    validated vs fno_bhavcopy). This rolls 'near' to the next month the moment the
    current month's expiry passes; the old 'current calendar month = near' logic
    referenced an EXPIRED contract for the days between a mid-month expiry and
    month-end. Falls back to the calendar-month construction if the expiry calendar
    is unavailable."""
    ul = _FUT_UNDERLYING.get(index_sym, "")
    if not ul:
        return []
    out = []
    try:
        from core.market_calendar import index_future_expiries
        for i, exp in enumerate(index_future_expiries(n=3)):
            sym = f"NSE:{ul}{str(exp.year)[2:]}{_MONTH_ABB[exp.month]}FUT"
            out.append({"symbol": sym, "label": _FUT_LABELS[i],
                        "month": f"{_MONTH_ABB[exp.month]} {exp.year}",
                        "expiry": exp.isoformat()})
        return out
    except Exception:
        pass
    now = datetime.datetime.now(tz=IST)
    for i in range(3):
        mo = (now.month - 1 + i) % 12 + 1
        yr = now.year + (now.month - 1 + i) // 12
        sym = f"NSE:{ul}{str(yr)[2:]}{_MONTH_ABB[mo]}FUT"
        out.append({"symbol": sym, "label": _FUT_LABELS[i],
                    "month": f"{_MONTH_ABB[mo]} {yr}"})
    return out


def fetch_futures(index_sym: str, on_fresh=None) -> list[dict]:
    """Fetch near/next/far futures quotes (no DB write of its own). Cached 30s.

    One request carries ALL indices' strips (12 symbols, cap is 50): the 60s signal
    eval asks for the four indices one after another, which was four calls a minute
    for what one call returns.

    on_fresh(index_sym, result): optional callback fired for every index in the batch,
    only on a real network fetch (not on a cache hit) — lets a caller persist the
    strips without the adapter knowing storage.
    """
    def cached():
        """The strip to return without a fetch, or None when one is due."""
        with _fut_lock:
            c = _fut_cache.get(index_sym)
            if c and time.time() - c["ts"] < (_FUT_TTL if _market_live() else _IDLE_TTL):
                return c["data"]
            if time.time() < _fut_fail_until:
                return []
            return None

    hit = cached()
    if hit is not None:
        return hit
    if index_sym not in _FUT_UNDERLYING:
        return []
    with _fut_flight:             # one fetch at a time; callers queued behind share it
        hit = cached()
        if hit is not None:
            return hit
        return _fetch_futures_batch(index_sym, on_fresh)


def _fetch_futures_batch(index_sym: str, on_fresh) -> list[dict]:
    global _fut_fail_until
    infos = {s: futures_symbols(s) for s in _FUT_UNDERLYING}
    if not infos.get(index_sym):
        return []
    raw, status = None, None
    if _acquire("futures"):
        try:
            resp = requests.get(
                QUOTES_URL,
                headers={"Authorization": auth_header(),
                         "Content-Type": "application/json", "version": "3"},
                params={"symbols": ",".join(s["symbol"] for si in infos.values() for s in si)},
                timeout=10,
            )
            status = resp.status_code
            raw = resp.json()
        except Exception:
            raw = None
        if not isinstance(raw, dict):
            raw = {"s": "error", "message": f"HTTP {status}"}
        if raw.get("s") != "ok":
            _note_reject("futures", raw.get("message"), status)
    if not raw or raw.get("s") != "ok":
        with _fut_lock:
            _fut_fail_until = time.time() + _error_ttl((raw or {}).get("message"), _FUT_ERR_TTL)
        return []

    q_map = {item["n"]: item.get("v", {}) for item in raw.get("d", [])}
    fresh = {}
    for s, sym_info in infos.items():
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
        fresh[s] = result

    with _fut_lock:
        ts = time.time()
        for s, result in fresh.items():
            _fut_cache[s] = {"data": result, "ts": ts}
    if on_fresh is not None:
        for s, result in fresh.items():
            if any(r["ltp"] for r in result):      # an all-zero strip is a miss, not a quote
                on_fresh(s, result)
    return fresh[index_sym]


_QUOTES_MAX = 50


def fetch_quotes(symbols: list[str]) -> dict:
    """Fetch last price (lp) for a list of Fyers symbols via /data/quotes."""
    if not symbols:
        return {}
    out: dict = {}
    symbols = list(dict.fromkeys(symbols))
    for i in range(0, len(symbols), _QUOTES_MAX):      # Standard plan: 50 symbols a request
        if not _acquire("quotes"):
            break
        status = None
        try:
            resp = requests.get(
                QUOTES_URL,
                headers={"Authorization": auth_header(), "version": "3"},
                params={"symbols": ",".join(symbols[i:i + _QUOTES_MAX])},
                timeout=8,
            )
            status = resp.status_code
            body = resp.json()
            if body.get("s") != "ok":
                _note_reject("quotes", body.get("message"), status)
            for item in (body.get("d") or []):
                n = item.get("n")
                lp = (item.get("v") or {}).get("lp")
                if n and lp is not None:
                    out[n] = float(lp)
        except Exception:
            if status == 429:
                _note_reject("quotes", "HTTP 429", status)
    return out
