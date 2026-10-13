"""Unit — the Fyers Data-API budget gate and the call-sharing it sits on.

Offline: requests.get is faked, the state file lives in tmp_path. These pin the
accounting (what counts, what is refused, what survives a restart) and the three
sharing rules that keep one dashboard under 5,000 Data calls a day — not any quota
Fyers itself enforces.
"""
import json

import pytest

from tradebot.adapters.broker import rest


class _Resp:
    status_code = 200
    text = ""

    def __init__(self, body):
        self._body = body

    def json(self):
        return self._body


def _chain_body(n):
    """A Fyers-shaped chain: the underlying row plus ATM ± n strikes, both sides."""
    rows = [{"strike_price": -1, "option_type": "", "symbol": "NSE:NIFTY50-INDEX"}]
    for k in range(-n, n + 1):
        for side in ("CE", "PE"):
            rows.append({"strike_price": 22000 + 50 * k, "option_type": side, "oi": 1})
    return {"s": "ok", "data": {"optionsChain": rows, "callOi": 1, "putOi": 1,
                                "expiryData": [{"date": "13-10-2026", "expiry": "111"},
                                               {"date": "19-10-2026", "expiry": "222"}]}}


@pytest.fixture
def net(monkeypatch, tmp_path):
    """Clean gate + caches, a fake network that records every request it is sent."""
    calls = []

    def fake_get(url, headers=None, params=None, timeout=None):
        calls.append((url, dict(params or {})))
        if url == rest.OC_URL:
            return _Resp(_chain_body(int(params["strikecount"])))
        return _Resp({"s": "ok", "d": [{"n": s, "v": {"lp": 100.0}}
                                       for s in params["symbols"].split(",")]})

    monkeypatch.setattr(rest, "_STATE_FILE", tmp_path / "calls.json")
    monkeypatch.setattr(rest, "_SEC_SOFT", 10 ** 6)        # no real-time waiting in units
    monkeypatch.setattr(rest, "_fut_fail_until", 0.0)
    monkeypatch.setattr(rest, "_market_live", lambda now=None: True)
    monkeypatch.setattr(rest, "_reserve", lambda now=None: 0)
    monkeypatch.setattr(rest, "_cool", {"until": 0.0, "level": 0, "last": 0.0})
    monkeypatch.setattr(rest, "auth_header", lambda: "app:token")
    monkeypatch.setattr(rest.requests, "get", fake_get)
    for store in (rest._day, rest._said, rest._recent, rest._oc_cache,
                  rest._oc_nearest, rest._oc_flight, rest._fut_cache):
        store.clear()
    yield calls
    for store in (rest._day, rest._said, rest._recent):
        store.clear()


def _strikes(data):
    return sorted({e["strike_price"] for e in data["data"]["optionsChain"]} - {-1})


def test_every_data_call_is_counted_by_kind(net):
    rest.fetch_option_chain("NSE:NIFTY50-INDEX")
    rest.fetch_quotes(["NSE:X"])
    rest.fetch_futures("NSE:NIFTY50-INDEX")
    st = rest.data_call_stats()
    assert st["used"] == 3 == len(net)
    assert st["by_kind"] == {"chain": 1, "quotes": 1, "futures": 1}
    assert rest.data_calls_remaining() == rest._DAY_SOFT - 3


def test_minute_budget_refuses_instead_of_sending(net, monkeypatch):
    monkeypatch.setattr(rest, "_MIN_SOFT", 3)
    for i in range(3):
        assert rest.fetch_quotes([f"NSE:S{i}"])
    assert rest.fetch_quotes(["NSE:S9"]) == {}             # refused: nothing went out
    assert len(net) == 3
    assert rest.data_call_stats()["denied"] == 1


def test_day_budget_spent_returns_an_error_chain_without_a_request(net, monkeypatch):
    monkeypatch.setattr(rest, "_DAY_SOFT", 1)
    assert rest.fetch_option_chain("NSE:NIFTY50-INDEX")["s"] == "ok"
    out = rest.fetch_option_chain("NSE:NIFTYBANK-INDEX")
    assert out["s"] == "error" and "rate gate" in out["message"]
    assert len(net) == 1


def test_count_survives_a_restart_but_not_midnight(net, monkeypatch):
    rest.fetch_quotes(["NSE:A"])
    rest._write(rest._snapshot(every=-1))
    rest._day.clear()                                       # a new process
    rest._recent.clear()
    st = rest.data_call_stats()
    assert st["used"] == 1 and st["last_minute"] == 1       # the minute window came back too
    stale = dict(json.loads(rest._STATE_FILE.read_text()), date="2000-01-01", used=4000)
    rest._STATE_FILE.write_text(json.dumps(stale))
    rest._day.clear()
    assert rest.data_call_stats()["used"] == 0


def test_fyers_side_limit_rejection_is_counted_not_swallowed(net, monkeypatch):
    monkeypatch.setattr(rest.requests, "get", lambda *a, **k: _Resp(
        {"s": "error", "message": "request limit reached"}))
    assert rest.fetch_quotes(["NSE:A"]) == {}
    assert rest.data_call_stats()["rejected"] == 1


def test_history_is_counted_apart_from_the_data_budget(net):
    rest.note_history_call()
    st = rest.data_call_stats()
    assert st["history"] == 1 and st["used"] == 0


def test_one_futures_request_serves_all_four_indices(net):
    seen = []
    first = rest.fetch_futures("NSE:NIFTY50-INDEX", on_fresh=lambda s, r: seen.append(s))
    assert len(net) == 1 and len(net[0][1]["symbols"].split(",")) == 12
    assert sorted(seen) == sorted(rest._FUT_UNDERLYING)     # every strip handed to storage
    assert first and all(f["ltp"] == 100.0 for f in first)
    for sym in rest._FUT_UNDERLYING:                        # the other three ride the cache
        assert rest.fetch_futures(sym, on_fresh=lambda s, r: seen.append(s))
    assert len(net) == 1 and len(seen) == 4


def test_failed_futures_fetch_backs_off_and_persists_nothing(net, monkeypatch):
    monkeypatch.setattr(rest.requests, "get", lambda *a, **k: _Resp({"s": "error", "message": "x"}))
    seen = []
    assert rest.fetch_futures("NSE:NIFTY50-INDEX", on_fresh=lambda s, r: seen.append(s)) == []
    used = rest.data_call_stats()["used"]
    assert rest.fetch_futures("NSE:NIFTY50-INDEX") == []    # inside the back-off: no retry
    assert rest.data_call_stats()["used"] == used and not seen


def test_narrow_caller_gets_its_width_from_a_wide_fetch(net):
    narrow = rest.fetch_option_chain("NSE:NIFTY50-INDEX", n_strikes=15)
    assert net[0][1]["strikecount"] == rest._OC_MIN_STRIKES  # fetched at capture width
    assert len(_strikes(narrow)) == 31 and _strikes(narrow)[15] == 22000
    assert any(e["strike_price"] == -1 for e in narrow["data"]["optionsChain"])
    wide = rest.fetch_option_chain("NSE:NIFTY50-INDEX", n_strikes=25)
    assert len(_strikes(wide)) == 51 and len(net) == 1       # poller reuses it, full width


def test_nearest_expiry_by_epoch_shares_the_blank_expiry_entry(net):
    rest.fetch_option_chain("NSE:NIFTY50-INDEX")
    rest.fetch_option_chain("NSE:NIFTY50-INDEX", "111")      # nearest, asked for by epoch
    assert len(net) == 1
    rest.fetch_option_chain("NSE:NIFTY50-INDEX", "222")      # a later expiry is its own chain
    assert len(net) == 2 and net[1][1]["timestamp"] == "222"


def test_max_age_rides_an_older_good_chain_but_never_an_error(net, monkeypatch):
    rest.fetch_option_chain("NSE:NIFTY50-INDEX")
    rest._oc_cache[("NSE:NIFTY50-INDEX", "")]["ts"] -= 40    # older than the 25s TTL
    rest.fetch_option_chain("NSE:NIFTY50-INDEX", max_age=75)
    assert len(net) == 1
    rest.fetch_option_chain("NSE:NIFTY50-INDEX")             # default caller refetches
    assert len(net) == 2
    e = rest._oc_cache[("NSE:NIFTY50-INDEX", "")]
    e["data"], e["ttl"] = {"s": "error", "message": "x"}, rest._OC_ERR_TTL
    e["ts"] -= 20
    rest.fetch_option_chain("NSE:NIFTY50-INDEX", max_age=75)
    assert len(net) == 3


# ── failure modes ──────────────────────────────────────────────────────────────

def test_a_fyers_rejection_pauses_every_data_call_and_escalates(net, monkeypatch):
    sent = []

    def rejecting(url, **k):
        sent.append(url)
        return _Resp({"s": "error", "message": "request limit reached"})

    monkeypatch.setattr(rest.requests, "get", rejecting)
    assert rest.fetch_quotes(["NSE:A"]) == {}                 # sent, rejected
    assert rest.fetch_option_chain("NSE:NIFTY50-INDEX")["s"] == "error"
    assert rest.fetch_futures("NSE:NIFTY50-INDEX") == []
    assert len(sent) == 1                                     # the pause held the other two
    assert 55 < rest._cool["until"] - rest.time.monotonic() <= 60
    rest._cool["until"] = 0.0                                 # the pause runs out ...
    rest._oc_cache.clear()
    rest.fetch_option_chain("NSE:NIFTY50-INDEX")              # ... and Fyers rejects again
    assert 115 < rest._cool["until"] - rest.time.monotonic() <= 120


def test_http_429_without_a_json_body_still_counts_as_a_rejection(net, monkeypatch):
    class _Raw(_Resp):
        status_code, text = 429, "Too Many Requests"

        def json(self):
            raise ValueError("no json")

    monkeypatch.setattr(rest.requests, "get", lambda *a, **k: _Raw(None))
    rest.fetch_option_chain("NSE:NIFTY50-INDEX")
    rest._cool["until"] = 0.0
    rest.fetch_quotes(["NSE:A"])
    rest._cool["until"] = 0.0
    rest.fetch_futures("NSE:NIFTY50-INDEX")
    assert rest.data_call_stats()["rejected"] == 3


def test_when_budget_is_short_only_the_essential_feeds_are_sent(net, monkeypatch):
    monkeypatch.setattr(rest, "_reserve", lambda now=None: 10 ** 6)   # everything is reserved
    assert rest.fetch_option_chain("NSE:NIFTYBANK-INDEX")["s"] == "error"
    assert rest.fetch_option_chain("NSE:NIFTY50-INDEX", "222")["s"] == "error"   # a far expiry
    assert len(net) == 0
    assert rest.fetch_option_chain("NSE:NIFTY50-INDEX")["s"] == "ok"
    assert rest.fetch_futures("NSE:NIFTY50-INDEX") and rest.fetch_quotes(["NSE:A"])
    assert len(net) == 3


def test_reserve_covers_the_rest_of_the_session_and_nothing_outside_it():
    import datetime as dt

    def at(day, h, m):                                        # 13 Oct 2026 is a Tuesday
        return dt.datetime(2026, 10, day, h, m, tzinfo=rest.IST)

    full = 377 * rest._ESSENTIAL_PER_MIN + rest._ESSENTIAL_TAIL
    assert rest._reserve(at(13, 7, 0)) == full                # before the open: the whole day
    assert rest._reserve(at(13, 9, 14)) == full
    assert rest._reserve(at(13, 15, 1)) == 30 * rest._ESSENTIAL_PER_MIN + rest._ESSENTIAL_TAIL
    assert rest._reserve(at(13, 15, 40)) == rest._ESSENTIAL_TAIL
    assert rest._reserve(at(13, 15, 46)) == 0
    assert rest._reserve(at(17, 11, 0)) == 0                  # Saturday
    assert rest._market_live(at(13, 9, 0)) and rest._market_live(at(13, 15, 45))
    assert not rest._market_live(at(13, 8, 59)) and not rest._market_live(at(17, 11, 0))


def test_market_shut_a_chain_and_a_strip_are_not_bought_again(net, monkeypatch):
    rest.fetch_option_chain("NSE:NIFTY50-INDEX")
    rest.fetch_futures("NSE:NIFTY50-INDEX")
    rest._oc_cache[("NSE:NIFTY50-INDEX", "")]["ts"] -= 300
    rest._fut_cache["NSE:NIFTY50-INDEX"]["ts"] -= 300
    monkeypatch.setattr(rest, "_market_live", lambda now=None: False)
    rest.fetch_option_chain("NSE:NIFTY50-INDEX")
    rest.fetch_futures("NSE:NIFTY50-INDEX")
    assert len(net) == 2                                      # a tab left open costs nothing
    monkeypatch.setattr(rest, "_market_live", lambda now=None: True)
    rest.fetch_option_chain("NSE:NIFTY50-INDEX")
    assert len(net) == 3                                      # the open: stale again at once


def test_a_dead_token_is_not_retried_every_few_seconds(net, monkeypatch):
    monkeypatch.setattr(rest.requests, "get", lambda *a, **k: _Resp(
        {"s": "error", "code": -15, "message": "Please provide valid token"}))
    rest.fetch_option_chain("NSE:NIFTY50-INDEX")
    rest.fetch_futures("NSE:NIFTY50-INDEX")
    rest._oc_cache[("NSE:NIFTY50-INDEX", "")]["ts"] -= 30     # past the ordinary 8s back-off
    monkeypatch.setattr(rest, "_fut_fail_until", rest._fut_fail_until - 30)
    used = rest.data_call_stats()["used"]
    rest.fetch_option_chain("NSE:NIFTY50-INDEX")
    rest.fetch_futures("NSE:NIFTY50-INDEX")
    assert rest.data_call_stats()["used"] == used
    assert rest.data_call_stats()["rejected"] == 0            # not a limit: no pause


def test_callers_arriving_together_share_one_fetch(net, monkeypatch):
    import threading
    import time as _t
    sent = []

    def slow(url, headers=None, params=None, timeout=None):
        sent.append(url)
        _t.sleep(0.3)
        if url == rest.OC_URL:
            return _Resp(_chain_body(25))
        return _Resp({"s": "ok", "d": [{"n": s, "v": {"lp": 1.0}}
                                       for s in params["symbols"].split(",")]})

    monkeypatch.setattr(rest.requests, "get", slow)
    out = []
    jobs = [threading.Thread(
        target=lambda: out.append(rest.fetch_option_chain("NSE:NIFTY50-INDEX")["s"]))
        for _ in range(6)]
    jobs += [threading.Thread(target=lambda s=s: out.append(bool(rest.fetch_futures(s))))
             for s in rest._FUT_UNDERLYING]
    for j in jobs:
        j.start()
    for j in jobs:
        j.join()
    assert sorted(sent) == sorted([rest.OC_URL, rest.QUOTES_URL])
    assert out.count("ok") == 6 and out.count(True) == 4


def test_a_lopsided_chain_is_recentred_on_the_underlying(net):
    rows = [{"strike_price": -1, "option_type": "", "ltp": 22010.0}]
    for k in range(-5, 35):                                   # strikes ran out below ATM
        rows.append({"strike_price": 22000 + 50 * k, "option_type": "CE"})
    data = {"s": "ok", "data": {"optionsChain": rows}}
    assert _strikes(rest._trim_chain(data, 3)) == [21850, 21900, 21950, 22000, 22050, 22100, 22150]
    rows[0]["ltp"] = 21760.0                                  # ATM at the very edge: still 2n+1
    edge = _strikes(rest._trim_chain(data, 3))
    assert edge[0] == 21750 and len(edge) == 7


def test_more_than_fifty_symbols_go_out_in_chunks(net):
    syms = [f"NSE:S{i}" for i in range(120)] + ["NSE:S0"]     # one duplicate
    assert len(rest.fetch_quotes(syms)) == 120
    assert [len(c[1]["symbols"].split(",")) for c in net] == [50, 50, 20]


def test_wall_clock_jumping_backwards_does_not_jam_the_gate(net, monkeypatch):
    rest.fetch_quotes(["NSE:A"])
    real = rest.time.time
    monkeypatch.setattr(rest.time, "time", lambda: real() - 86400)
    assert rest.fetch_quotes(["NSE:B"])                       # the window runs on monotonic
