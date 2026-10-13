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
    monkeypatch.setattr(rest, "_fut_fail_ts", 0.0)
    monkeypatch.setattr(rest, "auth_header", lambda: "app:token")
    monkeypatch.setattr(rest.requests, "get", fake_get)
    for store in (rest._day, rest._said, rest._recent, rest._oc_cache,
                  rest._oc_nearest, rest._fut_cache):
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
    rest._save(0, every=-1)
    rest._day.clear()                                       # a new process
    assert rest.data_call_stats()["used"] == 1
    stale = dict(json.loads(rest._STATE_FILE.read_text()), date="2000-01-01", used=4000)
    rest._STATE_FILE.write_text(json.dumps(stale))
    rest._day.clear()
    assert rest.data_call_stats()["used"] == 0


def test_fyers_side_limit_rejection_is_counted_not_swallowed(net, monkeypatch):
    monkeypatch.setattr(rest.requests, "get", lambda *a, **k: _Resp(
        {"s": "error", "message": "request limit reached"}))
    assert rest.fetch_quotes(["NSE:A"]) == {}
    assert rest.fetch_futures("NSE:NIFTY50-INDEX") == []
    assert rest.data_call_stats()["rejected"] == 2


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
