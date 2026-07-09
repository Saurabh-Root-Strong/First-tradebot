"""Unit — core.obs.warn_once: throttled, auto-labelling swallow observability."""
from core.obs import _reset, warn_counts, warn_once


def _raise_bad():
    raise ValueError("bad")


def test_warn_once_throttles_and_autolabels_by_raise_site():
    _reset()
    for _ in range(3):
        try:
            _raise_bad()
        except Exception as e:
            warn_once(e, every=1000)          # under threshold -> prints once, counts all
    counts = warn_counts()
    assert len(counts) == 1                    # same raise site -> one context
    assert list(counts.values())[0] == 3       # all three counted
    assert any("test_unit_obs.py" in k for k in counts)   # auto-labelled to the raise line


def test_warn_once_explicit_context_and_distinct_sites():
    _reset()
    try:
        raise KeyError("x")
    except Exception as e:
        warn_once(e, context="mylabel")
    try:
        _raise_bad()
    except Exception as e:
        warn_once(e)
    counts = warn_counts()
    assert "mylabel" in counts                 # explicit context honoured
    assert len(counts) == 2                    # explicit + auto are distinct contexts


def test_warn_once_returns_running_count():
    _reset()
    n = None
    for _ in range(5):
        try:
            _raise_bad()
        except Exception as e:
            n = warn_once(e, every=1000)
    assert n == 5
