"""Unit — core.obs.warn_once: throttled, auto-labelling swallow observability."""
import io
import sys

from core.obs import _reset, warn_counts, warn_once


def test_warn_once_never_raises_on_cp1252_stderr_with_non_ascii():
    """HARD CONTRACT: warn_once is called from `except` blocks that previously did `pass`.
    If it can raise, observability turns a gracefully-swallowed error into a CRASH.
    Regression: a cp1252 stderr (Windows .bat `>> log 2>&1`) + a '₹' in the exception message
    raised UnicodeEncodeError and killed the caller."""
    _reset()
    real = sys.stderr
    sys.stderr = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")
    try:
        try:
            raise ValueError("bad ₹ value — delivery ✓")
        except Exception as e:
            n = warn_once(e)          # must NOT raise
        assert n >= 1
    finally:
        sys.stderr = real
    assert len(warn_counts()) == 1


def test_warn_once_never_raises_on_broken_stderr():
    """Even a totally broken stderr must not propagate out of the except block."""
    _reset()
    real = sys.stderr

    class Boom:
        def write(self, *_a, **_k): raise OSError("stderr is gone")
        def flush(self, *_a, **_k): raise OSError("stderr is gone")

    sys.stderr = Boom()
    try:
        try:
            raise KeyError("x")
        except Exception as e:
            n = warn_once(e)          # must NOT raise
        assert n >= 1
    finally:
        sys.stderr = real


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
