"""core.obs — minimal observability for SWALLOWED exceptions.

The chain_snapshots capture died silently for WEEKS because a per-record insert error was
caught by `except: pass`. That is the highest-severity failure mode in this system: a
try/except that must not crash the caller (graceful degradation) yet must ALSO not HIDE a
systematic bug — a silently-defaulted field feeds a wrong number into a live score with no
signal that anything broke.

warn_once() records such an error and prints the FIRST occurrence (and every Nth) per site,
throttled so a hot loop can't spam stderr. It does NOT change the degradation behaviour —
the caller still swallows and continues; the failure just stops being invisible. Counts are
queryable (warn_counts) so a health check / badge can surface "N contexts degrading".
"""
from __future__ import annotations

import sys
import traceback

_counts: dict[str, int] = {}


def _emit(msg: str) -> None:
    """Write to stderr, NEVER raising. stderr may be a cp1252 stream (Windows, redirected by
    a .bat as `>> log 2>&1`), and this codebase is full of ₹ / — / ✓ — printing a non-ASCII
    exception message there raises UnicodeEncodeError. Since warn_once is called from INSIDE
    `except` blocks that previously did `pass`, any raise here would convert a gracefully
    swallowed error into a CRASH. Fall back to ASCII, then to silence."""
    try:
        print(msg, file=sys.stderr, flush=True)
    except Exception:
        try:
            sys.stderr.write(msg.encode("ascii", "replace").decode("ascii") + "\n")
            sys.stderr.flush()
        except Exception:
            pass                      # observability must never win over the caller


def warn_once(exc: BaseException, context: str | None = None, every: int = 500) -> int:
    """Record a swallowed exception; print the 1st + every `every`-th for that context.
    Returns the running count (-1 if the helper itself failed). `context` defaults to the
    file:line where the exception was RAISED (deepest traceback frame), so a bare
    warn_once(e) self-labels to the failing statement — one call site instruments many
    try/excepts with distinct contexts.

    HARD CONTRACT: this function NEVER raises. It is invoked from `except` blocks whose
    prior behaviour was `pass`; if it could raise, adding observability would change control
    flow and crash callers that used to degrade gracefully. Every path is guarded."""
    try:
        if context is None:
            try:
                frames = traceback.extract_tb(exc.__traceback__)
                f = frames[-1] if frames else None
                fname = f.filename.replace("\\", "/").rsplit("/", 1)[-1] if f else "?"
                context = f"{fname}:{f.lineno}" if f else "?"
            except Exception:
                context = "?"
        n = _counts.get(context, 0) + 1
        _counts[context] = n
        if n == 1 or n % max(every, 1) == 0:
            _emit(f"[warn-once x{n}] {context}: {type(exc).__name__}: {str(exc)[:180]}")
        return n
    except Exception:
        return -1                     # never propagate out of an except block


def warn_counts() -> dict[str, int]:
    """Snapshot of per-context swallow counts (0 contexts = nothing degrading)."""
    return dict(_counts)


def _reset() -> None:            # test hook
    _counts.clear()
