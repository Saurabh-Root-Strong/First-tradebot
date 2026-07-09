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


def warn_once(exc: BaseException, context: str | None = None, every: int = 500) -> int:
    """Record a swallowed exception; print the 1st + every `every`-th for that context.
    Returns the running count. `context` defaults to the file:line where the exception was
    RAISED (deepest traceback frame), so a bare warn_once(e) self-labels to the failing
    statement — one call site instruments many try/excepts with distinct contexts."""
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
        print(f"[warn-once x{n}] {context}: {type(exc).__name__}: {str(exc)[:180]}",
              file=sys.stderr, flush=True)
    return n


def warn_counts() -> dict[str, int]:
    """Snapshot of per-context swallow counts (0 contexts = nothing degrading)."""
    return dict(_counts)


def _reset() -> None:            # test hook
    _counts.clear()
