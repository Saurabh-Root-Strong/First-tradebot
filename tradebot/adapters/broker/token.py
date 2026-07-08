"""Single source of truth for the Fyers access token.

Before this, `TOKEN_FILE = Path("access_token.txt")` was redefined in 5 modules —
and inconsistently: dashboard/signals/download_historical used a CWD-relative path
while supervise used a script-relative one. Launched from anywhere but the repo
root, they'd read different (or missing) files. Here the path is anchored to
PROJECT_ROOT (via core.constants), so it is correct regardless of working dir.

Leaf adapter: imports only core + stdlib. No dashboard, no Dash, no cycles.
"""
from __future__ import annotations

import base64
import json
import os
import time

from core.constants import PROJECT_ROOT

# Fyers app id ("client_id"). Env override wins; the literal is the existing default.
APP_ID = os.environ.get("FYERS_APP_ID", "WVDZUTO6HL-100")

TOKEN_FILE = PROJECT_ROOT / "access_token.txt"

# A token that still parses but has less than this much life left is treated as
# NOT usable, so the supervisor/dashboard re-auth PROACTIVELY instead of starting
# a session that dies seconds later (mid-run 401 → watchdog restart churn).
# Fyers mints daily tokens that expire at a fixed ~06:00-IST cutoff — NOT 24h from
# issue — so a session token is normally refreshed long before this; the margin
# only guards the edge case of launching in the last minutes before that wall.
REFRESH_MARGIN_SEC = 300  # 5 min


def read_token() -> str:
    """Raw JWT string from access_token.txt, or "" if absent."""
    return TOKEN_FILE.read_text(encoding="utf-8").strip() if TOKEN_FILE.exists() else ""


def _claims(raw: str | None = None) -> dict:
    """Decoded JWT payload. The ONE place the token is parsed — everything else
    (remaining/fy_id/is_usable/describe) reads from here. Raises on any malformed
    or missing token; callers catch and apply their own fail-open/closed policy."""
    raw = read_token() if raw is None else raw
    payload = raw.split(".")[1]
    payload += "=" * (4 - len(payload) % 4)
    return json.loads(base64.urlsafe_b64decode(payload))


def token_remaining(raw: str | None = None) -> "float | None":
    """Seconds left on the JWT (exp - now), or None if unparseable. Reads file if raw is None."""
    try:
        return _claims(raw).get("exp", 0) - time.time()
    except Exception:
        return None


def fy_id(raw: str | None = None) -> str:
    """fy_id claim from the JWT, or "?" if unavailable."""
    try:
        return _claims(raw).get("fy_id", "?")
    except Exception:
        return "?"


def is_usable(raw: str | None = None, margin: float = REFRESH_MARGIN_SEC) -> bool:
    """The single validity gate. True only when the token parses AND has more than
    `margin` seconds of life left. Supervisor + dashboard both defer to this instead
    of re-deriving `exp > now` from their own inline JWT decode — one margin, one
    definition of 'usable', no drift."""
    rem = token_remaining(raw)
    return rem is not None and rem > margin


def describe(raw: str | None = None) -> str:
    """One-line human summary of the token lifecycle: fy_id, issued→expires, life
    left. Surfaces Fyers' fixed ~06:00-IST daily cutoff every launch so an overnight
    expiry is never again mistaken for a bug ('why did it die in a day?')."""
    try:
        c = _claims(raw)
    except Exception:
        return "token: unreadable - malformed or missing (access_token.txt)"
    import datetime
    fy   = c.get("fy_id", "?")
    exp  = c.get("exp", 0)
    iat  = c.get("iat", 0)
    rem  = exp - time.time()
    fmt  = "%d-%b %H:%M"
    issued  = datetime.datetime.fromtimestamp(iat).strftime(fmt) if iat else "?"
    expires = datetime.datetime.fromtimestamp(exp).strftime(fmt) if exp else "?"
    if rem > 0:
        life = f"{int(rem // 3600)}h {int((rem % 3600) // 60)}m left"
    else:
        dead = -rem
        life = f"EXPIRED {int(dead // 3600)}h {int((dead % 3600) // 60)}m ago"
    # ASCII only — this string is print()ed to Windows consoles (cp1252) by
    # supervise/dashboard; non-ASCII (arrows/dots) would raise UnicodeEncodeError.
    return f"token: fy_id {fy} | issued {issued} -> expires {expires} IST | {life}"


def auth_header(raw: str | None = None) -> str:
    """The "APP_ID:token" string Fyers REST + websocket expect. Always reads file by
    default — avoids module-global timing issues with Dash worker threads."""
    raw = read_token() if raw is None else raw
    return f"{APP_ID}:{raw}"
