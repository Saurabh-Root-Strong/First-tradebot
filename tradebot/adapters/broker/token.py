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


def read_token() -> str:
    """Raw JWT string from access_token.txt, or "" if absent."""
    return TOKEN_FILE.read_text(encoding="utf-8").strip() if TOKEN_FILE.exists() else ""


def token_remaining(raw: str | None = None) -> "float | None":
    """Seconds left on the JWT (exp - now), or None if unparseable. Reads file if raw is None."""
    raw = read_token() if raw is None else raw
    try:
        payload = raw.split(".")[1]
        payload += "=" * (4 - len(payload) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload))
        return claims.get("exp", 0) - time.time()
    except Exception:
        return None


def fy_id(raw: str | None = None) -> str:
    """fy_id claim from the JWT, or "?" if unavailable."""
    raw = read_token() if raw is None else raw
    try:
        p = raw.split(".")[1]
        p += "=" * (4 - len(p) % 4)
        return json.loads(base64.urlsafe_b64decode(p)).get("fy_id", "?")
    except Exception:
        return "?"


def auth_header(raw: str | None = None) -> str:
    """The "APP_ID:token" string Fyers REST + websocket expect. Always reads file by
    default — avoids module-global timing issues with Dash worker threads."""
    raw = read_token() if raw is None else raw
    return f"{APP_ID}:{raw}"
