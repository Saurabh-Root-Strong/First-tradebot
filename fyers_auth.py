"""
fyers_auth.py: Daily Fyers access-token generator with auto-capture.

Run this ONCE before starting TradeBot each trading day.
Tokens are valid until midnight IST on the day they are issued.

Requirements: only 'requests' (already installed)

Usage:
    python fyers_auth.py

What happens:
    1. A local server starts on http://127.0.0.1:8085
    2. Your browser opens the Fyers login page
    3. You log in — Fyers redirects to http://127.0.0.1:8085 automatically
    4. The server captures the auth_code — NO copy-paste needed
    5. Access token is generated and saved to access_token.txt
    6. TradeBot.py reads it automatically

IMPORTANT — One-time Fyers Dashboard setup:
    Your app's redirect URI must be set to: http://127.0.0.1:8085
    Dashboard → your app → Edit (⋮) → change http://127.0.0.1 to http://127.0.0.1:8085 → Save
"""

import hashlib
import http.server
import os
import sys
import webbrowser
from pathlib import Path
from urllib.parse import urlencode, urlparse, parse_qs

import requests

# ── Credentials ────────────────────────────────────────────────────────────────
CLIENT_ID    = os.environ.get("FYERS_APP_ID",     "WVDZUTO6HL-100")
SECRET_KEY   = os.environ.get("FYERS_SECRET_KEY", "QEUPA8AAA6")
LOCAL_PORT   = int(os.environ.get("FYERS_PORT",   "8085"))
REDIRECT_URI = f"http://127.0.0.1:{LOCAL_PORT}"

# Fyers V3 endpoints
_AUTH_CODE_URL = "https://api-t1.fyers.in/api/v3/generate-authcode"
_VALIDATE_URL  = "https://api-t1.fyers.in/api/v3/validate-authcode"

from tradebot.adapters.broker.token import TOKEN_FILE   # single broker-token source


# ── Local OAuth callback server ────────────────────────────────────────────────

class _CallbackHandler(http.server.BaseHTTPRequestHandler):
    """Handles the single GET redirect from Fyers after login."""

    captured: dict = {}   # shared across handler instances

    def do_GET(self) -> None:
        params = parse_qs(urlparse(self.path).query)
        # Fyers sends both 'code' and 'auth_code' — prefer 'auth_code'
        code = (params.get("auth_code") or params.get("code") or [None])[0]

        if code:
            _CallbackHandler.captured["auth_code"] = code
            html = (
                b"<html><body style='font-family:sans-serif;text-align:center;margin-top:80px'>"
                b"<h2 style='color:#2e7d32'>&#10003; Login successful!</h2>"
                b"<p>Auth code captured. Return to the terminal.</p>"
                b"</body></html>"
            )
        else:
            html = (
                b"<html><body style='font-family:sans-serif;text-align:center;margin-top:80px'>"
                b"<h2 style='color:#c62828'>&#10007; No auth code received.</h2>"
                b"<p>Run fyers_auth.py again.</p>"
                b"</body></html>"
            )

        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(html)

    def log_message(self, *_) -> None:
        pass  # suppress server access logs


def _wait_for_auth_code(timeout: int = 120) -> str:
    """Start local server, block until Fyers redirects with auth_code."""
    _CallbackHandler.captured.clear()
    server = http.server.HTTPServer(("127.0.0.1", LOCAL_PORT), _CallbackHandler)
    server.timeout = 2  # check every 2s so we can detect timeout

    deadline = timeout
    while "auth_code" not in _CallbackHandler.captured and deadline > 0:
        server.handle_request()
        deadline -= 2

    server.server_close()

    if "auth_code" not in _CallbackHandler.captured:
        raise TimeoutError(f"No redirect received within {timeout} seconds. Did you log in?")

    return _CallbackHandler.captured["auth_code"]


# ── Token exchange ─────────────────────────────────────────────────────────────

def generate_access_token(auth_code: str, secret_key: str) -> str:
    app_id_hash = hashlib.sha256(
        f"{CLIENT_ID}:{secret_key}".encode("utf-8")
    ).hexdigest()

    resp = requests.post(
        _VALIDATE_URL,
        json={
            "grant_type": "authorization_code",
            "appIdHash":  app_id_hash,
            "code":       auth_code,
        },
        headers={"Content-Type": "application/json"},
        timeout=20,
    )

    try:
        data = resp.json()
    except Exception:
        data = {}

    if not resp.ok:
        msg = data.get("message") or data.get("errmsg") or resp.text or "Unknown error"
        raise RuntimeError(f"HTTP {resp.status_code} from Fyers: {msg}")

    if data.get("s") != "ok":
        raise RuntimeError(f"Fyers rejected token: {data}")

    token = data.get("access_token", "")
    if not token:
        raise RuntimeError(f"Empty token in response: {data}")
    return token


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    sep = "=" * 62
    print(sep)
    print("  FYERS DAILY TOKEN GENERATOR")
    print(sep)
    print(f"  App ID      : {CLIENT_ID}")
    print(f"  Redirect URI: {REDIRECT_URI}")
    print(sep)

    secret_key = SECRET_KEY
    if not secret_key:
        secret_key = input(
            "\nEnter your App Secret (Fyers API Dashboard → your app): "
        ).strip()
        if not secret_key:
            print("ERROR: App Secret is required.")
            sys.exit(1)

    # Build Fyers login URL
    auth_url = (
        f"{_AUTH_CODE_URL}?"
        + urlencode({
            "client_id":     CLIENT_ID,
            "redirect_uri":  REDIRECT_URI,
            "response_type": "code",
            "state":         "tradebot",
            "scope":         "",
            "nonce":         "",
        })
    )

    print(f"\nStarting local callback server on {REDIRECT_URI} ...")
    print("Opening Fyers login in your browser...\n")

    # Open browser AFTER server is ready (server starts on first handle_request call)
    webbrowser.open(auth_url, new=1)

    print("Waiting for you to log in (timeout: 2 minutes)...")
    print("After login, the browser shows a green success page automatically.\n")

    try:
        auth_code = _wait_for_auth_code(timeout=120)
    except TimeoutError as exc:
        print(f"\nERROR: {exc}")
        print("Make sure the redirect URI in your Fyers Dashboard is exactly:")
        print(f"  {REDIRECT_URI}")
        sys.exit(1)

    print("Auth code captured automatically.")
    print("\nGenerating access token...")

    try:
        access_token = generate_access_token(auth_code, secret_key)
    except Exception as exc:
        print(f"\nERROR: {exc}")
        print("\nCommon causes:")
        print("  - auth_code expired  (took more than 60s after login)")
        print("  - wrong Secret Key")
        print("  - redirect URI mismatch between code and Fyers Dashboard")
        sys.exit(1)

    TOKEN_FILE.write_text(access_token, encoding="utf-8")

    print()
    print(sep)
    print("  SUCCESS — Access token generated")
    print(sep)
    print(f"  Saved to    : {TOKEN_FILE.resolve()}")
    print(f"  Token prefix: {access_token[:50]}...")
    print()
    print("  TradeBot.py will auto-read access_token.txt on startup.")
    print()
    print("  Now run:  .venv\\Scripts\\python.exe TradeBot.py")
    print(sep)


if __name__ == "__main__":
    main()
