"""
fyers_auth_headless.py — daily Fyers access token with NO browser (for the cloud).

The interactive fyers_auth.py opens a browser for OAuth — impossible on a headless
server. This reproduces a full login programmatically via Fyers' TOTP flow, so the
cloud deploy can refresh its token unattended every morning:

    send_login_otp  →  verify_otp (TOTP)  →  verify_pin  →  authcode  →  access_token

It writes the same access_token.txt the dashboard/supervisor read, so it is a drop-in
replacement: supervise.py calls THIS instead of fyers_auth.py when FYERS_HEADLESS=1.

Required env (set on the server, never commit — see .env.example):
    FYERS_FY_ID         your Fyers login/client id (e.g. XS04195)
    FYERS_PIN           your 4-digit trading PIN
    FYERS_TOTP_SECRET   the TOTP/secret key from Fyers (Profile → enable TOTP; the
                        base32 secret behind the QR, not the 6-digit code)
    FYERS_APP_ID        API app id, e.g. WVDZUTO6HL-100   (defaults to that)
    FYERS_SECRET_KEY    API app secret
    FYERS_REDIRECT_URI  the redirect URI registered for the app

NOTE: this hits Fyers' internal vagator endpoints. They are stable in practice but
undocumented — if Fyers changes them, fall back to the one-time browser login. This
file could not be tested here (it needs your live credentials + TOTP), so verify it
once interactively on the server before trusting the unattended schedule.
"""
from __future__ import annotations

import base64
import json
import os
import sys
import time
from pathlib import Path

import requests

HERE       = Path(__file__).parent
TOKEN_FILE = HERE / "access_token.txt"

APP_ID       = os.environ.get("FYERS_APP_ID", "WVDZUTO6HL-100")
SECRET_KEY   = os.environ.get("FYERS_SECRET_KEY", "")
FY_ID        = os.environ.get("FYERS_FY_ID", "")
PIN          = os.environ.get("FYERS_PIN", "")
TOTP_SECRET  = os.environ.get("FYERS_TOTP_SECRET", "")
REDIRECT_URI = os.environ.get("FYERS_REDIRECT_URI", "http://127.0.0.1:8085")

_VAGATOR   = "https://api-t1.fyers.in/vagator/v2"
_TOKEN_URL = "https://api-t1.fyers.in/api/v3/token"
_VALIDATE  = "https://api-t1.fyers.in/api/v3/validate-authcode"
_TIMEOUT   = 20


def _b64(s: str) -> str:
    return base64.b64encode(s.encode()).decode()


def _totp_now(secret: str) -> str:
    """RFC-6238 TOTP (30s window, 6 digits) — no external dependency."""
    import hashlib, hmac, struct
    key = base64.b32decode(secret.strip().replace(" ", "").upper() + "=" * ((8 - len(secret) % 8) % 8))
    counter = struct.pack(">Q", int(time.time()) // 30)
    digest = hmac.new(key, counter, hashlib.sha1).digest()
    off = digest[-1] & 0x0F
    code = (struct.unpack(">I", digest[off:off + 4])[0] & 0x7FFFFFFF) % 1_000_000
    return f"{code:06d}"


def _post(url: str, payload: dict, headers: dict | None = None) -> dict:
    r = requests.post(url, json=payload, headers=headers or {}, timeout=_TIMEOUT)
    try:
        data = r.json()
    except Exception:
        raise RuntimeError(f"{url} → HTTP {r.status_code}, non-JSON body")
    return data


def _login_to_authcode() -> str:
    # 1. send OTP request
    r1 = _post(f"{_VAGATOR}/send_login_otp_v2", {"fy_id": _b64(FY_ID), "app_id": "2"})
    rk = r1.get("request_key")
    if not rk:
        raise RuntimeError(f"send_login_otp failed: {r1}")

    # 2. verify TOTP (retry once across a 30s boundary)
    rk2 = None
    for attempt in range(2):
        r2 = _post(f"{_VAGATOR}/verify_otp", {"request_key": rk, "otp": _totp_now(TOTP_SECRET)})
        rk2 = r2.get("request_key")
        if rk2:
            break
        if attempt == 0:
            time.sleep(31 - (int(time.time()) % 30))   # wait for a fresh TOTP window
    if not rk2:
        raise RuntimeError("verify_otp failed — check FYERS_TOTP_SECRET")

    # 3. verify PIN → vagator access token
    r3 = _post(f"{_VAGATOR}/verify_pin",
               {"request_key": rk2, "identity_type": "pin", "identifier": _b64(PIN)})
    vtoken = (r3.get("data") or {}).get("access_token")
    if not vtoken:
        raise RuntimeError(f"verify_pin failed: {r3}")

    # 4. exchange for an auth_code
    app_base = APP_ID.split("-")[0]
    app_type = APP_ID.split("-")[1] if "-" in APP_ID else "100"
    r4 = _post(_TOKEN_URL, {
        "fyers_id": FY_ID, "app_id": app_base, "redirect_uri": REDIRECT_URI,
        "appType": app_type, "code_challenge": "", "state": "tradebot",
        "scope": "", "nonce": "", "response_type": "code", "create_cookie": True,
    }, headers={"Authorization": f"Bearer {vtoken}"})
    url = r4.get("Url") or r4.get("url")
    if not url:
        raise RuntimeError(f"token endpoint did not return a redirect Url: {r4}")
    from urllib.parse import urlparse, parse_qs
    code = parse_qs(urlparse(url).query).get("auth_code", [None])[0]
    if not code:
        raise RuntimeError(f"no auth_code in redirect Url: {url}")
    return code


def _exchange(auth_code: str) -> str:
    import hashlib
    app_hash = hashlib.sha256(f"{APP_ID}:{SECRET_KEY}".encode()).hexdigest()
    data = _post(_VALIDATE, {"grant_type": "authorization_code",
                             "appIdHash": app_hash, "code": auth_code})
    token = data.get("access_token")
    if not token:
        raise RuntimeError(f"validate-authcode failed: {data}")
    return token


def main() -> None:
    missing = [k for k, v in {"FYERS_FY_ID": FY_ID, "FYERS_PIN": PIN,
               "FYERS_TOTP_SECRET": TOTP_SECRET, "FYERS_SECRET_KEY": SECRET_KEY}.items() if not v]
    if missing:
        print(f"ERROR: missing env: {', '.join(missing)}")
        sys.exit(1)
    print("Headless Fyers auth (TOTP)…")
    try:
        token = _exchange(_login_to_authcode())
    except Exception as exc:
        print(f"FAILED: {exc}")
        sys.exit(1)
    TOKEN_FILE.write_text(token, encoding="utf-8")
    print(f"OK — access token written to {TOKEN_FILE}")


if __name__ == "__main__":
    main()
