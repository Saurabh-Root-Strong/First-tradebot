"""
supervise.py  --  Run-and-leave-it supervisor for the Tradebot dashboard.

Guarantees continuous live capture across a trading session:
  1. Ensures a valid Fyers token (launches fyers_auth.py for browser login if not).
  2. Launches dashboard.py.
  3. Watches it — restarts automatically if:
       - the process dies, OR
       - the WebSocket stalls during market hours (no tick for WS_STALL_SEC),
         detected via the data/ws_heartbeat.txt the dashboard writes every 10s.
  4. Logs every event + capture gap to logs/supervisor.log.

Morning flow:  .venv\\Scripts\\python.exe supervise.py
    → if the token is stale, a browser opens for Fyers login; log in once,
      then walk away — the supervisor keeps the dashboard alive all session.

Stop with Ctrl+C.
"""

import base64
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
HERE = Path(__file__).parent
PY   = HERE / ".venv" / "Scripts" / "python.exe"
PY   = PY if PY.exists() else Path(sys.executable)
TOKEN_FILE = HERE / "access_token.txt"
HEARTBEAT  = HERE / "data" / "ws_heartbeat.txt"
LOG        = HERE / "logs" / "supervisor.log"

MKT_OPEN     = datetime.time(9, 15)
MKT_CLOSE    = datetime.time(15, 30)
WS_STALL_SEC = 90      # no tick this long during market hrs → restart
HEALTH_POLL  = 15      # supervisor loop interval (s)
START_GRACE  = 120     # don't health-check a freshly-started process for this long


def log(msg: str) -> None:
    line = f"[{datetime.datetime.now(tz=IST):%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def token_valid() -> bool:
    try:
        raw = TOKEN_FILE.read_text(encoding="utf-8").strip()
        p = raw.split(".")[1]; p += "=" * (4 - len(p) % 4)
        exp = json.loads(base64.urlsafe_b64decode(p)).get("exp", 0)
        return (exp - time.time()) > 0
    except Exception:
        return False


def ensure_token() -> bool:
    if token_valid():
        return True
    log("Token missing/expired — launching fyers_auth.py (log in via the browser)…")
    try:
        subprocess.run([str(PY), str(HERE / "fyers_auth.py")], cwd=str(HERE))
    except Exception as exc:
        log(f"fyers_auth.py failed: {exc}")
    ok = token_valid()
    log("Token OK." if ok else "Token still invalid after auth attempt.")
    return ok


def is_market_hours(now: datetime.datetime) -> bool:
    return now.weekday() < 5 and MKT_OPEN <= now.time() <= MKT_CLOSE


def heartbeat_age() -> float:
    try:
        return time.time() - float(HEARTBEAT.read_text().strip())
    except Exception:
        return 1e9


def _free_port_8050() -> None:
    """Kill any process already bound to 8050 so a relaunch always starts clean
    (prevents the 'old instance holds the port, new code never loads' trap)."""
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-NetTCPConnection -LocalPort 8050 -State Listen -ErrorAction SilentlyContinue "
             "| ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }"],
            timeout=15,
        )
    except Exception:
        pass
    time.sleep(2)


def launch(first: bool = False) -> subprocess.Popen:
    _free_port_8050()
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"}
    if not first:
        env["TRADEBOT_NO_BROWSER"] = "1"   # only the first launch pops the browser
    log("Launching dashboard.py" + ("" if first else " (restart — no new browser tab)"))
    return subprocess.Popen([str(PY), str(HERE / "dashboard.py")], cwd=str(HERE), env=env)


def main() -> None:
    log("=== Supervisor start ===")
    if not ensure_token():
        log("No valid token — aborting. Run fyers_auth.py, then re-run supervise.py.")
        sys.exit(1)

    proc = launch(first=True)
    started = time.time()
    try:
        while True:
            time.sleep(HEALTH_POLL)
            now = datetime.datetime.now(tz=IST)

            # 1) process died → restart
            if proc.poll() is not None:
                log(f"Dashboard exited (code {proc.returncode}) — restarting")
                if not ensure_token():
                    log("Cannot restart without a valid token — waiting 60s.")
                    time.sleep(60)
                    continue
                proc = launch(); started = time.time()
                continue

            # 2) WS stalled during market hours → restart (after start grace)
            if is_market_hours(now) and (time.time() - started) > START_GRACE:
                age = heartbeat_age()
                if age > WS_STALL_SEC:
                    log(f"WS stalled — no tick for {age:.0f}s during market hours — restarting")
                    try:
                        proc.kill(); proc.wait(timeout=10)
                    except Exception:
                        pass
                    proc = launch(); started = time.time()
    except KeyboardInterrupt:
        log("Supervisor stopped by user — terminating dashboard")
        try:
            proc.terminate(); proc.wait(timeout=10)
        except Exception:
            pass
        log("=== Supervisor stop ===")


if __name__ == "__main__":
    main()
