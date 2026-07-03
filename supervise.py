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

from core.constants import IST   # single source of truth
from core.market_calendar import is_trading_day   # NSE holidays + weekends
HERE = Path(__file__).parent
PY   = HERE / ".venv" / "Scripts" / "python.exe"
PY   = PY if PY.exists() else Path(sys.executable)
from tradebot.adapters.broker.token import TOKEN_FILE   # single broker-token source
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
    # On a headless server (FYERS_HEADLESS=1) use the TOTP auth — no browser.
    headless = os.environ.get("FYERS_HEADLESS") == "1"
    auth_script = "fyers_auth_headless.py" if headless else "fyers_auth.py"
    log(f"Token missing/expired — launching {auth_script}"
        + ("" if headless else " (log in via the browser)") + "…")
    try:
        subprocess.run([str(PY), str(HERE / auth_script)], cwd=str(HERE))
    except Exception as exc:
        log(f"{auth_script} failed: {exc}")
    ok = token_valid()
    log("Token OK." if ok else "Token still invalid after auth attempt.")
    return ok


def is_market_hours(now: datetime.datetime) -> bool:
    # trading day (weekday AND not an NSE holiday) within session hours
    return is_trading_day(now.date()) and MKT_OPEN <= now.time() <= MKT_CLOSE


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


def _other_supervisors() -> list[int]:
    """PIDs of OTHER running supervise.py processes (cmdline scan, so it also sees
    instances started before this guard existed). Empty list on any scan failure —
    fail-open so a broken scan can't block the morning launch."""
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" "
             "-ErrorAction SilentlyContinue "
             "| ForEach-Object { \"$($_.ProcessId) $($_.CommandLine)\" }"],
            capture_output=True, text=True, timeout=20)
        pids = []
        for line in (r.stdout or "").splitlines():
            if "supervise.py" in line:
                head = line.strip().split()
                if head and head[0].isdigit() and int(head[0]) != os.getpid():
                    pids.append(int(head[0]))
        return pids
    except Exception:
        return []


def main() -> None:
    # SINGLE INSTANCE — a second supervisor is a port war: its launch kills the
    # first one's dashboard (_free_port_8050), then BOTH restart loops fight over
    # 8050 forever. Required now that a scheduled task can auto-launch this script
    # on wake/logon while a healthy instance is already running.
    others = _other_supervisors()
    if others:
        log(f"Supervisor already running (PID {others[0]}) — exiting, not a failure.")
        return
    log("=== Supervisor start ===")
    if not ensure_token():
        log("No valid token — aborting. Run fyers_auth.py, then re-run supervise.py.")
        sys.exit(1)

    proc = launch(first=True)
    started = time.time()
    harvested_date = None          # footprint-validation harvest runs once/day post-close
    try:
        while True:
            time.sleep(HEALTH_POLL)
            now = datetime.datetime.now(tz=IST)

            # 0) once per weekday after close, append today to the footprint-validation
            #    ledger (accumulates toward a real verdict). Fire-and-forget; safe —
            #    reads lock-free mirrors, never touches the live capture.
            if (now.weekday() < 5 and now.time() > datetime.time(15, 35)
                    and harvested_date != now.date()):
                harvested_date = now.date()
                try:
                    subprocess.Popen([str(PY), str(HERE / "footprint_validate.py"), "--harvest"],
                                     cwd=str(HERE))
                    log("Post-close: footprint validation harvest triggered")
                except Exception as exc:
                    log(f"harvest trigger failed: {exc}")

            # 1) process died → restart
            if proc.poll() is not None:
                log(f"Dashboard exited (code {proc.returncode}) — restarting")
                if not ensure_token():
                    log("Cannot restart without a valid token — waiting 60s.")
                    time.sleep(60)
                    continue
                proc = launch(); started = time.time()
                continue

            # 2) WS stalled during market hours → restart (after start grace).
            # ALSO require the market itself to have been open > WS_STALL_SEC: a
            # dashboard started pre-open has (correctly) no ticks yet, so at 09:15:00
            # sharp its heartbeat is already "stale" by wall-clock and the old check
            # restart-killed a healthy process in the open's first seconds. The WS
            # gets the same 90s from the OPEN as it gets from any mid-session stall.
            open_dt = datetime.datetime.combine(now.date(), MKT_OPEN, tzinfo=IST)
            since_open = (now - open_dt).total_seconds()
            if (is_market_hours(now) and (time.time() - started) > START_GRACE
                    and since_open > WS_STALL_SEC):
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
