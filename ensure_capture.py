"""
ensure_capture.py — self-healing launcher for the Tradebot capture stack.

Runs every 5 minutes from a (non-admin) Task Scheduler job. If it is a trading
day, inside the capture window, and NO supervise.py is running, it launches one
DETACHED (own process group, no console) — so capture survives the two failure
modes that killed it on 2026-07-03: closing the terminal window and a laptop
lid-close/sleep (on wake, the next 5-min tick resurrects capture).

Silent no-op otherwise. supervise.py's own single-instance guard is the second
lock, so even a double-fire cannot start a port war. Only ACTIONS are logged
(logs/ensure_capture.log) — no heartbeat spam.

Registered by:
  schtasks /Create /TN TradebotEnsureCapture /SC MINUTE /MO 5 /F
           /TR "<repo>\\.venv\\Scripts\\pythonw.exe <repo>\\ensure_capture.py"
"""
from __future__ import annotations

import datetime
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from core.constants import IST
from core.market_calendar import is_trading_day
from core.capture_role import is_capture_host

PY = HERE / ".venv" / "Scripts" / "python.exe"
LOG = HERE / "logs" / "ensure_capture.log"
WINDOW_START = datetime.time(8, 55)     # pre-open: token check + WS connect lead
WINDOW_END = datetime.time(15, 35)      # a start after this captures nothing


def _log(msg: str) -> None:
    line = f"[{datetime.datetime.now(tz=IST):%Y-%m-%d %H:%M:%S}] {msg}"
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _supervisor_running() -> bool:
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" "
             "-ErrorAction SilentlyContinue "
             "| ForEach-Object { \"$($_.ProcessId) $($_.CommandLine)\" }"],
            capture_output=True, text=True, timeout=20)
        return any("supervise.py" in ln for ln in (r.stdout or "").splitlines())
    except Exception:
        return True     # fail-closed: if the scan breaks, do NOT double-launch


def main() -> None:
    # This watchdog exists to keep a CAPTURER alive. On a viewer box there is nothing to
    # resurrect — and relaunching supervise.py there just churns a process every 5 minutes
    # that now refuses and exits (EXIT_NOT_CAPTURE_HOST). Silent no-op, matching the rest
    # of this script's behaviour outside its window; the loud explanation belongs to
    # supervise.py, which the operator runs by hand when they mean it.
    if not is_capture_host():
        return
    now = datetime.datetime.now(tz=IST)
    if not is_trading_day(now.date()):
        return
    if not (WINDOW_START <= now.time() <= WINDOW_END):
        return
    if _supervisor_running():
        return
    _log("No supervisor running inside the capture window — launching supervise.py")
    creation = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    subprocess.Popen(
        [str(PY), str(HERE / "supervise.py")],
        cwd=str(HERE),
        creationflags=creation,
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        env={**os.environ, "TRADEBOT_NO_BROWSER": "1"},   # auto-resurrections never pop a tab
    )
    _log("supervise.py launched (detached)")


if __name__ == "__main__":
    main()
