"""
morning_capture_alarm.py — a NOISY watchdog so a dead Fyers token never silently eats a
session again (2026-07-24: token expired 06:00, nobody noticed till 14:00 → ~5h of tick+chain
data lost forever).

Runs the existing check_vm_capture health probe. If capture is DOWN (expired token / stale
ticks during market hours) it raises a Windows toast + a foreground message box so you SEE it
the instant you're at the laptop — then you refresh the token and it self-heals via --fix.

It does NOT auto-fix: the token needs your Fyers login (headless TOTP is anti-bot-blocked), so
this is an ALARM, not a repair. Scheduled daily 09:20 IST + at-logon (the logon trigger is the
one that matters — it screams the moment you open a laptop that was shut at 09:20).

Zero market impact: read-only health probe, laptop-only, exits in seconds off-session.

    .venv\\Scripts\\python.exe morning_capture_alarm.py           # check, alarm if down
    .venv\\Scripts\\python.exe morning_capture_alarm.py --test    # force the alarm UI
"""
from __future__ import annotations

import datetime
import subprocess
import sys
from pathlib import Path

from core.constants import IST
from core.market_calendar import is_trading_day

ROOT = Path(__file__).resolve().parent
PY = ROOT / ".venv" / "Scripts" / "python.exe"
LOG = ROOT / "logs" / "morning_capture_alarm.log"


def _log(msg: str):
    LOG.parent.mkdir(exist_ok=True)
    ts = datetime.datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"{ts}  {msg}\n")


def _alarm(title: str, body: str):
    """Windows toast + a blocking message box (PowerShell, no third-party deps). Best-effort —
    a headless/locked session may only get the toast; the box waits for the next unlock."""
    ps = f'''
$ErrorActionPreference='SilentlyContinue'
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType=WindowsRuntime] | Out-Null
$t=[Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02)
$x=$t.GetElementsByTagName('text'); $x.Item(0).AppendChild($t.CreateTextNode('{title}'))|Out-Null
$x.Item(1).AppendChild($t.CreateTextNode('{body}'))|Out-Null
$n=[Windows.UI.Notifications.ToastNotification]::new($t)
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('Tradebot').Show($n)
Add-Type -AssemblyName PresentationFramework
[System.Windows.MessageBox]::Show('{body}','{title}','OK','Warning')|Out-Null
'''
    try:
        subprocess.run(["powershell", "-NoProfile", "-Command", ps], timeout=120)
    except Exception as exc:
        _log(f"alarm UI failed: {exc}")


def main():
    now = datetime.datetime.now(IST)
    if "--test" in sys.argv:
        _alarm("Tradebot CAPTURE DOWN",
               "TEST — token/capture check. Run check_vm_capture.py --fix after refreshing token.")
        _log("test alarm fired")
        return
    # off a trading day, nothing to guard
    if not is_trading_day(now.date()):
        _log("non-trading day — skip")
        return
    # before ~09:15 the morning token may legitimately not be pushed yet; only alarm from 09:20
    if now.time() < datetime.time(9, 20):
        _log(f"pre-09:20 ({now.time():%H:%M}) — skip")
        return
    r = subprocess.run([str(PY), str(ROOT / "check_vm_capture.py")],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=180)
    healthy = r.returncode == 0
    tail = (r.stdout or "").strip().splitlines()[-3:]
    _log(("HEALTHY" if healthy else "UNHEALTHY") + " | " + " · ".join(tail))
    if not healthy:
        _alarm("⚠ Tradebot CAPTURE DOWN",
               "Fyers token/capture is DOWN and the market is open. Refresh the token now, "
               "then run:  python check_vm_capture.py --fix   (every minute lost is lost forever)")


if __name__ == "__main__":
    main()
