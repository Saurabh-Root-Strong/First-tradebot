"""capture_role.py — is THIS process allowed to capture? One answer, one place.

The rule itself is old and unchanged (2026-07-06 role-safety, 2026-07-09 persistence
boundary): a box CAPTURES only if it is the DESIGNATED capture host — a `.capture_host`
marker beside the mirror dir, or `TRADEBOT_CAPTURE=1`. `DASH_VIEWER=1` always wins as
viewer. Anything else, including a bare `python dashboard.py` on the laptop, is a
read-only VIEWER, so it can never clobber the mirrors synced down from the VM.

WHY THIS MODULE EXISTS. The rule was written out three times — `dashboard._resolve_role`
(gates the WS + pollers), `intraday_db._is_capture_host` (gates every DB write, the
clobber guard), and, fatally, NOWHERE in `supervise.py`, which simply ASSUMED the process
it supervises is a capturer. On 2026-08-11 that assumption ran the laptop into a restart
loop for a whole session: the box correctly resolved to viewer, so `dashboard.py` never
started `_heartbeat_writer` (only the capturer branch does), so `data/ws_heartbeat.txt`
stayed 35 days stale, so `supervise.heartbeat_age()` read ~3,025,000s, so the supervisor
killed and relaunched the dashboard every ~2 minutes from the open — 09:47, 09:49, 09:51,
09:53, 09:55, 09:57 in `logs/supervisor.log`. Capture was never affected (the VM captures;
`sync_from_vm.py` feeds the laptop) but the UI died every two minutes and `_free_port_8050`
cleared the port each time.

The bug was not the rule. It was N copies of a safety rule that must agree, where one
layer thought "capturer" and another thought "viewer" about the SAME process. That is the
failure this module is meant to make structurally impossible: every caller asks here.
"""
from __future__ import annotations

import os

from core.constants import LIVE_DIR

MARKER_NAME = ".capture_host"
CAPTURER = "capturer"
VIEWER = "viewer"


def marker_path(live_dir=None):
    """The `.capture_host` marker, beside the mirror dir (data/intraday/)."""
    return (live_dir or LIVE_DIR).parent / MARKER_NAME


def is_capture_host(live_dir=None) -> bool:
    """True only for the designated capture host. SAFE BY DEFAULT — the dangerous role
    is never the fallthrough. Order matters: DASH_VIEWER is an explicit override and wins
    over both the env flag and the marker, so `dev.bat` can force a viewer on a box that
    happens to carry the marker."""
    if os.environ.get("DASH_VIEWER") == "1":
        return False
    if os.environ.get("TRADEBOT_CAPTURE") == "1":
        return True
    return marker_path(live_dir).exists()


def resolve_role(live_dir=None) -> str:
    return CAPTURER if is_capture_host(live_dir) else VIEWER


def why(live_dir=None) -> str:
    """One line naming WHICH input decided the role — so a refusal or a boot log says
    what to change, instead of leaving the operator to guess. The VM's documented deploy
    caveat (a rebuilt container without `touch data/intraday/.capture_host` fails safe to
    viewer and silently stops capturing) is exactly the case this has to spell out."""
    # ASCII-only: this string is printed by supervise.py's refusal, which is read on a
    # cp1252 Windows console and through `docker logs` on the VM.
    if os.environ.get("DASH_VIEWER") == "1":
        return "DASH_VIEWER=1 (explicit viewer override - wins over marker and env flag)"
    if os.environ.get("TRADEBOT_CAPTURE") == "1":
        return "TRADEBOT_CAPTURE=1 (capture forced by environment)"
    p = marker_path(live_dir)
    return (f"marker {p} present" if p.exists()
            else f"no marker at {p}, TRADEBOT_CAPTURE unset - safe default is viewer")
