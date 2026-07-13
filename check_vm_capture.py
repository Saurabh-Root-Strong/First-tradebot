"""
check_vm_capture.py — is the CAPTURE VM actually capturing right now?

THE GAP THIS CLOSES. Every monitor in this system watches signals or code: the DEGRADED
badge (swallowed exceptions), WRITE ERR (db failures), verify_nse_calendar (constant drift),
audit_signals --check (bias drift). NOTHING watched the most basic thing of all — "is data
being captured AT ALL?" On 2026-07-13 the VM ran a whole weekend on an EXPIRED token and
would have captured ZERO for the entire session, silently, because the morning-token task was
skipped (laptop asleep, no catch-up) and the dashboard's auto-auth refreshes the LOCAL token
without pushing it to the VM.

Checks, in order (fails fast, never hangs):
  1. LOCAL token usable?      -> if not, nothing can be fixed from here (go log in).
  2. VM reachable?            -> ssh with a hard timeout.
  3. VM token usable?         -> the silent killer: an expired token captures nothing.
  4. Ticks flowing?           -> only asserted DURING market hours on a trading day;
                                 outside them a quiet feed is correct, not a fault.

  --fix   push the local token to the VM, restart capture, re-verify (only ever pushes a
          token that is VALID locally — never replaces a good token with a dead one).

exit 0 = healthy | 1 = unhealthy (fixable: run with --fix) | 2 = unreachable / no local token
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from core.constants import IST
from core.market_calendar import is_trading_day

VM = os.environ.get("TRADEBOT_VM_HOST", "ubuntu@13.233.88.148")
KEY = os.environ.get("TRADEBOT_VM_KEY",
                     str(Path(os.path.expanduser("~")) / "Downloads" / "tradebot-key.pem"))
TOKEN = Path("access_token.txt")
STALE_SEC = 180          # ticks older than this DURING market hours = capture is down


def _ssh(cmd: str, timeout: int = 40):
    """Run a command on the VM. Returns (rc, stdout). rc=-1 on timeout/transport error."""
    try:
        r = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=15", "-o", "StrictHostKeyChecking=accept-new",
             "-o", "BatchMode=yes", "-i", KEY, VM, cmd],
            capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout or "").strip()
    except Exception as exc:
        return -1, f"{type(exc).__name__}: {exc}"


def _market_open(now) -> bool:
    return (is_trading_day(now.date())
            and dt.time(9, 15) <= now.time() <= dt.time(15, 30))


def _local_token_ok() -> bool:
    try:
        from tradebot.adapters.broker.token import describe, is_usable
        raw = TOKEN.read_text().strip() if TOKEN.exists() else ""
        if not raw:
            print("  LOCAL token : MISSING")
            return False
        ok = is_usable(raw)
        print(f"  LOCAL token : {'OK' if ok else 'EXPIRED'}  ({describe(raw)})")
        return bool(ok)
    except Exception as exc:
        print(f"  LOCAL token : check failed ({exc})")
        return False


# One python snippet on the VM answers both questions in a single ssh round-trip.
_VM_PROBE = r'''cd ~/tradebot && docker compose exec -T tradebot python -c "
import datetime, json
from pathlib import Path
out = {}
try:
    from tradebot.adapters.broker.token import is_usable, describe
    raw = Path('access_token.txt').read_text().strip()
    out['token_ok'] = bool(is_usable(raw)); out['token'] = describe(raw)
except Exception as e:
    out['token_ok'] = False; out['token'] = 'unreadable: %s' % e
try:
    from core.mirror_io import read_mirror
    from core.constants import IST
    now = datetime.datetime.now(IST)
    df = read_mirror('ticks', now.date().isoformat())
    if df is None or not len(df):
        out['tick_age'] = None; out['rows'] = 0
    else:
        out['rows'] = int(len(df))
        out['tick_age'] = float((now - df['ts'].max()).total_seconds())
except Exception as e:
    out['tick_age'] = None; out['rows'] = 0; out['err'] = str(e)
print(json.dumps(out))
"'''


def probe():
    rc, out = _ssh(_VM_PROBE, timeout=60)
    if rc != 0:
        return None, out
    import json
    for line in reversed(out.splitlines()):          # last line = our json
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line), out
            except Exception:
                continue
    return None, out


def fix() -> bool:
    """Push the local token + restart capture + re-verify. Only called with a VALID local token."""
    print("\n  FIXING — pushing local token to the VM...")
    r = subprocess.run(["scp", "-o", "ConnectTimeout=15", "-o", "StrictHostKeyChecking=accept-new",
                        "-i", KEY, str(TOKEN), f"{VM}:/home/ubuntu/tradebot/access_token.txt"],
                       capture_output=True, text=True, timeout=90)
    if r.returncode != 0:
        print(f"  scp FAILED: {(r.stderr or '').strip()}")
        return False
    print("  token uploaded. restarting capture...")
    rc, out = _ssh("cd ~/tradebot && docker compose restart tradebot", timeout=180)
    if rc != 0:
        print(f"  restart FAILED: {out}")
        return False
    print("  restarted. verifying...")
    import time
    time.sleep(30)
    rc, out = _ssh("cd ~/tradebot && docker compose logs --since 90s tradebot 2>&1 "
                   "| grep -E 'indices live|CAPTURER' | tail -2", timeout=60)
    print("  " + (out.replace("\n", "\n  ") if out else "(no log lines yet)"))
    return "indices live" in out or "CAPTURER" in out


def main() -> None:
    ap = argparse.ArgumentParser(description="verify (and optionally heal) VM capture")
    ap.add_argument("--fix", action="store_true",
                    help="push the local token + restart capture if the VM is unhealthy")
    args = ap.parse_args()

    now = dt.datetime.now(IST)
    live = _market_open(now)
    print("=" * 72)
    print(f"VM CAPTURE HEALTH — {now:%Y-%m-%d %H:%M} IST  "
          f"({'MARKET OPEN' if live else 'market closed'})")
    print("=" * 72)

    local_ok = _local_token_ok()

    p, raw = probe()
    if p is None:
        print(f"  VM          : UNREACHABLE ({raw[:120]})")
        print("\nRESULT: cannot verify the VM. Check network / VM / ssh key.")
        sys.exit(2)

    print(f"  VM token    : {'OK' if p.get('token_ok') else 'EXPIRED/BAD'}  ({p.get('token')})")
    age, rows = p.get("tick_age"), p.get("rows", 0)
    if age is None:
        print(f"  VM ticks    : NONE today (rows=0)")
    else:
        print(f"  VM ticks    : {rows:,} rows, newest {age:.0f}s ago")

    problems = []
    ticks_dead = age is None or age > STALE_SEC
    if not p.get("token_ok"):
        # A bad token FILE does not kill a running WS immediately -- the live session holds the
        # old token in memory. It kills capture on the next RESTART/reconnect. So distinguish
        # the ACTIVE failure (no ticks) from the LATENT one (ticks still flowing) instead of
        # crying "capturing NOTHING" while data is visibly landing.
        problems.append("VM token is expired -> capturing NOTHING" if ticks_dead else
                        "VM token is expired -> ticks still flowing on the in-memory session, "
                        "but capture DIES on the next restart/reconnect (latent)")
    if live and ticks_dead:
        problems.append(f"market is OPEN but ticks are "
                        f"{'absent' if age is None else f'{age:.0f}s stale'} -> capture is DOWN")

    print("-" * 72)
    if not problems:
        print("HEALTHY — the VM is capturing." if live else
              "HEALTHY — VM token valid (market closed, a quiet feed is expected).")
        sys.exit(0)

    for x in problems:
        print(f"  ✗ {x}")
    if not args.fix:
        print("\nRESULT: UNHEALTHY. Re-run with --fix to push the local token and restart capture.")
        sys.exit(1)
    if not local_ok:
        print("\nCannot fix: the LOCAL token is also dead. Run morning_token.bat and log in first.")
        sys.exit(2)
    ok = fix()
    print("\nRESULT: " + ("FIXED — capture is live again." if ok else
                          "FIX FAILED — inspect the VM manually."))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
