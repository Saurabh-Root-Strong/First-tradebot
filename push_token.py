"""
push_token.py — ship the local access_token.txt to the cloud VM.

WHY THIS EXISTS. Fyers tokens die at a fixed ~06:00-IST cutoff and the VM cannot
mint its own: `fyers_auth_headless.py` hits Fyers' internal vagator endpoints, which
now answer HTTP 404 (anti-bot). So the ONLY way the VM gets a token is an upload from
the laptop that ran the browser login. That upload used to live in exactly one place —
step [2/4] of morning_token.bat — and on 2026-08-28 the login ran while the upload did
not, so the VM sat in its "Cannot restart without a valid token" loop and captured
NOTHING from 09:15 to 09:46. A fresh LOCAL token is not the same as a fresh VM token,
and nothing made the two happen together.

Now fyers_auth.py calls push() the moment it writes the token, so minting and shipping
are ONE action. This module is also runnable on its own to retry an upload:

    .venv\\Scripts\\python.exe push_token.py           # upload (+ verify)
    .venv\\Scripts\\python.exe push_token.py --restart  # also bounce the VM container

NO RESTART BY DEFAULT, on purpose. supervise.py re-reads access_token.txt from disk on
every restart attempt (ensure_token → token_usable → read_token) and its dead-token
branch retries every 60s, so a bare upload is picked up within a minute with no lost
ticks. A restart during a HEALTHY session would drop the WebSocket and punch a hole in
the capture — the exact damage this script exists to prevent. Use --restart only when
the VM is known stuck.

Env overrides (same names sync_from_vm.py / check_vm_capture.py already use):
    TRADEBOT_VM_HOST        ubuntu@13.233.88.148
    TRADEBOT_VM_KEY         %USERPROFILE%/Downloads/tradebot-key.pem
    TRADEBOT_VM_TOKEN_PATH  /home/ubuntu/tradebot/access_token.txt
    TRADEBOT_NO_TOKEN_PUSH  set to 1 to disable the auto-push entirely
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tradebot.adapters.broker.token import TOKEN_FILE, describe, is_usable

HOST = os.environ.get("TRADEBOT_VM_HOST", "ubuntu@13.233.88.148")
KEY = os.environ.get("TRADEBOT_VM_KEY",
                     str(Path(os.path.expanduser("~")) / "Downloads" / "tradebot-key.pem"))
REMOTE = os.environ.get("TRADEBOT_VM_TOKEN_PATH", "/home/ubuntu/tradebot/access_token.txt")

# ssh/scp flags shared by every call: key-only (BatchMode → never sit on a password
# prompt inside an unattended morning script), bounded connect time, and accept-new so
# a first-ever connect to a rebuilt VM does not hang on the host-key question.
_SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
             "-o", "StrictHostKeyChecking=accept-new"]


def _run(cmd: list, timeout: int) -> tuple:
    """(rc, output). rc=-1 on timeout/transport error — never raises, because this
    whole module is best-effort: a network hiccup must not fail a token that is
    already safely written to disk."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, ((r.stdout or "") + (r.stderr or "")).strip()
    except Exception as exc:
        return -1, f"{type(exc).__name__}: {exc}"


def _local_sha() -> str:
    return hashlib.sha256(TOKEN_FILE.read_bytes()).hexdigest()


def push(restart: bool = False, verbose: bool = True) -> bool:
    """Upload access_token.txt to the VM and verify it landed byte-identical.

    Returns True only on a VERIFIED upload. Skips (returns False) when there is
    nothing sensible to push — no token, a dead token, no SSH key, or when running
    ON the VM itself. Never raises."""
    def say(msg: str) -> None:
        if verbose:
            print(msg)

    if os.environ.get("TRADEBOT_NO_TOKEN_PUSH") == "1":
        say("  VM push: disabled (TRADEBOT_NO_TOKEN_PUSH=1)")
        return False

    # The VM runs with FYERS_HEADLESS=1. Without this guard the VM's own supervisor
    # would try to scp the token to itself on every auth attempt.
    if os.environ.get("FYERS_HEADLESS") == "1":
        say("  VM push: skipped (running ON the VM)")
        return False

    if not TOKEN_FILE.exists():
        say(f"  VM push: skipped — no {TOKEN_FILE.name} to push")
        return False

    # Never ship a dead token. It cannot revive the VM (supervise rejects it) and it
    # overwrites the remote file, destroying the evidence of what the VM last had.
    try:
        raw = TOKEN_FILE.read_text(encoding="utf-8").strip()
        if not is_usable(raw):
            # describe(raw), never describe() — the no-arg form re-reads the adapter's
            # own path, which would report a DIFFERENT (healthy) token than the one
            # being refused and make the refusal read like a bug.
            say(f"  VM push: REFUSED — local token is not usable ({describe(raw)})")
            return False
    except Exception:
        say("  VM push: REFUSED — local token is unreadable/malformed")
        return False

    if not Path(KEY).exists():
        say(f"  VM push: skipped — SSH key not found at {KEY}")
        return False

    rc, out = _run(["scp", *_SSH_OPTS, "-i", KEY, str(TOKEN_FILE), f"{HOST}:{REMOTE}"], timeout=90)
    if rc != 0:
        say(f"  VM push FAILED (scp rc={rc}): {out}")
        say(f"     retry with:  .venv\\Scripts\\python.exe push_token.py")
        return False

    # VERIFY. scp has exited 0 on a truncated write before (and a silently-wrong token
    # on the VM is indistinguishable from a missing one until the market opens), so
    # compare digests rather than trusting the exit code.
    rc, out = _run(["ssh", *_SSH_OPTS, "-i", KEY, HOST, f"sha256sum {REMOTE}"], timeout=60)
    remote_sha = out.split()[0] if rc == 0 and out else ""
    if remote_sha != _local_sha():
        say(f"  VM push UNVERIFIED — remote digest {remote_sha[:12] or '?'} != "
            f"local {_local_sha()[:12]}. Re-run push_token.py.")
        return False

    say(f"  VM push OK → {HOST}:{REMOTE} (verified)")

    if restart:
        rc, out = _run(["ssh", *_SSH_OPTS, "-i", KEY, HOST,
                        "cd ~/tradebot && docker compose restart tradebot"], timeout=180)
        say("  VM capture restarted" if rc == 0 else f"  VM restart FAILED (rc={rc}): {out}")
    else:
        # Not a guess: supervise.py's dead-token branch sleeps 60s, then calls
        # ensure_token() again, which re-reads the file we just wrote.
        say("  VM supervisor picks it up within 60s (no restart needed)")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Upload the local Fyers token to the VM.")
    ap.add_argument("--restart", action="store_true",
                    help="also restart the VM container (only when it is stuck; a restart "
                         "during a healthy session drops ticks)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    if not a.quiet:
        print(f"  local token : {describe()}")
    return 0 if push(restart=a.restart, verbose=not a.quiet) else 1


if __name__ == "__main__":
    sys.exit(main())
