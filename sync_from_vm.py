"""
sync_from_vm.py — pull the VM's full-session intraday mirrors down to this laptop.

The VM is the authoritative capturer (runs 24/7, captures the whole session even
with the laptop closed). This laptop is a VIEWER. This script copies the VM's
lock-free parquet mirrors (data/intraday/live/<date>_*.parquet) into the local
LIVE_DIR so a local `DASH_VIEWER=1` dashboard renders the VM's complete day.

Only the parquet mirrors are synced — they are lock-free by design (safe to copy
mid-session). The per-day DuckDB is single-writer-locked on the VM and is NOT
synced; the mirrors carry everything the dashboard reads.

Usage
  python sync_from_vm.py                 # pull today's mirrors once
  python sync_from_vm.py --date 2026-06-22
  python sync_from_vm.py --watch 60      # re-pull every 60s (live viewing)

Config (env overrides the defaults):
  TRADEBOT_VM_HOST   ubuntu@13.233.88.148
  TRADEBOT_VM_KEY    %USERPROFILE%/Downloads/tradebot-key.pem
  TRADEBOT_VM_DIR    ~/tradebot/data/intraday/live
"""
from __future__ import annotations

import argparse
import datetime
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from core.constants import IST, LIVE_DIR

DEF_HOST = os.environ.get("TRADEBOT_VM_HOST", "ubuntu@13.233.88.148")
DEF_KEY  = os.environ.get("TRADEBOT_VM_KEY",
                          str(Path(os.path.expanduser("~")) / "Downloads" / "tradebot-key.pem"))
DEF_DIR  = os.environ.get("TRADEBOT_VM_DIR", "~/tradebot/data/intraday/live")


def _today() -> str:
    return datetime.datetime.now(tz=IST).date().isoformat()


def pull(date: str, host: str, key: str, remote_dir: str) -> bool:
    """scp the VM's <date>_*.parquet mirrors down, then ATOMICALLY publish each into
    LIVE_DIR. scp writes into a staging dir first; each finished file is os.replace()'d
    onto its live path (atomic on the same volume). A reader (the dashboard) therefore
    always sees a COMPLETE old-or-new file — never the half-written / locked target that
    an in-place scp exposes (which caused 10-15s retry stalls + torn 'not full' reads).
    Returns ok."""
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    if not Path(key).exists():
        print(f"  SSH key not found: {key}", file=sys.stderr)
        return False
    # staging lives beside LIVE_DIR (same volume → os.replace is atomic) and OUTSIDE it
    # so the dashboard's mirror glob never sees a half-synced file.
    stage = LIVE_DIR.parent / f".sync_staging_{date}"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True, exist_ok=True)
    remote = f"{host}:{remote_dir}/{date}_*.parquet"
    cmd = ["scp", "-q", "-o", "ConnectTimeout=20", "-o", "StrictHostKeyChecking=accept-new",
           "-i", key, remote, str(stage)]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except Exception as exc:
        print(f"  scp failed: {exc}", file=sys.stderr)
        shutil.rmtree(stage, ignore_errors=True)
        return False
    if r.returncode != 0:
        # No matching files yet (pre-open) is not a hard error — report softly.
        msg = (r.stderr or "").strip()
        print(f"  scp rc={r.returncode} {msg}", file=sys.stderr)
        shutil.rmtree(stage, ignore_errors=True)
        return False
    # atomically publish: os.replace each staged file onto its live path. Retry on the
    # brief PermissionError a reader can cause by having the old file open at that instant.
    published, mb = 0, 0.0
    for src in sorted(stage.glob(f"{date}_*.parquet")):
        dst = LIVE_DIR / src.name
        for _ in range(6):
            try:
                os.replace(src, dst)
                published += 1
                mb += dst.stat().st_size / 1e6
                break
            except PermissionError:
                time.sleep(0.2)
        else:
            try:                                   # last resort so we don't drop the update
                shutil.copy2(src, dst)
                published += 1
                mb += dst.stat().st_size / 1e6
            except Exception as exc:
                print(f"  publish failed for {src.name}: {exc}", file=sys.stderr)
    shutil.rmtree(stage, ignore_errors=True)
    print(f"  synced {published} mirrors ({mb:.1f} MB) for {date} in {time.time()-t0:.1f}s")
    return published > 0


def main() -> None:
    ap = argparse.ArgumentParser(description="pull VM intraday mirrors to local")
    ap.add_argument("--date", default=_today(), help="ISO date (default: today IST)")
    ap.add_argument("--host", default=DEF_HOST)
    ap.add_argument("--key",  default=DEF_KEY)
    ap.add_argument("--remote-dir", default=DEF_DIR)
    ap.add_argument("--watch", type=int, default=0, metavar="SEC",
                    help="re-pull every SEC seconds (0 = once)")
    args = ap.parse_args()

    print(f"sync_from_vm  {args.host}:{args.remote_dir}/{args.date}_*.parquet  ->  {LIVE_DIR}")
    ok = pull(args.date, args.host, args.key, args.remote_dir)
    if not args.watch:
        sys.exit(0 if ok else 1)
    print(f"  watching - re-pull every {args.watch}s (Ctrl+C to stop)")
    try:
        while True:
            time.sleep(args.watch)
            pull(args.date, args.host, args.key, args.remote_dir)
    except KeyboardInterrupt:
        print("\n  stopped.")


if __name__ == "__main__":
    main()
