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
    """scp the VM's <date>_*.parquet mirrors into the local LIVE_DIR. Returns ok."""
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    if not Path(key).exists():
        print(f"  SSH key not found: {key}", file=sys.stderr)
        return False
    remote = f"{host}:{remote_dir}/{date}_*.parquet"
    cmd = ["scp", "-q", "-o", "ConnectTimeout=20", "-o", "StrictHostKeyChecking=accept-new",
           "-i", key, remote, str(LIVE_DIR)]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except Exception as exc:
        print(f"  scp failed: {exc}", file=sys.stderr)
        return False
    if r.returncode != 0:
        # No matching files yet (pre-open) is not a hard error — report softly.
        msg = (r.stderr or "").strip()
        print(f"  scp rc={r.returncode} {msg}", file=sys.stderr)
        return False
    pulled = sorted(p.name for p in LIVE_DIR.glob(f"{date}_*.parquet"))
    mb = sum(p.stat().st_size for p in LIVE_DIR.glob(f"{date}_*.parquet")) / 1e6
    print(f"  synced {len(pulled)} mirrors ({mb:.1f} MB) for {date} in {time.time()-t0:.1f}s")
    return True


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
    print(f"  watching — re-pull every {args.watch}s (Ctrl+C to stop)")
    try:
        while True:
            time.sleep(args.watch)
            pull(args.date, args.host, args.key, args.remote_dir)
    except KeyboardInterrupt:
        print("\n  stopped.")


if __name__ == "__main__":
    main()
