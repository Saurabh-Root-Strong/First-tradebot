"""
eod_sync.py — post-close archival sync: pull ALL the VM's captured data down to
this laptop and merge it into the local archive. The companion to sync_from_vm.py
(which is the live, today-only viewer sync); this one is the nightly "bring
everything home" job, meant to run automatically after 15:30 IST.

What it does (idempotent, safe to re-run):
  PER-DAY files  → additive copy (a file is fetched only if missing locally or its
                   size differs from the VM's, so other days are never clobbered):
      data/intraday/live/<date>_*.parquet   (lock-free mirrors)
      data/intraday/<date>.duckdb           (full-tick canonical store)
  ACCUMULATING stores → row-level MERGE (union, nothing lost on either side):
      data/intraday_trades.db               (paper_trades, PK trade_id)
      data/validation/footprint_ledger.parquet

The VM is the authoritative capturer (24/7), so it is a superset on capture days;
the additive copy + union-merge means the local archive only ever grows.

Run:
  python eod_sync.py            # pull + merge everything
  python eod_sync.py --quiet    # less chatter (for the scheduled task)
Auto-schedule (runs ~15:45 IST weekdays, and ASAP if the laptop was asleep then):
  setup_eod_sync.bat
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from core.constants import PROJECT_ROOT, DATA_DIR, LIVE_DIR
from sync_from_vm import DEF_HOST, DEF_KEY            # reuse VM connection config

REMOTE_ROOT = os.environ.get("TRADEBOT_VM_ROOT", "~/tradebot")
_INTRADAY = DATA_DIR / "intraday"
_VALID    = DATA_DIR / "validation"


def _ssh(host: str, key: str, cmd: str, timeout: int = 60) -> str:
    r = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=20", "-o", "StrictHostKeyChecking=accept-new",
         "-i", key, host, cmd],
        capture_output=True, text=True, timeout=timeout)
    return r.stdout if r.returncode == 0 else ""


def _scp(host: str, key: str, remote_path: str, local_path: Path, timeout: int = 300) -> bool:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["scp", "-q", "-o", "ConnectTimeout=20", "-o", "StrictHostKeyChecking=accept-new",
         "-i", key, f"{host}:{remote_path}", str(local_path)],
        capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        print(f"    scp failed {remote_path}: {(r.stderr or '').strip()}", file=sys.stderr)
    return r.returncode == 0


def _remote_listing(host: str, key: str, *globs: str) -> dict[str, int]:
    """{remote_path: size_bytes} for the given remote globs (missing globs ignored)."""
    cmd = "ls -la --time-style=+%s " + " ".join(globs) + " 2>/dev/null"
    out = _ssh(host, key, cmd)
    files: dict[str, int] = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 7 or parts[0].startswith("d"):
            continue
        try:
            size = int(parts[4])
        except ValueError:
            continue
        files[parts[-1]] = size            # last field = path (no spaces in our names)
    return files


def _sync_perday(host: str, key: str, remote_glob: str, local_dir: Path,
                 quiet: bool) -> tuple[int, int]:
    """Additive copy: fetch each remote file only if local missing or size differs."""
    remote = _remote_listing(host, key, remote_glob)
    fetched = skipped = 0
    for rpath, rsize in sorted(remote.items()):
        lpath = local_dir / Path(rpath).name
        if lpath.exists() and lpath.stat().st_size == rsize:
            skipped += 1
            continue
        if _scp(host, key, rpath, lpath):
            fetched += 1
            if not quiet:
                print(f"    + {lpath.name}  ({rsize/1e6:.1f} MB)")
    return fetched, skipped


def _merge_trades(host: str, key: str, quiet: bool) -> int:
    """Union the VM's paper_trades into the local SQLite (INSERT OR IGNORE by PK)."""
    import sqlite3
    local_db = DATA_DIR / "intraday_trades.db"
    remote_db = f"{REMOTE_ROOT}/data/intraday_trades.db"
    tmp = Path(tempfile.mkdtemp()) / "vm_trades.db"
    if not _scp(host, key, remote_db, tmp):
        return 0
    local_db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(local_db), timeout=10.0)
    try:
        # Ensure the local table exists with the VM's schema, then union rows.
        con.execute("ATTACH DATABASE ? AS vm", (str(tmp),))
        has = con.execute("SELECT name FROM vm.sqlite_master "
                          "WHERE type='table' AND name='paper_trades'").fetchone()
        if not has:
            return 0
        ddl = con.execute("SELECT sql FROM vm.sqlite_master "
                          "WHERE name='paper_trades'").fetchone()[0]
        con.execute(ddl.replace("CREATE TABLE", "CREATE TABLE IF NOT EXISTS"))
        before = con.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0]
        con.execute("INSERT OR IGNORE INTO paper_trades SELECT * FROM vm.paper_trades")
        after = con.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0]
        con.commit()
        added = after - before
        if not quiet:
            print(f"    paper_trades: +{added} new (local now {after})")
        return added
    except Exception as exc:
        print(f"    trades merge skipped: {exc}", file=sys.stderr)
        return 0
    finally:
        con.close()


def _merge_ledger(host: str, key: str, quiet: bool) -> int:
    """Union the VM's footprint validation ledger (dedup identical rows)."""
    import pandas as pd
    local = _VALID / "footprint_ledger.parquet"
    remote = f"{REMOTE_ROOT}/data/validation/footprint_ledger.parquet"
    tmp = Path(tempfile.mkdtemp()) / "vm_ledger.parquet"
    if not _scp(host, key, remote, tmp):
        return 0
    try:
        vm_df = pd.read_parquet(tmp)
        if local.exists():
            loc_df = pd.read_parquet(local)
            before = len(loc_df)
            merged = (pd.concat([loc_df, vm_df], ignore_index=True)
                      .drop_duplicates().reset_index(drop=True))
        else:
            before, merged = 0, vm_df.drop_duplicates().reset_index(drop=True)
        local.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(local, index=False)
        added = len(merged) - before
        if not quiet:
            print(f"    footprint_ledger: +{added} new (local now {len(merged)})")
        return added
    except Exception as exc:
        print(f"    ledger merge skipped: {exc}", file=sys.stderr)
        return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="post-close archival sync from the VM")
    ap.add_argument("--host", default=DEF_HOST)
    ap.add_argument("--key",  default=DEF_KEY)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    q = args.quiet

    if not Path(args.key).exists():
        print(f"SSH key not found: {args.key}", file=sys.stderr)
        sys.exit(1)

    print(f"eod_sync  {args.host}:{REMOTE_ROOT}  ->  {PROJECT_ROOT}")
    print("  [1/4] per-day mirrors (live/*.parquet)")
    mf, ms = _sync_perday(args.host, args.key,
                          f"{REMOTE_ROOT}/data/intraday/live/*.parquet", LIVE_DIR, q)
    print(f"        fetched {mf}, up-to-date {ms}")
    print("  [2/4] per-day duckdb (intraday/*.duckdb)")
    df, ds = _sync_perday(args.host, args.key,
                          f"{REMOTE_ROOT}/data/intraday/*.duckdb", _INTRADAY, q)
    print(f"        fetched {df}, up-to-date {ds}")
    print("  [3/4] merge paper-trades")
    _merge_trades(args.host, args.key, q)
    print("  [4/4] merge footprint ledger")
    _merge_ledger(args.host, args.key, q)
    print("eod_sync done.")


if __name__ == "__main__":
    main()
