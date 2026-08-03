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

Then, LAST, it PURGES the VM of any day BEFORE today whose files are already verified
present locally (size match) — so the tiny t3.micro only ever holds the live (today's)
capture and never fills up. Today's still-capturing files are never touched. This makes
the laptop the permanent archive and keeps the VM lean. Disable with --no-purge.
NOTE: purging old DAYS frees DISK; it does not change the live process's RAM use.

Run:
  python eod_sync.py            # pull + merge everything, then purge archived days off the VM
  python eod_sync.py --quiet    # less chatter (for the scheduled task)
  python eod_sync.py --no-purge # pull + merge but leave the VM's copies in place
Auto-schedule (runs ~15:45 IST weekdays, and ASAP if the laptop was asleep then):
  setup_eod_sync.bat
"""
from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from core.constants import PROJECT_ROOT, DATA_DIR, LIVE_DIR, IST
from core.market_calendar import is_trading_day, holiday_name
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


def _push_calibration(host: str, key: str, quiet: bool) -> int:
    """Ship the freshly-learned calibration ledgers TO the VM.

    WHY THIS EXISTS (the VM was running blind). eod_sync's LEARN step runs on the LAPTOP and
    writes data/calibration/*.json there. data/ is gitignored, so a deploy never carries them
    and the VM cannot generate them itself (it has no archive to measure). Checked 2026-08-03:
    the VM had NO band_coverage.json at all, so hour_forecast.band_coverage returned
    conf='none' and the VM board showed no coverage tag next to the band whatsoever.

    Same failure class as the historical 5min store: silently missing, board looks fine.
    Hand-copying fixes it for a week and then rots, so it belongs in the nightly loop.

    ROUTE: the VM's data/ is a ROOT-OWNED bind mount, so `ubuntu` cannot scp into it
    directly (Permission denied). Land in /tmp, then `docker compose cp` as root — which
    writes through to the bind mount and survives container rebuilds.
    """
    src = DATA_DIR / "calibration"
    files = [p for p in (src / "band_coverage.json", src / "band_multipliers.json")
             if p.exists()]
    if not files:
        if not quiet:
            print("        no local calibration ledgers to push (run the LEARN step first)")
        return 0
    sent = 0
    for p in files:
        r = subprocess.run(
            ["scp", "-q", "-o", "ConnectTimeout=20", "-o", "StrictHostKeyChecking=accept-new",
             "-i", key, str(p), f"{host}:/tmp/{p.name}"],
            capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            print(f"    scp -> VM failed {p.name}: {(r.stderr or '').strip()}", file=sys.stderr)
            continue
        # ATOMIC: land under a .tmp name inside the container, then rename. `docker compose cp`
        # streams the file, so a live dashboard reading band_coverage.json mid-copy would hit
        # a truncated document — hour_forecast swallows the JSON error and silently drops to
        # cover=None (no tag on the board) until the next mtime change. A same-directory mv is
        # atomic, so a reader sees either the whole old file or the whole new one.
        out = _ssh(host, key,
                   f"cd {REMOTE_ROOT} && docker compose exec -T tradebot "
                   f"mkdir -p /app/data/calibration && "
                   f"docker compose cp /tmp/{p.name} tradebot:/app/data/calibration/{p.name}.tmp "
                   f"&& docker compose exec -T tradebot "
                   f"mv /app/data/calibration/{p.name}.tmp /app/data/calibration/{p.name} "
                   f"&& rm -f /tmp/{p.name} && echo OK", timeout=120)
        if "OK" in out:
            sent += 1
            if not quiet:
                print(f"    + VM calibration/{p.name}  ({p.stat().st_size/1024:.1f} KB)")
        else:
            print(f"    VM copy failed for {p.name} (container down?)", file=sys.stderr)
    print(f"        pushed {sent}/{len(files)} calibration ledger(s) to the VM")
    return sent


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


def _day_stem(name: str) -> str | None:
    """Extract the YYYY-MM-DD date from a per-day filename, else None.
    Handles '<date>_<table>.parquet' and '<date>.duckdb'."""
    stem = name.split("_")[0] if name.endswith(".parquet") else name.rsplit(".", 1)[0]
    try:
        datetime.date.fromisoformat(stem)
        return stem
    except ValueError:
        return None


def _purge_vm(host: str, key: str, quiet: bool, today: str) -> int:
    """Delete from the VM every per-day file that is (a) for a day BEFORE today and
    (b) already byte-for-byte present locally. This is what keeps the t3.micro lean:
    the laptop is the permanent archive, the VM only ever holds the live (today's)
    capture. SAFETY: today's still-capturing files are NEVER touched, and a file is
    removed only when the local copy's size matches the VM's (verified archived)."""
    remote = {}
    remote.update(_remote_listing(host, key, f"{REMOTE_ROOT}/data/intraday/live/*.parquet"))
    remote.update(_remote_listing(host, key, f"{REMOTE_ROOT}/data/intraday/*.duckdb"))
    targets: list[tuple[str, int]] = []
    for rpath, rsize in remote.items():
        name = Path(rpath).name
        stem = _day_stem(name)
        if not stem or stem >= today:           # skip non-dated + today/future (live)
            continue
        lpath = (LIVE_DIR / name) if name.endswith(".parquet") else (_INTRADAY / name)
        if lpath.exists() and lpath.stat().st_size == rsize:   # verified archived locally
            targets.append((rpath, rsize))
    if not targets:
        if not quiet:
            print("        nothing to purge (VM holds only today's live files / nothing verified yet)")
        return 0
    # one ssh call deletes them all
    _ssh(host, key, "rm -f " + " ".join(f"'{p}'" for p, _ in targets))
    freed = sum(s for _, s in targets)
    if not quiet:
        for p, s in sorted(targets):
            print(f"    - VM rm {Path(p).name}  ({s/1e6:.1f} MB)")
    print(f"        purged {len(targets)} VM file(s), freed {freed/1e6:.1f} MB")
    return freed


def main() -> None:
    ap = argparse.ArgumentParser(description="post-close archival sync from the VM")
    ap.add_argument("--host", default=DEF_HOST)
    ap.add_argument("--key",  default=DEF_KEY)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--no-purge", action="store_true",
                    help="skip deleting already-archived days from the VM (keep VM copies)")
    ap.add_argument("--force", action="store_true",
                    help="run even on a non-trading day (NSE holiday / weekend)")
    args = ap.parse_args()
    q = args.quiet
    _today = datetime.datetime.now(tz=IST).date()
    if not args.force and not is_trading_day(_today):
        why = holiday_name(_today) or "weekend"
        print(f"eod_sync: {_today} is not an NSE trading day ({why}) — "
              f"no session to archive. Use --force to run anyway.")
        return
    today = _today.isoformat()

    if not Path(args.key).exists():
        print(f"SSH key not found: {args.key}", file=sys.stderr)
        sys.exit(1)

    print(f"eod_sync  {args.host}:{REMOTE_ROOT}  ->  {PROJECT_ROOT}")
    print("  [1/6] per-day mirrors (live/*.parquet)")
    mf, ms = _sync_perday(args.host, args.key,
                          f"{REMOTE_ROOT}/data/intraday/live/*.parquet", LIVE_DIR, q)
    print(f"        fetched {mf}, up-to-date {ms}")
    print("  [2/6] per-day duckdb (intraday/*.duckdb)")
    df, ds = _sync_perday(args.host, args.key,
                          f"{REMOTE_ROOT}/data/intraday/*.duckdb", _INTRADAY, q)
    print(f"        fetched {df}, up-to-date {ds}")
    print("  [3/6] merge paper-trades")
    _merge_trades(args.host, args.key, q)
    print("  [4/6] merge footprint ledger")
    _merge_ledger(args.host, args.key, q)
    print("  [5/6] refresh hour-forecast ledger (today's labeled rows)")
    try:
        import backtest_hour_forecast as hfb
        new = hfb.harvest(hfb._captured_days())
        if not new.empty:
            comb = hfb._upsert(new)
            print(f"        ledger now {len(comb)} rows / {comb['date'].nunique()} days")
        else:
            print("        nothing to harvest yet")
    except Exception as exc:
        print(f"        skipped: {exc}")
    print("  [6/7] LEARN — refresh band coverage + recalibrate multipliers (L4 loop)")
    try:
        import backtest_band_horizon as bh
        import calibration_engine as cal
        # Measure every horizon the UI can ask for. Was [15, 60] only, so a 5m or 30m
        # selection silently borrowed the 15m cell and printed it as verified (n=612).
        # 240m is deliberately absent: LAST_PRED is 15:00, so a 4h window never resolves
        # inside the session and the cell would be permanently empty.
        bh.run(30, [5, 15, 30, 60, 120])  # regenerate data/calibration/band_coverage.json
        cal.learn()                       # shrink-recalibrate band_multipliers.json
    except Exception as exc:
        print(f"        skipped: {exc}")
    # LEARN runs here, on the laptop — so ship the result to the VM, which cannot compute it
    # and (checked 2026-08-03) had no coverage ledger at all.
    print("  [6b/7] PUSH calibration ledgers -> VM")
    try:
        _push_calibration(args.host, args.key, q)
    except Exception as exc:
        print(f"        skipped: {exc}")
    # Purge LAST — only after every fetch/merge above has safely landed locally.
    if args.no_purge:
        print("  [7/7] purge VM (skipped: --no-purge)")
    else:
        print(f"  [7/7] purge VM of already-archived days (< {today})")
        _purge_vm(args.host, args.key, q, today)
    print("eod_sync done.")


if __name__ == "__main__":
    main()
