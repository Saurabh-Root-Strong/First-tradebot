"""day_files.py — read ONE finished session's table, wherever that session is kept.

Two hosts, two layouts, and a reader that assumed only one of them:

    laptop (viewer)   data/intraday/live/<day>_<table>.parquet   ~80 days, grown by the daily sync
    VM (capturer)     data/intraday/<day>.duckdb                 93 days — but only the last 1–2
                                                                 days of parquet mirrors

range_flag and oi_flag need 10–20 PAST sessions. Reading mirrors alone they worked on the laptop
and were silent on the VM (found on the first deploy, 2026-10-10: "captured sessions: 1"). So:
the parquet mirror when it exists, else the per-day DuckDB file, opened read-only.

Past days only. TODAY is the capturer's open database (an exclusive writer) — today's rows come
from the live mirror through core.mirror_io, never from here.

Only the columns asked for are read, and nothing is cached here: callers reduce a day to a small
curve and keep that (see range_flag._load_ticks for why past tick frames must not sit in the
shared 32-slot cache of core.mirror_io).
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

MIN_DB_BYTES = 5_000_000      # a real session's database is ~50 MB; a dead capture's is a stub


def _dirs():
    from core.constants import LIVE_DIR
    return LIVE_DIR, LIVE_DIR.parent


def load(table: str, day: str, cols: list) -> "pd.DataFrame | None":
    """`table` for `day` with (the available subset of) `cols`, ts in IST, time-sorted. None if
    the session is kept nowhere readable."""
    from core.constants import IST, today_iso
    live, root = _dirs()
    df = None
    p = live / f"{day}_{table}.parquet"
    if p.exists() and p.stat().st_size >= 100:
        try:
            import pyarrow.parquet as pq
            have = set(pq.read_schema(p).names)
            df = pd.read_parquet(p, columns=[c for c in cols if c in have])
        except Exception:
            df = None
    if df is None:
        db = root / f"{day}.duckdb"
        # today_iso() is the IST session date — the VM's own clock is UTC
        if db.exists() and db.stat().st_size >= MIN_DB_BYTES and day != today_iso():
            con = None
            try:
                import duckdb
                con = duckdb.connect(str(db), read_only=True)
                have = {r[0] for r in con.execute(f"describe {table}").fetchall()}
                sel = ", ".join(c for c in cols if c in have)
                df = con.execute(f"select {sel} from {table}").fetchdf() if sel else None
            except Exception:
                df = None
            finally:
                if con is not None:
                    con.close()
    if df is None or not len(df) or "ts" not in df.columns:
        return None
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(IST)
    return df.sort_values("ts")


def days(table: str, min_parquet_bytes: int) -> list:
    """Every session `load(table, …)` could serve: a mirror of at least min_parquet_bytes, or a
    per-day database of real size. Sorted ISO dates; non-date filenames ignored."""
    live, root = _dirs()
    out = set()
    for p in live.glob(f"*_{table}.parquet"):
        if p.stat().st_size >= min_parquet_bytes:
            out.add(p.name[:10])
    for p in root.glob("*.duckdb"):
        # The database only stands in for a session that has NO mirror at all (the VM's past
        # days). Where a mirror exists it decides, including by being too small: on the laptop
        # a stub mirror is a day the VM failed to capture, and a local database beside it is the
        # viewer's own, not that session — counting those took the laptop's list from 76
        # sessions to 83, seven of them days with nothing in them.
        d = p.name[:10]
        if p.stat().st_size >= MIN_DB_BYTES and not (live / f"{d}_{table}.parquet").exists():
            out.add(d)
    good = []
    for d in out:
        try:
            dt.date.fromisoformat(d)
            good.append(d)
        except ValueError:
            continue
    return sorted(good)
