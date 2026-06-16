"""core.mirror_io — the one canonical reader for the lock-free intraday parquet mirrors.

Previously this loader was reimplemented (slightly differently) in timeframe_delta,
smart_money, intraday_tf, ... — which is how drift creeps in. One implementation now:
IST-localised ts, optional symbol filter, optional lookahead-free as_of cutoff.
"""
from __future__ import annotations

import datetime

import pandas as pd

from .constants import IST, LIVE_DIR, today_iso


def read_mirror(tbl: str, date: str | None = None,
                as_of: "datetime.datetime | None" = None,
                symbol: str | None = None) -> "pd.DataFrame | None":
    """Read data/intraday/live/<date>_<tbl>.parquet.

    date   : ISO day (default today) — replays a past captured session.
    as_of  : keep only rows with ts <= as_of (lookahead-free reconstruction).
    symbol : keep only that symbol's rows.
    Returns a ts-sorted DataFrame (ts in IST), or None if missing/empty.
    """
    p = LIVE_DIR / f"{date or today_iso()}_{tbl}.parquet"
    if not p.exists() or p.stat().st_size < 100:
        return None
    try:
        df = pd.read_parquet(p)
    except Exception:
        return None
    if symbol is not None and "symbol" in df.columns:
        df = df[df["symbol"] == symbol]
    if df.empty:
        return None
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(IST)
    if as_of is not None:
        df = df[df["ts"] <= pd.Timestamp(as_of)]
    return df.sort_values("ts").reset_index(drop=True) if len(df) else None
