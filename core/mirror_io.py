"""core.mirror_io — the one canonical reader for the lock-free intraday parquet mirrors.

Previously this loader was reimplemented (slightly differently) in timeframe_delta,
smart_money, intraday_tf, ... — which is how drift creeps in. One implementation now:
IST-localised ts, optional symbol filter, optional lookahead-free as_of cutoff.
"""
from __future__ import annotations

import datetime
import functools

import pandas as pd

from .constants import IST, LIVE_DIR, today_iso


@functools.lru_cache(maxsize=32)
def _load_processed(path_str: str, _mtime_ns: int, _size: int) -> "pd.DataFrame":
    """Parse + normalise a mirror parquet ONCE (ts→IST, epoch-junk dropped, sorted).
    Cached by (path, mtime, size) so repeated reads of a STATIC past-day file (the
    scout/backtest replay hammers the same file hundreds of times per render) skip
    the disk read + re-parse. A changed file (live append) gets a new mtime → new
    cache entry, so it is never stale. Only used for past days (see read_mirror)."""
    df = pd.read_parquet(path_str)
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(IST)
    df = df[df["ts"] >= pd.Timestamp("2000-01-01", tz=IST)]
    return df.sort_values("ts").reset_index(drop=True)


# TODAY's mirror is re-read on every scout/render pass (4 idx × many lookbacks × 34
# callbacks) — an uncached read re-parsed a growing multi-MB ticks file hundreds of
# times per render. Cache it too, keyed on (mtime, size): the file only changes when
# the writer exports (~every 10 s = the mirror's existing staleness bound), so a hit is
# never staler than a fresh read would have been. Bounded to ONE frame per live file
# (replaced on change, whole dict cleared past ~2 sessions' worth of tables) so it can't
# balloon the capture VM's RAM the way an unbounded lru would at day rollover.
_LIVE_CACHE: dict = {}   # path_str -> ((mtime_ns, size), df)
_LIVE_CACHE_CAP = 16     # ~8 live tables; slack for one day rollover before a full clear


def _load_live(p) -> "pd.DataFrame":
    """mtime/size-cached parse of TODAY's growing mirror (never stale, RAM-bounded)."""
    st = p.stat()
    key = (st.st_mtime_ns, st.st_size)
    hit = _LIVE_CACHE.get(str(p))
    if hit is not None and hit[0] == key:
        return hit[1]
    df = pd.read_parquet(p)
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(IST)
    df = df[df["ts"] >= pd.Timestamp("2000-01-01", tz=IST)]
    df = df.sort_values("ts").reset_index(drop=True)
    if len(_LIVE_CACHE) > _LIVE_CACHE_CAP:   # guard the capture VM's RAM at date rollover
        _LIVE_CACHE.clear()
    _LIVE_CACHE[str(p)] = (key, df)
    return df


def read_mirror(tbl: str, date: str | None = None,
                as_of: "datetime.datetime | None" = None,
                symbol: str | None = None) -> "pd.DataFrame | None":
    """Read data/intraday/live/<date>_<tbl>.parquet.

    date   : ISO day (default today) — replays a past captured session.
    as_of  : keep only rows with ts <= as_of (lookahead-free reconstruction).
    symbol : keep only that symbol's rows.
    Returns a ts-sorted DataFrame (ts in IST), or None if missing/empty.
    """
    day = date or today_iso()
    p = LIVE_DIR / f"{day}_{tbl}.parquet"
    if not p.exists() or p.stat().st_size < 100:
        return None
    try:
        if day == today_iso():
            # LIVE file grows all session — mtime/size-cached parse (see _load_live):
            # fresh on every export, but dedupes the hundreds of same-render re-reads.
            df = _load_live(p)
        else:
            # PAST day = static file → cached parse (mtime/size-keyed, never stale).
            st = p.stat()
            df = _load_processed(str(p), st.st_mtime_ns, st.st_size)
    except Exception:
        return None
    # NOTE: df may be a cached shared frame — only FILTER below (returns new frames),
    # never mutate it in place.
    if symbol is not None and "symbol" in df.columns:
        df = df[df["symbol"] == symbol]
    if df.empty:
        return None
    if as_of is not None:
        # Defensive: a tz-naive as_of would raise "Cannot compare tz-naive and
        # tz-aware" against the IST ts. Localise naive inputs to IST (callers should
        # pass tz-aware, but one naive caller shouldn't crash the whole read).
        ts_cut = pd.Timestamp(as_of)
        if ts_cut.tzinfo is None:
            ts_cut = ts_cut.tz_localize(IST)
        df = df[df["ts"] <= ts_cut]
    return df.sort_values("ts").reset_index(drop=True) if len(df) else None
