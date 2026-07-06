"""F2 unit — TODAY's mirror read is mtime/size-cached (dedupes the hundreds of same-render
re-reads) yet REFRESHES the moment the file changes, so it is never stale. Also asserts the
cache is bounded so it can't balloon the capture VM's RAM at day rollover.
"""
import time

import pandas as pd

import core.mirror_io as mio


def _write(p, ts_vals):
    pd.DataFrame({
        "ts": pd.to_datetime(ts_vals, utc=True),
        "symbol": ["X"] * len(ts_vals),
        "ltp": list(range(len(ts_vals))),
    }).to_parquet(p)


def test_live_read_cached_then_refreshes(tmp_path, monkeypatch):
    mio._LIVE_CACHE.clear()
    monkeypatch.setattr(mio, "LIVE_DIR", tmp_path)
    monkeypatch.setattr(mio, "today_iso", lambda: "2026-07-06")
    p = tmp_path / "2026-07-06_ticks.parquet"
    _write(p, ["2026-07-06T09:20:00", "2026-07-06T09:21:00"])

    df1 = mio.read_mirror("ticks")
    assert len(df1) == 2
    assert str(p) in mio._LIVE_CACHE               # cached after first read

    key_before = mio._LIVE_CACHE[str(p)][0]
    df2 = mio.read_mirror("ticks")                 # unchanged file → cache hit, not re-parsed
    assert len(df2) == 2
    assert mio._LIVE_CACHE[str(p)][0] == key_before

    time.sleep(0.01)
    _write(p, ["2026-07-06T09:20:00", "2026-07-06T09:21:00", "2026-07-06T09:22:00"])
    df3 = mio.read_mirror("ticks")                 # grew → new size/mtime → refreshed
    assert len(df3) == 3


def test_live_cache_is_bounded(tmp_path, monkeypatch):
    mio._LIVE_CACHE.clear()
    monkeypatch.setattr(mio, "LIVE_DIR", tmp_path)
    monkeypatch.setattr(mio, "today_iso", lambda: "2026-07-06")
    # Populate past the cap with distinct table files; cache must not grow unbounded.
    for i in range(mio._LIVE_CACHE_CAP + 5):
        tbl = f"t{i}"
        _write(tmp_path / f"2026-07-06_{tbl}.parquet", ["2026-07-06T09:20:00"])
        mio.read_mirror(tbl)
    assert len(mio._LIVE_CACHE) <= mio._LIVE_CACHE_CAP + 1
