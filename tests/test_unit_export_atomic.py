"""F1 unit — parquet export is ATOMIC (temp + os.replace) and failures are COUNTED,
not swallowed. Guards against the chain_snapshots silent-death class re-appearing on the
mirror-export path: a torn read (reader sees a half-written file) or a silent freeze
(export keeps failing with nothing on the header).

Uses a bare IntradayDB shell (no writer thread / duckdb) + a fake connection that emulates
DuckDB's `COPY ... TO '<path>'` so the test is fully offline.
"""
import datetime

import intraday_db as idb


def _bare_db(tmp_path, monkeypatch):
    db = idb.IntradayDB.__new__(idb.IntradayDB)      # shell — no thread, no duckdb
    db._date = datetime.date(2026, 7, 6)
    db._export_errors = {}
    monkeypatch.setattr(idb, "_LIVE_DIR", tmp_path)
    monkeypatch.setattr(idb, "_PARQUET_TABLES", ("ticks",))
    return db


class _OkCon:
    """Emulate COPY TO by materialising the target temp file."""
    def execute(self, sql):
        target = sql.split("'")[1]        # ...TO '<path>' (FORMAT ...)
        from pathlib import Path
        Path(target).write_bytes(b"PAR1-complete-snapshot")


class _FailCon:
    def execute(self, sql):
        raise RuntimeError("disk full")


def test_export_atomic_publishes_and_cleans_temp(tmp_path, monkeypatch):
    db = _bare_db(tmp_path, monkeypatch)
    db._export_parquet(_OkCon())

    final = tmp_path / "2026-07-06_ticks.parquet"
    assert final.exists()
    assert final.read_bytes() == b"PAR1-complete-snapshot"
    assert not (tmp_path / "2026-07-06_ticks.parquet.tmp").exists()   # replace consumed the temp
    assert db.export_error_count() == 0


def test_export_failure_counts_and_preserves_prior_file(tmp_path, monkeypatch):
    db = _bare_db(tmp_path, monkeypatch)
    final = tmp_path / "2026-07-06_ticks.parquet"
    final.write_bytes(b"PRIOR-GOOD")     # a previously-published complete mirror

    db._export_parquet(_FailCon())       # must NOT raise

    assert db.export_error_count() == 1                    # counted, surfaced on the badge
    assert final.read_bytes() == b"PRIOR-GOOD"             # prior file intact — no torn overwrite
    assert not (tmp_path / "2026-07-06_ticks.parquet.tmp").exists()
