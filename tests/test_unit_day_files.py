"""Unit — day_files: one finished session's table from the parquet mirror (laptop) or the
per-day DuckDB (VM). The bug it guards: range_flag / oi_flag read mirrors only, so on the VM —
which keeps 93 days of databases but 1–2 days of mirrors — both flags were silent."""
import duckdb
import pandas as pd

import day_files


def _frame(day, n=40, base=100.0):
    ts = pd.date_range(f"{day} 09:15", periods=n, freq="1min", tz="Asia/Kolkata")
    return pd.DataFrame({"ts": ts, "symbol": "X", "ltp": [base + i * 0.1 for i in range(n)], "junk": 1})


def _setup(tmp_path, monkeypatch):
    live = tmp_path / "live"
    live.mkdir()
    monkeypatch.setattr(day_files, "_dirs", lambda: (live, tmp_path))
    monkeypatch.setattr(day_files, "MIN_DB_BYTES", 1)          # test databases are tiny
    return live


def _db(path, df):
    con = duckdb.connect(str(path))
    con.execute("create table ticks as select * from df")
    con.close()


def test_parquet_mirror_is_read_with_only_the_asked_columns(tmp_path, monkeypatch):
    live = _setup(tmp_path, monkeypatch)
    _frame("2026-09-01").to_parquet(live / "2026-09-01_ticks.parquet")
    d = day_files.load("ticks", "2026-09-01", ["ts", "symbol", "ltp", "not_there"])
    assert list(d.columns) == ["ts", "symbol", "ltp"] and len(d) == 40
    assert d.ts.iloc[0].strftime("%Y-%m-%d %H:%M %z") == "2026-09-01 09:15 +0530"


def test_falls_back_to_the_per_day_database_when_there_is_no_mirror(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    _db(tmp_path / "2026-09-02.duckdb", _frame("2026-09-02", base=200.0))
    d = day_files.load("ticks", "2026-09-02", ["ts", "symbol", "ltp"])
    assert d is not None and len(d) == 40 and abs(d.ltp.iloc[0] - 200.0) < 1e-9
    assert d.ts.iloc[0].strftime("%Y-%m-%d %H:%M") == "2026-09-02 09:15"      # IST preserved
    assert day_files.load("oi_snapshots", "2026-09-02", ["ts"]) is None       # table absent -> None, no raise


def test_the_mirror_wins_when_both_exist_and_missing_days_are_none(tmp_path, monkeypatch):
    live = _setup(tmp_path, monkeypatch)
    _frame("2026-09-03", base=100.0).to_parquet(live / "2026-09-03_ticks.parquet")
    _db(tmp_path / "2026-09-03.duckdb", _frame("2026-09-03", base=999.0))
    assert abs(day_files.load("ticks", "2026-09-03", ["ts", "ltp"]).ltp.iloc[0] - 100.0) < 1e-9
    assert day_files.load("ticks", "2026-09-04", ["ts", "ltp"]) is None


def test_days_lists_sessions_from_either_store(tmp_path, monkeypatch):
    live = _setup(tmp_path, monkeypatch)
    _frame("2026-09-01").to_parquet(live / "2026-09-01_ticks.parquet")
    _db(tmp_path / "2026-09-02.duckdb", _frame("2026-09-02"))
    (live / "tmp_2026-06-29_ticks.parquet").write_bytes(b"x" * 500)           # a stray non-date name
    assert day_files.days("ticks", 100) == ["2026-09-01", "2026-09-02"]
    # a mirror that is too small is a DEAD capture: it is not rescued by a database beside it
    _db(tmp_path / "2026-09-01.duckdb", _frame("2026-09-01"))
    assert day_files.days("ticks", 10_000_000) == ["2026-09-02"]
