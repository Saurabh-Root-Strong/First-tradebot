"""Unit — eod_sync must not archive the database the capturer is still writing (today's):
a 15:45 copy is a mid-write snapshot, and the size-matched purge then deletes the VM's
whole copy. 10-14/10-15/10-19 archives stop at 15:45 because of this."""
import eod_sync as es


def test_todays_duckdb_is_left_for_the_next_run(tmp_path, monkeypatch):
    remote = {"~/tradebot/data/intraday/2026-10-21.duckdb": 100,
              "~/tradebot/data/intraday/2026-10-22.duckdb": 200}
    got = []
    monkeypatch.setattr(es, "_remote_listing", lambda h, k, g: remote)
    monkeypatch.setattr(es, "_scp", lambda h, k, r, l, timeout=300: got.append(l.name) or True)
    fetched, _ = es._sync_perday("h", "k", "g", tmp_path, True, skip_day="2026-10-22")
    assert got == ["2026-10-21.duckdb"] and fetched == 1


def test_without_skip_day_everything_missing_is_fetched(tmp_path, monkeypatch):
    remote = {"~/x/2026-10-22_ticks.parquet": 10, "~/x/2026-10-21_ticks.parquet": 10}
    got = []
    monkeypatch.setattr(es, "_remote_listing", lambda h, k, g: remote)
    monkeypatch.setattr(es, "_scp", lambda h, k, r, l, timeout=300: got.append(l.name) or True)
    es._sync_perday("h", "k", "g", tmp_path, True)
    assert sorted(got) == ["2026-10-21_ticks.parquet", "2026-10-22_ticks.parquet"]
