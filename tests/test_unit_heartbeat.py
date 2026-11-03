"""Unit — the supervisor must not restart a healthy capturer because it caught the heartbeat
file mid-rewrite. 2026-10-16 14:35 it did: "WS stalled — no tick for 1000000000s"."""
import os
import threading
import time

import supervise as sv


def test_heartbeat_age_reads_a_good_file(tmp_path, monkeypatch):
    hb = tmp_path / "hb.txt"
    hb.write_text(f"{time.time() - 12:.0f}")
    monkeypatch.setattr(sv, "HEARTBEAT", hb)
    age = sv.heartbeat_age(pause=0)
    assert 10 <= age <= 14


def test_unreadable_is_unknown_not_31_years_stale(tmp_path, monkeypatch):
    monkeypatch.setattr(sv, "HEARTBEAT", tmp_path / "missing.txt")
    assert sv.heartbeat_age(pause=0) is None
    hb = tmp_path / "hb.txt"
    hb.write_text("")                                         # caught mid-rewrite
    monkeypatch.setattr(sv, "HEARTBEAT", hb)
    assert sv.heartbeat_age(pause=0) is None


def test_a_moment_of_unreadability_never_triggers_a_restart_but_a_long_one_does():
    t0 = 1_000_000.0
    age, since = sv.stall_age(None, None, t0)                 # first unreadable read
    assert age == 0 and since == t0
    age, since = sv.stall_age(None, since, t0 + 30)           # still unreadable, 30s later
    assert age == 30 < sv.WS_STALL_SEC
    age, since = sv.stall_age(5.0, since, t0 + 45)            # readable again -> clock clears
    assert age == 5.0 and since is None
    age, since = sv.stall_age(None, None, t0 + 60)
    age, since = sv.stall_age(None, since, t0 + 60 + sv.WS_STALL_SEC + 1)
    assert age > sv.WS_STALL_SEC                              # unreadable too long = a real stall


def test_racing_an_in_place_rewriter_never_reports_a_stall(tmp_path, monkeypatch):
    """The OLD writer (truncate-then-write, as Path.write_text does) hammered the file while
    the reader polled: every result must be a sane age or None — never the 1e9 that
    restarted the capturer."""
    hb = tmp_path / "hb.txt"
    hb.write_text(f"{time.time():.0f}")
    monkeypatch.setattr(sv, "HEARTBEAT", hb)
    stop = threading.Event()

    def writer():
        while not stop.is_set():
            try:
                with open(hb, "w") as f:                      # truncate ...
                    f.flush()
                    f.write(f"{time.time():.0f}")             # ... then write
            except OSError:
                pass

    th = threading.Thread(target=writer, daemon=True)
    th.start()
    try:
        ages = [sv.heartbeat_age(retries=3, pause=0.001) for _ in range(400)]
    finally:
        stop.set()
        th.join(timeout=2)
    assert all(a is None or a < 5 for a in ages)


def test_atomic_writer_pattern_is_never_seen_empty(tmp_path):
    """The NEW writer (write a temp file, os.replace it in): a reader never sees it empty."""
    hb = tmp_path / "hb.txt"
    tmp = hb.with_suffix(".tmp")
    hb.write_text("1")
    stop = threading.Event()

    def writer():
        i = 0
        while not stop.is_set():
            i += 1
            try:
                tmp.write_text(str(i))
                os.replace(tmp, hb)
            except OSError:                                   # Windows: reader holds it open
                pass

    th = threading.Thread(target=writer, daemon=True)
    th.start()
    seen_empty = 0
    try:
        for _ in range(2000):
            try:
                if hb.read_text().strip() == "":
                    seen_empty += 1
            except OSError:
                pass
    finally:
        stop.set()
        th.join(timeout=2)
    assert seen_empty == 0
