"""Unit — range_flag: the big-move-day SIZE flag ("moved N× normal for this time of day").

Data-free: curves are built from synthetic tick frames, so these pin the arithmetic and the
guards (warm-up, early-morning, stale opening print), not any market result.
"""
import pandas as pd

import range_flag as rf


def _ticks(day, pts):
    """pts = [(HH:MM:SS, price), ...] -> a tick frame like the mirror's."""
    return pd.DataFrame({"ts": pd.to_datetime([f"{day} {t}" for t, _ in pts]).tz_localize("Asia/Kolkata"),
                         "ltp": [p for _, p in pts]})


def _session(day, move):
    """A full session: opens at 100, has reached +move by 09:50, then sits there until 15:30."""
    pts, t = [], pd.Timestamp(f"{day} 09:15:00")
    for i in range(375):
        m = t + pd.Timedelta(minutes=i)
        pts.append((m.strftime("%H:%M:%S"), 100.0 + move * min(i, 35) / 35.0))
    return _ticks(day, pts)


def test_curve_is_range_so_far_in_percent_of_the_open():
    c = rf.curve(_session("2026-10-01", 2.0))
    assert rf.at(c, 9 * 60 + 15) == 0.0                       # nothing has happened yet
    assert abs(rf.at(c, 9 * 60 + 50) - 2.0) < 1e-9             # 100 -> 102 by 09:50
    assert abs(rf.at(c, 14 * 60) - 2.0) < 1e-9                 # range-so-far never shrinks
    assert rf.at(c, 9 * 60) is None                            # before the first point


def test_multiple_is_today_over_the_median_of_priors_at_the_same_minute():
    priors = [rf.curve(_session(f"2026-09-{d:02d}", 1.0)) for d in range(1, 13)]   # 12 normal days
    today = rf.curve(_session("2026-10-01", 2.6))
    m = rf.multiple(today, priors, 10 * 60 + 15)
    assert abs(m - 2.6) < 1e-6
    assert rf.level(m) == "flag" and rf.level(1.7) == "elevated" and rf.level(1.2) == ""
    assert rf.expectation(m) == (1.82, 56)                     # the measured 2.5–3× row
    assert rf.expectation(9.0) == (1.89, 67)                   # never extrapolated past the table


def test_no_reading_before_0945_or_without_enough_history():
    priors = [rf.curve(_session(f"2026-09-{d:02d}", 1.0)) for d in range(1, 13)]
    today = rf.curve(_session("2026-10-01", 2.0))
    assert rf.multiple(today, priors, 9 * 60 + 30) is None     # too early: denominator is noise
    assert rf.multiple(today, priors[:9], 11 * 60) is None     # 9 priors < MIN_DAYS
    assert rf.multiple(today, priors[:10], 11 * 60) is not None


def test_only_the_last_20_priors_count():
    old = [rf.curve(_session(f"2026-08-{d:02d}", 5.0)) for d in range(1, 11)]      # 10 wild days, long ago
    new = [rf.curve(_session(f"2026-09-{d:02d}", 1.0)) for d in range(1, 21)]      # the last 20: calm
    today = rf.curve(_session("2026-10-01", 2.0))
    assert abs(rf.multiple(today, old + new, 11 * 60) - 2.0) < 1e-6


def test_stale_previous_close_print_in_the_opening_minute_does_not_set_the_base():
    # 09:15:00 carries yesterday's close (90); the real open (100) prints seconds later.
    pts = [("09:15:00", 90.0), ("09:15:20", 100.0)] + [(f"{9 + (16 + i) // 60:02d}:{(16 + i) % 60:02d}:00", 100.5)
                                                       for i in range(60)]
    c = rf.curve(_ticks("2026-10-01", pts))
    assert rf.at(c, 10 * 60) < 1.0                             # 0.5%, not the 11% a 90-base would give


def _exch(day, pts, open_=100.0):
    """Ticks carrying the exchange's running day_open / day_high / day_low, like the live feed."""
    t = _ticks(day, pts)
    t["day_open"] = open_
    t["day_high"] = [max(open_, *[p for _, p in pts[:i + 1]]) for i in range(len(pts))]
    t["day_low"] = [min(open_, *[p for _, p in pts[:i + 1]]) for i in range(len(pts))]
    return t


def test_exchange_day_range_is_used_and_survives_a_late_capture_start():
    # The day opened at 100 and had already been to 103 before capture came up at 10:00,
    # trading 101.0-101.2 since. The tick prices alone span 0.2%; the day has ranged 3%.
    pts = [(f"10:{m:02d}:00", 101.0 + 0.2 * (m % 2)) for m in range(60)]
    t = _exch("2026-10-01", pts)
    t["day_high"] = 103.0
    c = rf.curve(t)
    assert abs(rf.at(c, 10 * 60 + 30) - 3.0) < 1e-9
    assert rf.at(rf.curve(t.drop(columns=["day_open", "day_high", "day_low"])), 10 * 60 + 30) < 0.3


def test_rows_with_a_stale_day_open_or_another_date_are_dropped():
    pts = [(f"09:{15 + m}:00", 100.0) for m in range(40)]
    t = _exch("2026-10-01", pts)
    t.loc[:2, ["day_open", "day_high", "day_low"]] = [95.0, 99.0, 91.0]   # yesterday's fields, 3 ticks
    y = _exch("2026-09-30", [("15:29:00", 90.0)])                          # yesterday's last quote
    c = rf.curve(pd.concat([y, t], ignore_index=True))
    assert c.index[0] >= 9 * 60 + 18 and float(c.max()) == 0.0            # neither leaked in


def test_a_prior_session_that_stopped_capturing_is_not_used_later():
    full = [rf.curve(_session(f"2026-09-{d:02d}", 1.0)) for d in range(1, 11)]     # 10 complete days
    dead = rf.curve(_session("2026-09-11", 9.0)); dead = dead[dead.index <= 10 * 60 + 44]   # died 10:44
    today = rf.curve(_session("2026-10-01", 2.0))
    assert rf.at(dead, 13 * 60, rf.PRIOR_MAX_STALE) is None                 # stale -> no reading
    assert abs(rf.multiple(today, full + [dead], 13 * 60) - 2.0) < 1e-6     # the 9% day is ignored
    assert rf.multiple(today, full[:9] + [dead], 13 * 60) is None           # and cannot make up the numbers


def test_empty_or_thin_input_gives_no_curve():
    assert len(rf.curve(None)) == 0
    assert len(rf.curve(_ticks("2026-10-01", [("09:15:00", 100.0), ("09:16:00", 101.0)]))) == 0
    assert rf.level(None) == "" and rf.level(float("nan")) == ""
