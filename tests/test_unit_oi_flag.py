"""Unit — oi_flag: "a sudden heavy OI pile-up on one side", scored against this expiry so far.

Data-free: snapshots are synthetic, so these pin the arithmetic and the guards — the capture-gap
rule, the hold window, builds-only, the minimum pool, and which days form the pool."""
import numpy as np
import pandas as pd

import oi_flag as of


def _snap(day, rows):
    """rows = [(HH:MM, call_oi, put_oi), ...] -> an oi_snapshots-like frame."""
    return pd.DataFrame({
        "ts": pd.to_datetime([f"{day} {t}:00" for t, _, _ in rows]).tz_localize("Asia/Kolkata"),
        "total_call_oi": [c for _, c, _ in rows], "total_put_oi": [p for _, _, p in rows],
        "spot": 100.0, "atm": 100, "atm_call_prem": 5.0, "atm_put_prem": 5.0})


def _flat_day(day, start=1000.0, step=0.001):
    """A calm session: both sides creep up 0.1% every 5 minutes, 09:15..15:30."""
    rows, t = [], pd.Timestamp(f"{day} 09:15")
    for i in range(76):
        v = start * (1 + step) ** i
        rows.append(((t + pd.Timedelta(minutes=5 * i)).strftime("%H:%M"), v, v))
    return rows


def test_builds_is_the_15_minute_percent_change_per_side():
    rows = _flat_day("2026-10-01")
    b = of.builds(_snap("2026-10-01", rows))
    assert b.index[0] == 9 * 60 + 15 and np.isnan(b.ce.iloc[0])           # no window yet
    assert abs(b.ce.loc[9 * 60 + 30] - ((1.001 ** 3 - 1) * 100)) < 1e-9   # 3 steps of 0.1%
    assert abs(b.pe.loc[12 * 60] - b.ce.loc[12 * 60]) < 1e-12


def test_a_capture_gap_is_not_read_as_a_burst():
    # nothing captured 11:00-11:40, during which call OI really grew 10%. The first snapshot
    # back must NOT show a 10% "15-minute" build — there is no fresh reading 15 minutes earlier.
    rows = [r for r in _flat_day("2026-10-01") if not ("11:00" <= r[0] <= "11:40")]
    rows = [(t, c * (1.10 if t > "11:40" else 1.0), p) for t, c, p in rows]
    b = of.builds(_snap("2026-10-01", rows))
    assert np.isnan(b.ce.loc[11 * 60 + 45]) and np.isnan(b.ce.loc[11 * 60 + 55])
    assert abs(b.ce.loc[12 * 60]) < 1.0                                    # back to normal once 15m of data exist


def test_rows_from_another_date_are_ignored():
    y = _snap("2026-09-30", [("15:29", 5000.0, 5000.0)])                   # yesterday's last row
    b = of.builds(pd.concat([y, _snap("2026-10-01", _flat_day("2026-10-01"))], ignore_index=True))
    assert abs(b.ce.loc[9 * 60 + 30] - ((1.001 ** 3 - 1) * 100)) < 1e-9    # base is today's 09:15, not 5000


def _series(vals, start=10 * 60 + 15):
    return pd.Series(vals, index=[start + 5 * i for i in range(len(vals))])


def test_the_opening_flood_is_not_a_burst_and_is_not_the_yardstick_for_midday():
    # every session of this expiry: a 6% build in the first windows after the bell, then ~0.3%.
    def day():
        calm = [0.3 + 0.02 * (i % 5) for i in range(60)]          # 0.30-0.38%, never perfectly flat
        return pd.Series([6.0, 5.5, 5.0] + calm, index=[9 * 60 + 35 + 5 * i for i in range(63)])
    prior = pd.concat([day() for _ in range(4)])                # 4 earlier days of the expiry
    today = day()
    assert of.score(today, prior, 9 * 60 + 45) is None          # the daily opening flood: NOT flagged
    # a 2% build at 12:45 is ~7x a normal midday reading; against a pool polluted by the 6%
    # openings it would not clear z 2.5 — against midday readings alone it does.
    today.loc[12 * 60 + 45] = 2.0
    h = of.score(today, prior, 12 * 60 + 45)
    assert h and h["minute"] == 12 * 60 + 45 and h["z"] >= of.Z_FLAG


def test_score_flags_a_build_far_outside_the_pool_and_reports_the_multiple():
    prior = np.r_[np.full(30, 0.2), np.full(30, 0.4)]                      # normal: 0.2-0.4% per 15 min
    today = _series([0.3, 0.3, 0.3, 3.0])
    h = of.score(today, prior, today.index[-1])
    assert h and h["minute"] == today.index[-1] and h["live"] is True
    assert h["z"] >= of.Z_FLAG and h["ratio"] == 10.0                      # 3.0 / median|pool| 0.3


def test_score_ignores_unwinds_and_needs_a_minimum_pool():
    prior = np.full(40, 0.3) + np.linspace(-0.05, 0.05, 40)
    assert of.score(_series([0.3, -5.0]), prior, 10 * 60 + 20) is None      # a big UNWIND is not a pile-up
    assert of.score(_series([0.3, 3.0]), prior[:10], 10 * 60 + 20) is None  # 11 readings < MIN_POOL


def test_flag_holds_for_30_minutes_then_clears():
    prior = np.full(40, 0.3) + np.linspace(-0.05, 0.05, 40)
    today = _series([0.3, 3.0] + [0.3] * 8)                                # burst at 10:05, calm after
    m0 = today.index[1]
    h = of.score(today, prior, m0 + 25)
    assert h and h["minute"] == m0 and h["live"] is False                  # still shown, marked not-live
    assert of.score(today, prior, m0 + 35) is None                         # past the hold window


def test_today_is_scored_against_its_own_earlier_readings_too():
    # no prior days at all (day 1 of a new expiry): silent until today has 30 readings of its own
    calm = [0.3 + 0.01 * (i % 5) for i in range(30)]
    assert of.score(_series(calm[:20] + [3.0]), np.empty(0), 10 * 60 + 15 + 5 * 20) is None
    assert of.score(_series(calm + [3.0]), np.empty(0), 10 * 60 + 15 + 5 * 30) is not None


def test_stage_of_the_contract():
    # NIFTY weekly expiries are Tuesdays: 2026-09-29, 2026-10-06, 2026-10-13. 2026-10-02 is a holiday.
    assert of.stage("2026-10-06", weekly=True) == "EXPIRY"
    assert of.stage("2026-10-07", weekly=True) == "FIRST"      # the session after an expiry
    assert of.stage("2026-10-08", weekly=True) == "MID"
    assert of.stage("2026-10-05", weekly=True) == "LATE"       # 1 day to go
    assert of.stage("2026-10-07", weekly=False) != "FIRST"     # monthly contract: nothing expired the day before


def test_pool_days_compare_like_with_like():
    cap = ["2026-09-29", "2026-09-30", "2026-10-01", "2026-10-05", "2026-10-06", "2026-10-07", "2026-10-08"]
    # MID: this expiry's earlier MID days, topped up from the previous expiry until it has 2 of its own
    assert of.pool_days(cap, "2026-10-09", weekly=True) == ["2026-10-01", "2026-10-08"]
    # FIRST day of a new expiry: nothing earlier in it — judged against the previous FIRST day(s),
    # never against yesterday's expiry-day readings
    assert of.pool_days(cap, "2026-10-07", weekly=True) == ["2026-09-30"]
    # EXPIRY day: earlier expiry days only, never the calm days of its own cycle
    assert of.pool_days(cap, "2026-10-06", weekly=True) == ["2026-09-29"]
    # a MID day never borrows a FIRST / LATE / EXPIRY day
    assert not {"2026-09-29", "2026-09-30", "2026-10-05", "2026-10-06", "2026-10-07"} & set(
        of.pool_days(cap, "2026-10-09", weekly=True))
