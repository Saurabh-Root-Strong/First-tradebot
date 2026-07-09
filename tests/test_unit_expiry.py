"""Unit — days_to_expiry / weekly_expiry_on_or_after (the DTE / theta axis).

Locks the single-source-of-truth expiry math (core.market_calendar) that the theta-cliff
badge (dashboard._scout_dte) and the scenario backtest (backtest_scout_trades._dte) both
wrap. The tricky part is the HOLIDAY ROLL-BACK: an expiry landing on a holiday moves to the
prior trading day, so a holiday-Tuesday makes the WEEKLY expiry a Monday. Fully offline
(the NSE 2026 holiday list is baked into market_calendar).
"""
import datetime

from core import market_calendar as mc


def test_weekly_normal_tuesday():
    # a plain trading Tuesday IS the weekly expiry -> DTE 0
    assert mc.days_to_expiry("2026-07-14", weekly=True) == 0
    # mid-week points to the next Tuesday
    assert mc.days_to_expiry("2026-07-09", weekly=True) == 5   # Thu -> Tue 14th
    assert mc.days_to_expiry("2026-07-13", weekly=True) == 1   # Mon -> Tue 14th


def test_weekly_holiday_tuesday_rolls_back_to_monday():
    # 2026-03-31 (Tue) = Mahavir Jayanti holiday -> that week's weekly expiry is Mon 03-30
    assert mc.weekly_expiry_on_or_after("2026-03-30") == datetime.date(2026, 3, 30)
    assert mc.days_to_expiry("2026-03-30", weekly=True) == 0    # Monday IS the expiry
    assert mc.days_to_expiry("2026-03-27", weekly=True) == 3    # Fri before -> Mon 03-30
    # ON the holiday-Tuesday, this week's expiry already passed -> next Tuesday (Apr 7)
    assert mc.days_to_expiry("2026-03-31", weekly=True) == 7


def test_weekly_accepts_weekend_and_datetime_inputs():
    # Saturday input -> the coming Tuesday
    assert mc.days_to_expiry("2026-07-11", weekly=True) == 3          # Sat -> Tue 14th
    # datetime and date inputs resolve the same as the ISO string
    assert (mc.days_to_expiry(datetime.datetime(2026, 7, 9, 11, 30), weekly=True)
            == mc.days_to_expiry("2026-07-09", weekly=True))


def test_monthly_last_tuesday_and_holiday_roll():
    # monthly = last Tuesday; June 2026 last Tue = 06-30 (a trading day)
    assert mc.days_to_expiry("2026-06-30", weekly=False) == 0
    assert mc.days_to_expiry("2026-07-09", weekly=False) == 19        # -> Jul 28 (last Tue)
    # March 2026 last Tuesday (03-31) is a holiday -> monthly rolls back to Mon 03-30
    assert mc.monthly_expiry(2026, 3) == datetime.date(2026, 3, 30)


def test_bad_input_returns_minus_one():
    assert mc.days_to_expiry("not-a-date", weekly=True) == -1
    assert mc.days_to_expiry(None, weekly=False) == -1
