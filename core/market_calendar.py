"""core.market_calendar — NSE equity (capital-market) trading-day calendar.

A trading day = a weekday (Mon–Fri) that is NOT an NSE holiday. Before this module
the codebase only checked ``weekday() < 5``, so on a weekday holiday (e.g. Moharram,
Fri 2026-06-26) capture/auth/sync still fired and the dashboard showed a confusing
"warming up" with no live feed.

Source: the published NSE 2026 trading-holiday list, CROSS-VERIFIED against
Daily_Cash_Market ``fno_bhavcopy`` actual ``trade_date``s for H1-2026 — every listed
weekday holiday below was confirmed a non-trading day (no bhavcopy row). H2-2026
dates are from the published list (not yet in bhavcopy). Special evening "muhurat"
sessions (Diwali Laxmi Pujan) are NOT modelled — those are a separate one-off window.

Keep this current each year: replace the dict when NSE publishes the next list, and
re-verify the prior year against bhavcopy.
"""
from __future__ import annotations

import datetime

# ISO date -> holiday name (NSE equity segment, full-day closures).
NSE_HOLIDAYS: dict[str, str] = {
    "2026-01-15": "Maharashtra Municipal Elections",
    "2026-01-26": "Republic Day",
    "2026-03-03": "Holi",
    "2026-03-26": "Shri Ram Navami",
    "2026-03-31": "Shri Mahavir Jayanti",
    "2026-04-03": "Good Friday",
    "2026-04-14": "Dr. Ambedkar Jayanti",
    "2026-05-01": "Maharashtra Day",
    "2026-05-28": "Bakri Eid",
    "2026-06-26": "Moharram",
    "2026-09-14": "Ganesh Chaturthi",
    "2026-10-02": "Mahatma Gandhi Jayanti",
    "2026-10-20": "Dussehra",
    "2026-11-10": "Diwali Balipratipada",
    "2026-11-24": "Guru Nanak Jayanti",
    "2026-12-25": "Christmas",
}


def _as_date(d) -> datetime.date:
    if isinstance(d, datetime.datetime):
        return d.date()
    if isinstance(d, datetime.date):
        return d
    return datetime.date.fromisoformat(str(d)[:10])


def holiday_name(d=None) -> "str | None":
    """Holiday name if `d` (date/datetime/ISO str, default today) is an NSE holiday."""
    d = _as_date(d) if d is not None else datetime.date.today()
    return NSE_HOLIDAYS.get(d.isoformat())


def is_trading_holiday(d=None) -> bool:
    return holiday_name(d) is not None


def is_trading_day(d=None) -> bool:
    """True iff `d` is a weekday AND not an NSE holiday."""
    d = _as_date(d) if d is not None else datetime.date.today()
    return d.weekday() < 5 and d.isoformat() not in NSE_HOLIDAYS


def prev_trading_day(d=None) -> datetime.date:
    """Most recent trading day STRICTLY before `d` (skips weekends + holidays)."""
    d = _as_date(d) if d is not None else datetime.date.today()
    d -= datetime.timedelta(days=1)
    while not is_trading_day(d):
        d -= datetime.timedelta(days=1)
    return d


def next_trading_day(d=None) -> datetime.date:
    """Next trading day STRICTLY after `d` (skips weekends + holidays)."""
    d = _as_date(d) if d is not None else datetime.date.today()
    d += datetime.timedelta(days=1)
    while not is_trading_day(d):
        d += datetime.timedelta(days=1)
    return d


# ── Index F&O expiry calendar ────────────────────────────────────────────────────
# NSE index monthly expiry = the LAST TUESDAY of the month, rolled BACK to the prior
# trading day if that Tuesday is a holiday. VALIDATED against Daily_Cash_Market
# fno_bhavcopy FUTIDX expiry_dates for 2026 — matches all of Jan27, Feb24, Mar30(Mon,
# holiday-roll off Mar31 Mahavir Jayanti), Apr28, May26, Jun30, Jul28, Aug25. All four
# indices (NIFTY/BANKNIFTY/FINNIFTY/MIDCPNIFTY) share these monthly futures expiries.
# (NIFTY also has WEEKLY options every Tuesday; the others are monthly-only — but that
# is an options concern, handled by intraday_scout._expiry_kind.)
import calendar as _calendar


def monthly_expiry(year: int, month: int) -> datetime.date:
    """NSE index monthly (futures) expiry for the given month — last Tuesday,
    holiday-adjusted backward. Authoritative rule (see module note)."""
    weeks = _calendar.monthcalendar(year, month)
    tues = [w[_calendar.TUESDAY] for w in weeks if w[_calendar.TUESDAY]]
    d = datetime.date(year, month, tues[-1])
    while not is_trading_day(d):
        d -= datetime.timedelta(days=1)
    return d


def index_future_expiries(today=None, n: int = 3) -> "list[datetime.date]":
    """The next `n` index-futures monthly expiry dates with expiry >= today
    (holiday-adjusted). out[0]=near, out[1]=next, out[2]=far. Rolls correctly the
    moment a month's expiry passes (near becomes next month), fixing the naive
    'current calendar month = near' bug."""
    today = _as_date(today) if today is not None else datetime.date.today()
    out: list[datetime.date] = []
    y, m = today.year, today.month
    while len(out) < n:
        e = monthly_expiry(y, m)
        if e >= today:
            out.append(e)
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return out
