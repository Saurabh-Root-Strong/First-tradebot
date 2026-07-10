"""
verify_nse_calendar.py — cross-check the HARDCODED NSE constants against ground truth.

The bot hardcodes three things NSE periodically REVISES:
  • market_calendar.NSE_HOLIDAYS   — the yearly trading-holiday list
  • the weekly-expiry weekday rule  — NIFTY weekly = Tuesday (moved Thu→Tue 2025-09-01)
  • constants.LOT_SIZES            — F&O lot sizes (rebased Jan-2026: 75→65 etc)

Hardcoding is fine, drifting silently is not. This script verifies each against the
Daily_Cash_Market fno_bhavcopy GROUND TRUTH (the actual NSE trade_dates / expiry_dates /
turnover DCM already ingests from the exchange), so a revision we forgot to apply is
CAUGHT, not discovered live. Run it periodically (e.g. a Monday cron):

    .venv\\Scripts\\python.exe verify_nse_calendar.py
    exit 0 = all consistent · exit 1 = drift found (prints the mismatch)

Lot size is not a bhavcopy column, so it is DERIVED: value_lacs = close × lot × contracts
→ lot = value_lacs·1e5 / (close·contracts); the modal derived lot is the cross-check.
"""
from __future__ import annotations

import datetime
import glob
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from core.constants import LOT_SIZES, NSE_NAME
from core.market_calendar import NSE_HOLIDAYS, is_trading_day

_YEAR = 2026


def _con():
    p = glob.glob("../Daily_Cash_Market/**/market_data.duckdb", recursive=True)
    if not p:
        print("  DCM market_data.duckdb not found — cannot verify.", file=sys.stderr)
        sys.exit(2)
    import duckdb
    return duckdb.connect(p[0], read_only=True)


def check_holidays(con) -> list[str]:
    """Any WEEKDAY in <year> with no bhavcopy rows = a non-trading day; it must be in
    NSE_HOLIDAYS. Any date in NSE_HOLIDAYS that DID trade = a wrong entry. Only verifiable
    up to the last ingested trade_date (future holidays can't be ground-truthed yet)."""
    problems = []
    max_td = con.execute("select max(trade_date) from fno_bhavcopy").fetchone()[0]
    traded = {r[0] for r in con.execute(
        "select distinct trade_date from fno_bhavcopy where extract(year from trade_date)=?",
        [_YEAR]).fetchall()}
    d = datetime.date(_YEAR, 1, 1)
    while d <= min(max_td, datetime.date(_YEAR, 12, 31)):
        if d.weekday() < 5:                                  # weekday
            did_trade = d in traded
            listed = d.isoformat() in NSE_HOLIDAYS
            if not did_trade and not listed:
                problems.append(f"  MISSING: {d} ({d:%A}) did NOT trade but is not in "
                                "NSE_HOLIDAYS (new holiday? or a data gap)")
            if did_trade and listed:
                problems.append(f"  WRONG:   {d} ({d:%A}) is in NSE_HOLIDAYS "
                                f"'{NSE_HOLIDAYS[d.isoformat()]}' but the market TRADED")
        d += datetime.timedelta(days=1)
    print(f"  holidays: verified 2026-01-01..{max_td} against bhavcopy "
          f"({'OK' if not problems else str(len(problems))+' ISSUE(S)'})")
    # also flag future-listed holidays that fall on a weekend (harmless but sloppy)
    for iso in NSE_HOLIDAYS:
        dt = datetime.date.fromisoformat(iso)
        if dt.weekday() >= 5:
            problems.append(f"  WEEKEND: {iso} ({dt:%A}) in NSE_HOLIDAYS is already a "
                            "weekend (redundant)")
    return problems


def check_weekly_expiry(con) -> list[str]:
    """NIFTY OPTIDX expiry_dates should all be TUESDAY (weekday 1), or a Monday when that
    Tuesday was a holiday (rolled back). NIFTY ALSO lists long-dated quarterly / semi-annual
    options on the legacy last-THURSDAY; those are never the NEAREST expiry, so we validate
    the nearest expiry per recent day (what the bot trades + what days_to_expiry computes),
    which is false-positive-free vs checking every listed expiry."""
    # Nearest LIQUID expiry per recent day. Filter by contracts>=1000 to skip the phantom
    # near-zero-volume expiry_dates bhavcopy carries (e.g. the 2026-06-25 Thursday artifact
    # with 0 contracts) — those are never the tradeable front expiry the bot uses.
    import itertools
    rows = con.execute(
        "select trade_date, expiry_date, sum(contracts) c from fno_bhavcopy "
        "where instrument='OPTIDX' and symbol='NIFTY' and expiry_date >= trade_date "
        "and trade_date >= (select max(trade_date)-90 from fno_bhavcopy) "
        "group by trade_date, expiry_date having c >= 1000 "
        "order by trade_date, expiry_date").fetchall()
    bad, checked = set(), 0
    for td, grp in itertools.groupby(rows, key=lambda r: r[0]):
        nxt = next(iter(grp))[1]                             # nearest LIQUID expiry
        checked += 1
        wd = nxt.weekday()
        if wd == 1:                                          # Tuesday
            continue
        if wd == 0 and not is_trading_day(nxt + datetime.timedelta(days=1)):
            continue                                         # holiday-rolled Monday
        bad.add(f"  EXPIRY:  nearest LIQUID NIFTY expiry from {td} is {nxt} ({nxt:%A}) - "
                "not Tuesday (nor a holiday-rolled Monday). Weekly rule may have changed.")
    if checked == 0:
        # FAIL-CLOSED: zero liquid expiries found => the query/schema changed or bhavcopy has
        # a gap. Reporting "all Tue" here would be a pass on an empty set.
        print("  weekly expiry: CANNOT VERIFY (0 liquid expiries found) -> CHECK DID NOT RUN")
        return ["  EXPIRY:  could NOT verify the weekly rule — no liquid NIFTY OPTIDX "
                "expiries in the last 90d. Inspect bhavcopy schema / ingestion."]
    print(f"  weekly expiry: nearest-liquid-expiry checked over {checked} recent days "
          f"({'all Tue/rolled-Mon' if not bad else str(len(bad))+' OFF-DAY'})")
    return sorted(bad)


def check_lot_sizes(con) -> list[str]:
    """Derive the lot from FUTIDX turnover (value_lacs = close·lot·contracts) over the most
    recent trades and compare the modal derived lot to LOT_SIZES. Approximate — flags a
    lot that is OFF BY A LOT (a real revision), not rounding noise."""
    problems = []
    for sym, nse in NSE_NAME.items():
        rows = con.execute(
            "select value_lacs, close_price, contracts from fno_bhavcopy "
            "where instrument='FUTIDX' and symbol=? and contracts>0 and close_price>0 "
            "and trade_date >= (select max(trade_date)-30 from fno_bhavcopy) "
            "limit 5000", [nse]).fetchall()
        lots = [round(v * 1e5 / (c * n)) for v, c, n in rows if v and c and n]
        if not lots:
            # FAIL-CLOSED: silently skipping is how a check dies without anyone noticing.
            # No derivable rows => bhavcopy schema/ingestion changed, or a data gap. Either
            # way THIS CHECK DID NOT RUN, and must not be mistaken for a pass.
            problems.append(f"  LOTSIZE: {nse} — could NOT derive a lot from turnover "
                            "(no usable FUTIDX rows in the last 30d). The lot check did NOT "
                            "run; inspect bhavcopy schema / ingestion.")
            print(f"  lot {nse:11s}: CANNOT DERIVE (no usable rows)  -> CHECK DID NOT RUN")
            continue
        lots.sort()
        modal = max(set(lots), key=lots.count)
        want = LOT_SIZES.get(sym)
        # accept within 5% (turnover uses avg price vs close → small derive error)
        if want and abs(modal - want) / want > 0.05:
            problems.append(f"  LOTSIZE: {nse} derived lot ~{modal} but LOT_SIZES={want} "
                            "— possible NSE lot revision.")
        print(f"  lot {nse:11s}: derived ~{modal:4d}  hardcoded {want}  "
              f"{'OK' if want and abs(modal-want)/want<=0.05 else 'CHECK'}")
    return problems


def main() -> None:
    print("=" * 78)
    print("NSE CONSTANTS VERIFICATION — hardcoded vs DCM fno_bhavcopy ground truth")
    print("=" * 78)
    con = _con()
    try:
        problems = check_holidays(con) + check_weekly_expiry(con) + check_lot_sizes(con)
    finally:
        con.close()
    print("-" * 78)
    if problems:
        print(f"DRIFT FOUND ({len(problems)}):")
        for p in problems:
            print(p)
        sys.exit(1)
    print("ALL CONSISTENT — hardcoded NSE constants match ground truth.")
    sys.exit(0)


if __name__ == "__main__":
    main()
