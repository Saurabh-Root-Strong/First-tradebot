"""btst_panel.py — read-only view model for the BTST overnight paper ledger.

The BTST close-strength edge (clr >= 0.66 -> long index FUTURES at the close, exit next
~09:30) is the ONLY validated positive-expectancy signal in this system, yet it had no UI:
the negative-EV intraday arrow got a full ledger + badges + notifications while the thing
that actually works lived in a cron log. This module builds the rows/summary the dashboard
renders.

PURE + read-only: it never writes the ledger (the VM cron is the single writer) and never
imports dashboard. Rupee truth uses the FUTURES lot (same LOT_SIZES) so the number is what
moves in the account, not a percentage that flatters a 120-lot MIDCAP loss.

AUTHORITY NOTE: the ledger on a VIEWER laptop is a STALE COPY -- the VM's cron owns it.
`ledger_age_days()` lets the UI say so instead of implying the numbers are current.
"""
from __future__ import annotations

import datetime as dt

from core.constants import IST, LOT_SIZES

# NSE bhavcopy-style name -> Fyers index symbol (the ledger stores the short name)
_SYM = {"NIFTY": "NSE:NIFTY50-INDEX", "BANKNIFTY": "NSE:NIFTYBANK-INDEX",
        "FINNIFTY": "NSE:FINNIFTY-INDEX", "MIDCPNIFTY": "NSE:MIDCPNIFTY-INDEX"}
_LABEL = {"NIFTY": "NIFTY 50", "BANKNIFTY": "BANK NIFTY",
          "FINNIFTY": "FIN NIFTY", "MIDCPNIFTY": "MIDCAP NIFTY"}

BACKTEST_LO, BACKTEST_HI = 10.0, 13.0     # bps/night the 4yr backtest expects
REVIEW_GATE = 25                          # paper nights before any real-capital discussion

# The ledger stores DATES + PRICES, never clock stamps — the times are FIXED BY THE RULE,
# so they are derived here rather than invented per-row:
#   entry = the day's CLOSE (the emit cron fires ~15:28; entry_px is the true <=15:30 close)
#   exit  = btst_signal.EXIT_T (09:30 next morning)
# Tradeable window 15:10-15:30, VALIDATED leak-free (backtest_btst_entry.py, 2yr n=754; the
# 5min bars are BAR-START labelled so the as-of read uses hm < T, no 5-min lookahead): the
# clr>=0.66 signal is already set by 15:10 on 97% of days; commit-at-15:10 +14.5 / -15:15 +13.9
# bps vs close-entry +14.9 — all indistinguishable, all clear +10-13. Fade P&L on the ~5% that
# fade is small-sample noise (flips +-6bps). Enter anywhere in the window; it's safe.
ENTRY_T = "15:10–15:30"                   # executed into the close; entry_px = that close
def _exit_t() -> str:
    import btst_signal as bs
    return bs.EXIT_T.strftime("%H:%M")


def _lot(sym_short: str):
    return LOT_SIZES.get(_SYM.get(sym_short, ""), None)


def futures_contract(date) -> str:
    """The contract you actually BUY: the near-month index FUTURES (all four indices share the
    same last-Tuesday monthly expiry). BTST is NOT an option — there is no strike and no CE/PE.
    That is precisely why it works: a futures contract has NO THETA, so it can be held
    overnight without decaying. An option would bleed against you."""
    try:
        import btst_signal as bs
        from core.market_calendar import index_future_expiries
        e = index_future_expiries(bs._as_date(date), 1)[0]
        return f"{e:%d%b} FUT"
    except Exception:
        return "near-month FUT"


def notional(entry_px, sym_short) -> "int | None":
    """Contract value of ONE lot = index level × lot size. This is the EXPOSURE you carry
    overnight (margin is only ~10–12% of it) — the number that decides whether a gap-down
    hurts or ruins."""
    lot = _lot(sym_short)
    if not (lot and entry_px):
        return None
    try:
        return round(float(entry_px) * lot)
    except (TypeError, ValueError):
        return None


def net_rupees(entry_px, net_bps, sym_short) -> "int | None":
    """Net Rs on ONE futures lot = entry x (net_bps/1e4) x lot. net_bps already carries the
    3bps round-trip cost, so this is the account-true number, not a gross point move."""
    lot = _lot(sym_short)
    if not (lot and entry_px and net_bps is not None):
        return None
    try:
        return round(float(entry_px) * (float(net_bps) / 1e4) * lot)
    except (TypeError, ValueError):
        return None


def load():
    """(open_rows, stale_rows, closed_rows) from the paper ledger. Empty lists if absent."""
    import btst_signal as bs
    led = bs._load_ledger()
    if led is None or led.empty:
        return [], [], []
    fresh, stale = bs.open_positions(led)
    closed = led[led["status"] == "CLOSED"]

    xt = _exit_t()

    def _open_row(r, is_stale):
        d = bs._as_date(r.date)
        return {"index": _LABEL.get(r.sym, r.sym), "sym": r.sym,
                "clr": round(float(r.clr), 3) if r.clr is not None else None,
                "signal_date": str(d),
                # what you actually place: BUY <n> lots of the near-month FUTURES. No strike,
                # no CE/PE — the old panel showed neither the contract nor the exposure.
                "action": "BUY (long)",
                "contract": f"{r.sym} {futures_contract(d)}",
                "triggered": f"{d} {ENTRY_T}",
                "exits": f"{bs._next_trading_day(d)} {xt}",
                "entry": round(float(r.entry_px), 2) if r.entry_px else None,
                "notional": notional(r.entry_px, r.sym),
                "lot": _lot(r.sym), "stale": is_stale}

    def _closed_row(r):
        d, xd = bs._as_date(r.date), bs._as_date(r.exit_date)
        return {"index": _LABEL.get(r.sym, r.sym), "sym": r.sym,
                "clr": round(float(r.clr), 3) if r.clr is not None else None,
                "held": f"{d}→{xd}",
                "triggered": f"{d} {ENTRY_T}",
                "exited": f"{xd} {xt}",
                "entry": round(float(r.entry_px), 2) if r.entry_px else None,
                "exit": round(float(r.exit_px), 2) if r.exit_px else None,
                "net_bps": round(float(r.net_bps), 1) if r.net_bps is not None else None,
                "rupee": net_rupees(r.entry_px, r.net_bps, r.sym)}

    return ([_open_row(r, False) for r in fresh.itertuples()],
            [_open_row(r, True) for r in stale.itertuples()],
            [_closed_row(r) for r in closed.itertuples()])


def summary(closed_rows) -> dict:
    """Paper scorecard. `tracking` compares the realised mean to the backtest expectation —
    the only question that matters before the review gate."""
    bps = [r["net_bps"] for r in closed_rows if r.get("net_bps") is not None]
    rs = [r["rupee"] for r in closed_rows if r.get("rupee") is not None]
    n = len(bps)
    if not n:
        return {"n": 0, "wins": 0, "win_pct": None, "mean_bps": None,
                "total_rupees": None, "worst_bps": None, "tracking": "no closed nights yet",
                "gate_left": REVIEW_GATE}
    wins = sum(1 for b in bps if b > 0)
    mean = sum(bps) / n
    tracking = ("above expectation" if mean > BACKTEST_HI else
                "tracking" if mean >= BACKTEST_LO else "BELOW expectation")
    return {"n": n, "wins": wins, "win_pct": round(100.0 * wins / n, 1),
            "mean_bps": round(mean, 1), "total_rupees": sum(rs) if rs else None,
            "worst_bps": round(min(bps), 1), "tracking": tracking,
            "gate_left": max(REVIEW_GATE - n, 0)}


def per_index(closed_rows) -> list[dict]:
    out = {}
    for r in closed_rows:
        d = out.setdefault(r["index"], {"index": r["index"], "n": 0, "wins": 0,
                                        "bps": 0.0, "rupee": 0})
        d["n"] += 1
        if (r["net_bps"] or 0) > 0:
            d["wins"] += 1
        d["bps"] += r["net_bps"] or 0.0
        d["rupee"] += r["rupee"] or 0
    for d in out.values():
        d["mean_bps"] = round(d["bps"] / d["n"], 1)
        d["win_pct"] = round(100.0 * d["wins"] / d["n"], 0)
    return sorted(out.values(), key=lambda d: d["mean_bps"])


def forming_candidates(as_of=None) -> list[dict]:
    """LIVE forming close-strength per index from the tick mirror, AS OF now (leak-safe: the
    read is capped at as_of, so it only sees ticks up to this minute). During 15:10-15:30 this
    previews which indices are shaping into BTST-LONG candidates (clr>=0.66) BEFORE the ~15:28
    emit cron logs them — the missing piece that makes the 15:10-15:30 window actionable.

    PROVISIONAL: the final minutes can still nudge clr, but 97% of final-strong closes are
    already >=0.66 by 15:10 (backtest_btst_entry). Each row: index, clr, spot, candidate, note."""
    import datetime as dt
    from core.mirror_io import read_mirror
    import btst_signal as bs
    now = as_of or dt.datetime.now(IST)
    today = now.date()
    out = []
    for short, fy in bs.FY.items():
        row = {"index": _LABEL.get(short, short), "sym": short,
               "clr": None, "spot": None, "candidate": False, "note": ""}
        try:
            tk = read_mirror("ticks", today.isoformat(), now, f"NSE:{fy}-INDEX")
            if tk is None or len(tk) < 30:
                row["note"] = "warming up"
                out.append(row); continue
            m = tk[(tk["ts"].dt.date == today) & (tk["ts"].dt.time >= dt.time(9, 15))]
            if len(m) < 30:
                row["note"] = "warming up"
                out.append(row); continue
            ltp = m["ltp"].to_numpy(float)
            hi, lo, px = float(ltp.max()), float(ltp.min()), float(m.iloc[-1]["ltp"])
            clr = (px - lo) / (hi - lo) if hi > lo else 0.5
            row.update({"clr": round(clr, 3), "spot": round(px, 1),
                        "candidate": clr >= bs.CLR_TH})
        except Exception as exc:
            row["note"] = f"err: {exc}"
        out.append(row)
    return out


def ledger_age_days() -> "int | None":
    """Whole days since the ledger file was last written. On a VIEWER this is a stale copy of
    the VM's; surfacing the age stops the UI implying the numbers are live."""
    import btst_signal as bs
    try:
        if not bs.LEDGER.exists():
            return None
        mt = dt.datetime.fromtimestamp(bs.LEDGER.stat().st_mtime, tz=IST)
        return (dt.datetime.now(IST).date() - mt.date()).days
    except Exception:
        return None
