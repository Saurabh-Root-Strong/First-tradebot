"""horizon_ledger.py — the scout ledger the SELECTED trade horizon would have produced.

WHY THIS EXISTS. The live ledger is replayed from `scout_alerts`, and only the headless
poller writes that log — at _ALERT_TF=60, for every viewer. So the Charts trade-horizon
dropdown changes the READ (band, cover%, verdict, EV) and never the legs: there is exactly
one ledger and it is the 60m one. Ask "how many trades did 5m→15m do today?" and the board
has no answer, because nothing ever traded that horizon.

This answers it by SIMULATION, off the same captured ticks and chain the live engine reads,
with the same discipline the real ledger enforces:

  * one position per index at a time — no stacking
  * open only on a TRADE verdict carrying a direction (never on RANGE)
  * no new leg after 15:10 — measured: legs opened 15:10-15:30 returned mean -50.8%,
    win 13% (n=47) vs -5.6%, win 37% for the rest of the day
  * HOLD = THE HORIZON, not the poller's fixed 90m cap. This is the whole point: a horizon
    dropdown whose positions all run 90m is not a horizon dropdown.
  * exit on horizon elapse / direction flip / the bell, whichever comes first
  * P&L on the real captured ATM premium, net of the measured Zerodha round trip

IT IS A SIMULATION AND IS LABELLED AS ONE. No order was placed, no alert was written, and
the fills are captured mid-prices with no slippage beyond the modelled spread — so it is,
if anything, optimistic. It exists to size a horizon before you trade it, not to claim a
track record.

LEAK SAFETY. Every scan is pinned to its own grid instant and the whole walk is capped at
`as_of`, so a ghost/practice session cannot see past its sealed clock and a replay cannot
see past the viewed minute. The engine call is `verdict_only=True`, which is the same gate
the live board uses — 16ms vs 440ms for the full scan, which is what makes a full 5m day
grid (288 calls) run in ~4.5s instead of ~127s.
"""
from __future__ import annotations

import datetime as dt
import functools
import logging

from core.constants import INDEX_SYMBOLS, LABELS, LOT_SIZES
from core import broker_costs as bcosts

log = logging.getLogger(__name__)

OPEN_SETTLE = dt.time(9, 35)     # same warmup gate the live scout uses
NO_NEW_AFTER = dt.time(15, 10)   # do not open what the bell will immediately close
BELL = dt.time(15, 30)


def _grid(day: str, ltf: int, as_of: dt.datetime | None) -> list:
    """Decision instants: every ltf-minute mark from the settle to the bell, capped at
    as_of so a sealed practice session cannot be walked past its own clock."""
    from intraday_scout import IST
    d = dt.date.fromisoformat(day)
    t = dt.datetime.combine(d, OPEN_SETTLE).replace(tzinfo=IST)
    end = dt.datetime.combine(d, BELL).replace(tzinfo=IST)
    if as_of is not None:
        a = as_of if as_of.tzinfo else as_of.replace(tzinfo=IST)
        end = min(end, a)
    out = []
    while t <= end:
        out.append(t)
        t += dt.timedelta(minutes=ltf)
    return out


def _key(as_of):
    """Cache key for as_of — bucketed to the minute so a live board's per-second clock
    does not blow the cache on every tick."""
    return None if as_of is None else as_of.strftime("%Y-%m-%dT%H:%M")


@functools.lru_cache(maxsize=32)
def _simulate(day: str, ltf: int, horizon_min: int, as_of_key: str | None) -> tuple:
    import intraday_scout as sc
    from intraday_scout import IST
    as_of = (dt.datetime.strptime(as_of_key, "%Y-%m-%dT%H:%M").replace(tzinfo=IST)
             if as_of_key else None)
    pts = _grid(day, ltf, as_of)
    if not pts:
        return (), (), {"note": "no decision instants in this window"}
    opens, closed = [], []
    n_scans = 0
    for sym in INDEX_SYMBOLS:
        held = None
        for t in pts:
            try:
                r = sc.scan_index(sym, ltf, date=day, as_of=t, horizon_min=horizon_min,
                                  verdict_only=True)
                n_scans += 1
            except Exception:
                continue
            if not r.get("has_data"):
                continue
            side = (r.get("direction") or "").strip()
            is_trade = str(r.get("verdict", "")).startswith("TRADE")
            spot = r.get("spot")
            # ── manage an open leg ────────────────────────────────────────────────
            if held is not None:
                due = held["open_ts"] + dt.timedelta(minutes=horizon_min)
                flipped = is_trade and side and side != held["dir"]
                if t >= due or flipped or t.time() >= BELL:
                    held["close_ts"] = t
                    held["outcome"] = ("flipped" if flipped and t < due else
                                       "bell" if t.time() >= BELL else "horizon")
                    held["exit_spot"] = spot
                    closed.append(held)
                    held = None
                    # a flip immediately opens the other side, exactly like the live poller
                    if not flipped:
                        continue
            # ── open a new leg ────────────────────────────────────────────────────
            if held is None and is_trade and side in ("CE", "PE") and spot:
                if t.time() > NO_NEW_AFTER:
                    continue
                strike = sc._atm(spot, sym)
                held = {"sym": sym, "label": LABELS.get(sym, sym), "dir": side,
                        "strike": strike, "open_ts": t, "entry_spot": spot,
                        "close_ts": None, "outcome": None, "exit_spot": None}
        if held is not None:                      # still open at the walk's end
            opens.append(held)

    # ── price every leg on the captured chain, then net the real cost ─────────────
    import intraday_scout as sc
    for e in list(closed) + list(opens):
        lot = LOT_SIZES.get(e["sym"])
        try:
            e["entry"] = sc._opt_premium(e["sym"], day, e["open_ts"].to_pydatetime()
                                         if hasattr(e["open_ts"], "to_pydatetime")
                                         else e["open_ts"], e["strike"], e["dir"])
        except Exception:
            e["entry"] = None
        t_exit = e["close_ts"] or (as_of or pts[-1])
        try:
            e["exit"] = sc._opt_premium(e["sym"], day, t_exit, e["strike"], e["dir"])
        except Exception:
            e["exit"] = None
        en, ex = e.get("entry"), e.get("exit")
        e["pnl_pct"] = round((ex / en - 1) * 100, 1) if (en and ex and en > 0) else None
        e["gross"] = round((ex - en) * lot) if (en and ex and lot) else None
        c = bcosts.roundtrip(en, ex, lot) if (en and ex and lot) else None
        e["cost"] = round(c["total"]) if c else None
        e["net"] = (e["gross"] - e["cost"]) if (e["gross"] is not None
                                                and e["cost"] is not None) else None
        # index-side move, signed by the leg's direction
        if e.get("entry_spot") and e.get("exit_spot"):
            mv = (e["exit_spot"] / e["entry_spot"] - 1) * 100
            e["idx_pct"] = round(mv if e["dir"] == "CE" else -mv, 3)
        else:
            e["idx_pct"] = None
        e["held_min"] = (round((e["close_ts"] - e["open_ts"]).total_seconds() / 60)
                         if e["close_ts"] else None)
    meta = {"scans": n_scans, "grid": len(pts), "ltf": ltf, "horizon": horizon_min,
            "day": day, "as_of": as_of_key, "note": ""}
    return tuple(opens), tuple(closed), meta


def simulate(day: str, ltf: int, horizon_min: int, as_of=None) -> dict:
    """Public entry. Returns {open, closed, meta, stats}. Never raises."""
    try:
        op, cl, meta = _simulate(day, int(ltf), int(horizon_min), _key(as_of))
    except Exception as exc:                                     # noqa: BLE001
        log.warning("horizon ledger sim failed for %s %sm→%sm: %s",
                    day, ltf, horizon_min, exc)
        return {"open": [], "closed": [], "meta": {"note": f"unavailable ({exc})"},
                "stats": {}}
    op, cl = list(op), list(cl)
    graded = [e for e in cl if e.get("net") is not None]
    wins = sum(1 for e in graded if e["net"] > 0)
    idx_graded = [e for e in cl if e.get("idx_pct") is not None]
    idx_wins = sum(1 for e in idx_graded if e["idx_pct"] > 0)
    stats = {
        "n_open": len(op), "n_closed": len(cl), "n_graded": len(graded),
        "wins": wins, "losses": len(graded) - wins,
        "win_pct": round(100 * wins / len(graded), 1) if graded else None,
        # the INDEX-side hit rate: was the arrow's DIRECTION right, before the option
        # wrapper's theta and spread got to it. The two diverge a lot, and the gap IS
        # the cost floor made visible.
        "idx_win_pct": (round(100 * idx_wins / len(idx_graded), 1)
                        if idx_graded else None),
        "gross": sum(e["gross"] for e in cl if e.get("gross") is not None),
        "cost": sum(e["cost"] for e in cl if e.get("cost") is not None),
        "net": sum(e["net"] for e in graded),
        "avg_pct": (round(sum(e["pnl_pct"] for e in cl if e.get("pnl_pct") is not None)
                          / max(1, sum(1 for e in cl if e.get("pnl_pct") is not None)), 1)
                    if cl else None),
        "avg_hold": (round(sum(e["held_min"] for e in cl if e.get("held_min"))
                           / max(1, sum(1 for e in cl if e.get("held_min"))))
                     if cl else None),
    }
    return {"open": op, "closed": cl, "meta": meta, "stats": stats}


def clear_cache() -> None:
    _simulate.cache_clear()
