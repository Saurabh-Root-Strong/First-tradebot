"""
trade_journal.py — a memory engine for YOUR trades, not the machine's.

WHY THIS EXISTS
---------------
The harvest (harvest_scout_flow.py) records the MACHINE's intraday scout trades — a strategy
this project has measured as dead on options (5 independent ways). Nothing anywhere recorded
the thing that actually works: the user's DISCRETIONARY price-action reading. So the system
had memory for the losing strategy and none for the winning one. This closes that.

THE DESIGN, AND THE FAILURE MODES IT ANSWERS
--------------------------------------------
Journals die of friction and lie from hindsight. Both are handled by ONE decision: you log in
the EVENING with a timestamp, and the board context is RECONSTRUCTED as-of that minute using
the same leakage-safe as_of replay the backtests use. So:

  * near-zero friction  — 3 fields typed when you are calm, not mid-trade at an office desk
  * no hindsight bias   — you cannot "remember" the context favourably; it is recomputed from
                          the tape as it stood at your entry minute, blind to what came after
  * no selective memory — `--skip` logs the setups you PASSED on; a journal of only taken
                          trades cannot tell you what your discipline is worth
  * survives the office — evening laptop work, no mid-session interaction required
  * overnight-capable   — --exit-date supports a BTST carry (the one validated edge here)

WHAT IT DOES NOT DO — DELIBERATELY
----------------------------------
It does not adapt any rule. At ~3 trades/day an auto-learner fits noise by construction (this
project has the receipts: a 7-trade week said 40/60 was better; the 31-day sample said the
opposite). This is a MEASUREMENT loop, not an adaptive engine. It answers one question, slowly
and honestly: in WHICH CONDITIONS does the human actually win?

USAGE
-----
    # a trade you took (times are IST, 24h HH:MM; date defaults to today)
    python trade_journal.py log --sym NIFTY --side CE --entry 11:45 --exit 12:30 \
        --entry-px 85.7 --exit-px 96.2 --lots 1 --why "coil break above 15m wall"

    # an index-level / futures / BTST trade (no premium): give index prices instead
    python trade_journal.py log --sym BANK --side LONG --entry 15:20 --exit 09:30 \
        --exit-date 2026-07-29 --entry-px 57100 --exit-px 57340 --why "post-3pm strong close"

    # a setup you deliberately PASSED (discipline is data too)
    python trade_journal.py skip --sym FIN --side PE --at 13:15 --why "into a x3 support"

    python trade_journal.py review              # where do YOU actually win?
    python trade_journal.py list --n 20
"""
from __future__ import annotations

import argparse
import datetime
import os
import sys

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from core.constants import IST, DATA_DIR, INDEX_SYMBOLS, LABELS, LOT_SIZES

OUT = DATA_DIR / "validation" / "trade_journal.parquet"

_ALIAS = {
    "NIFTY": "NSE:NIFTY50-INDEX", "NIFTY50": "NSE:NIFTY50-INDEX", "N": "NSE:NIFTY50-INDEX",
    "BANK": "NSE:NIFTYBANK-INDEX", "BANKNIFTY": "NSE:NIFTYBANK-INDEX", "B": "NSE:NIFTYBANK-INDEX",
    "FIN": "NSE:FINNIFTY-INDEX", "FINNIFTY": "NSE:FINNIFTY-INDEX", "F": "NSE:FINNIFTY-INDEX",
    "MIDCAP": "NSE:MIDCPNIFTY-INDEX", "MIDCPNIFTY": "NSE:MIDCPNIFTY-INDEX",
    "M": "NSE:MIDCPNIFTY-INDEX",
}


def _resolve(s: str) -> str:
    k = (s or "").upper().replace(" ", "").replace("-INDEX", "")
    if k in _ALIAS:
        return _ALIAS[k]
    for sym in INDEX_SYMBOLS:
        if k in sym.upper():
            return sym
    raise SystemExit(f"unknown symbol '{s}' — use NIFTY / BANK / FIN / MIDCAP")


def _ts(date: str, hhmm: str) -> datetime.datetime:
    hh, mm = map(int, hhmm.split(":"))
    return datetime.datetime.combine(datetime.date.fromisoformat(date),
                                     datetime.time(hh, mm), tzinfo=IST)


def _has_capture(date: str) -> bool:
    """Is there real captured tape for this date? Guard added after a live trap: logging with
    a date that has NO capture still returned a PLAUSIBLE-LOOKING board tag, because the
    structure functions silently fall back to prior-day bars. Context that is confidently
    wrong is worse than context that is blank — so a date with no tape stores nulls."""
    from core.constants import LIVE_DIR
    p = LIVE_DIR / f"{date}_ticks.parquet"
    try:
        return p.exists() and p.stat().st_size > 100_000
    except OSError:
        return False


def _last_captured_day() -> str | None:
    from core.constants import LIVE_DIR
    import glob as _glob
    ds = []
    for p in _glob.glob(str(LIVE_DIR / "*_ticks.parquet")):
        if os.path.getsize(p) > 100_000:
            s = os.path.basename(p).split("_")[0]
            try:
                datetime.date.fromisoformat(s); ds.append(s)
            except ValueError:
                pass
    return max(ds) if ds else None


def _default_date() -> str:
    """Default log date = TODAY if it has tape, else the LAST CAPTURED SESSION. You usually
    journal in the evening — and after midnight `today` is a blank day, which silently
    produced wrong context before this guard. Always announced, never assumed quietly."""
    today = datetime.date.today().isoformat()
    if _has_capture(today):
        return today
    last = _last_captured_day()
    if last:
        print(f"  (no tape for {today} — defaulting to last captured session {last}; "
              f"pass --date to override)")
        return last
    return today


def _context(sym: str, date: str, at: datetime.datetime) -> dict:
    """The board EXACTLY as it stood at `at` — leakage-safe (as_of replay). Every field is
    optional: a day with no capture must degrade to NULLS (never to a stale-bar guess), and
    never crash the log."""
    ctx = {}
    if not _has_capture(date):
        ctx["ctx_err"] = f"no capture for {date} — context left blank (not guessed)"
        return ctx
    try:
        import tradeboard as tb
    except Exception:
        return ctx
    try:
        reg = tb.day_regime(date, at) or {}
        ctx.update({"day_type": reg.get("label"), "day_er": reg.get("er"),
                    "day_rng_pct": reg.get("rng"), "vol_state": reg.get("vol_label"),
                    "vol_ratio": reg.get("vol_ratio")})
    except Exception as exc:                 # never swallow silently — a blank field must say why
        ctx["ctx_err"] = f"day_regime: {type(exc).__name__}: {exc}"[:120]
    try:
        for r in tb.scout_scan(date, at, 15, 60):
            if r.get("sym") != sym:
                continue
            lv = r.get("levels") or {}
            ctx.update({
                "board_tag": r.get("tag"), "board_loc": r.get("loc"),
                "ltf_struct": r.get("ltf_struct"), "htf_struct": r.get("htf_struct"),
                "ltf_candle": r.get("ltf_pattern"), "tape_pct": r.get("tape_pct"),
                "spot": lv.get("spot"), "band_lo": lv.get("band_lo"),
                "band_hi": lv.get("band_hi"),
                "sup": lv.get("support"), "sup_t": lv.get("sup_touches"),
                "res": lv.get("resistance"), "res_t": lv.get("res_touches"),
                "headroom_atr": lv.get("headroom_atr"), "wall_warn": lv.get("wall_warn"),
                "board_lean": lv.get("lean"),
            })
            break
    except Exception as exc:
        ctx["ctx_err"] = ((ctx.get("ctx_err") or "") +
                          f" | scout_scan: {type(exc).__name__}: {exc}")[:240]
    return ctx


def _append(row: dict):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([row])
    if OUT.exists():
        old = pd.read_parquet(OUT)
        new = pd.concat([old, new], ignore_index=True)
    # idempotent: one row per (date, sym, entry_time, kind) — a re-log corrects, never dupes
    new = new.drop_duplicates(subset=["date", "sym", "entry_time", "kind"], keep="last")
    new = new.sort_values(["date", "entry_time"]).reset_index(drop=True)
    new.to_parquet(OUT, index=False)
    return new


def cmd_log(a):
    sym = _resolve(a.sym)
    date = a.date or _default_date()
    ent = _ts(date, a.entry)
    exit_date = a.exit_date or date
    exd = _ts(exit_date, a.exit) if a.exit else None
    side = (a.side or "").upper()
    lots = a.lots or 1
    lot = LOT_SIZES.get(sym, 1) if side in ("CE", "PE") else 1
    pnl = pnl_pct = hold = None
    if a.entry_px and a.exit_px:
        if side in ("CE", "PE"):                 # option premium: long the option either way
            pnl = (a.exit_px - a.entry_px) * lot * lots
        elif side in ("LONG",):
            pnl = (a.exit_px - a.entry_px) * lots
        else:                                     # SHORT
            pnl = (a.entry_px - a.exit_px) * lots
        pnl_pct = (a.exit_px / a.entry_px - 1.0) * 100.0 * (1 if side != "SHORT" else -1)
    if exd:
        hold = round((exd - ent).total_seconds() / 60.0)
    ctx = _context(sym, date, ent)
    # did the board AGREE with you at that moment? (context, never a verdict on you)
    lean = ctx.get("board_lean")
    want = "UP" if side in ("CE", "LONG") else "DOWN"
    agree = None if not lean else ("agree" if lean == want else "disagree")
    row = {"kind": "trade", "date": date, "sym": sym, "label": LABELS.get(sym, sym),
           "side": side, "entry_time": a.entry, "exit_time": a.exit, "exit_date": exit_date,
           "entry_px": a.entry_px, "exit_px": a.exit_px, "lots": lots, "strike": a.strike,
           "pnl_rs": round(pnl, 1) if pnl is not None else None,
           "pnl_pct": round(pnl_pct, 2) if pnl_pct is not None else None,
           "hold_min": hold, "why": a.why, "note": a.note,
           "board_agree": agree, **ctx}
    df = _append(row)
    _p = f"Rs{pnl:+,.0f}" if pnl is not None else "n/a"
    print(f"logged {LABELS.get(sym,sym)} {side} {a.entry}->{a.exit or '?'} P&L {_p}")
    print(f"  context @{a.entry}: day {ctx.get('day_type')} · board {ctx.get('board_tag')} "
          f"· {agree or 'n/a'} · headroom {ctx.get('headroom_atr')} · tape {ctx.get('tape_pct')}%")
    print(f"  journal now {len(df)} rows -> {OUT.name}")


def cmd_skip(a):
    """A setup you PASSED. Discipline is data: if your skips would have lost, the skip IS
    the edge; if they would have won, hesitation is costing you."""
    sym = _resolve(a.sym)
    date = a.date or _default_date()
    at = _ts(date, a.at)
    ctx = _context(sym, date, at)
    row = {"kind": "skip", "date": date, "sym": sym, "label": LABELS.get(sym, sym),
           "side": (a.side or "").upper(), "entry_time": a.at, "why": a.why, "note": a.note,
           **ctx}
    df = _append(row)
    print(f"logged SKIP {LABELS.get(sym,sym)} {a.side or ''} @{a.at} — {a.why}")
    print(f"  journal now {len(df)} rows")


def _bucket_report(df, col, label, minn=3):
    if col not in df.columns:
        return
    g = df.dropna(subset=[col])
    if g.empty:
        return
    lines = []
    for k, s in g.groupby(col):
        p = s.dropna(subset=["pnl_rs"])
        if len(p) < minn:
            continue
        lines.append(f"    {str(k):<22} n={len(p):>3}  win {100*(p.pnl_rs>0).mean():>3.0f}%  "
                     f"avg Rs{p.pnl_rs.mean():>+8,.0f}  net Rs{p.pnl_rs.sum():>+9,.0f}")
    if not lines:                      # print nothing rather than an empty promise
        return
    print(f"\n  by {label}:")
    print("\n".join(lines))


def cmd_review(a):
    if not OUT.exists():
        print("no journal yet — log a trade first"); return
    df = pd.read_parquet(OUT)
    t = df[df.kind == "trade"].copy()
    sk = df[df.kind == "skip"]
    print("=" * 78)
    print(f"TRADE JOURNAL — {len(t)} trades, {len(sk)} skips, "
          f"{df.date.nunique()} days ({df.date.min()}..{df.date.max()})")
    print("=" * 78)
    p = t.dropna(subset=["pnl_rs"])
    if p.empty:
        print("no priced trades yet"); return
    w = p[p.pnl_rs > 0]; l = p[p.pnl_rs <= 0]
    aw = f"Rs{w.pnl_rs.mean():+,.0f}" if len(w) else "—"
    al = f"Rs{l.pnl_rs.mean():+,.0f}" if len(l) else "—"
    print(f"  net Rs{p.pnl_rs.sum():+,.0f} · win {100*len(w)/len(p):.0f}% "
          f"({len(w)}W/{len(l)}L) · avg win {aw} / avg loss {al}"
          + (f" = payoff {abs(w.pnl_rs.mean()/l.pnl_rs.mean()):.2f}x" if len(l) and len(w) else ""))
    if "hold_min" in p and p.hold_min.notna().any():
        print(f"  median hold {p.hold_min.median():.0f} min")
    # THE question this whole file exists to answer
    _bucket_report(p, "day_type", "DAY TYPE (trend/mid/chop)")
    _bucket_report(p, "board_agree", "BOARD AGREEMENT (did the machine lean your way)")
    _bucket_report(p, "board_tag", "BOARD SETUP at your entry")
    _bucket_report(p, "sym", "INDEX")
    _bucket_report(p, "side", "SIDE")
    if "wall_warn" in p.columns and p.wall_warn.notna().any():
        _bucket_report(p, "wall_warn", "ENTERED AT A WALL (True = <0.5 ATR to a multi-touch)")
    if "entry_time" in p.columns:
        p = p.copy(); p["hour"] = p.entry_time.str.slice(0, 2) + ":00"
        _bucket_report(p, "hour", "ENTRY HOUR")
    if len(p) < 12:
        print(f"\n  (breakdowns appear once a bucket has >=3 priced trades — you have {len(p)}."
              f" Expect ~1 month of logging before any pattern is worth reading.)")
    print("\n  READ: these are DESCRIPTIONS, not rules. Below ~30 trades per bucket they are")
    print("  anecdotes. The point is to find WHERE YOU WIN, then trade more of that and less")
    print("  of the rest — no parameter here auto-changes anything.")
    if len(sk):
        print(f"\n  SKIPS logged: {len(sk)} — review them by hand; if your passes would have")
        print("  won, hesitation is the cost; if they would have lost, discipline is the edge.")


def cmd_list(a):
    if not OUT.exists():
        print("no journal yet"); return
    df = pd.read_parquet(OUT).tail(a.n)
    cols = [c for c in ("date", "label", "side", "strike", "entry_time", "exit_time",
                        "pnl_rs", "day_type", "board_tag", "board_agree", "why")
            if c in df.columns]
    print(df[cols].to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description="your discretionary trade journal")
    sub = ap.add_subparsers(dest="cmd", required=True)

    lg = sub.add_parser("log", help="log a trade you took")
    lg.add_argument("--sym", required=True); lg.add_argument("--side", required=True,
                    help="CE / PE / LONG / SHORT")
    lg.add_argument("--entry", required=True, help="HH:MM IST")
    lg.add_argument("--exit", help="HH:MM IST")
    lg.add_argument("--date"); lg.add_argument("--exit-date", dest="exit_date")
    lg.add_argument("--entry-px", type=float); lg.add_argument("--exit-px", type=float)
    lg.add_argument("--lots", type=int, default=1); lg.add_argument("--strike")
    lg.add_argument("--why", default=""); lg.add_argument("--note", default="")
    lg.set_defaults(fn=cmd_log)

    sk = sub.add_parser("skip", help="log a setup you deliberately passed")
    sk.add_argument("--sym", required=True); sk.add_argument("--side", default="")
    sk.add_argument("--at", required=True, help="HH:MM IST"); sk.add_argument("--date")
    sk.add_argument("--why", default=""); sk.add_argument("--note", default="")
    sk.set_defaults(fn=cmd_skip)

    rv = sub.add_parser("review", help="where do YOU actually win?")
    rv.set_defaults(fn=cmd_review)

    ls = sub.add_parser("list"); ls.add_argument("--n", type=int, default=15)
    ls.set_defaults(fn=cmd_list)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
