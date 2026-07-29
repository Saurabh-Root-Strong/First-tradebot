"""
update_historical_daily.py — keep the historical 5-min store CURRENT, incrementally.

WHY THIS EXISTS (a real, twice-repeated production bug)
-------------------------------------------------------
The TradeBoard ledger warms its 60m confirmation frame from PRIOR-day bars, which come from
data/historical/5min/*.parquet. That directory is gitignored, so a deploy never carries it,
and the VM keeps only TODAY's live parquet (older sessions roll into DuckDB). Result: the VM's
historical store froze on the day it was hand-copied and silently rotted.

  2026-07-27  VM store MISSING entirely  -> 60m could not warm  -> ledger fired 0 trades
  2026-07-29  VM store missing 07-28     -> 60m read 07-27 -> [HOLE] -> 07-29
              -> different HTF structure -> local showed 3 open trades, VM showed NONE

Both times the board looked fine and was quietly wrong. This script removes the whole bug
class: it appends each completed session's bars so the store is never more than a day stale.

WHY NOT `download_historical.py`
--------------------------------
That tool REBUILDS a file from a lookback window and OVERWRITES it. Run with a small window it
TRUNCATES 2 years of history to a few days (done by accident on 2026-07-25 and caught only by
a verify step). This script is strictly ADDITIVE: read existing -> append -> dedup on ts ->
sort -> write. It can never shorten the file, and it refuses to write a result smaller than
what it read.

TIMESTAMP CONTRACT: epoch -> IST -> tz-NAIVE, matching download_historical exactly. A mismatch
here would silently misalign every bar in the store.

    python update_historical_daily.py                # append the last few sessions
    python update_historical_daily.py --days 15      # wider catch-up after an outage
    python update_historical_daily.py --dry-run
"""
from __future__ import annotations

import argparse
import datetime
import sys
import time

import pandas as pd
import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from core.constants import IST, DATA_DIR, INDEX_SYMBOLS
from tradebot.adapters.broker.token import APP_ID, TOKEN_FILE

HISTORY_URL = "https://api-t1.fyers.in/data/history"
OUT_DIR = DATA_DIR / "historical" / "5min"
_MKT_CLOSE = datetime.time(15, 30)


def _safe(sym: str) -> str:
    return sym.replace(":", "_").replace("-", "_")


def _auth() -> str:
    return f"{APP_ID}:{TOKEN_FILE.read_text(encoding='utf-8').strip()}"


def _fetch(sym: str, d_from: str, d_to: str) -> pd.DataFrame:
    r = requests.get(HISTORY_URL,
                     headers={"Authorization": _auth(), "version": "3"},
                     params={"symbol": sym, "resolution": "5", "date_format": "1",
                             "range_from": d_from, "range_to": d_to, "cont_flag": "1"},
                     timeout=45)
    j = r.json()
    if j.get("s") != "ok" or not j.get("candles"):
        raise RuntimeError(f"{j.get('s')}: {str(j)[:120]}")
    df = pd.DataFrame(j["candles"], columns=["ts", "open", "high", "low", "close", "volume"])
    # SAME contract as download_historical: epoch -> IST -> naive
    df["ts"] = (pd.to_datetime(df["ts"], unit="s", utc=True)
                .dt.tz_convert("Asia/Kolkata").dt.tz_localize(None))
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype("float64")
    df["volume"] = df["volume"].astype("int64")
    return df


def update_symbol(sym: str, days: int, dry: bool) -> tuple[str, str]:
    path = OUT_DIR / f"{_safe(sym)}_5min.parquet"
    if not path.exists():
        return sym, "SKIP — no existing store (run download_historical once to seed it)"
    old = pd.read_parquet(path)
    old["ts"] = pd.to_datetime(old["ts"])
    n0 = len(old)
    last_old = old["ts"].max()
    d_to = datetime.date.today()
    d_from = d_to - datetime.timedelta(days=days)
    try:
        new = _fetch(sym, d_from.isoformat(), d_to.isoformat())
    except Exception as exc:
        return sym, f"FETCH FAILED — {exc}"
    # today's bars are only trustworthy once the session is done; before the close we still
    # append them (they are real, just partial) — the dedup on the next run replaces them.
    merged = (pd.concat([old, new], ignore_index=True)
              .drop_duplicates(subset=["ts"], keep="last")
              .sort_values("ts").reset_index(drop=True))
    if len(merged) < n0:                       # ADDITIVE-ONLY invariant, never truncate
        return sym, f"REFUSED — merge shrank the store ({n0} -> {len(merged)})"
    added = len(merged) - n0
    newest = merged["ts"].max()
    if dry:
        return sym, f"dry-run: would add {added} bars, newest {newest}"
    if added:
        merged.to_parquet(path, index=False)
    return sym, (f"+{added} bars ({n0} -> {len(merged)}), "
                 f"newest {last_old.date()} -> {newest.date()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=6,
                    help="calendar days to re-pull and merge (default 6 = covers a long weekend)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    now = datetime.datetime.now(IST)
    print(f"historical 5min top-up — {now:%Y-%m-%d %H:%M} IST"
          + ("  (session still open — today's bars will be partial, next run fixes them)"
             if now.time() < _MKT_CLOSE else ""))
    ok = True
    for sym in INDEX_SYMBOLS:
        s, msg = update_symbol(sym, a.days, a.dry_run)
        print(f"  {s:<24} {msg}")
        if "FAILED" in msg or "REFUSED" in msg:
            ok = False
        time.sleep(0.4)                        # be polite to /history
    print("done" if ok else "done WITH ERRORS")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
