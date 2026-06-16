"""
nse_oi.py — intraday FUTURES open-interest from NSE (the piece Fyers' feed lacks).

Fyers' data API does not serve intraday futures OI (verified: absent from /data/quotes
and /data/options-chain-v3). NSE's "OI Spurts — underlyings" endpoint does, for all
~216 F&O underlyings incl. the 4 indices. We poll it on a modest cadence and append to
a lock-free per-day mirror (data/intraday/live/<date>_futures_oi.parquet) that
intraday_tf reads — same pattern as the other captures, no DuckDB schema change.

  symbol  underlying name (NIFTY / BANKNIFTY / RELIANCE / …)
  oi      latest futures OI (contracts)
  chg_oi  change in OI vs previous day
  volume  futures volume

CAVEAT: NSE blocks datacenter IPs hard. This works from a home/office IP; from a cloud
VM it may 403. The handshake (cookie priming + browser headers) is required and is
re-done automatically on auth failure.

  .venv\\Scripts\\python.exe nse_oi.py            # one fetch, print the 4 indices
  .venv\\Scripts\\python.exe nse_oi.py --poll     # background-style loop (180s)
"""
from __future__ import annotations

import argparse
import datetime
import sys
import threading
import time
from pathlib import Path

import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
LIVE_DIR = Path(__file__).parent / "data" / "intraday" / "live"
_SPURTS = "https://www.nseindia.com/api/live-analysis-oi-spurts-underlyings"
_HOME   = "https://www.nseindia.com"
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/market-data/oi-spurts",
}
# OI-spurts underlying name → our index symbol
INDEX_MAP = {"NIFTY": "NSE:NIFTY50-INDEX", "BANKNIFTY": "NSE:NIFTYBANK-INDEX",
             "FINNIFTY": "NSE:FINNIFTY-INDEX", "MIDCPNIFTY": "NSE:MIDCPNIFTY-INDEX"}

_session: requests.Session | None = None
_lock = threading.Lock()


def _today() -> str:
    return datetime.datetime.now(tz=IST).date().isoformat()


def _new_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(_HEADERS)
    s.get(_HOME, timeout=12)                 # prime cookies
    return s


def fetch_oi() -> list[dict]:
    """Return [{symbol, oi, chg_oi, volume}] for all F&O underlyings, or [] on failure."""
    global _session
    for attempt in range(2):
        try:
            if _session is None:
                _session = _new_session()
            r = _session.get(_SPURTS, timeout=12)
            if r.status_code in (401, 403):
                _session = None              # cookies stale → re-handshake once
                continue
            if r.status_code != 200:
                return []
            data = r.json().get("data", [])
            return [{"symbol": d.get("symbol"), "oi": float(d.get("latestOI") or 0),
                     "chg_oi": float(d.get("changeInOI") or 0),
                     "volume": float(d.get("volume") or 0)} for d in data if d.get("symbol")]
        except Exception:
            _session = None
            time.sleep(1.0)
    return []


def record() -> int:
    """Fetch once and append a timestamped batch to today's mirror. Returns row count."""
    rows = fetch_oi()
    if not rows:
        return 0
    import pandas as pd
    ts = datetime.datetime.now(tz=IST)
    df = pd.DataFrame(rows)
    df.insert(0, "ts", ts)
    p = LIVE_DIR / f"{_today()}_futures_oi.parquet"
    with _lock:
        LIVE_DIR.mkdir(parents=True, exist_ok=True)
        if p.exists():
            try:
                old = pd.read_parquet(p)
                df = pd.concat([old, df], ignore_index=True)
            except Exception:
                pass
        df.to_parquet(p, index=False)
    return len(rows)


def poll_loop(interval: int = 180, market_hours_only: bool = True) -> None:
    """Background loop — call record() every `interval`s during market hours."""
    while True:
        now = datetime.datetime.now(tz=IST)
        in_mkt = datetime.time(9, 10) <= now.time() <= datetime.time(15, 35) and now.weekday() < 5
        if (not market_hours_only) or in_mkt:
            try:
                n = record()
                print(f"[nse_oi] {now:%H:%M:%S} captured {n} underlyings")
            except Exception as exc:
                print(f"[nse_oi] {now:%H:%M:%S} error: {exc}")
        time.sleep(interval)


def main() -> None:
    ap = argparse.ArgumentParser(description="NSE intraday futures OI")
    ap.add_argument("--poll", action="store_true", help="run the 180s capture loop")
    args = ap.parse_args()
    if args.poll:
        poll_loop()
        return
    rows = fetch_oi()
    if not rows:
        print("FAILED — NSE returned no data (blocked or down).")
        return
    by = {r["symbol"]: r for r in rows}
    print(f"OK — {len(rows)} underlyings. Index futures OI:")
    for name, _sym in INDEX_MAP.items():
        r = by.get(name)
        if r:
            print(f"  {name:10} OI {r['oi']/1e5:>8.1f}L   ΔOI {r['chg_oi']/1e5:>+7.1f}L   "
                  f"vol {r['volume']/1e5:>8.1f}L")


if __name__ == "__main__":
    main()
