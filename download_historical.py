"""
download_historical.py  —  Download OHLCV history for all 4 NSE indices.

Saves one file per symbol × timeframe to  data/historical/.
Default: 220 calendar days ≈ 150 trading days (excludes weekends/holidays).

Formats:
  parquet  — compact, fast, preserves dtypes  (recommended for backtesting)
  csv      — plain text, universally compatible

Columns in every file:
  ts        datetime  IST, timezone-naive (e.g. 2025-06-03 09:15:00)
  open      float
  high      float
  low       float
  close     float
  volume    int

Usage:
  .venv\\Scripts\\python.exe download_historical.py
  .venv\\Scripts\\python.exe download_historical.py --days 365
  .venv\\Scripts\\python.exe download_historical.py --format csv
  .venv\\Scripts\\python.exe download_historical.py --force   # re-download existing
"""

import argparse
import datetime
import sys
import time
from pathlib import Path

# Force UTF-8 on Windows consoles that default to cp1252 / cp936.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import requests

# ── Constants ──────────────────────────────────────────────────────────────────
APP_ID       = "WVDZUTO6HL-100"
TOKEN_FILE   = Path("access_token.txt")
DATA_DIR     = Path("data") / "historical"
HISTORY_URL  = "https://api-t1.fyers.in/data/history"
IST          = datetime.timezone(datetime.timedelta(hours=5, minutes=30))

SYMBOLS = [
    "NSE:NIFTY50-INDEX",
    "NSE:NIFTYBANK-INDEX",
    "NSE:FINNIFTY-INDEX",
    "NSE:MIDCPNIFTY-INDEX",
]

SYMBOL_LABEL = {
    "NSE:NIFTY50-INDEX":    "NIFTY 50",
    "NSE:NIFTYBANK-INDEX":  "BANK NIFTY",
    "NSE:FINNIFTY-INDEX":   "FIN NIFTY",
    "NSE:MIDCPNIFTY-INDEX": "MIDCAP NIFTY",
}

# (tf_key, fyers_resolution, max_calendar_days_per_api_call, display_label)
# Conservative batch sizes — Fyers allows more but throttling is safer.
TIMEFRAMES = [
    ("5min",  "5",  60,  "5-Min   intraday scalp"),
    ("15min", "15", 100, "15-Min  intraday swing"),
    ("60min", "60", 100, "1-Hour  BTST / positional"),
    ("daily", "D",  365, "Daily   swing positional"),
]

RETRY_DELAYS = [1.5, 3.0, 6.0]   # seconds between retries (exponential backoff)
API_SLEEP    = 0.4                # seconds between every API call (rate limit)

SEP  = "-" * 64
SEP2 = "=" * 64


# ── Helpers ────────────────────────────────────────────────────────────────────

def _safe(sym: str) -> str:
    """Symbol → safe filename stem."""
    return sym.replace(":", "_").replace("-", "_")


def _auth() -> str:
    return f"{APP_ID}:{TOKEN_FILE.read_text(encoding='utf-8').strip()}"


def _validate_token() -> None:
    import base64, json as _json
    if not TOKEN_FILE.exists():
        print("ERROR: access_token.txt not found.  Run fyers_auth.py first.")
        sys.exit(1)
    raw = TOKEN_FILE.read_text(encoding="utf-8").strip()
    try:
        payload = raw.split(".")[1]
        payload += "=" * (4 - len(payload) % 4)
        claims    = _json.loads(base64.urlsafe_b64decode(payload))
        remaining = claims.get("exp", 0) - time.time()
        if remaining <= 0:
            print("ERROR: Token has EXPIRED.  Run fyers_auth.py to refresh.")
            sys.exit(1)
        h, m = int(remaining // 3600), int((remaining % 3600) // 60)
        print(f"  Token OK  —  fy_id: {claims.get('fy_id', '?')}  "
              f"expires in {h}h {m}m")
    except SystemExit:
        raise
    except Exception:
        print("  Token: unreadable JWT — proceeding anyway")


# ── Single batch fetch ─────────────────────────────────────────────────────────

def _fetch_batch(sym: str, resolution: str,
                 d_from: str, d_to: str) -> pd.DataFrame:
    """
    One API call for [d_from, d_to].  Returns DataFrame or empty on failure.
    Retries up to len(RETRY_DELAYS) times with exponential back-off.
    """
    for attempt, delay in enumerate([0] + RETRY_DELAYS, start=1):
        if delay:
            time.sleep(delay)
        try:
            resp = requests.get(
                HISTORY_URL,
                headers={
                    "Authorization": _auth(),
                    "Content-Type":  "application/json",
                    "version":       "3",
                },
                params={
                    "symbol":      sym,
                    "resolution":  resolution,
                    "date_format": "1",
                    "range_from":  d_from,
                    "range_to":    d_to,
                    "cont_flag":   "1",
                },
                timeout=30,
            )
        except requests.RequestException as exc:
            tag = f"[attempt {attempt}]" if attempt <= len(RETRY_DELAYS) else "[failed]"
            print(f" {tag} network: {exc}", end="")
            continue

        try:
            data = resp.json()
        except Exception:
            print(f" [attempt {attempt}] bad JSON (HTTP {resp.status_code})", end="")
            continue

        if data.get("s") != "ok":
            msg = data.get("message") or data.get("errmsg") or data.get("s")
            print(f" [{msg}]", end="")
            # Non-retryable errors (invalid symbol, expired token, etc.)
            if "token" in str(msg).lower() or "expired" in str(msg).lower():
                print("\n  ERROR: token problem — run fyers_auth.py first.")
                sys.exit(1)
            return pd.DataFrame()

        candles = data.get("candles", [])
        if not candles:
            return pd.DataFrame()

        df = pd.DataFrame(candles,
                          columns=["ts", "open", "high", "low", "close", "volume"])

        # Convert Unix epoch → IST, then strip timezone for backtesting compat.
        # Stored as naive IST so any backtesting framework can use it directly.
        df["ts"] = (
            pd.to_datetime(df["ts"], unit="s", utc=True)
            .dt.tz_convert("Asia/Kolkata")
            .dt.tz_localize(None)
        )
        df[["open", "high", "low", "close"]] = (
            df[["open", "high", "low", "close"]].astype("float64")
        )
        df["volume"] = df["volume"].astype("int64")
        return df

    return pd.DataFrame()


# ── Full download for one symbol × timeframe ──────────────────────────────────

def _download_one(sym: str, tf_key: str, resolution: str,
                  batch_days: int, total_cal_days: int,
                  fmt: str, force: bool) -> tuple[int, Path]:
    """
    Download `total_cal_days` in chunks of `batch_days`, deduplicate, save.
    Returns (candle_count, out_path).  candle_count == 0 on failure/skip.
    """
    out_path = DATA_DIR / f"{_safe(sym)}_{tf_key}.{fmt}"

    if out_path.exists() and not force:
        size_kb = out_path.stat().st_size // 1024
        rows    = _row_count(out_path, fmt)
        print(f"    SKIP  {out_path.name}  ({rows:,} rows, {size_kb} KB already on disk)")
        return -1, out_path          # -1 = skipped

    now    = datetime.datetime.now(tz=IST)
    frames = []
    cursor = total_cal_days          # days remaining (counting backwards from today)

    while cursor > 0:
        chunk  = min(cursor, batch_days)
        d_from = (now - datetime.timedelta(days=cursor)).strftime("%Y-%m-%d")
        d_to   = (now - datetime.timedelta(days=cursor - chunk)).strftime("%Y-%m-%d")

        print(f"    {d_from} → {d_to}", end="", flush=True)
        df = _fetch_batch(sym, resolution, d_from, d_to)

        if not df.empty:
            frames.append(df)
            print(f"  {len(df):>6,} candles")
        else:
            print("  (empty)")

        cursor -= chunk
        time.sleep(API_SLEEP)

    if not frames:
        return 0, out_path

    result = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["ts"])
        .sort_values("ts")
        .reset_index(drop=True)
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        result.to_parquet(out_path, index=False)
    else:
        result.to_csv(out_path, index=False)

    return len(result), out_path


def _row_count(path: Path, fmt: str) -> int:
    try:
        if fmt == "parquet":
            import pyarrow.parquet as pq
            return pq.read_metadata(path).num_rows
        else:
            with open(path) as f:
                return sum(1 for _ in f) - 1   # subtract header
    except Exception:
        return 0


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download NSE index OHLCV history from Fyers API"
    )
    ap.add_argument(
        "--days", type=int, default=220,
        metavar="N",
        help="Calendar days to download (220 ≈ 150 trading days).  Default: 220",
    )
    ap.add_argument(
        "--format", choices=["parquet", "csv"], default="parquet",
        help="Output file format.  Default: parquet",
    )
    ap.add_argument(
        "--force", action="store_true",
        help="Re-download and overwrite files that already exist",
    )
    args = ap.parse_args()

    print(SEP2)
    print("  NSE HISTORICAL DATA DOWNLOADER")
    print(SEP2)
    print(f"  Calendar days  : {args.days}  (≈ {int(args.days * 5/7 * 0.95)} trading days)")
    print(f"  Symbols        : {len(SYMBOLS)}")
    print(f"  Timeframes     : {len(TIMEFRAMES)}")
    print(f"  Format         : {args.format}")
    print(f"  Output dir     : {DATA_DIR.resolve()}")
    print(f"  Force overwrite: {args.force}")
    print(SEP2)

    _validate_token()
    print()

    total_tasks = len(SYMBOLS) * len(TIMEFRAMES)
    task_no     = 0
    summary     = []

    for sym in SYMBOLS:
        label = SYMBOL_LABEL[sym]
        print(f"\n{'=' * 64}")
        print(f"  {label}  ({sym})")
        print(f"{'=' * 64}")

        for tf_key, resolution, batch_days, tf_label in TIMEFRAMES:
            task_no += 1
            print(f"\n  [{task_no:>2}/{total_tasks}]  {tf_label}")

            n, out_path = _download_one(
                sym, tf_key, resolution,
                batch_days, args.days, args.format, args.force,
            )

            if n == -1:          # skipped
                summary.append((label, tf_key, "SKIP", out_path))
            elif n > 0:
                size_kb = out_path.stat().st_size // 1024
                print(f"\n    OK  {n:,} candles  ->  {out_path.name}  ({size_kb} KB)")
                summary.append((label, tf_key, f"{n:,} rows", out_path))
            else:
                print(f"\n    FAIL  no data downloaded")
                summary.append((label, tf_key, "FAILED", out_path))

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{SEP2}")
    print("  DOWNLOAD SUMMARY")
    print(SEP2)
    print(f"  {'INDEX':<16}  {'TF':<6}  {'STATUS':<14}  FILE")
    print(f"  {'-'*15}  {'-'*5}  {'-'*13}  {'-'*30}")
    for sym_lbl, tf, status, path in summary:
        print(f"  {sym_lbl:<16}  {tf:<6}  {status:<14}  {path.name}")
    print(SEP2)
    print(f"\n  All files saved to:  {DATA_DIR.resolve()}")
    print(f"\n  Load in Python:")
    print(f"    import pandas as pd")
    print(f"    df = pd.read_parquet('data/historical/NSE_NIFTY50_INDEX_5min.parquet')")
    print(f"    # ts column is IST, timezone-naive  (e.g. 2025-11-01 09:15:00)")
    print()


if __name__ == "__main__":
    main()
