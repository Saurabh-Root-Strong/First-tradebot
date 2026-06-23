"""
download_historical.py  —  Download OHLCV history for the NSE F&O universe.

Covers every F&O underlying (indices + ~211 stock futures, see fno_universe.py)
across 4 timeframes, sized for backtesting the Tradebot engine.

Saves one file per symbol × timeframe to  data/historical/<timeframe>/.
Default depth (Fyers serves intraday 2+ years back):
  intraday (5/15/60-min) : 730 calendar days  (~2 years)
  daily                  : 1500 calendar days (~4 years)

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

The job is resumable: existing files are skipped unless --force, so an
interrupted run (token expiry, network) can simply be re-run.

Usage:
  .venv\\Scripts\\python.exe download_historical.py                 # full universe, default depth
  .venv\\Scripts\\python.exe download_historical.py --indices-only  # just the 5 indices
  .venv\\Scripts\\python.exe download_historical.py --intraday-days 400 --daily-days 750
  .venv\\Scripts\\python.exe download_historical.py --timeframes 5min,15min
  .venv\\Scripts\\python.exe download_historical.py --format csv
  .venv\\Scripts\\python.exe download_historical.py --force         # re-download existing
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

from fno_universe import ALL_SYMBOLS, INDEX_SYMBOLS, LABELS

# ── Constants ──────────────────────────────────────────────────────────────────
from tradebot.adapters.broker.token import APP_ID, TOKEN_FILE   # single broker-token source
DATA_DIR     = Path("data") / "historical"
HISTORY_URL  = "https://api-t1.fyers.in/data/history"
from core.constants import IST   # single source of truth  # noqa: E402

# Full F&O universe (indices + stock futures) sourced from fno_universe.py.
SYMBOLS      = ALL_SYMBOLS
SYMBOL_LABEL = LABELS

# (tf_key, fyers_resolution, max_calendar_days_per_api_call, is_intraday, display_label)
# Conservative batch sizes — Fyers allows more but throttling is safer.
TIMEFRAMES = [
    ("5min",  "5",  60,  True,  "5-Min   intraday scalp"),
    ("15min", "15", 100, True,  "15-Min  intraday swing"),
    ("60min", "60", 100, True,  "1-Hour  BTST / positional"),
    ("daily", "D",  365, False, "Daily   swing positional"),
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
    out_dir  = DATA_DIR / tf_key
    out_path = out_dir / f"{_safe(sym)}_{tf_key}.{fmt}"

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

    out_dir.mkdir(parents=True, exist_ok=True)
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
        description="Download NSE F&O-universe OHLCV history from Fyers API"
    )
    ap.add_argument(
        "--intraday-days", type=int, default=730, metavar="N",
        help="Calendar days for 5/15/60-min timeframes.  Default: 730 (~2 yrs)",
    )
    ap.add_argument(
        "--daily-days", type=int, default=1500, metavar="N",
        help="Calendar days for the daily timeframe.  Default: 1500 (~4 yrs)",
    )
    ap.add_argument(
        "--indices-only", action="store_true",
        help="Download only the index symbols (skip the ~211 stock futures)",
    )
    ap.add_argument(
        "--timeframes", type=str, default="",
        help="Comma list to restrict timeframes, e.g. '5min,15min'.  Default: all",
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

    symbols = INDEX_SYMBOLS if args.indices_only else SYMBOLS

    timeframes = TIMEFRAMES
    if args.timeframes.strip():
        wanted     = {t.strip() for t in args.timeframes.split(",")}
        timeframes = [tf for tf in TIMEFRAMES if tf[0] in wanted]
        if not timeframes:
            print(f"ERROR: no known timeframes in '{args.timeframes}'. "
                  f"Choose from: {', '.join(tf[0] for tf in TIMEFRAMES)}")
            sys.exit(1)

    print(SEP2)
    print("  NSE F&O HISTORICAL DATA DOWNLOADER")
    print(SEP2)
    print(f"  Intraday depth : {args.intraday_days} cal days  "
          f"(≈ {int(args.intraday_days * 5/7 * 0.95)} trading days)")
    print(f"  Daily depth    : {args.daily_days} cal days")
    print(f"  Symbols        : {len(symbols)}  "
          f"({'indices only' if args.indices_only else 'full F&O universe'})")
    print(f"  Timeframes     : {', '.join(tf[0] for tf in timeframes)}")
    print(f"  Format         : {args.format}")
    print(f"  Output dir     : {DATA_DIR.resolve()}")
    print(f"  Force overwrite: {args.force}")
    print(SEP2)

    _validate_token()
    print()

    total_tasks = len(symbols) * len(timeframes)
    task_no     = 0
    summary     = []

    for sym in symbols:
        label = SYMBOL_LABEL.get(sym, sym)
        print(f"\n{'=' * 64}")
        print(f"  {label}  ({sym})")
        print(f"{'=' * 64}")

        for tf_key, resolution, batch_days, is_intraday, tf_label in timeframes:
            task_no += 1
            total_cal_days = args.intraday_days if is_intraday else args.daily_days
            print(f"\n  [{task_no:>3}/{total_tasks}]  {tf_label}")

            n, out_path = _download_one(
                sym, tf_key, resolution,
                batch_days, total_cal_days, args.format, args.force,
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
    ok_rows = sum(1 for _, _, st, _ in summary if st not in ("SKIP", "FAILED"))
    failed  = [(s, t) for s, t, st, _ in summary if st == "FAILED"]
    print(f"\n  Files OK: {ok_rows}   Skipped: "
          f"{sum(1 for _,_,st,_ in summary if st=='SKIP')}   "
          f"Failed: {len(failed)}")
    if failed:
        print("  FAILED tasks (re-run to retry — existing files are skipped):")
        for s, t in failed:
            print(f"    - {s} {t}")
    print(f"\n  All files saved under:  {DATA_DIR.resolve()}\\<timeframe>\\")
    print(f"\n  Load in Python:")
    print(f"    import pandas as pd")
    print(f"    df = pd.read_parquet('data/historical/5min/NSE_RELIANCE_EQ_5min.parquet')")
    print(f"    # ts column is IST, timezone-naive  (e.g. 2025-11-01 09:15:00)")
    print()


if __name__ == "__main__":
    main()
