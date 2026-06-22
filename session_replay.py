"""
session_replay.py  —  Intraday session time-machine and pattern finder.

Lets you replay any past trading session candle-by-candle and search for
historical moments that looked structurally similar to what you are seeing
right now — so you can validate whether a 12:30 PM signal also appeared at
10:10 AM before asking "is this setup worth taking?"

Usage (run from project root):
  python session_replay.py                              # list saved sessions
  python session_replay.py --report                     # full today's report
  python session_replay.py --date 2026-06-03 --report   # replay a past day
  python session_replay.py --oi                         # OI timeline today
  python session_replay.py --signals                    # signal history today
  python session_replay.py --setups --tf 15min          # 15-min setups today
  python session_replay.py --candles --tf 5min          # 5-min bars today
  python session_replay.py --sym NIFTY --report         # one symbol only
  python session_replay.py --similar --score 1.5        # find score ~+1.5
  python session_replay.py --similar --score 1.5 --pcr 0.85  # score + PCR
  python session_replay.py --export csv                 # export today to CSV
  python session_replay.py --stats                      # row counts for today

  ── Tick commands (raw WebSocket price stream) ──────────────────────────────
  python session_replay.py --ticks                      # NIFTY ticks today
  python session_replay.py --ticks --sym BANK           # BANKNIFTY ticks
  python session_replay.py --ticks --from 10:00 --to 10:05  # time slice
  python session_replay.py --tick-stats                 # velocity + volatility
  python session_replay.py --tick-stats --sym NIFTY --from 09:15 --to 12:30
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

from core.constants import IST   # single source of truth (fixed +5:30, no DST)
DB_DIR = Path(__file__).parent / "data" / "intraday"

try:
    import duckdb
    import pandas as pd
    pd.set_option("display.max_rows",     250)
    pd.set_option("display.max_columns",  40)
    pd.set_option("display.width",        180)
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
except ImportError:
    print("ERROR: pip install duckdb pandas")
    sys.exit(1)

# ── Symbol helpers ────────────────────────────────────────────────────────────

_SYM_MAP = {
    "NIFTY":     "NSE:NIFTY50-INDEX",
    "NIFTY50":   "NSE:NIFTY50-INDEX",
    "BANK":      "NSE:NIFTYBANK-INDEX",
    "BANKNIFTY": "NSE:NIFTYBANK-INDEX",
    "FIN":       "NSE:FINNIFTY-INDEX",
    "FINNIFTY":  "NSE:FINNIFTY-INDEX",
    "MIDCAP":    "NSE:MIDCPNIFTY-INDEX",
    "MIDCPNIFTY":"NSE:MIDCPNIFTY-INDEX",
}
_LABELS = {
    "NSE:NIFTY50-INDEX":    "NIFTY 50",
    "NSE:NIFTYBANK-INDEX":  "BANK NIFTY",
    "NSE:FINNIFTY-INDEX":   "FIN NIFTY",
    "NSE:MIDCPNIFTY-INDEX": "MIDCAP NIFTY",
}

def _sym(s: str) -> str:
    return _SYM_MAP.get(s.upper(), s)

def _label(sym_full: str | None) -> str:
    return _LABELS.get(sym_full or "", sym_full or "ALL INDICES")


# ── DB helpers ────────────────────────────────────────────────────────────────

_LIVE_DIR = DB_DIR / "live"   # Parquet snapshots written by dashboard after each flush

_PARQUET_TABLES = (
    "ticks", "candles", "oi_snapshots",
    "futures_quotes", "signals", "trade_setups",
)

_live_notice_shown = False   # print the "[live] snapshot" banner at most once per run


def _open_parquet_fallback(date: datetime.date):
    """
    Return an in-memory DuckDB connection with VIEWs over today's live Parquet
    snapshots.  These files are written by the dashboard every ~10 s after each
    checkpoint — no DuckDB lock, fully concurrent.

    Returns None if no Parquet snapshots exist yet (dashboard just started or
    first flush hasn't fired).
    """
    found = False
    con = duckdb.connect()          # in-memory — zero locking
    for tbl in _PARQUET_TABLES:
        pq = _LIVE_DIR / f"{date}_{tbl}.parquet"
        if pq.exists():
            # Forward slashes required by DuckDB on Windows
            con.execute(
                f"CREATE VIEW {tbl} AS SELECT * FROM read_parquet('{pq.as_posix()}')"
            )
            found = True
        # If Parquet missing: no view created; SQL errors caught by _run() → empty df
    return con if found else None


def _open(date: datetime.date, read_only: bool = True):
    """
    Open the DuckDB session file.  Falls back to live Parquet snapshots when
    the dashboard has an exclusive write-lock on the file (Windows limitation).
    """
    path = DB_DIR / f"{date}.duckdb"
    if path.exists():
        try:
            return duckdb.connect(str(path), read_only=read_only)
        except Exception as exc:
            if "being used by another process" in str(exc) or \
               "different configuration" in str(exc):
                # Dashboard holds the exclusive lock.  Try Parquet snapshots.
                con = _open_parquet_fallback(date)
                if con is not None:
                    global _live_notice_shown
                    if not _live_notice_shown:
                        print(
                            "  [live]  Dashboard running — reading Parquet snapshot "
                            "(≤10 s stale).  Full accuracy after market close.\n"
                        )
                        _live_notice_shown = True
                    return con
                print(
                    f"\n  {date}.duckdb is locked and no live snapshot exists yet.\n"
                    "  Wait ~10 s for the first checkpoint, then retry.\n"
                )
            else:
                print(f"  Cannot open {path}: {exc}")
            return None
    return None


def _run(date: datetime.date, sql: str) -> pd.DataFrame:
    con = _open(date)
    if con is None:
        return pd.DataFrame()
    try:
        df = con.execute(sql).df()
        return df
    except Exception as exc:
        # Silently swallow expected "table / view does not exist" errors — these
        # happen when a Parquet snapshot hasn't been written yet for a table.
        # DuckDB raises "Catalog Error: Table with name X does not exist!" or
        # "Table X not found" depending on version — catch both patterns.
        _msg = str(exc).lower()
        if "not found" not in _msg and "does not exist" not in _msg:
            print(f"  [query error] {exc}")
        return pd.DataFrame()
    finally:
        try:
            con.close()
        except Exception:
            pass


def _ist_time(df: pd.DataFrame, col: str = "ts") -> pd.DataFrame:
    """Add a 'time' column (HH:MM IST) derived from a UTC timestamp column."""
    if col not in df.columns:
        return df
    df = df.copy()
    ts = pd.to_datetime(df[col], utc=True)
    df.insert(0, "time", ts.dt.tz_convert("Asia/Kolkata").dt.strftime("%H:%M"))
    df = df.drop(columns=[col])
    return df


# ── Display helpers ───────────────────────────────────────────────────────────

_SEP = "─" * 120

def _hdr(title: str) -> None:
    print(f"\n{_SEP}\n  {title}\n{_SEP}")

def _show(df: pd.DataFrame, empty_msg: str = "  No data.") -> None:
    if df.empty:
        print(empty_msg)
    else:
        print(df.to_string(index=False))


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_list() -> None:
    _hdr("SAVED INTRADAY SESSIONS")
    files = sorted(DB_DIR.glob("*.duckdb"))
    if not files:
        print("  No session data found yet.")
        print(f"  Data will appear at:  {DB_DIR}")
        print("  Run the dashboard with the intraday_db hook active to start recording.")
        return
    total_rows = 0
    for f in files:
        date  = datetime.date.fromisoformat(f.stem)
        size  = f.stat().st_size / 1024
        try:
            con = duckdb.connect(str(f), read_only=True)
            counts = {
                t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                for t in ("ticks", "candles", "oi_snapshots", "futures_quotes", "signals", "trade_setups")
            }
            con.close()
            row_total = sum(counts.values())
            total_rows += row_total
            parts = "  ".join(f"{k[:3]}:{v:,}" for k, v in counts.items())
            print(f"  {date}  {size:6.1f} KB  {row_total:,} rows   {parts}")
        except Exception:
            print(f"  {date}  {size:.1f} KB  [error]")
    print(f"\n  {len(files)} sessions  |  {total_rows:,} total rows  |  path: {DB_DIR}")


def cmd_stats(date: datetime.date) -> None:
    _hdr(f"SESSION STATS  {date}")
    tables = ("ticks", "candles", "oi_snapshots", "futures_quotes", "signals", "trade_setups")
    for t in tables:
        df = _run(date, f"SELECT COUNT(*) AS rows FROM {t}")
        n  = int(df.iloc[0]["rows"]) if not df.empty else 0
        print(f"  {t:22s}  {n:>7,d} rows")


def cmd_oi(date: datetime.date, sym: str | None) -> None:
    sf = f"AND symbol = '{_sym(sym)}'" if sym else ""
    df = _run(date, f"""
        SELECT ts, symbol,
               ROUND(spot, 0)       AS spot,
               atm,
               ROUND(pcr, 3)        AS pcr,
               ROUND(atm_iv, 2)     AS atm_iv,
               call_wall, put_wall, max_pain,
               ROUND(total_call_oi / 1e6, 2) AS call_oi_M,
               ROUND(total_put_oi  / 1e6, 2) AS put_oi_M,
               ROUND(atm_call_prem, 1)        AS ce_prem,
               ROUND(atm_put_prem,  1)        AS pe_prem,
               ROUND(put_skew, 2)             AS put_skew
        FROM oi_snapshots
        WHERE date = '{date}' {sf}
        ORDER BY ts, symbol
    """)
    _hdr(f"OI SNAPSHOT TIMELINE  {date}  —  {_label(_sym(sym) if sym else None)}")
    _show(_ist_time(df))


def cmd_signals(date: datetime.date, sym: str | None) -> None:
    sf = f"AND symbol = '{_sym(sym)}'" if sym else ""
    df = _run(date, f"""
        SELECT ts, symbol,
               ROUND(weighted_score, 2) AS w_score,
               overall,
               ROUND(score_5min,  2) AS s5m,
               ROUND(score_15min, 2) AS s15m,
               ROUND(score_60min, 2) AS s60m,
               ROUND(score_daily, 2) AS sD,
               signal_5min  AS sig5m,
               signal_15min AS sig15m,
               ROUND(rsi_15min, 1)   AS rsi15,
               ROUND(close_price, 0) AS close,
               bull_tfs, bear_tfs
        FROM signals
        WHERE date = '{date}' {sf}
        ORDER BY ts, symbol
    """)
    _hdr(f"SIGNAL TIMELINE  {date}  —  {_label(_sym(sym) if sym else None)}")
    _show(_ist_time(df))


def cmd_setups(date: datetime.date, sym: str | None, tf: str | None) -> None:
    sf  = f"AND symbol = '{_sym(sym)}'" if sym else ""
    tff = f"AND timeframe = '{tf}'"     if tf  else ""
    df  = _run(date, f"""
        SELECT ts, symbol, timeframe,
               signal,
               ROUND(composite_score, 2) AS score,
               confidence, direction, phase,
               ROUND(l1_tech,     2) AS L1,
               ROUND(l2_oi,       2) AS L2,
               ROUND(l3_velocity, 2) AS L3,
               ROUND(l4_inst,     2) AS L4,
               ROUND(l5_futures,  2) AS L5,
               ROUND(l6_iv,       2) AS L6,
               ROUND(l7_pcr,      2) AS L7,
               ROUND(l8_maxpain,  2) AS L8,
               ROUND(l9_context,  2) AS L9,
               agreement,
               ROUND(spot,   0) AS spot,
               ROUND(atm_iv, 1) AS atm_iv
        FROM trade_setups
        WHERE date = '{date}' {sf} {tff}
        ORDER BY ts, symbol, timeframe
    """)
    tf_tag = f"  [{tf}]" if tf else ""
    _hdr(f"TRADE SETUP HISTORY  {date}{tf_tag}  —  {_label(_sym(sym) if sym else None)}")
    _show(_ist_time(df))


def cmd_candles(date: datetime.date, sym: str | None, tf: str) -> None:
    sf = f"AND symbol = '{_sym(sym)}'" if sym else ""
    df = _run(date, f"""
        SELECT ts, symbol,
               ROUND(open,  0) AS O,
               ROUND(high,  0) AS H,
               ROUND(low,   0) AS L,
               ROUND(close, 0) AS C,
               ROUND(volume / 1e6, 2) AS vol_M
        FROM candles
        WHERE date = '{date}' AND resolution = '{tf}' {sf}
        ORDER BY ts, symbol
    """)
    _hdr(f"CANDLES  {date}  [{tf}]  —  {_label(_sym(sym) if sym else None)}")
    _show(_ist_time(df))


def cmd_report(date: datetime.date, sym: str | None) -> None:
    """
    Chronological multi-stream session replay — the full time-machine view.
    Shows 15-min candles, OI snapshots, signal history, and 15-min setups
    in separate panels so you can align them by timestamp.
    """
    sf       = f"AND symbol = '{_sym(sym)}'" if sym else ""
    sym_full = _sym(sym) if sym else None
    _hdr(f"SESSION REPLAY  {date}  —  {_label(sym_full)}")

    # 15-min candles
    c15 = _run(date, f"""
        SELECT ts, symbol,
               ROUND(open,0) AS O, ROUND(high,0) AS H,
               ROUND(low,0)  AS L, ROUND(close,0) AS C,
               ROUND(volume/1e6,2) AS vol_M
        FROM candles WHERE date='{date}' AND resolution='15min' {sf}
        ORDER BY ts, symbol
    """)

    # OI snapshots
    oi = _run(date, f"""
        SELECT ts, symbol,
               ROUND(spot,0) AS spot,
               ROUND(pcr,3)  AS pcr,
               ROUND(atm_iv,1) AS atm_iv,
               call_wall, put_wall
        FROM oi_snapshots WHERE date='{date}' {sf}
        ORDER BY ts, symbol
    """)

    # Multi-TF signals
    sig = _run(date, f"""
        SELECT ts, symbol,
               ROUND(weighted_score,2) AS w_score,
               overall,
               signal_15min AS sig15m,
               ROUND(rsi_15min,1) AS rsi15,
               bull_tfs, bear_tfs
        FROM signals WHERE date='{date}' {sf}
        ORDER BY ts, symbol
    """)

    # 15-min trade setups
    setups = _run(date, f"""
        SELECT ts, symbol, signal,
               ROUND(composite_score,2) AS score,
               confidence, agreement, phase,
               ROUND(atm_iv,1) AS atm_iv
        FROM trade_setups
        WHERE date='{date}' AND timeframe='15min' {sf}
        ORDER BY ts, symbol
    """)

    for title, df in [
        ("15-MIN CANDLES",   c15),
        ("OI SNAPSHOTS",     oi),
        ("SIGNALS",          sig),
        ("15-MIN SETUPS",    setups),
    ]:
        sep = "─" * (118 - len(title))
        print(f"\n── {title} {sep}")
        _show(_ist_time(df))


def cmd_similar(
    sym:       str | None,
    score_mid: float | None,
    score_min: float,
    score_max: float,
    pcr_mid:   float | None,
    pcr_min:   float,
    pcr_max:   float,
    tf:        str,
    n_days:    int,
) -> None:
    """
    Pattern search: scan historical sessions for intraday moments whose
    composite score (and optionally PCR) were in the given range.

    Use case: "At 12:30 I see score=+1.8 with PCR=0.85 — did this setup
    appear earlier today or on past days? What happened to price after?"
    """
    sym_full   = _sym(sym) if sym else None
    sf         = f"AND symbol = '{sym_full}'" if sym_full else ""
    score_desc = (f"~{score_mid:.2f}" if score_mid is not None
                  else f"[{score_min:.2f},{score_max:.2f}]")
    pcr_desc   = (f"~{pcr_mid:.2f}" if pcr_mid is not None
                  else f"[{pcr_min:.2f},{pcr_max:.2f}]")

    _hdr(
        f"PATTERN SEARCH  [{_label(sym_full)}]  tf:{tf}  "
        f"score:{score_desc}  pcr:{pcr_desc}  last {n_days} sessions"
    )

    files = sorted(DB_DIR.glob("*.duckdb"), reverse=True)[:n_days]
    if not files:
        print("  No session data found.")
        return

    all_rows: list[pd.DataFrame] = []

    for f in files:
        date = datetime.date.fromisoformat(f.stem)
        try:
            con = duckdb.connect(str(f), read_only=True)

            setups = con.execute(f"""
                SELECT ts, symbol, signal, composite_score,
                       confidence, agreement, phase, spot, atm_iv,
                       l1_tech, l3_velocity, l9_context
                FROM trade_setups
                WHERE timeframe = '{tf}'
                  AND composite_score BETWEEN {score_min} AND {score_max}
                  {sf}
                ORDER BY ts
            """).df()

            oi_df = pd.DataFrame()
            if not setups.empty:
                oi_df = con.execute(f"""
                    SELECT ts, symbol, pcr, call_wall, put_wall
                    FROM oi_snapshots
                    ORDER BY ts
                """).df()

            con.close()

            if setups.empty:
                continue

            setups["ts"]   = pd.to_datetime(setups["ts"], utc=True)
            setups["date"] = date
            setups["time"] = (setups["ts"]
                              .dt.tz_convert("Asia/Kolkata")
                              .dt.strftime("%H:%M"))

            # Nearest-timestamp join to pick up PCR at time of setup
            if not oi_df.empty:
                oi_df["ts"] = pd.to_datetime(oi_df["ts"], utc=True)
                setups = pd.merge_asof(
                    setups.sort_values("ts"),
                    oi_df[["ts","symbol","pcr","call_wall","put_wall"]].sort_values("ts"),
                    on="ts", by="symbol",
                    tolerance=pd.Timedelta("5min"),
                    direction="nearest",
                )

            # PCR filter (post-join, so missing PCR rows are kept)
            if "pcr" in setups.columns:
                mask = setups["pcr"].isna() | setups["pcr"].between(pcr_min, pcr_max)
                setups = setups[mask]

            all_rows.append(setups)

        except Exception:
            continue

    if not all_rows:
        print("  No matching setups found across the searched sessions.")
        return

    result = pd.concat(all_rows, ignore_index=True)

    # Build display frame
    keep = ["date","time","symbol","signal","composite_score","confidence",
            "agreement","phase"]
    if "pcr" in result.columns:
        keep += ["pcr","call_wall","put_wall"]
    keep += ["spot","atm_iv","l1_tech","l3_velocity","l9_context"]
    keep  = [c for c in keep if c in result.columns]
    disp  = result[keep].rename(columns={
        "composite_score": "score",
        "call_wall":       "c_wall",
        "put_wall":        "p_wall",
        "l1_tech":         "L1_tech",
        "l3_velocity":     "L3_vel",
        "l9_context":      "L9_ctx",
    })

    n_sess = len({str(r["date"]) for _, r in result.iterrows()})
    print(f"\n  {len(result)} matching moments across {n_sess} sessions\n")
    print(disp.to_string(index=False))

    # Summary statistics
    if len(result) > 1:
        print(f"\n  ── Signal distribution ──────────────────────────────────")
        print(result["signal"].value_counts().to_string())
        print(f"\n  ── Session phase ────────────────────────────────────────")
        print(result["phase"].value_counts().to_string())
        if "pcr" in result.columns and result["pcr"].notna().any():
            pcr_vals = result["pcr"].dropna()
            print(f"\n  ── PCR at matched moments ───────────────────────────────")
            print(f"     mean={pcr_vals.mean():.3f}  "
                  f"min={pcr_vals.min():.3f}  "
                  f"max={pcr_vals.max():.3f}  "
                  f"std={pcr_vals.std():.3f}")
        sc = result["composite_score"]
        print(f"\n  ── Score distribution ───────────────────────────────────")
        print(f"     mean={sc.mean():.3f}  "
              f"min={sc.min():.3f}  "
              f"max={sc.max():.3f}  "
              f"std={sc.std():.3f}")


def _fmt_time(t: str) -> str:
    """
    Normalise a user-supplied time string to HH:MM:SS.
    Accepts '9:15', '09:15', '09:15:00' — all return '09:15:00'.
    """
    parts = t.strip().split(":")
    h = parts[0].zfill(2)
    m = parts[1].zfill(2) if len(parts) > 1 else "00"
    s = parts[2].zfill(2) if len(parts) > 2 else "00"
    return f"{h}:{m}:{s}"


def _time_filter(date: datetime.date, from_t: str | None, to_t: str | None) -> str:
    """
    Build a SQL WHERE clause fragment that clips tick rows to an IST time range
    AND anchors to the given calendar date.

    from_t / to_t accept '9:15', '09:15', or '09:15:00'.
    Always includes a base-date guard so overnight ticks from the previous
    session that ended up in this DB file are excluded from the result.
    """
    # Base-date guard: only return ticks whose exchange timestamp falls on `date`
    parts = [
        f"(ts AT TIME ZONE 'Asia/Kolkata')::DATE = DATE '{date}'"
    ]
    if from_t:
        parts.append(
            f"(ts AT TIME ZONE 'Asia/Kolkata') >= "
            f"TIMESTAMP '{date} {_fmt_time(from_t)}'"
        )
    if to_t:
        parts.append(
            f"(ts AT TIME ZONE 'Asia/Kolkata') < "
            f"TIMESTAMP '{date} {_fmt_time(to_t)}'"
        )
    return "AND " + " AND ".join(parts)


def cmd_ticks(
    date:   datetime.date,
    sym:    str | None,
    from_t: str | None = None,
    to_t:   str | None = None,
) -> None:
    """
    Raw tick timeline — one row per WebSocket SymbolUpdate packet.

    Columns:
      time     — IST wall-clock timestamp of the exchange feed tick
      ltp      — last traded price (index level)
      delta    — change from previous tick (points)
      chp      — cumulative % change from previous session close
      day_high — expanding session high at this tick
      day_low  — expanding session low at this tick
      range    — day_high - day_low (expanding intraday range)
    """
    sym_full = _sym(sym) if sym else "NSE:NIFTY50-INDEX"
    tf       = _time_filter(date, from_t, to_t)
    time_tag = f"  [{from_t}–{to_t}]" if (from_t or to_t) else ""

    df = _run(date, f"""
        WITH ranked AS (
            SELECT
                ts,
                ltp,
                ltp - LAG(ltp) OVER (ORDER BY ts) AS delta,
                chp,
                day_high,
                day_low,
                day_high - day_low AS range_pts
            FROM ticks
            WHERE symbol = '{sym_full}'
              {tf}
        )
        SELECT
            ts,
            ROUND(ltp,       2) AS ltp,
            ROUND(delta,     2) AS delta,
            ROUND(chp,       2) AS chp_pct,
            ROUND(day_high,  2) AS day_high,
            ROUND(day_low,   2) AS day_low,
            ROUND(range_pts, 2) AS range_pts
        FROM ranked
        ORDER BY ts
    """)

    _hdr(f"TICK STREAM  {date}{time_tag}  —  {_label(sym_full)}")
    if df.empty:
        print("  No tick data.  Ticks are recorded during live trading sessions only.")
        return

    # Convert ts to readable IST time with seconds
    df_disp = df.copy()
    ts_col  = pd.to_datetime(df_disp["ts"], utc=True)
    df_disp.insert(0, "time", ts_col.dt.tz_convert("Asia/Kolkata").dt.strftime("%H:%M:%S"))
    df_disp = df_disp.drop(columns=["ts"])
    # First tick has no prior tick — show "—" instead of NaN for delta
    df_disp["delta"] = df_disp["delta"].fillna(0.0)

    print(df_disp.to_string(index=False))
    print(f"\n  {len(df):,} ticks   "
          f"ltp range: {df['ltp'].min():,.2f} – {df['ltp'].max():,.2f}   "
          f"span: {df['ltp'].max() - df['ltp'].min():.2f} pts")


def cmd_tick_stats(
    date:   datetime.date,
    sym:    str | None,
    from_t: str | None = None,
    to_t:   str | None = None,
) -> None:
    """
    Tick velocity and volatility analytics — the quant view of session quality.

    Sections:
      Overview      — tick count, interval, price range, ATR(1), net move
      Minute bars   — OHLCV reconstructed from raw ticks (verify candle accuracy)
      Top moves     — the 10 fastest 1-minute price swings of the session
      Tick density  — ticks-per-minute heatmap (reveals liquidity pockets)
    """
    sym_full = _sym(sym) if sym else "NSE:NIFTY50-INDEX"
    tf       = _time_filter(date, from_t, to_t)
    time_tag = f"  [{from_t}–{to_t}]" if (from_t or to_t) else ""
    label    = _label(sym_full)

    _hdr(f"TICK VELOCITY + VOLATILITY  {date}{time_tag}  —  {label}")

    # ── 1. Overview stats ─────────────────────────────────────────────────────
    ov = _run(date, f"""
        WITH t AS (
            SELECT ts, ltp,
                   ltp - LAG(ltp) OVER (ORDER BY ts) AS delta,
                   EXTRACT(EPOCH FROM
                       ts - LAG(ts) OVER (ORDER BY ts)) AS gap_sec
            FROM ticks
            WHERE symbol = '{sym_full}' {tf}
        )
        SELECT
            COUNT(*)                             AS total_ticks,
            ROUND(AVG(gap_sec),      2)          AS avg_interval_sec,
            ROUND(STDDEV(gap_sec),   2)          AS std_interval_sec,
            ROUND(MIN(ltp),          2)          AS session_low,
            ROUND(MAX(ltp),          2)          AS session_high,
            ROUND(MAX(ltp) - MIN(ltp), 2)        AS price_range_pts,
            ROUND(AVG(ABS(delta)),   2)          AS atr1_pts,
            ROUND(MAX(ABS(delta)),   2)          AS max_tick_move,
            ROUND(LAST(ltp  ORDER BY ts) -
                  FIRST(ltp ORDER BY ts), 2)     AS net_move_pts,
            ROUND((LAST(ltp  ORDER BY ts) -
                   FIRST(ltp ORDER BY ts)) /
                   FIRST(ltp ORDER BY ts) * 100, 3) AS net_move_pct
        FROM t
    """)

    if ov.empty or ov["total_ticks"].iloc[0] == 0:
        print("  No tick data for this session / time range.")
        print("  Ticks are recorded during live WebSocket sessions only.")
        return

    r = ov.iloc[0]
    # All aggregate fields can be NULL when there is only 1 tick (LAG → NULL gap).
    # Coerce to float with a 0 fallback so format strings never receive None/NaN.
    def _f(key: str, default: float = 0.0) -> float:
        v = r.get(key)
        import math
        return default if (v is None or (isinstance(v, float) and math.isnan(v))) else float(v)

    print(f"\n  ── Overview ────────────────────────────────────────────────────────")
    print(f"     Ticks captured    : {int(r['total_ticks']):,}")
    avg_int = _f('avg_interval_sec')
    std_int = _f('std_interval_sec')
    intvl   = f"{avg_int:.2f}s  (±{std_int:.2f}s)" if avg_int > 0 else "< 2 ticks — N/A"
    print(f"     Avg interval      : {intvl}")
    print(f"     Session range     : {_f('session_low'):,.2f}  –  "
          f"{_f('session_high'):,.2f}  ({_f('price_range_pts'):.2f} pts)")
    atr1 = _f('atr1_pts')
    print(f"     ATR(1) tick       : {atr1:.2f} pts  "
          f"{'(avg absolute tick-to-tick move)' if atr1 > 0 else '(N/A — need ≥2 ticks)'}")
    print(f"     Max single tick   : {_f('max_tick_move'):.2f} pts")
    net = _f('net_move_pts')
    pct = _f('net_move_pct')
    net_arrow = "▲" if net >= 0 else "▼"
    print(f"     Net move          : {net_arrow} {abs(net):.2f} pts  ({pct:+.3f}%)")

    # ── 2. Minute bars reconstructed from ticks ───────────────────────────────
    min_df = _run(date, f"""
        SELECT
            date_trunc('minute',
                ts AT TIME ZONE 'Asia/Kolkata')    AS minute_ist,
            COUNT(*)                               AS ticks,
            ROUND(arg_min(ltp, ts), 2)            AS open,
            ROUND(MAX(ltp),         2)            AS high,
            ROUND(MIN(ltp),         2)            AS low,
            ROUND(arg_max(ltp, ts), 2)            AS close,
            ROUND(MAX(ltp) - MIN(ltp), 2)         AS range_pts,
            ROUND(STDDEV(ltp),      4)            AS sigma
        FROM ticks
        WHERE symbol = '{sym_full}' {tf}
        GROUP BY 1
        ORDER BY 1
    """)

    if not min_df.empty:
        min_df["minute"] = pd.to_datetime(min_df["minute_ist"]).dt.strftime("%H:%M")
        min_df = min_df.drop(columns=["minute_ist"])
        cols = ["minute","ticks","open","high","low","close","range_pts","sigma"]
        print(f"\n  ── 1-Minute bars from ticks ────────────────────────────────────")
        print(min_df[cols].to_string(index=False))

        # ── 3. Top 10 fastest 1-min swings ───────────────────────────────────
        top10 = min_df.nlargest(10, "range_pts")[["minute","open","close","range_pts","ticks"]]
        print(f"\n  ── Top 10 fastest 1-min moves ──────────────────────────────────")
        print(top10.to_string(index=False))

        # ── 4. Tick density summary ───────────────────────────────────────────
        avg_ticks = min_df["ticks"].mean()
        low_dens  = min_df[min_df["ticks"] < avg_ticks * 0.4]
        high_dens = min_df[min_df["ticks"] > avg_ticks * 2.0]
        print(f"\n  ── Tick density ────────────────────────────────────────────────")
        print(f"     Avg ticks/min  : {avg_ticks:.1f}")
        if not high_dens.empty:
            print(f"     High activity  : {', '.join(high_dens['minute'].tolist())}")
        if not low_dens.empty:
            print(f"     Low activity   : {', '.join(low_dens['minute'].tolist())}")

        # ── 5. Rolling 5-min realised volatility ─────────────────────────────
        if len(min_df) >= 5:
            import math
            # Per-minute return (close-to-close)
            min_df["ret"] = min_df["close"].pct_change()
            vol_5m = (min_df["ret"].rolling(5).std()
                      * (252 * 375) ** 0.5 * 100)   # annualised %
            valid   = vol_5m.dropna()
            if not valid.empty:
                peak_idx = vol_5m.idxmax()
                # idxmax() returns NaN (not None) when all values are NaN
                peak_min = (min_df.loc[peak_idx, "minute"]
                            if peak_idx is not None and not (isinstance(peak_idx, float) and math.isnan(peak_idx))
                            else "—")
                vol_mean = valid.mean()
                vol_peak = valid.max()
                vol_last = vol_5m.iloc[-1]
                vol_last = vol_last if not math.isnan(vol_last) else 0.0
                print(f"\n  ── 5-min Realised Volatility (annualised) ──────────────────")
                print(f"     Mean           : {vol_mean:.1f}%")
                print(f"     Peak           : {vol_peak:.1f}%  @ {peak_min}")
                print(f"     Current        : {vol_last:.1f}%")

    print()


def cmd_export(date: datetime.date, fmt: str, out_dir: Path) -> None:
    """Export all tables to CSV or Parquet files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    con = _open(date)
    if con is None:
        print(f"  No session data for {date}")
        return

    for tbl in ("ticks","candles","oi_snapshots","futures_quotes","signals","trade_setups"):
        try:
            df   = con.execute(f"SELECT * FROM {tbl} ORDER BY ts").df()
            stem = f"{date}_{tbl}"
            if fmt == "parquet":
                path = out_dir / f"{stem}.parquet"
                df.to_parquet(path, index=False)
            else:
                path = out_dir / f"{stem}.csv"
                df.to_csv(path, index=False)
            print(f"  {tbl:22s}  {len(df):>6d} rows  →  {path}")
        except Exception as exc:
            print(f"  {tbl}: {exc}")
    con.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="session_replay",
        description="Intraday session time-machine and pattern finder",
    )
    p.add_argument("--date", default=None,
                   help="Target date YYYY-MM-DD (default: today)")
    p.add_argument("--sym",  default=None,
                   help="Symbol short name: NIFTY | BANKNIFTY | FINNIFTY | MIDCAP")
    p.add_argument("--tf",   default=None,
                   help="Timeframe: 5min | 15min | 60min | daily (default depends on command)")

    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--list",       action="store_true", help="List all saved sessions")
    mode.add_argument("--stats",      action="store_true", help="Row counts for the target date")
    mode.add_argument("--oi",         action="store_true", help="OI snapshot timeline")
    mode.add_argument("--signals",    action="store_true", help="Signal history")
    mode.add_argument("--setups",     action="store_true", help="Trade setup history")
    mode.add_argument("--candles",    action="store_true", help="Candle history")
    mode.add_argument("--report",     action="store_true", help="Full session replay (default)")
    mode.add_argument("--similar",    action="store_true", help="Find similar historical setups")
    mode.add_argument("--ticks",      action="store_true",
                      help="Raw tick stream (one row per WebSocket message)")
    mode.add_argument("--tick-stats", action="store_true", dest="tick_stats",
                      help="Tick velocity + volatility analytics")
    mode.add_argument("--export",     choices=["csv","parquet"],
                      help="Export all tables to files")

    # Time-range filter — applies to --ticks and --tick-stats
    p.add_argument("--from", default=None, dest="from_t", metavar="HH:MM",
                   help="Start time (IST) for tick slice, e.g. 10:00")
    p.add_argument("--to",   default=None, dest="to_t",   metavar="HH:MM",
                   help="End time (IST) for tick slice, e.g. 12:30")

    # Pattern search options
    p.add_argument("--score",     type=float, default=None,
                   help="Composite score centre for --similar (±0.5 band)")
    p.add_argument("--score-min", type=float, default=-4.0, dest="score_min")
    p.add_argument("--score-max", type=float, default=+4.0, dest="score_max")
    p.add_argument("--pcr",       type=float, default=None,
                   help="PCR centre for --similar (±0.15 band)")
    p.add_argument("--pcr-min",   type=float, default=0.0,  dest="pcr_min")
    p.add_argument("--pcr-max",   type=float, default=9.9,  dest="pcr_max")
    p.add_argument("--n-days",    type=int,   default=30,   dest="n_days",
                   help="Number of past sessions to search (default: 30)")

    p.add_argument("--out-dir",   type=Path,  default=Path("exports"), dest="out_dir",
                   help="Output directory for --export (default: exports/)")
    return p


def main() -> None:
    args = _parser().parse_args()

    # Resolve date
    if args.date:
        try:
            date = datetime.date.fromisoformat(args.date)
        except ValueError:
            print(f"ERROR: invalid date '{args.date}' — use YYYY-MM-DD format")
            sys.exit(1)
    else:
        date = datetime.datetime.now(tz=IST).date()

    if args.list:
        cmd_list()
        return

    if args.export:
        cmd_export(date, args.export, args.out_dir)
        return

    if args.stats:
        cmd_stats(date)
        return

    if args.similar:
        score_min = (args.score - 0.5) if args.score is not None else args.score_min
        score_max = (args.score + 0.5) if args.score is not None else args.score_max
        pcr_min   = (args.pcr   - 0.15) if args.pcr  is not None else args.pcr_min
        pcr_max   = (args.pcr   + 0.15) if args.pcr  is not None else args.pcr_max
        cmd_similar(
            args.sym,
            args.score, score_min, score_max,
            args.pcr,   pcr_min,   pcr_max,
            args.tf or "15min",
            args.n_days,
        )
        return

    if args.ticks:
        cmd_ticks(date, args.sym, args.from_t, args.to_t)
    elif args.tick_stats:
        cmd_tick_stats(date, args.sym, args.from_t, args.to_t)
    elif args.oi:
        cmd_oi(date, args.sym)
    elif args.signals:
        cmd_signals(date, args.sym)
    elif args.setups:
        cmd_setups(date, args.sym, args.tf)
    elif args.candles:
        cmd_candles(date, args.sym, args.tf or "15min")
    else:
        # Default: full session report
        cmd_report(date, args.sym)


if __name__ == "__main__":
    main()
