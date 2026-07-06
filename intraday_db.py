"""
intraday_db.py  —  Intraday session persistence engine.

Captures 5 live data streams to DuckDB while the trading session runs:

  candles        OHLCV bar-close events (all 4 resolutions: 1m/5m/15m/60m)
  oi_snapshots   Option-chain state every ~3 min (PCR / IV / walls / OI)
  futures_quotes Near / next / far futures strip every ~30 s
  signals        Multi-timeframe technical consensus every ~60 s
  trade_setups   9-layer composite scores + full layer breakdown every ~30 s

Architecture
  DuckDB on Windows takes an exclusive file lock.  To allow both writes AND
  reads during a live session without a second process contending for the lock,
  ALL database access (reads included) is routed through a single background
  writer thread that owns the connection.

  Callers push records or query requests onto a bounded Queue(maxsize=5000).
  Queue overflow silently drops data records — the trading loop is NEVER blocked.
  Query requests block the caller for up to 8 s (timeout returns empty DataFrame).

  Data records are flushed in batches (up to 100) or every 10 s, first wins.
  Query records flush any pending batch immediately, then execute in-connection.

  One DuckDB file per calendar day:  data/intraday/YYYY-MM-DD.duckdb
  Past-session queries open the file read-only (no active writer → no conflict).

  Graceful degradation: DuckDB import failure logs once and keeps trading.
"""

from __future__ import annotations

import datetime
import os
import queue
import threading
import time
from pathlib import Path
from zoneinfo import ZoneInfo

from core.constants import IST   # single source of truth (fixed +5:30, no DST)
_DB_DIR  = Path(__file__).parent / "data" / "intraday"
_LIVE_DIR = _DB_DIR / "live"   # Parquet snapshots for concurrent reads during live session

# Tables exported to Parquet after every checkpoint so session_replay.py can
# query live data without needing a second DuckDB connection (which Windows
# exclusive-locks prevent).
_PARQUET_TABLES = (
    "ticks", "candles", "oi_snapshots",
    "futures_quotes", "signals", "trade_setups",
    "chain_snapshots", "scout_alerts",
)

# ── Schema ────────────────────────────────────────────────────────────────────

# Per-day DuckDB schema is the single source of truth in the storage layer.
from tradebot.storage.schema import INTRADAY_DDL as _DDL  # noqa: F401  (kept for compat)
from tradebot.storage.schema import init_intraday as _init_intraday

_SENTINEL = object()   # poison pill sent by shutdown()


class _QueryReq:
    """Carries an SQL query from caller → writer thread → result back."""
    __slots__ = ("sql", "result")

    def __init__(self, sql: str) -> None:
        self.sql    = sql
        self.result: queue.Queue = queue.Queue(maxsize=1)


# ── Writer engine ─────────────────────────────────────────────────────────────

class IntradayDB:
    """
    Non-blocking background persistence engine.
    Use the module-level singleton `idb` — don't instantiate directly.
    """

    BATCH_SIZE  = 100    # flush when this many data records are waiting
    FLUSH_EVERY = 10.0   # also flush every N seconds regardless of batch size
    QUERY_TIMEOUT = 8.0  # max seconds a caller waits for a live query result

    def __init__(self) -> None:
        self._q: queue.Queue      = queue.Queue(maxsize=5000)  # 5000: ticks add ~4/sec at peak
        self._conn                = None          # owned by writer thread only
        self._date: datetime.date | None = None   # date the current conn covers
        self._duckdb              = None
        self._ok: bool            = False
        self._errors: int         = 0
        # Per-record-kind insert failures. NEVER silent: the chain_snapshots capture
        # died for weeks because per-record errors were swallowed — count every one,
        # log the first per kind (+ every 500th), expose via session_stats().
        self._insert_errors: dict[str, int] = {}
        # Per-table PARQUET-EXPORT failures. Same rule, same reason: a persistent export
        # failure FREEZES the lock-free mirror (readers see stale data) — must not be
        # silent. Counted here, folded into the write-health badge via export_error_count().
        self._export_errors: dict[str, int] = {}

        # Per-symbol throttle — futures API fires every 2 s; we persist every 30 s
        self._fut_last: dict[str, float] = {}

        # Per-symbol cumulative volume baseline for tick_vol derivation.
        # Maintained in writer thread only (single-threaded) — no lock needed.
        # Reset each new session day via _get_conn().
        self._tick_prev_vol: dict[str, int] = {}
        self._FUT_THROTTLE = 30.0

        self._ready = threading.Event()   # set once writer has imported duckdb
        self._thread = threading.Thread(
            target=self._writer_loop, name="IntradayDB-writer", daemon=True,
        )
        self._thread.start()

    # ── Public write API ──────────────────────────────────────────────────────

    def write_tick(
        self,
        sym:      str,
        ts,                    # exchange feed timestamp (datetime, tz-aware)
        ltp:      float,
        cum_vol:  float = 0.0,
        day_open: float = 0.0,
        day_high: float = 0.0,
        day_low:  float = 0.0,
        ch:       float = 0.0,
        chp:      float = 0.0,
    ) -> None:
        """
        Persist one raw WebSocket tick.  Non-blocking — drops silently if queue full.

        tick_vol (per-tick volume increment) is derived inside the writer thread
        using _tick_prev_vol so the public API stays simple.
        """
        self._push(("tick", sym, ts, ltp, int(cum_vol),
                    day_open, day_high, day_low, ch, chp))

    def write_candle(self, sym: str, res: str, bar: dict) -> None:
        self._push(("candle", sym, res, bar))

    def write_oi_snapshot(self, snap) -> None:
        self._push(("oi", snap))

    def write_futures(self, sym: str, futures: list, spot: float) -> None:
        now = time.monotonic()
        if now - self._fut_last.get(sym, 0.0) < self._FUT_THROTTLE:
            return
        self._fut_last[sym] = now
        self._push(("futures", sym, futures, spot))

    def write_signal(self, result: dict) -> None:
        self._push(("signal", result))

    def write_chain(self, sym: str, ts, rows: list) -> None:
        """
        Persist per-strike chain legs for one index at one moment.
        rows: [(strike, side, ltp, ltpch, oi, oich, volume), ...]
        Non-blocking — drops silently if queue full.
        """
        if rows:
            self._push(("chain", sym, ts, rows))

    def write_scout_alert(self, sym: str, ts, alert: dict) -> None:
        """Persist ONE fired scout lifecycle alert (NEW/SL/TARGET/BAND) to the
        canonical per-day store + parquet mirror.

        This is the authoritative, archived, multi-device evening-review log — the
        dashboard's localStorage alert list is only a per-browser convenience copy.
        Non-blocking; drops silently if the queue is full.
        Decision-support only (the CE/PE arrow is measured negative-EV)."""
        if alert:
            self._push(("scout_alert", sym, ts, alert))

    def write_setup(
        self,
        sym: str, tf: str, signal: str,
        composite: float, confidence: int, direction: str,
        l1: float, l2: float, l3: float, l4: float, l5: float,
        l6: float, l7: float, l8: float, l9: float,
        agreement: int, phase: str, spot: float, atm_iv: float,
    ) -> None:
        self._push((
            "setup", sym, tf, signal, composite, confidence, direction,
            l1, l2, l3, l4, l5, l6, l7, l8, l9,
            agreement, phase, spot, atm_iv,
        ))

    # ── Public read API ───────────────────────────────────────────────────────

    def query(self, sql: str, date: datetime.date | None = None):
        """
        Execute SQL and return a pandas DataFrame.

        Today's file: routed through the writer thread (avoids Windows lock
        contention — same connection reads and writes).
        Past sessions: opened read-only directly (writer is on a different file).
        Returns an empty DataFrame on error or missing data.
        """
        import pandas as pd
        target = date or datetime.datetime.now(tz=IST).date()
        today  = datetime.datetime.now(tz=IST).date()

        if target == today and self._ok and self._thread.is_alive():
            req = _QueryReq(sql)
            self._push(("query", req))
            try:
                return req.result.get(timeout=self.QUERY_TIMEOUT)
            except queue.Empty:
                return pd.DataFrame()

        # Past session — open read-only (writer holds today's file, not this one)
        path = _DB_DIR / f"{target}.duckdb"
        if not path.exists():
            return pd.DataFrame()
        try:
            import duckdb
            con = duckdb.connect(str(path), read_only=True)
            df  = con.execute(sql).df()
            con.close()
            return df
        except Exception as exc:
            if "being used by another process" in str(exc):
                print(
                    f"[IntradayDB] {target}.duckdb is locked by another process.\n"
                    "  Run session_replay.py after the trading session ends."
                )
            return pd.DataFrame()

    def list_sessions(self) -> list[datetime.date]:
        out = []
        for p in sorted(_DB_DIR.glob("*.duckdb")):
            try:
                out.append(datetime.date.fromisoformat(p.stem))
            except ValueError:
                pass
        return out

    def session_stats(self, date: datetime.date | None = None) -> dict[str, int]:
        target = date or datetime.datetime.now(tz=IST).date()
        counts: dict[str, int] = {}
        for tbl in ("ticks", "candles", "oi_snapshots", "futures_quotes", "signals", "trade_setups", "chain_snapshots", "scout_alerts"):
            df = self.query(f"SELECT COUNT(*) AS n FROM {tbl}", target)
            counts[tbl] = int(df.iloc[0]["n"]) if not df.empty else 0
        for kind, n in self._insert_errors.items():
            counts[f"insert_errors_{kind}"] = n
        for tbl, n in self._export_errors.items():
            counts[f"export_errors_{tbl}"] = n
        return counts

    def insert_error_count(self) -> int:
        """Total per-record insert failures this process (0 = healthy). LIGHT — reads the
        in-memory counter, no DB query, safe to poll. Surfaced as the dashboard write-health
        badge so a SYSTEMATIC write failure (schema drift — the chain_snapshots class of bug)
        is visible on the header, not only buried in supervisor.log."""
        return sum(self._insert_errors.values())

    def export_error_count(self) -> int:
        """Total parquet-export failures this process (0 = healthy). LIGHT in-memory counter.
        Nonzero = the lock-free mirror is not advancing → readers see FROZEN data. Folded into
        the write-health badge alongside insert errors so a stalled export can't stay silent."""
        return sum(self._export_errors.values())

    def shutdown(self) -> None:
        """Flush remaining records, checkpoint, close.  Call at process exit."""
        self._q.put(_SENTINEL)
        self._thread.join(timeout=15)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _push(self, record: tuple) -> None:
        try:
            self._q.put_nowait(record)
        except queue.Full:
            pass   # drop data — never block the caller

    # ── Writer thread ─────────────────────────────────────────────────────────

    def _writer_loop(self) -> None:
        try:
            import duckdb as _ddb
            self._duckdb = _ddb
            self._ok = True
        except ImportError:
            print("[IntradayDB] duckdb not available — session data will NOT be persisted")
            self._ready.set()
            return
        self._ready.set()

        batch: list[tuple] = []
        last_flush = time.monotonic()

        while True:
            remaining = self.FLUSH_EVERY - (time.monotonic() - last_flush)
            try:
                record = self._q.get(timeout=max(0.1, remaining))

                if record is _SENTINEL:
                    if batch:
                        self._flush_data(batch)
                    self._close_conn()
                    return

                if record[0] == "query":
                    # Flush any pending writes first, then answer the query
                    if batch:
                        self._flush_data(batch)
                        batch = []
                        last_flush = time.monotonic()
                    self._exec_query(record[1])
                else:
                    batch.append(record)
                    # Drain the queue without sleeping (grab whatever is ready)
                    while len(batch) < self.BATCH_SIZE:
                        try:
                            r = self._q.get_nowait()
                            if r is _SENTINEL:
                                if batch:
                                    self._flush_data(batch)
                                self._close_conn()
                                return
                            if r[0] == "query":
                                self._flush_data(batch)
                                batch = []
                                last_flush = time.monotonic()
                                self._exec_query(r[1])
                            else:
                                batch.append(r)
                        except queue.Empty:
                            break

            except queue.Empty:
                pass   # timeout — fall through to flush check

            now = time.monotonic()
            if batch and (len(batch) >= self.BATCH_SIZE or now - last_flush >= self.FLUSH_EVERY):
                self._flush_data(batch)
                batch = []
                last_flush = now

    # ── Connection management ─────────────────────────────────────────────────

    def _get_conn(self):
        today = datetime.datetime.now(tz=IST).date()
        if self._conn is None or self._date != today:
            self._close_conn()
            _DB_DIR.mkdir(parents=True, exist_ok=True)
            path = _DB_DIR / f"{today}.duckdb"
            self._conn = self._duckdb.connect(str(path))
            _init_intraday(self._conn)   # create tables + apply idempotent migrations
            self._date = today
            self._tick_prev_vol.clear()   # new session — reset volume baselines
        return self._conn

    def _close_conn(self) -> None:
        if self._conn:
            try:
                self._conn.execute("CHECKPOINT")
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    # ── Flush helpers ─────────────────────────────────────────────────────────

    def _flush_data(self, batch: list[tuple]) -> None:
        try:
            con = self._get_conn()
            for rec in batch:
                try:
                    self._insert(con, rec)
                except Exception as exc:
                    # Count + log (first per kind, then every 500th) — a schema/column
                    # mismatch must surface the same day, not weeks later.
                    kind = rec[0] if rec and isinstance(rec[0], str) else "?"
                    n = self._insert_errors.get(kind, 0) + 1
                    self._insert_errors[kind] = n
                    if n == 1 or n % 500 == 0:
                        print(f"[IntradayDB] {kind} insert FAILED (x{n}): {exc}",
                              flush=True)
            con.execute("CHECKPOINT")
            self._export_parquet(con)   # snapshot for concurrent reads
        except Exception as exc:
            if self._errors < 3:
                print(f"[IntradayDB] write error: {exc}")
                self._errors += 1

    def _export_parquet(self, con) -> None:
        """
        After each checkpoint, export all tables to Parquet files in data/intraday/live/.

        These files are plain files — no DuckDB lock — so session_replay.py can read
        them while the dashboard's write connection holds the exclusive .duckdb lock.

        Written ATOMICALLY: COPY to a per-table .tmp, then os.replace onto the live path
        (same dir → same volume → atomic rename). A concurrent reader therefore always
        sees a COMPLETE file, never the torn/truncated mid-COPY snapshot an in-place write
        exposes. Per-table failures are COUNTED + throttle-logged (not swallowed): a silent
        export freeze — mirror stops advancing with nothing on the header — is the exact
        chain_snapshots silent-death class. A failed table keeps its PREVIOUS good file
        (replace didn't fire) and self-heals next flush.

        Staleness: up to FLUSH_EVERY (10 s).  Acceptable for market analysis.
        Cost: ~100–200 ms per flush (negligible vs 10-second interval).
        """
        if self._date is None:
            return
        try:
            _LIVE_DIR.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            self._note_export_error("mkdir", exc)
            return
        today = str(self._date)
        for tbl in _PARQUET_TABLES:
            pq  = _LIVE_DIR / f"{today}_{tbl}.parquet"
            tmp = _LIVE_DIR / f"{today}_{tbl}.parquet.tmp"
            try:
                # Use forward slashes — DuckDB COPY TO needs POSIX paths on Windows
                con.execute(
                    f"COPY {tbl} TO '{tmp.as_posix()}' (FORMAT PARQUET, CODEC SNAPPY)"
                )
                os.replace(tmp, pq)          # atomic publish (never a torn read)
            except Exception as exc:
                self._note_export_error(tbl, exc)
                try:
                    tmp.unlink()             # don't leak a half-written temp
                except OSError:
                    pass

    def _note_export_error(self, tbl: str, exc: Exception) -> None:
        """Count + throttle-log a parquet-export failure (first per table, then every
        500th). Surfaced on the write-health badge so a frozen mirror can't stay silent."""
        n = self._export_errors.get(tbl, 0) + 1
        self._export_errors[tbl] = n
        if n == 1 or n % 500 == 0:
            print(f"[IntradayDB] {tbl} parquet export FAILED (x{n}): {exc}", flush=True)

    def _exec_query(self, req: _QueryReq) -> None:
        import pandas as pd
        try:
            con = self._get_conn()
            df  = con.execute(req.sql).df()
        except Exception:
            df = pd.DataFrame()
        try:
            req.result.put_nowait(df)
        except queue.Full:
            pass

    # ── Row insertion ─────────────────────────────────────────────────────────

    def _insert(self, con, rec: tuple) -> None:
        now   = datetime.datetime.now(tz=IST)
        today = now.date()
        kind  = rec[0]

        if kind == "tick":
            # ("tick", sym, ts, ltp, cum_vol, day_open, day_high, day_low, ch, chp)
            _, sym, ts, ltp, cum_vol, day_open, day_high, day_low, ch, chp = rec
            if hasattr(ts, "tzinfo") and ts.tzinfo is None:
                ts = ts.replace(tzinfo=IST)
            # Derive per-tick volume increment (safe against WebSocket resets)
            prev_vol = self._tick_prev_vol.get(sym, cum_vol)
            tick_vol = max(0, cum_vol - prev_vol)
            self._tick_prev_vol[sym] = cum_vol
            con.execute(
                """INSERT INTO ticks
                       (ts, symbol, ltp, tick_vol, cum_vol,
                        day_open, day_high, day_low, ch, chp)
                   VALUES (?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT DO NOTHING""",
                [ts, sym, ltp, tick_vol, cum_vol,
                 day_open if day_open else None,
                 day_high if day_high else None,
                 day_low  if day_low  else None,
                 ch, chp],
            )

        elif kind == "candle":
            _, sym, res, bar = rec
            ts = bar.get("ts")
            if not ts:
                return
            if hasattr(ts, "tzinfo") and ts.tzinfo is None:
                ts = ts.replace(tzinfo=IST)
            con.execute(
                """INSERT INTO candles
                       (ts, date, symbol, resolution, open, high, low, close, volume)
                   VALUES (?,?,?,?,?,?,?,?,?)
                   ON CONFLICT DO NOTHING""",
                [ts, today, sym, res,
                 bar.get("open"), bar.get("high"), bar.get("low"),
                 bar.get("close"), int(bar.get("volume") or 0)],
            )

        elif kind == "oi":
            snap = rec[1]
            ts   = snap.ts
            if hasattr(ts, "tzinfo") and ts.tzinfo is None:
                ts = ts.replace(tzinfo=IST)
            con.execute(
                """INSERT INTO oi_snapshots
                       (ts, date, symbol, spot, atm, pcr,
                        total_call_oi, total_put_oi, atm_call_oi, atm_put_oi,
                        atm_call_iv, atm_put_iv, atm_iv,
                        atm_call_prem, atm_put_prem,
                        call_wall, put_wall, max_pain,
                        near_call_oi, near_put_oi, put_skew)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT DO NOTHING""",
                [ts, today, snap.sym,
                 round(float(snap.spot), 2),
                 int(snap.atm),
                 round(float(snap.pcr), 4),
                 int(snap.total_call_oi), int(snap.total_put_oi),
                 int(snap.atm_call_oi),   int(snap.atm_put_oi),
                 round(float(snap.atm_call_iv), 3),
                 round(float(snap.atm_put_iv),  3),
                 round(float(snap.atm_iv),      3),
                 round(float(snap.atm_call_prem), 2),
                 round(float(snap.atm_put_prem),  2),
                 int(snap.call_wall), int(snap.put_wall), int(snap.max_pain),
                 int(snap.near_call_oi), int(snap.near_put_oi),
                 round(float(snap.put_skew), 3)],
            )

        elif kind == "chain":
            _, sym, ts, rows = rec
            if hasattr(ts, "tzinfo") and ts.tzinfo is None:
                ts = ts.replace(tzinfo=IST)
            con.executemany(
                # Name the 12 columns explicitly — the table has a 13th (`expiry`,
                # NOT NULL DEFAULT 0, added by migration + part of the PK). A bare
                # VALUES(...12...) against the 13-col table errored on EVERY insert and
                # was swallowed by the writer's per-record except → chain_snapshots
                # stayed empty all session. expiry takes its DEFAULT 0 (legacy single
                # nearest-expiry marker, which is exactly what the capture stores).
                """INSERT INTO chain_snapshots
                       (ts, date, symbol, strike, side, ltp, ltpch, oi, oich,
                        volume, delta, iv)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT DO NOTHING""",
                [[ts, today, sym, int(sp), side,
                  round(float(ltp or 0), 2), round(float(ltpch or 0), 2),
                  int(oi or 0), int(oich or 0), int(vol or 0),
                  (round(float(delta), 4) if delta is not None else None),
                  (round(float(iv), 3) if iv is not None else None)]
                 for sp, side, ltp, ltpch, oi, oich, vol, delta, iv in rows],
            )

        elif kind == "futures":
            _, sym, futures, spot = rec
            if len(futures) < 2:
                return
            near     = futures[0]
            nxt      = futures[1]
            far      = futures[2] if len(futures) > 2 else {}
            near_ltp = float(near.get("ltp") or 0)
            next_ltp = float(nxt.get("ltp")  or 0)
            far_ltp  = float(far.get("ltp")  or 0)
            spot_f   = float(spot or 0)
            near_bas = round(near_ltp - spot_f, 2) if (near_ltp and spot_f) else 0.0
            next_bas = round(next_ltp - spot_f, 2) if (next_ltp and spot_f) else 0.0
            roll     = round(next_ltp - near_ltp, 2) if (near_ltp and next_ltp) else 0.0
            term     = "CONTANGO" if roll > 0 else "BACKWARDATION"
            con.execute(
                """INSERT INTO futures_quotes
                       (ts, date, symbol, near_ltp, next_ltp, far_ltp,
                        near_basis, next_basis, roll_spread, term_structure,
                        near_vol, next_vol)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT DO NOTHING""",
                [now, today, sym, near_ltp, next_ltp, far_ltp,
                 near_bas, next_bas, roll, term,
                 int(near.get("vol") or 0), int(nxt.get("vol") or 0)],
            )

        elif kind == "signal":
            result = rec[1]
            sym    = result.get("symbol", "")
            tfs    = result.get("timeframes", {})

            def _sc(k):  return round(float(tfs.get(k, {}).get("score")    or 0), 3)
            def _sg(k):  return str(tfs.get(k, {}).get("signal")           or "")
            def _rsi(k): return round(float(tfs.get(k, {}).get("rsi")      or 0), 2)

            overall = result.get("overall") or ("",)
            overall_label = overall[0] if isinstance(overall, (list, tuple)) else str(overall)

            con.execute(
                """INSERT INTO signals
                       (ts, date, symbol, weighted_score, overall,
                        score_5min, score_15min, score_60min, score_daily,
                        signal_5min, signal_15min, signal_60min, signal_daily,
                        rsi_5min, rsi_15min, macd_hist_15m, close_price,
                        vwap_15min, bull_tfs, bear_tfs)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT DO NOTHING""",
                [now, today, sym,
                 round(float(result.get("weighted_score") or 0), 3),
                 overall_label,
                 _sc("5min"), _sc("15min"), _sc("60min"), _sc("daily"),
                 _sg("5min"), _sg("15min"), _sg("60min"), _sg("daily"),
                 _rsi("5min"), _rsi("15min"),
                 round(float(tfs.get("15min", {}).get("macd_hist") or 0), 4),
                 round(float(tfs.get("15min", {}).get("close")     or 0), 2),
                 round(float(tfs.get("15min", {}).get("vwap")      or 0), 2),
                 int(result.get("bull_timeframes") or 0),
                 int(result.get("bear_timeframes") or 0)],
            )

        elif kind == "setup":
            (_, sym, tf, signal, composite, confidence, direction,
             l1, l2, l3, l4, l5, l6, l7, l8, l9,
             agreement, phase, spot, atm_iv) = rec
            con.execute(
                """INSERT INTO trade_setups
                       (ts, date, symbol, timeframe, signal, composite_score,
                        confidence, direction,
                        l1_tech, l2_oi, l3_velocity, l4_inst, l5_futures,
                        l6_iv, l7_pcr, l8_maxpain, l9_context,
                        agreement, phase, spot, atm_iv)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT DO NOTHING""",
                [now, today, sym, tf,
                 str(signal),
                 round(float(composite),  3),
                 int(confidence),
                 str(direction),
                 round(float(l1), 3), round(float(l2), 3), round(float(l3), 3),
                 round(float(l4), 3), round(float(l5), 3), round(float(l6), 3),
                 round(float(l7), 3), round(float(l8), 3), round(float(l9), 3),
                 int(agreement),
                 str(phase),
                 round(float(spot   or 0), 2),
                 round(float(atm_iv or 0), 3)],
            )

        elif kind == "scout_alert":
            _, sym, ts, a = rec
            if hasattr(ts, "tzinfo") and ts.tzinfo is None:
                ts = ts.replace(tzinfo=IST)

            def _f(x):
                try:
                    return round(float(x), 2) if x is not None else None
                except (TypeError, ValueError):
                    return None

            con.execute(
                """INSERT INTO scout_alerts
                       (ts, date, symbol, kind, label, side, strike,
                        entry, sl, tgt, cur, spot, band_dir, head, body, thin)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT DO NOTHING""",
                [ts, today, sym, str(a.get("kind") or ""),
                 a.get("label"), a.get("side"),
                 (int(a["strike"]) if a.get("strike") is not None else None),
                 _f(a.get("entry")), _f(a.get("sl")), _f(a.get("tgt")),
                 _f(a.get("cur")), _f(a.get("spot")),
                 a.get("band_dir"), a.get("head"), a.get("body"),
                 bool(a.get("thin"))],
            )


# ── Module singleton — import this everywhere ─────────────────────────────────

idb = IntradayDB()
