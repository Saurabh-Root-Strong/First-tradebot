"""
intraday_trades.py  --  Component C: intraday paper-trade ledger + track record.

The engine (build_recommendation) emits an actionable CE/PE setup with a full
ticket (strike, entry premium, SL, T1, T2). This module turns each fresh signal
into a tracked paper trade, follows the chosen option's live LTP to resolution
(T1 / SL / session-close), and records the honest outcome — hit-rate, R-multiple,
MFE/MAE — exactly the feedback loop the EOD memory engine has, but intraday and
at the option level.

Lifecycle:
    maybe_open(index_sym, rec)      -> opens ONE trade per (index, direction) at a
                                       time (no churn from the 30s re-fire).
    update_open_trades({sym: ltp})  -> updates MFE/MAE; resolves on SL/T1/T2.
    close_eod({sym: ltp})           -> marks remaining open trades to last price.
    recent() / stats()              -> track-record panel + honest accuracy.

Storage: SQLite (data/intraday_trades.db) — a small mutable ledger with frequent
UPDATEs, so SQLite (not the append-only DuckDB tick store) is the right tool.
Thread-safe: a single lock guards all writes (poller thread + Dash callbacks).
"""

from __future__ import annotations

import datetime
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
_DB = Path(__file__).parent / "data" / "intraday_trades.db"

# Don't re-enter the same (index, direction) within this window after a close.
_REENTRY_COOLDOWN_MIN = 10
# Minimum confidence to log a trade. 0 = track every directional (non-NEUTRAL)
# call and tag its conviction (LOW/MODERATE/HIGH) — the dedup + cooldown prevent
# spam, and this builds the full validation dataset (do LOW setups lose, HIGH win?).
_MIN_CONFIDENCE = 0

_DDL = """
CREATE TABLE IF NOT EXISTS paper_trades (
    trade_id     TEXT PRIMARY KEY,
    opened_ts    TEXT NOT NULL,
    date         TEXT NOT NULL,
    index_sym    TEXT NOT NULL,
    option_sym   TEXT,
    direction    TEXT,
    strike       REAL,
    expiry       TEXT,
    tf           TEXT,
    conviction   TEXT,
    confidence   INTEGER,
    composite    REAL,
    spot_entry   REAL,
    entry_ltp    REAL,
    sl           REAL,
    t1           REAL,
    t2           REAL,
    risk         REAL,
    mfe          REAL,
    mae          REAL,
    last_ltp     REAL,
    updated_ts   TEXT,
    status       TEXT,
    exit_ltp     REAL,
    exit_ts      TEXT,
    r_multiple   REAL,
    outcome      TEXT,
    reason       TEXT
);
"""


def _now() -> datetime.datetime:
    return datetime.datetime.now(tz=IST)


def _today() -> str:
    return _now().date().isoformat()


class TradeLedger:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        _DB.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as con:
            con.execute(_DDL)
            try:
                con.execute("ALTER TABLE paper_trades ADD COLUMN reason TEXT")
            except Exception:
                pass   # column already exists
            con.commit()

    def _conn(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(_DB), timeout=8.0)
        con.row_factory = sqlite3.Row
        return con

    # ── Open ──────────────────────────────────────────────────────────────────
    def maybe_open(self, index_sym: str, rec: dict,
                   oi_dyn=None, cont=None,
                   ts: Optional[datetime.datetime] = None) -> Optional[str]:
        """Open a tracked trade from an actionable rec. Returns trade_id or None.

        oi_dyn / cont are the live OI-Dynamics (A) and overnight-continuity (B)
        reads at signal time — woven into the trade's `reason` for the cockpit so
        the 'why' carries the real intraday edge, not just the 9-layer signal.
        """
        if not rec or rec.get("neutral") or rec.get("direction") not in ("CE", "PE"):
            return None
        if int(rec.get("confidence") or 0) < _MIN_CONFIDENCE:
            return None
        entry = float(rec.get("ltp") or 0)
        sl    = float(rec.get("sl") or 0)
        risk  = round(entry - sl, 2)
        if entry <= 0 or risk <= 0:
            return None

        ts  = ts or _now()
        day = ts.date().isoformat()
        direction = rec["direction"]

        with self._lock:
            con = self._conn()
            try:
                # Skip if a trade for this (index, direction) is already OPEN today.
                if con.execute(
                    "SELECT 1 FROM paper_trades WHERE index_sym=? AND direction=? "
                    "AND status='OPEN' AND date=? LIMIT 1",
                    [index_sym, direction, day],
                ).fetchone():
                    return None
                # Cooldown: skip if one closed very recently (anti-churn).
                last = con.execute(
                    "SELECT exit_ts FROM paper_trades WHERE index_sym=? AND direction=? "
                    "AND date=? AND exit_ts IS NOT NULL ORDER BY exit_ts DESC LIMIT 1",
                    [index_sym, direction, day],
                ).fetchone()
                if last and last["exit_ts"]:
                    try:
                        gap = (ts - datetime.datetime.fromisoformat(last["exit_ts"])).total_seconds() / 60
                        if gap < _REENTRY_COOLDOWN_MIN:
                            return None
                    except Exception:
                        pass

                # Capture the edge: live OI-Dynamics (A) + overnight continuity (B)
                # + the top 9-layer signal, so the cockpit 'why' shows the real read.
                bits: list = []
                try:
                    if oi_dyn is not None and getattr(oi_dyn, "has_data", False):
                        bits.append("OI▸ " + str(oi_dyn.headline))
                except Exception:
                    pass
                try:
                    if cont is not None and getattr(cont, "has_data", False):
                        bits.append("Overnight▸ " + str(cont.verdict))
                except Exception:
                    pass
                _sigs = rec.get("opt_signals") or []
                if _sigs:
                    bits.append(str(_sigs[0][1]))
                reason = " | ".join(bits)[:280] if bits else str(rec.get("warning") or "")[:280]

                tid = uuid.uuid4().hex[:12]
                con.execute(
                    "INSERT INTO paper_trades (trade_id, opened_ts, date, index_sym, "
                    "option_sym, direction, strike, expiry, tf, conviction, confidence, "
                    "composite, spot_entry, entry_ltp, sl, t1, t2, risk, mfe, mae, last_ltp, "
                    "updated_ts, status, outcome, reason) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    [tid, ts.isoformat(), day, index_sym,
                     rec.get("option_sym"), direction, rec.get("strike"), str(rec.get("exp_date") or ""),
                     rec.get("tf_label"), rec.get("conviction"), int(rec.get("confidence") or 0),
                     float(rec.get("score") or 0), rec.get("spot"),
                     entry, sl, float(rec.get("t1") or 0), float(rec.get("t2") or 0), risk,
                     entry, entry, entry, ts.isoformat(), "OPEN", "OPEN", reason],
                )
                con.commit()
                return tid
            finally:
                con.close()

    # ── Update / resolve ──────────────────────────────────────────────────────
    def update_open_trades(self, price_map: dict,
                           ts: Optional[datetime.datetime] = None) -> int:
        """Apply latest option LTPs to all open trades. Returns #rows touched."""
        if not price_map:
            return 0
        ts = ts or _now()
        touched = 0
        with self._lock:
            con = self._conn()
            try:
                opens = con.execute("SELECT * FROM paper_trades WHERE status='OPEN'").fetchall()
                for t in opens:
                    ltp = price_map.get(t["option_sym"])
                    if ltp is None:
                        continue
                    self._apply(con, t, float(ltp), ts)
                    touched += 1
                con.commit()
            finally:
                con.close()
        return touched

    def _apply(self, con, t, ltp: float, ts) -> None:
        entry = t["entry_ltp"]; sl = t["sl"]; t1 = t["t1"]; t2 = t["t2"]
        risk  = t["risk"] or (entry - sl) or 1.0
        mfe = max(t["mfe"] if t["mfe"] is not None else ltp, ltp)
        mae = min(t["mae"] if t["mae"] is not None else ltp, ltp)

        status = exit_ltp = outcome = r = exit_ts = None
        if ltp <= sl:
            status, exit_ltp, outcome, r, exit_ts = "SL", sl, "LOSS", -1.0, ts.isoformat()
        elif t2 and ltp >= t2:
            status, exit_ltp, outcome, r, exit_ts = "T2", t2, "WIN", round((t2 - entry) / risk, 2), ts.isoformat()
        elif t1 and ltp >= t1:
            status, exit_ltp, outcome, r, exit_ts = "T1", t1, "WIN", round((t1 - entry) / risk, 2), ts.isoformat()

        if status:
            con.execute(
                "UPDATE paper_trades SET mfe=?, mae=?, last_ltp=?, updated_ts=?, "
                "status=?, exit_ltp=?, exit_ts=?, r_multiple=?, outcome=? WHERE trade_id=?",
                [mfe, mae, ltp, ts.isoformat(), status, exit_ltp, exit_ts, r, outcome, t["trade_id"]],
            )
        else:
            con.execute(
                "UPDATE paper_trades SET mfe=?, mae=?, last_ltp=?, updated_ts=? WHERE trade_id=?",
                [mfe, mae, ltp, ts.isoformat(), t["trade_id"]],
            )

    def close_eod(self, price_map: Optional[dict] = None,
                  ts: Optional[datetime.datetime] = None) -> int:
        """Mark every still-open trade to its last/given price at session end."""
        ts = ts or _now()
        price_map = price_map or {}
        closed = 0
        with self._lock:
            con = self._conn()
            try:
                for t in con.execute("SELECT * FROM paper_trades WHERE status='OPEN'").fetchall():
                    ltp = float(price_map.get(t["option_sym"], t["last_ltp"] or t["entry_ltp"]))
                    entry = t["entry_ltp"]; risk = t["risk"] or 1.0
                    r = round((ltp - entry) / risk, 2)
                    outcome = "WIN" if ltp > entry else "LOSS" if ltp < entry else "SCRATCH"
                    con.execute(
                        "UPDATE paper_trades SET last_ltp=?, updated_ts=?, status='EOD', "
                        "exit_ltp=?, exit_ts=?, r_multiple=?, outcome=? WHERE trade_id=?",
                        [ltp, ts.isoformat(), ltp, ts.isoformat(), r, outcome, t["trade_id"]],
                    )
                    closed += 1
                con.commit()
            finally:
                con.close()
        return closed

    # ── Reads ─────────────────────────────────────────────────────────────────
    def open_trades(self) -> list[dict]:
        with self._lock:
            con = self._conn()
            try:
                return [dict(r) for r in con.execute(
                    "SELECT * FROM paper_trades WHERE status='OPEN' ORDER BY opened_ts").fetchall()]
            finally:
                con.close()

    def recent(self, limit: int = 25, date: Optional[str] = None) -> list[dict]:
        with self._lock:
            con = self._conn()
            try:
                if date:
                    rows = con.execute(
                        "SELECT * FROM paper_trades WHERE date=? ORDER BY opened_ts DESC LIMIT ?",
                        [date, limit]).fetchall()
                else:
                    rows = con.execute(
                        "SELECT * FROM paper_trades ORDER BY opened_ts DESC LIMIT ?",
                        [limit]).fetchall()
                return [dict(r) for r in rows]
            finally:
                con.close()

    def stats(self, date: Optional[str] = None) -> dict:
        """Honest track-record stats over RESOLVED trades."""
        with self._lock:
            con = self._conn()
            try:
                q = "SELECT direction, conviction, outcome, r_multiple FROM paper_trades WHERE status != 'OPEN'"
                params: list = []
                if date:
                    q += " AND date=?"; params.append(date)
                rows = con.execute(q, params).fetchall()
            finally:
                con.close()

        n = len(rows)
        if n == 0:
            return {"n": 0}
        wins = sum(1 for r in rows if r["outcome"] == "WIN")
        loss = sum(1 for r in rows if r["outcome"] == "LOSS")
        rs   = [r["r_multiple"] for r in rows if r["r_multiple"] is not None]
        decided = wins + loss
        by_dir: dict = {}
        for d in ("CE", "PE"):
            drs = [r["r_multiple"] for r in rows if r["direction"] == d and r["r_multiple"] is not None]
            dw  = sum(1 for r in rows if r["direction"] == d and r["outcome"] == "WIN")
            dd  = sum(1 for r in rows if r["direction"] == d and r["outcome"] in ("WIN", "LOSS"))
            if dd:
                by_dir[d] = {"n": dd, "hit": round(dw / dd * 100), "avg_r": round(sum(drs) / len(drs), 2) if drs else 0.0}
        return {
            "n": n, "wins": wins, "losses": loss,
            "hit_rate": round(wins / decided * 100) if decided else None,
            "avg_r": round(sum(rs) / len(rs), 2) if rs else None,
            "expectancy": round(sum(rs) / len(rs), 2) if rs else None,
            "best_r": round(max(rs), 2) if rs else None,
            "worst_r": round(min(rs), 2) if rs else None,
            "by_dir": by_dir,
        }


_ledger: Optional[TradeLedger] = None
_ledger_lock = threading.Lock()


def get_ledger() -> TradeLedger:
    global _ledger
    with _ledger_lock:
        if _ledger is None:
            _ledger = TradeLedger()
    return _ledger
