"""
intraday_store.py  —  Intraday session data engine.

Two stores, one purpose: give every signal layer a full session memory.

  CandleStore          Live WebSocket ticks → real-time OHLCV candles
                       (1-min / 5-min / 15-min / 1-hour, forming candle always live)

  OITimeSeriesStore    3-minute option-chain snapshots from 9:15 to 3:30
                       Velocity queries: OI change, IV trend, wall shift, PCR slope
                       over any lookback window (15m / 30m / 1hr)

Thread-safe. Session-scoped RAM.
Completed bars and OI snapshots are also persisted to DuckDB via intraday_db
(background, non-blocking) so the session is replayable after market close.
"""

import datetime
import threading
import time as _time
from collections import deque
from zoneinfo import ZoneInfo

import pandas as pd

from core.constants import IST   # single source of truth (fixed +5:30, no DST)

# resolution name → seconds per candle
RESOLUTIONS = {
    "1sec":  1,
    "5sec":  5,
    "15sec": 15,
    "30sec": 30,
    "1min":  60,
    "5min":  300,
    "15min": 900,
    "60min": 3600,
}

# keep enough candles for full session + small buffer (in-memory deque length).
# Sub-minute resolutions keep a recent rolling window (full history lives in the
# persisted `candles` table + raw `ticks` ledger).
_MAXLEN = {
    "1sec":  900,   # ~15 min of 1-sec (live scalping window)
    "5sec":  720,   # ~60 min
    "15sec": 480,   # ~2 hr
    "30sec": 360,   # ~3 hr
    "1min":  400,   # ~6.5 hr at 1-min
    "5min":  80,
    "15min": 30,
    "60min": 8,
}

# ── OI snapshot cadence + retention (DERIVED, so they cannot drift apart) ────────
# The old pair was MAX_OI_SNAPSHOTS = 150 with the comment "150 x 3 min = 7.5 hours —
# more than full session", but the accept-throttle was 90s, not 180s. 150 x 90s = 3.75h
# against a 6.25h session, so the in-memory series SILENTLY DROPPED the first ~2.5 hours
# of every day — the morning, which carries the single largest OI moves (09:00-09:30 is
# 16.4% of the session's total |dOI|, measured over 6 sessions). Two readers were seeing
# a truncated day: the OI panel (dashboard) and intraday_shock.
#
# Fixed by deriving the bound from the interval instead of hard-coding a number that has
# to be re-reasoned every time the cadence changes.
OI_SNAPSHOT_SEC = 30     # accept one snapshot per index per this many seconds
_SESSION_SEC = int(6.25 * 3600)                                   # 09:15 -> 15:30
MAX_OI_SNAPSHOTS = _SESSION_SEC // OI_SNAPSHOT_SEC + 50           # + pre-open/margin


# ── Candle Builder ────────────────────────────────────────────────────────────

class _CandleBuilder:
    """
    Assembles a raw tick stream into OHLCV candles for ONE resolution.

    Fyers SymbolUpdate gives cumulative day-volume on every tick.
    Candle volume = cum_vol_at_close − cum_vol_at_open.
    """

    def __init__(self, resolution_seconds: int, maxlen: int):
        self._res  = resolution_seconds
        self._lock = threading.Lock()
        self._done: deque = deque(maxlen=maxlen)
        self._live: dict | None = None   # forming (incomplete) candle
        self._vol0: float = 0.0          # cumulative volume at candle start

    # ── internal helpers ──────────────────────────────────────────────────────

    def _bucket(self, ts: datetime.datetime) -> datetime.datetime:
        """Snap ts to the start of its candle bucket."""
        epoch   = int(ts.timestamp())
        snapped = (epoch // self._res) * self._res
        return datetime.datetime.fromtimestamp(snapped, tz=IST)

    # ── public API ────────────────────────────────────────────────────────────

    def on_tick(self, ts: datetime.datetime, ltp: float, cum_vol: float) -> "dict | None":
        """
        Process one tick.  Returns the completed bar dict when a bar closes,
        None otherwise.  Callers use the return value to trigger DB writes.
        """
        with self._lock:
            b = self._bucket(ts)

            # Guard: Fyers resets cumulative volume to 0 at day open, and can
            # also reset on WebSocket reconnect mid-session.  If cum_vol drops
            # below our baseline, treat it as a new baseline so volume never
            # goes negative and future candles remain correct.
            if cum_vol < self._vol0:
                self._vol0 = cum_vol

            if self._live is None:
                self._live = {
                    "ts": b, "open": ltp, "high": ltp,
                    "low": ltp, "close": ltp, "volume": 0.0,
                }
                self._vol0 = cum_vol
                return None

            if b > self._live["ts"]:
                # Volume was updated on every tick in the else-branch below,
                # so self._live["volume"] is already correct for the old candle.
                # Using cum_vol here would include this first tick of the NEW
                # candle in the OLD candle's volume — wrong.
                closed = dict(self._live)
                self._done.append(closed)
                # open the next candle; this tick belongs to it
                self._live = {
                    "ts": b, "open": ltp, "high": ltp,
                    "low": ltp, "close": ltp, "volume": 0.0,
                }
                self._vol0 = cum_vol
                return closed   # ← signal to caller that a bar just closed
            else:
                self._live["high"]   = max(self._live["high"], ltp)
                self._live["low"]    = min(self._live["low"],  ltp)
                self._live["close"]  = ltp
                self._live["volume"] = max(0.0, cum_vol - self._vol0)
                return None

    def forming(self) -> dict | None:
        with self._lock:
            return dict(self._live) if self._live else None

    def completed(self) -> list[dict]:
        with self._lock:
            return list(self._done)


class CandleStore:
    """
    Per-symbol, per-resolution candle store.
    One _CandleBuilder for every (symbol, resolution) pair.
    """

    def __init__(self, symbols: list[str]):
        self._builders: dict[str, dict[str, _CandleBuilder]] = {
            sym: {
                res: _CandleBuilder(secs, _MAXLEN[res])
                for res, secs in RESOLUTIONS.items()
            }
            for sym in symbols
        }

    def on_tick(self, sym: str, ltp: float, cum_vol: float,
                ts: datetime.datetime | None = None):
        """Call on every WebSocket tick for the given symbol."""
        if sym not in self._builders:
            return
        t = ts or datetime.datetime.now(tz=IST)
        for res, builder in self._builders[sym].items():
            closed = builder.on_tick(t, ltp, cum_vol)
            if closed:
                _idb_write_candle(sym, res, closed)

    def forming_candle(self, sym: str, res: str) -> dict | None:
        b = self._builders.get(sym, {}).get(res)
        return b.forming() if b else None

    def to_dataframe(self, sym: str, res: str,
                     include_forming: bool = True) -> pd.DataFrame:
        """
        Returns completed candles + (optionally) the live forming candle
        as a sorted DataFrame with columns [ts, open, high, low, close, volume].
        """
        b = self._builders.get(sym, {}).get(res)
        if not b:
            return pd.DataFrame()
        rows = b.completed()
        if include_forming:
            f = b.forming()
            if f:
                rows = rows + [f]
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)


# ── DB write helpers (non-blocking; lazy-import to avoid circular deps) ───────

def _idb_write_candle(sym: str, res: str, bar: dict) -> None:
    try:
        from intraday_db import idb
        idb.write_candle(sym, res, bar)
    except Exception:
        pass


def _idb_write_oi(snap) -> None:
    try:
        from intraday_db import idb
        idb.write_oi_snapshot(snap)
    except Exception:
        pass


def record_tick(
    sym:      str,
    ts,
    ltp:      float,
    cum_vol:  float = 0.0,
    day_open: float = 0.0,
    day_high: float = 0.0,
    day_low:  float = 0.0,
    ch:       float = 0.0,
    chp:      float = 0.0,
) -> None:
    """
    Persist a raw WebSocket tick to DuckDB and feed the candle builders.

    Call this from every WebSocket SymbolUpdate callback — it is the single
    entry point that keeps both the in-memory candle store and the persistent
    tick ledger in sync.

    All DB I/O is non-blocking (queue-based background thread).
    """
    # 1. Feed candle builders for all 4 resolutions
    candle_store.on_tick(sym, ltp, cum_vol, ts)

    # 2. Persist raw tick to DuckDB
    try:
        from intraday_db import idb
        idb.write_tick(sym, ts, ltp, cum_vol,
                       day_open, day_high, day_low, ch, chp)
    except Exception:
        pass


# ── OI Snapshot ───────────────────────────────────────────────────────────────

class OISnapshot:
    """
    One timestamped capture of option-chain state for one index.
    Lightweight: __slots__ keeps memory low for 150 snapshots/index.
    """
    __slots__ = (
        "ts", "sym", "spot", "atm", "pcr",
        "total_call_oi", "total_put_oi",
        "atm_call_oi", "atm_put_oi",
        "atm_call_iv", "atm_put_iv", "atm_iv",
        "atm_call_prem", "atm_put_prem",
        "call_wall", "put_wall", "max_pain",
        "near_call_oi", "near_put_oi",
        "put_skew",
    )

    def __init__(
        self, ts, sym, spot, atm, pcr,
        total_call_oi, total_put_oi,
        atm_call_oi, atm_put_oi,
        atm_call_iv, atm_put_iv,
        atm_call_prem, atm_put_prem,
        call_wall, put_wall, max_pain,
        near_call_oi, near_put_oi, put_skew,
    ):
        self.ts            = ts
        self.sym           = sym
        self.spot          = spot
        self.atm           = atm
        self.pcr           = pcr
        self.total_call_oi = total_call_oi
        self.total_put_oi  = total_put_oi
        self.atm_call_oi   = atm_call_oi
        self.atm_put_oi    = atm_put_oi
        self.atm_call_iv   = atm_call_iv
        self.atm_put_iv    = atm_put_iv
        self.atm_iv        = ((atm_call_iv + atm_put_iv) / 2
                               if (atm_call_iv and atm_put_iv)
                               else (atm_call_iv or atm_put_iv or 0.0))
        self.atm_call_prem = atm_call_prem
        self.atm_put_prem  = atm_put_prem
        self.call_wall     = call_wall
        self.put_wall      = put_wall
        self.max_pain      = max_pain
        self.near_call_oi  = near_call_oi
        self.near_put_oi   = near_put_oi
        self.put_skew      = put_skew


# ── OI Time-Series Store ──────────────────────────────────────────────────────

class OITimeSeriesStore:
    """
    Stores timestamped OI snapshots for all indices.
    Exposes velocity() — the core quant query:
      "Compared to X minutes ago, what changed and how fast?"
    """

    def __init__(self, symbols: list[str]):
        self._lock = threading.Lock()
        self._s: dict[str, deque] = {
            sym: deque(maxlen=MAX_OI_SNAPSHOTS) for sym in symbols
        }
        # throttle: don't accept two snapshots within OI_SNAPSHOT_SEC of each other
        self._last_ts: dict[str, float] = {sym: 0.0 for sym in symbols}

    def add(self, snap: OISnapshot, min_interval_sec: int = OI_SNAPSHOT_SEC):
        """Add snapshot. Silently drops if called too soon after last add.

        This is a SECOND, independent rate limit on top of the poller's sleep, and the two
        disagreed: the poller ran at 30s before 11:30, but this throttle was a flat 90s, so
        oi_snapshots (PCR, OI walls, max pain, ATM IV) was 90-second data ALL DAY even in
        the fast window. Chain legs went to the mirror 3x more often than the aggregate
        computed from the very same fetch. Now both are OI_SNAPSHOT_SEC."""
        accepted = False
        with self._lock:
            now = _time.time()
            if now - self._last_ts.get(snap.sym, 0) < min_interval_sec:
                return
            if snap.sym in self._s:
                self._s[snap.sym].append(snap)
                self._last_ts[snap.sym] = now
                accepted = True
        if accepted:
            _idb_write_oi(snap)

    def latest(self, sym: str) -> OISnapshot | None:
        with self._lock:
            q = self._s.get(sym)
            return q[-1] if q else None

    def series(self, sym: str) -> list:
        """Copy of this index's session OI snapshots (oldest→newest) for charting."""
        with self._lock:
            return list(self._s.get(sym, []))

    def count(self, sym: str) -> int:
        with self._lock:
            return len(self._s.get(sym, []))

    @staticmethod
    def _nearest_in(series: list, minutes_ago: int) -> "OISnapshot | None":
        """
        Find the snapshot in `series` closest to `minutes_ago` ago.
        Tolerance fixed at 270 s (3 × 90-second interval).
        Works on an already-copied list so no lock is needed.
        """
        if not series:
            return None
        target = datetime.datetime.now(tz=IST) - datetime.timedelta(minutes=minutes_ago)
        best   = min(series, key=lambda s: abs((s.ts - target).total_seconds()))
        if abs((best.ts - target).total_seconds()) > 270:
            return None
        return best

    # ── velocity report ───────────────────────────────────────────────────────

    def velocity(self, sym: str) -> dict:
        """
        Full velocity report for one index.

        Sections:
          oi       — call/put OI change over 30m and 1hr, near-ATM 30m
          iv       — ATM IV change, put skew change, regime tag
          premium  — ATM call/put premium change over 15m and 30m
          walls    — call/put wall position now vs 30m and 1hr ago
          pcr      — PCR level + slope over 30m

        Returns {"has_data": False} when fewer than 3 snapshots exist
        (need ~9 minutes of session history before results are meaningful).

        Single lock acquisition: the deque is copied once, then all lookbacks
        are computed on the snapshot — no repeated lock contention.
        """
        with self._lock:
            series = list(self._s.get(sym, []))

        # Discard any snapshots from a previous calendar day — comparing
        # today's OI against yesterday's would produce completely wrong
        # velocity numbers if the app runs past midnight or is left running.
        today = datetime.datetime.now(tz=IST).date()
        series = [s for s in series if s.ts.date() == today]

        snap_count = len(series)
        now = series[-1] if series else None
        if not now or snap_count < 3:
            return {"has_data": False, "snap_count": snap_count}

        s15 = self._nearest_in(series, 15)
        s30 = self._nearest_in(series, 30)
        s60 = self._nearest_in(series, 60)

        def _d(attr: str, ref: OISnapshot | None):
            if ref is None:
                return None
            return getattr(now, attr, 0) - getattr(ref, attr, 0)

        # ── OI section ────────────────────────────────────────────────────────
        oi = {
            "call_30m":      _d("total_call_oi", s30),
            "put_30m":       _d("total_put_oi",  s30),
            "call_1hr":      _d("total_call_oi", s60),
            "put_1hr":       _d("total_put_oi",  s60),
            "atm_call_30m":  _d("atm_call_oi",   s30),
            "atm_put_30m":   _d("atm_put_oi",    s30),
            "near_call_30m": _d("near_call_oi",  s30),
            "near_put_30m":  _d("near_put_oi",   s30),
        }

        # ── IV section ────────────────────────────────────────────────────────
        iv_ch_1hr = _d("atm_iv", s60) or 0
        iv = {
            "now":          now.atm_iv,
            "change_30m":   _d("atm_iv",   s30),
            "change_1hr":   iv_ch_1hr,
            "skew_now":     now.put_skew,
            "skew_ch_1hr":  _d("put_skew", s60),
            "regime": (
                "expanding"   if iv_ch_1hr >  1.0 else
                "contracting" if iv_ch_1hr < -1.0 else
                "stable"
            ),
        }

        # ── Premium section ───────────────────────────────────────────────────
        premium = {
            "call_now":    now.atm_call_prem,
            "put_now":     now.atm_put_prem,
            "call_ch_15m": _d("atm_call_prem", s15),
            "put_ch_15m":  _d("atm_put_prem",  s15),
            "call_ch_30m": _d("atm_call_prem", s30),
            "put_ch_30m":  _d("atm_put_prem",  s30),
        }

        # ── Walls section ─────────────────────────────────────────────────────
        cws_1hr = _d("call_wall", s60)
        pws_1hr = _d("put_wall",  s60)
        walls = {
            "call_now":       now.call_wall,
            "put_now":        now.put_wall,
            "call_30m_ago":   s30.call_wall if s30 else None,
            "put_30m_ago":    s30.put_wall  if s30 else None,
            "call_1hr_ago":   s60.call_wall if s60 else None,
            "put_1hr_ago":    s60.put_wall  if s60 else None,
            "call_shift_1hr": cws_1hr,
            "put_shift_1hr":  pws_1hr,
        }

        # ── PCR section ───────────────────────────────────────────────────────
        pcr_ch_30m = _d("pcr", s30) or 0
        pcr = {
            "now":        now.pcr,
            "30m_ago":    s30.pcr if s30 else None,
            "1hr_ago":    s60.pcr if s60 else None,
            "change_30m": pcr_ch_30m,
            "change_1hr": _d("pcr", s60),
            "trend": (
                "rising"  if pcr_ch_30m >  0.05 else
                "falling" if pcr_ch_30m < -0.05 else
                "stable"
            ),
        }

        return {
            "has_data":   True,
            "snap_count": snap_count,
            "oi":      oi,
            "iv":      iv,
            "premium": premium,
            "walls":   walls,
            "pcr":     pcr,
        }


# ── Snapshot builder (called from dashboard after every OI fetch) ─────────────

def build_oi_snapshot(
    sym: str,
    spot: float,
    strike_map: dict,
    total_c_oi: int,
    total_p_oi: int,
    pcr: float,
    max_pain: float,
) -> OISnapshot | None:
    """
    Extract an OISnapshot from data already fetched by fetch_option_chain().
    Returns None when inputs are insufficient.
    """
    if not strike_map or not spot:
        return None

    ts      = datetime.datetime.now(tz=IST)
    strikes = sorted(strike_map.keys())
    atm     = min(strikes, key=lambda x: abs(x - spot))
    atm_idx = strikes.index(atm)

    def _iv(sp: int, side: str) -> float:
        return ((strike_map.get(sp, {}).get(side, {}).get("greeks") or {}).get("iv") or 0.0)

    atm_ce = strike_map.get(atm, {}).get("CE", {})
    atm_pe = strike_map.get(atm, {}).get("PE", {})

    # near-ATM zone: ±5 strikes
    near = strikes[max(0, atm_idx - 5): atm_idx + 6]
    near_c_oi = sum((strike_map[sp].get("CE", {}).get("oi") or 0) for sp in near)
    near_p_oi = sum((strike_map[sp].get("PE", {}).get("oi") or 0) for sp in near)

    call_wall = max(strikes, key=lambda sp: (strike_map[sp].get("CE", {}).get("oi") or 0))
    put_wall  = max(strikes, key=lambda sp: (strike_map[sp].get("PE", {}).get("oi") or 0))

    # OTM skew: 1.5% OTM strikes
    otm_p = min(strikes, key=lambda x: abs(x - spot * 0.985))
    otm_c = min(strikes, key=lambda x: abs(x - spot * 1.015))
    put_skew = _iv(otm_p, "PE") - _iv(otm_c, "CE")

    return OISnapshot(
        ts=ts, sym=sym, spot=spot, atm=atm, pcr=pcr,
        total_call_oi=total_c_oi,   total_put_oi=total_p_oi,
        atm_call_oi=atm_ce.get("oi") or 0,
        atm_put_oi=atm_pe.get("oi") or 0,
        atm_call_iv=_iv(atm, "CE"),
        atm_put_iv=_iv(atm, "PE"),
        atm_call_prem=atm_ce.get("ltp") or 0.0,
        atm_put_prem=atm_pe.get("ltp") or 0.0,
        call_wall=call_wall, put_wall=put_wall, max_pain=max_pain,
        near_call_oi=near_c_oi, near_put_oi=near_p_oi,
        put_skew=put_skew,
    )


# ── Session helpers ───────────────────────────────────────────────────────────

def session_phase(at=None) -> tuple[str, float, str]:
    """(phase_name, confidence_multiplier, caution) — a thin projection of
    session_strategy(), the single source of truth for the time-of-day phase.

    Previously this carried its OWN divergent windows/multipliers (OPENING ended
    09:45 here but 10:00 in session_strategy; conf-mult 0.65 vs 0.70), so the
    overview panel and the Trade Book could disagree on the phase between 09:45 and
    10:00. Now both derive from one table — no drift."""
    s = session_strategy(at)
    return s["phase"], s["conf_mult"], s["caution"]


def session_strategy(at=None) -> dict:
    """
    Time-of-day STRATEGY profile — the richer sibling of session_phase().

    The same six windows, but each now carries the actual trade-shaping rules,
    not just a confidence multiplier:

      allow_new_entry  — hard gate: no new trades open in noisy windows
                         (first 5 min, lunch, final 15 min).
      min_conf         — phase-specific confidence floor to OPEN (>= global 50).
      stop_mult /      — risk geometry: wider stops at the volatile open,
      target_mult        tighter/faster targets into the theta-heavy close.
      reentry_cooldown_min — anti-churn gap, longer in chop.
      bias             — one-line description of the regime + what to favour,
                         shown in the Trade Book so the 'why now' is explicit.

    Boundary note: OPENING runs to 10:00 (the user's mental anchor) — before
    10 is gap / opening-range territory, after 10 is established trend.
    """
    T   = datetime.time
    now = (at or datetime.datetime.now(tz=IST)).time()
    if   now < T(9, 20):
        return dict(phase="PRE-OPEN", conf_mult=0.40, allow_new_entry=False, min_conf=70,
                    stop_mult=1.25, target_mult=1.00, reentry_cooldown_min=10,
                    bias="First 5 min — price discovery, no entries",
                    caution="⚠ First 5 min — wait for the range to set")
    elif now < T(10, 0):
        return dict(phase="OPENING", conf_mult=0.70, allow_new_entry=True, min_conf=58,
                    stop_mult=1.20, target_mult=1.05, reentry_cooldown_min=12,
                    bias="Opening range — gap/EOD-led, wider stops, higher bar",
                    caution="⚠ Opening (pre-10) — volatile, size down")
    elif now < T(12, 0):
        return dict(phase="MORNING", conf_mult=1.00, allow_new_entry=True, min_conf=50,
                    stop_mult=1.00, target_mult=1.00, reentry_cooldown_min=10,
                    bias="Trend-follow — prime window (post-10)", caution="")
    elif now < T(13, 0):
        return dict(phase="LUNCH", conf_mult=0.65, allow_new_entry=False, min_conf=65,
                    stop_mult=1.00, target_mult=0.90, reentry_cooldown_min=20,
                    bias="Stand aside — low-volume chop, manage only",
                    caution="⚠ Lunch — no new entries, signals unreliable")
    elif now < T(14, 30):
        return dict(phase="AFTERNOON", conf_mult=1.00, allow_new_entry=True, min_conf=52,
                    stop_mult=1.00, target_mult=1.00, reentry_cooldown_min=10,
                    bias="Institutional momentum — trend continuation", caution="")
    elif now < T(15, 15):
        return dict(phase="PRE-CLOSE", conf_mult=0.85, allow_new_entry=True, min_conf=58,
                    stop_mult=0.85, target_mult=0.80, reentry_cooldown_min=12,
                    bias="Theta-aware scalps — tighter, faster targets",
                    caution="Pre-close: take profits faster, no fresh swings")
    else:
        return dict(phase="CLOSE", conf_mult=0.30, allow_new_entry=False, min_conf=90,
                    stop_mult=0.80, target_mult=0.70, reentry_cooldown_min=30,
                    bias="Square-off only — no new option buys",
                    caution="⚠ Final 15 min — theta collapse, exit don't enter")


# ── Module singletons (imported by signals.py, trade_setup.py, dashboard.py) ──

_SYMBOLS = [
    "NSE:NIFTY50-INDEX",
    "NSE:NIFTYBANK-INDEX",
    "NSE:FINNIFTY-INDEX",
    "NSE:MIDCPNIFTY-INDEX",
]

candle_store = CandleStore(_SYMBOLS)
oi_store     = OITimeSeriesStore(_SYMBOLS)
