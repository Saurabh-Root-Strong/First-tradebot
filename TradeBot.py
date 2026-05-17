"""
TradeBot: Production-grade NIFTY 50 intraday trading agent for Fyers.

Architectural improvements over FirstTradeBot.py:
  1. DailyTradeJournal  — thread-safe CSV log; persists every trade with P&L,
                          status, and order-id. Survives restarts on same day.
  2. Parallel scanning   — ThreadPoolExecutor processes all 50 symbols concurrently
                          instead of sequentially with 2-s sleep → ~10x faster cycle.
  3. Thread-local HTTP   — one requests.Session per worker thread (Session is NOT
                          thread-safe for concurrent use).
  4. VWAP fix            — resets at each trading day boundary (was cumulating
                          across 7 days, giving a 7-day VWAP, not intraday VWAP).
  5. Market hours guard  — skips cycles outside NSE hours (9:15–15:30 IST, weekdays).
  6. Auto squareoff      — stops new orders at 15:20 IST to avoid holding overnight.
  7. Circuit breakers    — max daily trade count + max daily drawdown % halt new orders.
  8. Balance cache       — refreshed every 5 min; was fetched once per symbol (50×/cycle).
  9. Position dedup      — skips re-entry if the symbol already has an open/filled trade.
 10. API rate limiter    — token bucket prevents hitting Fyers rate limits under parallel load.
 11. Resolution fix      — Fyers data API uses integer strings ("5"), not "5minute".
"""

import csv
import math
import os
import time
import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd
import requests

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")

# ── Config (all overridable via env vars) ──────────────────────────────────────
FYERS_APP_ID        = os.environ.get("FYERS_APP_ID", "WVDZUTO6HL-100")

# Access token: env var wins; fallback reads access_token.txt (written by fyers_auth.py)
FYERS_ACCESS_TOKEN  = os.environ.get("FYERS_ACCESS_TOKEN", "")
if not FYERS_ACCESS_TOKEN:
    _token_file = Path("access_token.txt")
    if _token_file.exists():
        FYERS_ACCESS_TOKEN = _token_file.read_text(encoding="utf-8").strip()
        # log.info call deferred — logger not ready at module load time
FYERS_API_V2_HOST   = os.environ.get("FYERS_API_V2_HOST", "https://api.fyers.in/api/v2")
FYERS_API_V3_HOST   = os.environ.get("FYERS_API_V3_HOST", "https://api-t1.fyers.in/api/v3")
FYERS_QUOTE_HOST    = os.environ.get("FYERS_QUOTE_HOST", "https://api-t1.fyers.in/data")
TRADING_MODE        = os.environ.get("TRADING_MODE", "DRY").upper()

DEFAULT_RESOLUTION  = os.environ.get("RESOLUTION", "5")        # Fyers: "5" = 5-min candles
MAX_RISK_PERCENT    = float(os.environ.get("MAX_RISK_PERCENT", "1.0"))
RISK_REWARD_RATIO   = float(os.environ.get("RISK_REWARD_RATIO", "2.0"))
LOOKBACK_DAYS       = int(os.environ.get("LOOKBACK_DAYS", "7"))
CYCLE_DELAY_SECONDS = int(os.environ.get("CYCLE_DELAY_SECONDS", "60"))
DATA_POLL_INTERVAL  = int(os.environ.get("DATA_POLL_INTERVAL", "5"))
MAX_WORKERS         = int(os.environ.get("MAX_WORKERS", "8"))
MAX_DAILY_TRADES    = int(os.environ.get("MAX_DAILY_TRADES", "20"))
MAX_DAILY_LOSS_PCT  = float(os.environ.get("MAX_DAILY_LOSS_PCT", "3.0"))
API_RATE_PER_SEC    = float(os.environ.get("API_RATE_PER_SEC", "5.0"))
TRADES_DIR          = Path(os.environ.get("TRADES_DIR", "trades"))
SQUAREOFF_HOUR      = int(os.environ.get("SQUAREOFF_HOUR", "15"))
SQUAREOFF_MINUTE    = int(os.environ.get("SQUAREOFF_MINUTE", "20"))

MIN_CANDLES = 30

NIFTY50_SYMBOLS = [
    "NSE:RELIANCE-EQ",  "NSE:TCS-EQ",        "NSE:HDFCBANK-EQ",  "NSE:INFY-EQ",
    "NSE:ICICIBANK-EQ", "NSE:SBIN-EQ",        "NSE:HINDUNILVR-EQ","NSE:KOTAKBANK-EQ",
    "NSE:ITC-EQ",       "NSE:LT-EQ",          "NSE:AXISBANK-EQ",  "NSE:BAJFINANCE-EQ",
    "NSE:ASIANPAINT-EQ","NSE:WIPRO-EQ",        "NSE:HCLTECH-EQ",   "NSE:BHARTIARTL-EQ",
    "NSE:MARUTI-EQ",    "NSE:ULTRACEMCO-EQ",  "NSE:POWERGRID-EQ", "NSE:ONGC-EQ",
    "NSE:SBILIFE-EQ",   "NSE:BAJAJFINSV-EQ",  "NSE:COALINDIA-EQ", "NSE:NTPC-EQ",
    "NSE:HDFC-EQ",      "NSE:BPCL-EQ",        "NSE:TECHM-EQ",     "NSE:ADANIPORTS-EQ",
    "NSE:DRREDDY-EQ",   "NSE:BAJAJ-AUTO-EQ",  "NSE:IOC-EQ",       "NSE:TITAN-EQ",
    "NSE:JSWSTEEL-EQ",  "NSE:BRITANNIA-EQ",   "NSE:UPL-EQ",       "NSE:HEROMOTOCO-EQ",
    "NSE:DIVISLAB-EQ",  "NSE:ADANIENT-EQ",    "NSE:GRASIM-EQ",    "NSE:SBICARD-EQ",
    "NSE:M&M-EQ",       "NSE:SHREECEM-EQ",    "NSE:INDUSINDBK-EQ","NSE:NESTLEIND-EQ",
    "NSE:TATASTEEL-EQ", "NSE:TATAMOTORS-EQ",  "NSE:SUNPHARMA-EQ", "NSE:LTIM-EQ",
    "NSE:BAJAJHLDNG-EQ","NSE:TVSMOTOR-EQ",
]


# ══════════════════════════════════════════════════════════════════════════════
# Trade record
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Trade:
    trade_id:    str
    timestamp:   str          # ISO-8601 in IST
    symbol:      str
    side:        str          # BUY | SELL
    qty:         int
    entry_price: float
    stop_loss:   float
    target:      float
    order_id:    str   = ""
    status:      str   = "PENDING"   # PENDING | DRY | FILLED | REJECTED | CLOSED
    exit_price:  float = 0.0
    pnl:         float = 0.0
    notes:       str   = ""


# ══════════════════════════════════════════════════════════════════════════════
# Daily Trade Journal
# ══════════════════════════════════════════════════════════════════════════════

class DailyTradeJournal:
    """
    Thread-safe CSV journal for all trades in the current trading session.

    File: trades/trades_YYYY-MM-DD.csv
    Loaded automatically on startup so a bot restart doesn't forget morning trades.
    """

    COLUMNS = [
        "trade_id", "timestamp", "symbol", "side", "qty",
        "entry_price", "stop_loss", "target",
        "order_id", "status", "exit_price", "pnl", "notes",
    ]

    def __init__(self, trades_dir: Path = TRADES_DIR) -> None:
        self._dir = trades_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._path = self._dir / f"trades_{date.today().isoformat()}.csv"
        self._trades: list[Trade] = []
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            with open(self._path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self.COLUMNS).writeheader()
            return
        with open(self._path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self._trades.append(Trade(
                    trade_id    = row["trade_id"],
                    timestamp   = row["timestamp"],
                    symbol      = row["symbol"],
                    side        = row["side"],
                    qty         = int(row["qty"]),
                    entry_price = float(row["entry_price"]),
                    stop_loss   = float(row["stop_loss"]),
                    target      = float(row["target"]),
                    order_id    = row.get("order_id", ""),
                    status      = row.get("status", "PENDING"),
                    exit_price  = float(row.get("exit_price") or 0),
                    pnl         = float(row.get("pnl") or 0),
                    notes       = row.get("notes", ""),
                ))
        log.info("Loaded %d existing trades from %s", len(self._trades), self._path)

    # ── Write operations ───────────────────────────────────────────────────────

    def log_trade(self, trade: Trade) -> None:
        """Append a new trade record to memory and CSV."""
        with self._lock:
            self._trades.append(trade)
            with open(self._path, "a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self.COLUMNS).writerow(asdict(trade))

    def update_trade(self, trade_id: str, **fields) -> None:
        """Update fields on an existing trade and rewrite the CSV."""
        with self._lock:
            for t in self._trades:
                if t.trade_id == trade_id:
                    for k, v in fields.items():
                        setattr(t, k, v)
                    self._rewrite()
                    return

    def _rewrite(self) -> None:
        with open(self._path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.COLUMNS)
            w.writeheader()
            for t in self._trades:
                w.writerow(asdict(t))

    # ── Read operations ────────────────────────────────────────────────────────

    @property
    def filepath(self) -> Path:
        return self._path

    def all_trades(self) -> list[Trade]:
        with self._lock:
            return list(self._trades)

    def active_symbols(self) -> set[str]:
        """Symbols with an open or pending position — used to skip re-entry."""
        with self._lock:
            return {t.symbol for t in self._trades if t.status in ("PENDING", "FILLED", "DRY")}

    @property
    def trade_count(self) -> int:
        with self._lock:
            return len(self._trades)

    @property
    def realized_pnl(self) -> float:
        with self._lock:
            return sum(t.pnl for t in self._trades if t.status == "CLOSED")

    def print_summary(self) -> None:
        trades = self.all_trades()
        sep = "-" * 100
        log.info(sep)
        log.info("DAILY TRADE SUMMARY  %s", date.today().isoformat())
        log.info(sep)
        log.info("%-8s  %-20s  %-4s  %5s  %9s  %9s  %9s  %-9s  %+8s",
                 "ID", "Symbol", "Side", "Qty", "Entry", "SL", "Target", "Status", "P&L")
        log.info(sep)
        for t in trades:
            log.info("%-8s  %-20s  %-4s  %5d  %9.2f  %9.2f  %9.2f  %-9s  %+8.2f",
                     t.trade_id, t.symbol, t.side, t.qty,
                     t.entry_price, t.stop_loss, t.target,
                     t.status, t.pnl)
        log.info(sep)
        closed  = [t for t in trades if t.status == "CLOSED"]
        wins    = [t for t in closed  if t.pnl > 0]
        losses  = [t for t in closed  if t.pnl < 0]
        open_n  = len([t for t in trades if t.status in ("PENDING", "FILLED")])
        log.info(
            "Total=%d | Open=%d | Closed=%d (W:%d L:%d) | Realized P&L=%+.2f",
            len(trades), open_n, len(closed), len(wins), len(losses), self.realized_pnl,
        )
        log.info(sep)


# ══════════════════════════════════════════════════════════════════════════════
# API Rate Limiter  (token-bucket, thread-safe)
# ══════════════════════════════════════════════════════════════════════════════

class RateLimiter:
    def __init__(self, calls_per_second: float) -> None:
        self._interval = 1.0 / max(calls_per_second, 0.01)
        self._lock = threading.Lock()
        self._next_slot = time.monotonic()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                if now >= self._next_slot:
                    self._next_slot = now + self._interval
                    return
                wait = self._next_slot - now
            time.sleep(wait)


# ══════════════════════════════════════════════════════════════════════════════
# Fyers REST Client  (thread-safe: one Session per worker thread)
# ══════════════════════════════════════════════════════════════════════════════

class FyersClient:
    def __init__(
        self,
        access_token: str,
        app_id: str,
        api_v2_host: str,
        api_v3_host: str,
        quote_host: str,
        rate_limiter: Optional[RateLimiter] = None,
    ) -> None:
        if TRADING_MODE == "LIVE" and not (access_token and app_id):
            raise ValueError("FYERS_APP_ID and FYERS_ACCESS_TOKEN are required for LIVE mode.")
        self._auth        = f"{app_id}:{access_token}" if (app_id and access_token) else ""
        self.api_v2_host  = api_v2_host
        self.api_v3_host  = api_v3_host
        self.quote_host   = quote_host
        self._rate        = rate_limiter
        self._local       = threading.local()

    @property
    def _session(self) -> requests.Session:
        if not hasattr(self._local, "s"):
            s = requests.Session()
            if self._auth:
                s.headers.update({
                    "Authorization": self._auth,
                    "Content-Type": "application/json",
                })
            self._local.s = s
        return self._local.s

    def _throttle(self) -> None:
        if self._rate:
            self._rate.acquire()

    def get_historical_data(self, symbol: str, resolution: str = DEFAULT_RESOLUTION) -> pd.DataFrame:
        now_ist   = datetime.now(tz=IST)
        range_to  = now_ist.strftime("%Y-%m-%d")
        range_from = (now_ist - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
        self._throttle()
        resp = self._session.get(
            f"{self.quote_host}/history",
            params={
                "symbol":      symbol,
                "resolution":  resolution,
                "date_format": 1,
                "range_from":  range_from,
                "range_to":    range_to,
                "cont_flag":   "",
            },
            timeout=25,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("s") != "ok":
            raise RuntimeError(f"History error [{symbol}]: {data}")
        candles = data.get("candles") or []
        if not candles:
            raise RuntimeError(f"No candles for {symbol}")
        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        # Convert epoch → IST timezone-aware timestamps for correct date grouping in VWAP
        df["timestamp"] = (
            pd.to_datetime(df["timestamp"], unit="s", utc=True)
            .dt.tz_convert("Asia/Kolkata")
        )
        return df

    def get_account_balance(self) -> float:
        if not self._auth:
            return 100_000.0
        self._throttle()
        try:
            resp = self._session.get(f"{self.api_v2_host}/funds", timeout=15)
            resp.raise_for_status()
            funds = resp.json().get("funds", {})
            val = funds.get("available_margin") or funds.get("cash") or 100_000.0
            return float(val)
        except Exception:
            log.warning("Funds fetch failed — using fallback capital 100,000")
            return 100_000.0

    def get_quotes(self, symbols: list[str]) -> dict:
        if not symbols:
            return {}
        self._throttle()
        resp = self._session.get(
            f"{self.quote_host}/quotes",
            params={"symbols": ",".join(symbols)},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("s") != "ok":
            raise RuntimeError(f"Quotes error: {data}")
        return data

    def place_order(
        self, symbol: str, qty: int, side: str, stop_loss: float, target: float
    ) -> dict:
        if not self._auth:
            raise RuntimeError("Cannot place order: no credentials")
        self._throttle()
        resp = self._session.post(
            f"{self.api_v3_host}/orders/sync",
            json={
                "symbol":      symbol,
                "qty":         qty,
                "type":        2,                          # Market order
                "side":        1 if side == "BUY" else -1,
                "productType": "INTRADAY",
                "limitPrice":  0,
                "stopPrice":   0,
                "validity":    "DAY",
                "stopLoss":    round(stop_loss, 2),
                "takeProfit":  round(target, 2),
                "offlineOrder": False,
                "disclosedQty": 0,
                "isSliceOrder": False,
            },
            timeout=25,
        )
        resp.raise_for_status()
        return resp.json()


# ══════════════════════════════════════════════════════════════════════════════
# Live Quote Poller  (background thread)
# ══════════════════════════════════════════════════════════════════════════════

class LiveMarketDataPoller:
    def __init__(
        self, client: FyersClient, symbols: list[str], interval: int = DATA_POLL_INTERVAL
    ) -> None:
        self._client   = client
        self._symbols  = list(symbols)
        self._interval = interval
        self._quotes: dict[str, dict] = {}
        self._lock  = threading.Lock()
        self._stop  = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="QuotePoller")

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=10)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                data = self._client.get_quotes(self._symbols)
                self._ingest(data)
            except Exception as exc:
                log.warning("Quote poll error: %s", exc)
            time.sleep(self._interval)

    def _ingest(self, data: dict) -> None:
        with self._lock:
            for item in data.get("d") or []:
                if not isinstance(item, dict):
                    continue
                sym = item.get("n")
                val = item.get("v") or {}
                if sym:
                    self._quotes[sym] = {
                        "lp":     val.get("lp"),
                        "bid":    val.get("bid"),
                        "ask":    val.get("ask"),
                        "volume": val.get("volume"),
                        "ts":     datetime.now(tz=IST),
                    }

    def last_price(self, symbol: str) -> Optional[float]:
        with self._lock:
            q = self._quotes.get(symbol)
            return float(q["lp"]) if q and q.get("lp") is not None else None


# ══════════════════════════════════════════════════════════════════════════════
# Technical Indicators
# ══════════════════════════════════════════════════════════════════════════════

def compute_macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    sig  = macd.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"macd": macd, "signal": sig, "hist": macd - sig})


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1 / period, min_periods=period).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1 / period, min_periods=period).mean()
    return 100 - (100 / (1 + gain / loss))


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Intraday VWAP — resets at the start of each calendar date (IST).
    The original code cumulated across 7 days giving a multi-day VWAP,
    which is meaningless for intraday entries.
    """
    tp  = (df["high"] + df["low"] + df["close"]) / 3
    tpv = tp * df["volume"]
    # Group by IST date so the cumsum resets each trading day
    grp_date = df["timestamp"].dt.date
    cum_tpv  = tpv.groupby(grp_date).cumsum()
    cum_vol  = df["volume"].groupby(grp_date).cumsum()
    return cum_tpv / cum_vol


def signal_from_indicators(df: pd.DataFrame) -> dict:
    macd_df  = compute_macd(df["close"])
    rsi_vals = compute_rsi(df["close"])
    vwap     = compute_vwap(df)

    m_now,  m_prev = float(macd_df["macd"].iloc[-1]),   float(macd_df["macd"].iloc[-2])
    s_now,  s_prev = float(macd_df["signal"].iloc[-1]), float(macd_df["signal"].iloc[-2])
    rsi_val        = float(rsi_vals.iloc[-1])
    vwap_val       = float(vwap.iloc[-1])
    close          = float(df["close"].iloc[-1])
    low            = float(df["low"].iloc[-1])
    high           = float(df["high"].iloc[-1])

    bull_cross = m_prev < s_prev and m_now > s_now
    bear_cross = m_prev > s_prev and m_now < s_now
    rsi_ok     = 30 < rsi_val < 70

    if bull_cross and close > vwap_val and rsi_ok:
        sl = min(low, vwap_val)
        tp = close + (close - sl) * RISK_REWARD_RATIO
        return {"signal": "BUY",  "entry_price": close, "stop_loss": sl, "target": tp,
                "rsi": rsi_val, "vwap": vwap_val}

    if bear_cross and close < vwap_val and rsi_ok:
        sl = max(high, vwap_val)
        tp = close - (sl - close) * RISK_REWARD_RATIO
        return {"signal": "SELL", "entry_price": close, "stop_loss": sl, "target": tp,
                "rsi": rsi_val, "vwap": vwap_val}

    return {"signal": "HOLD"}


def calculate_quantity(balance: float, entry: float, stop_loss: float) -> int:
    risk_amount    = balance * (MAX_RISK_PERCENT / 100)
    risk_per_share = abs(entry - stop_loss)
    if risk_per_share <= 0:
        return 0
    return max(int(math.floor(risk_amount / risk_per_share)), 0)


# ══════════════════════════════════════════════════════════════════════════════
# Trading Engine  (main orchestrator)
# ══════════════════════════════════════════════════════════════════════════════

class TradingEngine:
    """
    Drives the full trading loop:
      - market hours + squareoff guard
      - parallel symbol scanning
      - circuit breakers (trade count + daily drawdown)
      - order dispatch + journal logging
    """

    def __init__(
        self,
        client:   FyersClient,
        journal:  DailyTradeJournal,
        symbols:  list[str],
        live_data: Optional[LiveMarketDataPoller] = None,
    ) -> None:
        self._client  = client
        self._journal = journal
        self._symbols = symbols
        self._live    = live_data
        # Balance cache
        self._balance:    float = 0.0
        self._balance_ts: float = 0.0
        self._bal_lock    = threading.Lock()

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _now() -> datetime:
        return datetime.now(tz=IST)

    def is_market_open(self) -> bool:
        now = self._now()
        if now.weekday() >= 5:
            return False
        o = now.replace(hour=9,  minute=15, second=0, microsecond=0)
        c = now.replace(hour=15, minute=30, second=0, microsecond=0)
        return o <= now < c

    def is_squareoff_time(self) -> bool:
        now = self._now()
        sq  = now.replace(hour=SQUAREOFF_HOUR, minute=SQUAREOFF_MINUTE, second=0, microsecond=0)
        return now >= sq

    def _get_balance(self) -> float:
        with self._bal_lock:
            if time.monotonic() - self._balance_ts > 300:   # 5-min TTL
                self._balance    = self._client.get_account_balance()
                self._balance_ts = time.monotonic()
                log.info("Balance refreshed: INR %.2f", self._balance)
            return self._balance

    def _circuit_ok(self) -> bool:
        if self._journal.trade_count >= MAX_DAILY_TRADES:
            log.warning("Circuit breaker: max daily trades (%d) reached", MAX_DAILY_TRADES)
            return False
        bal      = self._get_balance()
        max_loss = bal * (MAX_DAILY_LOSS_PCT / 100)
        if self._journal.realized_pnl <= -max_loss:
            log.warning(
                "Circuit breaker: daily loss limit hit  P&L=%.2f  limit=%.2f",
                self._journal.realized_pnl, -max_loss,
            )
            return False
        return True

    # ── Per-symbol worker (runs in thread pool) ────────────────────────────────

    def _process_symbol(self, symbol: str) -> None:
        try:
            # Skip if we already hold a position in this symbol today
            if symbol in self._journal.active_symbols():
                return

            df = self._client.get_historical_data(symbol)
            if len(df) < MIN_CANDLES:
                log.info("%-22s  skip: only %d candles", symbol, len(df))
                return

            sig = signal_from_indicators(df)
            if sig["signal"] == "HOLD":
                return

            # Prefer live tick price over last historical close
            if self._live:
                lp = self._live.last_price(symbol)
                if lp:
                    diff = abs(lp - sig["stop_loss"])
                    sig["entry_price"] = lp
                    sig["target"] = (lp + diff * RISK_REWARD_RATIO
                                     if sig["signal"] == "BUY"
                                     else lp - diff * RISK_REWARD_RATIO)

            balance = self._get_balance()
            qty     = calculate_quantity(balance, sig["entry_price"], sig["stop_loss"])
            if qty == 0:
                log.info("%-22s  skip: qty=0 (risk too small)", symbol)
                return

            log.info(
                "%-22s  %s  entry=%.2f  sl=%.2f  tp=%.2f  qty=%d  rsi=%.1f  vwap=%.2f",
                symbol, sig["signal"], sig["entry_price"],
                sig["stop_loss"], sig["target"], qty,
                sig["rsi"], sig["vwap"],
            )

            trade = Trade(
                trade_id    = str(uuid.uuid4())[:8],
                timestamp   = self._now().isoformat(),
                symbol      = symbol,
                side        = sig["signal"],
                qty         = qty,
                entry_price = sig["entry_price"],
                stop_loss   = sig["stop_loss"],
                target      = sig["target"],
            )

            if TRADING_MODE == "LIVE":
                result   = self._client.place_order(
                    symbol=symbol, qty=qty, side=sig["signal"],
                    stop_loss=sig["stop_loss"], target=sig["target"],
                )
                order_id = result.get("id", "")
                status   = "FILLED" if result.get("s") == "ok" else "REJECTED"
                notes    = "" if status == "FILLED" else str(result.get("message", ""))
                trade.order_id = order_id
                trade.status   = status
                trade.notes    = notes
                log.info("%-22s  order %s → %s", symbol, order_id, status)
            else:
                trade.status = "DRY"
                log.info("%-22s  DRY RUN — order not sent", symbol)

            self._journal.log_trade(trade)

        except Exception as exc:
            log.exception("Error processing %s: %s", symbol, exc)

    # ── Cycle + main loop ──────────────────────────────────────────────────────

    def run_cycle(self) -> None:
        if not self._circuit_ok():
            return
        log.info("-- Scanning %d symbols  workers=%d --", len(self._symbols), MAX_WORKERS)
        with ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="scanner") as pool:
            futures = {pool.submit(self._process_symbol, sym): sym for sym in self._symbols}
            for fut in as_completed(futures):
                if (exc := fut.exception()):
                    log.error("Worker error [%s]: %s", futures[fut], exc)
        self._journal.print_summary()

    def run(self) -> None:
        log.info("TradingEngine started | mode=%s | journal=%s", TRADING_MODE, self._journal.filepath)
        try:
            while True:
                if not self.is_market_open():
                    log.info("[%s IST] Market closed — waiting 60 s",
                             self._now().strftime("%H:%M"))
                    time.sleep(60)
                    continue

                if self.is_squareoff_time():
                    log.info("[%s IST] Squareoff time — no new orders",
                             self._now().strftime("%H:%M"))
                    time.sleep(CYCLE_DELAY_SECONDS)
                    continue

                self.run_cycle()
                log.info("Sleeping %d s before next cycle", CYCLE_DELAY_SECONDS)
                time.sleep(CYCLE_DELAY_SECONDS)

        except KeyboardInterrupt:
            log.info("Interrupted")
        finally:
            self._journal.print_summary()


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # Report token source so it's clear which credentials are active
    if not FYERS_ACCESS_TOKEN:
        log.warning("No access token found. Run fyers_auth.py first, or set FYERS_ACCESS_TOKEN.")
    elif Path("access_token.txt").exists() and not os.environ.get("FYERS_ACCESS_TOKEN"):
        log.info("Access token loaded from access_token.txt")
    else:
        log.info("Access token loaded from environment variable")

    log.info("App ID: %s | Mode: %s", FYERS_APP_ID, TRADING_MODE)

    rate_limiter = RateLimiter(API_RATE_PER_SEC)

    client = FyersClient(
        access_token = FYERS_ACCESS_TOKEN,
        app_id       = FYERS_APP_ID,
        api_v2_host  = FYERS_API_V2_HOST,
        api_v3_host  = FYERS_API_V3_HOST,
        quote_host   = FYERS_QUOTE_HOST,
        rate_limiter = rate_limiter,
    )

    journal = DailyTradeJournal(TRADES_DIR)

    live_data = None
    if TRADING_MODE == "LIVE" and FYERS_ACCESS_TOKEN and FYERS_APP_ID:
        live_data = LiveMarketDataPoller(client, NIFTY50_SYMBOLS)
        live_data.start()
        log.info("Live quote poller started")
    else:
        log.info("DRY mode — live quotes disabled")

    engine = TradingEngine(client, journal, NIFTY50_SYMBOLS, live_data)
    try:
        engine.run()
    finally:
        if live_data:
            live_data.stop()


if __name__ == "__main__":
    main()
