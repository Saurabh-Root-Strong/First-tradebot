"""FirstTradeBot: NIFTY 50 auto trading scaffold for Fyers.

This agent evaluates MACD, RSI, and VWAP to generate signals for NIFTY 50 stocks.
Use DRY mode first. Live trading requires a valid Fyers access token and API permissions.

Note: the provided Fyers docs link points to API Connect JS button integration.
This Python bot uses the REST/HTTP trading API, so you need a REST session token
from Fyers MyAPI and the appropriate live trading permissions.
"""

import os
import time
import math
import logging
import threading
from datetime import datetime, timedelta

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

FYERS_APP_ID = os.environ.get("FYERS_APP_ID")
FYERS_ACCESS_TOKEN = os.environ.get("FYERS_ACCESS_TOKEN")
FYERS_USER_ID = os.environ.get("FYERS_USER_ID")
FYERS_ACCOUNT_ID = os.environ.get("FYERS_ACCOUNT_ID")
FYERS_API_V2_HOST = os.environ.get("FYERS_API_V2_HOST", "https://api.fyers.in/api/v2")
FYERS_API_V3_HOST = os.environ.get("FYERS_API_V3_HOST", "https://api-t1.fyers.in/api/v3")
FYERS_QUOTE_HOST = os.environ.get("FYERS_QUOTE_HOST", "https://api-t1.fyers.in/data")
TRADING_MODE = os.environ.get("TRADING_MODE", "DRY").upper()
DEFAULT_RESOLUTION = os.environ.get("RESOLUTION", "5minute")
MAX_RISK_PERCENT = float(os.environ.get("MAX_RISK_PERCENT", "1.0"))
RISK_REWARD_RATIO = float(os.environ.get("RISK_REWARD_RATIO", "2.0"))
LOOKBACK_DAYS = int(os.environ.get("LOOKBACK_DAYS", "7"))
CYCLE_DELAY_SECONDS = int(os.environ.get("CYCLE_DELAY_SECONDS", "60"))
ORDER_DELAY_SECONDS = int(os.environ.get("ORDER_DELAY_SECONDS", "2"))
DATA_POLL_INTERVAL = int(os.environ.get("DATA_POLL_INTERVAL", "5"))
MIN_CANDLES = 30

NIFTY50_SYMBOLS = [
    "NSE:RELIANCE-EQ",
    "NSE:TCS-EQ",
    "NSE:HDFCBANK-EQ",
    "NSE:INFY-EQ",
    "NSE:ICICIBANK-EQ",
    "NSE:SBIN-EQ",
    "NSE:HINDUNILVR-EQ",
    "NSE:KOTAKBANK-EQ",
    "NSE:ITC-EQ",
    "NSE:LT-EQ",
    "NSE:AXISBANK-EQ",
    "NSE:BAJFINANCE-EQ",
    "NSE:ASIANPAINT-EQ",
    "NSE:WIPRO-EQ",
    "NSE:HCLTECH-EQ",
    "NSE:BHARTIARTL-EQ",
    "NSE:MARUTI-EQ",
    "NSE:ULTRACEMCO-EQ",
    "NSE:POWERGRID-EQ",
    "NSE:ONGC-EQ",
    "NSE:SBILIFE-EQ",
    "NSE:BAJAJFINSV-EQ",
    "NSE:COALINDIA-EQ",
    "NSE:NTPC-EQ",
    "NSE:HDFC-EQ",
    "NSE:BPCL-EQ",
    "NSE:TECHM-EQ",
    "NSE:ADANIPORTS-EQ",
    "NSE:DRREDDY-EQ",
    "NSE:BAJAJ-AUTO-EQ",
    "NSE:IOC-EQ",
    "NSE:TITAN-EQ",
    "NSE:JSWSTEEL-EQ",
    "NSE:BRITANNIA-EQ",
    "NSE:UPL-EQ",
    "NSE:HEROMOTOCO-EQ",
    "NSE:DIVISLAB-EQ",
    "NSE:ADANIENT-EQ",
    "NSE:GRASIM-EQ",
    "NSE:SBICARD-EQ",
    "NSE:M&M-EQ",
    "NSE:SHREECEM-EQ",
    "NSE:INDUSINDBK-EQ",
    "NSE:NESTLEIND-EQ",
    "NSE:TATASTEEL-EQ",
    "NSE:TATAMOTORS-EQ",
    "NSE:SUNPHARMA-EQ",
    "NSE:LTIM-EQ",
    "NSE:BAJAJHLDNG-EQ",
    "NSE:TVSMOTOR-EQ",
]

def aligned_log(symbol: str, message: str) -> None:
    logging.info("%s | %s", symbol.ljust(18), message)


class FyersClient:
    def __init__(self, access_token: str, app_id: str, api_v2_host: str, api_v3_host: str, quote_host: str):
        if TRADING_MODE == "LIVE" and (not access_token or not app_id):
            raise ValueError("FYERS_APP_ID and FYERS_ACCESS_TOKEN are required for LIVE trading mode.")

        self.access_token = access_token
        self.app_id = app_id
        self.api_v2_host = api_v2_host
        self.api_v3_host = api_v3_host
        self.quote_host = quote_host
        self.session = requests.Session()
        if access_token and app_id:
            self.session.headers.update({
                "Authorization": f"{app_id}:{access_token}",
                "Content-Type": "application/json",
            })

    def get_historical_data(self, symbol: str, resolution: str) -> pd.DataFrame:
        today = datetime.today()
        start_date = (today - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
        end_date = today.strftime("%Y-%m-%d")
        url = f"{self.quote_host}/history"
        params = {
            "symbol": symbol,
            "resolution": resolution,
            "date_format": 1,
            "range_from": start_date,
            "range_to": end_date,
            "cont_flag": "",
        }
        response = self.session.get(url, params=params, timeout=25)
        response.raise_for_status()
        data = response.json()
        if data.get("s") != "ok":
            raise RuntimeError(f"Fyers history error: {data}")
        candles = data.get("candles", [])
        if not candles:
            raise RuntimeError(f"Fyers history returned no candles: {data}")
        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        return df

    def get_account_balance(self) -> float:
        if not self.access_token:
            return 100000.0
        try:
            url = f"{self.api_v2_host}/funds"
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()
            funds = data.get("funds", {})
            cash = funds.get("cash", 0.0)
            available_margin = funds.get("available_margin", 0.0)
            return float(available_margin or cash or 100000.0)
        except Exception:
            logging.warning("Unable to fetch live funds, using fallback capital.")
            return 100000.0

    def get_quotes(self, symbols: list[str]) -> dict:
        if not symbols:
            return {}
        url = f"{self.quote_host}/quotes?symbols={','.join(symbols)}"
        response = self.session.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        if data.get("s") != "ok":
            raise RuntimeError(f"Fyers quotes error: {data}")
        return data

    def place_order(self, symbol: str, qty: int, side: str, stop_loss: float, target: float) -> dict:
        if not self.access_token:
            raise RuntimeError("Cannot place order without FYERS_ACCESS_TOKEN")

        payload = {
            "symbol": symbol,
            "qty": qty,
            "type": 2,
            "side": 1 if side == "BUY" else -1,
            "productType": "INTRADAY",
            "limitPrice": 0,
            "stopPrice": 0,
            "validity": "DAY",
            "stopLoss": round(stop_loss, 2),
            "takeProfit": round(target, 2),
            "offlineOrder": False,
            "disclosedQty": 0,
            "isSliceOrder": False,
        }
        url = f"{self.api_v3_host}/orders"
        response = self.session.post(url, json=payload, timeout=25)
        response.raise_for_status()
        return response.json()


class LiveMarketDataPoller:
    def __init__(self, client: FyersClient, symbols: list[str], interval: int = DATA_POLL_INTERVAL):
        self.client = client
        self.symbols = list(symbols)
        self.interval = interval
        self.quotes: dict[str, dict] = {}
        self.lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                data = self.client.get_quotes(self.symbols)
                self._update_quotes(data)
            except Exception as exc:
                logging.warning("Live quote poll failed: %s", exc)
            time.sleep(self.interval)

    def _update_quotes(self, data: dict) -> None:
        raw_quotes = data.get("d") or []
        with self.lock:
            for item in raw_quotes:
                if not isinstance(item, dict):
                    continue
                symbol = item.get("n")
                value = item.get("v", {})
                if symbol and isinstance(value, dict):
                    self.quotes[symbol] = {
                        "lp": value.get("lp"),
                        "bid": value.get("bid"),
                        "ask": value.get("ask"),
                        "volume": value.get("volume"),
                        "timestamp": datetime.utcnow(),
                    }

    def get_last_price(self, symbol: str) -> float | None:
        with self.lock:
            quote = self.quotes.get(symbol)
            return quote.get("lp") if quote else None


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return pd.DataFrame({"macd": macd, "signal": signal_line, "histogram": histogram})


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cumulative_tp_vol = (typical_price * df["volume"]).cumsum()
    cumulative_vol = df["volume"].cumsum()
    return cumulative_tp_vol / cumulative_vol


def calculate_quantity(account_balance: float, entry_price: float, stop_loss: float) -> int:
    risk_amount = account_balance * (MAX_RISK_PERCENT / 100)
    risk_per_share = abs(entry_price - stop_loss)
    if risk_per_share <= 0:
        return 0
    qty = int(math.floor(risk_amount / risk_per_share))
    return max(qty, 0)


def signal_from_indicators(df: pd.DataFrame) -> dict:
    macd_df = compute_macd(df["close"])
    rsi_series = compute_rsi(df["close"])
    vwap_series = compute_vwap(df)

    df = df.copy()
    df["macd"] = macd_df["macd"]
    df["signal_line"] = macd_df["signal"]
    df["rsi"] = rsi_series
    df["vwap"] = vwap_series

    latest = df.iloc[-1]
    previous = df.iloc[-2]

    bullish_cross = previous["macd"] < previous["signal_line"] and latest["macd"] > latest["signal_line"]
    bearish_cross = previous["macd"] > previous["signal_line"] and latest["macd"] < latest["signal_line"]
    rsi_ok = 30 < latest["rsi"] < 70

    if bullish_cross and latest["close"] > latest["vwap"] and rsi_ok:
        stop_loss = min(latest["low"], latest["vwap"])
        target = latest["close"] + (latest["close"] - stop_loss) * RISK_REWARD_RATIO
        return {
            "signal": "BUY",
            "entry_price": float(latest["close"]),
            "stop_loss": float(stop_loss),
            "target": float(target),
            "rsi": float(latest["rsi"]),
            "vwap": float(latest["vwap"]),
        }

    if bearish_cross and latest["close"] < latest["vwap"] and rsi_ok:
        stop_loss = max(latest["high"], latest["vwap"])
        target = latest["close"] - (stop_loss - latest["close"]) * RISK_REWARD_RATIO
        return {
            "signal": "SELL",
            "entry_price": float(latest["close"]),
            "stop_loss": float(stop_loss),
            "target": float(target),
            "rsi": float(latest["rsi"]),
            "vwap": float(latest["vwap"]),
        }

    return {"signal": "HOLD"}


def run_trading_cycle(client: FyersClient, live_data: LiveMarketDataPoller | None, symbol: str) -> None:
    try:
        df = client.get_historical_data(symbol, DEFAULT_RESOLUTION)
        if len(df) < MIN_CANDLES:
            aligned_log(symbol, f"skipping, only {len(df)} candles available")
            return

        signal = signal_from_indicators(df)
        live_price = live_data.get_last_price(symbol) if live_data else None
        if live_price is not None and signal["signal"] != "HOLD":
            signal["entry_price"] = live_price
            if signal["signal"] == "BUY":
                signal["target"] = live_price + (live_price - signal["stop_loss"]) * RISK_REWARD_RATIO
            else:
                signal["target"] = live_price - (signal["stop_loss"] - live_price) * RISK_REWARD_RATIO

        if signal["signal"] == "HOLD":
            aligned_log(symbol, "no trade signal")
            return

        capital = client.get_account_balance()
        qty = calculate_quantity(capital, signal["entry_price"], signal["stop_loss"])
        if qty == 0:
            aligned_log(symbol, "zero quantity, trade skipped")
            return

        aligned_log(
            symbol,
            f"{signal['signal']} | entry={signal['entry_price']:.2f} stop={signal['stop_loss']:.2f} target={signal['target']:.2f} qty={qty} rsi={signal['rsi']:.1f} vwap={signal['vwap']:.2f}",
        )

        if TRADING_MODE == "LIVE":
            result = client.place_order(
                symbol=symbol,
                qty=qty,
                side=signal["signal"],
                stop_loss=signal["stop_loss"],
                target=signal["target"],
            )
            aligned_log(symbol, f"live order sent: {result}")
        else:
            aligned_log(symbol, "dry run: order not sent")
    except Exception as exc:
        logging.exception("Error processing %s: %s", symbol, exc)


def main() -> None:
    logging.info("Starting trading agent in %s mode", TRADING_MODE)
    client = FyersClient(
        access_token=FYERS_ACCESS_TOKEN,
        app_id=FYERS_APP_ID,
        api_v2_host=FYERS_API_V2_HOST,
        api_v3_host=FYERS_API_V3_HOST,
        quote_host=FYERS_QUOTE_HOST,
    )

    live_data = None
    if TRADING_MODE == "LIVE" and FYERS_ACCESS_TOKEN and FYERS_APP_ID:
        live_data = LiveMarketDataPoller(client, NIFTY50_SYMBOLS, interval=DATA_POLL_INTERVAL)
        live_data.start()
    else:
        logging.info("Live quotes disabled: running in DRY mode or missing live credentials.")

    try:
        while True:
            for symbol in NIFTY50_SYMBOLS:
                run_trading_cycle(client, live_data, symbol)
                time.sleep(ORDER_DELAY_SECONDS)
            logging.info("Cycle complete, waiting %s seconds", CYCLE_DELAY_SECONDS)
            time.sleep(CYCLE_DELAY_SECONDS)
    finally:
        if live_data is not None:
            live_data.stop()


if __name__ == "__main__":
    main()
