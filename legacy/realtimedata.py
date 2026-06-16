"""
Live tick feed for NSE indices: Nifty 50, Bank Nifty, Fin Nifty, Midcap Nifty.
Token is read from access_token.txt — run fyers_auth.py to refresh each morning.

Usage:  .venv\Scripts\python.exe realtimedata.py
Stop:   Ctrl+C
"""

import datetime
import sys
from pathlib import Path

from fyers_apiv3.FyersWebsocket import data_ws

APP_ID     = "WVDZUTO6HL-100"
TOKEN_FILE = Path("access_token.txt")

INDEX_SYMBOLS = [
    "NSE:NIFTY50-INDEX",
    "NSE:NIFTYBANK-INDEX",
    "NSE:FINNIFTY-INDEX",
    "NSE:MIDCPNIFTY-INDEX",
]

LABELS = {
    "NSE:NIFTY50-INDEX":    "NIFTY 50    ",
    "NSE:NIFTYBANK-INDEX":  "BANK NIFTY  ",
    "NSE:FINNIFTY-INDEX":   "FIN NIFTY   ",
    "NSE:MIDCPNIFTY-INDEX": "MIDCAP NIFTY",
}

IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
SEP = "─" * 102

_state: dict[str, dict] = {}
_dashboard_lines = 0


def _now_ist() -> str:
    return datetime.datetime.now(tz=IST).strftime("%H:%M:%S IST")


def _fmt_ts(epoch: int) -> str:
    if not epoch:
        return "--:--:--"
    return datetime.datetime.fromtimestamp(epoch, tz=IST).strftime("%H:%M:%S")


def _print_dashboard() -> None:
    global _dashboard_lines
    if _dashboard_lines:
        sys.stdout.write(f"\033[{_dashboard_lines}A")

    rows = [SEP, f"  INDEX LIVE FEED  [{_now_ist()}]", SEP]

    for sym in INDEX_SYMBOLS:
        tick = _state.get(sym)
        if tick is None:
            rows.append(f"  {LABELS[sym]}   waiting for first tick...")
            continue

        ltp  = tick.get("ltp", 0)
        ch   = tick.get("ch", 0)
        chp  = tick.get("chp", 0)
        o    = tick.get("open_price", 0)
        h    = tick.get("high_price", 0)
        l    = tick.get("low_price", 0)
        ts   = _fmt_ts(tick.get("exch_feed_time", 0))
        clr  = "\033[92m" if ch >= 0 else "\033[91m"   # green / red
        rst  = "\033[0m"
        sign = "+" if ch >= 0 else ""

        rows.append(
            f"  {LABELS[sym]}"
            f"  {clr}{ltp:>11,.2f}{rst}"
            f"  {clr}{sign}{ch:>+8.2f}  {sign}{chp:>+6.2f}%{rst}"
            f"  O {o:>10,.2f}  H {h:>10,.2f}  L {l:>10,.2f}"
            f"  [{ts}]"
        )

    rows.append(SEP)
    # \033[K erases from cursor to end of line — clears any residual chars from a
    # previously longer line before writing the new (possibly shorter) content
    for row in rows:
        sys.stdout.write(row + "\033[K\n")
    sys.stdout.flush()
    _dashboard_lines = len(rows)   # one cursor-up per printed line


def onmessage(message: dict) -> None:
    # Status/control messages have a 'code' key but no price data
    if "code" in message:
        print(f"  [{_now_ist()}] {message.get('message', message)}")
        return

    sym = message.get("symbol")
    if sym not in INDEX_SYMBOLS:
        return

    _state[sym] = message
    _print_dashboard()


def onerror(message) -> None:
    print(f"\n  ERROR [{_now_ist()}]: {message}")


def onclose(message) -> None:
    print(f"\n  Closed [{_now_ist()}]: {message}")


def onopen() -> None:
    fyers.subscribe(symbols=INDEX_SYMBOLS, data_type="SymbolUpdate")
    fyers.keep_running()


# ── Startup ────────────────────────────────────────────────────────────────────

if not TOKEN_FILE.exists():
    print("ERROR: access_token.txt not found. Run fyers_auth.py first.")
    sys.exit(1)

raw_token    = TOKEN_FILE.read_text(encoding="utf-8").strip()
access_token = f"{APP_ID}:{raw_token}"

print(SEP)
print("  INDEX LIVE TICK FEED  —  Nifty 50 | Bank Nifty | Fin Nifty | Midcap Nifty")
print(f"  Connecting...  (Ctrl+C to stop)")
print(SEP)

fyers = data_ws.FyersDataSocket(
    access_token  = access_token,
    log_path      = "",
    litemode      = False,
    write_to_file = False,
    reconnect     = True,
    on_connect    = onopen,
    on_close      = onclose,
    on_error      = onerror,
    on_message    = onmessage,
)

fyers.connect()
