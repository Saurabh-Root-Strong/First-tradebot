"""Verify OHLCV data for all resolutions and all 4 indices."""
import requests, datetime
from pathlib import Path

APP_ID = "WVDZUTO6HL-100"
token  = Path("access_token.txt").read_text(encoding="utf-8").strip()
auth   = f"{APP_ID}:{token}"
IST    = datetime.timezone(datetime.timedelta(hours=5, minutes=30))

now       = datetime.datetime.now(tz=IST)
today     = now.strftime("%Y-%m-%d")
week_ago  = (now - datetime.timedelta(days=7)).strftime("%Y-%m-%d")
month_ago = (now - datetime.timedelta(days=60)).strftime("%Y-%m-%d")

SYMBOLS = {
    "NIFTY 50":     "NSE:NIFTY50-INDEX",
    "BANK NIFTY":   "NSE:NIFTYBANK-INDEX",
    "FIN NIFTY":    "NSE:FINNIFTY-INDEX",
    "MIDCAP NIFTY": "NSE:MIDCPNIFTY-INDEX",
}
RESOLUTIONS = {
    "5-min":  ("5",  week_ago,  today),
    "15-min": ("15", week_ago,  today),
    "60-min": ("60", week_ago,  today),
    "Daily":  ("D",  month_ago, today),
}

def fetch(sym, res, frm, to):
    resp = requests.get(
        "https://api-t1.fyers.in/data/history",
        headers={"Authorization": auth, "Content-Type": "application/json", "version": "3"},
        params={"symbol": sym, "resolution": res, "date_format": "1",
                "range_from": frm, "range_to": to, "cont_flag": "1"},
        timeout=15,
    )
    d = resp.json()
    candles = d.get("candles", [])
    return len(candles), d.get("s"), candles[-1] if candles else None

print(f"{'INDEX':<16} {'5-min':>8} {'15-min':>8} {'60-min':>8} {'Daily':>8}")
print("-" * 56)
for name, sym in SYMBOLS.items():
    row = f"{name:<16}"
    for tf, (res, frm, to) in RESOLUTIONS.items():
        n, s, last = fetch(sym, res, frm, to)
        row += f"  {n:>6}" if s == "ok" else f"  {'ERR':>6}"
    print(row)

print()
# Show last 5min candle for NIFTY
n, s, last = fetch("NSE:NIFTY50-INDEX", "5", week_ago, today)
if last:
    ts = datetime.datetime.fromtimestamp(last[0], tz=IST)
    print(f"Last 5-min candle NIFTY: {ts:%H:%M} O={last[1]} H={last[2]} L={last[3]} C={last[4]} V={last[5]}")
