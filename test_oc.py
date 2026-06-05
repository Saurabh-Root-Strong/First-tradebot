import requests
from pathlib import Path

APP_ID = "WVDZUTO6HL-100"
token  = Path("access_token.txt").read_text(encoding="utf-8").strip()
auth   = f"{APP_ID}:{token}"

SYMS = [
    "NSE:NIFTY50-INDEX",
    "NSE:NIFTYBANK-INDEX",
    "NSE:FINNIFTY-INDEX",
    "NSE:MIDCPNIFTY-INDEX",
]

for sym in SYMS:
    resp = requests.get(
        "https://api-t1.fyers.in/data/options-chain-v3",
        headers={"Authorization": auth, "Content-Type": "application/json", "version": "3"},
        params={"symbol": sym, "strikecount": "3", "timestamp": "", "greeks": "1"},
        timeout=15,
    )
    try:
        d       = resp.json()
        strikes = [x for x in d.get("data", {}).get("optionsChain", []) if x.get("strike_price", 0) > 0]
        print(f"{sym}: s={d.get('s')}  strikes={len(strikes)}  callOi={d.get('data',{}).get('callOi')}")
    except Exception as e:
        print(f"{sym}: PARSE ERROR  status={resp.status_code}  raw={resp.text[:120]}")
