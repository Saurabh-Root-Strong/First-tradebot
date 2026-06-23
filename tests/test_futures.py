"""Inspect all fields in futures quote response."""
import requests, datetime
from pathlib import Path

APP_ID = "WVDZUTO6HL-100"
token  = Path("access_token.txt").read_text(encoding="utf-8").strip()
auth   = f"{APP_ID}:{token}"

resp = requests.get(
    "https://api-t1.fyers.in/data/quotes",
    headers={"Authorization": auth, "Content-Type": "application/json", "version": "3"},
    params={"symbols": "NSE:NIFTY26JUNFUT,NSE:NIFTY26JULFUT,NSE:NIFTY26AUGFUT"},
    timeout=15,
)
data = resp.json()

for item in data.get("d", [])[:1]:  # just first item
    print("Symbol:", item.get("n"))
    print("All fields in v:")
    for k, v in sorted(item.get("v", {}).items()):
        print(f"  {k:<30} = {v}")
