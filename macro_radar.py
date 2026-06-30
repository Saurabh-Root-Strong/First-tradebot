"""
macro_radar.py — LIVE global macro risk board for the Indian market.

Tracks the price-based factors that move India (crude, USD/INR, gold, silver, copper,
DXY, US 10Y yield, S&P/Nasdaq futures, India VIX), flags SUDDEN moves (z-score vs the
factor's own recent daily vol), maps each to its India-equity impact sign, and prints a
net risk-on/off tilt. Price-based = reliable, no headline-scraping hallucination.

HONEST USE: liquid macro is priced in seconds and the predictable next-day gap is
pre-priced by GIFT Nifty (see macro_overnight.py). This is a RISK / CONTEXT radar —
know WHY the tape is moving, flatten/avoid on a shock, size the range band — NOT a
front-running edge. Headlines (the WHY behind a spike) belong in news_events.py feeds.

Data: yfinance (Fyers does not serve globals).

    .venv\\Scripts\\python.exe macro_radar.py
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import yfinance as yf

# (ticker, label, India-equity impact sign of a RISE, weight)
FACTORS = [
    ("BZ=F",       "Brent crude",   -1, 1.0),   # dearer oil = bad for importer India
    ("INR=X",      "USD/INR",       -1, 1.0),   # weaker rupee = FII outflow risk
    ("DX-Y.NYB",   "DXY dollar",    -1, 1.0),   # strong dollar = EM outflow
    ("^TNX",       "US 10Y yield",  -1, 0.8),   # rising yields = risk-off
    ("ES=F",       "S&P 500 fut",   +1, 1.0),   # risk-on
    ("NQ=F",       "Nasdaq fut",    +1, 0.8),   # risk-on
    ("HG=F",       "Copper",        +1, 0.6),   # global growth = risk-on
    ("GC=F",       "Gold",          -1, 0.5),   # safe-haven bid = mild risk-off
    ("SI=F",       "Silver",        -1, 0.3),
    ("^INDIAVIX",  "India VIX",     -1, 0.8),   # fear gauge
]
SPIKE_Z = 1.5


def _fetch(tkr):
    try:
        d = yf.download(tkr, period="40d", interval="1d", progress=False, auto_adjust=True)
        if d is None or len(d) < 10:
            return None
        c = d["Close"]
        if isinstance(c, pd.DataFrame):
            c = c.iloc[:, 0]
        return c.dropna()
    except Exception:
        return None


def compute():
    """Fetch all factors -> (rows, tilt, lean). rows: list of dicts sorted by |z|.

    Importable by the dashboard poller (no subprocess). Network-bound — call from a
    background thread, never the request path.
    """
    out = []
    tilt = 0.0
    for tkr, lab, sgn, w in FACTORS:
        c = _fetch(tkr)
        if c is None or len(c) < 10:
            out.append({"factor": lab, "level": None, "chg": None, "z": None,
                        "impact": "n/a", "spike": False})
            continue
        ret = c.pct_change() * 100
        chg = float(ret.iloc[-1])
        vol = float(ret.iloc[-21:-1].std())
        z = chg / vol if vol > 0 else 0.0
        impact = "UP" if (sgn * np.sign(chg)) > 0 else "DOWN" if (sgn * np.sign(chg)) < 0 else "flat"
        tilt += sgn * z * w
        out.append({"factor": lab, "level": float(c.iloc[-1]), "chg": chg, "z": z,
                    "impact": impact, "spike": abs(z) >= SPIKE_Z})
    out.sort(key=lambda r: (abs(r["z"]) if r["z"] is not None else -1), reverse=True)
    lean = ("RISK-ON / India tailwind" if tilt > 1 else
            "RISK-OFF / India headwind" if tilt < -1 else "MIXED / neutral")
    return out, tilt, lean


def main():
    print("=" * 74)
    print("  LIVE MACRO RADAR -> INDIA  (risk/context, not a trade signal)")
    print("=" * 74)
    data, tilt, lean = compute()
    print(f"  {'factor':14s} {'level':>10} {'chg%':>7} {'z':>6}  {'India':>6}  flag")
    print("  " + "-" * 64)
    for r in data:
        if r["level"] is None:
            print(f"  {r['factor']:14s} {'n/a':>10}  (fetch failed)"); continue
        flag = "  <-- SPIKE" if r["spike"] else ""
        print(f"  {r['factor']:14s} {r['level']:>10.2f} {r['chg']:>+6.2f}% {r['z']:>+5.1f}  "
              f"{r['impact']:>6}  {flag}")
    print("  " + "-" * 64)
    print(f"  NET MACRO TILT for India: {tilt:+.1f}  ->  {lean}")
    print("\n  NOTE: this is CONTEXT/RISK — public macro is priced in seconds and the next-day")
    print("  gap is pre-priced by GIFT Nifty. Use to know WHY / flatten on a shock / size the")
    print("  band — not to front-run. Pair with news_events.py for the headline behind a SPIKE.")


if __name__ == "__main__":
    main()
