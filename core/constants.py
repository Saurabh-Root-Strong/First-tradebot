"""core.constants — the single source of truth for shared constants + tiny helpers.

Previously IST was redefined in 19 files, INDEX_SYMBOLS in 8, LABELS in 7, etc.
Everything anchors to PROJECT_ROOT so it is correct regardless of the working dir.
"""
from __future__ import annotations

import datetime
from pathlib import Path

IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LIVE_DIR = DATA_DIR / "intraday" / "live"        # lock-free parquet mirrors (today)
HIST_DIR = DATA_DIR / "historical"               # downloaded OHLCV per timeframe

# The four live-captured indices.
NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY = (
    "NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX", "NSE:FINNIFTY-INDEX", "NSE:MIDCPNIFTY-INDEX")
INDEX_SYMBOLS = [NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY]

LABELS = {NIFTY: "NIFTY 50", BANKNIFTY: "BANK NIFTY",
          FINNIFTY: "FIN NIFTY", MIDCPNIFTY: "MIDCAP NIFTY"}

# label / alias (case-insensitive, spaces stripped) -> index symbol
LABEL_TO_SYM = {v.replace(" ", ""): k for k, v in LABELS.items()}   # NIFTY50, BANKNIFTY, ...
LABEL_TO_SYM.update({"NIFTY": NIFTY, "BANKNIFTY": BANKNIFTY, "FINNIFTY": FINNIFTY,
                     "MIDCPNIFTY": MIDCPNIFTY, "MIDCAPNIFTY": MIDCPNIFTY})

# option strike step per index
STRIKE_STEP = {NIFTY: 50, BANKNIFTY: 100, FINNIFTY: 50, MIDCPNIFTY: 25}

# index symbol <-> NSE F&O underlying name (oi-spurts / bhavcopy)
NSE_NAME = {NIFTY: "NIFTY", BANKNIFTY: "BANKNIFTY", FINNIFTY: "FINNIFTY", MIDCPNIFTY: "MIDCPNIFTY"}
SYM_BY_NSE = {v: k for k, v in NSE_NAME.items()}


def today_iso() -> str:
    """Current trading date (IST), ISO format — used in mirror filenames."""
    return datetime.datetime.now(tz=IST).date().isoformat()


def safe_sym(sym: str) -> str:
    """Symbol -> filesystem-safe stem (NSE:NIFTY50-INDEX -> NSE_NIFTY50_INDEX)."""
    return sym.replace(":", "_").replace("-", "_")


def resolve_sym(token: str) -> str:
    """Accept a full symbol or a label/alias and return the canonical index symbol."""
    return LABEL_TO_SYM.get(token.upper().replace(" ", ""), token)
