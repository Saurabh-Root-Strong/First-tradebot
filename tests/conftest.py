"""Put the repo root on sys.path so tests can import the (still-flat) prod modules.

Removed once prod code lives under the installed `tradebot/` package.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The legacy live-API smoke scripts hit Fyers with a real token AT IMPORT TIME — they
# are manual, market-hours checks, not offline units. Skip COLLECTING them when there is
# no token, so `pytest` is a clean offline regression run (the test_unit_*.py suite). They
# still collect when a token is present, to run intentionally during market hours.
_LIVE_API = ["test_futures.py", "test_oc.py", "test_signals.py", "test_signals2.py"]
if not (ROOT / "access_token.txt").exists():
    collect_ignore = _LIVE_API
