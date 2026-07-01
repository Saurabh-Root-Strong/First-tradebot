@echo off
REM ── BTST morning routine — run at ~09:35 IST (after the 09:30 exit) ──────────────
REM  1) refresh index daily/5min (so exit prices are available)
REM  2) reconcile: fill last night's positions at the real ~09:30 exit
REM  3) scorecard: paper P&L vs the +10-13 bps backtest expectation
cd /d "d:\Python Projects\Tradebot"

echo [1/3] Refreshing index bars...
.venv\Scripts\python.exe download_historical.py --indices-only --force --timeframes 5min,daily

echo.
echo [2/3] Reconciling last night's BTST exits...
.venv\Scripts\python.exe btst_signal.py --reconcile

echo.
echo [3/3] Paper scorecard:
.venv\Scripts\python.exe btst_signal.py --scorecard

echo.
pause
