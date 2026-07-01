@echo off
REM ── BTST close-strength EOD routine — run at ~15:20 IST ─────────────────────────
REM  1) refresh today's index daily bar (so clr = near-final close)
REM  2) emit tonight's strong-close BTST-LONG candidates (PAPER — nothing auto-executes)
REM  Run again if you like; it UPDATES today's entry to the latest close.
cd /d "d:\Python Projects\Tradebot"

echo [1/2] Refreshing today's index daily bars...
.venv\Scripts\python.exe download_historical.py --indices-only --force --timeframes daily

echo.
echo [2/2] BTST close-strength candidates (PAPER, exit next ~09:30 on FUTURES):
.venv\Scripts\python.exe btst_signal.py

echo.
echo Next morning ~09:35, run: btst_morning.bat  (fills exits + scorecard)
pause
