@echo off
REM ---------------------------------------------------------------------------
REM  run_sync.bat -- Launcher for the Tradebot nightly snapshot sync.
REM
REM  Called by the Windows Task Scheduler task "TradebotNightlySync"
REM  (Mon-Fri, 20:30 and 23:55). Runs AFTER Daily_Cash_Market's NSE ingestion
REM  (19:30 primary, 23:30 retry) so it always pulls the freshest DuckDB into
REM  the local snapshot.
REM
REM  --force is used so the late (23:55) run refreshes even if the 20:30 run
REM  already marked today done -- picks up data DCM's 23:30 retry may have added.
REM  PYTHONUTF8=1 avoids cp1252 UnicodeEncodeError on the arrow/box-draw glyphs.
REM
REM  NOTE: keep this file ASCII-only -- cmd.exe mis-parses non-ASCII bytes.
REM ---------------------------------------------------------------------------
set PYTHONUTF8=1
cd /d "%~dp0"

if exist "%~dp0.venv\Scripts\python.exe" (
    set "PY=%~dp0.venv\Scripts\python.exe"
) else (
    set "PY=python"
)

"%PY%" "%~dp0nightly_sync.py" --force >> "%~dp0data\nightly_sync.log" 2>&1
