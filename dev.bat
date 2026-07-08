@echo off
REM ============================================================================
REM  dev.bat  --  LOCAL development / staging against the VM's REAL session data.
REM
REM  The VM is the sole authoritative capturer (do NOT touch it). This laptop is a
REM  VIEWER. This script:
REM    1. pulls the VM's live parquet mirrors down (real full-session data)
REM    2. starts a background 60s sync watcher (keeps the local view current)
REM    3. launches a READ-ONLY viewer dashboard on http://127.0.0.1:8050
REM
REM  It can NEVER capture: DASH_VIEWER=1 forces viewer, and no .capture_host marker
REM  is created. So you build + verify changes on prod-realistic data with ZERO risk
REM  to the VM's live capture. Nothing here is ever synced UP to the VM.
REM
REM  Loop:  see issue on the VM URL -> reproduce here on synced data -> fix + verify
REM         -> commit -> deploy_vm.bat (after 15:35).
REM ============================================================================
setlocal
cd /d "%~dp0"
set PY=.venv\Scripts\python.exe

echo [1/4] pulling today's VM mirrors (pre-open may be empty — that's fine)...
"%PY%" sync_from_vm.py

echo [2/4] freeing port 8050 so the viewer binds cleanly...
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 8050 -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }"

echo [3/4] starting background sync watcher (every 60s; close its window to stop)...
start "tradebot-sync" "%PY%" sync_from_vm.py --watch 60

echo [4/4] launching LOCAL VIEWER (read-only) on http://127.0.0.1:8050 ...
set DASH_VIEWER=1
set TRADEBOT_NO_BROWSER=
"%PY%" dashboard.py
endlocal
