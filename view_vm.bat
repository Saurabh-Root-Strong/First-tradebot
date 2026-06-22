@echo off
REM view_vm.bat - view the VM's full live session on this laptop, read-only.
REM Stops the local capturer (so it can't clobber synced files), pulls the VM's
REM mirrors, keeps them fresh in the background, and runs the dashboard in
REM VIEWER mode (no capture). Use this instead of supervise.py when you just want
REM to WATCH the VM's data locally.
cd /d "%~dp0"
set PY=.venv\Scripts\python.exe
if not exist "%PY%" set PY=python

echo [1/4] Stopping any local capture holding port 8050...
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 8050 -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }"

echo [2/4] Pulling today's mirrors from the VM...
"%PY%" sync_from_vm.py
if errorlevel 1 echo     (sync issue - check VM/key; viewer shows whatever is local)

echo [3/4] Starting background sync (every 60s)...
start "tradebot-sync" /min "%PY%" sync_from_vm.py --watch 60

echo [4/4] Launching VIEWER dashboard (read-only) at http://127.0.0.1:8050 ...
set DASH_VIEWER=1
"%PY%" dashboard.py
