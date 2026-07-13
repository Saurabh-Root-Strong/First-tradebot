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

REM weekly NSE-constants drift check — runs at most once / 7 days on launch (keyed off the
REM log mtime, which the Monday scheduled task also updates, so the two never double-run).
REM Always re-warns if a prior run flagged drift, so it stays loud until the constant is fixed.
powershell -NoProfile -Command "$l='logs\verify_nse.log'; if(-not (Test-Path $l) -or (Get-Item $l).LastWriteTime -lt (Get-Date).AddDays(-6)){ Write-Host '[NSE] weekly drift check...'; & '.\verify_nse.bat' }; $w= if(Test-Path $l){(Get-Item $l).LastWriteTime.ToString('yyyy-MM-dd HH:mm')}else{'never'}; $bad=$false; if(Test-Path 'logs\verify_nse.DRIFT'){ $bad=$true; Write-Host ''; Write-Host ('  *** NSE CONSTANT DRIFT DETECTED ('+$w+') - see logs\verify_nse.log ***') -ForegroundColor Red }; if(Test-Path 'logs\signal_health.DRIFT'){ $bad=$true; Write-Host ''; Write-Host '  *** SIGNAL BIAS DRIFT (weekly health check) - see logs\verify_nse.log ***' -ForegroundColor Red }; if(-not $bad){ Write-Host ('[health] NSE constants + signal bias OK (last check '+$w+')') -ForegroundColor Green } else { Write-Host '' }"

REM ── VM CAPTURE HEALTH + SELF-HEAL ────────────────────────────────────────────────
REM The VM is the sole capturer, but NOTHING used to verify it was actually capturing. On
REM 2026-07-13 it sat on an EXPIRED token all weekend (morning-token task skipped: laptop
REM asleep + no catch-up) and the dashboard's auto-auth refreshed only the LOCAL token --
REM the VM would have captured ZERO for the whole session, silently. Running the tradebot is
REM the one thing you always do, so it is the right place to check + heal: --fix pushes the
REM local token and restarts capture, but ONLY if the local token is valid (it will never
REM replace a good VM token with a dead one). No-op when everything is healthy.
echo [1/5] checking VM capture health (heals a dead token automatically)...
"%PY%" check_vm_capture.py --fix
if errorlevel 2 (
  powershell -NoProfile -Command "Write-Host ''; Write-Host '  *** VM UNREACHABLE or LOCAL TOKEN DEAD - the VM may be capturing NOTHING. ***' -ForegroundColor Red; Write-Host '  *** Run morning_token.bat (log in), then re-run dev.bat.              ***' -ForegroundColor Red; Write-Host ''"
) else if errorlevel 1 (
  powershell -NoProfile -Command "Write-Host ''; Write-Host '  *** VM CAPTURE UNHEALTHY and the auto-fix did not take - check the VM. ***' -ForegroundColor Red; Write-Host ''"
)

echo [2/5] pulling today's VM mirrors (pre-open may be empty — that's fine)...
"%PY%" sync_from_vm.py

echo [3/5] freeing port 8050 so the viewer binds cleanly...
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 8050 -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }"

REM Start the 60s mirror watcher DETACHED via powershell, not `start` -- `start` inherits the
REM console and did NOT survive when dev.bat itself was launched non-interactively (that is
REM why the viewer silently showed Friday's data while Monday's session ran).
echo [4/5] starting background sync watcher (every 60s)...
powershell -NoProfile -Command "Start-Process -FilePath '.\.venv\Scripts\python.exe' -ArgumentList 'sync_from_vm.py','--watch','60' -WorkingDirectory '%CD%' -WindowStyle Minimized"

echo [5/5] launching LOCAL VIEWER (read-only) on http://127.0.0.1:8050 ...
set DASH_VIEWER=1
set TRADEBOT_NO_BROWSER=
"%PY%" dashboard.py
endlocal
