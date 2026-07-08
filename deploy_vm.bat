@echo off
REM ============================================================================
REM  deploy_vm.bat  --  push local commits, redeploy them on the VM, ENSURE the
REM  capture marker, and VERIFY the WebSocket actually came back.
REM
REM  Run AFTER market close (>= 15:35). A mid-session run restarts the capturer
REM  (brief capture gap). Build-first (`up -d --build`) means a build FAILURE leaves
REM  the OLD container running — no downtime, VM just stays on old code.
REM
REM  Flow:  git push origin <branch>   (VM pulls from origin — stale without this)
REM      -> VM: fetch + checkout + ff-only pull
REM      -> touch data/intraday/.capture_host   (WITHOUT it the VM boots as a VIEWER
REM         and SILENTLY STOPS capturing — the documented fail-safe trap)
REM      -> docker compose up -d --build
REM      -> poll fresh logs until the WS auths / ticks (not just "container up").
REM ============================================================================
setlocal
set BRANCH=refactor/layered-architecture
set KEY=%USERPROFILE%\Downloads\tradebot-key.pem
set VM=ubuntu@13.233.88.148
set LOG=%~dp0logs\vm_deploy.log

echo ================================================================ >> "%LOG%"
echo === VM DEPLOY  %date% %time% === >> "%LOG%"

echo [1/4] pushing %BRANCH% -> origin (so the VM can pull it)...
git push origin %BRANCH%
if errorlevel 1 (
  echo   PUSH FAILED — VM would deploy STALE code. Aborting.
  echo PUSH FAILED >> "%LOG%"
  pause & exit /b 1
)

echo [2/4] VM: pull + ensure capture marker + rebuild (build-first = no downtime on fail)...
ssh -i "%KEY%" -o ConnectTimeout=25 -o StrictHostKeyChecking=accept-new %VM% "cd ~/tradebot && git stash push -m ci-deploy -- docker-compose.yml 2>/dev/null; git fetch origin && git checkout %BRANCH% && git pull --ff-only origin %BRANCH% && { git stash pop 2>/dev/null || true; } && mkdir -p data/intraday && touch data/intraday/.capture_host && { docker compose up -d --build 2>&1 || docker-compose up -d --build 2>&1; } && echo DEPLOYED_HEAD=$(git rev-parse --short HEAD)" >> "%LOG%" 2>&1
if errorlevel 1 (
  echo   VM DEPLOY STEP FAILED — see %LOG%. OLD container should still be running.
  pause & exit /b 1
)

echo [3/4] verifying capture health (polls fresh logs up to ~40s)...
ssh -i "%KEY%" -o ConnectTimeout=25 %VM% "cd ~/tradebot && for i in $(seq 1 8); do sleep 5; if docker compose logs --since 90s tradebot 2>&1 | grep -q 'All 4 indices live'; then echo VERIFY_OK_TICKS; break; fi; done; echo '----- fresh capturer log -----'; docker compose logs --since 90s tradebot 2>&1 | grep -iE 'CAPTURER|VIEWER|Token  |\[WS\]|indices live' | tail -14"

echo.
echo [4/4] deploy finished %time%.  Read the lines above:
echo    OK  (market OPEN)   : 'VERIFY_OK_TICKS'  or  '[WS] All 4 indices live'
echo    OK  (market CLOSED) : '[CAPTURER]' + 'Token OK' + '[WS] Authentication done'  (no ticks till open)
echo    BAD                : any 'VIEWER' line, or no '[CAPTURER]' / no WS auth = CAPTURE DOWN
echo    Rollback if BAD    : ssh %VM% "cd ~/tradebot && git reset --hard HEAD~1 && docker compose up -d --build"
echo    Full log: %LOG%
endlocal
