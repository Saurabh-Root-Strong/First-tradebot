@echo off
REM ============================================================================
REM  push_news.bat  --  refresh stock news locally + push the mirrors to the VM
REM
REM  The cloud VM (EC2) cannot reach NSE (datacenter IPs are blocked), so the
REM  stock-filings feed is dead there. This laptop CAN reach NSE, so:
REM    1. Backfill the last few days of NSE corporate announcements (scored).
REM    2. Upload every per-day news mirror to the VM's live data dir.
REM  The VM dashboard globs that dir each tick, so it picks them up with no
REM  restart. RBI/macro still fetches fine on the VM itself.
REM
REM  Run it whenever you want the cloud news fresh (stock filings land all day),
REM  or schedule it via Task Scheduler (e.g. every 30 min, 9:00-16:00, Mon-Fri).
REM ============================================================================

setlocal enabledelayedexpansion
set REPO=d:\Python Projects\Tradebot
set PEM=%USERPROFILE%\Downloads\tradebot-key.pem
set VM=ubuntu@13.233.88.148
set RDIR=/home/ubuntu/tradebot/data/intraday/live

cd /d "%REPO%"
icacls "%PEM%" /inheritance:r /grant:r "%USERNAME%:R" >nul 2>&1

echo [1/3] Refreshing NSE stock news locally (last 3 days)...
".venv\Scripts\python.exe" news_events.py --backfill 3
if errorlevel 1 ( echo  BACKFILL FAILED & pause & exit /b 1 )

echo.
echo [2/3] Ensuring remote dir exists + writable...
REM Container restart resets %RDIR% to root ownership, which blocks the scp
REM below with "Permission denied". Reclaim it for the ssh (ubuntu) user first;
REM the container runs as root so it can still write regardless of owner.
ssh -i "%PEM%" -o StrictHostKeyChecking=accept-new %VM% "mkdir -p %RDIR% && sudo chown -R ubuntu:ubuntu %RDIR%"

echo.
echo [3/3] Uploading news mirrors to the VM...
set N=0
for %%f in ("data\intraday\live\*_news_events.parquet") do (
  scp -i "%PEM%" -o StrictHostKeyChecking=accept-new "%%f" %VM%:%RDIR%/ && set /a N+=1
)
echo.
echo Done. Pushed !N! news mirror file(s) to the VM. The cloud dashboard will show
echo them within ~60s (it re-reads the live dir each tick).
endlocal
