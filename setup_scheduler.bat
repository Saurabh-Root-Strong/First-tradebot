@echo off
REM ─────────────────────────────────────────────────────────────────────────────
REM  setup_scheduler.bat  --  Register Tradebot Nightly Sync with Windows Task Scheduler
REM
REM  Run this script ONCE (as Administrator) to set up the scheduled task.
REM  The task will run nightly_sync.py every weekday (Mon-Fri) at 7:30 PM.
REM
REM  Daily_Cash_Market updates its DuckDB at ~6:30 PM.  7:30 PM gives it a
REM  comfortable 60-minute window to complete before Tradebot pulls a snapshot.
REM
REM  To check task status:   schtasks /Query /TN "TradebotNightlySync" /FO LIST
REM  To run manually:        schtasks /Run   /TN "TradebotNightlySync"
REM  To remove task:         schtasks /Delete /TN "TradebotNightlySync" /F
REM  To check sync status:   python nightly_sync.py --status
REM ─────────────────────────────────────────────────────────────────────────────

setlocal

set TASK_NAME=TradebotNightlySync
set SCRIPT_DIR=%~dp0
set SCRIPT_DIR=%SCRIPT_DIR:~0,-1%

REM Use the venv python if available, else fall back to system python
if exist "%SCRIPT_DIR%\.venv\Scripts\python.exe" (
    set PYTHON=%SCRIPT_DIR%\.venv\Scripts\python.exe
) else (
    set PYTHON=python
)

set SCRIPT=%SCRIPT_DIR%\nightly_sync.py
set LOG_FILE=%SCRIPT_DIR%\data\nightly_sync.log

REM Verify the script exists before registering
if not exist "%SCRIPT%" (
    echo ERROR: nightly_sync.py not found at %SCRIPT%
    pause
    exit /b 1
)

echo.
echo ─── Tradebot Nightly Sync — Task Scheduler Setup ───────────────────────────
echo.
echo   Python : %PYTHON%
echo   Script : %SCRIPT%
echo   Log    : %LOG_FILE%
echo   Schedule: Mon-Fri at 19:30 (7:30 PM)
echo.

REM Delete existing task if present (idempotent re-registration)
schtasks /Delete /TN "%TASK_NAME%" /F >nul 2>&1

REM Register the task
REM  /SC WEEKLY  /D  = weekdays only
REM  /ST 19:30   = 7:30 PM
REM  /RL HIGHEST = run with highest available privileges (avoids UAC blocking)
REM  /F          = force overwrite
schtasks /Create ^
  /TN "%TASK_NAME%" ^
  /TR "\"%PYTHON%\" \"%SCRIPT%\" >> \"%LOG_FILE%\" 2>&1" ^
  /SC WEEKLY ^
  /D MON,TUE,WED,THU,FRI ^
  /ST 19:30 ^
  /RL HIGHEST ^
  /F

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: Task registration failed.
    echo Try running this script as Administrator.
    pause
    exit /b 1
)

echo.
echo ─── Task registered successfully ───────────────────────────────────────────
echo.
echo   Task name : %TASK_NAME%
echo   Runs at   : 7:30 PM  Mon-Fri
echo.
echo   To verify : schtasks /Query /TN "%TASK_NAME%" /FO LIST
echo   To test   : python nightly_sync.py --force
echo   To check  : python nightly_sync.py --status
echo.

REM Offer to run the first sync right now
set /p RUNNOW="Run first sync now? (y/n): "
if /i "%RUNNOW%"=="y" (
    echo.
    echo Running first sync...
    "%PYTHON%" "%SCRIPT%" --force
    echo.
    "%PYTHON%" "%SCRIPT%" --status
)

echo.
pause
