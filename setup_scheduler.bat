@echo off
REM ─────────────────────────────────────────────────────────────────────────────
REM  setup_scheduler.bat  --  Register Tradebot Nightly Sync with Windows Task Scheduler
REM
REM  Run this script ONCE to set up the scheduled task (single trigger fallback).
REM  The task runs run_sync.bat every weekday (Mon-Fri) at 8:30 PM.
REM
REM  Daily_Cash_Market's NSE ingestion task runs at 7:30 PM (with a 11:30 PM
REM  retry).  Tradebot must run AFTER it -- 8:30 PM gives the 7:30 PM ingestion
REM  a 60-minute window.  (Do NOT schedule at 7:30 PM -- that races DCM.)
REM
REM  NOTE: the live task registered via PowerShell adds a second 11:55 PM trigger
REM  to recover days when DCM's 7:30 PM run failed and only the 11:30 PM retry
REM  succeeded.  schtasks below registers a single 8:30 PM trigger only.
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

set LAUNCHER=%SCRIPT_DIR%\run_sync.bat
set LOG_FILE=%SCRIPT_DIR%\data\nightly_sync.log

REM Verify the launcher exists before registering
if not exist "%LAUNCHER%" (
    echo ERROR: run_sync.bat not found at %LAUNCHER%
    pause
    exit /b 1
)

echo.
echo --- Tradebot Nightly Sync -- Task Scheduler Setup --------------------------
echo.
echo   Launcher : %LAUNCHER%
echo   Log      : %LOG_FILE%
echo   Schedule : Mon-Fri at 20:30 (8:30 PM)
echo.

REM Delete existing task if present (idempotent re-registration)
schtasks /Delete /TN "%TASK_NAME%" /F >nul 2>&1

REM Register the task
REM  /SC WEEKLY  /D  = weekdays only
REM  /ST 20:30   = 8:30 PM (after DCM's 7:30 PM ingestion)
REM  /F          = force overwrite
REM  (No /RL HIGHEST: the sync needs no elevation, and HIGHEST requires running
REM   this script as Administrator. run_sync.bat handles the log redirect.)
schtasks /Create ^
  /TN "%TASK_NAME%" ^
  /TR "cmd /c \"%LAUNCHER%\"" ^
  /SC WEEKLY ^
  /D MON,TUE,WED,THU,FRI ^
  /ST 20:30 ^
  /F

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: Task registration failed.
    echo Try running this script as Administrator.
    pause
    exit /b 1
)

echo.
echo --- Task registered successfully ------------------------------------------
echo.
echo   Task name : %TASK_NAME%
echo   Runs at   : 8:30 PM  Mon-Fri
echo.
echo   To verify : schtasks /Query /TN "%TASK_NAME%" /FO LIST
echo   To test   : schtasks /Run /TN "%TASK_NAME%"
echo   To check  : python nightly_sync.py --status
echo.

REM Offer to run the first sync right now
set /p RUNNOW="Run first sync now? (y/n): "
if /i "%RUNNOW%"=="y" (
    echo.
    echo Running first sync...
    call "%LAUNCHER%"
    echo.
    set PYTHONUTF8=1
    python "%SCRIPT_DIR%\nightly_sync.py" --status
)

echo.
pause
