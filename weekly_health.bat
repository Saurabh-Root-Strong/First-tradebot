@echo off
REM weekly_health.bat - the FULL weekly health check (Task Scheduler, Mondays, background).
REM
REM   1. NSE constants drift  (verify_nse_calendar.py)  -- fast (~1s)
REM   2. Signal structural bias (audit_signals.py --check) -- SLOW (minutes; harvests every
REM      captured day). This is the check that would have caught the 95%-CALL flow bias.
REM
REM Both append to logs\verify_nse.log. Either failing drops logs\verify_nse.DRIFT, which
REM dev.bat surfaces in red on every launch until fixed.
REM
REM NOTE: dev.bat's LAUNCH hook deliberately runs only verify_nse.bat (the fast NSE check) so
REM starting the tradebot never blocks for minutes. The slow signal audit runs here, in the
REM background, where nobody is waiting.
cd /d "d:\Python Projects\Tradebot"
if not exist logs mkdir logs

REM 1) NSE constants (this also freshens the log mtime + clears/sets the marker)
call "verify_nse.bat"

REM 2) Signal structural-bias invariants
echo --- signal health (structural bias invariants) --- >> logs\verify_nse.log
".venv\Scripts\python.exe" audit_signals.py --check >> logs\verify_nse.log 2>&1
if errorlevel 1 (
  echo SIGNAL DRIFT: component bias outside band %DATE% %TIME% - see logs\verify_nse.log >> logs\verify_nse.DRIFT
  echo [DRIFT] signal bias - see logs\verify_nse.DRIFT >> logs\verify_nse.log
) else (
  echo [OK] signal structural invariants hold >> logs\verify_nse.log
)
