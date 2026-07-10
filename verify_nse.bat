@echo off
REM verify_nse.bat - weekly NSE-constants drift check (Task Scheduler: Mondays).
REM Cross-checks hardcoded NSE_HOLIDAYS / weekly-expiry / LOT_SIZES vs DCM ground truth.
REM Appends to logs\verify_nse.log; drops logs\verify_nse.DRIFT on drift (exit 1), clears it when clean.
cd /d "d:\Python Projects\Tradebot"
if not exist logs mkdir logs
echo ==================== %DATE% %TIME% ==================== >> logs\verify_nse.log
REM exit 0 = consistent | 1 = drift | 2 = COULD NOT VERIFY (DCM ground-truth DB unreachable).
REM 2 must not be mistaken for a pass. Test errorlevel 2 BEFORE 1 (`if errorlevel N` = ">= N").
".venv\Scripts\python.exe" verify_nse_calendar.py >> logs\verify_nse.log 2>&1
if errorlevel 2 (
  echo NSE CHECK COULD NOT VERIFY: DCM ground-truth DB unreachable %DATE% %TIME% > logs\verify_nse.DRIFT
  echo [DRIFT] nse check could not verify - see logs\verify_nse.log >> logs\verify_nse.log
) else if errorlevel 1 (
  echo NSE CONSTANT DRIFT detected %DATE% %TIME% - see logs\verify_nse.log > logs\verify_nse.DRIFT
  echo [DRIFT] see logs\verify_nse.DRIFT >> logs\verify_nse.log
) else (
  if exist logs\verify_nse.DRIFT del logs\verify_nse.DRIFT
  echo [OK] all consistent >> logs\verify_nse.log
)
