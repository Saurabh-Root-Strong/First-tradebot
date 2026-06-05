@echo off
:: replay.bat — always runs session_replay.py with the project venv Python.
:: Usage: replay [all session_replay.py args]
::   replay --stats
::   replay --ticks --sym NIFTY --from 09:15 --to 10:00
::   replay --tick-stats --sym BANK
::   replay --candles --tf 5min
::   replay --report
::   replay --date 2026-06-03 --report
"%~dp0.venv\Scripts\python.exe" "%~dp0session_replay.py" %*
