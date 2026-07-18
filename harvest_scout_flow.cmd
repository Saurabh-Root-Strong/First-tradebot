@echo off
REM EOD scout flow-context harvest. Self-healing (catches up all missed days).
REM Self-guarded (refuses today's harvest before 15:35 IST). Zero live impact.
cd /d "D:\Python Projects\Tradebot"
".venv\Scripts\python.exe" harvest_scout_flow.py >> logs\harvest_scout_flow.log 2>&1
