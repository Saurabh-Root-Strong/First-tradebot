@echo off
REM ---------------------------------------------------------------------------
REM  run_supervised.bat -- one-command run-and-leave-it launcher.
REM
REM  Starts supervise.py, which ensures a valid Fyers token (browser login if
REM  needed), launches the dashboard, and auto-restarts it on crash or a stalled
REM  WebSocket during market hours. Logs to logs\supervisor.log.
REM
REM  Just run this in the morning and walk away. Ctrl+C to stop.
REM  Keep this file ASCII-only -- cmd.exe mis-parses non-ASCII bytes.
REM ---------------------------------------------------------------------------
set PYTHONUTF8=1
cd /d "%~dp0"

if exist "%~dp0.venv\Scripts\python.exe" (
    set "PY=%~dp0.venv\Scripts\python.exe"
) else (
    set "PY=python"
)

"%PY%" "%~dp0supervise.py"
