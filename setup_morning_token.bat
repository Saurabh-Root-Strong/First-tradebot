@echo off
REM setup_morning_token.bat - one-time: register the daily Fyers token refresh as
REM a Windows Scheduled Task (09:00 Mon-Fri, runs-if-missed). Run once.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_morning_token.ps1"
echo.
pause
