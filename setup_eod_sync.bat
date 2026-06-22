@echo off
REM setup_eod_sync.bat - one-time: register the post-close VM->local archival sync
REM as a Windows Scheduled Task (15:45 Mon-Fri, runs-if-missed). Run once.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_eod_sync.ps1"
echo.
pause
