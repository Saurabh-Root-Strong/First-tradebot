@echo off
REM Morning capture watchdog - alarms (toast + box) if Fyers token/capture is DOWN during
REM market hours. Read-only, laptop-only. Daily 09:20 IST + at-logon.
cd /d "D:\Python Projects\Tradebot"
REM Console output goes to its OWN file: redirecting into morning_capture_alarm.log (which the
REM script appends to itself) locked it, and every run from 24 Jul to 4 Nov 2026 crashed.
".venv\Scripts\python.exe" morning_capture_alarm.py >> logs\morning_capture_alarm.console.log 2>&1
