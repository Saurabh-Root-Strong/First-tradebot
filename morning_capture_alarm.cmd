@echo off
REM Morning capture watchdog — alarms (toast + box) if Fyers token/capture is DOWN during
REM market hours. Read-only, laptop-only. Daily 09:20 IST + at-logon.
cd /d "D:\Python Projects\Tradebot"
".venv\Scripts\python.exe" morning_capture_alarm.py >> logs\morning_capture_alarm.log 2>&1
