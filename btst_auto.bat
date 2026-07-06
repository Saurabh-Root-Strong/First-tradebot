@echo off
REM ── BTST unattended runner — HEADLESS + IDEMPOTENT (no pause, safe to run anytime) ──
REM  Reconcile fills any position whose exit day has arrived (skips already-closed);
REM  emit logs/updates tonight's strong-close candidates; scorecard prints the record.
REM  Reconcile self-heals from the TICK MIRROR when the historical archive lags, so this
REM  needs NO same-day download and NO Fyers token. Schedule it twice a day:
REM     ~09:35 IST  → fills last night's exits + scorecard
REM     ~15:25 IST  → logs tonight's candidates (re-run updates to the latest close)
REM
REM  Register with Windows Task Scheduler (run once, from this folder):
REM     schtasks /Create /TN "BTST morning" /TR "\"%CD%\btst_auto.bat\"" /SC DAILY /ST 09:35
REM     schtasks /Create /TN "BTST eod"     /TR "\"%CD%\btst_auto.bat\"" /SC DAILY /ST 15:25
cd /d "%~dp0"

REM Best-effort archive refresh (ignored if no token — reconcile falls back to the mirror).
.venv\Scripts\python.exe download_historical.py --indices-only --force --timeframes 5min,daily 1>nul 2>nul

.venv\Scripts\python.exe btst_signal.py --reconcile
.venv\Scripts\python.exe btst_signal.py
.venv\Scripts\python.exe btst_signal.py --scorecard
