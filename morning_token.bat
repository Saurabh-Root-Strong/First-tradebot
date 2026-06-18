@echo off
REM ============================================================================
REM  morning_token.bat  --  one-click daily Fyers token refresh for the cloud VM
REM
REM  Fyers tokens die at midnight IST, and Fyers' unattended TOTP API login is
REM  anti-bot blocked, so each trading morning run this once on your laptop:
REM    1. Opens a browser, you log into Fyers (FY_ID / PIN / TOTP)
REM    2. Saves the fresh access_token.txt locally
REM    3. Uploads it to the VM over SSH (key-only)
REM    4. Restarts the VM capture so it picks up the new token
REM  Then close the laptop -- the VM captures + serves all session, anywhere.
REM ============================================================================

setlocal
set REPO=d:\Python Projects\Tradebot
set PEM=%USERPROFILE%\Downloads\tradebot-key.pem
set VM=ubuntu@13.233.88.148

cd /d "%REPO%"

REM --- make sure Windows ssh accepts the key (perms must be private) ----------
icacls "%PEM%" /inheritance:r /grant:r "%USERNAME%:R" >nul 2>&1

echo.
echo [1/4] Opening Fyers login in your browser -- log in (FY_ID, PIN, TOTP)...
".venv\Scripts\python.exe" fyers_auth.py
if errorlevel 1 (
  echo.
  echo  LOGIN FAILED -- token not refreshed. Fix and re-run.
  pause & exit /b 1
)

echo.
echo [2/4] Uploading token to the VM...
scp -i "%PEM%" -o StrictHostKeyChecking=accept-new access_token.txt %VM%:/home/ubuntu/tradebot/access_token.txt
if errorlevel 1 ( echo  UPLOAD FAILED & pause & exit /b 1 )

echo.
echo [3/4] Restarting capture on the VM...
ssh -i "%PEM%" %VM% "cd ~/tradebot && docker compose restart tradebot"
if errorlevel 1 ( echo  RESTART FAILED & pause & exit /b 1 )

echo.
echo [4/4] Done. The VM is now capturing with a fresh token.
echo       Open  https://13.233.88.148.sslip.io   (user: admin)
echo       You can close the laptop now.
echo.
start "" https://13.233.88.148.sslip.io
pause
