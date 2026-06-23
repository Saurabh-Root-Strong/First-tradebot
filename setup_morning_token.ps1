# setup_morning_token.ps1 - register the daily Fyers token refresh as a Windows
# Scheduled Task. Fires 09:00 Mon-Fri (before the 09:15 open). StartWhenAvailable
# means if the laptop was asleep at 09:00 it runs as soon as it next wakes.
#
# NOTE: this is NOT fully unattended. morning_token.bat opens a browser for the
# Fyers login (FY_ID / PIN / TOTP) - Fyers blocks headless TOTP. The task just
# pops it at the right time; you tap TOTP, then scp + restart + news-push run on
# their own. A visible console is launched so you can complete the login.
$ErrorActionPreference = "Stop"
$dir = $PSScriptRoot
$bat = Join-Path $dir "morning_token.bat"

# Launch via cmd /c start so a visible console window appears for the login.
$action  = New-ScheduledTaskAction -Execute "cmd.exe" `
    -Argument "/c start `"Tradebot Morning Token`" `"$bat`"" -WorkingDirectory $dir
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At "09:00"
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 30) `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries

Register-ScheduledTask -TaskName "Tradebot Morning Token" -Action $action -Trigger $trigger `
    -Settings $settings -Description "Daily Fyers token refresh + push to VM (interactive login)" -Force

Write-Host ""
Write-Host "Registered task 'Tradebot Morning Token':"
Write-Host "  - fires 09:00 Mon-Fri (before the 09:15 open)"
Write-Host "  - if the laptop was asleep then, fires as soon as it wakes"
Write-Host "  - opens a console + browser; you complete the TOTP login,"
Write-Host "    then scp token -> VM, restart, news-push run automatically"
Write-Host ""
Write-Host "Test it now with:  Start-ScheduledTask -TaskName 'Tradebot Morning Token'"
Write-Host "Remove it with:    Unregister-ScheduledTask -TaskName 'Tradebot Morning Token' -Confirm:`$false"
