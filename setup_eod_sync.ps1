# setup_eod_sync.ps1 - register the post-close VM->local archival sync as a
# Windows Scheduled Task. Runs 15:45 Mon-Fri; StartWhenAvailable means if the
# laptop was asleep/off at 15:45 it runs as soon as it next wakes (so a closed
# lid at the office never skips the day's archive).
$ErrorActionPreference = "Stop"
$dir = $PSScriptRoot
$py  = Join-Path $dir ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

$action  = New-ScheduledTaskAction -Execute $py -Argument "eod_sync.py --quiet" -WorkingDirectory $dir
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At "15:45"
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 30) `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries

Register-ScheduledTask -TaskName "Tradebot EOD Sync" -Action $action -Trigger $trigger `
    -Settings $settings -Description "Pull + merge all VM intraday data after market close" -Force

Write-Host ""
Write-Host "Registered task 'Tradebot EOD Sync':"
Write-Host "  - runs 15:45 Mon-Fri"
Write-Host "  - if the laptop was asleep then, runs as soon as it wakes"
Write-Host "  - command:  $py eod_sync.py --quiet  (in $dir)"
Write-Host ""
Write-Host "Test it now with:  Start-ScheduledTask -TaskName 'Tradebot EOD Sync'"
Write-Host "Remove it with:    Unregister-ScheduledTask -TaskName 'Tradebot EOD Sync' -Confirm:`$false"
