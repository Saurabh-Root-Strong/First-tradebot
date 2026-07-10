# install_verify_task.ps1 — (re)register the weekly HEALTH check as a Windows Task Scheduler
# job. Idempotent (-Force replaces). Run once per machine:
#     powershell -ExecutionPolicy Bypass -File .\install_verify_task.ps1
#
# Runs weekly_health.bat every Monday 08:00 = NSE-constants drift + SIGNAL structural-bias
# invariants (the check that would have caught the 95%-CALL flow bias). Critical non-default
# settings:
#   -StartWhenAvailable        : a missed Monday (laptop off) runs on the NEXT wake
#   -AllowStartIfOnBatteries    : DEFAULT BLOCKS battery runs → would silently skip; allow it
#   -DontStopIfGoingOnBatteries : don't kill it mid-run on unplug
#   -ExecutionTimeLimit 45m     : the signal audit harvests EVERY captured day and grows with
#                                 the archive (minutes today) — 10m would eventually kill it.
# (dev.bat runs only the FAST NSE check on launch so startup never blocks; it still shows the
#  red DRIFT warning that either check may have raised.)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
$bat  = Join-Path $repo "weekly_health.bat"

$action  = New-ScheduledTaskAction -Execute $bat -WorkingDirectory $repo
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday -At 8:00AM
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable `
             -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
             -ExecutionTimeLimit (New-TimeSpan -Minutes 45)

Register-ScheduledTask -TaskName "TradebotVerifyNSE" -Action $action -Trigger $trigger `
    -Settings $settings -Description "Weekly health check: NSE constants drift + scout signal structural-bias invariants" -Force | Out-Null

$info = Get-ScheduledTaskInfo -TaskName "TradebotVerifyNSE"
Write-Host "Registered TradebotVerifyNSE  |  next run: $($info.NextRunTime)  |  Monday 08:00, catch-up + battery ON, 45m limit"
