# install_verify_task.ps1 — (re)register the weekly NSE-constants drift check as a
# Windows Task Scheduler job. Idempotent (-Force replaces). Run once per machine:
#     powershell -ExecutionPolicy Bypass -File .\install_verify_task.ps1
#
# The task runs verify_nse.bat every Monday 08:00. Critical non-default settings:
#   -StartWhenAvailable        : a missed Monday (laptop off) runs on the NEXT wake
#   -AllowStartIfOnBatteries    : DEFAULT BLOCKS battery runs → would silently skip; allow it
#   -DontStopIfGoingOnBatteries : don't kill it mid-run on unplug
# (dev.bat ALSO runs the check on project launch if >7 days stale — same log-mtime clock,
#  so the two never double-run. This task is the "even if I never open the project" path.)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
$bat  = Join-Path $repo "verify_nse.bat"

$action  = New-ScheduledTaskAction -Execute $bat -WorkingDirectory $repo
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday -At 8:00AM
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable `
             -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
             -ExecutionTimeLimit (New-TimeSpan -Minutes 10)

Register-ScheduledTask -TaskName "TradebotVerifyNSE" -Action $action -Trigger $trigger `
    -Settings $settings -Description "Weekly NSE constants drift check (holidays/expiry/lot sizes vs DCM ground truth)" -Force | Out-Null

$info = Get-ScheduledTaskInfo -TaskName "TradebotVerifyNSE"
Write-Host "Registered TradebotVerifyNSE  |  next run: $($info.NextRunTime)  |  Monday 08:00, catch-up + battery ON"
