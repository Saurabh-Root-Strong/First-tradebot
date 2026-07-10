# install_btst_tasks.ps1 — (re)register the BTST paper-loop tasks. Idempotent (-Force).
#     powershell -ExecutionPolicy Bypass -File .\install_btst_tasks.ps1
#
# WHY THIS EXISTS: the tasks were originally created with the schtasks recipe in
# btst_auto.bat's header:  schtasks /TR "\"%CD%\btst_auto.bat\""
# which stores  Execute=cmd  Args=/c ""D:\...\btst_auto.bat""  — DOUBLED quotes. cmd
# mis-parses that and the run dies with 0x80070002 (file not found). That is exactly how
# "BTST morning" failed on 2026-07-07, leaving a MIDCPNIFTY position unreconciled; the
# scorecard silently dropped it and read +14.3 bps when the truth was +6.6 bps.
#
# The cmdlet form below sets Execute to the .bat directly (no cmd wrapper, no quote
# mangling) + an explicit WorkingDirectory.
#
# Both tasks run btst_auto.bat — the HEADLESS, idempotent runner (reconcile -> emit ->
# scorecard). Do NOT schedule btst_eod.bat / btst_morning.bat: they contain `pause` and
# would hang forever as a scheduled task.
#   09:35  fills last night's exits (reconcile) + scorecard
#   15:25  logs tonight's strong-close candidates
# Running on a holiday is harmless — btst_signal picks the most recent trading day and the
# runner is idempotent (re-running updates, never double-books).

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
$bat  = Join-Path $repo "btst_auto.bat"
if (-not (Test-Path $bat)) { throw "btst_auto.bat not found at $bat" }

$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable `
             -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
             -ExecutionTimeLimit (New-TimeSpan -Minutes 30)
$action = New-ScheduledTaskAction -Execute $bat -WorkingDirectory $repo

foreach ($job in @(@{Name="BTST morning"; At="09:35"}, @{Name="BTST eod"; At="15:25"})) {
    $trigger = New-ScheduledTaskTrigger -Daily -At $job.At
    Register-ScheduledTask -TaskName $job.Name -Action $action -Trigger $trigger `
        -Settings $settings -Description "BTST close-strength PAPER loop (nothing auto-executes)" -Force | Out-Null
    Enable-ScheduledTask -TaskName $job.Name | Out-Null
    $i = Get-ScheduledTaskInfo -TaskName $job.Name
    $s = (Get-ScheduledTask -TaskName $job.Name).State
    Write-Host ("{0,-14} state={1,-8} next={2}" -f $job.Name, $s, $i.NextRunTime)
}
