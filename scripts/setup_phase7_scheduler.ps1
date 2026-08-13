# Phase 7 Windows Task Scheduler Setup Script
#
# ONE-TIME, HUMAN-RUN setup artifact (D-02)
#
# This script documents and registers a Windows Task Scheduler entry that will
# invoke phase7_daily_runner.py daily. It must be run manually, as Administrator,
# by the user when they choose to start the 4-6 week paper trading validation window.
#
# IMPORTANT: This script is NOT invoked automatically by any other script in the repo.
# It is a documented artifact only -- the user runs it manually when ready to begin.
#
# Usage (PowerShell, as Administrator):
#   .\setup_phase7_scheduler.ps1
#
# To remove the scheduled task later:
#   Unregister-ScheduledTask -TaskName "Phase7-DailyValidation" -Confirm:$false
#
# Exit codes:
#   0: Task registered successfully
#   1: Script was not run as Administrator
#

param(
    [string]$PythonExe = "python",
    [string]$TriggerTime = "16:05",
    [string]$TaskName = "Phase7-DailyValidation"
)

# Check if running as Administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")
if (-not $isAdmin) {
    Write-Host "ERROR: This script must be run as Administrator." -ForegroundColor Red
    Write-Host "Please open PowerShell as Administrator and try again."
    exit 1
}

# Resolve the absolute path to phase7_daily_runner.py relative to this script
$scriptPath = Join-Path $PSScriptRoot "phase7_daily_runner.py"

if (-not (Test-Path $scriptPath)) {
    Write-Host "ERROR: phase7_daily_runner.py not found at $scriptPath" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Phase 7 Windows Task Scheduler Setup" -ForegroundColor Cyan
Write-Host ("=" * 80)
Write-Host ""
Write-Host "This script will register a daily scheduled task that runs the Phase 7"
Write-Host "paper trading validation runner at $TriggerTime local time each day."
Write-Host ""
Write-Host "Task Name:      $TaskName"
Write-Host "Python Exe:     $PythonExe"
Write-Host "Script Path:    $scriptPath"
Write-Host "Trigger Time:   $TriggerTime (local time, after US market close)"
Write-Host ""

# Create the trigger
$trigger = New-ScheduledTaskTrigger -Daily -At $TriggerTime
if ($null -eq $trigger) {
    Write-Host "ERROR: Failed to create trigger" -ForegroundColor Red
    exit 1
}

# Create the action
$action = New-ScheduledTaskAction -Execute $PythonExe -Argument "`"$scriptPath`""
if ($null -eq $action) {
    Write-Host "ERROR: Failed to create action" -ForegroundColor Red
    exit 1
}

# Settings: this task must survive the machine being off/asleep/on-battery at the
# trigger time -- D-08's retry-then-skip only guards failures *inside* a running
# process, it does nothing if the process never launches. Without StartWhenAvailable,
# a missed trigger (PC off at 16:05) silently drops the entire day with zero record,
# not even a skip log entry, since phase7_daily_runner.py never starts.
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries

# Register the scheduled task
try {
    Register-ScheduledTask `
        -TaskName $TaskName `
        -Trigger $trigger `
        -Action $action `
        -Settings $settings `
        -RunLevel Highest `
        -Description "Phase 7 paper trading validation daily runner (D-02)" `
        -Force
    Write-Host "[OK] Task registered successfully!" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Failed to register task" -ForegroundColor Red
    Write-Host $_.Exception.Message
    exit 1
}

# Verify the task was created
try {
    $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction Stop
    Write-Host ""
    Write-Host "Task Details:" -ForegroundColor Cyan
    Write-Host "  Name:       $($task.TaskName)"
    Write-Host "  State:      $($task.State)"
    Write-Host "  Enabled:    $($task.Settings.Enabled)"
    Write-Host ""
} catch {
    Write-Host "WARNING: Task was registered but could not be verified." -ForegroundColor Yellow
}

Write-Host ""
Write-Host ("=" * 80)
Write-Host ""
Write-Host "To view the scheduled task:" -ForegroundColor Cyan
Write-Host "  taskkill.exe /FI `"TASKNAME eq $TaskName`" /V"
Write-Host ""
Write-Host "To remove this scheduled task (when the 4-6 week window is complete):" -ForegroundColor Cyan
Write-Host "  Unregister-ScheduledTask -TaskName `"$TaskName`" -Confirm:`$false"
Write-Host ""
Write-Host "The task will now run daily at $TriggerTime until manually removed."
Write-Host ""

exit 0
