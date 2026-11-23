[CmdletBinding()]
param(
    [ValidateRange(66, 2147483647)]
    [int]$Budget = 500000,
    [string]$Results,
    [string]$PythonExecutable,
    [switch]$Smoke
)

$ErrorActionPreference = 'Stop'
$codeDirectory = $PSScriptRoot

if (-not $PythonExecutable) {
    # Get-Command can return several python.exe installations. Pass one path,
    # never an Object[] of paths, to Start-Process.
    $pythonCommand = Get-Command python -CommandType Application | Select-Object -First 1
    if (-not $pythonCommand) { throw 'Python was not found. Supply -PythonExecutable PATH.' }
    $PythonExecutable = $pythonCommand.Source
}
& $PythonExecutable -c 'import numpy, scipy, pandas, matplotlib, gymnasium, ale_py'
if ($LASTEXITCODE -ne 0) {
    throw "Dependencies are missing for $PythonExecutable. Install requirements.txt with that Python."
}

if (-not $Results) {
    $directoryName = if ($Smoke) { 'validation_parallel_' + (Get-Date -Format 'yyyyMMdd_HHmmss') }
                     else { 'assignment4_v8_500k_parallel' }
    $Results = Join-Path (Split-Path $codeDirectory -Parent) "results\$directoryName"
}
$resultsDirectory = [IO.Path]::GetFullPath($Results)
if (Test-Path -LiteralPath $resultsDirectory) {
    if (-not (Test-Path -LiteralPath $resultsDirectory -PathType Container)) {
        throw "Results path is not a directory: $resultsDirectory"
    }
    if (@(Get-ChildItem -LiteralPath $resultsDirectory -Force).Count -gt 0) {
        throw "Results directory is not empty. Choose a fresh -Results directory: $resultsDirectory"
    }
} else {
    New-Item -ItemType Directory -Path $resultsDirectory | Out-Null
}

$studyProcesses = New-Object 'System.Collections.Generic.List[System.Diagnostics.Process]'

function Start-StudyShard {
    param([string]$Shard, [switch]$Tune)
    $name = $Shard.Replace(':', '_')
    $launchArguments = @(
        '-u', '-X', 'utf8', 'part2_run.py', '--budget', "$Budget",
        '--only', $Shard, '--results', ('"{0}"' -f $resultsDirectory)
    )
    if ($Smoke) { $launchArguments += '--smoke' }
    if ($Tune) { $launchArguments += '--tune' }

    $startedProcess = Start-Process -FilePath $PythonExecutable `
        -ArgumentList $launchArguments -WorkingDirectory $codeDirectory `
        -WindowStyle Hidden -PassThru -ErrorAction Stop `
        -RedirectStandardOutput "$resultsDirectory\stdout_$name.log" `
        -RedirectStandardError "$resultsDirectory\stderr_$name.log"
    if ($null -eq $startedProcess) { throw "Failed to launch $Shard. No process returned." }
    # Retain a process handle so its exit code remains available after exit.
    $null = $startedProcess.Handle
    $studyProcesses.Add($startedProcess)
    Write-Host "Started $Shard (PID $($startedProcess.Id))"
    return $startedProcess
}

Write-Host "Python: $PythonExecutable"
Write-Host "Results and logs: $resultsDirectory"
if ($Smoke) {
    Write-Host 'Technical smoke test: one seed, 256 interactions, two evaluation episodes; no tuning.'
    $firstProcess = Start-StudyShard 'Q:A'
} else {
    Write-Host 'Tuning both algorithms once before launching the other five shards.'
    $firstProcess = Start-StudyShard 'Q:A' -Tune
    $selectionPath = Join-Path $resultsDirectory 'selected_Q_A.json'
    while ($true) {
        $selection = $null
        if (Test-Path -LiteralPath $selectionPath) {
            # The runner writes this small file directly. Wait for complete JSON.
            try { $selection = Get-Content -LiteralPath $selectionPath -Raw | ConvertFrom-Json }
            catch { $selection = $null }
            if ($selection -and $selection.selected) { break }
        }
        $firstProcess.Refresh()
        if ($firstProcess.HasExited) {
            throw "Tuning exited before selection was available. Check $resultsDirectory\stderr_Q_A.log"
        }
        Start-Sleep -Seconds 2
    }
    Write-Host ('Selected: ' + ($selection.selected | ConvertTo-Json -Compress))
}

foreach ($shard in 'Q:B', 'Q:C', 'PPO:A', 'PPO:B', 'PPO:C') {
    $null = Start-StudyShard $shard
}

$lastProgress = [DateTime]::MinValue
while ($true) {
    $runningCount = 0
    foreach ($studyProcess in $studyProcesses) {
        $studyProcess.Refresh()
        if (-not $studyProcess.HasExited) { $runningCount++ }
        elseif ($studyProcess.ExitCode -ne 0) {
            throw "Shard PID $($studyProcess.Id) failed with exit code $($studyProcess.ExitCode). Check stderr logs in $resultsDirectory. Other shards may still be running."
        }
    }
    if ($runningCount -eq 0) { break }
    if (([DateTime]::Now - $lastProgress).TotalSeconds -ge 30) {
        $summaryRows = @(Get-ChildItem -LiteralPath $resultsDirectory -Filter 'seed_results*.csv' |
            ForEach-Object { Import-Csv -LiteralPath $_.FullName })
        $trainedCount = @($summaryRows | Where-Object stage -eq 'main').Count
        $targetCount = if ($Smoke) { 6 } else { 30 }
        Write-Host "$runningCount shards active; $trainedCount/$targetCount trained agents recorded."
        $lastProgress = [DateTime]::Now
    }
    Start-Sleep -Seconds 2
}

$summaryRows = @(Get-ChildItem -LiteralPath $resultsDirectory -Filter 'seed_results*.csv' |
    ForEach-Object { Import-Csv -LiteralPath $_.FullName })
$expectedCount = if ($Smoke) { 8 } else { 40 }
if ($summaryRows.Count -ne $expectedCount) {
    throw "Expected $expectedCount result rows, found $($summaryRows.Count). Inspect the logs."
}
& $PythonExecutable -X utf8 (Join-Path $codeDirectory 'part2_tables.py') --results $resultsDirectory
if ($LASTEXITCODE -ne 0) { throw 'Runs completed, but report generation failed.' }
Write-Host "Finished successfully. Report: $resultsDirectory\part2_tables.md"
