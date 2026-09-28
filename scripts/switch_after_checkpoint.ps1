param(
    [Parameter(Mandatory = $true)]
    [string]$Dataset
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$resultsDir = Join-Path $repoRoot 'results\main_protocol_ucr64_v1_4'
$rawPath = Join-Path $resultsDir 'raw_results.csv'
$selectionPath = Join-Path $resultsDir 'selection_results.csv'
$pidPath = Join-Path $resultsDir 'benchmark_pid.txt'
$statusPath = Join-Path $resultsDir 'switch_status.txt'
$oldProcessId = [int](Get-Content -LiteralPath $pidPath -Raw)

try {
    while ($true) {
        if ((Test-Path -LiteralPath $rawPath) -and
            (Test-Path -LiteralPath $selectionPath)) {
            $rawRows = @(Import-Csv -LiteralPath $rawPath |
                Where-Object { $_.dataset -eq $Dataset })
            $selectionRows = @(Import-Csv -LiteralPath $selectionPath |
                Where-Object { $_.dataset -eq $Dataset })
            if ($rawRows.Count -eq 360 -and $selectionRows.Count -eq 120) {
                break
            }
        }
        if (-not (Get-Process -Id $oldProcessId -ErrorAction SilentlyContinue)) {
            throw "Original benchmark process $oldProcessId exited before $Dataset checkpoint."
        }
        Start-Sleep -Seconds 5
    }

    $oldProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $oldProcessId"
    if ($oldProcess -and $oldProcess.CommandLine -notmatch 'src\.run_benchmark') {
        throw "Process $oldProcessId is not the expected benchmark process."
    }
    if ($oldProcess) {
        $children = @(Get-CimInstance Win32_Process -Filter "ParentProcessId = $oldProcessId" |
            Where-Object { $_.Name -eq 'python.exe' })
        foreach ($child in $children) {
            Stop-Process -Id $child.ProcessId -Force -ErrorAction SilentlyContinue
        }
        Stop-Process -Id $oldProcessId -Force -ErrorAction SilentlyContinue
    }

    $python = Join-Path $repoRoot '.venv\Scripts\python.exe'
    $newProcess = Start-Process -FilePath $python `
        -ArgumentList @('-m', 'src.run_benchmark', '--config',
            'configs\main_protocol_balanced.yaml') `
        -WorkingDirectory $repoRoot -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $resultsDir 'benchmark_optimized.log') `
        -RedirectStandardError (Join-Path $resultsDir 'benchmark_optimized.err.log') `
        -PassThru
    Set-Content -LiteralPath $pidPath -Value $newProcess.Id
    Set-Content -LiteralPath $statusPath -Value (
        "Switched after $Dataset checkpoint at $(Get-Date -Format s). " +
        "Old PID: $oldProcessId; new PID: $($newProcess.Id)."
    )
} catch {
    Set-Content -LiteralPath $statusPath -Value "Switch failed: $_"
    throw
}
