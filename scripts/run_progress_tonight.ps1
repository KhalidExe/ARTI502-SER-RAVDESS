# Progress-report batch (Windows PowerShell, from 02_code with the .venv active):
#   .\scripts\run_progress_tonight.ps1
# Stops at the first failure. Logs go to results\logs\.
$ErrorActionPreference = "Continue"   # native stderr (warnings) must not abort; exit codes are checked
$env:PYTHONUNBUFFERED = "1"
Set-Location (Split-Path $PSScriptRoot -Parent)
New-Item -ItemType Directory -Force -Path results\logs | Out-Null

function Step($name, [string[]]$cmd) {
    Write-Host "`n=== $name ===" -ForegroundColor Cyan
    $log = "results\logs\$name.log"
    & python @cmd 2>&1 | Tee-Object -Variable out
    $code = $LASTEXITCODE
    $out | Out-File -FilePath $log -Encoding utf8
    # Windows can crash at interpreter shutdown (0xC0000409) after results are written, so the exit
    # code alone is not trusted: require "[done] exit code 0" or exit 0, AND complete outputs on disk.
    $done = Select-String -Path $log -Pattern '^\[done\] exit code 0' -Quiet
    & python -m src.verify_runs --log $log
    $verified = ($LASTEXITCODE -eq 0)
    if (((($code -ne 0) -and -not $done)) -or -not $verified) {
        Write-Host "FAILED: $name (see $log)" -ForegroundColor Red; exit 1
    }
}

Step "checks"      @("-m", "src.checks", "--overfit")
Step "prelim_m1"   @("-m", "src.run_experiments", "--model", "m1", "--folds", "0-5", "--seeds", "0", "--tag", "prelim", "--skip-existing")
Step "prelim_m3"   @("-m", "src.run_experiments", "--model", "m3", "--folds", "0-5", "--seeds", "0", "--tag", "prelim", "--skip-existing")
Step "prelim_m2"   @("-m", "src.run_experiments", "--model", "m2", "--folds", "0-5", "--seeds", "0", "--tag", "prelim", "--skip-existing")
Write-Host "`nDone. Open notebooks\03_results_analysis.ipynb with TAGS = ['prelim']." -ForegroundColor Green
