#!/usr/bin/env pwsh
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$examplesPath = (Resolve-Path (Join-Path $repoRoot "examples")).Path
$pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Python executable not found at $pythonExe"
}

$tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("faceless-regression-" + [guid]::NewGuid().ToString("N"))
$workDir = Join-Path $tempRoot "examples"

New-Item -ItemType Directory -Path $tempRoot | Out-Null
Copy-Item -LiteralPath $examplesPath -Destination $tempRoot -Recurse -Force

$labelsPath = Join-Path $workDir "labels"
if (Test-Path -LiteralPath $labelsPath) {
    Remove-Item -LiteralPath $labelsPath -Recurse -Force
}

& $pythonExe -m faceless $workDir -Auto -Label
if ($LASTEXITCODE -ne 0) {
    throw "faceless run failed with exit code $LASTEXITCODE"
}

$labelCount = (Get-ChildItem -LiteralPath $labelsPath -Filter "*.txt" -File -ErrorAction SilentlyContinue | Measure-Object).Count
if ($labelCount -le 0) {
    throw "Regression failed: no labels were generated."
}

Write-Host "Regression passed: generated $labelCount labels in $workDir"

if (Test-Path -LiteralPath $tempRoot) {
    Remove-Item -LiteralPath $tempRoot -Recurse -Force
}
