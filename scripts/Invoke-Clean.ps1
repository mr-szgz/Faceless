[CmdletBinding()]
param(
    [string]$Python,
    [string]$ProjectRoot = (Join-Path $PSScriptRoot ".."),
    [string]$OutputDir = "dist",
    [switch]$Clean,
    [string]$Prefix = (Split-Path -Parent $PSScriptRoot)
)
. "$Prefix\scripts\Set-Environment.ps1" -Prefix $Prefix

$fixedTargets = @(
    "build",
    "dist",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".nox",
    ".coverage",
    "dependencies/venv"
)

foreach ($target in $fixedTargets) {
    $fullPath = Join-Path $Prefix $target
    if (Test-Path -LiteralPath $fullPath) {
        Remove-Item -LiteralPath $fullPath -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Get-ChildItem -Path $Prefix -Directory -Filter "*.egg-info" -Force -ErrorAction SilentlyContinue |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

Get-ChildItem -Path $Prefix -Directory -Filter "__pycache__" -Force -Recurse -ErrorAction SilentlyContinue |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
