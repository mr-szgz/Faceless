[CmdletBinding()]
param(
    [string]$Prefix = (Split-Path -Parent $PSScriptRoot)
)

. "$Prefix\scripts\Set-Environment.ps1" -Prefix $Prefix

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$outputDir = Join-Path $projectRoot "dist"
$specPath = Join-Path $projectRoot "run_faceless.spec"
$python = "python"

if (-not (Test-Path -LiteralPath $specPath -PathType Leaf)) {
    throw "Missing spec file: $specPath"
}

Push-Location -LiteralPath $projectRoot
try {
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null

    Get-Process -Name "run-faceless" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue

    & $python -m pip install --upgrade pyinstaller build wheel
    if ($LASTEXITCODE -ne 0) {
        throw "pip install failed."
    }

    & $python -m PyInstaller --noconfirm --distpath $outputDir --workpath (Join-Path $projectRoot "build") $specPath
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller build failed."
    }

    & $python -m build --wheel --sdist --outdir $outputDir
    if ($LASTEXITCODE -ne 0) {
        throw "python -m build failed."
    }

    Write-Host "Build artifacts:"
    Get-ChildItem -LiteralPath $outputDir -File | Sort-Object Name | ForEach-Object {
        Write-Host " - $($_.Name)"
    }
} finally {
    Pop-Location
}
