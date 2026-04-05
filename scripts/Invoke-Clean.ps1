$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

$fixedTargets = @(
    "build",
    "dist",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".nox",
    ".coverage"
)

foreach ($target in $fixedTargets) {
    $fullPath = Join-Path $root $target
    if (Test-Path -LiteralPath $fullPath) {
        Remove-Item -LiteralPath $fullPath -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Get-ChildItem -Path $root -Directory -Filter "*.egg-info" -Force -ErrorAction SilentlyContinue |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

Get-ChildItem -Path $root -Directory -Filter "__pycache__" -Force -Recurse -ErrorAction SilentlyContinue |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
