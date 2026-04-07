[CmdletBinding()]
param(
    [string]$Prefix = (Split-Path -Parent $PSScriptRoot)
)
. "$Prefix\scripts\Set-Environment.ps1" -Prefix $Prefix
# ensure dependnecies are up to date for development
uv sync --extra dev 

# install install current uv preferred environment
uv pip install -e . 

$env:PATH = (Join-Path -Path $Prefix -ChildPath "dist/") + [IO.Path]::PathSeparator + $env:PATH 

# get the powershell discovered path (probably .venv path)
(Get-Command faceless).Source
(Get-Command run-faceless).Source

mise run new-menu
