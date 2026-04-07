[CmdletBinding()]
param(
    [string]$Prefix = (Split-Path -Parent $PSScriptRoot)
)
. "$Prefix\scripts\Set-Environment.ps1" -Prefix $Prefix

$env:PATH = "$(Split-Path -Parent (Get-Command python).Source)" + [IO.Path]::PathSeparator + $env:PATH

$env:UV_VENV_CLEAR = 0
uv venv --python 3.10.16 --seed --cache-dir (Join-Path -Path $Prefix -ChildPath "dependencies/uv/.cache")
uv run python -m ensurepip
uv run python -m pip install -U pip

mise run "clean"
mise run "wheel"

(Get-Command python).Source
python -m faceless --help

mise run "build"

$env:PATH = (Join-Path -Path $Prefix -ChildPath "dist") + [IO.Path]::PathSeparator + $env:PATH 

(Get-Command run-faceless).Source
run-faceless --install-dependencies $Prefix
