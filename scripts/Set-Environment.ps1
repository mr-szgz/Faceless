[CmdletBinding()]
param(
    [string]$Prefix = (Split-Path -Parent $PSScriptRoot)
)

$env:TORCH_HOME = "$Prefix/dependencies/TORCH_HOME"

$env:MISE_CEILING_PATHS = (Split-Path -Parent $Prefix)
mise use uv
mise use powershell@7
$env:PATH = "$(mise where uv)" + [IO.Path]::PathSeparator + $env:PATH
$env:PATH = "$(mise where powershell)" + [IO.Path]::PathSeparator + $env:PATH

$env:VENV = Join-Path -Path $Prefix -ChildPath ".venv"
$env:UV_PROJECT_ENVIRONMENT = $env:VENV
$env:PATH = (Join-Path -Path $env:VENV -ChildPath "Scripts") + [IO.Path]::PathSeparator + $env:PATH 
$env:PATH = (Join-Path -Path $env:VENV -ChildPath "Scripts") + [IO.Path]::PathSeparator + $env:PATH 


$env:PATH = (Join-Path -Path $Prefix -ChildPath "dist") + [IO.Path]::PathSeparator + $env:PATH 
$env:PATH = (Join-Path -Path $Prefix -ChildPath "dependencies/git") + [IO.Path]::PathSeparator + $env:PATH 

