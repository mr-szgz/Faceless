[CmdletBinding()]
param(
    [string]$Prefix = (Split-Path -Parent $PSScriptRoot),
    [switch]$Force
)
. "$Prefix\scripts\Set-Environment.ps1" -Prefix $Prefix

$examples = Join-Path $Prefix "examples"
$zip = Join-Path $Prefix "examples.zip"

if (Test-Path -Path $examples) {
  $count = (Get-ChildItem -Path $examples -File).Count
  if (-not $Force) {
    $answer = Read-Host "Delete examples '$examples' and reset from '$zip'. Continue and remove $count files? (Y/n)"
    if ($answer -match '^n') {
        return -1
    }
  }

  Remove-Item -Path $examples -Recurse -Force
}

if (Test-Path -Path $zip) {
  Expand-Archive -Path $zip -DestinationPath $examples

  # unpack zip wrapping folder
  $children = Get-ChildItem -Path $examples
  if ($children.Count -eq 1 -and $children[0].PSIsContainer) {
    $wrapper = $children[0].FullName
    Get-ChildItem -Path $wrapper | Move-Item -Destination $examples
    Remove-Item -Path $wrapper -Recurse -Force
  }

  $count = (Get-ChildItem -Path $examples -File).Count
  Write-Host "Done. $count examples in '$examples'"
}
