[CmdletBinding()]
param()

mise run build

uv run python scripts/New-ContextMenu.py -Exe ((Get-Command faceless).Source)
