# ensure dependnecies are up to date for development
uv sync --extra dev 

# install install current uv preferred environment
uv pip install -e . 

# get the powershell discovered path (probably .venv path)
(Get-Command faceless).Source
# TODO: if Where-Is cmdlet is avaialble also call-it with `faceless`
