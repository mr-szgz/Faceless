[CmdletBinding()]
param(
    [string]$Python,
    [string]$ProjectRoot = (Join-Path $PSScriptRoot ".."),
    [string]$SpecPath = "run_faceless.spec",
    [switch]$Clean,
    [string]$DistDir,
    [string]$BuildDir,
    [bool]$PreferProjectVenv = $true
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$resolvedProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
$resolvedSpecPath = if ([IO.Path]::IsPathRooted($SpecPath)) {
    $SpecPath
} else {
    Join-Path $resolvedProjectRoot $SpecPath
}

if (-not (Test-Path -LiteralPath $resolvedSpecPath -PathType Leaf)) {
    throw "Spec file not found: $resolvedSpecPath"
}

function Resolve-PythonInterpreter {
    param(
        [string]$ConfiguredPython,
        [string]$ProjectRootPath,
        [bool]$UseProjectVenv
    )

    if (-not [string]::IsNullOrWhiteSpace($ConfiguredPython)) {
        return $ConfiguredPython
    }

    if ($UseProjectVenv) {
        $venvCandidates = @(
            (Join-Path $ProjectRootPath ".venv\Scripts\python.exe"),
            (Join-Path $ProjectRootPath ".venv/bin/python"),
            (Join-Path $ProjectRootPath ".venv/bin/python3")
        )
        foreach ($candidate in $venvCandidates) {
            if (Test-Path -LiteralPath $candidate -PathType Leaf) {
                return $candidate
            }
        }
    }

    return "python"
}

function Test-CommandExists {
    param([string]$Name)
    return $null -ne (Get-Command -Name $Name -ErrorAction SilentlyContinue)
}

function Test-PythonModule {
    param(
        [string]$PythonPath,
        [string]$ModuleName
    )

    & $PythonPath -c "import $ModuleName" *> $null
    return $LASTEXITCODE -eq 0
}

function Install-PythonPackages {
    param(
        [string]$PythonPath,
        [string[]]$Packages
    )

    if (Test-PythonModule -PythonPath $PythonPath -ModuleName "pip") {
        & $PythonPath -m pip install --upgrade @Packages
        if ($LASTEXITCODE -eq 0) {
            return
        }
    } elseif (Test-PythonModule -PythonPath $PythonPath -ModuleName "ensurepip") {
        & $PythonPath -m ensurepip --upgrade
        if ($LASTEXITCODE -eq 0) {
            & $PythonPath -m pip install --upgrade @Packages
            if ($LASTEXITCODE -eq 0) {
                return
            }
        }
    }

    if (Test-CommandExists -Name "uv") {
        & uv pip install --python $PythonPath --upgrade @Packages
        if ($LASTEXITCODE -eq 0) {
            return
        }
    }

    throw "Unable to install required Python packages. Ensure pip or uv is available for $PythonPath."
}

$resolvedPython = Resolve-PythonInterpreter -ConfiguredPython $Python -ProjectRootPath $resolvedProjectRoot -UseProjectVenv $PreferProjectVenv
Write-Host "Using Python interpreter: $resolvedPython"

$resolvedDistDir = if ([string]::IsNullOrWhiteSpace($DistDir)) {
    Join-Path $resolvedProjectRoot "dist"
} elseif ([IO.Path]::IsPathRooted($DistDir)) {
    $DistDir
} else {
    Join-Path $resolvedProjectRoot $DistDir
}

$resolvedBuildDir = if ([string]::IsNullOrWhiteSpace($BuildDir)) {
    Join-Path $resolvedProjectRoot "build"
} elseif ([IO.Path]::IsPathRooted($BuildDir)) {
    $BuildDir
} else {
    Join-Path $resolvedProjectRoot $BuildDir
}

Push-Location -LiteralPath $resolvedProjectRoot
try {
    if ($Clean) {
        Remove-Item -LiteralPath $resolvedBuildDir -Recurse -Force -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $resolvedDistDir -Recurse -Force -ErrorAction SilentlyContinue
    }

    if (-not (Test-PythonModule -PythonPath $resolvedPython -ModuleName "PyInstaller")) {
        Install-PythonPackages -PythonPath $resolvedPython -Packages @("pyinstaller")
    }

    $pyinstallerArgs = @()
    if ($resolvedDistDir) {
        $pyinstallerArgs += @("--distpath", $resolvedDistDir)
    }
    if ($resolvedBuildDir) {
        $pyinstallerArgs += @("--workpath", $resolvedBuildDir)
    }
    $pyinstallerArgs += $resolvedSpecPath

    & $resolvedPython -m PyInstaller @pyinstallerArgs
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller build failed."
    }

    Write-Host "Build artifacts:"
    if (Test-Path -LiteralPath $resolvedDistDir) {
        Get-ChildItem -LiteralPath $resolvedDistDir -File | Sort-Object Name | ForEach-Object {
            Write-Host " - $($_.Name)"
        }
    } else {
        Write-Warning "No dist directory found at $resolvedDistDir."
    }
} finally {
    Pop-Location
}
