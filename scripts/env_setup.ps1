#
# env_setup.ps1 -- shared Windows build environment setup
#
# Dot-source this from other scripts:  . "$PSScriptRoot\env_setup.ps1"
#
# Provides:
#   $cmake    -- path to cmake.exe (from VS installation)
#   $ninja    -- path to ninja.exe (from VS installation)
#   $ctest    -- path to ctest.exe (from VS installation)
#   $python   -- path to python.exe in the tycho conda env
#   $repoRoot -- path to the repository root
#
# Also loads the VS developer environment and Intel OneAPI/MKL env vars.
#

# -- Repository root (one level up from scripts/) --
$repoRoot = Split-Path -Parent $PSScriptRoot

# -- Intel OneAPI / MKL --
$env:ONEAPI_ROOT = 'C:\Program Files (x86)\Intel\oneAPI'
$env:MKLROOT     = 'C:\Program Files (x86)\Intel\oneAPI\mkl\latest'

# -- Locate Visual Studio via vswhere --
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) {
    Write-Error "vswhere not found at $vswhere -- is Visual Studio installed?"
    exit 1
}
$vsInstall = & $vswhere -latest -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath
if (-not $vsInstall) {
    Write-Error "No Visual Studio installation with C++ tools found."
    exit 1
}

# -- Import VS developer environment --
$vcvarsall = Join-Path $vsInstall 'VC\Auxiliary\Build\vcvarsall.bat'
$tempFile  = [System.IO.Path]::GetTempFileName() + '.bat'
@'
@echo off
call "%1" x64
set
'@ | Set-Content $tempFile

$envOutput = cmd /c $tempFile $vcvarsall 2>&1
Remove-Item $tempFile

foreach ($line in $envOutput) {
    if ($line -match '^([^=]+)=(.*)$') {
        [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
    }
}

# Restore OneAPI env vars (vcvarsall may overwrite them)
$env:ONEAPI_ROOT = 'C:\Program Files (x86)\Intel\oneAPI'
$env:MKLROOT     = 'C:\Program Files (x86)\Intel\oneAPI\mkl\latest'

# -- Tool paths from VS installation --
$vsCmakeDir = Join-Path $vsInstall 'Common7\IDE\CommonExtensions\Microsoft\CMake'
$cmake = Join-Path $vsCmakeDir 'CMake\bin\cmake.exe'
$ctest = Join-Path $vsCmakeDir 'CMake\bin\ctest.exe'
$ninja = Join-Path $vsCmakeDir 'Ninja\ninja.exe'

foreach ($tool in @($cmake, $ninja, $ctest)) {
    if (-not (Test-Path $tool)) {
        Write-Error "Tool not found: $tool"
        exit 1
    }
}

# -- Python from tycho conda environment --
$python = (conda run -n tycho where python 2>$null) | Select-Object -First 1
if (-not $python -or -not (Test-Path $python)) {
    Write-Error "Python not found in tycho conda environment. Run: conda create -n tycho python=3.13"
    exit 1
}
