#
# config_and_build.ps1 — configure and build Tycho on Windows
#
# Prerequisites:
#   - LLVM/clang-cl installed to C:\Program Files\LLVM
#   - Intel OneAPI Base Toolkit (for MKL)
#   - Visual Studio 2022 (for MSVC headers/libs used by clang-cl)
#   - conda environment named "tycho" with Python 3.13 and runtime deps:
#       conda create -n tycho python=3.13
#       conda activate tycho
#       pip install numpy scipy matplotlib spiceypy seaborn
#
# Usage:
#   powershell -File scripts/config_and_build.ps1
#
# The built module is installed directly into the tycho conda environment's
# site-packages. Activate the environment before running examples:
#   conda activate tycho
#   python examples/Brachistochrone.py

. "$PSScriptRoot\env_setup.ps1"

Write-Host "Configuring Tycho with clang-cl + MSVC environment..."
& $cmake --preset x64-Clang-Release `
    "-DPython_EXECUTABLE=$python" `
    "-DCMAKE_MAKE_PROGRAM=$ninja"

if ($LASTEXITCODE -ne 0) {
    Write-Error "CMake configuration failed (exit code $LASTEXITCODE)"
    exit $LASTEXITCODE
}

Write-Host "Building Tycho..."
& $cmake --build --preset x64-Clang-Release --parallel 8 2>&1
