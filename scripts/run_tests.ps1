#
# run_tests.ps1 — run C++ unit tests via CTest on Windows
#
# Usage:
#   powershell -File scripts/run_tests.ps1

. "$PSScriptRoot\env_setup.ps1"

$buildDir = Join-Path $repoRoot 'out\build\x64-Clang-Release'

Write-Host "Running C++ unit tests..."
& $ctest --output-on-failure --test-dir $buildDir -C Release 2>&1
exit $LASTEXITCODE
