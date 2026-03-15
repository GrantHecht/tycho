#
# build_tests.ps1 — rebuild C++ tests + benchmarks on Windows (already configured)
#
# Usage:
#   powershell -File scripts/build_tests.ps1

. "$PSScriptRoot\env_setup.ps1"

Write-Host "Building tycho_tests and bench_all (-j6)..."
& $cmake --build --preset x64-Clang-Release --parallel 6 --target tycho_tests --target bench_all 2>&1

exit $LASTEXITCODE
