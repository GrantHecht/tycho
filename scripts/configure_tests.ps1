#
# configure_tests.ps1 — configure and build C++ tests + benchmarks on Windows
#
# Usage:
#   powershell -File scripts/configure_tests.ps1

. "$PSScriptRoot\env_setup.ps1"

Write-Host "Configuring with BUILD_CPP_TESTS=ON BUILD_CPP_BENCHMARKS=ON..."
& $cmake --preset x64-Clang-Release `
    "-DPython_EXECUTABLE=$python" `
    "-DCMAKE_MAKE_PROGRAM=$ninja" `
    '-DBUILD_CPP_TESTS=ON' `
    '-DBUILD_CPP_BENCHMARKS=ON'

if ($LASTEXITCODE -ne 0) {
    Write-Error "CMake configuration failed (exit code $LASTEXITCODE)"
    exit $LASTEXITCODE
}

Write-Host "Building tycho_tests, tycho_tests_light, and bench_all (-j6)..."
& $cmake --build --preset x64-Clang-Release --parallel 6 --target tycho_tests --target tycho_tests_light --target bench_all 2>&1

if ($LASTEXITCODE -ne 0) {
    Write-Error "Build failed (exit code $LASTEXITCODE)"
    exit $LASTEXITCODE
}

Write-Host "Build complete."
