# Configure with tests and benchmarks enabled
$env:ONEAPI_ROOT = 'C:\Program Files (x86)\Intel\oneAPI'
$env:MKLROOT    = 'C:\Program Files (x86)\Intel\oneAPI\mkl\latest'

$vcvarsall = 'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat'
$tempFile = [System.IO.Path]::GetTempFileName() + '.bat'
@"
@echo off
call "$vcvarsall" x64
set
"@ | Set-Content $tempFile

$envOutput = cmd /c $tempFile 2>&1
Remove-Item $tempFile

foreach ($line in $envOutput) {
    if ($line -match '^([^=]+)=(.*)$') {
        [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
    }
}

$env:ONEAPI_ROOT = 'C:\Program Files (x86)\Intel\oneAPI'
$env:MKLROOT    = 'C:\Program Files (x86)\Intel\oneAPI\mkl\latest'

$cmake  = 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
$ninja  = 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe'
$python = 'C:\Users\grant\miniconda3\envs\tycho\python.exe'

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

Write-Host "Building tycho_tests and bench_all (-j6)..."
& $cmake --build --preset x64-Clang-Release --parallel 6 --target tycho_tests --target bench_all 2>&1

if ($LASTEXITCODE -ne 0) {
    Write-Error "Build failed (exit code $LASTEXITCODE)"
    exit $LASTEXITCODE
}

Write-Host "Build complete."
