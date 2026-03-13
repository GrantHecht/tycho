# Set up Intel OneAPI environment variables
$env:ONEAPI_ROOT = 'C:\Program Files (x86)\Intel\oneAPI'
$env:MKLROOT = 'C:\Program Files (x86)\Intel\oneAPI\mkl\latest'

# Import VS developer environment (sets include/lib paths for clang-cl)
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

# Restore OneAPI env vars
$env:ONEAPI_ROOT = 'C:\Program Files (x86)\Intel\oneAPI'
$env:MKLROOT = 'C:\Program Files (x86)\Intel\oneAPI\mkl\latest'

$python = 'C:\Users\grant\miniconda3\envs\tycho\python.exe'

# Build the wheel (scikit-build-core drives CMake internally).
# Compiler and generator are passed via --config-setting so scikit-build-core
# passes them directly to cmake, matching the x64-Clang-Release preset.
# Paths use the 8.3 short form (PROGRA~1) to avoid cmake argument-splitting on spaces.
$env:CMAKE_BUILD_PARALLEL_LEVEL = '8'

Write-Host "Building wheel..."
& $python -m build --wheel --no-isolation `
    -C "cmake.args=-GNinja" `
    -C "cmake.args=-DTYCHO_FP_MODE=SAFE_FAST" `
    -C "cmake.args=-DCMAKE_CXX_COMPILER=C:/PROGRA~1/LLVM/bin/clang-cl.exe" `
    -C "cmake.args=-DCMAKE_C_COMPILER=C:/PROGRA~1/LLVM/bin/clang-cl.exe" `
    -C "cmake.args=-DCMAKE_LINKER=C:/PROGRA~1/LLVM/bin/lld-link.exe" `
    -C "cmake.args=-DCMAKE_CXX_COMPILER_AR=C:/PROGRA~1/LLVM/bin/llvm-ar.exe" `
    -C "cmake.args=-DCMAKE_CXX_COMPILER_RANLIB=C:/PROGRA~1/LLVM/bin/llvm-ranlib.exe"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Bundle DLL dependencies into the wheel
Write-Host "Repairing wheel with delvewheel..."
$whl = Get-ChildItem dist\*.whl | Select-Object -Last 1
& $python -m delvewheel repair $whl.FullName `
    --add-path 'C:\Program Files (x86)\Intel\oneAPI\mkl\latest\bin;C:\Program Files\LLVM\bin' `
    --wheel-dir dist\repaired
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Wheel ready: dist\repaired\"
Get-ChildItem dist\repaired\*.whl | Select-Object Name
