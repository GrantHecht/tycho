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

# Build the wheel (scikit-build-core drives CMake internally)
Write-Host "Building wheel..."
& $python -m build --wheel --no-isolation
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
