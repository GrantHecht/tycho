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

# Restore OneAPI env vars (vcvarsall may overwrite them)
$env:ONEAPI_ROOT = 'C:\Program Files (x86)\Intel\oneAPI'
$env:MKLROOT = 'C:\Program Files (x86)\Intel\oneAPI\mkl\latest'

$cmake = 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
$ninja = 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe'
$python = 'C:\Users\grant\miniconda3\envs\tycho\python.exe'

Write-Host "Configuring Tycho with clang-cl + MSVC environment..."
& $cmake --preset x64-Clang-Release `
    "-DPython_EXECUTABLE=$python" `
    "-DCMAKE_MAKE_PROGRAM=$ninja"
