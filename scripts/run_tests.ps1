# Run C++ unit tests via CTest
$ctest = 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\ctest.exe'
$buildDir = 'C:\Users\grant\src\tycho\out\build\x64-Clang-Release'

Write-Host "Running C++ unit tests..."
& $ctest --output-on-failure --test-dir $buildDir -C Release 2>&1
exit $LASTEXITCODE
