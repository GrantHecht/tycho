# Phase 3: CMake Library Packaging Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Make Tycho consumable by external C++ projects via `FetchContent`/`add_subdirectory` and `find_package` after installation.

**Architecture:** Add CMake packaging infrastructure so `tycho::tycho` is a proper exported target. `FetchContent` support comes first (simpler), then `install()` rules + `find_package` config generation. Eigen is the only dependency that matters for C++ consumers — fmt, autodiff, and nanobind are implementation details.

**Tech Stack:** CMake 3.20+, `CMakePackageConfigHelpers`

---

### Task 1: Create feature branch

**Step 1: Create and check out branch**

```bash
git checkout -b feat/cmake-packaging main
```

---

### Task 2: Add `TYCHO_MASTER_PROJECT` guard

**Files:**
- Modify: `CMakeLists.txt` (root)

**Step 1: Detect if Tycho is the top-level project**

Near the top of `CMakeLists.txt`, after `project()`:

```cmake
# Detect if Tycho is the top-level project or consumed as a subdirectory
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(TYCHO_MASTER_PROJECT ON)
else()
    set(TYCHO_MASTER_PROJECT OFF)
endif()
```

**Step 2: Default options based on master project status**

Update the option defaults:
```cmake
option(ENABLE_PYTHON_BINDINGS "Build Python bindings (_tychopy nanobind module)"
    ${TYCHO_MASTER_PROJECT})
option(BUILD_CPP_EXAMPLES "Build standalone C++ example executables"
    ${TYCHO_MASTER_PROJECT})
option(BUILD_CPP_BENCHMARKS "Build C++ benchmarks using Google Benchmark"
    OFF)
option(BUILD_CPP_BENCHMARKS_LEGACY "Build legacy hand-rolled benchmark executables"
    OFF)
option(BUILD_CPP_TESTS "Build C++ unit tests"
    OFF)
```

When consumed as a subdirectory, Python bindings and examples are OFF by default.
Tests and benchmarks remain OFF regardless (they're opt-in).

**Step 3: Guard Python discovery**

Ensure the `find_package(Python ...)` and nanobind setup only runs when
`ENABLE_PYTHON_BINDINGS` is ON:

```cmake
if(ENABLE_PYTHON_BINDINGS)
    # ... existing Python/nanobind discovery ...
endif()
```

(This may already be guarded — verify and ensure completeness.)

**Step 4: Build and verify (as master project)**

```bash
cd build && rm -rf * && cmake --preset macos-llvm-release && ninja -j2 all
```

Everything should work identically since `TYCHO_MASTER_PROJECT` is ON.

**Step 5: Commit**

```bash
git add CMakeLists.txt
git commit -m "feat: add TYCHO_MASTER_PROJECT guard for subdirectory consumption"
```

---

### Task 3: Ensure `tycho::tycho` alias target exists

**Files:**
- Modify: `src/CMakeLists.txt`

**Step 1: Verify alias target**

Ensure this exists (should have been added in Phase 2):
```cmake
add_library(tycho::tycho ALIAS tycho)
```

If `tycho` is an INTERFACE library, this should work directly. If it's a STATIC
library, it also works.

**Step 2: Verify public include directory**

```cmake
target_include_directories(tycho INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
```

The `BUILD_INTERFACE` genex ensures the `include/` dir is used when building
in-tree (FetchContent/add_subdirectory). The `INSTALL_INTERFACE` genex is
for installed usage.

**Step 3: Commit if changes needed**

```bash
git add src/CMakeLists.txt
git commit -m "feat: ensure tycho::tycho alias and public include dirs"
```

---

### Task 4: Add Eigen dependency handling

**Files:**
- Modify: `CMakeLists.txt` (root)
- Create: `cmake/TychoEigenConfig.cmake` (helper module)

**Step 1: Add `TYCHO_USE_SYSTEM_EIGEN` option**

```cmake
option(TYCHO_USE_SYSTEM_EIGEN
    "Use system-installed Eigen3 instead of bundled dep/eigen" OFF)
```

**Step 2: Implement Eigen discovery logic**

```cmake
if(TYCHO_USE_SYSTEM_EIGEN)
    find_package(Eigen3 3.4 REQUIRED)
    message(STATUS "Using system Eigen3: ${Eigen3_DIR}")
else()
    # Use bundled Eigen from dep/eigen
    if(NOT TARGET Eigen3::Eigen)
        add_library(Eigen3::Eigen INTERFACE IMPORTED)
        target_include_directories(Eigen3::Eigen INTERFACE
            ${CMAKE_CURRENT_SOURCE_DIR}/dep/eigen
        )
    endif()
    message(STATUS "Using bundled Eigen from dep/eigen")
endif()
```

**Step 3: Link Eigen to the tycho target**

In `src/CMakeLists.txt`, ensure Eigen is a public dependency:
```cmake
target_link_libraries(tycho INTERFACE Eigen3::Eigen)
```

This ensures consumers automatically get Eigen on their include path.

**Step 4: Build and verify both paths**

```bash
# Bundled (default)
cd build && rm -rf * && cmake --preset macos-llvm-release && ninja -j2 all

# System (if Eigen3 is installed via Homebrew or similar)
cd build && rm -rf * && cmake --preset macos-llvm-release \
    -DTYCHO_USE_SYSTEM_EIGEN=ON && ninja -j2 all
```

**Step 5: Commit**

```bash
git add CMakeLists.txt src/CMakeLists.txt cmake/
git commit -m "feat: add TYCHO_USE_SYSTEM_EIGEN option with bundled/system Eigen support"
```

---

### Task 5: Add `FetchContent` / `add_subdirectory` test

**Files:**
- Create: `tests/cmake/fetch_content_test/CMakeLists.txt`
- Create: `tests/cmake/fetch_content_test/main.cpp`

**Step 1: Create a minimal consumer project**

`tests/cmake/fetch_content_test/CMakeLists.txt`:
```cmake
cmake_minimum_required(VERSION 3.20)
project(tycho_fetch_test CXX)

# Simulate FetchContent by adding tycho as a subdirectory
# (In real usage, this would be FetchContent_Declare/MakeAvailable)
add_subdirectory(${TYCHO_SOURCE_DIR} ${CMAKE_BINARY_DIR}/tycho-build)

add_executable(fetch_test main.cpp)
target_link_libraries(fetch_test PRIVATE tycho::tycho)
```

`tests/cmake/fetch_content_test/main.cpp`:
```cpp
#include <tycho/tycho.h>
#include <iostream>

int main() {
    // Verify core types are accessible
    Eigen::VectorXd v(3);
    v << 1.0, 2.0, 3.0;

    auto args = tycho::Arguments<3>();

    std::cout << "Tycho FetchContent test: OK" << std::endl;
    return 0;
}
```

**Step 2: Test it**

```bash
cd /tmp && mkdir tycho_fc_test && cd tycho_fc_test
cmake /path/to/tycho/tests/cmake/fetch_content_test \
    -DTYCHO_SOURCE_DIR=/path/to/tycho \
    -DCMAKE_CXX_COMPILER=/opt/homebrew/Cellar/llvm/22.1.0/bin/clang++
make
./fetch_test
```

Expected: prints "Tycho FetchContent test: OK"

**Step 3: Commit**

```bash
git add tests/cmake/
git commit -m "test: add FetchContent/add_subdirectory consumer test"
```

---

### Task 6: Add `install()` rules

**Files:**
- Modify: `src/CMakeLists.txt` — add install targets
- Modify: `CMakeLists.txt` (root) — add install for headers

**Step 1: Install the tycho library target**

In `src/CMakeLists.txt`:
```cmake
include(GNUInstallDirs)

install(TARGETS tycho
    EXPORT tychoTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Also install the internal static libraries that tycho depends on
install(TARGETS psiopt utils optimalcontrol astro
    EXPORT tychoTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
```

Note: `vectorfunctions` is INTERFACE (header-only), so it doesn't need an
ARCHIVE destination, but it should be in the export set:
```cmake
install(TARGETS vectorfunctions
    EXPORT tychoTargets
)
```

**Step 2: Install public headers**

In root `CMakeLists.txt`:
```cmake
install(DIRECTORY include/tycho
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h"
)
```

**Step 3: Install bundled Eigen headers (when not using system Eigen)**

```cmake
if(NOT TYCHO_USE_SYSTEM_EIGEN)
    install(DIRECTORY dep/eigen/Eigen
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
    install(DIRECTORY dep/eigen/unsupported
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
endif()
```

**Step 4: Commit**

```bash
git add src/CMakeLists.txt CMakeLists.txt
git commit -m "feat: add install() rules for tycho library and headers"
```

---

### Task 7: Generate and install CMake config files

**Files:**
- Create: `cmake/tychoConfig.cmake.in`
- Modify: `CMakeLists.txt` (root)

**Step 1: Create config template**

`cmake/tychoConfig.cmake.in`:
```cmake
@PACKAGE_INIT@

include(CMakeFindDependencyMacro)

# If tycho was built with system Eigen, consumers need it too
set(TYCHO_USE_SYSTEM_EIGEN @TYCHO_USE_SYSTEM_EIGEN@)
if(TYCHO_USE_SYSTEM_EIGEN)
    find_dependency(Eigen3 3.4)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/tychoTargets.cmake")

check_required_components(tycho)
```

**Step 2: Add config generation to root CMakeLists.txt**

```cmake
include(CMakePackageConfigHelpers)

# Generate version file
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/tychoConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

# Generate config file from template
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/tychoConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/tychoConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tycho
)

# Install the export set
install(EXPORT tychoTargets
    FILE tychoTargets.cmake
    NAMESPACE tycho::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tycho
)

# Install config and version files
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/tychoConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/tychoConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tycho
)
```

**Step 3: Add version to project()**

Ensure `project(tycho VERSION 0.1.0)` is set in the root CMakeLists.txt.

**Step 4: Commit**

```bash
git add cmake/tychoConfig.cmake.in CMakeLists.txt
git commit -m "feat: generate and install tychoConfig.cmake for find_package support"
```

---

### Task 8: Test `install()` + `find_package()` flow

**Files:**
- Create: `tests/cmake/find_package_test/CMakeLists.txt`
- Create: `tests/cmake/find_package_test/main.cpp`

**Step 1: Create consumer project**

`tests/cmake/find_package_test/CMakeLists.txt`:
```cmake
cmake_minimum_required(VERSION 3.20)
project(tycho_find_test CXX)

find_package(tycho REQUIRED)

add_executable(find_test main.cpp)
target_link_libraries(find_test PRIVATE tycho::tycho)
```

`tests/cmake/find_package_test/main.cpp`:
```cpp
#include <tycho/tycho.h>
#include <iostream>

int main() {
    Eigen::VectorXd v(3);
    v << 1.0, 2.0, 3.0;
    std::cout << "Tycho find_package test: OK" << std::endl;
    return 0;
}
```

**Step 2: Test the full install + find flow**

```bash
# Install tycho to a temporary prefix
cd build
cmake --install . --prefix /tmp/tycho_install

# Build consumer against installed tycho
cd /tmp && mkdir tycho_fp_test_build && cd tycho_fp_test_build
cmake /path/to/tycho/tests/cmake/find_package_test \
    -DCMAKE_PREFIX_PATH=/tmp/tycho_install \
    -DCMAKE_CXX_COMPILER=/opt/homebrew/Cellar/llvm/22.1.0/bin/clang++
make
./find_test
```

Expected: prints "Tycho find_package test: OK"

**Step 3: Commit**

```bash
git add tests/cmake/find_package_test/
git commit -m "test: add find_package consumer test"
```

---

### Task 9: Full validation

**Step 1: Clean build and test all Tycho targets**

```bash
cd build && rm -rf *
cmake --preset macos-llvm-release -DBUILD_CPP_TESTS=ON
ninja -j2 all
ctest --output-on-failure
```

**Step 2: Run Python examples**

```bash
python scripts/run_examples.py
```

**Step 3: Run C++ brachistochrone**

```bash
./examples/cpp_examples/brachistochrone/brachistochrone_cpp
```

**Step 4: Test FetchContent consumer**

```bash
cd /tmp && rm -rf tycho_fc_test && mkdir tycho_fc_test && cd tycho_fc_test
cmake /path/to/tycho/tests/cmake/fetch_content_test \
    -DTYCHO_SOURCE_DIR=/path/to/tycho \
    -DCMAKE_CXX_COMPILER=/opt/homebrew/Cellar/llvm/22.1.0/bin/clang++
make && ./fetch_test
```

**Step 5: Test install + find_package consumer**

```bash
cd /path/to/tycho/build
cmake --install . --prefix /tmp/tycho_install
cd /tmp && rm -rf tycho_fp_test_build && mkdir tycho_fp_test_build && cd tycho_fp_test_build
cmake /path/to/tycho/tests/cmake/find_package_test \
    -DCMAKE_PREFIX_PATH=/tmp/tycho_install \
    -DCMAKE_CXX_COMPILER=/opt/homebrew/Cellar/llvm/22.1.0/bin/clang++
make && ./find_test
```

**Step 6: Commit any fixes**

---

## Validation Checklist

Before marking Phase 3 complete:

- [ ] `TYCHO_MASTER_PROJECT` properly detected
- [ ] Options default OFF when consumed as subdirectory
- [ ] `tycho::tycho` alias target works
- [ ] `TYCHO_USE_SYSTEM_EIGEN` option works (both bundled and system)
- [ ] FetchContent/add_subdirectory consumer compiles and runs
- [ ] `cmake --install` succeeds
- [ ] `find_package(tycho)` works from a separate project
- [ ] Installed consumer compiles and runs
- [ ] All C++ tests pass
- [ ] All 38 Python examples pass
- [ ] C++ brachistochrone converges
