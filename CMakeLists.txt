cmake_minimum_required(VERSION 3.16)
project(Tycho VERSION 0.1.0 LANGUAGES CXX)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Detect if Tycho is the top-level project or consumed as a subdirectory
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(TYCHO_MASTER_PROJECT ON)
else()
    set(TYCHO_MASTER_PROJECT OFF)
endif()

set( CMAKE_MESSAGE_LOG_LEVEL "STATUS" )
################################################################################
################## Includes ####################################################
################################################################################

include(cmake/git-submodule-init.cmake)
include(cmake/git-submodule-update.cmake)
include(CheckCXXCompilerFlag)


################################################################################
############### Warnings and Errors ############################################
################################################################################

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there. You may need to remove CMakeCache.txt.")
endif()

if(${CMAKE_CXX_COMPILER_ID} STREQUAL "MSVC")
  message(FATAL_ERROR "You cannot build Tycho with the Microsoft Compiler. Please use Clang or GCC and try again.")
endif()

################################################################################
############### Build Settings #################################################
################################################################################


set(LINK_TIME_OPT FALSE)          ### DO LTO - Recommended for full release
set(CLANG_MAX_INLINE_DEPTH 225)
#set(BUILD_TYCHO_WHEEL TRUE)      ### CI Build for generic wheels
#set(BUILD_SPHINX_DOCS TRUE)

# Floating-point optimization policy.
# STRICT: no fast-math relaxations
# SAFER_FAST: all fast-math optimizations except finite-math-only
# FAST: full fast-math
set(_TYCHO_FP_MODE_DEFAULT "SAFER_FAST")

set(TYCHO_FP_MODE "${_TYCHO_FP_MODE_DEFAULT}" CACHE STRING
    "Floating-point optimization mode. Valid values: STRICT, SAFER_FAST, FAST."
)
set_property(CACHE TYCHO_FP_MODE PROPERTY STRINGS STRICT SAFER_FAST FAST)

# Normalize to uppercase for comparison without FORCE-writing back to the cache,
# so that ccmake / cmake-gui edits are not silently overwritten on re-configure.
string(TOUPPER "${TYCHO_FP_MODE}" TYCHO_FP_MODE)

set(_TYCHO_VALID_FP_MODES STRICT SAFER_FAST FAST)
list(FIND _TYCHO_VALID_FP_MODES "${TYCHO_FP_MODE}" _TYCHO_FP_MODE_INDEX)
if(_TYCHO_FP_MODE_INDEX EQUAL -1)
    message(FATAL_ERROR
        "Invalid TYCHO_FP_MODE='${TYCHO_FP_MODE}'. Valid values: STRICT, SAFER_FAST, FAST."
    )
endif()

set(_TYCHO_WINDOWS_CLANG_NO_FINITE_MATH_ONLY_FLAG "-fno-finite-math-only")
if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang"
   AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    # clang-cl ignores raw GNU-style frontend flags; forward them explicitly.
    set(_TYCHO_WINDOWS_CLANG_NO_FINITE_MATH_ONLY_FLAG "/clang:-fno-finite-math-only")
endif()

if(WIN32 AND TYCHO_FP_MODE STREQUAL "SAFER_FAST")
    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        message(FATAL_ERROR
            "TYCHO_FP_MODE=SAFER_FAST requires Clang on Windows to disable finite-math-only."
        )
    endif()

    check_cxx_compiler_flag(
        "${_TYCHO_WINDOWS_CLANG_NO_FINITE_MATH_ONLY_FLAG}"
        TYCHO_HAS_NO_FINITE_MATH_ONLY_FLAG
    )
    if(NOT TYCHO_HAS_NO_FINITE_MATH_ONLY_FLAG)
        message(FATAL_ERROR
            "The active Windows Clang frontend does not support "
            "${_TYCHO_WINDOWS_CLANG_NO_FINITE_MATH_ONLY_FLAG}, so "
            "TYCHO_FP_MODE=SAFER_FAST is unavailable."
        )
    endif()
endif()

option(ENABLE_PYTHON_BINDINGS "Build Python bindings (_tychopy nanobind module)"
    ${TYCHO_MASTER_PROJECT})
option(BUILD_CPP_EXAMPLES "Build standalone C++ example executables"
    ${TYCHO_MASTER_PROJECT})
option(BUILD_CPP_BENCHMARKS "Build C++ benchmarks using Google Benchmark (requires FetchContent)" OFF)
option(BUILD_CPP_BENCHMARKS_LEGACY "Build legacy hand-rolled benchmark executables" OFF)
option(BUILD_CPP_TESTS "Build C++ unit tests (requires Google Test via FetchContent)" OFF)
option(OPTIMIZE_BINDINGS "Compile Python binding code at -O3 instead of -Os (nanobind default)" OFF)
option(COMPILE_TIME_TRACE "Enable Clang -ftime-trace for build profiling (Clang only)" OFF)
option(TYCHO_USE_SYSTEM_EIGEN
    "Use system-installed Eigen3 instead of bundled dep/eigen" OFF)

# Auto-detect ccache to accelerate repeated builds (brew install ccache)
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    message(STATUS "ccache found: ${CCACHE_PROGRAM}")
endif()


################################################################################
##########################  Set Compiler Flags  ################################
################################################################################

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SIMD_FLAGS)
set(FP_FLAGS)


# Release Flags
if(NOT WIN32)
    list(APPEND RELEASE_FLAGS "-O3")
endif()


## Generic Binary Flags
if(BUILD_TYCHO_WHEEL)
    # Detect architecture for wheel builds
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64|ARM64")
        # ARM64 wheels are macOS-only (Apple Silicon); target M1 baseline
        list(APPEND SIMD_FLAGS "-mcpu=apple-m1")
        message(STATUS "Building ARM64 wheel targeting Apple M1+")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        # x86-64 (Intel/AMD) - use AVX2 (x86-64-v3 flags written out for older compilers)
        list(APPEND SIMD_FLAGS "-mcx16")
        list(APPEND SIMD_FLAGS "-mpopcnt")
        list(APPEND SIMD_FLAGS "-msse3")
        list(APPEND SIMD_FLAGS "-msse4.1")
        list(APPEND SIMD_FLAGS "-msse4.2")
        list(APPEND SIMD_FLAGS "-mssse3")
        list(APPEND SIMD_FLAGS "-mavx")
        list(APPEND SIMD_FLAGS "-mavx2")
        list(APPEND SIMD_FLAGS "-mbmi")
        list(APPEND SIMD_FLAGS "-mbmi2")
        list(APPEND SIMD_FLAGS "-mf16c")
        list(APPEND SIMD_FLAGS "-mfma")
        list(APPEND SIMD_FLAGS "-mlzcnt")
        list(APPEND SIMD_FLAGS "-mmovbe")
        list(APPEND SIMD_FLAGS "-mxsave")
        message(STATUS "Building x86-64 wheel with AVX2 support")
    else()
        # Fallback for other architectures
        list(APPEND SIMD_FLAGS "-march=native")
        message(WARNING "Unknown architecture ${CMAKE_SYSTEM_PROCESSOR}, using -march=native")
    endif()

    # Windows runs out of ram with LINK_TIME_OPT on gh-actions
    if(NOT WIN32)
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
            list(APPEND SIMD_FLAGS "-mtune=skylake")
        endif()
    endif()
    set(LINK_TIME_OPT TRUE)   ### DO LTO - Recommended for full release
    set(CLANG_MAX_INLINE_DEPTH 400)
    
else()
    list(APPEND SIMD_FLAGS "-march=native")
    if(NOT WIN32)
        list(APPEND SIMD_FLAGS "-mtune=native")
    endif()
endif()


if(WIN32)
     list(APPEND RELEASE_FLAGS "/MD")
     list(APPEND RELEASE_FLAGS "/DNDEBUG")
     list(APPEND RELEASE_FLAGS "/O2")
     list(APPEND RELEASE_FLAGS "/Ob2")
     list(APPEND SIMD_FLAGS "/arch:AVX2")
     list(APPEND RELEASE_FLAGS "/GS-")
     list(APPEND RELEASE_FLAGS "/Gw")
endif()

if(NOT WIN32)
    # Debugging Flags
    set(DEBUG_FLAGS "-g" "-ggdb3")
    # Common Flags
    list(APPEND COMMON_FLAGS "-pthread")
endif()
 

if(LINK_TIME_OPT)
  if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    if(WIN32)
        ## clang-cl flag
        if(BUILD_TYCHO_WHEEL)
            ## Do full LTO for CI builds since it is single threaded anyway
            list(APPEND RELEASE_FLAGS "-flto=full")
            list(APPEND RELEASE_FLAGS "/opt:lldltojobs=1")
        else()
            list(APPEND RELEASE_FLAGS "-flto=thin")
            list(APPEND RELEASE_FLAGS "/opt:lldltojobs=8")
        endif()
    else()
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
    endif()

  elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
    set(CMAKE_CXX_COMPILE_OPTIONS_IPO ${CMAKE_CXX_COMPILE_OPTIONS_IPO} -flto=1)
  endif()

else()
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
endif()



list(APPEND COMMON_FLAGS "$<$<STREQUAL:${CMAKE_CXX_COMPILER_ID},Clang>:-mllvm>")
list(APPEND COMMON_FLAGS "$<$<STREQUAL:${CMAKE_CXX_COMPILER_ID},Clang>:-inline-threshold=${CLANG_MAX_INLINE_DEPTH}>")



if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU" AND FALSE)
    list(APPEND RELEASE_FLAGS "-finline-limit=1000")
    list(APPEND RELEASE_FLAGS "--param=inline-unit-growth=500")    #40,40
    list(APPEND RELEASE_FLAGS "--param=early-inlining-insns=50")   #14,14
    list(APPEND RELEASE_FLAGS "--param=large-function-insns=3500") #2700,2700
    list(APPEND RELEASE_FLAGS "--param=large-function-growth=120") #100,100
    list(APPEND RELEASE_FLAGS "--param=inline-min-speedup=10")     #30,15
endif()


if(NOT WIN32)
  list(APPEND RELEASE_FLAGS "-fomit-frame-pointer")
  list(APPEND RELEASE_FLAGS "-fno-stack-protector")
  #list(APPEND RELEASE_FLAGS "-fcf-protection=none")
  list(APPEND RELEASE_FLAGS "-fno-asynchronous-unwind-tables")
endif()

if(TYCHO_FP_MODE STREQUAL "STRICT")
    # On non-Windows, the compiler already defaults to strict FP semantics,
    # so no flags are needed.  On Windows, /fp:strict must be set explicitly.
    if(WIN32)
        list(APPEND FP_FLAGS "/fp:strict")
    endif()
elseif(TYCHO_FP_MODE STREQUAL "SAFER_FAST")
    if(WIN32)
        list(APPEND FP_FLAGS "/fp:fast")
        list(APPEND FP_FLAGS "${_TYCHO_WINDOWS_CLANG_NO_FINITE_MATH_ONLY_FLAG}")
    else()
        list(APPEND FP_FLAGS "-ffast-math")
        list(APPEND FP_FLAGS "-fno-finite-math-only")
    endif()
elseif(TYCHO_FP_MODE STREQUAL "FAST")
    if(WIN32)
        list(APPEND FP_FLAGS "/fp:fast")
    else()
        list(APPEND FP_FLAGS "-ffast-math")
    endif()
endif()

list(APPEND RELEASE_FLAGS ${SIMD_FLAGS})
list(APPEND RELEASE_FLAGS ${FP_FLAGS})
 

# Combine Flags
set(COMPILE_FLAGS ${COMMON_FLAGS})
# Wrap each flag individually in a genex to avoid broken genex fragments.
# Embedding a semicolon-separated list inside a single genex breaks the
# generator expression when CMake stores it as a list property (e.g.,
# COMPILE_OPTIONS), because semicolons inside the genex become list separators.
# The REUSE_FROM PCH mechanism then evaluates these fragments incorrectly.
foreach(_flag IN LISTS RELEASE_FLAGS)
    list(APPEND COMPILE_FLAGS "$<$<OR:$<CONFIG:RELEASE>,$<CONFIG:RELWITHDEBINFO>>:${_flag}>")
endforeach()
foreach(_flag IN LISTS DEBUG_FLAGS)
    list(APPEND COMPILE_FLAGS "$<$<OR:$<CONFIG:DEBUG>,$<CONFIG:RELWITHDEBINFO>>:${_flag}>")
endforeach()

if(COMPILE_TIME_TRACE AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options(-ftime-trace)
    message(STATUS "Compile-time tracing enabled (-ftime-trace)")
endif()



################################################################################
################## TYCHO PREPROCCESSOR MACROS  #################################
################################################################################

add_definitions(-DEIGEN_INITIALIZE_MATRICES_BY_ZERO) ##NEVER EVER GET RID OF THIS EVER
add_definitions(-DEIGEN_DONT_PARALLELIZE) ##NEVER EVER GET RID OF THIS EVER


add_definitions(-DTYCHO_DEFAULT_QP_THREADS=8) ## no spaces between = !!!! otherwise it is set to one

###  initial size of asset arena memory allocator in number of doubles/superscalars
add_definitions(-DTYCHO_DEFAULT_ARENA_SIZE=256) ## no spaces between = !!!! otherwise it is set to one

### Arena memory allocator is always on (TYCHO_MEMORYMAN removed — BumpAllocator rewrite)

add_definitions(-DTYCHO_COMPILERVERSION="${CMAKE_CXX_COMPILER_VERSION}")
add_definitions(-DTYCHO_VERSIONSTRING="${CMAKE_PROJECT_VERSION}")
add_definitions(-DTYCHO_COMPILERSTRING="${CMAKE_CXX_COMPILER_ID}")
add_definitions(-DTYCHO_COMPILERVERSION="${CMAKE_CXX_COMPILER_VERSION}")
add_definitions(-DTYCHO_OS="${CMAKE_SYSTEM_NAME}")
add_definitions(-DTYCHO_OSVERSION="${CMAKE_SYSTEM_VERSION}")
add_definitions(-DFMT_HEADER_ONLY)
add_definitions(-DFMT_USE_LOCALE=0)

if(WIN32)
## See : https://developercommunity.visualstudio.com/t/Access-violation-in-_Thrd_yield-after-up/10664660#T-N10669129-N10678728
add_definitions(-D_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
endif()



################################################################################
###################### Find and set our dependencies ###########################
################################################################################

#if(APPLE)
#  include(FindPythonEnv)
#endif()

# Set up submodule dependencies
if(TYCHO_USE_SYSTEM_EIGEN)
    find_package(Eigen3 3.4 REQUIRED)
    # Global include so internal targets (pch, psiopt, etc.) see Eigen headers.
    # Prefer modern target-based approach; fall back to legacy variable.
    if(TARGET Eigen3::Eigen)
        get_target_property(_eigen3_inc Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
        if(_eigen3_inc)
            include_directories(${_eigen3_inc})
        else()
            message(FATAL_ERROR
                "Eigen3::Eigen target exists but has no INTERFACE_INCLUDE_DIRECTORIES. "
                "Set EIGEN3_INCLUDE_DIR manually or use -DTYCHO_USE_SYSTEM_EIGEN=OFF.")
        endif()
    else()
        if(NOT EIGEN3_INCLUDE_DIR)
            message(FATAL_ERROR
                "find_package(Eigen3) succeeded but EIGEN3_INCLUDE_DIR is not set.")
        endif()
        include_directories(${EIGEN3_INCLUDE_DIR})
    endif()
    message(STATUS "Using system Eigen3: ${Eigen3_DIR}")
else()
    if(NOT TARGET Eigen3::Eigen)
        add_library(Eigen3::Eigen INTERFACE IMPORTED)
        target_include_directories(Eigen3::Eigen INTERFACE
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/dep/eigen>
        )
    endif()
    # Global include so internal targets (pch, psiopt, etc.) see Eigen headers
    include_directories(dep/eigen)
    message(STATUS "Using bundled Eigen from dep/eigen")
endif()
include_directories(dep/fmt/include)
include_directories(dep/autodiff)

# fmt is used by core libraries (utils) so it must always be available.
add_subdirectory(dep/fmt)

if(ENABLE_PYTHON_BINDINGS)
    find_package(Python ${PYVERSION_EXACT} COMPONENTS Interpreter Development REQUIRED)
    add_subdirectory(dep/nanobind)
endif()

# Set up external dependencies
find_package(Threads REQUIRED)

if(UNIX AND NOT APPLE)
  # find_package(OpenMP REQUIRED)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    list(APPEND COMPILE_FLAGS "-fopenmp=libiomp5")
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GCC")
    list(APPEND COMPILE_FLAGS "-fopenmp")
  endif()
elseif(APPLE)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    if(CMAKE_CXX_COMPILER_ID MATCHES "AppleClang")
      # find_package(OpenMP) fails with AppleClang, so detect libomp manually.
      find_path(TYCHO_OMP_INCLUDE_DIR omp.h
          PATHS
              /opt/homebrew/opt/libomp/include   # Apple Silicon Homebrew
              /usr/local/opt/libomp/include      # Intel Mac Homebrew
          REQUIRED
      )
      find_library(TYCHO_OMP_LIBRARY omp
          PATHS
              /opt/homebrew/opt/libomp/lib
              /usr/local/opt/libomp/lib
          REQUIRED
      )
      include_directories("${TYCHO_OMP_INCLUDE_DIR}")
      get_filename_component(_tycho_omp_lib_dir "${TYCHO_OMP_LIBRARY}" DIRECTORY)
      link_directories("${_tycho_omp_lib_dir}")
      list(APPEND COMPILE_FLAGS "-Xclang")
      list(APPEND COMPILE_FLAGS "-fopenmp")
      list(APPEND OpenMP_CXX_FLAGS "-lomp")
    else()
      # With homebrew LLVM clang compiler on Mac, this seems to work well for 
      # finding homebrew OpenMP 
      find_package(OpenMP REQUIRED)

      list(APPEND COMPILE_FLAGS "-fopenmp")
      list(APPEND OpenMP_CXX_FLAGS "-lomp")
    endif()
  endif()
endif()

# Binding-specific compile flags: -Os (nanobind best practice) instead of -O3.
# Computed AFTER all platform flags (OpenMP etc.) are appended to COMPILE_FLAGS.
# Core libs (psiopt, utils, etc.) use PRIVATE ${COMPILE_FLAGS} so their -O3 does
# not propagate into _tychopy TUs; pch_bindings and _tychopy both use only
# BINDING_COMPILE_FLAGS, keeping PCH and TU flags in sync.
if(OPTIMIZE_BINDINGS)
    set(BINDING_COMPILE_FLAGS ${COMPILE_FLAGS})
else()
    string(REPLACE "-O3" "-Os" BINDING_COMPILE_FLAGS "${COMPILE_FLAGS}")
    if(WIN32)
        string(REPLACE "/O2" "/Os" BINDING_COMPILE_FLAGS "${BINDING_COMPILE_FLAGS}")
    endif()
endif()

if(APPLE)
  find_package(AccelerateSparse REQUIRED)
  add_compile_definitions(USE_ACCELERATE_SPARSE)
else()
  find_package(MKL)
  if(NOT MKL_FOUND)
      message(FATAL_ERROR
          "Tycho requires Intel MKL. Set MKLROOT to the MKL installation directory.\n"
          "  See CLAUDE.md for installation instructions.")
  endif()
  # Validate that no individual MKL library is NOTFOUND. FindMKL.cmake does not
  # include MKL_OMP_LIBRARY in FIND_PACKAGE_HANDLE_STANDARD_ARGS, so MKL_FOUND
  # can be TRUE while MKL_LIBRARIES contains a NOTFOUND entry (libiomp5).
  foreach(_mkl_lib IN LISTS MKL_LIBRARIES)
      if("${_mkl_lib}" MATCHES "NOTFOUND")
          message(FATAL_ERROR
              "MKL library component not found: ${_mkl_lib}\n"
              "Your MKL installation may be incomplete. "
              "Ensure MKLROOT points to a complete MKL installation.")
      endif()
  endforeach()
endif()

if(ENABLE_PYTHON_BINDINGS)
    find_package(Python ${PYVERSION_EXACT} REQUIRED COMPONENTS Interpreter Development)
endif()

# Set dependency variables
set(INCLUDE_DIRS ${MKL_INCLUDE_DIRS})
# On Linux, static MKL archives have circular dependencies and must be wrapped
# in --start-group/--end-group so the linker rescans them.
if(UNIX AND NOT APPLE)
  set(LINK_LIBS -Wl,--start-group ${MKL_LIBRARIES} -Wl,--end-group Threads::Threads ${CMAKE_DL_LIBS})
else()
  set(LINK_LIBS ${MKL_LIBRARIES} Threads::Threads ${CMAKE_DL_LIBS})
endif()

# Handle OpenMP for Apple
if (APPLE)
  if(CMAKE_CXX_COMPILER_ID MATCHES "AppleClang")
    list(APPEND LINK_LIBS ${OpenMP_CXX_FLAGS})
  else()
    list(APPEND LINK_LIBS OpenMP::OpenMP_CXX)
  endif()
endif()

if(ENABLE_PYTHON_BINDINGS)
    # Handle Python executable variable
    if (NOT DEFINED PYTHON_EXECUTABLE)
        set(PYTHON_EXECUTABLE ${Python_EXECUTABLE})
    endif()
    if(DEFINED Python_INCLUDE_DIRS)
        list(APPEND INCLUDE_DIRS ${Python_INCLUDE_DIRS})
    elseif(DEFINED PYTHON_INCLUDE_DIRS)
        list(APPEND INCLUDE_DIRS ${PYTHON_INCLUDE_DIRS})
    endif()
    # NOTE: Python_LIBRARIES is intentionally NOT added to LINK_LIBS.
    # LINK_LIBS flows into the tycho::tycho INTERFACE target, which must
    # remain pure C++ for find_package/FetchContent consumers.
    # nanobind handles Python symbol resolution on all platforms:
    #   - Linux: MODULE libraries resolve CPython symbols at runtime
    #   - Windows: nanobind-static links Python::Module
    #   - macOS: -undefined dynamic_lookup (Python symbols bound at load time)

endif()

if(UNIX)
  list(APPEND LINK_LIBS m)
endif()



################################################################################
################ Linker Flags ##################################################


if(UNIX AND NOT APPLE)  # Linux (MKL group wrapping is in LINK_LIBS)
  list(APPEND LINKER_FLAGS ${OpenMP_CXX_FLAGS} -Wl,--no-undefined)
elseif(APPLE)  # Apple Dynamic and Static Linking
  list(APPEND LINKER_FLAGS ${OpenMP_CXX_FLAGS})
  list(APPEND LINKER_FLAGS "-Wl,-bind_at_load")
  list(APPEND LINKER_FLAGS "-Wl,-undefined,dynamic_lookup")
else()  # Windows Static Linking
  list(APPEND LINKER_FLAGS "${OpenMP_CXX_FLAGS}")
endif()

###########################################################
######## Find Correct Python Install Location  ############

if(ENABLE_PYTHON_BINDINGS)
    if(NOT PIP_INSTALL)
      if(NOT PYTHON_LOCAL_INSTALL_DIR)
        if(WIN32)
          set(PYTHON_LOCAL_INSTALL_DIR ${Python_SITEARCH})
        else()
          execute_process ( COMMAND ${PYTHON_EXECUTABLE} -m site --user-site
          OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)
          set(PYTHON_LOCAL_INSTALL_DIR ${PYTHON_SITE_PACKAGES})
        endif()
      endif()

      if(PYTHON_LOCAL_INSTALL_DIR)
        message(STATUS "Tycho will be installed to " ${PYTHON_LOCAL_INSTALL_DIR})
      else()
        message(STATUS "Could not find appropriate install directory")
      endif()
    endif()
endif()

message(STATUS "Floating-point mode: ${TYCHO_FP_MODE}")
if(TYCHO_FP_MODE STREQUAL "STRICT")
    message(STATUS "IEEE 754 compliance: STRICT (no fast-math relaxations)")
elseif(TYCHO_FP_MODE STREQUAL "SAFER_FAST")
    message(STATUS "IEEE 754 compliance: SAFER_FAST (fast-math except finite-math-only)")
else()
    message(STATUS "IEEE 754 compliance: FAST (full fast-math)")
endif()

################################################################################
# Code Directories
include(GNUInstallDirs)
add_subdirectory(src)

add_subdirectory(extensions)


# Documentation
# READ: pip install sphinx,breathe,furo before trying to build docs

if(BUILD_SPHINX_DOCS)
  add_subdirectory(doc)
endif()


if(BUILD_CPP_EXAMPLES)
    add_subdirectory(examples/cpp_examples)
endif()

if(BUILD_CPP_BENCHMARKS)
    add_subdirectory(bench/cpp)
endif()

if(BUILD_CPP_BENCHMARKS_LEGACY)
    add_subdirectory(bench/legacy)
endif()

if(BUILD_CPP_TESTS)
    enable_testing()
    add_subdirectory(tests/cpp)
endif()

# Formatting
include(cmake/clang-format.cmake)

################################################################################
# Install rules — headers, CMake package config, and Find modules

install(DIRECTORY include/tycho
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h"
)

if(NOT TYCHO_USE_SYSTEM_EIGEN)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/dep/eigen/Eigen/Core")
        message(FATAL_ERROR
            "Bundled Eigen not found at dep/eigen/. "
            "Run: git submodule update --init dep/eigen")
    endif()
    install(DIRECTORY dep/eigen/Eigen
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
    install(DIRECTORY dep/eigen/unsupported
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
endif()

# Install fmt headers (header-only, required by tycho public API)
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/dep/fmt/include/fmt/format.h")
    message(FATAL_ERROR
        "Bundled fmt not found at dep/fmt/. "
        "Run: git submodule update --init dep/fmt")
endif()
install(DIRECTORY dep/fmt/include/fmt
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install autodiff headers (header-only, required by tycho public API)
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/dep/autodiff/autodiff/forward/dual.hpp")
    message(FATAL_ERROR
        "Bundled autodiff not found at dep/autodiff/. "
        "Run: git submodule update --init dep/autodiff")
endif()
install(DIRECTORY dep/autodiff/autodiff
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp"
    PATTERN "tests" EXCLUDE
    PATTERN "pybind11" EXCLUDE
)

# CMake package config for find_package(tycho)
include(CMakePackageConfigHelpers)

# Variables substituted into tychoConfig.cmake.in
if(APPLE)
    set(TYCHO_SPARSE_SOLVER "AccelerateSparse")
    # macOS needs libomp linked explicitly. On Linux/Windows, MKL_LIBRARIES
    # includes Intel's libiomp5 which satisfies OpenMP runtime linkage.
    set(TYCHO_NEEDS_OPENMP ON)
    # Derive libomp prefix from the library found at configure time.
    # TYCHO_OMP_LIBRARY is set by the AppleClang find_library() block above;
    # for Homebrew LLVM, find_package(OpenMP) handles it, but the config
    # template still needs a hint path for consumer-side re-detection.
    if(TYCHO_OMP_LIBRARY)
        get_filename_component(_tycho_omp_lib_dir "${TYCHO_OMP_LIBRARY}" DIRECTORY)
        get_filename_component(TYCHO_LIBOMP_DIR "${_tycho_omp_lib_dir}" DIRECTORY)
    else()
        # Homebrew LLVM fallback: OpenMP found via find_package; derive
        # hint path for consumer-side libomp re-detection.
        if(EXISTS "/opt/homebrew/opt/libomp/lib")
            set(TYCHO_LIBOMP_DIR "/opt/homebrew/opt/libomp")
        elseif(EXISTS "/usr/local/opt/libomp/lib")
            set(TYCHO_LIBOMP_DIR "/usr/local/opt/libomp")
        else()
            set(TYCHO_LIBOMP_DIR "/opt/homebrew/opt/libomp")
            message(WARNING
                "tycho: could not detect libomp prefix for consumer hint. "
                "Defaulting to ${TYCHO_LIBOMP_DIR}. Consumers may need to "
                "install libomp: brew install libomp")
        endif()
    endif()
else()
    set(TYCHO_SPARSE_SOLVER "MKL")
    set(TYCHO_NEEDS_OPENMP OFF)
endif()

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/tychoConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/tychoConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/tychoConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tycho
)

install(EXPORT tychoTargets
    FILE tychoTargets.cmake
    NAMESPACE tycho::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tycho
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/tychoConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/tychoConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tycho
)

# Install Find modules so consumers can re-find transitive dependencies
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindMKL.cmake"
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindAccelerateSparse.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tycho
)
