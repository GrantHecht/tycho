cmake_minimum_required(VERSION 3.16)
project(Tycho VERSION 0.1.0 LANGUAGES CXX)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

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

option(ENABLE_PYTHON_BINDINGS "Build Python bindings (_tycho pybind11 module)" ON)
option(BUILD_CPP_EXAMPLES "Build standalone C++ example executables" ON)
option(BUILD_CPP_BENCHMARKS "Build C++ benchmarks using Google Benchmark (requires FetchContent)" OFF)
option(BUILD_CPP_BENCHMARKS_LEGACY "Build legacy hand-rolled benchmark executables" OFF)
option(BUILD_CPP_TESTS "Build C++ unit tests (requires Google Test via FetchContent)" OFF)
option(OPTIMIZE_BINDINGS "Compile Python binding code at -O3 instead of -Os (nanobind default)" OFF)
option(COMPILE_TIME_TRACE "Enable Clang -ftime-trace for build profiling (Clang only)" OFF)

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
add_definitions(-DTYCHO_DEFAULT_FUNC_THREADS=16) ## no spaces between = !!!! otherwise it is set to one

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
include_directories(dep/eigen)
include_directories(dep/fmt/include)
include_directories(dep/autodiff)
include_directories(dep/thread-pool/include)

if(ENABLE_PYTHON_BINDINGS)
    find_package(Python ${PYVERSION_EXACT} COMPONENTS Interpreter Development REQUIRED)
    add_subdirectory(dep)
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
      # fina_package(OpenMP REQUIRED) will fail with AppleClang so 
      # we need this to point towards the homebrew libomp
      include_directories("/opt/homebrew/Cellar/libomp/22.1.0/include/")
      link_directories("/opt/homebrew/Cellar/libomp/22.1.0/lib/")
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
# not propagate into _tycho TUs; pch_bindings and _tycho both use only
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
  find_package(MKL REQUIRED)
endif()

if(ENABLE_PYTHON_BINDINGS)
    find_package(Python ${PYVERSION_EXACT} REQUIRED COMPONENTS Interpreter Development)
endif()

# Set dependency variables
set(INCLUDE_DIRS ${MKL_INCLUDE_DIRS})
set(LINK_LIBS ${MKL_LIBRARIES} Threads::Threads ${CMAKE_DL_LIBS})

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
    if (NOT APPLE)
        # Don't directly link with python library on mac and instead rely on -undefined dynamic_lookup
        # to bind to the python symbols at runtime from the interpreter
        if(DEFINED Python_LIBRARIES)
            list(APPEND LINK_LIBS ${Python_LIBRARIES})
        elseif(DEFINED PYTHON_LIBRARIES)
            list(APPEND LINK_LIBS ${PYTHON_LIBRARIES})
        endif()
    endif()

endif()

if(UNIX)
  list(APPEND LINK_LIBS m)
endif()



################################################################################
################ Linker Flags ##################################################


if(BUILD_SHARED_LIBS AND NOT APPLE AND NOT WIN32)  # Windows and Linux Dynamic Linking
  list(APPEND LINKER_FLAGS "${OpenMP_CXX_FLAGS} -Wl,--no-undefined -Wl,--start-group ${MKL_LIBRARIES_LIST} -Wl, --end-group")
elseif(UNIX AND NOT APPLE)  # Linux Static Linking
  list(APPEND LINKER_FLAGS "${OpenMP_CXX_FLAGS} -Wl,--no-undefined -Wl,--start-group ${MKL_LIBRARIES_LIST} -Wl,--end-group")
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

message(STATUS "CMAKE DIR LOC" ${CMAKE_CURRENT_SOURCE_DIR})
message(STATUS ${OpenMP_CXX_FLAGS})
message(STATUS ${OpenMP_CXX_LIBRARIES})
message(STATUS ${PYTHON_EXECUTABLE})

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
