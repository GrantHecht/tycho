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
# Ninja job pool for memory-heavy translation units.
# Many binding, test, example, and core library TUs consume 3-8 GB RAM each.  Limiting
# concurrent heavy compiles prevents OOM while still allowing light TUs to
# build in parallel at the full -j level.
#
# TYCHO_HEAVY_COMPILE_JOBS controls the pool depth (max concurrent heavy TUs).
# Default: auto-detected from system RAM (1 per 10 GB, minimum 1).
# Override: cmake -DTYCHO_HEAVY_COMPILE_JOBS=<N> ...
################################################################################
if(NOT DEFINED TYCHO_HEAVY_COMPILE_JOBS)
    cmake_host_system_information(RESULT _TYCHO_TOTAL_RAM_MB QUERY TOTAL_PHYSICAL_MEMORY)
    math(EXPR _TYCHO_POOL_DEPTH "${_TYCHO_TOTAL_RAM_MB} / 10240")
    if(_TYCHO_POOL_DEPTH LESS 1)
        set(_TYCHO_POOL_DEPTH 1)
    endif()
    set(TYCHO_HEAVY_COMPILE_JOBS ${_TYCHO_POOL_DEPTH} CACHE STRING
        "Max concurrent compilations of memory-heavy TUs (auto: 1 per 10 GB RAM)")
endif()
if(TYCHO_HEAVY_COMPILE_JOBS LESS 1)
    message(WARNING "TYCHO_HEAVY_COMPILE_JOBS=${TYCHO_HEAVY_COMPILE_JOBS} is invalid (must be >= 1), resetting to 1.")
    set(TYCHO_HEAVY_COMPILE_JOBS 1 CACHE STRING
        "Max concurrent compilations of memory-heavy TUs" FORCE)
endif()
message(STATUS "Heavy compile pool depth: ${TYCHO_HEAVY_COMPILE_JOBS} (override with -DTYCHO_HEAVY_COMPILE_JOBS=N)")
set_property(GLOBAL APPEND PROPERTY JOB_POOLS heavy_compile=${TYCHO_HEAVY_COMPILE_JOBS})
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
option(BUILD_CPP_EXAMPLES "Build standalone C++ example executables" OFF)
option(BUILD_CPP_BENCHMARKS "Build C++ benchmarks using Google Benchmark (requires FetchContent)" OFF)
option(BUILD_CPP_BENCHMARKS_LEGACY "Build legacy hand-rolled benchmark executables" OFF)
option(BUILD_CPP_TESTS "Build C++ unit tests (requires Google Test via FetchContent)" OFF)
option(OPTIMIZE_BINDINGS "Compile Python binding code at -O3 instead of -Os (nanobind default)" OFF)
option(COMPILE_TIME_TRACE "Enable Clang -ftime-trace for build profiling (Clang only)" OFF)
option(ENABLE_ENZYME_AD
    "Enable experimental Enzyme automatic differentiation (requires Clang + Enzyme plugin)"
    OFF)
option(BUILD_SPHINX_DOCS
    "Build Sphinx documentation (requires sphinx, breathe, doxygen, pydata-sphinx-theme)"
    OFF)

if(ENABLE_ENZYME_AD)
    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        message(FATAL_ERROR
            "ENABLE_ENZYME_AD requires Clang. Got CMAKE_CXX_COMPILER_ID="
            "${CMAKE_CXX_COMPILER_ID}. Override -DCMAKE_CXX_COMPILER=/path/to/clang++ "
            "(typically a custom-built LLVM installation matching your Enzyme plugin).")
    endif()

    string(REGEX MATCH "^([0-9]+)" _tycho_clang_major "${CMAKE_CXX_COMPILER_VERSION}")
    set(_tycho_clang_major "${CMAKE_MATCH_1}")

    find_package(Enzyme CONFIG REQUIRED)
    if(NOT Enzyme_LLVM_VERSION_MAJOR)
        message(FATAL_ERROR
            "find_package(Enzyme) succeeded but Enzyme_LLVM_VERSION_MAJOR is unset. "
            "Check that your Enzyme installation provides EnzymeConfig.cmake "
            "(this variable is set by the upstream config file).")
    endif()

    if(NOT "${Enzyme_LLVM_VERSION_MAJOR}" STREQUAL "${_tycho_clang_major}")
        message(FATAL_ERROR
            "Enzyme was built against LLVM ${Enzyme_LLVM_VERSION_MAJOR}, "
            "but compiler is clang ${_tycho_clang_major}. Versions must match. "
            "Rebuild Enzyme against the active Clang or override "
            "-DCMAKE_CXX_COMPILER to a matching clang.")
    endif()

    set(_tycho_enzyme_target "ClangEnzyme-${Enzyme_LLVM_VERSION_MAJOR}")
    if(NOT TARGET ${_tycho_enzyme_target})
        message(FATAL_ERROR
            "Imported target ${_tycho_enzyme_target} not defined by EnzymeConfig.cmake. "
            "Reinstall Enzyme or check CMAKE_PREFIX_PATH.")
    endif()
    get_target_property(_tycho_enzyme_plugin_path ${_tycho_enzyme_target} IMPORTED_LOCATION)
    if(NOT _tycho_enzyme_plugin_path)
        # EnzymeConfig.cmake only sets per-config IMPORTED_LOCATION_<CONFIG>, not the
        # generic IMPORTED_LOCATION. Try the imported configurations in turn.
        get_target_property(_tycho_enzyme_imported_configs
            ${_tycho_enzyme_target} IMPORTED_CONFIGURATIONS)
        if(_tycho_enzyme_imported_configs)
            foreach(_tycho_cfg IN LISTS _tycho_enzyme_imported_configs)
                get_target_property(_tycho_enzyme_plugin_path
                    ${_tycho_enzyme_target} IMPORTED_LOCATION_${_tycho_cfg})
                if(_tycho_enzyme_plugin_path)
                    break()
                endif()
            endforeach()
        endif()
    endif()
    if(NOT _tycho_enzyme_plugin_path)
        message(FATAL_ERROR
            "${_tycho_enzyme_target} has no IMPORTED_LOCATION (or per-config "
            "IMPORTED_LOCATION_<CONFIG>). Enzyme installation appears incomplete.")
    endif()

    set(TYCHO_ENZYME_COMPILE_FLAGS "-fplugin=${_tycho_enzyme_plugin_path}"
        CACHE STRING "Enzyme plugin compile flag(s); applied to Enzyme-instantiating targets only" FORCE)
    add_compile_definitions(TYCHO_HAS_ENZYME_AD)

    message(STATUS "Enzyme AD: enabled (LLVM ${Enzyme_LLVM_VERSION_MAJOR}, plugin ${_tycho_enzyme_plugin_path})")

    include("${PROJECT_SOURCE_DIR}/cmake/EnzymeTryCompile.cmake")
    tycho_enzyme_try_compile_fwddiff()

    # Phase 2: probe the FoR Hessian-via-nested-Enzyme strategy.  ForwardOverForward
    # is archived research scaffolding — the helpers live in dense_enzyme.h for
    # future revival but are no longer cmake-selectable.  Failure here is fatal
    # (the user can still pair <EnzymeAD, FDiffFwd> with ENABLE_ENZYME_AD=ON or
    # set -DENABLE_ENZYME_AD=OFF if Enzyme Hessians are not needed at all).
    tycho_enzyme_try_compile_nested_for(TYCHO_ENZYME_HAS_FOR)
    if(NOT TYCHO_ENZYME_HAS_FOR)
        message(FATAL_ERROR
            "Enzyme nested-FoR smoke test failed. Disable Enzyme Hessians by "
            "selecting <EnzymeAD, FDiffFwd> template parameters or set "
            "-DENABLE_ENZYME_AD=OFF.")
    endif()

    # Phase 2 nested-Enzyme strategy is fixed at ForwardOverReverse; FoF was
    # archived (see dense_enzyme.h archive block for revival conditions).
    add_compile_definitions(TYCHO_ENZYME_HESSIAN_STRATEGY_ForwardOverReverse)

    # Phase 3 batched forward-mode Jacobian width.  Drives the chunked
    # __enzyme_fwddiff(..., enzyme_width, W, enzyme_dupv, ...) loop in the
    # scalar Jacobian helper.  W=1 disables batching and falls back to the
    # Phase 1 single-tangent loop.  W=4 / W=8 trade per-call Enzyme overhead
    # for parallel tangent propagation; see CLAUDE.md "Rule of thumb" +
    # bench/cpp/bench_enzyme.cpp for the W=4 vs W=8 head-to-head.
    set(TYCHO_ENZYME_BATCH_WIDTH "4" CACHE STRING
        "Phase 3 batch width for Enzyme forward-mode Jacobian (1, 4, or 8)")
    set_property(CACHE TYCHO_ENZYME_BATCH_WIDTH PROPERTY STRINGS "1" "4" "8")
    if(NOT TYCHO_ENZYME_BATCH_WIDTH MATCHES "^(1|4|8)$")
        message(FATAL_ERROR
            "TYCHO_ENZYME_BATCH_WIDTH must be 1, 4, or 8 (got: "
            "${TYCHO_ENZYME_BATCH_WIDTH}).  STRINGS is a UI hint, not a "
            "constraint; this regex pins the allowed values.")
    endif()
    add_compile_definitions(TYCHO_ENZYME_BATCH_WIDTH=${TYCHO_ENZYME_BATCH_WIDTH})
    message(STATUS
        "Enzyme AD: TYCHO_ENZYME_BATCH_WIDTH=${TYCHO_ENZYME_BATCH_WIDTH} — "
        "VFs with compile-time IR < ${TYCHO_ENZYME_BATCH_WIDTH} silently "
        "use the single-tangent tail-loop fallback.")

    # Phase 6: direct-SIMD EnzymeAD Hessian for is_vectorizable=true VFs at
    # SuperScalar Scalar.  When OFF, falls through to the Phase 5a
    # scalarize-per-lane fallback so we can A/B in the same binary.
    option(TYCHO_ENZYME_SIMD_HESSIAN
        "Phase 6: direct-SIMD EnzymeAD Hessian for Vectorizable VFs"
        ON)
    if(TYCHO_ENZYME_SIMD_HESSIAN)
        add_compile_definitions(TYCHO_ENZYME_SIMD_HESSIAN_ENABLED)
    endif()
    message(STATUS
        "Enzyme AD: Phase 6 SIMD Hessian "
        "${TYCHO_ENZYME_SIMD_HESSIAN} (TYCHO_ENZYME_SIMD_HESSIAN)")
endif()

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


################################################################################
# Sanitizers (opt-in; off by default)
#
# Two mutually exclusive modes:
#
# TYCHO_ENABLE_SANITIZERS=ON
#     AddressSanitizer + UndefinedBehaviorSanitizer. Catches heap/stack
#     overflow, use-after-free, leaks, and common UB (null derefs, signed
#     overflow, misaligned loads, ...). Does NOT catch uninitialized reads.
#     Usage from Python:
#       LD_PRELOAD=/usr/lib/clang/<ver>/lib/.../libclang_rt.asan.so \
#         conda run -n tycho python <script.py>
#
# TYCHO_ENABLE_MSAN=ON (clang only)
#     MemorySanitizer. Catches reads of uninitialized memory. Much higher
#     false-positive rate because any write by a non-instrumented library
#     (MKL, libiomp5, libstdc++ on Fedora) "poisons" its outputs. Use with
#     MSAN_OPTIONS=exitcode=0:halt_on_error=0 and grep tycho source paths
#     out of the noise. The MSan runtime is linked statically — no
#     LD_PRELOAD needed, just run the sanitized binary directly.
#
# You cannot enable both at the same time (ASan and MSan are mutually
# exclusive in the clang runtime).
################################################################################
set(TYCHO_ENABLE_SANITIZERS OFF CACHE BOOL
    "Enable AddressSanitizer + UndefinedBehaviorSanitizer on tycho targets")
set(TYCHO_ENABLE_MSAN OFF CACHE BOOL
    "Enable MemorySanitizer on tycho targets (clang only; mutually exclusive with TYCHO_ENABLE_SANITIZERS)")
set(SANITIZER_COMPILE_FLAGS "")
if(TYCHO_ENABLE_SANITIZERS AND TYCHO_ENABLE_MSAN)
    message(FATAL_ERROR
        "TYCHO_ENABLE_SANITIZERS and TYCHO_ENABLE_MSAN are mutually exclusive. Pick one.")
endif()
if(TYCHO_ENABLE_SANITIZERS)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        list(APPEND SANITIZER_COMPILE_FLAGS
             "-fsanitize=address,undefined"
             "-fno-omit-frame-pointer"
             # Default to recoverable UBSan so the process continues past the
             # first diagnostic; use UBSAN_OPTIONS=halt_on_error=1 at runtime
             # to turn this back on.
             "-fsanitize-recover=all"
             "-g")
        # Apply link flags globally so _tychopy and any executable target
        # (tests, examples) link against libasan/libubsan correctly.
        add_link_options(-fsanitize=address,undefined)
        # On clang, force the dynamic sanitizer runtime so _tychopy.so can be
        # loaded into Python via LD_PRELOAD=$(clang -print-file-name=
        # libclang_rt.asan-x86_64.so). Without this, clang links libclang_rt.asan.a
        # statically, which produces duplicate-symbol errors at dlopen time.
        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            list(APPEND SANITIZER_COMPILE_FLAGS "-shared-libsan")
            add_link_options(-shared-libsan)
        endif()
        message(STATUS "Sanitizers ENABLED (AddressSanitizer + UBSan)")
    else()
        message(WARNING "TYCHO_ENABLE_SANITIZERS: unsupported compiler, ignoring")
    endif()
endif()
if(TYCHO_ENABLE_MSAN)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        list(APPEND SANITIZER_COMPILE_FLAGS
             "-fsanitize=memory"
             "-fsanitize-memory-track-origins=2"
             "-fno-omit-frame-pointer"
             "-fsanitize-recover=all"
             "-g")
        add_link_options(-fsanitize=memory)
        # Note: Fedora compiler-rt only ships static libclang_rt.msan.a, so we
        # cannot LD_PRELOAD an MSan runtime into an un-instrumented Python
        # interpreter the way we do for ASan. To use MSan you must build a
        # standalone C++ binary (e.g. a cpp_examples/ reproducer) that links
        # the MSan runtime statically and exercises the suspect code path
        # directly. Loading _tychopy.so into Python under MSan will fail with
        # unresolved __msan_* symbols — this is expected.
        message(STATUS "MemorySanitizer ENABLED (clang, track-origins=2) — standalone binaries only")
    else()
        message(FATAL_ERROR "TYCHO_ENABLE_MSAN requires clang (got ${CMAKE_CXX_COMPILER_ID})")
    endif()
endif()

################################################################################
# Auto-init trivial stack/aggregate variables (debug aid)
#
# TYCHO_AUTO_INIT_STACK=ON fills uninitialized trivial stack variables with
# 0xAAAA... (clang/gcc 12+). This makes reads of uninitialized memory produce
# deterministic poison values instead of whatever the compiler happens to
# leave on the stack — handy for reproducing UB that ASan misses.
#
# Cheap to enable (no runtime cost at -O3 for values that get optimized out,
# small cost for values that actually live on the stack). Off by default to
# match production behavior, but safe to ship.
################################################################################
set(TYCHO_AUTO_INIT_STACK OFF CACHE BOOL
    "Fill trivial uninitialized stack variables with 0xAA poison pattern (debug aid)")
if(TYCHO_AUTO_INIT_STACK)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        list(APPEND SANITIZER_COMPILE_FLAGS "-ftrivial-auto-var-init=pattern")
        message(STATUS "Trivial auto-var init ENABLED (pattern 0xAA)")
    else()
        message(WARNING "TYCHO_AUTO_INIT_STACK: unsupported compiler, ignoring")
    endif()
endif()

# Combine Flags
set(COMPILE_FLAGS ${COMMON_FLAGS})
list(APPEND COMPILE_FLAGS ${SANITIZER_COMPILE_FLAGS})
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
# pocketfft (header-only, BSD-3) — provides DCT-I for Chebyshev interpolation.
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/dep/pocketfft/pocketfft_hdronly.h")
    message(FATAL_ERROR
        "Bundled pocketfft not found at dep/pocketfft/. "
        "Run: git submodule update --init dep/pocketfft")
endif()
include_directories(dep/pocketfft)
# pocketfft threading: construction-time DCT-I is safe to run multithreaded
# (pocketfft threads only run during interpolant construction, never during
# solve-time evaluation). Thread count is exposed via ChebTable::from_values
# nthreads parameter (default 1). No global compile-time disable needed.

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
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GCC|GNU")
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
    # SKBUILD is set by scikit-build-core when driving CMake via `pip wheel .`
    # or `pip install .`; pip handles its own install layout, so the dev-build
    # POST_BUILD copy in src/bindings/CMakeLists.txt must NOT fire.
    # PIP_INSTALL is the legacy opt-out flag (never auto-set; honored for
    # callers that pass -DPIP_INSTALL=ON explicitly).
    if(NOT SKBUILD AND NOT PIP_INSTALL)
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

# add_subdirectory(extensions)
# Unbuilt as a side-effect of cr3bp_model.h removal (Tycho_Extensions.h
# referenced the deleted legacy CR3BP type). The extension mechanism
# itself is under review.


# Documentation
if(BUILD_SPHINX_DOCS)
  find_package(Doxygen REQUIRED)
  # Sphinx is invoked via `${Python_EXECUTABLE} -m sphinx`; no find_package
  # needed — relies on docs/requirements-docs.txt being installed.
  add_subdirectory(docs)
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
