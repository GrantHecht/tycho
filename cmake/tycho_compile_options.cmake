################################################################################
# tycho_compile_options()
#
# Assembles Tycho's shared C++ compile-flag set: SIMD architecture flags,
# release/debug flags, LTO/IPO settings, floating-point mode (derived from
# TYCHO_FP_MODE), sanitizer instrumentation, and the combined COMPILE_FLAGS
# list consumed by targets across the project (src/, tests/cpp/, bench/,
# examples/, extensions/, src/bindings/ via BINDING_COMPILE_FLAGS).
#
# Implemented as a macro, not a function: it sets many caller-scope variables
# (COMPILE_FLAGS, CMAKE_POSITION_INDEPENDENT_CODE, CMAKE_CXX_STANDARD,
# CMAKE_INTERPROCEDURAL_OPTIMIZATION, ...) that must be visible to code that
# runs after this is called (add_subdirectory() calls elsewhere in the root
# CMakeLists.txt). A function() would confine all of those to a function-local
# scope and drop them on return unless every one were forwarded individually
# via PARENT_SCOPE; a macro avoids that entirely by executing directly in the
# caller's scope, matching the behavior of the code this was extracted from.
#
# Extracted verbatim from the root CMakeLists.txt "Set Compiler Flags"
# section -- no flag values were changed by this move.
################################################################################

macro(tycho_compile_options)

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

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang"
   AND NOT CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    # GNU-style Clang frontend only: clang-cl (MSVC frontend variant) does not
    # reliably forward arbitrary raw GNU-style -W flags without an explicit
    # /clang: prefix -- see the _TYCHO_WINDOWS_CLANG_NO_FINITE_MATH_ONLY_FLAG
    # precedent above. Not gating on x64-Clang-Release (clang-cl) for now;
    # revisit with /clang:-Wabsolute-value if Windows warning coverage is
    # wanted later.
    add_compile_options(-Wabsolute-value)
endif()

endmacro()
