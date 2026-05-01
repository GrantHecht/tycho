# Configure-time end-to-end smoke tests for the Enzyme Clang plugin.
# Each helper compiles a tiny program with the plugin flag, runs it, and
# requires exit code 0. Any failure → FATAL_ERROR with an actionable message.
#
# Plugin-flag plumbing:
#   The plugin flag (e.g. "-fplugin=/path/to/ClangEnzyme-22.so") must reach
#   the *compile* line so the plugin runs on the IR; it must also stay on the
#   *link* line, because without the plugin's lowering pass __enzyme_fwddiff
#   and enzyme_dup remain undefined symbols.
#
#   We use CheckCXXSourceRuns (from CheckCXXSourceRuns), which honors
#   CMAKE_REQUIRED_FLAGS for both compilation and linking. This is more robust
#   than try_run's CMAKE_FLAGS path, which silently dropped the flag in
#   testing on this machine — likely because "-DCMAKE_CXX_FLAGS=<flags>" inside
#   CMAKE_FLAGS is parsed as separate list elements and the inner value never
#   reaches the compile command.

include(CheckCXXSourceRuns)

function(tycho_enzyme_try_compile_fwddiff)
    set(_src_path "${PROJECT_SOURCE_DIR}/cmake/try_compile_sources/enzyme_fwddiff.cpp")

    if(NOT EXISTS "${_src_path}")
        message(FATAL_ERROR
            "Enzyme fwddiff smoke-test source not found at: ${_src_path}")
    endif()

    file(READ "${_src_path}" _src_content)

    # Save and override CMAKE_REQUIRED_FLAGS / CMAKE_REQUIRED_FLAGS-adjacent vars.
    set(_saved_required_flags "${CMAKE_REQUIRED_FLAGS}")
    set(CMAKE_REQUIRED_FLAGS "${TYCHO_ENZYME_COMPILE_FLAGS} -O1 -std=c++20")

    # Force a fresh check — never let a stale cache hide a broken plugin install.
    unset(TYCHO_ENZYME_FWDDIFF_SMOKE_TEST CACHE)
    check_cxx_source_runs("${_src_content}" TYCHO_ENZYME_FWDDIFF_SMOKE_TEST)

    set(CMAKE_REQUIRED_FLAGS "${_saved_required_flags}")

    if(NOT TYCHO_ENZYME_FWDDIFF_SMOKE_TEST)
        message(FATAL_ERROR
            "Enzyme fwddiff smoke test failed (compile, link, or wrong runtime "
            "answer for f'(3) where f(x) = x*x; expected 6).\n"
            "See ${CMAKE_BINARY_DIR}/CMakeFiles/CMakeConfigureLog.yaml "
            "(or CMakeError.log on older CMake) for the full compile/link "
            "command and output.\n"
            "Likely causes:\n"
            "  * Enzyme plugin version mismatch with active Clang.\n"
            "  * Clang does not accept -fplugin=<plugin> at this path.\n"
            "  * Plugin path '${TYCHO_ENZYME_COMPILE_FLAGS}' is wrong "
            "or the .so is unreadable.")
    endif()
    message(STATUS "Enzyme AD: fwddiff smoke test passed.")
endfunction()

# Helper: shared body for the nested smoke test. Compiles + runs the named
# source with the plugin flag attached; sets ${OUTVAR} to TRUE on pass, FALSE
# otherwise. Does NOT FATAL_ERROR — the caller decides how to react.
function(_tycho_enzyme_run_nested_smoke SRC_NAME CACHE_VAR LABEL OUTVAR)
    set(_src_path "${PROJECT_SOURCE_DIR}/cmake/try_compile_sources/${SRC_NAME}")
    if(NOT EXISTS "${_src_path}")
        message(FATAL_ERROR
            "Enzyme nested ${LABEL} smoke-test source not found at: ${_src_path}")
    endif()

    file(READ "${_src_path}" _src_content)

    set(_saved_required_flags "${CMAKE_REQUIRED_FLAGS}")
    set(CMAKE_REQUIRED_FLAGS "${TYCHO_ENZYME_COMPILE_FLAGS} -O1 -std=c++20")

    unset(${CACHE_VAR} CACHE)
    check_cxx_source_runs("${_src_content}" ${CACHE_VAR})

    set(CMAKE_REQUIRED_FLAGS "${_saved_required_flags}")

    if(${CACHE_VAR})
        message(STATUS "Enzyme AD: nested-${LABEL} smoke test passed.")
        set(${OUTVAR} TRUE PARENT_SCOPE)
    else()
        message(STATUS
            "Enzyme AD: nested-${LABEL} smoke test FAILED. The ${LABEL} Hessian "
            "strategy will be unavailable. See "
            "${CMAKE_BINARY_DIR}/CMakeFiles/CMakeConfigureLog.yaml for details.")
        set(${OUTVAR} FALSE PARENT_SCOPE)
    endif()
endfunction()

function(tycho_enzyme_try_compile_nested_for OUTVAR)
    _tycho_enzyme_run_nested_smoke(
        enzyme_nested_for.cpp
        TYCHO_ENZYME_NESTED_FOR_SMOKE_TEST
        FoR
        ${OUTVAR})
    set(${OUTVAR} ${${OUTVAR}} PARENT_SCOPE)
endfunction()
