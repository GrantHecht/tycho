# =============================================================================
# Generic wrapper for compile-fail probes.
#
# Shared by every compile-fail probe in the C++ suite (the Enzyme dispatch
# probes under enzyme/compile_fail/, the function-entry seam probes under
# solvers/compile_fail/).  It is generic: it knows nothing about either.
#
# Drives `cmake --build` on a target that is *expected* to fail to compile.
# Test passes (script exits 0) iff:
#   - the build fails (cmake --build returns non-zero), AND
#   - the build output contains a substring matching EXPECTED_REGEX
#     (pins the failure to the intended diagnostic), AND
#   - the build output does NOT contain FORBIDDEN_REGEX, when one is given.
#
# The wrapper-script form is used (instead of CTest's WILL_FAIL +
# PASS_REGULAR_EXPRESSION) because in CMake 3.16+ those two properties
# interact awkwardly: with WILL_FAIL TRUE the regex match is inverted, so a
# matching regex *fails* the test.  A small wrapper makes the intent explicit
# and portable.
#
# Required input variables (passed on the command line via -D):
#   BINARY_DIR     — top-level build dir to invoke `cmake --build` from
#   TARGET         — the build target expected to fail
#   CONFIG         — build config (e.g. Release)
#   EXPECTED_REGEX — REQUIRED.  Pattern that must appear in the build log to
#                    pin the failure to the intended diagnostic.  Each caller
#                    must supply this; there is no default, so a typo in the
#                    -D variable surfaces as a clear "regex was empty" error.
#   FORBIDDEN_REGEX — OPTIONAL.  Pattern that must NOT appear in the build log.
#                    Presence-only checking is too weak on its own: a probe
#                    whose authored diagnostic fires CORRECTLY but which then
#                    goes on to instantiate the machinery it was supposed to
#                    stop short of will emit raw compiler errors beside the
#                    authored one and still pass.  A caller that claims "the
#                    authored message is the only failure" passes the raw
#                    pattern here so the claim is actually tested.  Anchor it
#                    on severity where that matters: a compiler often repeats a
#                    phrase inside explanatory NOTES attached to the authored
#                    diagnostic, and forbidding those would forbid the
#                    diagnostic from explaining itself.
# =============================================================================

if(NOT DEFINED BINARY_DIR)
    message(FATAL_ERROR "BINARY_DIR not set")
endif()
if(NOT DEFINED TARGET)
    message(FATAL_ERROR "TARGET not set")
endif()
if(NOT DEFINED CONFIG)
    set(CONFIG "")
endif()
if(NOT DEFINED EXPECTED_REGEX OR EXPECTED_REGEX STREQUAL "")
    message(FATAL_ERROR
        "EXPECTED_REGEX not set or empty.  Each compile-fail probe must "
        "supply a regex that pins the diagnostic, otherwise an unrelated "
        "build break would silently pass.  See "
        "tests/cpp/enzyme/CMakeLists.txt for examples.")
endif()

set(_build_args --build "${BINARY_DIR}" --target "${TARGET}")
if(CONFIG)
    list(APPEND _build_args --config "${CONFIG}")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" ${_build_args}
    RESULT_VARIABLE _rc
    OUTPUT_VARIABLE _out
    ERROR_VARIABLE  _err
)

set(_combined "${_out}${_err}")

if(_rc EQUAL 0)
    message("--- build STDOUT ---\n${_out}")
    message("--- build STDERR ---\n${_err}")
    message(FATAL_ERROR
        "Compile-fail probe target '${TARGET}' built successfully — "
        "the expected diagnostic did NOT fire.")
endif()

# Pin the failure to the expected diagnostic so an unrelated build error
# (e.g. missing header) doesn't pass as a fake success.
string(REGEX MATCH "${EXPECTED_REGEX}" _match "${_combined}")

if(NOT _match)
    message("--- build STDOUT ---\n${_out}")
    message("--- build STDERR ---\n${_err}")
    message(FATAL_ERROR
        "Compile-fail probe target '${TARGET}' failed to build, but the "
        "expected diagnostic regex '${EXPECTED_REGEX}' was not found in "
        "the output.  This may indicate an unrelated build break.")
endif()

if(DEFINED FORBIDDEN_REGEX AND NOT FORBIDDEN_REGEX STREQUAL "")
    string(REGEX MATCH "${FORBIDDEN_REGEX}" _forbidden "${_combined}")
    if(_forbidden)
        message("--- build STDOUT ---\n${_out}")
        message("--- build STDERR ---\n${_err}")
        message(FATAL_ERROR
            "Compile-fail probe target '${TARGET}' produced the expected "
            "diagnostic, but ALSO produced the forbidden pattern "
            "'${FORBIDDEN_REGEX}' (matched: '${_forbidden}').  The refusal is "
            "not clean: something was instantiated behind the failed check.")
    endif()
endif()

message(STATUS
    "Compile-fail probe target '${TARGET}' failed as expected, "
    "with the expected diagnostic present.")
