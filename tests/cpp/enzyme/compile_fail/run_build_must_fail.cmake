# =============================================================================
# Wrapper for the Phase 1 Vectorizable+EnzymeAD compile-fail probe.
#
# Drives `cmake --build` on a target that is *expected* to fail to compile.
# Test passes (script exits 0) iff:
#   - the build fails (cmake --build returns non-zero), AND
#   - the build output contains a substring proving the Phase 1 SuperScalar
#     static_assert (or its EIGEN_STATIC_ASSERT cousin) actually fired.
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
#   EXPECTED_REGEX — (optional) pattern that must appear in the build log to
#                    pin the failure to the intended diagnostic.  Defaults to
#                    the Phase 1 Vectorizable+EnzymeAD assertion text.
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
    set(EXPECTED_REGEX "(does not yet support SuperScalar|Vectorizable<Derived>|IsSuperScalar)")
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

message(STATUS
    "Compile-fail probe target '${TARGET}' failed as expected, "
    "with the expected diagnostic present.")
