# =============================================================================
# Enzyme-mode test executable.  The Enzyme Clang plugin flag is scoped to this
# target only; tycho_core / bindings / PCH stay plugin-free so the bulk of the
# build is unaffected.
# =============================================================================

add_executable(tycho_enzyme_tests
    test_enzyme_jacobian.cpp
    test_enzyme_hessian.cpp
    test_enzyme_phase_e2e.cpp
)

target_compile_options(tycho_enzyme_tests PRIVATE ${TYCHO_ENZYME_COMPILE_FLAGS})

target_link_libraries(tycho_enzyme_tests PRIVATE tycho GTest::gtest_main)

target_include_directories(tycho_enzyme_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/tests/cpp           # for test_utils.h (if needed)
    ${CMAKE_SOURCE_DIR}/extensions          # for Tycho_Extensions.h reference
)

# Heavy template instantiation + plugin overhead -> join the existing
# heavy_compile job pool so concurrent builds don't OOM the system.
set_target_properties(tycho_enzyme_tests PROPERTIES
    JOB_POOL_COMPILE heavy_compile)

# Do NOT reuse the PCH on Enzyme targets.  The Enzyme plugin injects
# preprocessor and AST-level state during parse; mixing that with a PCH
# built without the plugin is a known source of fragile failures.
set_target_properties(tycho_enzyme_tests PROPERTIES
    DISABLE_PRECOMPILE_HEADERS ON)

# Tests don't need -O3.  Lower optimization for compile-time savings.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(tycho_enzyme_tests PRIVATE $<$<CONFIG:Release>:-O1>)
endif()

include(GoogleTest)
gtest_discover_tests(tycho_enzyme_tests)

# =============================================================================
# Compile-fail probe: opting an EnzymeAD VectorFunction into Vectorizable
# (is_vectorizable = true with a SuperScalar Eigen::Array Scalar) must trigger
# the Phase 1 SuperScalar static_assert in dense_enzyme.h.  This is implemented
# as a separate target that we expect to fail to compile, plus a CTest case
# that drives `cmake --build` on it.  The test passes iff the build fails AND
# the build output contains a substring that proves the static_assert fired
# (rather than some unrelated build error).
#
# EXCLUDE_FROM_ALL keeps this target out of `ninja all` / `ninja
# tycho_enzyme_tests`; only `cmake --build --target` (run by ctest) builds it.
# =============================================================================
add_executable(tycho_enzyme_compile_fail_vectorizable
    compile_fail/vectorizable_phase1.cpp)

target_compile_options(tycho_enzyme_compile_fail_vectorizable
    PRIVATE ${TYCHO_ENZYME_COMPILE_FLAGS})

target_link_libraries(tycho_enzyme_compile_fail_vectorizable
    PRIVATE tycho)

target_include_directories(tycho_enzyme_compile_fail_vectorizable PRIVATE
    ${CMAKE_SOURCE_DIR}/tests/cpp
    ${CMAKE_SOURCE_DIR}/extensions)

set_target_properties(tycho_enzyme_compile_fail_vectorizable PROPERTIES
    JOB_POOL_COMPILE heavy_compile
    DISABLE_PRECOMPILE_HEADERS ON
    EXCLUDE_FROM_ALL TRUE)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(tycho_enzyme_compile_fail_vectorizable
        PRIVATE $<$<CONFIG:Release>:-O1>)
endif()

# CTest driver: invoke `cmake --build` on the compile-fail target through a
# small wrapper script.  The wrapper exits 0 iff the build fails AND the
# output contains the Phase 1 SuperScalar static_assert message — that pins
# the failure to the assert we care about and avoids passing on unrelated
# build breaks.
#
# (CTest's WILL_FAIL + PASS_REGULAR_EXPRESSION interact awkwardly: with
# WILL_FAIL TRUE a matching regex is treated as failure.  The wrapper-script
# form makes the intent explicit and portable across CMake versions.)
add_test(
    NAME EnzymeFwdNotVectorizableYet_BuildMustFail
    COMMAND ${CMAKE_COMMAND}
        -DBINARY_DIR=${CMAKE_BINARY_DIR}
        -DTARGET=tycho_enzyme_compile_fail_vectorizable
        -DCONFIG=$<CONFIG>
        -P ${CMAKE_CURRENT_SOURCE_DIR}/compile_fail/run_build_must_fail.cmake
)
