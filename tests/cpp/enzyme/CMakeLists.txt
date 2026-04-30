# =============================================================================
# Enzyme-mode test executable.  The Enzyme Clang plugin flag is scoped to this
# target only; tycho_core / bindings / PCH stay plugin-free so the bulk of the
# build is unaffected.
# =============================================================================

add_executable(tycho_enzyme_tests
    test_enzyme_jacobian.cpp
    test_enzyme_hessian.cpp
    test_enzyme_nontemplate.cpp
    test_enzyme_vectorized.cpp
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
# Compile-fail probes for the Phase 4 non-templated `compute_impl` constraints
# (CLAUDE.md "Non-templated `compute_impl` (Phase 4)" section).  Each probe is
# a TU that MUST FAIL to compile; ctest passes iff cmake --build returns
# non-zero AND the EXPECTED_REGEX matches the diagnostic.
# =============================================================================
set(_enzyme_probes
    non_templated_with_fdiff
    non_templated_vectorizable
    non_templated_calls_compute
)
foreach(_probe ${_enzyme_probes})
    add_library(enzyme_compile_fail_${_probe} OBJECT EXCLUDE_FROM_ALL
        compile_fail/${_probe}.cpp)
    target_link_libraries(enzyme_compile_fail_${_probe} PRIVATE tycho)
    target_compile_options(enzyme_compile_fail_${_probe}
        PRIVATE ${TYCHO_ENZYME_COMPILE_FLAGS})
    set_target_properties(enzyme_compile_fail_${_probe} PROPERTIES
        DISABLE_PRECOMPILE_HEADERS ON
        JOB_POOL_COMPILE heavy_compile)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(enzyme_compile_fail_${_probe}
            PRIVATE $<$<CONFIG:Release>:-O1>)
    endif()
endforeach()

# All three probes fire the same root-cause diagnostic: a non-templated
# compute_impl that takes Eigen::Map<const Matrix<double,...>> cannot accept
# the Eigen::Matrix<Array<double,W,1>,...> (or other) type the dispatch path
# wraps it with.  The instantiation chain printed by clang distinguishes the
# three constraints; the per-probe target name distinguishes them in ctest.
add_test(NAME enzyme_compile_fail_non_templated_with_fdiff
    COMMAND ${CMAKE_COMMAND}
        -DBINARY_DIR=${CMAKE_BINARY_DIR}
        -DTARGET=enzyme_compile_fail_non_templated_with_fdiff
        -DCONFIG=$<CONFIG>
        "-DEXPECTED_REGEX=no viable conversion.*Eigen::Map"
        -P ${CMAKE_CURRENT_SOURCE_DIR}/compile_fail/run_build_must_fail.cmake)

add_test(NAME enzyme_compile_fail_non_templated_vectorizable
    COMMAND ${CMAKE_COMMAND}
        -DBINARY_DIR=${CMAKE_BINARY_DIR}
        -DTARGET=enzyme_compile_fail_non_templated_vectorizable
        -DCONFIG=$<CONFIG>
        "-DEXPECTED_REGEX=no viable conversion.*Eigen::Map"
        -P ${CMAKE_CURRENT_SOURCE_DIR}/compile_fail/run_build_must_fail.cmake)

add_test(NAME enzyme_compile_fail_non_templated_calls_compute
    COMMAND ${CMAKE_COMMAND}
        -DBINARY_DIR=${CMAKE_BINARY_DIR}
        -DTARGET=enzyme_compile_fail_non_templated_calls_compute
        -DCONFIG=$<CONFIG>
        "-DEXPECTED_REGEX=no viable conversion.*Eigen::Map"
        -P ${CMAKE_CURRENT_SOURCE_DIR}/compile_fail/run_build_must_fail.cmake)
