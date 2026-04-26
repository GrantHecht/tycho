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
# Phase 4 compile-fail probe: non-templated compute_impl + AutodiffFwd MUST
# fail to compile.  AutodiffFwd's compute_jacobian_impl calls compute() with
# `dual<double>`-typed Eigen vectors; a non-templated compute_impl that takes
# concrete `double` Map cannot accept those types — substitution failure.
# =============================================================================
add_executable(tycho_enzyme_compile_fail_nontemplate_autodiff
    compile_fail/non_template_autodiff.cpp)

target_link_libraries(tycho_enzyme_compile_fail_nontemplate_autodiff
    PRIVATE tycho)

target_include_directories(tycho_enzyme_compile_fail_nontemplate_autodiff
    PRIVATE
        ${CMAKE_SOURCE_DIR}/tests/cpp
        ${CMAKE_SOURCE_DIR}/extensions)

set_target_properties(tycho_enzyme_compile_fail_nontemplate_autodiff PROPERTIES
    JOB_POOL_COMPILE heavy_compile
    DISABLE_PRECOMPILE_HEADERS ON
    EXCLUDE_FROM_ALL TRUE)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(tycho_enzyme_compile_fail_nontemplate_autodiff
        PRIVATE $<$<CONFIG:Release>:-O1>)
endif()

add_test(
    NAME EnzymeNonTemplateAutodiff_BuildMustFail
    COMMAND ${CMAKE_COMMAND}
        -DBINARY_DIR=${CMAKE_BINARY_DIR}
        -DTARGET=tycho_enzyme_compile_fail_nontemplate_autodiff
        -DCONFIG=$<CONFIG>
        -DEXPECTED_REGEX=no\ matching\ function|cannot\ deduce|no\ viable|substitution\ failure
        -P ${CMAKE_CURRENT_SOURCE_DIR}/compile_fail/run_build_must_fail.cmake
)
