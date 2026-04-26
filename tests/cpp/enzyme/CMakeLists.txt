# =============================================================================
# Enzyme-mode test executable.  The Enzyme Clang plugin flag is scoped to this
# target only; tycho_core / bindings / PCH stay plugin-free so the bulk of the
# build is unaffected.
# =============================================================================

add_executable(tycho_enzyme_tests
    test_enzyme_jacobian.cpp
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
