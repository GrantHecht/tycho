include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.17.0
    GIT_SHALLOW    TRUE
    EXCLUDE_FROM_ALL
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Clang 22.1.0 with MSVC headers: suppress char8_t->char32_t implicit conversion
# warning in gtest-printers.h.  gtest compiles with -WX so this becomes an error
# without the suppression.
if(WIN32)
    foreach(_t gtest gtest_main)
        if(TARGET ${_t})
            target_compile_options(${_t} PRIVATE -Wno-character-conversion)
        endif()
    endforeach()
endif()

include(GoogleTest)

#-------------------------------------------------------------------------------
# Test sources — split into two executables by memory weight.
#
# tycho_tests_light: utility and leaf-header tests that do NOT include the
#   full tycho/tycho.h umbrella (~200 MB per TU).
#
# tycho_tests: VF, OC, integrator, astro, and solver tests that include
#   tycho/tycho.h (directly or via test_utils.h).  Compiled without a PCH;
#   typical per-TU RSS is 4-7 GB due to heavy template instantiation.
#-------------------------------------------------------------------------------
set(TYCHO_TEST_LIGHT_SOURCES
    utils/test_timer.cpp
    utils/test_math_functions.cpp
    utils/test_sizing_helpers.cpp
    utils/test_eigen_stl.cpp
    utils/test_lambda_jump_table.cpp
    utils/test_misc_utils.cpp
    utils/test_flat_map.cpp
    utils/test_type_storage.cpp
    utils/test_tuple_iterator.cpp
    utils/test_var_registry.cpp
    optimal_control/test_ode_sizes.cpp
)

set(TYCHO_TEST_HEAVY_SOURCES
    # Utils (include tycho/tycho.h)
    utils/test_thread_pool.cpp
    utils/test_memory_management.cpp

    # VectorFunctions
    vector_functions/test_args_segment.cpp
    vector_functions/test_constant_scaled.cpp
    vector_functions/test_scaled_nesting.cpp
    vector_functions/test_multi_param_ode.cpp
    vector_functions/test_var_tags.cpp
    vector_functions/test_ode_syntax_prototypes.cpp
    vector_functions/test_sum_dot_cross.cpp
    vector_functions/test_norms.cpp
    vector_functions/test_normalized_stacked_padded.cpp
    vector_functions/test_composition_trig.cpp
    vector_functions/test_elements_cwiseops.cpp
    vector_functions/test_generic_function_basic.cpp
    vector_functions/test_generic_function_copy_move.cpp
    vector_functions/test_generic_function_interfaces.cpp
    vector_functions/test_vf_ode_expressions.cpp
    vector_functions/test_vf_composition_chains.cpp
    vector_functions/test_vf_hessian.cpp
    vector_functions/test_vf_interp_compose.cpp
    vector_functions/test_vf_value_lock.cpp
    vector_functions/test_vf_matrix_view_errors.cpp
    vector_functions/test_vector_functions.cpp
    vector_functions/test_cwise_segment_sum_jacobian.cpp
    vector_functions/test_cwise_jacobian_stress.cpp
    vector_functions/test_mixed_size_ops.cpp
    vector_functions/test_vf_concepts.cpp

    # Astro
    astro/test_kepler_conversions.cpp
    astro/test_kepler_edge_cases.cpp
    astro/test_kepler_propagation.cpp
    astro/test_lambert_solver.cpp
    astro/test_cr3bp.cpp

    # Integrators
    integrators/test_integration_basic.cpp
    integrators/test_integration_convergence.cpp
    integrators/test_integration_stm.cpp
    integrators/test_integration_adaptive.cpp
    integrators/test_integration_dense.cpp

    # OptimalControl
    optimal_control/test_phase_construction.cpp
    optimal_control/test_ocp_multiphase.cpp
    optimal_control/test_lgl_collocation.cpp
    optimal_control/test_mesh_refinement.cpp
    optimal_control/test_runtime_ode.cpp
    optimal_control/test_ode_builder.cpp
    optimal_control/test_phase_wrapper.cpp
    optimal_control/test_builder_brachistochrone.cpp
    optimal_control/test_control_modes.cpp

    # Solvers
    solvers/test_psiopt_convergence.cpp
    solvers/test_solver_interfaces.cpp
    solvers/test_jet_batch.cpp
    solvers/test_nlp_structure.cpp
)

#-------------------------------------------------------------------------------
# Common configuration for both test executables.
#-------------------------------------------------------------------------------
set(_TYCHO_TEST_COMMON_INCLUDES
    ${INCLUDE_DIRS}
    ${TYCHO_INCLUDES}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

function(tycho_add_test_executable target sources)
    add_executable(${target} ${${sources}})
    target_compile_options(${target} PRIVATE ${COMPILE_FLAGS})
    target_include_directories(${target} PRIVATE ${_TYCHO_TEST_COMMON_INCLUDES})
    target_link_libraries(${target} PRIVATE tycho GTest::gtest_main)
    if(UNIX)
        target_link_options(${target} PRIVATE ${LINKER_FLAGS})
    endif()
    gtest_discover_tests(${target})
endfunction()

#-------------------------------------------------------------------------------
# Light test executable — no PCH, ~200 MB per TU
#-------------------------------------------------------------------------------
tycho_add_test_executable(tycho_tests_light TYCHO_TEST_LIGHT_SOURCES)
target_include_directories(tycho_tests_light PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/optimal_control
)

#-------------------------------------------------------------------------------
# Heavy test executable — no PCH, ~4-7 GB per TU due to heavy template instantiation
#-------------------------------------------------------------------------------
tycho_add_test_executable(tycho_tests TYCHO_TEST_HEAVY_SOURCES)
target_include_directories(tycho_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/astro           # for astro_test_utils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/integrators     # for integrator_test_utils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/optimal_control # for oc_test_utils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/solvers         # for solver_test_utils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/vector_functions # for vf_test_utils.h
)
set_property(TARGET tycho_tests PROPERTY JOB_POOL_COMPILE heavy_compile)

# Unity build — merge test TUs to eliminate cross-TU template duplication.
# Batch size 8 merges heavy test sources into fewer unity TUs.
# This significantly reduces serial compile time at the cost of higher per-TU RSS.
set_target_properties(tycho_tests PROPERTIES
    UNITY_BUILD ON
    UNITY_BUILD_BATCH_SIZE 8
)

# Reduce optimization level for tests to improve compile time.
# With Clang/GCC, the last -O flag takes precedence over -O3 from COMPILE_FLAGS.
# Tests don't need -O3 — correctness is unaffected by optimization level.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(tycho_tests PRIVATE $<$<CONFIG:Release>:-O1>)
    target_compile_options(tycho_tests_light PRIVATE $<$<CONFIG:Release>:-O1>)
endif()
