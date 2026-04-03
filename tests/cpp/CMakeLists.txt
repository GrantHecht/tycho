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
#   full tycho/tycho.h umbrella.  Compiled WITHOUT a precompiled header so
#   each TU only parses what it actually needs (~200 MB per TU instead of
#   the ~6 GB PCH baseline).
#
# tycho_tests: VF, OC, integrator, astro, and solver tests that include
#   tycho/tycho.h (directly or via test_utils.h).  Also compiled without
#   a PCH — measurements show that per-TU RSS drops ~30% for most of
#   these tests when the full-library PCH is not force-loaded.
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
    # Utils (include test_utils.h -> tycho/tycho.h)
    utils/test_thread_pool.cpp
    utils/test_memory_management.cpp

    # VectorFunctions
    vector_functions/test_args_segment.cpp
    vector_functions/test_constant_scaled.cpp
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
    vector_functions/test_vector_functions.cpp
    vector_functions/test_cwise_segment_sum_jacobian.cpp
    vector_functions/test_cwise_jacobian_stress.cpp
    vector_functions/test_mixed_size_ops.cpp

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
# Light test executable — no PCH, ~200 MB per TU
#-------------------------------------------------------------------------------
add_executable(tycho_tests_light ${TYCHO_TEST_LIGHT_SOURCES})
target_compile_options(tycho_tests_light PRIVATE ${COMPILE_FLAGS})
target_include_directories(tycho_tests_light PRIVATE
    ${INCLUDE_DIRS}
    ${TYCHO_INCLUDES}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/optimal_control
)
target_link_libraries(tycho_tests_light PRIVATE tycho GTest::gtest_main)
if(UNIX)
    target_link_options(tycho_tests_light PRIVATE ${LINKER_FLAGS})
endif()
gtest_discover_tests(tycho_tests_light)

#-------------------------------------------------------------------------------
# Heavy test executable — no PCH, ~4-7 GB per TU (still better than the
# ~6 GB uniform cost when a full-library PCH was force-loaded on every TU)
#-------------------------------------------------------------------------------
add_executable(tycho_tests ${TYCHO_TEST_HEAVY_SOURCES})
target_compile_options(tycho_tests PRIVATE ${COMPILE_FLAGS})
target_include_directories(tycho_tests PRIVATE
    ${INCLUDE_DIRS}
    ${TYCHO_INCLUDES}
    ${CMAKE_CURRENT_SOURCE_DIR}                 # for test_utils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/astro           # for astro_test_utils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/integrators     # for integrator_test_utils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/optimal_control # for oc_test_utils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/solvers         # for solver_test_utils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/vector_functions # for vf_test_utils.h
)
target_link_libraries(tycho_tests PRIVATE tycho GTest::gtest_main)
set_property(TARGET tycho_tests PROPERTY JOB_POOL_COMPILE heavy_compile)
if(UNIX)
    target_link_options(tycho_tests PRIVATE ${LINKER_FLAGS})
endif()
gtest_discover_tests(tycho_tests)
