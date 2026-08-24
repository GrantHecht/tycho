# ── Static (legacy DSL) examples ────────────────────────────────────────────
add_subdirectory(static/brachistochrone)
add_subdirectory(static/zermelo)
add_subdirectory(static/hypersens)
add_subdirectory(static/delta3_launch)

# ── Builder API examples ────────────────────────────────────────────────────
add_subdirectory(builder/brachistochrone)
add_subdirectory(builder/zermelo)
add_subdirectory(builder/hypersens)
add_subdirectory(builder/delta3_launch)

# New builder examples (Tier 1: simple single-phase)
add_subdirectory(builder/van_der_pol)
add_subdirectory(builder/bryson_denham)
add_subdirectory(builder/mountain_car)
add_subdirectory(builder/analytic_example)
add_subdirectory(builder/bike_obstacle)
add_subdirectory(builder/free_flying_robot)

# New builder examples (Tier 2: single-phase advanced)
add_subdirectory(builder/cart_pole)
add_subdirectory(builder/parallel_parking)
add_subdirectory(builder/hanging_chain)
add_subdirectory(builder/reentry)
add_subdirectory(builder/minimum_time_to_climb)

# New builder examples (Tier 3: multi-phase)
add_subdirectory(builder/goddard_rocket)
add_subdirectory(builder/multi_phase_cannon)
add_subdirectory(builder/multi_phase_zermelo)

# New builder examples (Tier 4: low-thrust astrodynamics)
add_subdirectory(builder/simple_low_thrust)
add_subdirectory(builder/topputto_low_thrust)
add_subdirectory(builder/dionysus_low_thrust)
add_subdirectory(builder/betts_low_thrust)
add_subdirectory(builder/orbit_continuation)

# New builder examples (Tier 5: advanced astrodynamics)
add_subdirectory(builder/heteroclinic)
add_subdirectory(builder/optimal_docking)
add_subdirectory(builder/multi_spacecraft_opt)

# New builder examples (Tier 6: interpolation tables)
add_subdirectory(builder/min_time_climb_tables)

# ── Assign all example targets to the heavy_compile job pool ────────────────
set(_all_example_targets
    # Static
    brachistochrone_cpp zermelo_cpp hypersens_cpp delta3_launch_cpp
    # Builder (existing)
    brachistochrone_builder zermelo_builder hypersens_builder delta3_launch_builder
    # Builder (Tier 1)
    van_der_pol_builder bryson_denham_builder mountain_car_builder
    analytic_example_builder bike_obstacle_builder free_flying_robot_builder
    # Builder (Tier 2)
    cart_pole_builder parallel_parking_builder hanging_chain_builder
    reentry_builder minimum_time_to_climb_builder
    # Builder (Tier 3)
    goddard_rocket_builder multi_phase_cannon_builder multi_phase_zermelo_builder
    # Builder (Tier 4)
    simple_low_thrust_builder topputto_low_thrust_builder dionysus_low_thrust_builder
    betts_low_thrust_builder orbit_continuation_builder
    # Builder (Tier 5)
    heteroclinic_builder optimal_docking_builder multi_spacecraft_opt_builder
    # Builder (Tier 6)
    min_time_climb_tables_builder
)

foreach(_ex IN LISTS _all_example_targets)
    if(TARGET ${_ex})
        set_property(TARGET ${_ex} PROPERTY JOB_POOL_COMPILE heavy_compile)
    endif()
endforeach()

# ── Aggregate target: build all C++ examples ────────────────────────────────
add_custom_target(cpp_examples_all)
foreach(_ex IN LISTS _all_example_targets)
    if(TARGET ${_ex})
        add_dependencies(cpp_examples_all ${_ex})
    endif()
endforeach()

# ── CTest registration ──────────────────────────────────────────────────────
include(CTest)

# multi_spacecraft_opt_builder is registered separately below, with an argument
# the others do not take.
foreach(_ex IN LISTS _all_example_targets)
    if(TARGET ${_ex} AND NOT _ex STREQUAL "multi_spacecraft_opt_builder")
        add_test(NAME cpp_example_${_ex} COMMAND ${_ex})
        set_tests_properties(cpp_example_${_ex} PROPERTIES TIMEOUT 300)
    endif()
endforeach()

# Jet::map() sweep examples need more time than single-solve examples
if(TARGET hanging_chain_builder)
    set_tests_properties(cpp_example_hanging_chain_builder PROPERTIES TIMEOUT 600)
endif()

# multi_spacecraft_opt_builder is the one example whose gate invocation differs
# from the example itself, and --deterministic is why.
#
# The example runs its solve over several evaluation partitions, so the order in
# which threads reduce into the KKT assembly varies from run to run and the
# iterate sequence varies with it. On this problem the last continuation step
# (ltacc = 0.005) converges marginally enough for that difference to decide the
# outcome: measured on an idle machine with nothing else running at all, it
# fails one run in eight, and the failing run diverges -- a final time of ~1712
# periods against ~3.3 on a passing one -- rather than missing a tolerance.
# Running it alone does not fix that; the nondeterminism is inside the solve,
# not in what else the machine is doing, which is why this test carries no
# RUN_SERIAL property.
#
# --deterministic pins the solve to one evaluation partition and one QP thread,
# leaving nothing to reduce across threads, and it then produces the same
# arithmetic every run. It costs roughly three times the runtime (~49 s against
# ~16 s), which is the price of a gate that does not flake.
#
# The threaded path is still what the example demonstrates: only this
# invocation passes the flag, and running the binary by hand gets the demo.
# The marginal continuation step itself is untouched and still worth a look.
if(TARGET multi_spacecraft_opt_builder)
    add_test(NAME cpp_example_multi_spacecraft_opt_builder
             COMMAND multi_spacecraft_opt_builder --deterministic)
    set_tests_properties(cpp_example_multi_spacecraft_opt_builder PROPERTIES TIMEOUT 300)
endif()
