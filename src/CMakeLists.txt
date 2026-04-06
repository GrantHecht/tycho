set(_TYCHO_SRC_INCLUDES
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/optimal_control
  ${CMAKE_CURRENT_SOURCE_DIR}/solvers
  ${CMAKE_CURRENT_SOURCE_DIR}/typedefs
  ${CMAKE_CURRENT_SOURCE_DIR}/utils
  ${CMAKE_CURRENT_SOURCE_DIR}/vf
  ${CMAKE_CURRENT_SOURCE_DIR}/integrators
)
set(TYCHO_INCLUDES ${_TYCHO_SRC_INCLUDES} PARENT_SCOPE)

#-------------------------------------------------------------------------------
## PCH
add_library(pch STATIC pch.h pch.cpp)
target_precompile_headers(pch PRIVATE pch.h
    vf/tycho_vector_functions.h)
target_include_directories(pch PRIVATE ${_TYCHO_SRC_INCLUDES} ${INCLUDE_DIRS})
target_compile_options(pch PRIVATE ${COMPILE_FLAGS})


## Put all ".h"/".cpp" pairs here
# Example: add_library(myODE STATIC odes/myode.h odes/myode.cpp)

# vectorfunctions is header-only (all VF code is template-based);
# expose it as an INTERFACE library so dependents get the include path.
add_library(vectorfunctions INTERFACE)
target_include_directories(vectorfunctions INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/vf>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)




add_library(optimalcontrol STATIC
    optimal_control/tycho_optimal_control.h
    optimal_control/ode_phase_base.cpp
    optimal_control/phase_indexer.cpp
    optimal_control/optimal_control_problem.cpp
    optimal_control/lgl_interp_table.cpp
    optimal_control/runtime_ode.cpp
)

target_include_directories(optimalcontrol PRIVATE ${_TYCHO_SRC_INCLUDES} ${INCLUDE_DIRS})
target_precompile_headers(optimalcontrol REUSE_FROM pch)
target_link_libraries(optimalcontrol PRIVATE psiopt)
target_compile_options(optimalcontrol PRIVATE ${COMPILE_FLAGS})
set_property(TARGET optimalcontrol PROPERTY JOB_POOL_COMPILE heavy_compile)





add_library(psiopt STATIC
    solvers/psiopt.cpp
    solvers/psiopt_print.cpp
    solvers/non_linear_program.cpp
    solvers/optimization_problem.cpp
    solvers/solver_init.cpp
)
target_include_directories(psiopt PRIVATE ${_TYCHO_SRC_INCLUDES} ${INCLUDE_DIRS})
target_compile_options(psiopt PRIVATE ${COMPILE_FLAGS})
target_link_libraries(psiopt PRIVATE utils)
if(APPLE)
    target_link_libraries(psiopt PUBLIC AccelerateSparse::AccelerateSparse)
endif()



add_library(utils STATIC
    utils/color_text.h utils/color_text.cpp
    utils/get_core_count.cpp
    utils/memory_management.cpp
    utils/thread_pool.cpp
)
target_include_directories(utils PRIVATE ${_TYCHO_SRC_INCLUDES} ${INCLUDE_DIRS})
target_compile_options(utils PRIVATE ${COMPILE_FLAGS})
# fmt is header-only (FMT_HEADER_ONLY defined globally); include path comes from
# the global include_directories(dep/fmt/include) in the root CMakeLists.txt.
# We do not link fmt::fmt-header-only because it would require exporting the fmt
# target alongside tycho.


add_library(astro STATIC
    astro/tycho_astro.h
    astro/lambert_solvers.cpp
    astro/accelleration_models.cpp
)

target_precompile_headers(astro REUSE_FROM pch)
target_include_directories(astro PRIVATE ${_TYCHO_SRC_INCLUDES} ${INCLUDE_DIRS})
target_link_libraries(astro PRIVATE optimalcontrol)
target_compile_options(astro PRIVATE ${COMPILE_FLAGS})


#-------------------------------------------------------------------------------
# tycho: public INTERFACE target bundling all core static libs.
# Used by Python bindings, C++ consumers (find_package/FetchContent), and future frontends.
add_library(tycho INTERFACE)
target_link_libraries(tycho INTERFACE psiopt utils optimalcontrol vectorfunctions astro)
# CRITICAL: LINK_LIBS is wrapped in BUILD_INTERFACE to avoid encoding
# host-specific absolute paths in the export set. ALL libraries in LINK_LIBS
# (set in root CMakeLists.txt) must be re-found and re-attached in
# cmake/tychoConfig.cmake.in. If you modify LINK_LIBS, update the config
# template to match.
target_link_libraries(tycho INTERFACE $<BUILD_INTERFACE:${LINK_LIBS}>)
target_link_libraries(tycho INTERFACE Eigen3::Eigen)
target_include_directories(tycho INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
add_library(tycho::tycho ALIAS tycho)
target_compile_features(tycho INTERFACE cxx_std_20)

# Compile definitions required by public headers
target_compile_definitions(tycho INTERFACE
    FMT_HEADER_ONLY
    FMT_USE_LOCALE=0
    EIGEN_INITIALIZE_MATRICES_BY_ZERO
    EIGEN_DONT_PARALLELIZE
    TYCHO_DEFAULT_QP_THREADS=8
)
if(APPLE)
    target_compile_definitions(tycho INTERFACE USE_ACCELERATE_SPARSE)
endif()
# Note: TYCHO_DEFAULT_ARENA_SIZE is intentionally omitted — it is only
# used in src/utils/memory_management.cpp (not in any public header).

# fmt, autodiff, and sparse-solver include paths for build-tree consumers
# (FetchContent). Install-tree consumers get these via tychoConfig.cmake.in.
target_include_directories(tycho INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/dep/fmt/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/dep/autodiff>
)
# MKL include directories are needed by public headers (PardisoInterface.h
# includes <mkl_pardiso.h>). On Apple, AccelerateSparse headers are part of
# the SDK and propagated via the AccelerateSparse::AccelerateSparse target.
if(MKL_INCLUDE_DIRS)
    target_include_directories(tycho INTERFACE
        $<BUILD_INTERFACE:${MKL_INCLUDE_DIRS}>
    )
endif()

# Compile options required for correct ABI with header-template code.
# Tycho's public headers contain Eigen types and template definitions that are
# instantiated in the consumer's translation units. Consumers MUST compile with
# matching SIMD, FP, and OpenMP flags to ensure:
#   - Eigen vectorization/alignment matches the library (prevents segfaults)
#   - Floating-point semantics match for template ABI consistency
#   - OpenMP runtime is linked (static libs contain OpenMP symbols)
# Generator expressions ensure correct flags per compiler/platform, and work
# in both build-tree and install-tree usage (CMake stores genexes as-is in
# tychoTargets.cmake, evaluating them against the consumer's compiler).
# Note: ${CLANG_MAX_INLINE_DEPTH} is a CMake variable expanded at configure
# time, so the inline threshold value is baked into the installed target.

# SIMD flags — Eigen alignment/vectorization must match between library and consumer
if(SIMD_FLAGS)
    target_compile_options(tycho INTERFACE ${SIMD_FLAGS})
endif()
# FP mode flags
if(FP_FLAGS)
    target_compile_options(tycho INTERFACE ${FP_FLAGS})
endif()
# Inline threshold — critical for VectorFunction expression template performance.
# Without aggressive inlining, the compiler cannot flatten the DSL expression tree
# into efficient machine code, causing 5-10x slowdowns in NLP evaluation.
target_compile_options(tycho INTERFACE
    $<$<CXX_COMPILER_ID:Clang>:-mllvm>
    $<$<CXX_COMPILER_ID:Clang>:-inline-threshold=${CLANG_MAX_INLINE_DEPTH}>
    $<$<CXX_COMPILER_ID:AppleClang>:-mllvm>
    $<$<CXX_COMPILER_ID:AppleClang>:-inline-threshold=${CLANG_MAX_INLINE_DEPTH}>
)
# OpenMP compile flags
# TODO: Add Windows (clang-cl) OpenMP flags for INTERFACE consumers
target_compile_options(tycho INTERFACE
    $<$<AND:$<PLATFORM_ID:Linux>,$<CXX_COMPILER_ID:Clang>>:-fopenmp=libiomp5>
    $<$<AND:$<PLATFORM_ID:Linux>,$<CXX_COMPILER_ID:GNU>>:-fopenmp>
    $<$<AND:$<PLATFORM_ID:Darwin>,$<CXX_COMPILER_ID:AppleClang>>:-Xclang>
    $<$<AND:$<PLATFORM_ID:Darwin>,$<CXX_COMPILER_ID:AppleClang>>:-fopenmp>
    $<$<AND:$<PLATFORM_ID:Darwin>,$<CXX_COMPILER_ID:Clang>,$<NOT:$<CXX_COMPILER_ID:AppleClang>>>:-fopenmp>
)

#-------------------------------------------------------------------------------
# Install rules

install(TARGETS tycho psiopt utils optimalcontrol astro vectorfunctions
    EXPORT tychoTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

#-------------------------------------------------------------------------------

if(ENABLE_PYTHON_BINDINGS)
    add_subdirectory(bindings)
endif()
