set(_TYCHO_SRC_INCLUDES
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/OptimalControl
  ${CMAKE_CURRENT_SOURCE_DIR}/Solvers
  ${CMAKE_CURRENT_SOURCE_DIR}/TypeDefs
  ${CMAKE_CURRENT_SOURCE_DIR}/Utils
  ${CMAKE_CURRENT_SOURCE_DIR}/VectorFunctions
  ${CMAKE_CURRENT_SOURCE_DIR}/Integrators
)
set(TYCHO_INCLUDES ${_TYCHO_SRC_INCLUDES} PARENT_SCOPE)

#-------------------------------------------------------------------------------
## PCH
add_library(pch STATIC pch.h pch.cpp)
target_precompile_headers(pch PRIVATE pch.h
    VectorFunctions/Tycho_VectorFunctions.h)
target_include_directories(pch PRIVATE ${_TYCHO_SRC_INCLUDES} ${INCLUDE_DIRS})
target_compile_options(pch PRIVATE ${COMPILE_FLAGS})


## Put all ".h"/".cpp" pairs here
# Example: add_library(myODE STATIC odes/myode.h odes/myode.cpp)

# vectorfunctions is header-only (all VF code is template-based);
# expose it as an INTERFACE library so dependents get the include path.
add_library(vectorfunctions INTERFACE)
target_include_directories(vectorfunctions INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/VectorFunctions>
)




add_library(optimalcontrol STATIC OptimalControl/Tycho_OptimalControl.h
OptimalControl/ODEPhaseBase.h
OptimalControl/ODEPhaseBase.cpp
OptimalControl/PhaseIndexer.h
OptimalControl/PhaseIndexer.cpp
OptimalControl/OptimalControlProblem.h
OptimalControl/OptimalControlProblem.cpp
OptimalControl/LGLInterpTable.h
OptimalControl/LGLInterpTable.cpp
)

target_include_directories(optimalcontrol PRIVATE ${_TYCHO_SRC_INCLUDES} ${INCLUDE_DIRS})
target_precompile_headers(optimalcontrol REUSE_FROM pch)
target_link_libraries(optimalcontrol PRIVATE psiopt)
target_compile_options(optimalcontrol PRIVATE ${COMPILE_FLAGS})





add_library(psiopt STATIC
  Solvers/PSIOPT.h Solvers/PSIOPT.cpp
  Solvers/NonLinearProgram.h Solvers/NonLinearProgram.cpp
  Solvers/Jet.h
  Solvers/OptimizationProblemBase.h
  Solvers/OptimizationProblem.h Solvers/OptimizationProblem.cpp
  Solvers/SolverInit.h Solvers/SolverInit.cpp
 )
target_include_directories(psiopt PRIVATE ${_TYCHO_SRC_INCLUDES} ${INCLUDE_DIRS})
target_compile_options(psiopt PRIVATE ${COMPILE_FLAGS})
target_link_libraries(psiopt PRIVATE utils)
if(APPLE)
    target_link_libraries(psiopt PUBLIC AccelerateSparse::AccelerateSparse)
endif()



add_library(utils STATIC
  Utils/ColorText.h Utils/ColorText.cpp
  Utils/GetCoreCount.h Utils/GetCoreCount.cpp
  Utils/MemoryManagement.h Utils/MemoryManagement.cpp
  Utils/ThreadPool.h Utils/ThreadPool.cpp
 )
target_include_directories(utils PRIVATE ${_TYCHO_SRC_INCLUDES} ${INCLUDE_DIRS})
target_compile_options(utils PRIVATE ${COMPILE_FLAGS})
# fmt is header-only (FMT_HEADER_ONLY defined globally); include path comes from
# the global include_directories(dep/fmt/include) in the root CMakeLists.txt.
# We do not link fmt::fmt-header-only because it would require exporting the fmt
# target alongside tycho.


add_library(astro STATIC
  Astro/Tycho_Astro.h
  Astro/KeplerUtils.h
  Astro/KeplerModel.h
  Astro/LambertSolvers.h
  Astro/LambertSolvers.cpp
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
)
if(APPLE)
    target_compile_definitions(tycho INTERFACE USE_ACCELERATE_SPARSE)
endif()

# fmt and autodiff include paths for build-tree consumers (FetchContent).
# Install-tree consumers use the existing $<INSTALL_INTERFACE:include> above.
target_include_directories(tycho INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/dep/fmt/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/dep/autodiff>
)

#-------------------------------------------------------------------------------
# Install rules

include(GNUInstallDirs)

install(TARGETS tycho psiopt utils optimalcontrol astro vectorfunctions
    EXPORT tychoTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

#-------------------------------------------------------------------------------

if(ENABLE_PYTHON_BINDINGS)
    add_subdirectory(Bindings)
endif()
