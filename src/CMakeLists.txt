include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/OptimalControl)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Solvers)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/TypeDefs)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Utils)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/VectorFunctions)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Integrators)

include_directories(${PROJECT_SOURCE_DIR}/extensions)



set(TYCHO_INCLUDES
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/OptimalControl
  ${CMAKE_CURRENT_SOURCE_DIR}/Solvers
  ${CMAKE_CURRENT_SOURCE_DIR}/TypeDefs
  ${CMAKE_CURRENT_SOURCE_DIR}/Utils
  ${CMAKE_CURRENT_SOURCE_DIR}/VectorFunctions
  ${CMAKE_CURRENT_SOURCE_DIR}/Integrators
  PARENT_SCOPE
)

#-------------------------------------------------------------------------------
## PCH
add_library(pch STATIC pch.h pch.cpp  )
target_precompile_headers(pch PRIVATE pch.h
VectorFunctions/Tycho_VectorFunctions.h)

target_include_directories(pch PRIVATE ${INCLUDE_DIRS})
#target_link_libraries(pch PRIVATE fmt::fmt-header-only)
target_compile_options(pch PRIVATE ${COMPILE_FLAGS})


## Put all ".h"/".cpp" pairs here
# Example: add_library(myODE STATIC odes/myode.h odes/myode.cpp)

# vectorfunctions is header-only (all VF code is template-based);
# expose it as an INTERFACE library so dependents get the include path.
add_library(vectorfunctions INTERFACE)
target_include_directories(vectorfunctions INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/VectorFunctions)




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

target_include_directories(optimalcontrol PRIVATE ${INCLUDE_DIRS})
target_precompile_headers(optimalcontrol REUSE_FROM pch)

target_compile_options(optimalcontrol PRIVATE ${COMPILE_FLAGS})





add_library(psiopt STATIC
  Solvers/PSIOPT.h Solvers/PSIOPT.cpp
  Solvers/NonLinearProgram.h Solvers/NonLinearProgram.cpp
  Solvers/Jet.h Solvers/Jet.cpp
  Solvers/OptimizationProblemBase.h
  Solvers/OptimizationProblem.h Solvers/OptimizationProblem.cpp
  Solvers/MKLInit.h Solvers/MKLInit.cpp
 )
target_include_directories(psiopt PRIVATE ${INCLUDE_DIRS})
target_compile_options(psiopt PRIVATE ${COMPILE_FLAGS})
if(APPLE)
    target_link_libraries(psiopt PUBLIC AccelerateSparse::AccelerateSparse)
endif()



add_library(utils STATIC
  Utils/ColorText.h Utils/ColorText.cpp
  Utils/GetCoreCount.h Utils/GetCoreCount.cpp
  Utils/MemoryManagement.h Utils/MemoryManagement.cpp
 )
target_include_directories(utils PRIVATE ${INCLUDE_DIRS})
target_compile_options(utils PRIVATE ${COMPILE_FLAGS})
target_link_libraries(utils PRIVATE fmt::fmt-header-only)


add_library(astro STATIC
  Astro/Tycho_Astro.h
  Astro/KeplerUtils.h
  Astro/KeplerModel.h
  Astro/LambertSolvers.h
  Astro/LambertSolvers.cpp
  )

target_precompile_headers(astro REUSE_FROM pch)
target_include_directories(astro PRIVATE ${INCLUDE_DIRS})
target_compile_options(astro PRIVATE ${COMPILE_FLAGS})


#-------------------------------------------------------------------------------
# tycho_core bundles all core static libs for use by bindings and future language frontends
add_library(tycho_core INTERFACE)
target_link_libraries(tycho_core INTERFACE psiopt utils optimalcontrol vectorfunctions astro)

#-------------------------------------------------------------------------------

message(STATUS "CMAKE DIR LOC" ${CMAKE_CURRENT_SOURCE_DIR})
message(STATUS "CMAKE PROJ LOC" ${CMAKE_SOURCE_DIR})

if(ENABLE_PYTHON_BINDINGS)
    add_subdirectory(Bindings)
endif()
