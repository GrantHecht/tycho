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
target_compile_options(pch PUBLIC ${COMPILE_FLAGS})


## Put all ".h"/".cpp" pairs here
# Example: add_library(myODE STATIC odes/myode.h odes/myode.cpp)

add_library(vectorfunctions STATIC VectorFunctions/Tycho_VectorFunctions.h
    VectorFunctions/Tycho_VectorFunctions.cpp
    VectorFunctions/VectorFunctionBuildPart1.cpp
    VectorFunctions/VectorFunctionBuildPart2.cpp
    VectorFunctions/ArgsSegBuildPart1.cpp
    VectorFunctions/ArgsSegBuildPart2.cpp
    VectorFunctions/ArgsSegBuildPart3.cpp
    VectorFunctions/ArgsSegBuildPart4.cpp
    VectorFunctions/ArgsSegBuildPart5.cpp
    VectorFunctions/BulkOperationsBuild.cpp
    VectorFunctions/FreeFunctionsBuild.cpp
    VectorFunctions/MatrixFunctionBuild.cpp
    VectorFunctions/PythonArgParsing.cpp
    )
target_include_directories(vectorfunctions PRIVATE ${INCLUDE_DIRS})
target_compile_options(vectorfunctions PUBLIC ${COMPILE_FLAGS})
target_precompile_headers(vectorfunctions REUSE_FROM pch)
    





add_library(optimalcontrol STATIC OptimalControl/Tycho_OptimalControl.h
OptimalControl/Tycho_OptimalControl.cpp
OptimalControl/ODEPhaseBase.h
OptimalControl/ODEPhaseBase.cpp
OptimalControl/ODEPhaseBaseBind.cpp
OptimalControl/GenericODESBuildPart1.cpp
OptimalControl/GenericODESBuildPart2.cpp
OptimalControl/GenericODESBuildPart3.cpp
OptimalControl/GenericODESBuildPart4.cpp
OptimalControl/GenericODESBuildPart5.cpp
OptimalControl/GenericODESBuildPart6.cpp
OptimalControl/PhaseIndexer.h
OptimalControl/PhaseIndexer.cpp
OptimalControl/OptimalControlProblem.h
OptimalControl/OptimalControlProblem.cpp
OptimalControl/OptimalControlProblemBind.cpp
OptimalControl/LGLInterpTable.h
OptimalControl/LGLInterpTable.cpp
)

target_include_directories(optimalcontrol PRIVATE ${INCLUDE_DIRS})
target_precompile_headers(optimalcontrol REUSE_FROM pch)

target_compile_options(optimalcontrol PUBLIC ${COMPILE_FLAGS})
  






add_library(psiopt STATIC
  Solvers/PSIOPT.h Solvers/PSIOPT.cpp
  Solvers/NonLinearProgram.h Solvers/NonLinearProgram.cpp
  Solvers/Jet.h Solvers/Jet.cpp
  Solvers/OptimizationProblemBase.h Solvers/OptimizationProblemBase.cpp
  Solvers/OptimizationProblem.h Solvers/OptimizationProblem.cpp
 )
target_include_directories(psiopt PRIVATE ${INCLUDE_DIRS})
target_compile_options(psiopt PUBLIC ${COMPILE_FLAGS})



add_library(utils STATIC
  Utils/ColorText.h Utils/ColorText.cpp
  Utils/GetCoreCount.h Utils/GetCoreCount.cpp
  Utils/MemoryManagement.h Utils/MemoryManagement.cpp
 )
target_include_directories(utils PRIVATE ${INCLUDE_DIRS})
target_compile_options(utils PUBLIC ${COMPILE_FLAGS})
target_link_libraries(utils PRIVATE fmt::fmt-header-only)


add_library(astro STATIC
  Astro/Tycho_Astro.h 
  Astro/Tycho_Astro.cpp
  Astro/KeplerUtils.h 
  Astro/KeplerUtils.cpp
  Astro/KeplerModel.h 
  Astro/KeplerModel.cpp
  Astro/LambertSolvers.h
  Astro/LambertSolvers.cpp
  )

target_precompile_headers(astro REUSE_FROM pch)
target_include_directories(astro PRIVATE ${INCLUDE_DIRS})
target_compile_options(astro PUBLIC ${COMPILE_FLAGS})





#---------------------------------------------------------------------------------





#-------------------------------------------------------------------------------
## Tycho Python Module
pybind11_add_module(_tycho main.cpp)


target_compile_options(_tycho PUBLIC  ${COMPILE_FLAGS})
target_link_libraries(_tycho PUBLIC psiopt)
target_link_libraries(_tycho PUBLIC utils)

target_link_libraries(_tycho PUBLIC optimalcontrol)
target_link_libraries(_tycho PUBLIC vectorfunctions)
target_link_libraries(_tycho PUBLIC astro)
target_link_libraries(_tycho PUBLIC tycho_extensions)

target_include_directories(_tycho PRIVATE ${INCLUDE_DIRS})
target_link_libraries(_tycho PRIVATE ${LINK_LIBS})
#set_target_properties(asset PROPERTIES CXX_VISIBILITY_PRESET hidden)

if(UNIX)
    target_link_options(_tycho PRIVATE ${LINKER_FLAGS})
endif()

add_custom_target(pytychosrc ALL DEPENDS ${CMAKE_SOURCE_DIR}/tycho)


message(STATUS "CMAKE DIR LOC" ${CMAKE_CURRENT_SOURCE_DIR})
message(STATUS "CMAKE PROJ LOC" ${CMAKE_SOURCE_DIR})

if(PYTHON_LOCAL_INSTALL_DIR)
    add_custom_command(TARGET _tycho POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:_tycho> ${PYTHON_LOCAL_INSTALL_DIR}/
    COMMENT "Copied _tycho extension to ${PYTHON_LOCAL_INSTALL_DIR}"
    )
    message(STATUS "Cleaning tycho install directory")

    add_custom_command(TARGET pytychosrc POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E rm -rf ${PYTHON_LOCAL_INSTALL_DIR}/tycho
                    COMMENT "Cleaning tycho install directory")

    add_custom_command(TARGET pytychosrc POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_directory  ${CMAKE_SOURCE_DIR}/tycho
                                                ${PYTHON_LOCAL_INSTALL_DIR}/tycho
                    DEPENDS ${CMAKE_SOURCE_DIR}/tycho
                    COMMENT "Copied tycho to ${PYTHON_LOCAL_INSTALL_DIR}")

endif()
