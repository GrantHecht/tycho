

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/src/OptimalControl)
include_directories(${PROJECT_SOURCE_DIR}/src/Solvers)
include_directories(${PROJECT_SOURCE_DIR}/src/TypeDefs)
include_directories(${PROJECT_SOURCE_DIR}/src/Utils)
include_directories(${PROJECT_SOURCE_DIR}/src/VectorFunctions)
include_directories(${PROJECT_SOURCE_DIR}/src/Integrators)

add_library(tycho_extensions STATIC
Tycho_Extensions.h
Tycho_Extensions.cpp)

add_dependencies(tycho_extensions pch_bindings)
target_compile_definitions(tycho_extensions PRIVATE TYCHO_PYTHON_BINDINGS)
target_include_directories(tycho_extensions PRIVATE ${INCLUDE_DIRS})

message(STATUS  ${INCLUDE_DIRS})


target_compile_options(tycho_extensions PUBLIC ${BINDING_COMPILE_FLAGS})
target_precompile_headers(tycho_extensions REUSE_FROM pch_bindings)

