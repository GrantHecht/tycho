# Pure C++ extension library — no Python dependency.
# Extension ODE models (CR3BPAD, ModifiedDynamicsAD, etc.) live in Tycho_Extensions.h
# and are usable from standalone C++ programs without pulling in the Python C API.
add_library(tycho_extensions STATIC Tycho_Extensions.h Tycho_Extensions.cpp)
target_precompile_headers(tycho_extensions REUSE_FROM pch)
target_link_libraries(tycho_extensions PUBLIC tycho)
target_include_directories(tycho_extensions PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${INCLUDE_DIRS})
target_compile_options(tycho_extensions PRIVATE ${COMPILE_FLAGS})

if(ENABLE_PYTHON_BINDINGS)
    # Python binding layer for extensions — compiled WITH TYCHO_PYTHON_BINDINGS.
    # Contains TychoBind<> specializations and ExtensionsBuild().
    # FunctionRegistry, TychoBind<T>, and binding helpers are provided by the
    # pch_bindings PCH (pch_nb.h -> FunctionRegistry.h, VectorFunctionsBind.h).
    add_library(tycho_extensions_bindings STATIC Bindings/TychoExtensionsBind.cpp)
    target_compile_definitions(tycho_extensions_bindings PRIVATE TYCHO_PYTHON_BINDINGS)
    target_precompile_headers(tycho_extensions_bindings REUSE_FROM pch_bindings)
    target_link_libraries(tycho_extensions_bindings PUBLIC tycho_extensions)
    target_include_directories(tycho_extensions_bindings PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${INCLUDE_DIRS})
    target_compile_options(tycho_extensions_bindings PUBLIC ${BINDING_COMPILE_FLAGS})
endif()
