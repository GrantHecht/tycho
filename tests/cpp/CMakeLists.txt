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

include(GoogleTest)

# Helper — mirrors the pattern from examples/cpp_examples/brachistochrone/CMakeLists.txt
function(tycho_add_test test_name test_source)
    add_executable(${test_name} ${test_source})
    target_compile_options(${test_name} PRIVATE ${COMPILE_FLAGS})
    target_include_directories(${test_name} PRIVATE ${TYCHO_INCLUDES} ${INCLUDE_DIRS})
    target_link_libraries(${test_name} PRIVATE tycho_core ${LINK_LIBS} GTest::gtest_main)
    target_precompile_headers(${test_name} REUSE_FROM pch)
    if(UNIX)
        target_link_options(${test_name} PRIVATE ${LINKER_FLAGS})
    endif()
    gtest_discover_tests(${test_name})
endfunction()

tycho_add_test(test_astro test_astro.cpp)
tycho_add_test(test_type_storage test_type_storage.cpp)
tycho_add_test(test_vector_functions test_vector_functions.cpp)
