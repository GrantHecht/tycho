add_executable(gf_eval_bench bench_gf_eval.cpp)

target_compile_options(gf_eval_bench PUBLIC ${COMPILE_FLAGS})
target_include_directories(gf_eval_bench PRIVATE ${INCLUDE_DIRS})
target_link_libraries(gf_eval_bench PRIVATE tycho_core)
target_precompile_headers(gf_eval_bench REUSE_FROM pch)

if(UNIX)
    target_link_options(gf_eval_bench PRIVATE ${LINKER_FLAGS})
endif()
