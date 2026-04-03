add_subdirectory(static/brachistochrone)
add_subdirectory(static/zermelo)
add_subdirectory(static/hypersens)
add_subdirectory(static/delta3_launch)

add_subdirectory(builder/brachistochrone)
add_subdirectory(builder/zermelo)
add_subdirectory(builder/hypersens)
add_subdirectory(builder/delta3_launch)

# C++ examples consume 4-8 GB RAM each — restrict to the heavy_compile pool.
foreach(_ex
    brachistochrone_cpp zermelo_cpp hypersens_cpp delta3_launch_cpp
    brachistochrone_builder zermelo_builder hypersens_builder delta3_launch_builder)
    if(TARGET ${_ex})
        set_property(TARGET ${_ex} PROPERTY JOB_POOL_COMPILE heavy_compile)
    endif()
endforeach()
