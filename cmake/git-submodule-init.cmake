add_custom_target(tycho-git-submodule-init COMMAND git submodule init
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
