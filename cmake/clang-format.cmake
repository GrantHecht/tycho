file(GLOB_RECURSE ALL_SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/extensions/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/extensions/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/tycho/detail/*.h
)

find_program(CLANG_FORMAT_EXE
    NAMES clang-format clang-format-22 clang-format-21 clang-format-20 clang-format-19 clang-format-18
    HINTS /opt/homebrew/opt/llvm/bin /usr/bin
    DOC "clang-format executable"
)

if(CLANG_FORMAT_EXE)
    add_custom_target(clang-format
        COMMAND ${CLANG_FORMAT_EXE}
        -style=file
        -i
        ${ALL_SOURCE_FILES}
        COMMENT "Running clang-format (LLVM style) on source files"
    )
else()
    message(WARNING "clang-format not found; 'clang-format' target will not be available.")
endif()
