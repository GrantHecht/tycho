file(GLOB_RECURSE ALL_SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/extensions/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/extensions/*.h
)

find_program(CLANG_FORMAT_EXE
    NAMES clang-format
    HINTS /opt/homebrew/opt/llvm/bin
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
