# docs/CMakeLists.txt — Sphinx + Doxygen site build.
# Activated when BUILD_SPHINX_DOCS=ON in the root CMakeLists.txt.

set(DOCS_SOURCE_DIR  "${CMAKE_CURRENT_SOURCE_DIR}/source")
set(DOCS_BUILD_DIR   "${CMAKE_CURRENT_BINARY_DIR}")
set(DOCS_HTML_DIR    "${DOCS_BUILD_DIR}/html")
set(DOCS_DOCTREE_DIR "${DOCS_BUILD_DIR}/.doctrees")
set(DOXYGEN_OUTPUT_DIR "${DOCS_BUILD_DIR}/doxygen")

if(NOT TARGET _tychopy)
    message(FATAL_ERROR
        "BUILD_SPHINX_DOCS=ON requires ENABLE_PYTHON_BINDINGS=ON "
        "(autodoc imports tychopy to render the Python API reference). "
        "Re-configure with -DENABLE_PYTHON_BINDINGS=ON.")
endif()

# 1) Substitute conf.py and Doxyfile into the build dir; Sphinx is invoked
#    with `-c <build-dir>` so it picks up the substituted copies.
configure_file(
    "${DOCS_SOURCE_DIR}/conf.py"
    "${DOCS_BUILD_DIR}/conf.py"
    @ONLY)
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in"
    "${DOCS_BUILD_DIR}/Doxyfile"
    @ONLY)
# Strict, VF-subsystem-scoped Doxygen config: enforces full docstring coverage
# (undocumented public symbol => hard failure). Consumed by the docs-build CI
# gate and by the local strict-check command in docs/source/contributing.
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.vf-strict.in"
    "${DOCS_BUILD_DIR}/Doxyfile.vf-strict"
    @ONLY)
# Strict, OC-subsystem-scoped Doxygen config: enforces full docstring coverage
# (undocumented public symbol => hard failure) for the OptimalControl subsystem.
# Consumed by the docs-build CI gate and by the local strict-check command in
# docs/source/contributing.
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.oc-strict.in"
    "${DOCS_BUILD_DIR}/Doxyfile.oc-strict"
    @ONLY)

# 2) Doxygen XML target.
file(GLOB_RECURSE _TYCHO_PUBLIC_HEADERS
    CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/include/tycho/*.h"
    "${PROJECT_SOURCE_DIR}/include/tycho/*.hpp")

add_custom_command(
    OUTPUT  "${DOXYGEN_OUTPUT_DIR}/xml/index.xml"
    COMMAND Doxygen::doxygen "${DOCS_BUILD_DIR}/Doxyfile"
    WORKING_DIRECTORY "${DOCS_BUILD_DIR}"
    DEPENDS ${_TYCHO_PUBLIC_HEADERS} "${DOCS_BUILD_DIR}/Doxyfile"
    COMMENT "Running Doxygen on include/tycho/"
    VERBATIM)

add_custom_target(tycho_doxygen
    DEPENDS "${DOXYGEN_OUTPUT_DIR}/xml/index.xml")

# 3) Sphinx HTML build.
add_custom_target(tycho_docs
    COMMAND ${Python_EXECUTABLE} -m sphinx
            -b html
            -c "${DOCS_BUILD_DIR}"
            -d "${DOCS_DOCTREE_DIR}"
            -W --keep-going -n
            "${DOCS_SOURCE_DIR}" "${DOCS_HTML_DIR}"
    DEPENDS tycho_doxygen _tychopy
    COMMENT "Building Sphinx HTML site")

# 4) Doctest runner.
add_custom_target(tycho_docs_doctest
    COMMAND ${Python_EXECUTABLE} -m sphinx
            -b doctest
            -c "${DOCS_BUILD_DIR}"
            -d "${DOCS_DOCTREE_DIR}"
            "${DOCS_SOURCE_DIR}" "${DOCS_BUILD_DIR}/doctest"
    DEPENDS tycho_doxygen _tychopy
    COMMENT "Running doctest builder")

# 5) Linkcheck.
add_custom_target(tycho_docs_linkcheck
    COMMAND ${Python_EXECUTABLE} -m sphinx
            -b linkcheck
            -c "${DOCS_BUILD_DIR}"
            "${DOCS_SOURCE_DIR}" "${DOCS_BUILD_DIR}/linkcheck"
    COMMENT "Checking external links (network-bound, slow)")

# 6) Local preview server.
add_custom_target(tycho_docs_serve
    COMMAND ${Python_EXECUTABLE} -m http.server 8000
            --directory "${DOCS_HTML_DIR}"
    DEPENDS tycho_docs
    COMMENT "Serving docs at http://localhost:8000 (Ctrl-C to stop)")

# 7) Clean.
add_custom_target(tycho_docs_clean
    COMMAND ${CMAKE_COMMAND} -E rm -rf
            "${DOCS_HTML_DIR}"
            "${DOCS_BUILD_DIR}/doctest"
            "${DOCS_BUILD_DIR}/linkcheck"
            "${DOXYGEN_OUTPUT_DIR}"
            "${DOCS_DOCTREE_DIR}"
    COMMENT "Cleaning generated docs output")

message(STATUS "Sphinx documentation targets: tycho_docs, tycho_doxygen, "
               "tycho_docs_doctest, tycho_docs_linkcheck, tycho_docs_serve, "
               "tycho_docs_clean")
