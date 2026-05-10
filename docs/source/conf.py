# Configured by CMake; usable directly by sphinx-build with fallback defaults
# when CMake variables are not yet substituted.
import os
from pathlib import Path


def _resolve(value: str, default: str) -> str:
    """Return default if value is an unsubstituted @CMake_VAR@ placeholder."""
    return default if value.startswith("@") and value.endswith("@") else value


# -- Project information -----------------------------------------------------
project = _resolve("@PROJECT_NAME@", "Tycho")
copyright = "2026, Grant Hecht"
author = "Grant Hecht"
release = _resolve("@PROJECT_VERSION@", "0.1.dev")

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_parser",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx_togglebutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.doctest",
    "breathe",
]

# When CMake substitutes @DOCS_SOURCE_DIR@, conf.py lives in the build dir but
# templates/static assets live in the source tree. Resolve to absolute paths so
# Sphinx finds them regardless of where conf.py is loaded from. The fallback
# (relative path) is correct for direct sphinx-build invocations from the
# source tree, when @DOCS_SOURCE_DIR@ has not been substituted.
_DOCS_SOURCE_DIR = _resolve("@DOCS_SOURCE_DIR@", str(Path(__file__).parent.resolve()))

templates_path = [str(Path(_DOCS_SOURCE_DIR) / "_templates")]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# MyST configuration
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "attrs_inline",
    "substitution",
    "tasklist",
    "dollarmath",
]
myst_heading_anchors = 3

# Napoleon (NumPy-style docstrings)
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True

# Autodoc / autosummary
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"

# Intersphinx
intersphinx_mapping = {
    "python":     ("https://docs.python.org/3", None),
    "numpy":      ("https://numpy.org/doc/stable/", None),
    "scipy":      ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "spiceypy":   ("https://spiceypy.readthedocs.io/en/main/", None),
}

# Strict mode (enabled in CI; relaxed locally if desired)
nitpicky = True
# Breathe renders qualified C++ symbol names as a chain of cpp:identifier
# cross-references (one per namespace component). When the namespace itself is
# not yet documented in the cpp domain, Sphinx's nitpicky mode warns. Each
# entry below is explicit and narrowly scoped to one specific namespace; the
# list shrinks naturally as docstring coverage expands through Plans 2-5 of
# the docs strategy umbrella.
nitpick_ignore = [
    ("cpp:identifier", "tycho"),
    ("cpp:identifier", "tycho::integrators"),
]

# -- HTML output -------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = "Tycho"
html_static_path = [str(Path(_DOCS_SOURCE_DIR) / "_static")]
html_css_files = ["custom.css"]

html_theme_options = {
    "github_url": "https://github.com/GrantHecht/tycho",
    "use_edit_page_button": True,
    "show_toc_level": 2,
    "navigation_depth": 3,
    "header_links_before_dropdown": 6,
    "icon_links": [],
    "footer_start": ["copyright"],
    "footer_end": ["theme-version"],
    "navbar_align": "left",
    "show_prev_next": True,
}

html_context = {
    "github_user": "GrantHecht",
    "github_repo": "tycho",
    "github_version": "main",
    "doc_path": "docs/source",
}

# -- Copy button -------------------------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# -- Doctest -----------------------------------------------------------------
doctest_global_setup = """
import numpy as np
"""

# -- Breathe (Doxygen XML → Sphinx) -----------------------------------------
# breathe_projects.tycho is set via:
#   - CMake configure_file substitution (in Task 4) → "@CMAKE_CURRENT_BINARY_DIR@/doxygen/xml"
#   - `-D breathe_projects.tycho=<path>` on the sphinx-build CLI for direct invocations
#   - Default fallback to "../build/doxygen/xml" relative to docs/source/, useful for
#     post-CMake local previews
breathe_projects = {"tycho": _resolve("@CMAKE_CURRENT_BINARY_DIR@/doxygen/xml", "../build/doxygen/xml")}
breathe_default_project = "tycho"
breathe_default_members = ("members", "undoc-members")
breathe_show_define_initializer = True
breathe_show_enumvalue_initializer = True
