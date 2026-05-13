# Configured by CMake; usable directly by sphinx-build with fallback defaults
# when CMake variables are not yet substituted.
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

# Resolve to absolute paths so Sphinx finds _templates/_static whether
# conf.py is loaded from the source tree or the CMake build dir.
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
# Breathe renders qualified C++ symbol names as chains of cpp:identifier
# cross-references; nitpicky mode warns when a namespace component is
# undocumented. List shrinks as Doxygen coverage expands.
nitpick_ignore = [
    ("cpp:identifier", "tycho"),
    ("cpp:identifier", "tycho::integrators"),
]
# External libraries that appear in our function signatures — Doxygen is
# deliberately scoped to include/tycho/, so xrefs into these namespaces
# will never resolve. Extend as new external-lib xrefs surface from
# Breathe-rendered signatures; do not pre-populate.
nitpick_ignore_regex = [
    ("cpp:identifier", r"Eigen.*"),
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
# breathe_projects.tycho resolves to the CMake-substituted XML output dir;
# falls back to ../build/doxygen/xml for direct sphinx-build invocations.
breathe_projects = {"tycho": _resolve("@CMAKE_CURRENT_BINARY_DIR@/doxygen/xml", "../build/doxygen/xml")}
breathe_default_project = "tycho"
breathe_default_members = ("members", "undoc-members")
breathe_show_define_initializer = True
breathe_show_enumvalue_initializer = True
