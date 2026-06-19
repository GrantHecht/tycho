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
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "spiceypy": ("https://spiceypy.readthedocs.io/en/main/", None),
}

# Strict mode (enabled in CI; relaxed locally if desired)
nitpicky = True
# Breathe renders qualified C++ symbol names as chains of cpp:identifier
# cross-references; nitpicky mode warns when a namespace component or a
# template parameter is undocumented. These are auto-generated signature
# xrefs (not authored links), so they are absorbed by the regex list below
# rather than enumerated one-by-one.
nitpick_ignore = []
# External libraries that appear in our function signatures — Doxygen is
# deliberately scoped to include/tycho/, so xrefs into these namespaces
# will never resolve. Extend as new external-lib xrefs surface from
# Breathe-rendered signatures; do not pre-populate.
nitpick_ignore_regex = [
    # C++: Breathe renders qualified internal symbol names and template
    # instantiations (e.g. ``tycho::vf::Norm``,
    # ``DenseFunctionBase<Func, IR, OR>``) and bare template-parameter names
    # (``Derived``, ``Scalar``) as cpp:identifier xrefs in member signatures.
    # Doxygen coverage of the VF subsystem is curated, not exhaustive, so many
    # never resolve. Absorb the whole class rather than chase each one.
    # NOTE: nitpick_ignore_regex patterns are full-matched against the target,
    # so trailing context must be absorbed (use ``.*<.*`` not ``.*<.*>``).
    ("cpp:identifier", r"tycho(::.*)?"),
    ("cpp:identifier", r"Eigen(::.*)?"),
    ("cpp:identifier", r".*<.*"),  # any template instantiation, incl. ``::value`` tails
    ("cpp:identifier", r".*::.*"),  # any unqualified-rendered nested name (Base::IRC, …)
    ("cpp:identifier", r"Dense(First|Second)Derivatives.*"),
    ("cpp:identifier", r"DenseDerivativeMode.*"),
    ("cpp:identifier", r"DenseVectorFunction"),
    ("cpp:identifier", r"Kepler_Impl.*"),
    ("cpp:identifier", r"SZ_\w+.*"),
    ("cpp:identifier", r"return_type_t.*"),
    ("cpp:identifier", r"(Derived|Scalar|Func\d*|IR\d*|OR\d*|ST\d*|JMode)"),
    # C++ optimal-control reference page: Breathe renders the member signatures
    # of the ``tycho::oc`` phase/problem types with bare (unqualified) internal
    # typedef names and prose references that Doxygen's curated scope does not
    # document as resolvable cross-reference targets. These are auto-generated
    # signature xrefs, not authored links, so they are absorbed by class rather
    # than enumerated. ``ScaleType``/``RegionType``/``VarIndexType`` are
    # phase-internal index typedefs; ``MeshIterateInfo`` and ``PSIOPT`` are
    # cross-subsystem types not documented under the OC page's Doxygen subset;
    # a lone ``vf`` is a bare namespace fragment from a return-type signature.
    ("cpp:identifier", r"(ScaleType|RegionType|VarIndexType|MeshIterateInfo|PSIOPT|vf)"),
    # Python: autodoc'ing the compiled tychopy.vector_functions module surfaces
    # the nanobind-bound VF expression types and the NumPy-style docstring prose
    # pseudo-types (``shape (n,)``, ``tuple[numpy.ndarray, ...]``, ``(int, int)``,
    # ``array_like``, descriptive Returns phrases, etc.) as py:class
    # cross-references that never resolve. A valid Python cross-reference target
    # is a pure dotted identifier (``module.ClassName``) — it can never contain
    # whitespace, a paren, a comma, or a bracket. So any target containing one of
    # those characters is necessarily docstring prose (often a comma-split type
    # fragment), and ignoring the whole character class is SAFE: it cannot mask a
    # real broken reference, which would still be a bare dotted identifier.
    ("py:class", r".*[\s(),\[\]].*"),
    # Identifier-shaped pseudo-types that slip past the character-class rule
    # (bare words autodoc emits from nanobind signatures / NumPy "type" lines).
    ("py:class", r"_tychopy\.vector_functions\..*"),
    ("py:class", r"(VectorFunction|ScalarFunction|GenericFunction|Segment\d*|Element|Arguments)"),
    ("py:class", r"array_like"),
    ("py:class", r"ndarray"),
    ("py:class", r"callable"),  # NumPy "type" line on Py/Numba function bindings
    ("py:class", r"dtype.*"),
    ("py:class", r"Sign"),  # docstring Returns-prose pseudo-type (sign())
    # Python optimal-control reference page: autodoc'ing the compiled
    # tychopy.optimal_control module surfaces the same class of NumPy-style
    # docstring prose pseudo-types as the VF page, plus a cross-module base.
    # ``np.ndarray`` is the abbreviated NumPy alias used in a Returns line;
    # ``sequence``, ``shape``, ``numsegs``, and ``Integrator`` are bare-word
    # prose pseudo-types (no whitespace, so the character-class rule above does
    # not catch them); ``_tychopy.solvers.OptimizationProblemBase`` is the
    # nanobind base class, documented under the solvers subsystem rather than
    # this page's Doxygen subset.
    ("py:class", r"np\.ndarray"),
    ("py:class", r"(sequence|shape|numsegs|Integrator)"),
    ("py:class", r"_tychopy\.solvers\..*"),
    # Dunder/method names mentioned in docstring prose (e.g. a class blurb that
    # references ``__and__``/``__or__``, or ``eval``) are not autodoc targets.
    # ``solve``/``optimize`` are prose method mentions in OC docstrings that
    # point at methods not exposed on the documented class.
    ("py:meth", r"(__\w+__|eval|solve|optimize)"),
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
from tychopy import vector_functions as vf
"""

# -- Breathe (Doxygen XML → Sphinx) -----------------------------------------
# breathe_projects.tycho resolves to the CMake-substituted XML output dir;
# falls back to ../build/doxygen/xml for direct sphinx-build invocations.
breathe_projects = {
    "tycho": _resolve("@CMAKE_CURRENT_BINARY_DIR@/doxygen/xml", "../build/doxygen/xml")
}
breathe_default_project = "tycho"
# Opt-in member rendering: a bare ``doxygenstruct`` renders the class brief and
# signature only. Reference pages add ``:members:`` explicitly on the headline
# types whose method surface is worth documenting. This keeps the curated
# expression/scaling catalog compact and avoids the nested-member duplicate
# declarations that full ``undoc-members`` expansion produces across the deeply
# inherited VF hierarchy.
breathe_default_members = ()
breathe_show_define_initializer = True
breathe_show_enumvalue_initializer = True

# Breathe renders Doxygen inheritance graphs ("Subclassed by …"), ``@ref``
# links, and member back-references as Sphinx ``:ref:`` cross-references to
# auto-generated Doxygen labels (e.g. ``structtycho_1_1vf_1_1VectorFunction``).
# Those labels only exist when *every* referenced compound — including types in
# other subsystems (``tycho::oc``, ``tycho::astro``) and undocumented ``_Impl``
# helpers — is itself rendered on a page, which a curated reference never does.
# The resulting "undefined label" warnings are unavoidable noise from Breathe's
# rendering, so the ``ref.ref`` category is suppressed.
#
# TRADE-OFF: ``ref.ref`` is also the category of a broken *authored* MyST
# ``{ref}`` label link, so those are NOT validated while this suppression is in
# place. To keep authored cross-references checked, this project deliberately
# avoids the ``{ref}`` role: page-to-page links use ``{doc}`` (validated under
# ``ref.doc``) and API links use ``:py:...:`` / ``:cpp:...:`` roles (validated
# under ``ref.class``/``ref.cpp``). Prefer those roles over ``{ref}`` in new
# pages so the suppression cannot mask a real authored-link error.
suppress_warnings = ["ref.ref"]
