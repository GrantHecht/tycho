Tutorials
=========

The examples below are meant to introduce ASSET in two contexts: strictly as a python package, and as an open-source library that you may extend.
New readers should begin at the beginning, since it's important to understand the library architecture and organization before using it to build anything.

.. warning::

   These tutorials largely pre-date the Tycho Python-API snake_case rename.
   Inline code samples often use camelCase identifiers (``addBoundaryValue``,
   ``XVar``, ``setControlMode``, ``vf.Stack``) and the legacy
   ``LinkConstraint`` / ``LinkObjective`` / ``LinkFlags`` link API. The
   bindings now expose only snake_case forms (``add_boundary_value``,
   ``x_var``, ``set_control_mode``, ``stack``, ``add_link_equal_con``).
   Copy-pasting samples will fail with ``AttributeError``. For runnable,
   current code see the scripts under ``examples/python_examples/``;
   heavily-affected pages also carry their own per-file warning banners.


.. toctree::
    :maxdepth: 2

    Install
    VectorFunctionGuide
    ODEGuide
    IntegratorGuide
    PhaseGuide
    OptimalControlProblem
    PSIOPT
    AutoMeshGuide
    NewInterface
    OptimalControlUtilites
    

    

