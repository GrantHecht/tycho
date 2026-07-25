"""Shared solve driver for corpus problem modules.

A problem module provides ``TIER``, ``TIMEOUT``, ``SOLVE_MODE``, and
``build() -> problem`` (fully constructed, unsolved -- everything the old
``build_and_solve`` did up to but excluding the ``configure(...)`` call).
A module may also define, at most one of:

- ``NOTES`` -- a static string (the same static text the old
  ``build_and_solve`` used to hard-code into its returned dict's
  ``"notes"``; most modules have none, i.e. ``""``).
- ``POST_SOLVE(prob) -> str`` -- a hook called after a psiopt-backend solve
  to compute notes that depend on post-solve state (e.g.
  ``hard_hypersens_stiff``'s ``phase.mesh_converged`` check). Only
  consulted on the psiopt path -- see "Backend semantics" below.

This driver applies configuration, dispatches the selected backend, runs
the solve, and normalizes the result dict -- one place instead of
seventeen. It reproduces exactly the flag/objective/iterations extraction
every pre-split module's ``build_and_solve`` tail did inline (verified
identical across all 17 modules): ``flag.name``;
``float(optimizer.last_obj_val)`` / ``int(optimizer.last_iter_num)``, each
independently guarded by ``try/except AttributeError: ... = None``.

Backend semantics:

- ``psiopt``: ``configure(prob.optimizer)`` then the module's
  ``SOLVE_MODE`` entry point (``getattr(prob, module.SOLVE_MODE)()``),
  exactly as the pre-split contract did. The returned ``"flag"`` is the
  raw ``ConvergenceFlags`` member name (e.g. ``"CONVERGED"``) -- this
  driver does not map it to a status string; ``scripts/run_corpus.py``'s
  parent process already owns that mapping (``_FLAG_TO_STATUS``) and
  continues to perform it unchanged, so duplicating it here would risk a
  second, drifting copy of the flag vocabulary.
- ``ipopt``: sets ``prob.nlp_solver = NLPSolvers.ipopt`` and forwards
  ``backend_options`` (verbatim strings) into ``prob.ipopt_options``, then
  always calls ``prob.optimize()`` -- the ipopt backend performs a single
  NLP solve regardless of a module's ``SOLVE_MODE`` (the staging modes
  have no Ipopt analog, per ``OptimizationProblemBase.nlp_solver``'s own
  docstring), so ``SOLVE_MODE`` is recorded in ``notes`` instead of being
  honored. ``prob.optimize()`` still returns a ``ConvergenceFlags`` member
  under this backend (verified in
  ``tychopy/test/test_Solvers/test_nlp_solver_backend.py``), so
  ``"flag"`` stays uniform across both backends. Objective/iterations come
  from ``prob.last_ipopt_result`` (``optimizer.last_obj_val`` /
  ``last_iter_num`` reflect only the most recent PSIOPT run and are left
  untouched by an ipopt-backend solve, per the same docstring) --
  ``configure``/``POST_SOLVE`` are not used on this path.
"""

import tychopy.solvers as solvs


def run(module, configure, backend="psiopt", backend_options=None):
    """Build, configure, solve, and normalize the result for one problem module.

    Returns the harness result-dict schema: ``{"flag", "objective",
    "iterations", "notes"}``, plus ``"backend"``.
    """
    prob = module.build()

    if backend == "psiopt":
        configure(prob.optimizer)
        flag = getattr(prob, module.SOLVE_MODE)()

        optimizer = prob.optimizer
        try:
            objective = float(optimizer.last_obj_val)
        except AttributeError:
            objective = None
        try:
            iterations = int(optimizer.last_iter_num)
        except AttributeError:
            iterations = None

        post_solve = getattr(module, "POST_SOLVE", None)
        notes = (
            post_solve(prob) if post_solve is not None else getattr(module, "NOTES", "")
        )

        result = {
            "flag": flag.name,
            "objective": objective,
            "iterations": iterations,
            "notes": notes,
        }
    elif backend == "ipopt":
        if not solvs.ipopt_available():
            raise RuntimeError(
                "--backend ipopt requires a Tycho build configured with -DENABLE_IPOPT=ON"
            )
        prob.nlp_solver = solvs.NLPSolvers.ipopt
        prob.ipopt_options = dict(backend_options or {})
        flag = prob.optimize()

        info = prob.last_ipopt_result
        result = {
            "flag": flag.name,
            "objective": float(info.objective),
            "iterations": int(info.iterations),
            "notes": (
                f"ipopt backend: single-solve mapping of SOLVE_MODE="
                f"{module.SOLVE_MODE!r}; ipopt status={info.status!r}"
            ),
        }
    else:
        raise ValueError(f"unknown backend: {backend!r}")

    result["backend"] = backend
    return result
