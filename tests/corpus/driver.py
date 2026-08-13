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

- ``psiopt``: ``configure(prob.optimizer)`` then the entry point selected by
  ``call_shape`` -- ``"module"`` (default) runs the module's declared
  ``SOLVE_MODE`` (``getattr(prob, module.SOLVE_MODE)()``), exactly as the
  pre-split contract did; ``"optimize"`` always runs ``prob.optimize()``
  instead, regardless of ``SOLVE_MODE``, for cross-backend comparability with
  the ipopt backend's single-solve mapping below. The returned ``"flag"`` is
  the raw ``ConvergenceFlags`` member name (e.g. ``"CONVERGED"``) -- this
  driver does not map it to a status string; ``scripts/run_corpus.py``'s
  parent process already owns that mapping (``_FLAG_TO_STATUS``) and
  continues to perform it unchanged, so duplicating it here would risk a
  second, drifting copy of the flag vocabulary.
- ``ipopt``: sets ``prob.nlp_solver = NLPSolvers.ipopt`` and forwards
  ``backend_options`` (verbatim strings) into ``prob.ipopt_options``, then
  always calls ``prob.optimize()`` -- the ipopt backend performs a single
  NLP solve regardless of a module's ``SOLVE_MODE`` or ``call_shape`` (the
  staging modes have no Ipopt analog, per
  ``OptimizationProblemBase.nlp_solver``'s own docstring), so ``SOLVE_MODE``
  and ``call_shape`` are recorded in ``notes`` instead of being honored.
  ``prob.optimize()`` still returns a ``ConvergenceFlags`` member under this
  backend (verified in ``tychopy/test/test_Solvers/test_nlp_solver_backend.py``),
  so ``"flag"`` stays uniform across both backends. Objective/iterations come
  from ``prob.last_ipopt_result`` (``optimizer.last_obj_val`` /
  ``last_iter_num`` reflect only the most recent interior-point solver run and are left
  untouched by an ipopt-backend solve, per the same docstring) --
  ``configure``/``POST_SOLVE`` are not used on this path.
"""

import tychopy.solvers as solvs


def run(module, configure, backend="psiopt", backend_options=None, call_shape="module"):
    """Build, configure, solve, and normalize the result for one problem module.

    ``call_shape`` selects which entry point the psiopt backend invokes:
    ``"module"`` (default) runs the module's declared ``SOLVE_MODE``, today's
    behavior; ``"optimize"`` always runs a single ``optimize()`` call instead,
    for cross-backend comparability with the ipopt backend's single-solve
    mapping (see the ``ipopt`` branch below, which always runs a single solve
    regardless of ``call_shape``).

    Returns the harness result-dict schema: ``{"flag", "objective",
    "iterations", "notes"}``, plus ``"backend"`` and ``"call_shape"``.
    """
    prob = module.build()

    if backend == "psiopt":
        configure(prob.optimizer)
        solve_attr = "optimize" if call_shape == "optimize" else module.SOLVE_MODE
        flag = getattr(prob, solve_attr)()

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
        module_notes = getattr(module, "NOTES", "")
        ipopt_notes = (
            f"ipopt backend: single-solve mapping of SOLVE_MODE="
            f"{module.SOLVE_MODE!r} (call_shape={call_shape!r} — always a "
            f"single solve on this backend); ipopt status={info.status!r}"
        )
        result = {
            "flag": flag.name,
            "objective": float(info.objective),
            "iterations": int(info.iterations),
            "notes": f"{module_notes}; {ipopt_notes}" if module_notes else ipopt_notes,
        }
    else:
        raise ValueError(f"unknown backend: {backend!r}")

    result["call_shape"] = call_shape
    result["backend"] = backend
    return result
