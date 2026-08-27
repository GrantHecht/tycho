"""Shared solve driver for corpus problem modules.

A problem module provides ``TIER``, ``TIMEOUT``, ``SOLVE_CALL``, and
``build() -> problem`` (fully constructed, unsolved -- everything the old
``build_and_solve`` did up to but excluding the ``configure(...)`` call). A
module may also define, at most one of:

- ``NOTES`` -- a static string (the same static text the old
  ``build_and_solve`` used to hard-code into its returned dict's
  ``"notes"``; most modules have none, i.e. ``""``).
- ``POST_SOLVE(prob) -> str`` -- a hook called after a psiopt-backend solve
  to compute notes that depend on post-solve state (e.g.
  ``hard_hypersens_stiff``'s ``phase.mesh_converged`` check). Only
  consulted on the psiopt path -- see "Backend semantics" below.

It may also, optionally, define ``configure(engine)`` -- a hook that applies
any engine settings the *problem itself* owns (tolerances, line-search mode,
thread count, ...). This replaces the earlier pattern of tweaking
``phase.optimizer`` directly inside ``build()``: the engine no longer exists
until the driver constructs it, so a problem-specific tuning that used to be
baked into the phase's own (always-present) optimizer now has to wait for
the driver to hand it an engine. ``module.configure`` runs BEFORE the
harness's own ``configure`` callback (the ``--config KEY=VALUE`` overrides),
so a CLI override still wins over whatever a problem module sets -- the same
precedence the earlier code had (build()'s inline tweaks, then the harness's
``configure(prob.optimizer)`` applied on top).

``SOLVE_CALL`` is a plain kwargs dict for ``prob.solve(engine, **SOLVE_CALL)``
(e.g. ``dict(mode="optimal")`` or ``dict(mode="optimal", presolve=True)``),
plus two reserved keys the driver itself consumes and never forwards
verbatim to a single ``solve()`` call -- see ``_dispatch_psiopt_solve``
below for exactly how each is expressed as sequential ``solve()`` calls:

- ``presolve`` (bool, default False) -- an unconditional Feasible stage
  before the main call, mirroring the retired ``solve_optimize()`` /
  ``solve_optimize_solve()`` combo methods' leading Solve phase. Run as its
  own ``solve(engine, mode="feasible")`` call rather than forwarded to
  ``solve()``'s own ``presolve=`` argument -- see
  ``_dispatch_psiopt_solve``'s docstring for why.
- ``feasible_fallback`` (bool, default False) -- if the main call above
  does not converge, run a further solve with ``mode="feasible"`` and
  ``warm=<main-call result>``, mirroring the retired ``optimize_solve()`` /
  ``solve_optimize_solve()`` combo methods' trailing *conditional* Solve
  phase (a Feasible attempt only runs when the preceding Optimal attempt
  didn't converge) -- see ``examples/python_examples/HyperSens.py`` for
  the same pattern used directly by a caller.

  The old five-name ``SOLVE_MODE`` vocabulary maps onto this convention as:

  - ``"solve"``                -> ``dict(mode="feasible")``
  - ``"optimize"``              -> ``dict(mode="optimal")``
  - ``"solve_optimize"``        -> ``dict(mode="optimal", presolve=True)``
  - ``"optimize_solve"``        -> ``dict(mode="optimal", feasible_fallback=True)``
  - ``"solve_optimize_solve"``  -> ``dict(mode="optimal", presolve=True, feasible_fallback=True)``

This driver constructs the engine, applies configuration, dispatches the
selected backend, runs the solve, and normalizes the result dict -- one
place instead of seventeen. It reproduces exactly the flag/objective/
iterations extraction every pre-split module's ``build_and_solve`` tail did
inline (verified identical across all 17 modules): ``flag.name``;
``float(engine.last_obj_val)`` / ``int(engine.last_iter_num)``, each
independently guarded by ``try/except AttributeError: ... = None``.

Backend semantics:

- ``psiopt``: constructs an ``InteriorPointSolver`` engine, applies
  ``module.configure`` (if defined) then the harness's ``configure``, then
  dispatches the solve selected by ``call_shape`` -- ``"module"`` (default)
  runs the module's declared ``SOLVE_CALL`` (with the ``feasible_fallback``
  conditional chain above), exactly as the pre-split contract's
  ``SOLVE_MODE`` did; ``"optimize"`` always runs a single
  ``dict(mode="optimal")`` call instead, regardless of ``SOLVE_CALL``, for
  cross-backend comparability with the ipopt backend's single-solve mapping
  below. The returned ``"flag"`` is the raw ``ConvergenceFlags`` member name
  (e.g. ``"CONVERGED"``) -- this driver does not map it to a status string;
  ``scripts/run_corpus.py``'s parent process already owns that mapping
  (``_FLAG_TO_STATUS``) and continues to perform it unchanged, so
  duplicating it here would risk a second, drifting copy of the flag
  vocabulary.
- ``ipopt``: constructs an ``IpoptSolver`` engine and forwards
  ``backend_options`` (verbatim strings) into its ``.options`` dict, then
  always runs a single ``prob.solve(engine)`` -- the ipopt backend performs
  a single NLP solve regardless of a module's ``SOLVE_CALL`` or
  ``call_shape`` (the staging shapes have no Ipopt analog), so ``SOLVE_CALL``
  and ``call_shape`` are recorded in ``notes`` instead of being honored.
  ``"flag"`` stays uniform across both backends (the returned
  ``SolveResult.flag`` either way). Objective/iterations come from the
  ``SolveResult`` itself (``result.objective()`` / ``result.iterations()``)
  -- ``engine.last_obj_val`` / ``last_iter_num`` reflect only the most
  recent ``InteriorPointSolver`` run and are irrelevant to an
  ``IpoptSolver``-driven solve -- ``module.configure``/``POST_SOLVE`` are
  not used on this path.
"""

import tychopy.solvers as solvs


def _dispatch_psiopt_solve(prob, engine, solve_call):
    """Run one problem's psiopt-backend solve per its SOLVE_CALL.

    Expresses the old combo-method chain shapes (``solve_optimize`` /
    ``optimize_solve`` / ``solve_optimize_solve``) as explicit sequential
    ``solve()`` calls on the same engine -- see the module docstring's
    ``SOLVE_CALL`` section for the exact mapping.

    ``presolve=True`` is handled as its own unconditional ``solve(engine,
    mode="feasible")`` call BEFORE the main call, rather than being
    forwarded to ``solve()``'s own ``presolve=`` convenience path: that
    path hands the presolve stage's ending point to the main stage through
    the public, validated ``WarmStartData`` channel, which raises if that
    point includes a non-finite value -- exactly what a corpus problem's
    presolve stage can legitimately produce while genuinely diverging
    (observed on ``hard_zermelo_wrongbasin``). Running the presolve stage
    as its own ``solve()`` call instead relies on the same in-place primal
    continuation the retired ``run_phase_sequence({SOE, OPT})`` used (see
    the frozen ``psiopt/src/psiopt.cpp``), which tolerates it: a
    ``solve()`` call with no explicit ``warm=`` picks up wherever the
    problem's own active variables already sit, the same continuity a
    bare re-solve without retranscription always had.
    """
    call = dict(solve_call)
    feasible_fallback = call.pop("feasible_fallback", False)
    if call.pop("presolve", False):
        prob.solve(engine, mode="feasible")
    result = prob.solve(engine, **call)
    if feasible_fallback and not result:
        result = prob.solve(engine, mode="feasible", warm=result)
    return result


def run(module, configure, backend="psiopt", backend_options=None, call_shape="module"):
    """Build, configure, solve, and normalize the result for one problem module.

    ``call_shape`` selects which solve the psiopt backend runs: ``"module"``
    (default) runs the module's declared ``SOLVE_CALL``, today's behavior;
    ``"optimize"`` always runs a single Optimal-mode solve instead, for
    cross-backend comparability with the ipopt backend's single-solve
    mapping (see the ``ipopt`` branch below, which always runs a single
    solve regardless of ``call_shape``).

    Returns the harness result-dict schema: ``{"flag", "objective",
    "iterations", "notes"}``, plus ``"backend"`` and ``"call_shape"``.
    """
    prob = module.build()

    if backend == "psiopt":
        engine = solvs.IPM()
        module_configure = getattr(module, "configure", None)
        if module_configure is not None:
            module_configure(engine)
        configure(engine)

        solve_call = (
            {"mode": "optimal"} if call_shape == "optimize" else module.SOLVE_CALL
        )
        result = _dispatch_psiopt_solve(prob, engine, solve_call)
        flag = result.flag

        try:
            objective = float(engine.last_obj_val)
        except AttributeError:
            objective = None
        try:
            iterations = int(engine.last_iter_num)
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
        engine = solvs.IpoptSolver()
        engine.options = dict(backend_options or {})
        solve_result = prob.solve(engine)
        flag = solve_result.flag

        module_notes = getattr(module, "NOTES", "")
        ipopt_notes = (
            f"ipopt backend: single-solve mapping of SOLVE_CALL="
            f"{module.SOLVE_CALL!r} (call_shape={call_shape!r} — always a "
            f"single solve on this backend)"
        )
        result = {
            "flag": flag.name,
            "objective": float(solve_result.objective()),
            "iterations": int(solve_result.iterations()),
            "notes": f"{module_notes}; {ipopt_notes}" if module_notes else ipopt_notes,
        }
    else:
        raise ValueError(f"unknown backend: {backend!r}")

    result["call_shape"] = call_shape
    result["backend"] = backend
    return result
