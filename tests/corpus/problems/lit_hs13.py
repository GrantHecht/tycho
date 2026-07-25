"""tests/corpus/problems/lit_hs13.py — E2 G0 literature tier.

Hock-Schittkowski problem 13 (HS13): a classic small NLP where the
linear-independence constraint qualification (LICQ) fails at the unique
solution (both the inequality constraint's gradient and the two active
bound constraints' gradients become linearly dependent there), so
KKT-based methods commonly report multiplier blow-up, an "acceptable but
not fully converged" solution, or slow tail convergence rather than a
clean CONVERGED flag.

Citation and verification trail
--------------------------------
Primary source: W. Hock and K. Schittkowski, "Test Examples for
Nonlinear Programming Codes," Lecture Notes in Economics and Mathematical
Systems, vol. 187, Springer, 1981 -- problem 13 of that collection.

Direct access to the original 1981 book was not available, so the
formulation (objective, constraint, AND the standard starting point) was
verified against the AMPL model file for this exact problem, fetched
directly: https://vanderbei.princeton.edu/ampl/nlmodels/hs/hs013.mod
(R. Vanderbei's AMPL translation of the full Hock-Schittkowski
collection, referenced as a standard machine-readable source for this
test set in K. Schittkowski's own "306 Test Problems for Nonlinear
Programming" user's guide, klaus-schittkowski.de/test_problems.pdf,
Section 1). The fetched file reads verbatim:

    var x {1..2} >= 0;
    minimize obj: (x[1] - 2)^2 + x[2]^2;
    subject to constr1: (1-x[1])^3 >= x[2];
    let x[1] := -2;
    let x[2] := -2;

i.e. min (x1-2)^2 + x2^2 s.t. (1-x1)^3 - x2 >= 0, x1 >= 0, x2 >= 0,
starting point (x1, x2) = (-2, -2). Known solution x* = (1, 0), f* = 1
(also given in the same file as a commented-out alternate starting
point). No correction to the brief's formulation was needed; this
confirms both the formula AND the standard start point the brief asked
to verify.

Encoding
--------
Static NLP via ``tychopy.solvers.OptimizationProblem`` (see
tests/corpus/README.md). PSIOPT's inequality convention is
``g(x) <= 0`` (doc-legacy/tutorials/PhaseGuide.rst), so:
  - "(1-x1)^3 - x2 >= 0"  ->  "x2 - (1-x1)^3 <= 0"
  - "x1 >= 0"             ->  "-x1 <= 0"
  - "x2 >= 0"             ->  "-x2 <= 0"

Observed on defaults 2026-07-16: ACCEPTABLE (harness status
"acceptable"), 77 iterations, objective 0.985042 (vs. the true f* = 1),
byte-identical across a --repeat 2 determinism check; final primal point
~(1.0075, ~0). This matches the expected pathology named in the Task 4
brief: the LICQ failure at x* keeps PSIOPT from reaching a clean
CONVERGED flag, landing instead in the "acceptable" band after a
comparatively high iteration count for a 2-variable problem.
"""

import tychopy as typy

vf = typy.vector_functions
solvs = typy.solvers
Args = vf.Arguments

TIER = "literature"
TIMEOUT = 30


def build_and_solve(configure) -> dict:
    """Construct, configure, and solve HS13.

    See tests/corpus/README.md for the full problem-module contract.
    """
    prob = solvs.OptimizationProblem()
    prob.set_vars([-2.0, -2.0])

    # min (x1 - 2)^2 + x2^2
    prob.add_objective((Args(2)[0] - 2.0) ** 2 + Args(2)[1] ** 2, [0, 1])

    # (1 - x1)^3 - x2 >= 0  ->  x2 - (1 - x1)^3 <= 0
    prob.add_inequal_con(Args(2)[1] - (1.0 - Args(2)[0]) ** 3, [0, 1])
    # x1 >= 0, x2 >= 0  ->  -x1 <= 0, -x2 <= 0
    prob.add_inequal_con(-Args(1)[0], [0])
    prob.add_inequal_con(-Args(1)[0], [1])

    configure(prob.optimizer)
    flag = prob.optimize()

    optimizer = prob.optimizer
    try:
        objective = float(optimizer.last_obj_val)
    except AttributeError:
        objective = None
    try:
        iterations = int(optimizer.last_iter_num)
    except AttributeError:
        iterations = None

    return {
        "flag": flag.name,
        "objective": objective,
        "iterations": iterations,
        "notes": "",
    }
