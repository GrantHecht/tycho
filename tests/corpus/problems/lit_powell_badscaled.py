"""tests/corpus/problems/lit_powell_badscaled.py — E2 G0 literature tier.

Powell's badly scaled function, restated as a zero-objective equality-
constrained FEASIBILITY problem (rather than the more commonly quoted
sum-of-squares unconstrained form): the two residuals differ in
magnitude by a factor of ~1e4, a classic scaling stress test for
Newton-based methods (the Jacobian's rows have wildly different natural
scales, and becomes singular when x1 == x2).

Citation and verification trail
--------------------------------
Primary source: J. J. More, B. S. Garbow, and K. E. Hillstrom, "Testing
Unconstrained Optimization Software," ACM Transactions on Mathematical
Software, 7(1):17-41, 1981 -- problem #3 ("Powell badly scaled function")
of that collection. Direct access to the 1981 ACM TOMS article was not
available, so the formulation was cross-checked against two independent
secondary sources, both fetched directly and in agreement:

1. https://al-roomi.org/benchmarks/unconstrained/2-dimensions/65-powell-s-badly-scaled-function
   (a benchmark-function reference site), which states the two
   components f1 = 10000*x1*x2 - 1, f2 = exp(-x1) + exp(-x2) - 1.0001
   (with the overall unconstrained objective being f1^2 + f2^2) and cites
   More-Garbow-Hillstrom (1981) by name as its source.
2. A web search of the standard MINPACK/CUTEst framing of this same test
   problem (multiple independent summaries, including discussion of the
   x0 = (0, 1) vs. x0 = (1, 1) starting-point distinction for Newton's
   method on this system) corroborates x0 = (0, 1) as the standard
   starting point for this problem across the literature.

The brief's stated formulation (10^4*x1*x2 - 1 = 0,
exp(-x1) + exp(-x2) - 1.0001 = 0, start (0, 1), zero objective) matches
both of these exactly; no correction was needed.

Encoding
--------
Static NLP via ``tychopy.solvers.OptimizationProblem`` (see
tests/corpus/README.md). Following the same pattern as
deg_zero_objective.py in the degenerate tier, NO ``add_objective`` call is
made at all -- this is pure feasibility, exercising PSIOPT on a
badly-conditioned pair of equality constraints with nothing to minimize.

Observed on defaults 2026-07-16: CONVERGED, 22 iterations, objective 0
(as expected for a feasibility problem), byte-identical across a
--repeat 2 determinism check. This is a FINDING TO REPORT: despite the
~1e4 scale disparity between the two residuals that makes this problem a
standard Newton-method scaling stress test, PSIOPT's defaults handle it
without any difficulty -- no scaling stress genuinely manifests here,
mirroring the analogous finding for deg_zero_objective.py in the
degenerate tier (a named pathology that turns out not to bite this
particular solver/problem combination).
"""

import tychopy as typy

vf = typy.vector_functions
solvs = typy.solvers
Args = vf.Arguments

TIER = "literature"
TIMEOUT = 30
SOLVE_MODE = "optimize"


def build():
    """Construct Powell's badly scaled system (unsolved).

    See tests/corpus/README.md for the full problem-module contract.
    """
    prob = solvs.OptimizationProblem()
    prob.set_vars([0.0, 1.0])

    # 1e4 * x1 * x2 - 1 = 0
    prob.add_equal_con(1.0e4 * Args(2)[0] * Args(2)[1] - 1.0, [0, 1])
    # exp(-x1) + exp(-x2) - 1.0001 = 0
    prob.add_equal_con((-Args(2)[0]).exp() + (-Args(2)[1]).exp() - 1.0001, [0, 1])
    # Deliberately no objective term of any kind (pure feasibility).

    return prob
