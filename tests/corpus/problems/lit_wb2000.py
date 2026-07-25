"""tests/corpus/problems/lit_wb2000.py — E2 G0 literature tier.

The Wächter-Biegler (2000) counterexample: a small, well-posed NLP
deliberately constructed so that a class of line-search primal-dual
interior-point methods converges to a feasible-looking but NON-stationary
point ("jamming"), rather than to the true optimum, from a specific start.

Citation and verification trail
--------------------------------
Primary source: A. Wächter and L. T. Biegler, "Failure of global
convergence for a class of interior point methods for nonlinear
programming," Mathematical Programming, 88(3):565-574, 2000
(DOI 10.1007/PL00011386). This journal article and its optimization-online
preprint (https://optimization-online.org/2000/08/189/) were both
consulted directly; neither exposes the full problem text through
un-authenticated web access (Springer login-gated; the preprint page is
abstract-only with no PDF link).

The problem statement AND Wächter-Biegler's own sufficient-condition
theorem were verified instead against a secondary source that quotes them
verbatim and was fetched and read in full: H. Y. Benson, D. F. Shanno,
and R. J. Vanderbei, "Interior-Point Methods for Nonconvex Nonlinear
Programming: Jamming and Comparative Numerical Testing," Princeton
ORFE-00-02 technical report (revised 2000-08-28), Section 4 "The
Wächter-Biegler Example"
(https://vanderbei.princeton.edu/ps/loqo3_5.pdf, PDF pages 7-8 of the
fetched document). That section states (their eq. 14):

    minimize   x1
    subject to x1^2 - x2 + a = 0
               x1 - x3 - b   = 0
               x2, x3 >= 0

and reproduces Wächter-Biegler's own theorem: if b >= 0,
a - m*b <= min(0, -a/2), x1(0) < 0, and x2(0), x3(0) > 0, where
m = (x1(0)^2 - x2(0) + a) / |x1(0) - x3(0) - b|, then certain line-search
interior-point algorithms produce iterates that remain bounded away from
the optimum.

CORRECTION vs. the Task 4 brief: the brief's starting constants
(a=1, b=0.5, start (-2, 3, 1)) do NOT satisfy the theorem above -- plugging
them in gives m = (4 - 3 + 1)/|-2 - 1 - 0.5| = 2/3.5 ~= 0.571 and
a - m*b = 1 - 0.286 = 0.714, which is NOT <= min(0, -0.5) = -0.5. That
combination is therefore not guaranteed (by Wächter-Biegler's own result)
to reproduce the failure mode this problem exists to probe. This module
instead uses a = -1, b = 1 (the exact numeric instance Benson, Shanno, and
Vanderbei ran through LOQO in the same section, which they confirm jams
near x1 = -1.127 without their shifting remedy), combined with the
brief's proposed start point (-2, 3, 1), which DOES satisfy the theorem:
m = (4 - 3 + (-1))/|-2 - 1 - 1| = 0/4 = 0, and
a - m*b = -1 - 0 = -1 <= min(0, 0.5) = 0. All four hypotheses
(b=1>=0, -1<=0, x1(0)=-2<0, x2(0)=3>0, x3(0)=1>0) hold.

Encoding
--------
Static NLP via ``tychopy.solvers.OptimizationProblem`` (see
tests/corpus/README.md, "Encoding for the literature tier" -- tychopy DOES
expose a first-class static-NLP container, so no trivial-dynamics/phase
workaround is needed for this tier). Inequality constraints use PSIOPT's
``g(x) <= 0`` convention (verified in doc-legacy/tutorials/PhaseGuide.rst,
"Inequality Constraints" section), so "x2 >= 0" / "x3 >= 0" are encoded as
"-x2 <= 0" / "-x3 <= 0".

Note on the "solution": Benson et al. report the reduced form's solution
as x1 = 1 (i.e., the b-bound is active); PSIOPT is a different codebase
from LOQO/IPOPT so it is NOT expected to reproduce a specific failure
mechanism -- the point of this corpus entry is to record whatever PSIOPT
actually does with today's defaults, not to force a particular flag.

Observed on defaults 2026-07-16: NOTCONVERGED (harness status "failed"),
500 iterations (hits max_iters), objective -0.968231, byte-identical
across a --repeat 2 determinism check. PSIOPT plateaus well short of the
true optimum (x1* = 1) without diverging outright -- consistent in
spirit with the "jamming" phenomenon this problem is designed to probe,
though PSIOPT is a different codebase from LOQO/IPOPT so the precise
mechanism is not expected (or claimed) to match theirs.
"""

import tychopy as typy

vf = typy.vector_functions
solvs = typy.solvers
Args = vf.Arguments

TIER = "literature"
TIMEOUT = 30
SOLVE_MODE = "optimize"

# Wächter-Biegler constants verified against Benson-Shanno-Vanderbei's
# reproduction (see docstring above) -- NOT the brief's original (a=1, b=0.5),
# which fails the cited theorem's own sufficient condition.
_A = -1.0
_B = 1.0


def build():
    """Construct the Wächter-Biegler counterexample (unsolved).

    See tests/corpus/README.md for the full problem-module contract.
    """
    prob = solvs.OptimizationProblem()
    prob.set_vars([-2.0, 3.0, 1.0])

    # minimize x1
    prob.add_objective(Args(1)[0], [0])

    # x1^2 - x2 + a = 0  (depends on x1, x2 -> indices [0, 1])
    prob.add_equal_con(Args(2)[0] ** 2 - Args(2)[1] + _A, [0, 1])
    # x1 - x3 - b = 0  (depends on x1, x3 -> indices [0, 2])
    prob.add_equal_con(Args(2)[0] - Args(2)[1] - _B, [0, 2])

    # x2 >= 0, x3 >= 0  ->  -x2 <= 0, -x3 <= 0 (PSIOPT's g(x) <= 0 convention)
    prob.add_inequal_con(-Args(1)[0], [1])
    prob.add_inequal_con(-Args(1)[0], [2])

    return prob
