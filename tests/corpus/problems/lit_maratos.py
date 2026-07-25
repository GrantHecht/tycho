"""tests/corpus/problems/lit_maratos.py — E2 G0 literature tier.

The Maratos-effect example: a step that is a genuine (Q-quadratic-rate)
improvement toward the optimum nonetheless INCREASES both the objective
and the constraint violation, so any merit function that penalizes both
jointly rejects it -- the textbook illustration of why naive merit-
function line searches can stall arbitrarily close to a solution.

Citation and verification trail
--------------------------------
Primary source: J. Nocedal and S. J. Wright, "Numerical Optimization,"
2nd edition, Springer, 2006. Fetched directly (convexoptimization.com/
TOOLS/nocedal.pdf, 683-page scan) and read at the primary location:
Chapter 15 "Fundamentals of Algorithms for Nonlinear Constrained
Optimization," Section 15.5 "The Maratos Effect," Example 15.4 (book
page 441; PDF page 460 of the fetched scan), attributed there to
Powell [M. J. D. Powell, reference [255] in the book's bibliography].
Equation (15.34) reads verbatim:

    min f(x1, x2) = 2(x1^2 + x2^2 - 1) - x1,  subject to  x1^2 + x2^2 - 1 = 0.

The book states the optimal solution is x* = (1, 0)^T with Lagrange
multiplier lambda* = 3/2, and parametrizes a feasible iterate as
xk = (cos(theta), sin(theta))^T for arbitrary theta, showing the step
pk = (sin^2(theta), -sin(theta)cos(theta))^T both increases f and the
constraint violation for ANY nonzero theta.

The specific numeric starting point (0, 1) proposed by the Task 4 brief
(theta = pi/2) was cross-checked against a secondary reproduction of this
same example (a thesis at
http://www.applied-mathematics.net/mythesis/node51.html, section on the
l1 merit function), which independently states: objective
2(x^2+y^2-1)-x, constraint x^2+y^2-1=0, starting point x_1=(0,1)^T,
optimum x*=(1,0)^T, direction delta_1=(1,0) -- an exact match to
Nocedal-Wright's general pk formula evaluated at theta=pi/2
(sin^2(pi/2)=1, -sin(pi/2)cos(pi/2)=0). No correction to the brief was
needed for this problem; the brief's formulation and start point are
both verified as-is.

Section 15.6 ("Second-Order Correction and Nonmonotone Techniques,"
including the "Nonmonotone (Watchdog) Strategy" subsection, book page
444) cites "Chamberlain et al [57]" -- i.e. the same 1982 paper targeted
by lit_cycling.py -- as one of "the earliest references on nonmonotone
methods," but does not reproduce that paper's own motivating cycling
example; see lit_cycling.py (SKIPPED) for why that example itself could
not be independently verified.

Encoding
--------
Static NLP via ``tychopy.solvers.OptimizationProblem`` (see
tests/corpus/README.md). No inequality constraints or bounds are needed:
just the objective and the single equality constraint.

Observed on defaults 2026-07-16: DIVERGING (harness status "diverged"),
2 iterations, objective ~1e16, byte-identical across a --repeat 2
determinism check. Console trace: iteration 0 takes a short step
(alpha_p = alpha_d = 0.25) from the exactly-feasible start; iteration 1's
step blows the equality-constraint residual up to ~5e15 and PSIOPT
immediately reports "Solution Diverging." This is a FINDING TO REPORT: on
this classic textbook example, PSIOPT does not exhibit the "slow/stalled
progress near the solution" flavor of the Maratos effect that motivates
the example in Nocedal-Wright -- instead it diverges outright, in just 2
iterations, from a start that is already exactly ON the constraint
manifold. Not force-fixed or reformulated to produce a different
outcome, per the Task 4 instructions to record observed behavior rather
than engineer a particular one.
"""

import tychopy as typy

vf = typy.vector_functions
solvs = typy.solvers
Args = vf.Arguments

TIER = "literature"
TIMEOUT = 30
SOLVE_MODE = "optimize"


def build():
    """Construct the Maratos-effect example (unsolved).

    See tests/corpus/README.md for the full problem-module contract.
    """
    prob = solvs.OptimizationProblem()
    prob.set_vars([0.0, 1.0])

    # min 2*(x1^2 + x2^2 - 1) - x1
    prob.add_objective(2.0 * (Args(2).squared_norm() - 1.0) - Args(2)[0], [0, 1])

    # x1^2 + x2^2 - 1 = 0
    prob.add_equal_con(Args(2).squared_norm() - 1.0, [0, 1])

    return prob
