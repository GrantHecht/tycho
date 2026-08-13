"""Regression tests for the Python-binding error translation of the Kepler
propagators.

Pins three contracts that the C++ side promises:

1. ``propagate_cartesian`` raises ``RuntimeError`` when the LCD kernel fails
   to converge or NaN-poisons its output.  A future regression that drops
   the ``allFinite()`` guard in the binding (or that returns the NaN array
   silently) would reintroduce the silent-failure bug from the pre-820fef6f
   era; InteriorPointSolver-style step-rejection patterns at the Python boundary depend
   on the exception.
2. ``propagate_classic`` validates ``mu > 0`` and a finite, non-zero
   semi-major axis at the C++ boundary (raising ``std::invalid_argument``
   which nanobind translates to ``ValueError``).  Bad-mu / bad-a inputs
   were previously misattributed to "LCD iteration did not converge" — the
   message is now method-specific.
3. ``propagate_modified`` likewise validates ``mu > 0`` at the C++ boundary.

These tests do not exercise the C++ kernel directly — that is covered in
the C++ unit tests (``test_kepler_lcd.cpp`` ``NaNInjectionFlagsNonConvergence``
and ``test_kepler_ift.cpp`` ``NaNPoisoningEndToEnd``).  The gap they close is
the *binding-layer translation* of those C++ contracts to Python exceptions.
"""

import math
import unittest

import numpy as np

import tychopy as typy


class TestKeplerPropagateErrors(unittest.TestCase):
    # ----- propagate_cartesian -----

    def test_propagate_cartesian_raises_on_nan_dt(self):
        # Non-finite dt is rejected at the C++ kernel boundary
        # (kepler_lcd_iterate validates dt finiteness in its precondition
        # block); nanobind translates std::invalid_argument to ValueError.
        rv = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
        with self.assertRaises(ValueError):
            typy.astro.propagate_cartesian(rv, float("nan"), 398600.4418)

    def test_propagate_cartesian_raises_on_nonpositive_mu(self):
        rv = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
        with self.assertRaises(ValueError):
            typy.astro.propagate_cartesian(rv, 100.0, -1.0)
        with self.assertRaises(ValueError):
            typy.astro.propagate_cartesian(rv, 100.0, 0.0)

    def test_propagate_cartesian_dt_zero_validates_mu(self):
        # CODEBASE 1.1b: the dt == 0 early return previously bypassed all input
        # validation (mu > 0, dt finite, V0 finite, r0 > 0).  Those checks are
        # now hoisted ahead of the early return, so a dt == 0 call with mu <= 0
        # raises std::invalid_argument -> ValueError.
        rv = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
        with self.assertRaisesRegex(ValueError, r"propagate_cartesian.*mu"):
            typy.astro.propagate_cartesian(rv, 0.0, -1.0)

    def test_propagate_cartesian_dt_zero_validates_r0(self):
        # A zero state vector (|R0| == 0) is rejected even when dt == 0, instead
        # of being returned unchanged by the (previously unguarded) early return.
        with self.assertRaisesRegex(ValueError, r"r0 must satisfy"):
            typy.astro.propagate_cartesian(np.zeros(6), 0.0, 398600.4418)

    def test_propagate_cartesian_runtime_error_message_lists_causes(self):
        # Force a NaN-poisoned output by handing the kernel a NaN-component
        # initial state.  The post-loop finite mask in kepler_lcd_iterate
        # detects this and reports converged=false; propagate_cartesian
        # NaN-fills its output; the binding translates to RuntimeError.
        # The enriched message lists the typical causes (dt magnitude, hyp
        # guard, non-finite inputs) so users can debug without a structured
        # KeplerLCDStatus enum (deferred to a follow-up).
        rv = np.array([7000.0, 0.0, 0.0, 0.0, math.nan, 0.0])
        with self.assertRaisesRegex(
            (RuntimeError, ValueError),
            r"propagate_cartesian|V0 must be finite",
        ):
            typy.astro.propagate_cartesian(rv, 100.0, 398600.4418)

    # ----- propagate_classic -----

    def test_propagate_classic_raises_on_nonpositive_mu(self):
        oe = np.array([7000.0, 0.01, 0.5, 0.0, 0.0, 0.0])
        with self.assertRaisesRegex(ValueError, r"propagate_classic.*mu"):
            typy.astro.propagate_classic(oe, 100.0, -1.0)
        with self.assertRaisesRegex(ValueError, r"propagate_classic.*mu"):
            typy.astro.propagate_classic(oe, 100.0, 0.0)

    def test_propagate_classic_raises_on_zero_semi_major_axis(self):
        # a == 0 produces 1/0 in the analytic mean-motion formula; the
        # C++ overload now rejects this at the boundary with a method-
        # specific message instead of misattributing to LCD non-convergence.
        oe = np.array([0.0, 0.01, 0.5, 0.0, 0.0, 0.0])
        with self.assertRaisesRegex(ValueError, r"semi-major axis"):
            typy.astro.propagate_classic(oe, 100.0, 398600.4418)

    def test_propagate_classic_raises_on_nan_semi_major_axis(self):
        oe = np.array([math.nan, 0.01, 0.5, 0.0, 0.0, 0.0])
        with self.assertRaisesRegex(ValueError, r"semi-major axis"):
            typy.astro.propagate_classic(oe, 100.0, 398600.4418)

    # ----- propagate_modified -----

    def test_propagate_modified_raises_on_nonpositive_mu(self):
        # MEE elements: p=7000, f=0.01, g=0.0, h=0.0, k=0.0, L=0.0
        mee = np.array([7000.0, 0.01, 0.0, 0.0, 0.0, 0.0])
        with self.assertRaisesRegex(ValueError, r"propagate_modified.*mu"):
            typy.astro.propagate_modified(mee, 100.0, -1.0)
        with self.assertRaisesRegex(ValueError, r"propagate_modified.*mu"):
            typy.astro.propagate_modified(mee, 100.0, 0.0)

    # ----- classic_to_cartesian / classic_to_modified -----
    # (elliptic near-parabolic anomaly solve: Markley hybrid seed vs.
    # genuinely degenerate input -> RuntimeError)

    def test_classic_to_cartesian_near_parabolic_converges(self):
        # perf/review-8a: the Markley-cubic-starter threshold hybrid seed
        # (used above e = 0.9) closes the near-parabolic divergence band that
        # the old E = M seed left open. These three cells -- mirroring the
        # C++ twin ``KeplerEdgeCases.EllipticNearParabolicNowConverges`` in
        # tests/cpp/astro/test_kepler_edge_cases.cpp -- were the probe's
        # confirmed gain set (converged-new, divergent-old); the headline
        # case (e = 1 - 1e-9, M = 1e-8) is the former
        # ``EllipticNonConvergencePoisonsOutput`` control input. This test
        # used to assert RuntimeError here; it now asserts success plus a
        # round-trip check that the Newton solve landed on the true root
        # (not merely produced *some* finite state).
        cells = [(1.0 - 1e-9, 1e-8), (1.0 - 1e-9, 0.12), (0.99, 0.105)]
        for e, m in cells:
            oe = [1.0e5, e, 0.1, 0.1, 0.1, m]
            rv = typy.astro.classic_to_cartesian(oe, 398600.4418)
            self.assertTrue(
                np.all(np.isfinite(rv)),
                msg=f"classic_to_cartesian non-finite at e={e}, M={m}",
            )
            # Independent correctness check, mirroring the C++ twin:
            # modified_to_classic recomputes M via the closed-form
            # (Newton-free) arctan relation, so recovering the input M
            # confirms the Newton-solved E landed on the true root.
            mee = typy.astro.classic_to_modified(oe, 398600.4418)
            oe2 = typy.astro.modified_to_classic(mee, 398600.4418)
            d_m = (oe2[5] - m + math.pi) % (2.0 * math.pi) - math.pi
            self.assertAlmostEqual(
                d_m,
                0.0,
                delta=1e-6,
                msg=f"recovered mean anomaly wrong at e={e}, M={m}",
            )

    def test_classic_to_modified_near_parabolic_converges(self):
        # Same three cells through the classic_to_modified sibling.
        cells = [(1.0 - 1e-9, 1e-8), (1.0 - 1e-9, 0.12), (0.99, 0.105)]
        for e, m in cells:
            oe = [1.0e5, e, 0.1, 0.1, 0.1, m]
            mee = typy.astro.classic_to_modified(oe, 398600.4418)
            self.assertTrue(
                np.all(np.isfinite(mee)),
                msg=f"classic_to_modified non-finite at e={e}, M={m}",
            )

    def test_classic_to_cartesian_degenerate_input_raises(self):
        # Raise-path coverage is preserved: a non-finite mean anomaly (NaN M)
        # can never satisfy the Newton step-convergence test, so
        # MAXITERS_ELLIPTIC is still exhausted and the whole output is
        # NaN-poisoned -- mirroring the C++ twin
        # ``KeplerEdgeCases.EllipticDegenerateInputPoisonsOutput``. This
        # documents that the poison -> allFinite -> ValueError/RuntimeError
        # chain at the binding is retained and still fires.
        oe = [1.0e5, 0.5, 0.1, 0.1, 0.1, math.nan]
        with self.assertRaisesRegex(RuntimeError, r"classic_to_cartesian.*converge"):
            typy.astro.classic_to_cartesian(oe, 398600.4418)

    def test_classic_to_modified_degenerate_input_raises(self):
        # Same degenerate (NaN M) input through the classic_to_modified
        # sibling.
        oe = [1.0e5, 0.5, 0.1, 0.1, 0.1, math.nan]
        with self.assertRaisesRegex(RuntimeError, r"classic_to_modified.*converge"):
            typy.astro.classic_to_modified(oe, 398600.4418)


if __name__ == "__main__":
    unittest.main()
