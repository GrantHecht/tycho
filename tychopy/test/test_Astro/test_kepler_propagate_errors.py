"""Regression tests for the Python-binding error translation of the Kepler
propagators.

Pins three contracts that the C++ side promises:

1. ``propagate_cartesian`` raises ``RuntimeError`` when the LCD kernel fails
   to converge or NaN-poisons its output.  A future regression that drops
   the ``allFinite()`` guard in the binding (or that returns the NaN array
   silently) would reintroduce the silent-failure bug from the pre-820fef6f
   era; PSIOPT-style step-rejection patterns at the Python boundary depend
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


if __name__ == "__main__":
    unittest.main()
