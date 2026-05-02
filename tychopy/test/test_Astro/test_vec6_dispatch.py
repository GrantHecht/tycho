"""Regression tests for tychopy.Astro._vec6_wrap dispatch (issue #45).

The wrapper must route arg-like VF types (Segment, Element, Arguments) — which
deliberately do not expose ``eval`` / ``__call__`` in the bindings — to the
VectorFunction overload, not the numpy ``np.asarray`` fallback.
"""

import unittest

import numpy as np

import tychopy as typy
from tychopy import VectorFunctions as vf


class TestVec6Dispatch(unittest.TestCase):
    def test_segment_dispatches_to_vf_overload(self):
        X = vf.Arguments(7)
        mee = X.head(6)
        out = typy.Astro.modified_to_cartesian(mee, 1.0)
        self.assertEqual(out.input_rows(), 7)
        self.assertEqual(out.output_rows(), 6)

    def test_arguments_dispatches_to_vf_overload(self):
        A6 = vf.Arguments(6)
        for name in (
            "modified_to_cartesian",
            "cartesian_to_classic",
            "cartesian_to_modified",
        ):
            out = getattr(typy.Astro, name)(A6, 1.0)
            self.assertEqual(out.input_rows(), 6, name)
            self.assertEqual(out.output_rows(), 6, name)

    def test_numeric_paths_unaffected(self):
        arr = np.array([1.0, 0.1, 0.0, 0.0, 0.0, 0.0])
        out_arr = typy.Astro.modified_to_cartesian(arr, 1.0)
        self.assertEqual(out_arr.shape, (6,))

        out_list = typy.Astro.modified_to_cartesian([1.0, 0.1, 0.0, 0.0, 0.0, 0.0], 1.0)
        self.assertEqual(out_list.shape, (6,))

        out_tup = typy.Astro.modified_to_cartesian((1.0, 0.1, 0.0, 0.0, 0.0, 0.0), 1.0)
        self.assertEqual(out_tup.shape, (6,))

    def test_numeric_only_function_rejects_vf_with_typeerror(self):
        # `cartesian_to_classic_true` and the propagators have no VF overload —
        # passing a VF must surface as a clean TypeError from nanobind dispatch,
        # not a confusing np.asarray failure.
        mee = vf.Arguments(7).head(6)
        with self.assertRaises(TypeError):
            typy.Astro.cartesian_to_classic_true(mee, 1.0)
        with self.assertRaises(TypeError):
            typy.Astro.propagate_cartesian(mee, 1.0, 1.0)


if __name__ == "__main__":
    unittest.main()
