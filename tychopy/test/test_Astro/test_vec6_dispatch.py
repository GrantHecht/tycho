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

    def test_element_dispatches_to_vf_overload(self):
        # Element is Segment<-1, 1, -1> — a 1-vector. Stack six of them to
        # form a 6-row arg-like expression that routes through _vec6_wrap.
        A6 = vf.Arguments(6)
        self.assertEqual(type(A6[0]).__name__, "Element")
        stacked = vf.Stack([A6[i] for i in range(6)])
        out = typy.Astro.modified_to_cartesian(stacked, 1.0)
        self.assertEqual(out.input_rows(), 6)
        self.assertEqual(out.output_rows(), 6)

    def test_numeric_paths_unaffected(self):
        arr = np.array([1.0, 0.1, 0.0, 0.0, 0.0, 0.0])
        out_arr = typy.Astro.modified_to_cartesian(arr, 1.0)
        self.assertEqual(out_arr.shape, (6,))
        self.assertTrue(np.all(np.isfinite(out_arr)))

        out_list = typy.Astro.modified_to_cartesian([1.0, 0.1, 0.0, 0.0, 0.0, 0.0], 1.0)
        self.assertEqual(out_list.shape, (6,))
        self.assertTrue(np.all(np.isfinite(out_list)))

        out_tup = typy.Astro.modified_to_cartesian((1.0, 0.1, 0.0, 0.0, 0.0, 0.0), 1.0)
        self.assertEqual(out_tup.shape, (6,))
        self.assertTrue(np.all(np.isfinite(out_tup)))

    def test_numeric_only_function_rejects_vf_with_typeerror(self):
        # `cartesian_to_classic_true` and the propagators have no VF overload —
        # passing a VF must surface as a clean TypeError from nanobind dispatch
        # ("incompatible function arguments"), not a confusing np.asarray
        # failure. Pinning on the nanobind substring guards against a future
        # regression that re-routes VFs to np.asarray and produces a different
        # TypeError from numpy.
        mee = vf.Arguments(7).head(6)
        nanobind_re = r"incompatible function arguments"
        with self.assertRaisesRegex(TypeError, nanobind_re):
            typy.Astro.cartesian_to_classic_true(mee, 1.0)
        with self.assertRaisesRegex(TypeError, nanobind_re):
            typy.Astro.propagate_cartesian(mee, 1.0, 1.0)


if __name__ == "__main__":
    unittest.main()
