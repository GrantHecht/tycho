"""
Regression tests for tychopy.optimal_control.ode_base_class.ODEBase.

Two defects fixed here (CODEBASE 1.2):

1. ``_make_index_set`` and ``get_vars`` used
   ``isinstance(x, (int, np.int32, np.intc))`` to recognize integer indices.
   ``np.int64`` -- numpy's default integer dtype on 64-bit Linux, and the
   element dtype of ``np.arange(...)`` -- fails that exact-type check and
   falls through to the "Invalid index"/"Invalid name specifier" raise
   branches.  Fixed by widening to ``(int, np.integer)``.
2. ``get_vars``'s trajectory branch built the ``(n_points, n_vars)`` output
   with a double Python loop (``output[j][i] = source[j][idx]``).  Fixed by
   vectorizing with ``np.asarray(source)[:, idxs]``.

The minimal ODE below mirrors the ``SHO_Ode`` / ``ode_x.ode`` subclass
pattern used by test_SetUnitsDtypes.py, but is built as a
``tychopy.optimal_control.ODEBase`` subclass (rather than a raw
``oc.ode_x.ode`` subclass) since the defects live in the pure-Python
``ODEBase`` wrapper, not the compiled extension.
"""

import unittest

import numpy as np

import tychopy.optimal_control as oc
import tychopy.vector_functions as vf


class RVOde(oc.ODEBase):
    """Toy 6-state (position + velocity) ODE, no controls.

    Registers a 3-element "R" (position) variable group and a 3-element
    "V" (velocity) group, mirroring the Vgroups idiom used by
    examples/python_examples/Delta3Launch.py's RocketODE.
    """

    def __init__(self):
        Xvars = 6
        args = oc.ODEArguments(Xvars)
        R = args.x_vec().head(3)
        V = args.x_vec().segment(3, 3)
        # Dynamics content is irrelevant to these tests (get_vars/add_Vgroups
        # never integrate or optimize) -- just a valid size-6 VectorFunction.
        xdot = vf.stack([V, V])
        super().__init__(xdot, Xvars)
        self.add_Vgroups({"R": [0, 1, 2], "V": [3, 4, 5]})


class TestOdeBaseClass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ode = RVOde()

    def test_add_vgroups_accepts_np_arange(self):
        """np.arange(...) elements are np.int64 on 64-bit Linux; must not be
        rejected by an exact np.int32/np.intc dtype check."""
        ode = RVOde()
        ode.add_Vgroups({"Rarange": np.arange(3)})  # must not raise
        np.testing.assert_array_equal(ode.idx("Rarange"), [0, 1, 2])

    def test_add_vgroups_int_scalar_still_accepted(self):
        """Plain python int scalars (the pre-existing supported case) still
        work after widening the isinstance check."""
        ode = RVOde()
        ode.add_Vgroups({"single": 4})  # must not raise
        np.testing.assert_array_equal(ode.idx("single"), [4])

    def test_get_vars_accepts_np_int64_name(self):
        """A bare np.int64 index mixed into the ``names`` list must not be
        rejected by an exact np.int32/np.intc dtype check."""
        state_vec = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 0.0])
        out = self.ode.get_vars(["R", np.int64(3)], state_vec)
        np.testing.assert_array_equal(out, [10.0, 11.0, 12.0, 13.0])

    def test_get_vars_single_vector_retscalar_branch_unchanged(self):
        """The single-vector branch (not the trajectory branch) with
        retscalar=True returns a bare scalar when the result has one
        element, unchanged by the trajectory-branch vectorization fix."""
        state_vec = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 0.0])
        out = self.ode.get_vars(["V"], state_vec)
        np.testing.assert_array_equal(out, [13.0, 14.0, 15.0])

        mass_like = self.ode.get_vars([np.int64(6)], state_vec, retscalar=True)
        self.assertIsInstance(mass_like, (float, np.floating))
        self.assertEqual(mass_like, 0.0)

    def test_get_vars_trajectory_matches_loop_reference_list_of_arrays(self):
        """Vectorized get_vars trajectory path matches a hand-written loop
        reference when the source is a list of 1-D arrays."""
        r_idxs = list(self.ode.idx("R"))
        traj = [np.arange(7, dtype=float) + k for k in range(5)]
        out = self.ode.get_vars(["R"], traj)
        ref = np.array([[row[i] for i in r_idxs] for row in traj])
        np.testing.assert_array_equal(out, ref)
        self.assertEqual(out.shape, (5, len(r_idxs)))

    def test_get_vars_trajectory_matches_loop_reference_2d_ndarray(self):
        """Vectorized get_vars trajectory path matches a hand-written loop
        reference when the source is a single 2-D ndarray (as opposed to a
        list of 1-D arrays)."""
        r_idxs = list(self.ode.idx("R"))
        traj = np.array([np.arange(7, dtype=float) + k for k in range(5)])
        out = self.ode.get_vars(["R"], traj)
        ref = np.array([[row[i] for i in r_idxs] for row in traj])
        np.testing.assert_array_equal(out, ref)
        self.assertEqual(out.shape, (5, len(r_idxs)))


if __name__ == "__main__":
    unittest.main()
