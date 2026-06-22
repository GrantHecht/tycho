"""Regression test: the CartesianToClassic VectorFunction must agree with the
scalar ``cartesian_to_classic`` across the whole orbit, including post-periapsis
(descending, R.V < 0) states.

The VF expression previously had its true-anomaly quadrant branches swapped
relative to the scalar implementation, producing the wrong anomaly (and hence the
wrong mean anomaly M and argument of periapsis quadrant) for descending states.
There was no VF-vs-scalar test, which is why the bug survived. This test closes
that gap.
"""

import numpy as np

from tychopy import astro

MU = 398600.4418  # Earth, km^3/s^2


def _sample_states():
    """Cartesian states sampled around one elliptical orbit (covers R.V > 0 and < 0)."""
    states = []
    for mean_anom_deg in (20.0, 90.0, 170.0, 200.0, 280.0, 340.0):
        oe = [
            8000.0,              # a  [km]
            0.1,                 # e
            np.radians(30.0),    # i
            np.radians(40.0),    # RAAN
            np.radians(50.0),    # arg. periapsis
            np.radians(mean_anom_deg),  # mean anomaly
        ]
        states.append(np.asarray(astro.classic_to_cartesian(oe, MU), dtype=float))
    return states


def test_cartesian_to_classic_vf_matches_scalar():
    vf = astro.CartesianToClassic(MU)
    saw_descending = False

    for state in _sample_states():
        rdotv = float(np.dot(state[:3], state[3:]))
        if rdotv < 0.0:
            saw_descending = True

        scalar = np.asarray(astro.cartesian_to_classic(state, MU), dtype=float)
        vf_out = np.asarray(vf.compute(state), dtype=float)

        # a, e compare directly.
        np.testing.assert_allclose(scalar[:2], vf_out[:2], rtol=1e-9, atol=1e-9)
        # i, RAAN, arg-periapsis, M are angles: compare wrap-safe via cos/sin.
        np.testing.assert_allclose(np.cos(scalar[2:]), np.cos(vf_out[2:]), atol=1e-9)
        np.testing.assert_allclose(np.sin(scalar[2:]), np.sin(vf_out[2:]), atol=1e-9)

    assert saw_descending, "test must exercise at least one post-periapsis (R.V < 0) state"
