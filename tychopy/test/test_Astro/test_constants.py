"""Regression tests for tychopy.astro.constants and CSIThruster (CODEBASE 1.2).

Two live defects fixed here:

1. ``constants.MuNeptune`` was bit-for-bit equal to ``MuNeptuneBarycenter``
   (the Neptune *system* GM), breaking this module's own convention —
   verified for Jupiter/Saturn/Uranus — that the un-suffixed constant is the
   planet-only GM. Fixed to the planet-only value (system GM minus Triton's
   GM, ~1427.6 km^3/s^2); ``MuNeptuneBarycenter`` is unchanged. Flagged for
   human review as a physical constant.
2. ``ThrusterModels.CSIThruster`` computed mass flow rate using
   ``9.8065`` instead of the standard gravity constant ``9.80665``
   (exact by definition).
"""

import pytest

import tychopy.astro.constants as c
from tychopy.astro.Extensions import ThrusterModels


def test_mu_neptune_is_planet_only():
    assert c.MuNeptune != c.MuNeptuneBarycenter
    assert c.MuNeptune == pytest.approx(6.8350995e15, rel=1e-6)
    assert c.MuNeptuneBarycenter == pytest.approx(6.83652710058002e15, rel=1e-12)


def test_csi_thruster_uses_standard_gravity():
    t = ThrusterModels.CSIThruster(F=1.0, Isp=1000.0, M=100.0)
    assert t.Mdot == pytest.approx(1.0 / (1000.0 * 9.80665), rel=1e-12)
