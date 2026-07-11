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

A further defect surfaced by (1): fixing ``MuNeptune`` to the planet-only
value exposed that ``SpiceBodyProps["NEPTUNE BARYCENTER"]`` mapped to
``MuNeptune`` — pre-fix that entry carried the (numerically correct)
barycenter value only because of the ``MuNeptune``/``MuNeptuneBarycenter``
mislabeling; post-fix it silently carried the wrong (planet-only) value for
a barycenter entry. ``SpiceBodyProps["URANUS BARYCENTER"]`` had the same
bug, pre-existing and independent of (1). Both are fixed here to map to
their respective ``*Barycenter`` constants.
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


def test_spice_body_props_barycenter_entries_use_barycenter_constants():
    """Every "* BARYCENTER" key must map to the distinct MuXBarycenter
    constant, not the planet-only Mu constant, whenever a distinct
    Barycenter constant exists for that body (Jupiter, Saturn, Uranus,
    Neptune). Mars has no distinct MuMarsBarycenter constant, so
    "MARS BARYCENTER" falling back to the planet-only MuMars is out of
    scope for this convention check.
    """
    for key, props in c.SpiceBodyProps.items():
        if not key.endswith(" BARYCENTER"):
            continue
        planet = key[: -len(" BARYCENTER")].title()
        barycenter_name = f"Mu{planet}Barycenter"
        planet_only_name = f"Mu{planet}"
        barycenter_mu = getattr(c, barycenter_name, None)
        if barycenter_mu is None:
            continue
        assert props["Mu"] == barycenter_mu, (
            f"SpiceBodyProps[{key!r}] should use {barycenter_name}"
        )
        planet_only_mu = getattr(c, planet_only_name, None)
        if planet_only_mu is not None:
            assert props["Mu"] != planet_only_mu, (
                f"SpiceBodyProps[{key!r}] should not use the planet-only "
                f"{planet_only_name}"
            )


def test_neptune_and_uranus_barycenter_entries_fixed():
    assert c.SpiceBodyProps["NEPTUNE BARYCENTER"]["Mu"] == c.MuNeptuneBarycenter
    assert c.SpiceBodyProps["URANUS BARYCENTER"]["Mu"] == c.MuUranusBarycenter
