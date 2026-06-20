// =============================================================================
// Originally from ASSET (AlabamaASRL/asset_asrl)
// Copyright 2020-present The University of Alabama-Astrodynamics and Space
//   Research Lab. Licensed under the Apache License, Version 2.0
// License: notices/asset-apache2.txt.
// Source: https://github.com/AlabamaASRL/asset_asrl
// Original Developer: James B. Pezent
//
// Modifications in Tycho fork (Copyright 2026-present Grant R. Hecht,
//   Apache 2.0 — see LICENSE.txt):
//   - Binding code extracted from ASSET source and reorganized (PR 2 — binding decoupling)
//   - Migrated pybind11 -> nanobind (PR 3)
//   - Migrated to tycho:: sub-namespaces (PR #35)
// =============================================================================

#include "astro/tycho_astro.h"

#include "tycho/detail/vf/operators/root_finder.h"

namespace tycho {
using namespace tycho::vf;
using namespace tycho::oc;
using namespace tycho::astro;
using namespace tycho::integrators;

template <> struct TychoBind<ModifiedToCartesian> {
    static void build(nb::module_ &m, const char *name) {
        auto obj = nb::class_<ModifiedToCartesian>(m, name, R"doc(MEE-to-Cartesian conversion VectorFunction (IR=6, OR=6).

Accepts Modified Equinoctial Elements ``[p, f, g, h, k, L]`` and returns
Cartesian state ``[rx, ry, rz, vx, vy, vz]`` via direct closed-form
conversion.  Useful for embedding element conversions inside an optimal
control expression graph.

Parameters
----------
mu : float
    Gravitational parameter (must be > 0).

See Also
--------
modified_to_cartesian : Scalar overload for numeric arrays.
)doc").def(nb::init<double>());
        bind::DenseBaseBuild<ModifiedToCartesian>(obj);
    }
};

template <> struct TychoBind<CartesianToClassic> {
    static void build(nb::module_ &m, const char *name) {
        auto obj = nb::class_<CartesianToClassic>(m, name, R"doc(Cartesian-to-classical orbital elements differentiable VectorFunction (IR=6, OR=6).

Accepts Cartesian state ``[rx, ry, rz, vx, vy, vz]`` and computes classical
orbital elements ``[a, e, i, Omega, omega, nu_or_M]`` as a differentiable
VectorFunction expression for embedding inside optimal control expression
graphs.

Parameters
----------
mu : float
    Gravitational parameter (must be > 0).

Notes
-----
Intended for use as a VectorFunction component in expression graphs.
For direct numeric conversion of array inputs, use
:func:`cartesian_to_classic` or :func:`cartesian_to_classic_true`.
)doc").def(nb::init<double>());
        bind::DenseBaseBuild<CartesianToClassic>(obj);
    }
};

template <> struct TychoBind<CartesianToMEE> {
    static void build(nb::module_ &m, const char *name) {
        auto obj = nb::class_<CartesianToMEE>(m, name, R"doc(Cartesian-to-MEE conversion VectorFunction (IR=6, OR=6).

Accepts Cartesian state ``[rx, ry, rz, vx, vy, vz]`` and computes Modified
Equinoctial Elements ``[p, f, g, h, k, L]`` (L = true longitude).
Implements the direct closed-form conversion as a differentiable
VectorFunction for use inside optimal control expression graphs.

Parameters
----------
mu : float
    Gravitational parameter (must be > 0).

See Also
--------
cartesian_to_modified : Scalar overload for numeric arrays.
)doc").def(nb::init<double>());
        bind::DenseBaseBuild<CartesianToMEE>(obj);
    }
};

void kepler_utils_build(FunctionRegistry &reg, nb::module_ &m);
} // namespace tycho

void tycho::kepler_utils_build(FunctionRegistry &reg, nb::module_ &m) {

    ////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////              Conversions                  /////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////

    m.def("cartesian_to_classic",
          [](const Vector6<double> &XV, double mu) { return cartesian_to_classic(XV, mu); },
          R"doc(Convert a Cartesian state to classical orbital elements (mean anomaly).

Parameters
----------
XV : array_like, shape (6,)
    Cartesian state ``[rx, ry, rz, vx, vy, vz]``.  Units must be
    consistent with ``mu`` (e.g. km and km/s).
mu : float
    Gravitational parameter (km³/s² or consistent units).  Must be > 0.

Returns
-------
ndarray, shape (6,)
    Classical elements ``[a, e, i, Omega, omega, M]``, where

    * ``a`` — semi-major axis (< 0 for hyperbolic orbits).
    * ``e`` — eccentricity.
    * ``i``, ``Omega``, ``omega`` — inclination, RAAN, argument of perigee (radians).
    * ``M`` — mean anomaly (radians): elliptic (M = E − e sin E) for e < 1,
      hyperbolic (M = e sinh H − H) for e > 1.

See Also
--------
cartesian_to_classic_true : Same conversion but returns true anomaly.
classic_to_cartesian : Inverse conversion.
)doc");
    m.def("cartesian_to_modified",
          [](const Vector6<double> &XV, double mu) { return cartesian_to_modified(XV, mu); },
          R"doc(Convert a Cartesian state to Modified Equinoctial Elements (MEE).

Direct closed-form conversion with no Kepler-equation iteration.
Singular at exactly i = 180° (retrograde equatorial); all other
inclinations including non-equatorial retrograde are supported.

Parameters
----------
XV : array_like, shape (6,)
    Cartesian state ``[rx, ry, rz, vx, vy, vz]``.
mu : float
    Gravitational parameter (must be > 0).

Returns
-------
ndarray, shape (6,)
    MEE ``[p, f, g, h, k, L]`` where ``L`` is the true longitude (radians).

See Also
--------
modified_to_cartesian : Inverse conversion.
)doc");
    m.def("classic_to_cartesian", [](const Vector6<double> &oelems, double mu) {
        return classic_to_cartesian(oelems, mu);
    },
          R"doc(Convert classical orbital elements to a Cartesian state.

Solves Kepler's equation via Newton iteration (elliptic) or the hyperbolic
analogue, then rotates to the inertial frame via the Euler-angle sequence
(Omega, i, omega).

Parameters
----------
oelems : array_like, shape (6,)
    Classical elements ``[a, e, i, Omega, omega, M]``.  For hyperbolic
    orbits ``a < 0`` by convention and ``M`` is the hyperbolic mean anomaly
    (M = e sinh H − H).  All angles in radians.
mu : float
    Gravitational parameter (must be > 0).

Returns
-------
ndarray, shape (6,)
    Cartesian state ``[rx, ry, rz, vx, vy, vz]``.

See Also
--------
cartesian_to_classic : Inverse conversion.
)doc");
    m.def("classic_to_modified",
          [](const Vector6<double> &oelems, double mu) { return classic_to_modified(oelems, mu); },
          R"doc(Convert classical orbital elements to Modified Equinoctial Elements (MEE).

Solves Kepler's equation for the true anomaly from the mean anomaly, then
computes MEE algebraically.

Parameters
----------
oelems : array_like, shape (6,)
    Classical elements ``[a, e, i, Omega, omega, M]``.  All angles in
    radians.  ``mu`` is accepted for API symmetry but is not used in the
    algebraic conversion.
mu : float
    Gravitational parameter (not used; accepted for API symmetry).

Returns
-------
ndarray, shape (6,)
    MEE ``[p, f, g, h, k, L]`` where ``L`` is the true longitude (radians).

See Also
--------
modified_to_classic : Inverse conversion.
)doc");

    m.def("modified_to_cartesian", [](const Vector6<double> &meelems, double mu) {
        return modified_to_cartesian(meelems, mu);
    },
          R"doc(Convert Modified Equinoctial Elements (MEE) to a Cartesian state.

Direct closed-form conversion using the standard MEE frame vectors.

Parameters
----------
meelems : array_like, shape (6,)
    MEE ``[p, f, g, h, k, L]``: semi-latus rectum ``p``, equinoctial
    eccentricity components ``f``, ``g``, inclination components ``h``,
    ``k``, and true longitude ``L`` (radians).
mu : float
    Gravitational parameter (must be > 0).

Returns
-------
ndarray, shape (6,)
    Cartesian state ``[rx, ry, rz, vx, vy, vz]``.

See Also
--------
cartesian_to_modified : Inverse conversion.
)doc");
    m.def("modified_to_classic", [](const Vector6<double> &meelems, double mu) {
        return modified_to_classic(meelems, mu);
    },
          R"doc(Convert Modified Equinoctial Elements (MEE) to classical orbital elements.

Uses algebraic relations from MEE to classical elements, then solves the
true-anomaly-to-mean-anomaly step analytically.

Parameters
----------
meelems : array_like, shape (6,)
    MEE ``[p, f, g, h, k, L]`` where ``L`` is the true longitude (radians).
mu : float
    Gravitational parameter (not used; accepted for API symmetry).

Returns
-------
ndarray, shape (6,)
    Classical elements ``[a, e, i, Omega, omega, M]``, where ``M`` is the
    elliptic mean anomaly (e < 1) or hyperbolic mean anomaly (e > 1).
    All angles in radians.

See Also
--------
classic_to_modified : Inverse conversion.
)doc");
    m.def("cartesian_to_classic_true",
          [](const Vector6<double> &XV, double mu) { return cartesian_to_classic_true(XV, mu); },
          R"doc(Convert a Cartesian state to classical orbital elements (true anomaly).

Like :func:`cartesian_to_classic` but returns true anomaly ``nu`` as the
sixth element instead of mean anomaly ``M``.

Parameters
----------
XV : array_like, shape (6,)
    Cartesian state ``[rx, ry, rz, vx, vy, vz]``.
mu : float
    Gravitational parameter (must be > 0).

Returns
-------
ndarray, shape (6,)
    Classical elements ``[a, e, i, Omega, omega, nu]`` where ``nu`` is the
    true anomaly (radians).

See Also
--------
cartesian_to_classic : Identical conversion but returns mean anomaly.
)doc");

    ////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////       Conversions as VectorFunctions      /////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////

    m.def("modified_to_cartesian", [](const GenericFunction<-1, -1> &meelems, double mu) {
        return GenericFunction<-1, -1>(ModifiedToCartesian(mu).eval(meelems));
    },
          R"doc(Compose a MEE-to-Cartesian conversion with a VectorFunction.

Returns a new VectorFunction that evaluates ``meelems`` and then converts
its MEE output ``[p, f, g, h, k, L]`` to Cartesian ``[rx, ry, rz, vx, vy, vz]``.

Parameters
----------
meelems : VectorFunction
    A VectorFunction whose output is MEE ``[p, f, g, h, k, L]``.
mu : float
    Gravitational parameter (must be > 0).

Returns
-------
VectorFunction
    Composed VectorFunction with output shape (6,) in Cartesian coordinates.

Notes
-----
For numeric-array input use the ``(array, mu)`` overload.
)doc");

    reg.build_register<ModifiedToCartesian>(m, "ModifiedToCartesian");

    m.def("cartesian_to_classic", [](const GenericFunction<-1, -1> &RV, double mu) {
        return GenericFunction<-1, -1>(CartesianToClassic(mu).eval(RV));
    },
          R"doc(Compose a Cartesian-to-classical conversion with a VectorFunction.

Returns a new VectorFunction that evaluates ``RV`` and then converts its
Cartesian output ``[rx, ry, rz, vx, vy, vz]`` to classical orbital elements
``[a, e, i, Omega, omega, M]`` (M = mean anomaly).

Parameters
----------
RV : VectorFunction
    A VectorFunction whose output is a Cartesian state vector (shape 6).
mu : float
    Gravitational parameter (must be > 0).

Returns
-------
VectorFunction
    Composed VectorFunction with output shape (6,) in classical elements.

Notes
-----
For numeric-array input use the ``(array, mu)`` overload.
)doc");

    reg.build_register<CartesianToClassic>(m, "CartesianToClassic");

    m.def("cartesian_to_modified", [](const GenericFunction<-1, -1> &RV, double mu) {
        return GenericFunction<-1, -1>(CartesianToMEE(mu).eval(RV));
    },
          R"doc(Compose a Cartesian-to-MEE conversion with a VectorFunction.

Returns a new VectorFunction that evaluates ``RV`` and then converts its
Cartesian output ``[rx, ry, rz, vx, vy, vz]`` to MEE ``[p, f, g, h, k, L]``.

Parameters
----------
RV : VectorFunction
    A VectorFunction whose output is a Cartesian state vector (shape 6).
mu : float
    Gravitational parameter (must be > 0).

Returns
-------
VectorFunction
    Composed VectorFunction with output shape (6,) in MEE.

Notes
-----
For numeric-array input use the ``(array, mu)`` overload.
)doc");

    reg.build_register<CartesianToMEE>(m, "CartesianToMEE");

    ////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////              Propagators                  /////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////

    // Python contract: propagate_* raises RuntimeError on non-convergence
    // or NaN-poisoned output.  The C++ kernels return a NaN-filled Vector6
    // (PSIOPT step-rejection semantics), but Python users expect exceptions;
    // translate at the boundary.  Detection via allFinite() is cheap (6
    // doubles) and avoids exposing the internal KeplerLCDResult::converged
    // flag.  A structured KeplerLCDStatus enum that distinguishes the
    // bailout causes (max_iters / hyp_guard / NaN) is the more principled
    // diagnostic surface and is intentionally deferred to a follow-up; the
    // enriched messages below list the typical causes so users can debug
    // without it.
    m.def("propagate_cartesian", [](const Vector6<double> &RV, double dt, double mu) {
        const auto rv = propagate_cartesian(RV, dt, mu);
        if (!rv.allFinite())
            throw std::runtime_error(
                "propagate_cartesian: LCD iteration did not converge "
                "(check dt magnitude, hyperbolic asymptote at sqrt(-alpha)*X = 30, "
                "or non-finite inputs)");
        return rv;
    },
          R"doc(Propagate a Cartesian state forward under two-body gravity.

Uses the universal-variable LCD (Stumpff-function) iteration followed by
the Goodyear f-g map.  Supports elliptic and hyperbolic trajectories.

Parameters
----------
RV : array_like, shape (6,)
    Initial Cartesian state ``[rx, ry, rz, vx, vy, vz]``.
dt : float
    Time-of-flight.  Units must be consistent with ``mu``.
mu : float
    Gravitational parameter (must be > 0).

Returns
-------
ndarray, shape (6,)
    Propagated Cartesian state ``[rx', ry', rz', vx', vy', vz']``.

Raises
------
ValueError
    If ``mu <= 0``, ``dt`` is non-finite, ``RV`` contains non-finite
    entries, or ``|R0| == 0``.
RuntimeError
    If the LCD iteration fails to converge (near-parabolic orbits, very
    large time steps) or the hyperbolic-asymptote guard fires
    (``sqrt(-alpha)*X ≈ 30``).
)doc");
    // propagate_classic / propagate_modified do not run an LCD iteration —
    // input validation in the C++ overloads (see kepler_propagation.h) raises
    // std::invalid_argument for mu <= 0 or non-finite/zero semi-major axis,
    // which nanobind translates to ValueError.  This RuntimeError fires only
    // if the analytic path itself produces non-finite output despite valid
    // inputs (e.g. a numerically extreme intermediate via modified_to_classic).
    m.def("propagate_classic", [](const Vector6<double> &oelems, double dt, double mu) {
        const auto out = propagate_classic(oelems, dt, mu);
        if (!out.allFinite())
            throw std::runtime_error("propagate_classic: produced non-finite output");
        return out;
    },
          R"doc(Propagate classical orbital elements forward by a time step.

Advances the mean anomaly analytically: ``M_new = M + n*dt`` where
``n = sqrt(mu / |a|^3)`` is the mean motion.  Only the mean anomaly
element (index 5) is updated; all other elements are unchanged.

Parameters
----------
oelems : array_like, shape (6,)
    Classical elements ``[a, e, i, Omega, omega, M]``.  All angles in
    radians.
dt : float
    Time-of-flight.  Units consistent with ``mu``.
mu : float
    Gravitational parameter (must be > 0).

Returns
-------
ndarray, shape (6,)
    Propagated classical elements with updated mean anomaly.

Raises
------
ValueError
    If ``mu <= 0`` or the semi-major axis is zero or non-finite.
RuntimeError
    If the propagation produces non-finite output.
)doc");

    m.def("propagate_modified", [](const Vector6<double> &meelems, double dt, double mu) {
        const auto out = propagate_modified(meelems, dt, mu);
        if (!out.allFinite())
            throw std::runtime_error("propagate_modified: produced non-finite output");
        return out;
    },
          R"doc(Propagate Modified Equinoctial Elements (MEE) forward by a time step.

Round-trips through classical elements for the analytic mean-anomaly update:
MEE → classic → propagate_classic → classic → MEE.

Parameters
----------
meelems : array_like, shape (6,)
    MEE ``[p, f, g, h, k, L]`` where ``L`` is the true longitude (radians).
dt : float
    Time-of-flight.  Units consistent with ``mu``.
mu : float
    Gravitational parameter (must be > 0).

Returns
-------
ndarray, shape (6,)
    Propagated MEE state.

Raises
------
ValueError
    If ``mu <= 0``, or if the semi-major axis derived from the MEE
    elements is zero or non-finite (raised by the internal
    ``propagate_classic`` call).
RuntimeError
    If the propagation produces non-finite output.
)doc");
}
