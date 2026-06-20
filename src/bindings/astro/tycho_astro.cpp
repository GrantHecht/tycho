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

namespace tycho {
using namespace tycho::vf;
using namespace tycho::oc;
using namespace tycho::astro;
using namespace tycho::integrators;
void astro_build(FunctionRegistry &reg, nb::module_ &m);
void BuildKeplerMod(FunctionRegistry &reg, nb::module_ &m);
void BuildKeplerIntegrator(FunctionRegistry &reg, nb::module_ &m);
void kepler_utils_build(FunctionRegistry &reg, nb::module_ &m);
void lambert_solvers_build(FunctionRegistry &reg, nb::module_ &m);
} // namespace tycho

void tycho::astro_build(FunctionRegistry &reg, nb::module_ &m) {
    auto mod = m.def_submodule("astro");

    BuildKeplerMod(reg, mod);
    BuildKeplerIntegrator(reg, mod);
    kepler_utils_build(reg, mod);
    lambert_solvers_build(reg, mod);

    /////////////////////////////////////////////////////////////
    //////////// Binding Misc CPP Functions here for now ////////
    /////////////////////////////////////////////////////////////

    mod.def("modified_dynamics",
            [](double mu) { return GenericFunction<-1, -1>(MEEDynamics(mu)); },
            R"doc(Build the MEE two-body + RSW-thrust dynamics VectorFunction (IR=9, OR=6).

Returns a VectorFunction suitable for use as the ODE of an optimal-control
:class:`~tychopy.optimal_control.PhaseInterface`.

Input layout ``[state(6), control(3)]``:

* State (indices 0–5): ``[p, f, g, h, k, L]`` — Modified Equinoctial
  Elements where ``L`` is the true longitude (radians).
* Control (indices 6–8): ``[ur, ut, un]`` — RSW-frame accelerations
  (radial, tangential/along-track, out-of-plane/normal).

Output (6,): state derivative ``[dp/dt, df/dt, dg/dt, dh/dt, dk/dt, dL/dt]``.

Parameters
----------
mu : float
    Gravitational parameter (must be > 0).

Returns
-------
VectorFunction (IR=9, OR=6)
    MEE equations of motion with analytic Jacobian and Hessian.

See Also
--------
cartesian_dynamics : Equivalent model in Cartesian coordinates.
)doc");

    mod.def("cartesian_dynamics",
            [](double mu) { return GenericFunction<-1, -1>(CartesianDynamics(mu)); },
            R"doc(Build the Cartesian two-body + thrust dynamics VectorFunction (IR=9, OR=6).

Returns a VectorFunction suitable for use as the ODE of an optimal-control
:class:`~tychopy.optimal_control.PhaseInterface`.

Input layout ``[state(6), control(3)]``:

* State (indices 0–5): ``[rx, ry, rz, vx, vy, vz]`` — inertial Cartesian
  position and velocity.
* Control (indices 6–8): ``[ax_ctrl, ay_ctrl, az_ctrl]`` — inertial control
  acceleration.

Output (6,): state derivative
``[vx, vy, vz, ax_grav + ax_ctrl, ay_grav + ay_ctrl, az_grav + az_ctrl]``.

Parameters
----------
mu : float
    Gravitational parameter (must be > 0).

Returns
-------
VectorFunction (IR=9, OR=6)
    Cartesian equations of motion with analytic Jacobian and Hessian.

See Also
--------
modified_dynamics : Equivalent model in Modified Equinoctial Elements.
)doc");

    mod.def("crtbp_dynamics", [](double mu) { return GenericFunction<-1, -1>(CRTBPDynamics(mu)); },
            R"doc(Build the CR3BP dynamics VectorFunction in the synodic frame (IR=9, OR=6).

Returns a VectorFunction suitable for use as the ODE of an optimal-control
:class:`~tychopy.optimal_control.PhaseInterface`.

Input layout ``[state(6), control(3)]``:

* State (indices 0–5): ``[x, y, z, vx, vy, vz]`` — position and velocity
  in the rotating (synodic) frame with the primaries on the x-axis.
* Control (indices 6–8): ``[ax_ctrl, ay_ctrl, az_ctrl]`` — control
  acceleration in the synodic frame.

Output (6,): state derivative
``[vx, vy, vz, x_ddot, y_ddot, z_ddot]`` in the synodic frame
(includes Coriolis and centrifugal terms).

Parameters
----------
mu : float
    CR3BP mass ratio ``mu = m2 / (m1 + m2)``, must satisfy ``0 < mu < 1``.
    This is **not** a gravitational parameter — it is the dimensionless
    mass fraction of the smaller primary.

Returns
-------
VectorFunction (IR=9, OR=6)
    CR3BP equations of motion with analytic Jacobian and Hessian.
)doc");

    mod.def("j2_cartesian", [](double mu, double J2, double Rb) {
        return GenericFunction<-1, -1>(J2Cartesian_Impl::Definition(mu, J2, Rb));
    },
            R"doc(Build the J2 perturbation acceleration VectorFunction in Cartesian coordinates.

Returns a VectorFunction that computes the J2 gravitational perturbation
acceleration given position and north-pole direction.

Input layout (IR=6): ``[rx, ry, rz, north_px, north_py, north_pz]``.
The north-pole direction need not be pre-normalized.

Output (OR=3): J2 acceleration vector ``[ax_J2, ay_J2, az_J2]``.

Parameters
----------
mu : float
    Gravitational parameter (must be > 0).
J2 : float
    J2 zonal harmonic coefficient (dimensionless, typically ~1.08e-3 for
    Earth).
Rb : float
    Reference body radius (same units as position).

Returns
-------
VectorFunction (IR=6, OR=3)
    J2 perturbation acceleration with analytic derivatives.
)doc");

    mod.def("non_ideal_solar_sail", [](double mu, double beta, double n1, double n2, double t1) {
        return GenericFunction<-1, -1>(NonIdealSolarSail_Impl::Definition(mu, beta, n1, n2, t1));
    },
            R"doc(Build the non-ideal solar-sail thrust acceleration VectorFunction.

Returns a VectorFunction that models a non-ideal solar sail, accounting for
absorption efficiency and tangential force, in addition to the ideal
radiation-pressure term.

Input layout (IR=6): ``[rx, ry, rz, nx, ny, nz]`` — inertial position and
sail-normal direction.  The sail normal need not be pre-normalized.

Output (OR=3): sail acceleration vector ``[ax, ay, az]``.

Parameters
----------
mu : float
    Gravitational parameter of the central body (used to scale the
    radiation-pressure force via the lightness number; must be > 0).
beta : float
    Lightness number (ratio of solar radiation pressure to gravity at
    the reference distance).
n1 : float
    Normal force efficiency coefficient.
n2 : float
    Absorption efficiency coefficient.
t1 : float
    Tangential force efficiency coefficient.

Returns
-------
VectorFunction (IR=6, OR=3)
    Non-ideal solar-sail acceleration with analytic derivatives.
)doc");
}
