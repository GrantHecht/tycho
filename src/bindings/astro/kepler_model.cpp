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
#include "optimal_control/tycho_optimal_control.h"

// TychoBind<KeplerPropagator> — defined here since KeplerPropagator.h's inline Build was removed.
namespace tycho {
using namespace tycho::vf;
using namespace tycho::oc;
using namespace tycho::astro;
using namespace tycho::integrators;
template <> struct TychoBind<KeplerPropagator> {
    static void build(nb::module_ &m, const char *name) {
        auto obj = nb::class_<KeplerPropagator>(m, name, R"doc(Universal-variable Kepler propagator VectorFunction (IR=7, OR=6).

Maps ``[rx, ry, rz, vx, vy, vz, dt]`` to the propagated Cartesian state
``[rx', ry', rz', vx', vy', vz']`` using the LCD (Stumpff-function)
universal-variable iteration with IFT-composed analytic Jacobian and
Hessian.

Intended for embedding inside optimal-control expression graphs (e.g. as a
shooting constraint inside a :class:`~astro.kepler.phase`).

Parameters
----------
mu : float
    Gravitational parameter (must be > 0).

Notes
-----
Input vector layout: indices 0–5 are the initial Cartesian state, index 6
is the time-of-flight ``dt``.  Units of position, velocity, and time must
all be consistent with ``mu``.
)doc");
        obj.def(nb::init<double>());
        bind::DenseBaseBuild<KeplerPropagator>(obj);
    }
};
} // namespace tycho

// TychoBind<KeplerPhase> — KeplerPhase is a subclass of ODEPhase<Kepler> with extra members.
namespace tycho {
using namespace tycho::vf;
using namespace tycho::oc;
using namespace tycho::astro;
using namespace tycho::integrators;
template <> struct TychoBind<KeplerPhase> {
    static void build(nb::module_ &m) {
        auto phase = nb::class_<KeplerPhase, ODEPhaseBase>(m, "phase", R"doc(Optimal-control phase for Keplerian two-body propagation.

Specialised :class:`~tychopy.optimal_control.PhaseInterface` for the Kepler
ODE.  When ``use_kepler_propagator`` is ``True`` (default), shooting
intervals use the analytic LCD Kepler propagator
(:class:`~astro.kepler.KeplerPropagator`) instead of numerical integration,
giving exact two-body trajectories.  The shooting defect is a midpoint
(central) shooting defect: each boundary state is propagated by the LCD
Kepler propagator to the interval midpoint and the two results are
differenced — the analytic analogue of ``CentralShootingDefect``.

Attributes
----------
use_kepler_propagator : bool
    If ``True`` (default), shooting segments use the LCD Kepler propagator.
    Set to ``False`` to fall back to numerical integration (e.g. when testing
    collocation-only transcriptions).
integrator : Integrator
    Numerical integrator used for non-Kepler shooting segments.

Notes
-----
Construct via :meth:`astro.kepler.ode.phase`, not directly.  All
constraint-building and solve methods are inherited from
:class:`~tychopy.optimal_control.PhaseInterface`.
)doc");
        bind::ODEPhaseBuildImpl<Kepler>(phase);
        phase.def_rw("integrator", &KeplerPhase::integrator_);
        phase.def_rw("use_kepler_propagator", &KeplerPhase::use_kepler_propagator_);
    }
};
} // namespace tycho

// TychoBind<Kepler> — the Kepler ODE type.
namespace tycho {
using namespace tycho::vf;
using namespace tycho::oc;
using namespace tycho::astro;
using namespace tycho::integrators;
template <> struct TychoBind<Kepler> {
    static void build(nb::module_ &m, const char *name) {
        auto obj = nb::class_<Kepler>(m, name, R"doc(Keplerian two-body ODE for use with an optimal-control Phase.

Defines the 6-state Cartesian dynamics ``dX/dt = [V, -mu*R/|R|^3]`` (no
perturbations, no control).  Typically used via :meth:`phase` to build a
:class:`~astro.kepler.phase` that uses the analytic Kepler propagator.

Parameters
----------
mu : float
    Gravitational parameter (km³/s² or consistent units).

Examples
--------
Build a Kepler phase with LGL3 collocation::

    from tychopy.astro import kepler
    ode = kepler.ode(mu)
    ph = ode.phase("LGL3", traj, num_segs)
    ph.use_kepler_propagator = True   # default
)doc").def(nb::init<double>());
        bind::DenseBaseBuild<Kepler>(obj);
        obj.def("phase", [](const Kepler &od, TranscriptionModes Tmode) {
            return std::make_shared<KeplerPhase>(od, Tmode);
        },
                R"doc(Build a Kepler phase with the given transcription mode and no initial trajectory.

Parameters
----------
mode : TranscriptionModes or str
    Transcription scheme (e.g. ``"LGL3"``, ``"LGL5"``, ``"Trapezoidal"``,
    ``"CentralShooting"``).

Returns
-------
phase
    A :class:`~astro.kepler.phase` ready for constraint configuration.
)doc");
        obj.def("phase", [](const Kepler &od, TranscriptionModes Tmode,
                            const std::vector<Eigen::VectorXd> &Traj, int numdef) {
            return std::make_shared<KeplerPhase>(od, Tmode, Traj, numdef);
        },
                R"doc(Build a Kepler phase with an initial-guess trajectory.

Parameters
----------
mode : TranscriptionModes or str
    Transcription scheme.
traj : list of ndarray
    Initial-guess trajectory nodes.  Each node is a 7-element vector
    ``[rx, ry, rz, vx, vy, vz, t]`` (Kepler ODE has no controls or
    parameters).
num_segs : int
    Number of defect (collocation/shooting) intervals to discretize
    the trajectory into.

Returns
-------
phase
    A :class:`~astro.kepler.phase` initialised with the given trajectory.
)doc");

        obj.def("phase", [](const Kepler &od, std::string Tmode) {
            return std::make_shared<KeplerPhase>(od, Tmode);
        },
                R"doc(Build a Kepler phase (string transcription mode, no initial trajectory).

Parameters
----------
mode : str
    Transcription scheme name (e.g. ``"LGL3"``).

Returns
-------
phase
    A :class:`~astro.kepler.phase` ready for constraint configuration.
)doc");
        obj.def("phase",
                [](const Kepler &od, std::string Tmode, const std::vector<Eigen::VectorXd> &Traj,
                   int numdef) { return std::make_shared<KeplerPhase>(od, Tmode, Traj, numdef); },
                R"doc(Build a Kepler phase (string transcription mode, with initial trajectory).

Parameters
----------
mode : str
    Transcription scheme name.
traj : list of ndarray
    Initial-guess trajectory nodes.  Each node is a 7-element vector
    ``[rx, ry, rz, vx, vy, vz, t]`` (Kepler ODE has no controls or
    parameters).
num_segs : int
    Number of defect (collocation/shooting) intervals to discretize
    the trajectory into.

Returns
-------
phase
    A :class:`~astro.kepler.phase` initialised with the given trajectory.
)doc");

        bind::IntegratorBuildConstructors<Kepler>(obj);
    }
};
void BuildKeplerMod(FunctionRegistry &reg, nb::module_ &m);
} // namespace tycho

void tycho::BuildKeplerMod(FunctionRegistry &reg, nb::module_ &m) {
    auto odemod = m.def_submodule("kepler");
    reg.template build_register<Kepler>(odemod, "ode");
    reg.build_register<KeplerPropagator>(odemod, "KeplerPropagator");
    TychoBind<KeplerPhase>::build(odemod);
}
