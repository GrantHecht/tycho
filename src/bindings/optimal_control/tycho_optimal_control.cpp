// =============================================================================
// Originally from ASSET (AlabamaASRL/asset_asrl)
// Copyright 2020-present The University of Alabama-Astrodynamics and Space
//   Research Lab. Licensed under the Apache License, Version 2.0
// License: notices/asset-apache2.txt.
// Source: https://github.com/AlabamaASRL/asset_asrl
// Original Developer: James B. Pezent
//
// Modifications in Tycho fork (Copyright 2026-present Grant R. Hecht,
//   Apache 2.0 — see LICENSE.txt).
// =============================================================================

#include "tycho_optimal_control.h"
#include "mesh_iterate_info_bind.h"
#include "ode_arguments_bind.h"

#include "tycho/detail/integrators/integrator.h"

namespace tycho {

// Forward declarations for builders defined in separate .cpp files.
void GenericODESBuildPart1(FunctionRegistry &reg, nb::module_ &m);
void GenericODESBuildPart2(FunctionRegistry &reg, nb::module_ &m);
void GenericODESBuildPart3(FunctionRegistry &reg, nb::module_ &m);
void GenericODESBuildPart4(FunctionRegistry &reg, nb::module_ &m);
void GenericODESBuildPart6(FunctionRegistry &reg, nb::module_ &m);

void GenericODESIntegratorPart1(FunctionRegistry &reg, nb::module_ &m);
void GenericODESIntegratorPart2(FunctionRegistry &reg, nb::module_ &m);
void GenericODESIntegratorPart3(FunctionRegistry &reg, nb::module_ &m);
void GenericODESIntegratorPart4(FunctionRegistry &reg, nb::module_ &m);
void GenericODESIntegratorPart6(FunctionRegistry &reg, nb::module_ &m);

void RKFlagsBuild(nb::module_ &m);

static void OCPFlagsBuild(nb::module_ &m) {
    nb::enum_<TranscriptionModes>(m, "TranscriptionModes",
                                  R"doc(Transcription scheme used to discretize a phase.

Passed (as an enum or its string name, e.g. ``"LGL3"``) when constructing
a phase to select how the continuous dynamics are converted into a
finite-dimensional set of constraints -- either collocation defects (the LGL
and ``Trapezoidal`` schemes) or shooting defects (``CentralShooting``).
)doc")
        .value("LGL3", TranscriptionModes::LGL3, "3rd-order Legendre-Gauss-Lobatto collocation.")
        .value("LGL5", TranscriptionModes::LGL5, "5th-order Legendre-Gauss-Lobatto collocation.")
        .value("LGL7", TranscriptionModes::LGL7, "7th-order Legendre-Gauss-Lobatto collocation.")
        .value("Trapezoidal", TranscriptionModes::Trapezoidal,
               "Trapezoidal (2nd-order) collocation.")
        .value("CentralShooting", TranscriptionModes::CentralShooting,
               "Central (forward/backward) shooting transcription.");

    nb::enum_<IntegralModes>(
        m, "IntegralModes",
        R"doc(Quadrature rule used to evaluate integral terms over a phase.)doc")
        .value("BaseIntegral", IntegralModes::BaseIntegral,
               "Default quadrature matched to the transcription order.")
        .value("TrapIntegral", IntegralModes::TrapIntegral, "Trapezoidal-rule quadrature.");

    nb::enum_<ControlModes>(m, "ControlModes",
                            R"doc(Interpolation/representation scheme for the control history.)doc")
        .value("HighestOrderSpline", ControlModes::HighestOrderSpline,
               "Spline matching the transcription's highest order.")
        .value("FirstOrderSpline", ControlModes::FirstOrderSpline,
               "Piecewise-linear (first-order) control spline.")
        .value("NoSpline", ControlModes::NoSpline,
               "Independent control values, no inter-node spline.")
        .value("BlockConstant", ControlModes::BlockConstant,
               "Control held constant over each transcription block.");

    nb::enum_<PhaseRegionFlags>(
        m, "PhaseRegionFlags",
        R"doc(Region of a phase that a state function (constraint or objective) acts on.

Selects which discretized state(s) -- endpoints, interior nodes, or
parameter blocks -- are passed as arguments to a state function when it is
attached to a phase. Most phase methods accept either the enum value or its
string name (e.g. ``"Front"``, ``"Path"``).
)doc")
        .value("Front", PhaseRegionFlags::Front, "The first (initial-time) node of the phase.")
        .value("Back", PhaseRegionFlags::Back, "The last (final-time) node of the phase.")
        .value("Path", PhaseRegionFlags::Path, "Every node of the phase (full path constraint).")
        .value("NodalPath", PhaseRegionFlags::NodalPath, "Every nodal (mesh) point along the path.")
        .value("FrontandBack", PhaseRegionFlags::FrontandBack,
               "Both endpoints, front state followed by back state.")
        .value("BackandFront", PhaseRegionFlags::BackandFront,
               "Both endpoints, back state followed by front state.")
        .value("Params", PhaseRegionFlags::Params, "The combined parameter vector of the phase.")
        .value("InnerPath", PhaseRegionFlags::InnerPath,
               "Every interior node, excluding the two endpoints.")
        .value("ODEParams", PhaseRegionFlags::ODEParams, "The ODE (dynamics) parameter sub-vector.")
        .value("StaticParams", PhaseRegionFlags::StaticParams,
               "The static (non-ODE) parameter sub-vector.")
        .value("PairWisePath", PhaseRegionFlags::PairWisePath,
               "Each consecutive pair of nodes along the path.");

    nb::enum_<ScaleModes>(m, "ScaleModes",
                          R"doc(Source of the variable/equation scaling applied to a phase.)doc")
        .value("AUTO", ScaleModes::AUTO, "Scales are computed automatically from the problem.")
        .value("CUSTOM", ScaleModes::CUSTOM, "Scales are supplied explicitly by the user.")
        .value("NONE", ScaleModes::NONE, "No scaling is applied (unit scales).");

    nb::enum_<MeshErrorEstimators>(
        m, "MeshErrorEstimators",
        R"doc(Method used to estimate the per-interval mesh (discretization) error.)doc")
        .value("DEBOOR", MeshErrorEstimators::DEBOOR,
               "de Boor collocation-residual error estimate.")
        .value("INTEGRATOR", MeshErrorEstimators::INTEGRATOR,
               "Error estimate from a reference numerical integrator.");

    nb::enum_<MeshErrorAggregation>(
        m, "MeshErrorAggregation",
        R"doc(Reduction used to aggregate per-interval mesh errors into a scalar.)doc")
        .value("MAX", MeshErrorAggregation::MAX, "Maximum error over all intervals.")
        .value("AVG", MeshErrorAggregation::AVG, "Arithmetic mean of per-interval errors.")
        .value("GEOMETRIC", MeshErrorAggregation::GEOMETRIC,
               "Geometric mean of per-interval errors.")
        .value("ENDTOEND", MeshErrorAggregation::ENDTOEND,
               "End-to-end re-integration trajectory error.");

    m.def("strto_phase_region_flag", nb::overload_cast<const std::string &>(&strto_PhaseRegionFlag),
          nb::arg("name"),
          R"doc(Convert a string name to the corresponding ``PhaseRegionFlags`` value.

Raises ``ValueError`` if ``name`` does not match any known flag.

Parameters
----------
name : str
    Case-sensitive string name of the phase-region flag, e.g. ``"Front"``,
    ``"Back"``, ``"Path"``, ``"FrontandBack"``, ``"BackandFront"``,
    ``"Params"``, ``"NodalPath"``, ``"InnerPath"``, ``"ODEParams"``,
    ``"StaticParams"``, ``"PairWisePath"``.

Returns
-------
PhaseRegionFlags
    The enum value corresponding to ``name``.
)doc");
}

void optimal_control_build(FunctionRegistry &reg, nb::module_ &m) {
    auto &oc = reg.getOptimalControlModule();
    RKFlagsBuild(oc);
    OCPFlagsBuild(oc);

    TychoBind<StateFunction<GenericFunction<-1, -1>>>::build(oc, "StateConstraint");
    TychoBind<StateFunction<GenericFunction<-1, 1>>>::build(oc, "StateObjective");

    TychoBind<MeshIterateInfo>::build(oc);

    // EventPack: 3-arg ctor for explicit C++ construction, tuple-accepting
    // init so Python callers can pass (vf, direction, stop_count) tuples
    // directly.
    nb::class_<tycho::integrators::EventPack>(
        oc, "EventPack",
        R"doc(Specification for a single event (zero-crossing) to be tracked by an integrator.

An event function is a scalar-output :class:`VectorFunction` that is evaluated
at each step. When its sign changes between two successive steps the integrator
records the bracketed interval as a crossing. The ``direction`` filter
restricts which sign changes are recorded, and ``stop_count`` can halt
integration after a given number of events.

A list of ``EventPack`` objects is passed to the integrator via its
``integrate_events`` family of methods. Each element of the list corresponds
to an independent event function, and the returned event-time arrays are
indexed accordingly.

``EventPack`` objects are implicitly constructible from a 3-tuple
``(vf, direction, stop_count)`` so callers may pass plain tuples wherever an
``EventPack`` is expected.
)doc")
        .def(nb::init<tycho::vf::GenericFunction<-1, 1>, int, int>(), nb::arg("vf"),
             nb::arg("direction") = 0, nb::arg("stop_count") = 0,
             R"doc(Construct an event specification.

Parameters
----------
vf : VectorFunction
    Scalar-output (output rows == 1) function over the ODE input
    ``[x, t, u, p]``. A zero crossing of this function defines the event.
direction : int, optional
    Crossing direction filter.  ``-1`` records only falling crossings
    (positive-to-negative); ``0`` (default) records any crossing;
    ``+1`` records only rising crossings (negative-to-positive).
stop_count : int, optional
    Number of crossings after which integration halts.  ``0`` (default)
    means "record crossings but never stop"; ``N > 0`` stops after the
    *N*-th recorded crossing.

Raises
------
ValueError
    If ``direction`` is not in ``{-1, 0, +1}`` or ``stop_count < 0``.
)doc")
        .def(
            "__init__",
            [](tycho::integrators::EventPack *self, const nb::tuple &t) {
                if (t.size() != 3)
                    throw nb::type_error(
                        "EventPack(tuple) expects exactly 3 elements: (vf, direction, stop_count)");
                new (self)
                    tycho::integrators::EventPack(nb::cast<tycho::vf::GenericFunction<-1, 1>>(t[0]),
                                                  nb::cast<int>(t[1]), nb::cast<int>(t[2]));
            },
            nb::arg("t"),
            R"doc(Construct from a 3-tuple ``(vf, direction, stop_count)``.

Convenience form used by the implicit conversion from ``tuple``.  The
elements are interpreted the same way as the explicit 3-argument constructor.

Raises
------
TypeError
    If the tuple does not have exactly 3 elements.
ValueError
    If ``direction`` is not in ``{-1, 0, +1}`` or ``stop_count < 0``.
)doc")
        .def_rw("vf", &tycho::integrators::EventPack::vf,
                R"doc(The scalar-output event function.

A :class:`VectorFunction` with exactly one output row.  The integrator
evaluates this function at every accepted step point; a sign change between
consecutive evaluations triggers an event recording.
)doc")
        .def_prop_rw(
            "direction", &tycho::integrators::EventPack::direction,
            &tycho::integrators::EventPack::set_direction,
            R"doc(Crossing direction filter: ``-1`` (falling), ``0`` (any), or ``+1`` (rising).

Only crossings matching the specified direction are recorded.  Assigning a
value outside ``{-1, 0, +1}`` raises ``ValueError``.
)doc")
        .def_prop_rw("stop_count", &tycho::integrators::EventPack::stop_count,
                     &tycho::integrators::EventPack::set_stop_count,
                     R"doc(Number of crossings after which integration stops.

``0`` (default) means all crossings are recorded but integration is never
halted by this event.  ``N > 0`` halts integration after the *N*-th
crossing is recorded.  Assigning a negative value raises ``ValueError``.
)doc");
    nb::implicitly_convertible<nb::tuple, tycho::integrators::EventPack>();

    TychoBind<ODEPhaseBase>::build(oc);
    TychoBind<OptimalControlProblemBase>::build(oc);
    TychoBind<LGLInterpTable>::build(oc);

    reg.build_register<InterpFunction<-1>>(oc, "InterpFunction");
    reg.build_register<InterpFunction<1>>(oc, "InterpFunction_1");
    reg.build_register<InterpFunction<3>>(oc, "InterpFunction_3");
    reg.build_register<InterpFunction<6>>(oc, "InterpFunction_6");

    TychoBind<FDDerivArbitrary<Eigen::VectorXd>>::build(oc, "FiniteDiffTable");
    TychoBind<ODEArguments<-1, -1, -1>>::build(oc, "ODEArguments");

    GenericODESBuildPart1(reg, oc);
    GenericODESBuildPart2(reg, oc);
    GenericODESBuildPart3(reg, oc);
    GenericODESBuildPart4(reg, oc);
    GenericODESBuildPart6(reg, oc);

    // Integrator bindings — must be registered after the core ODE parts above
    // so that the submodules already exist.
    GenericODESIntegratorPart1(reg, oc);
    GenericODESIntegratorPart2(reg, oc);
    GenericODESIntegratorPart3(reg, oc);
    GenericODESIntegratorPart4(reg, oc);
    GenericODESIntegratorPart6(reg, oc);
}

} // namespace tycho
