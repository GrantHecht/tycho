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

#pragma once
#ifdef TYCHO_PYTHON_BINDINGS

// TychoBind specializations for ODEPhase<DODE>, StateFunction<T>,
// FDDerivArbitrary<T>, ODEPhaseBase, LGLInterpTable, and OptimalControlProblemBase.

#include "tycho/detail/optimal_control/core/state_function.h"
#include "tycho/detail/optimal_control/phase/ode_phase.h"
#include "tycho/detail/optimal_control/phase/ode_phase_base.h"
#include "tycho/detail/optimal_control/phase/optimal_control_problem.h"
#include "tycho/detail/optimal_control/transcription/lgl_interp_functions.h"
#include "tycho/detail/vf/derivatives/fd_deriv_arbitrary.h"

namespace tycho {

using namespace tycho::vf;
using namespace tycho::oc;

namespace bind {

template <class DODE, class PyClass> void ODEPhaseBuildImpl(PyClass &phase) {
    phase.def(nb::init<DODE, TranscriptionModes>());
    phase.def(nb::init<DODE, TranscriptionModes, const std::vector<Eigen::VectorXd> &, int>());
    phase.def(
        nb::init<DODE, TranscriptionModes, const std::vector<Eigen::VectorXd> &, int, bool>());

    phase.def(nb::init<DODE, std::string>());
    phase.def(nb::init<DODE, std::string, const std::vector<Eigen::VectorXd> &>());

    phase.def(nb::init<DODE, std::string, const std::vector<Eigen::VectorXd> &, int>());
    phase.def(nb::init<DODE, std::string, const std::vector<Eigen::VectorXd> &, int, bool>());
}

} // namespace bind

template <class DODE> struct TychoBind<ODEPhase<DODE>> {
    static void build(nb::module_ &m) {
        auto phase = nb::class_<ODEPhase<DODE>, ODEPhaseBase>(m, "phase");

        bind::ODEPhaseBuildImpl<DODE>(phase);
        phase.def_rw("integrator", &ODEPhase<DODE>::integrator_,
                     R"doc(Integrator: the reference Runge-Kutta integrator for this phase's ODE.

Used by re-integration and mesh-error estimation. Adjust its step-size
or tolerances before calling :meth:`optimize` to control propagation
accuracy.
)doc");
        phase.def_rw("enable_hessian_sparsity", &ODEPhase<DODE>::enable_hessian_sparsity_,
                     R"doc(bool: whether to exploit Hessian sparsity in the defect constraints.

When ``True``, the phase exploits the block-sparse structure of the
defect Hessian during NLP assembly. Defaults to ``False``.
)doc");
        phase.def_rw("old_shooting_defect", &ODEPhase<DODE>::old_shooting_defect_,
                     R"doc(bool: use the legacy shooting-defect formulation.

When ``True``, the ``CentralShooting`` transcription uses the original
defect implementation from the ASSET codebase. Defaults to ``False``
(i.e. the updated formulation).
)doc");

        phase.def("get_input_scale", &ODEPhase<DODE>::get_input_scale,
                  R"doc(Return the per-input scale vector for a function binding.

Computes representative input magnitudes over the chosen phase region
and variable indices, used internally to set automatic output scales.

Parameters
----------
flag : PhaseRegionFlags or str
    Phase region the function acts on.
xtu_vars : numpy.ndarray of int
    State/time/control variable indices.
op_vars : numpy.ndarray of int
    ODE-parameter variable indices.
sp_vars : numpy.ndarray of int
    Static-parameter variable indices.

Returns
-------
numpy.ndarray
    Per-input scale factors derived from the current trajectory.
)doc");
        phase.def("get_defect", &ODEPhase<DODE>::get_defect,
                  R"doc(Return the ODE defect VectorFunction for the active transcription mode.

Returns
-------
VectorFunction
    A type-erased defect function matching the phase's current
    transcription scheme (LGL3, LGL5, LGL7, Trapezoidal, or
    CentralShooting).
)doc");
    }
};

template <class FuncType> struct TychoBind<StateFunction<FuncType>> {
    static void build(nb::module_ &m, const char *name) {
        auto obj = nb::class_<StateFunction<FuncType>>(m, name);
        // Set the class docstring based on the concrete Python name.
        if (std::string{name} == "StateConstraint") {
            obj.doc() =
                R"doc(A vector-valued constraint bound to a phase region.

Bundles a VectorFunction with the phase region it applies to and the
variable indices it reads. This is the internal record a phase stores
for each equality or inequality constraint added via
:meth:`~PhaseInterface.add_equal_con` / :meth:`~PhaseInterface.add_inequal_con`.
Users typically do not construct ``StateConstraint`` objects directly --
they are built and registered by the ``add_*`` methods on a phase.

Parameters
----------
func : VectorFunction
    The constraint function (vector-valued, output dimension = number of
    constraints).
region : PhaseRegionFlags or str
    Phase region the constraint acts on (e.g. ``"Front"``, ``"Back"``,
    ``"Path"``).
vars : numpy.ndarray of int
    Variable indices within each node vector that the function reads.
)doc";
        } else {
            obj.doc() =
                R"doc(A scalar objective bound to a phase region.

Bundles a scalar VectorFunction (output rows = 1) with the phase region
it applies to and the variable indices it reads. This is the internal
record a phase stores for each objective added via
:meth:`~PhaseInterface.add_state_objective`. Users typically do not construct
``StateObjective`` objects directly -- they are built and registered by
the ``add_*`` methods on a phase.

Parameters
----------
func : VectorFunction
    The objective function (scalar-valued, output rows = 1).
region : PhaseRegionFlags or str
    Phase region the objective acts on (e.g. ``"Front"``, ``"Back"``,
    ``"Path"``).
vars : numpy.ndarray of int
    Variable indices within each node vector that the function reads.
)doc";
        }
        obj.def(nb::init<FuncType, PhaseRegionFlags, Eigen::VectorXi, Eigen::VectorXi,
                         Eigen::VectorXi>(),
                R"doc(Bind a function to a phase region with explicit variable index groups.

Parameters
----------
func : VectorFunction
    The vector function to attach.
region : PhaseRegionFlags or str
    Phase region the function acts on (e.g. ``"Front"``, ``"Back"``,
    ``"Path"``).
xtu_vars : numpy.ndarray of int
    State/time/control variable indices within each node vector that the
    function reads.
op_vars : numpy.ndarray of int
    ODE-parameter variable indices the function reads.
sp_vars : numpy.ndarray of int
    Static-parameter variable indices the function reads.
)doc");
        obj.def(nb::init<FuncType, PhaseRegionFlags, Eigen::VectorXi, PhaseRegionFlags,
                         Eigen::VectorXi>(),
                R"doc(Bind a function spanning both a state region and a parameter region.

Parameters
----------
func : VectorFunction
    The vector function to attach.
region : PhaseRegionFlags or str
    Phase region for the state/time/control variables.
xtu_vars : numpy.ndarray of int
    State/time/control variable indices the function reads.
param_region : PhaseRegionFlags or str
    Parameter region — must be ``"ODEParams"`` or ``"StaticParams"``.
param_vars : numpy.ndarray of int
    Parameter variable indices the function reads.
)doc");
        obj.def(nb::init<FuncType, PhaseRegionFlags, Eigen::VectorXi>(),
                R"doc(Bind a function to a phase region with a single variable index group.

Parameters
----------
func : VectorFunction
    The vector function to attach.
region : PhaseRegionFlags or str
    Phase region the function acts on. When set to ``"ODEParams"`` or
    ``"StaticParams"``, ``vars`` is interpreted as the corresponding
    parameter indices; otherwise it is the state/time/control indices.
vars : numpy.ndarray of int
    Variable indices, interpreted according to ``region``.
)doc");
    }
};

template <class DType> struct TychoBind<FDDerivArbitrary<DType>> {
    static void build(nb::module_ &m, const char *name) {
        auto obj = nb::class_<FDDerivArbitrary<DType>>(
            m, name,
            R"doc(Finite-difference differentiator for non-uniformly-sampled vector data.

Computes per-sample derivatives of a sequence of vector samples with
respect to a chosen independent axis. The samples do not need to be
equally spaced: stencil weights are solved from the actual node offsets
at each evaluation point via a Vandermonde system, using forward,
central, or backward stencils near the boundaries.

Parameters
----------
axis : int
    Index of the independent axis (e.g. time) within each sample vector.
data : list of numpy.ndarray
    The sampled data series.
)doc");
        obj.def(nb::init<int, const std::vector<DType> &>(),
                R"doc(Construct a differentiator with the given axis index and data series.

Parameters
----------
axis : int
    Index of the independent axis within each sample vector.
data : list of numpy.ndarray
    The sampled data series to differentiate.
)doc");
        obj.def("all_derivs", &FDDerivArbitrary<DType>::template allderiv_python<DType>,
                R"doc(Compute finite-difference derivatives at every sample index.

Parameters
----------
order : int
    Order of the derivative to compute (e.g. ``1`` for first derivative).
accuracy : int
    Finite-difference accuracy order (rounded up to the nearest even
    number internally).

Returns
-------
list of numpy.ndarray
    One derivative vector per sample, in the same order as the input data.
)doc");
        obj.def("deriv", &FDDerivArbitrary<DType>::template ithderiv_python<DType>,
                R"doc(Compute the finite-difference derivative at a single sample index.

Parameters
----------
i : int
    Zero-based sample index at which to evaluate the derivative.
order : int
    Order of the derivative to compute.
accuracy : int
    Finite-difference accuracy order (rounded up to the nearest even
    number internally).

Returns
-------
numpy.ndarray
    The derivative vector at sample index ``i``.
)doc");
    }
};

// Forward declarations for out-of-line TychoBind implementations
template <> struct TychoBind<ODEPhaseBase> {
    static void build(nb::module_ &m);
};

template <> struct TychoBind<LGLInterpTable> {
    static void build(nb::module_ &m);
};

template <> struct TychoBind<OptimalControlProblemBase> {
    static void build(nb::module_ &m);
};

} // namespace tycho

#endif // TYCHO_PYTHON_BINDINGS
