#pragma once

/// @file optimal_control.h
/// @brief Umbrella header for the OptimalControl subsystem.
///
/// Includes the full public OptimalControl API: ODE definitions, phases and
/// multi-phase problems, the collocation/transcription machinery, mesh
/// refinement, the interpolation tables, and the high-level builder wrappers.
/// Include this single header to define and solve optimal control problems.

/// @defgroup optimal_control OptimalControl
/// @brief Phases, ODEs, collocation/transcription, and mesh refinement.
///
/// The OptimalControl subsystem (namespace `tycho::oc`) converts continuous
/// optimal control problems into finite-dimensional NLPs via direct
/// collocation. It provides:
/// - **ODEs** (@ref tycho::ODE) wrapping user dynamics as VectorFunctions,
/// - **Phases** (@ref tycho::Phase) carrying boundary values, path
///   constraints, integral objectives, and a discretization mesh,
/// - **Multi-phase problems** (@ref tycho::OptimalControlProblem) that link
///   phases together with link functions,
/// - **Transcription schemes** (LGL collocation, trapezoidal, shooting) and
///   **mesh-error estimation/refinement**, and
/// - **builder wrappers** that expose the above as the Python-facing API.
///
/// Everything in the problem-definition layer is expressed in terms of the
/// VectorFunction subsystem (`tycho::vf`).

// Tycho — Phase/ODE transcription, collocation, mesh refinement

#include "tycho/vector_functions.h"

#include "tycho/detail/optimal_control/builder/ocp_wrapper.h"
#include "tycho/detail/optimal_control/builder/ode_builder.h"
#include "tycho/detail/optimal_control/builder/phase_wrapper.h"
#include "tycho/detail/optimal_control/builder/runtime_ode.h"
#include "tycho/detail/optimal_control/builder/var_registry.h"
#include "tycho/detail/optimal_control/core/interface_types.h"
#include "tycho/detail/optimal_control/core/link_function.h"
#include "tycho/detail/optimal_control/core/ode_arguments.h"
#include "tycho/detail/optimal_control/core/ode_sizes.h"
#include "tycho/detail/optimal_control/core/optimal_control_flags.h"
#include "tycho/detail/optimal_control/core/state_function.h"
#include "tycho/detail/optimal_control/interp/interp_table_1d.h"
#include "tycho/detail/optimal_control/interp/interp_table_2d.h"
#include "tycho/detail/optimal_control/interp/interp_table_3d.h"
#include "tycho/detail/optimal_control/interp/interp_table_4d.h"
#include "tycho/detail/optimal_control/phase/mesh_iterate_info.h"
#include "tycho/detail/optimal_control/phase/ode.h"
#include "tycho/detail/optimal_control/phase/ode_phase.h"
#include "tycho/detail/optimal_control/phase/ode_phase_base.h"
#include "tycho/detail/optimal_control/phase/optimal_control_problem.h"
#include "tycho/detail/optimal_control/transcription/lgl_interp_functions.h"
#include "tycho/detail/optimal_control/transcription/lgl_interp_table.h"
#include "tycho/detail/vf/operators/interp_compose.h"
