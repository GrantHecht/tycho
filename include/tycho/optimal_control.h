#pragma once

// Tycho — Phase/ODE transcription, collocation, mesh refinement

#include "tycho/vector_functions.h"

#include "tycho/detail/optimal_control/core/optimal_control_flags.h"
#include "tycho/detail/optimal_control/core/ode_sizes.h"
#include "tycho/detail/optimal_control/core/ode_arguments.h"
#include "tycho/detail/optimal_control/core/interface_types.h"
#include "tycho/detail/optimal_control/core/state_function.h"
#include "tycho/detail/optimal_control/core/link_function.h"
#include "tycho/detail/optimal_control/phase/ode.h"
#include "tycho/detail/optimal_control/phase/ode_phase_base.h"
#include "tycho/detail/optimal_control/phase/ode_phase.h"
#include "tycho/detail/optimal_control/phase/optimal_control_problem.h"
#include "tycho/detail/optimal_control/transcription/lgl_interp_table.h"
#include "tycho/detail/optimal_control/transcription/lgl_interp_functions.h"
#include "tycho/detail/optimal_control/interp/interp_table_1d.h"
#include "tycho/detail/optimal_control/interp/interp_table_2d.h"
#include "tycho/detail/optimal_control/interp/interp_table_3d.h"
#include "tycho/detail/optimal_control/interp/interp_table_4d.h"
#include "tycho/detail/vf/operators/interp_compose.h"
#include "tycho/detail/optimal_control/phase/mesh_iterate_info.h"
#include "tycho/detail/optimal_control/builder/var_registry.h"
#include "tycho/detail/optimal_control/builder/runtime_ode.h"
#include "tycho/detail/optimal_control/builder/ode_builder.h"
#include "tycho/detail/optimal_control/builder/phase_wrapper.h"
#include "tycho/detail/optimal_control/builder/ocp_wrapper.h"
