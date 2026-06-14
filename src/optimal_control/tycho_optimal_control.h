#pragma once
#include "tycho/optimal_control.h"

// Public detail headers (not pulled in by optimal_control.h directly)
#include "tycho/detail/vf/derivatives/fd_deriv_arbitrary.h"

// Binding headers (guarded by #ifdef TYCHO_PYTHON_BINDINGS, defined on binding CMake targets)
#include "bindings/integrators/integrator_bind.h"
#include "bindings/optimal_control/ode_bind.h"
#include "bindings/optimal_control/ode_phase_bind.h"
#include "bindings/optimal_control/ode_sizes_bind.h"

namespace tycho {} // namespace tycho
