#pragma once
#include "tycho/optimal_control.h"

// Public detail headers (not pulled in by optimal_control.h directly)
#include "tycho/detail/vf/derivatives/fd_deriv_arbitrary.h"
#include "tycho/detail/vf/derivatives/fd_deriv_uniform.h"

// Binding headers (guarded by #ifdef TYCHO_PYTHON_BINDINGS, defined on binding CMake targets)
#include "Bindings/Integrators/IntegratorBind.h"
#include "Bindings/OptimalControl/ODEBind.h"
#include "Bindings/OptimalControl/ODEPhaseBind.h"
#include "Bindings/OptimalControl/ODESizesBind.h"

namespace Tycho {} // namespace Tycho
