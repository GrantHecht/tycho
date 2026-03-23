#pragma once
#include "tycho/optimal_control.h"

// Forwarding headers to public detail/ headers (included here, not via optimal_control.h)
#include "FDDerivArbitrary.h"
#include "FDDerivUniform.h"

// Binding headers (guarded by #ifdef TYCHO_PYTHON_BINDINGS, defined on binding CMake targets)
#include "Bindings/Integrators/IntegratorBind.h"
#include "Bindings/OptimalControl/ODEBind.h"
#include "Bindings/OptimalControl/ODEPhaseBind.h"
#include "Bindings/OptimalControl/ODESizesBind.h"

namespace Tycho {} // namespace Tycho
