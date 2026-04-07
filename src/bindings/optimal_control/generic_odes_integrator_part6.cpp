// =============================================================================
// Tycho — Integrator bindings for generic ODE size combinations (part 6).
// Split from generic_odes_build_part6.cpp to reduce per-TU memory usage.
// Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt
// =============================================================================

#include "tycho_optimal_control.h"

namespace tycho {
using namespace tycho::vf;
using namespace tycho::oc;
using namespace tycho::integrators;

void GenericODESIntegratorPart6(FunctionRegistry &reg, nb::module_ &m) {

    bind::BuildGenODEIntegrator<GenericFunction<-1, -1>, 6, 0, 0>("ode_6", m, reg);
}

} // namespace tycho
