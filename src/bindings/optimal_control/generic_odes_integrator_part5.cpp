// =============================================================================
// Tycho — Integrator bindings for generic ODE size combinations (part 5).
// Split from generic_odes_build_part5.cpp to reduce per-TU memory usage.
// Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt
// =============================================================================

#include "tycho_optimal_control.h"

namespace tycho {
using namespace tycho::vf;
using namespace tycho::oc;
using namespace tycho::integrators;

void GenericODESIntegratorPart5(FunctionRegistry &reg, nb::module_ &m) {

    bind::BuildGenODEIntegrator<GenericFunction<-1, -1>, 2, 1, 0>("ode_2_1", m, reg);
}

} // namespace tycho
