// =============================================================================
// Tycho — Integrator<Kepler> bindings.
// Split from kepler_model.cpp to reduce per-TU memory usage.
// Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt
// =============================================================================

#include "astro/tycho_astro.h"
#include "optimal_control/tycho_optimal_control.h"

namespace tycho {
using namespace tycho::vf;
using namespace tycho::oc;
using namespace tycho::astro;
using namespace tycho::integrators;
void BuildKeplerIntegrator(FunctionRegistry &reg, nb::module_ &m);
} // namespace tycho

void tycho::BuildKeplerIntegrator(FunctionRegistry &reg, nb::module_ &m) {
    auto odemod = nb::borrow<nb::module_>(m.attr("Kepler"));
    reg.template Build_Register<Integrator<Kepler>>(odemod, "integrator");
}
