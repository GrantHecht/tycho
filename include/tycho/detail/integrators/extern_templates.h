// =============================================================================
// Tycho fork (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================

#pragma once

// Extern template declarations for frequently-used Integrator<DODE>
// specializations. Consumers that include this header opt into skipping
// implicit instantiation; the matching explicit instantiation definitions
// live in src/integrators/extern_template_instantiations.cpp (the
// integrators_instantiations static library).
//
// This header is deliberately NOT included from tycho/integrators.h because
// the extern template declaration requires the full Kepler type to be
// visible (Kepler::IRC is used by Integrator<Kepler>'s VectorFunction base
// specialization). Most consumers of integrator.h do not need or want the
// transitive kepler_model.h include; include this bridge header explicitly
// from the TU that benefits (currently: src/bindings/astro/kepler_integrator.cpp
// and src/integrators/extern_template_instantiations.cpp).

#include "tycho/detail/astro/kepler_model.h"
#include "tycho/detail/integrators/integrator.h"

namespace tycho::integrators {
extern template struct Integrator<tycho::astro::Kepler>;
} // namespace tycho::integrators
