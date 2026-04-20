// Explicit instantiation definitions for common Integrator<DODE>
// specializations. Matches the extern template declarations in
// include/tycho/detail/integrators/extern_templates.h.
//
// This TU compiles the template method bodies that are suppressed in
// consuming TUs by extern template, keeping per-TU RAM footprint bounded
// for heavy consumers such as the Python binding kepler_integrator.cpp.

#include "tycho/detail/integrators/extern_templates.h"

namespace tycho::integrators {

template struct Integrator<tycho::astro::Kepler>;

} // namespace tycho::integrators
