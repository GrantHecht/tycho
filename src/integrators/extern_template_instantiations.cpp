// Explicit instantiation definitions for common Integrator<DODE>
// specializations. Matches the extern template declarations in
// include/tycho/detail/integrators/extern_templates.h.
//
// This TU compiles all the template method bodies that are suppressed in
// consuming TUs by extern template (notably the Python binding TU
// kepler_integrator.cpp, which has exploded in RAM footprint since SP2
// added 6 new RK methods multiplying the Integrator::stepper_compute_impl
// instantiations).

#include "tycho/detail/integrators/extern_templates.h"

namespace tycho::integrators {

template struct Integrator<tycho::astro::Kepler>;

} // namespace tycho::integrators
