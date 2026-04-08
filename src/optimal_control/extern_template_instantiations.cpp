// Explicit instantiation definitions for common OC/integrator types.
// Matches the extern template declarations in
// include/tycho/detail/optimal_control/extern_templates.h.
//
// This TU compiles all the template method bodies that are suppressed in
// consuming TUs by extern template. It will be heavy (~4-8 GB RSS).

#include "tycho/tycho.h"

namespace tycho::oc {

using DynODE_  = GenericODE<vf::GenericFunction<-1, -1>, -1, -1, -1>;
using DynBODE_ = Blocked_ODE_Wrapper<DynODE_>;

// Defect types — direct ODE
template struct LGLDefects<DynODE_, 2>;
template struct LGLDefects<DynODE_, 3>;
template struct LGLDefects<DynODE_, 4>;
template struct TrapezoidalDefects<DynODE_>;

// Defect types — blocked ODE wrapper
template struct LGLDefects<DynBODE_, 2>;
template struct LGLDefects<DynBODE_, 3>;
template struct LGLDefects<DynBODE_, 4>;
template struct TrapezoidalDefects<DynBODE_>;

} // namespace tycho::oc

namespace tycho::integrators {

using DynODE_ = oc::GenericODE<vf::GenericFunction<-1, -1>, -1, -1, -1>;

template struct Integrator<DynODE_>;
template struct RKStepper<DynODE_, RKOptions::DOPRI5>;
template struct RKStepper<DynODE_, RKOptions::DOPRI87>;

} // namespace tycho::integrators
