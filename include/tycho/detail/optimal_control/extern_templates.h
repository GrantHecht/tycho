#pragma once

// Extern template declarations for frequently-used OC/integrator types.
//
// These suppress implicit instantiation in consuming TUs. The matching
// explicit instantiation definitions live in
// src/optimal_control/extern_template_instantiations.cpp (the oc_instantiations
// library).
//
// Phase 1: fully-dynamic GenericODE variant only. Fixed-size variants
// (e.g., GenericODE<GF, 6, 3, 0>) can be added incrementally.

#include "tycho/detail/optimal_control/phase/ode.h"
#include "tycho/detail/optimal_control/transcription/lgl_defects.h"
#include "tycho/detail/optimal_control/transcription/trapezoidal_defects.h"
#include "tycho/detail/optimal_control/transcription/blocked_ode_wrapper.h"
#include "tycho/detail/integrators/integrator.h"
#include "tycho/detail/integrators/rk_steppers.h"

namespace tycho::oc {

// ---------------------------------------------------------------------------
// GenericODE<GenericFunction<-1,-1>, -1, -1, -1>  (fully dynamic ODE)
// ---------------------------------------------------------------------------
using DynODE_  = GenericODE<vf::GenericFunction<-1, -1>, -1, -1, -1>;
using DynBODE_ = Blocked_ODE_Wrapper<DynODE_>;

// Defect types — direct ODE
extern template struct LGLDefects<DynODE_, 2>;
extern template struct LGLDefects<DynODE_, 3>;
extern template struct LGLDefects<DynODE_, 4>;
extern template struct TrapezoidalDefects<DynODE_>;

// Defect types — blocked ODE wrapper
extern template struct LGLDefects<DynBODE_, 2>;
extern template struct LGLDefects<DynBODE_, 3>;
extern template struct LGLDefects<DynBODE_, 4>;
extern template struct TrapezoidalDefects<DynBODE_>;

} // namespace tycho::oc

namespace tycho::integrators {

using DynODE_ = oc::GenericODE<vf::GenericFunction<-1, -1>, -1, -1, -1>;

extern template struct Integrator<DynODE_>;
extern template struct RKStepper<DynODE_, RKOptions::DOPRI5>;
extern template struct RKStepper<DynODE_, RKOptions::DOPRI87>;

} // namespace tycho::integrators
