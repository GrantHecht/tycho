///////////////////////////////////////////////////////////////////////////////
// Brachistochrone ODE — type-erased factory
//
// Declares a factory that returns a GenericFunction<-1,-1> wrapping the
// concrete Brachistochrone VectorFunction. The concrete type and all its
// compute_jacobian_adjointgradient_adjointhessian template instantiations
// live in brachistochrone_ode.cpp so main.cpp does not re-instantiate them.
//
// See docs/dev/notes/user_guide_example_tu_split.md for the pattern rationale.
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <tycho/tycho.h>

namespace tycho_examples {

// Factory. Returns a type-erased VectorFunction implementing:
//   State  : [x, y, v]
//   Control: [theta]
//   xdot =  sin(theta) * v
//   ydot = -cos(theta) * v
//   vdot =  g * cos(theta)
tycho::vf::GenericFunction<-1, -1> make_brachistochrone_ode(double g);

} // namespace tycho_examples
