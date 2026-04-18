///////////////////////////////////////////////////////////////////////////////
// Zermelo navigation ODEs — type-erased factories
//
// Four wind models, one factory each, returning GenericFunction<-1,-1> so
// heavy compute_jacobian_adjointgradient_adjointhessian instantiations for
// each concrete wind-model expression tree live once in zermelo_ode.cpp.
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <tycho/tycho.h>

namespace tycho_examples {

// State: [x, y]  (2D position), Control: [theta]
// ODE:  xdot = vMax·cos(theta) + wx(x,y,t)
//       ydot = vMax·sin(theta) + wy(x,y,t)

tycho::vf::GenericFunction<-1, -1> make_zermelo_no_wind_ode(double vMax);

tycho::vf::GenericFunction<-1, -1>
make_zermelo_uniform_wind_ode(double vMax, double wVel, double wAng);

tycho::vf::GenericFunction<-1, -1>
make_zermelo_const_dir_wind_ode(double vMax, double wAng);

tycho::vf::GenericFunction<-1, -1> make_zermelo_var_wind_ode(double vMax);

} // namespace tycho_examples
