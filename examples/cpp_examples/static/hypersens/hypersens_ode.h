///////////////////////////////////////////////////////////////////////////////
// HyperSens ODE — type-erased factory
//
// Declares a factory returning a GenericFunction<-1,-1> wrapping the concrete
// HyperSens VectorFunction. Heavy template instantiations live in
// hypersens_ode.cpp so main.cpp doesn't re-emit them.
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <tycho/tycho.h>

namespace tycho_examples {

// Linear hypersensitive ODE: xdot = u - x
tycho::vf::GenericFunction<-1, -1> make_hypersens_ode();

} // namespace tycho_examples
