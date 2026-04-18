///////////////////////////////////////////////////////////////////////////////
// HyperSens ODE — concrete VectorFunction + type-erased factory
//
// Heavy compute_jacobian_adjointgradient_adjointhessian instantiations for
// HyperSens live here (one TU, one emission) rather than in main.cpp.
///////////////////////////////////////////////////////////////////////////////

#include "hypersens_ode.h"

namespace tycho_examples {

namespace {

using namespace tycho;
using namespace tycho::vf;
using namespace tycho::oc;

struct HyperSens_Impl : ODESize<1, 1, 0> {
    static auto Definition() {
        auto args = ODEArguments<1, 1, 0>();
        auto x = args[XVar<0>];
        auto u = args[UVar<0>];
        return u - x;
    }
};

BUILD_ODE_FROM_EXPRESSION(HyperSens, HyperSens_Impl);

} // namespace

tycho::vf::GenericFunction<-1, -1> make_hypersens_ode() {
    return tycho::vf::GenericFunction<-1, -1>(HyperSens());
}

} // namespace tycho_examples
