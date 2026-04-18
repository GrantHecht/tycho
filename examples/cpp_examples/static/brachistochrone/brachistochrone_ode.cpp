///////////////////////////////////////////////////////////////////////////////
// Brachistochrone ODE — concrete VectorFunction + type-erased factory
//
// Defines the concrete Brachistochrone_Impl + BUILD_ODE_FROM_EXPRESSION here
// so that all heavy compute_jacobian_adjointgradient_adjointhessian template
// instantiations for this ODE are emitted exactly once — in this TU — rather
// than in main.cpp.
//
// See doc/user_guide_example_tu_split.md for the pattern rationale.
///////////////////////////////////////////////////////////////////////////////

#include "brachistochrone_ode.h"

namespace tycho_examples {

namespace {

using namespace tycho;
using namespace tycho::vf;
using namespace tycho::oc;

struct Brachistochrone_Impl : ODESize<3, 1, 0> {
    static auto Definition(double g) {
        auto args = ODEArguments<3, 1, 0>();
        auto v = args[XVar<2>];
        auto theta = args[UVar<0>];

        auto xdot = sin(theta) * v;
        auto ydot = cos(theta) * v * (-1.0);
        auto vdot = g * cos(theta);

        return StackedOutputs{xdot, ydot, vdot};
    }
};

BUILD_ODE_FROM_EXPRESSION(Brachistochrone, Brachistochrone_Impl, double);

} // namespace

tycho::vf::GenericFunction<-1, -1> make_brachistochrone_ode(double g) {
    return tycho::vf::GenericFunction<-1, -1>(Brachistochrone(g));
}

} // namespace tycho_examples
