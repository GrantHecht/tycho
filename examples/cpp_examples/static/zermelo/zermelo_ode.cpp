///////////////////////////////////////////////////////////////////////////////
// Zermelo navigation ODEs — concrete VectorFunctions + erased factories
///////////////////////////////////////////////////////////////////////////////

#include "zermelo_ode.h"

#include <cmath>

namespace tycho_examples {

namespace {

using namespace tycho;
using namespace tycho::vf;
using namespace tycho::oc;

// No wind
struct ZermeloNoWind_Impl : ODESize<2, 1, 0> {
    static auto Definition(double vMax) {
        auto args = ODEArguments<2, 1, 0>();
        auto theta = args[UVar<0>];
        return StackedOutputs{vMax * cos(theta), vMax * sin(theta)};
    }
};
BUILD_ODE_FROM_EXPRESSION(ZermeloNoWind, ZermeloNoWind_Impl, double);

// Uniform wind
struct ZermeloUniformWind_Impl : ODESize<2, 1, 0> {
    static auto Definition(double vMax, double wVel, double wAng) {
        auto args = ODEArguments<2, 1, 0>();
        auto theta = args[UVar<0>];
        return StackedOutputs{vMax * cos(theta) + wVel * std::cos(wAng),
                              vMax * sin(theta) + wVel * std::sin(wAng)};
    }
};
BUILD_ODE_FROM_EXPRESSION(ZermeloUniformWind, ZermeloUniformWind_Impl, double, double, double);

// Constant-direction wind with position-dependent speed
struct ZermeloConstDirWind_Impl : ODESize<2, 1, 0> {
    static auto Definition(double vMax, double wAng) {
        auto args = ODEArguments<2, 1, 0>();
        auto pos = args[XVec];
        auto theta = args[UVar<0>];
        auto vel = cos(pos.norm());
        return StackedOutputs{vMax * cos(theta) + vel * std::cos(wAng),
                              vMax * sin(theta) + vel * std::sin(wAng)};
    }
};
BUILD_ODE_FROM_EXPRESSION(ZermeloConstDirWind, ZermeloConstDirWind_Impl, double, double);

// Variable-direction wind: speed and angle depend on position
struct ZermeloVarWind_Impl : ODESize<2, 1, 0> {
    static auto Definition(double vMax) {
        auto args = ODEArguments<2, 1, 0>();
        auto pos = args[XVec];
        auto x = args[XVar<0>];
        auto y = args[XVar<1>];
        auto theta = args[UVar<0>];
        auto vel = sin(pos.norm());
        auto ang = 2.0 * (x + y);
        return StackedOutputs{vMax * cos(theta) + vel * cos(ang),
                              vMax * sin(theta) + vel * sin(ang)};
    }
};
BUILD_ODE_FROM_EXPRESSION(ZermeloVarWind, ZermeloVarWind_Impl, double);

} // namespace

tycho::vf::GenericFunction<-1, -1> make_zermelo_no_wind_ode(double vMax) {
    return tycho::vf::GenericFunction<-1, -1>(ZermeloNoWind(vMax));
}

tycho::vf::GenericFunction<-1, -1>
make_zermelo_uniform_wind_ode(double vMax, double wVel, double wAng) {
    return tycho::vf::GenericFunction<-1, -1>(ZermeloUniformWind(vMax, wVel, wAng));
}

tycho::vf::GenericFunction<-1, -1>
make_zermelo_const_dir_wind_ode(double vMax, double wAng) {
    return tycho::vf::GenericFunction<-1, -1>(ZermeloConstDirWind(vMax, wAng));
}

tycho::vf::GenericFunction<-1, -1> make_zermelo_var_wind_ode(double vMax) {
    return tycho::vf::GenericFunction<-1, -1>(ZermeloVarWind(vMax));
}

} // namespace tycho_examples
