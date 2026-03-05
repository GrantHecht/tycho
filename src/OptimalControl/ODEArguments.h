#pragma once
#include "ODESizes.h"
#include "VectorFunctions/Tycho_VectorFunctions.h"

namespace Tycho {

template <int _XV, int _UV, int _PV>
struct ODEArguments : Arguments<ODESize<_XV, _UV, _PV>::XtUPV>, ODESize<_XV, _UV, _PV> {

    using Base = Arguments<ODESize<_XV, _UV, _PV>::XtUPV>;

    ODEArguments(int Xv, int Uv, int Pv) : Base(Xv + Uv + Pv + 1) { this->setXtUPVars(Xv, Uv, Pv); }
    ODEArguments(int Xv, int Uv) : ODEArguments(Xv, Uv, 0) {}
    ODEArguments(int Xv) : ODEArguments(Xv, 0) {}
};

} // namespace Tycho
