#include "Tycho_OptimalControl.h"

namespace Tycho {

template <int XV, int UV, int PV>
using GODE = GenericODE<GenericFunction<-1, (XV == 1) ? 1 : -1>, XV, UV, PV>;

void GenericODESBuildPart5(FunctionRegistry &reg, nb::module_ &m) {

    Bind::BuildGenODEModule<GenericFunction<-1, -1>, 2, 1, 0>("ode_2_1", m, reg);
}

} // namespace Tycho