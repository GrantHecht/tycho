#include "Tycho_OptimalControl.h"

namespace Tycho {

void GenericODESBuildPart1(FunctionRegistry &reg, nb::module_ &m) {

    Bind::BuildGenODEModule<GenericFunction<-1, -1>, -1, 0, 0>("ode_x", m, reg);
    Bind::BuildGenODEModule<GenericFunction<-1, -1>, -1, -1, 0>("ode_x_u", m, reg);
}

} // namespace Tycho