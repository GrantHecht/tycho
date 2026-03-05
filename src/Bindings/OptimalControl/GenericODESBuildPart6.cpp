#include "Tycho_OptimalControl.h"

namespace Tycho {

void GenericODESBuildPart6(FunctionRegistry &reg, nb::module_ &m) {

    Bind::BuildGenODEModule<GenericFunction<-1, -1>, 6, 0, 0>("ode_6", m, reg);

    Bind::BuildGenODEModule<GenericFunction<-1, -1>, 4, 0, 0>("ode_4", m, reg);
}

} // namespace Tycho