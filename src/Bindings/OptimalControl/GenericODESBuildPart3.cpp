#include "Tycho_OptimalControl.h"

namespace Tycho {

void GenericODESBuildPart3(FunctionRegistry &reg, nb::module_ &m) {

    Bind::BuildGenODEModule<GenericFunction<-1, -1>, 6, 3, 0>("ode_6_3", m, reg);
}

} // namespace Tycho