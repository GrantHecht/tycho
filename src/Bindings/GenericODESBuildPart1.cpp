#include "Tycho_OptimalControl.h"

namespace Tycho {

void GenericODESBuildPart1(FunctionRegistry &reg, nb::module_ &m) {

    PythonGenericODE<-1, 0, 0>::BuildGenODEModule("ode_x", m, reg);
    PythonGenericODE<-1, -1, 0>::BuildGenODEModule("ode_x_u", m, reg);
}

} // namespace Tycho