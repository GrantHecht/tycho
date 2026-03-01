#include "Tycho_OptimalControl.h"

namespace Tycho {

void GenericODESBuildPart6(FunctionRegistry &reg, py::module &m) {

    PythonGenericODE<6, 0, 0>::BuildGenODEModule("ode_6", m, reg);

    PythonGenericODE<4, 0, 0>::BuildGenODEModule("ode_4", m, reg);
}

} // namespace Tycho