#include "Tycho_OptimalControl.h"

namespace Tycho {

void GenericODESBuildPart3(FunctionRegistry &reg, nb::module_ &m) {

    PythonGenericODE<6, 3, 0>::BuildGenODEModule("ode_6_3", m, reg);
}

} // namespace Tycho