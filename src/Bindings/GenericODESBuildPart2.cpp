#include "Tycho_OptimalControl.h"

namespace Tycho {

void GenericODESBuildPart2(FunctionRegistry &reg, nb::module_ &m) {

    Bind::BuildGenODEModule<GenericFunction<-1, -1>, -1, -1, -1>("ode_x_u_p", m, reg);
}

} // namespace Tycho