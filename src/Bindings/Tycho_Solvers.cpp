#include "Solvers/Tycho_Solvers.h"
#include "JetBind.h"
#include "OptimizationProblemBind.h"
#include "PSIOPTBind.h"

namespace Tycho {

void SolversBuild(FunctionRegistry &reg, nb::module_ &m) {
    auto &sol = reg.getSolversModule();
#ifndef USE_ACCELERATE_SPARSE
    int DSECOND = dsecnd();
#endif
    TychoBind<PSIOPT>::Build(sol);
    TychoBind<OptimizationProblemBase>::Build(sol);
    TychoBind<Jet>::Build(sol);
    TychoBind<OptimizationProblem>::Build(sol);
}

} // namespace Tycho
