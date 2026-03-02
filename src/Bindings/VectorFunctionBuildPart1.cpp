#include "Tycho_VectorFunctions.h"

namespace Tycho {
void VectorFunctionBuildPart1(FunctionRegistry &reg, nb::module_ &m) {
    using Gen = GenericFunction<-1, -1>;
    using GenS = GenericFunction<-1, 1>;

    Bind::UnaryMathBuild<Gen>(reg.vfuncx);
    Bind::BinaryMathBuild<Gen>(reg.vfuncx);
}

} // namespace Tycho