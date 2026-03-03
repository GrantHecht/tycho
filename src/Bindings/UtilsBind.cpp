#include "MemoryManagerBind.h"
#include "Utils/Tycho_Utils.h"

namespace Tycho {

void UtilsBuild(nb::module_ &m) {
    auto um = m.def_submodule("Utils", "Contains miscilanaeous utilities");
    um.def("get_core_count", &Tycho::get_core_count);
    TychoBind<MemoryManager>::Build(um);
}

} // namespace Tycho
