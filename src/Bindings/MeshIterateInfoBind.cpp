#include "MeshIterateInfoBind.h"

using namespace Tycho;

void TychoBind<MeshIterateInfo>::Build(nb::module_ &m) {
    auto obj = nb::class_<MeshIterateInfo>(m, "MeshIterateInfo");

    obj.def_ro("times", &MeshIterateInfo::times);
    obj.def_ro("error", &MeshIterateInfo::error);
    obj.def_ro("distribution", &MeshIterateInfo::distribution);
    obj.def_ro("distintegral", &MeshIterateInfo::distintegral);
    obj.def_ro("avg_error", &MeshIterateInfo::avg_error);
    obj.def_ro("max_error", &MeshIterateInfo::max_error);
    obj.def_ro("numsegs", &MeshIterateInfo::numsegs);
    obj.def_ro("converged", &MeshIterateInfo::converged);
}
