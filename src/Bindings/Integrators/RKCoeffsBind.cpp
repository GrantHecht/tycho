#include "Integrators/RKCoeffs.h"

namespace Tycho {

void RKFlagsBuild(nb::module_ &m) {
    nb::enum_<RKOptions>(m, "RKOptions")
        .value("RK4", RKOptions::RK4Classic)
        .value("DOPRI54", RKOptions::DOPRI54)
        .value("DOPRI87", RKOptions::DOPRI87);
}

} // namespace Tycho
