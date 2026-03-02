#pragma once
#ifdef TYCHO_PYTHON_BINDINGS

// TychoBind<ODEPhase<DODE>> specialization and ODEPhaseBuildImpl free function.
// Replaces the out-of-class ODEPhase<DODE>::Build() and BuildImpl() definitions
// that were previously included from ODEPhase.h.

namespace Tycho {

namespace Bind {

template <class DODE, class PyClass>
void ODEPhaseBuildImpl(PyClass &phase) {
    phase.def(nb::init<DODE, TranscriptionModes>());
    phase.def(nb::init<DODE, TranscriptionModes, const std::vector<Eigen::VectorXd> &, int>());
    phase.def(
        nb::init<DODE, TranscriptionModes, const std::vector<Eigen::VectorXd> &, int, bool>());

    phase.def(nb::init<DODE, std::string>());
    phase.def(nb::init<DODE, std::string, const std::vector<Eigen::VectorXd> &>());

    phase.def(nb::init<DODE, std::string, const std::vector<Eigen::VectorXd> &, int>());
    phase.def(nb::init<DODE, std::string, const std::vector<Eigen::VectorXd> &, int, bool>());
}

} // namespace Bind

template <class DODE>
struct TychoBind<ODEPhase<DODE>> {
    static void Build(nb::module_ &m) {
        auto phase = nb::class_<ODEPhase<DODE>, ODEPhaseBase>(m, "phase");

        Bind::ODEPhaseBuildImpl<DODE>(phase);
        phase.def_rw("integrator", &ODEPhase<DODE>::integrator);
        phase.def_rw("EnableHessianSparsity", &ODEPhase<DODE>::EnableHessianSparsity);
        phase.def_rw("OldShootingDefect", &ODEPhase<DODE>::OldShootingDefect);

        phase.def("get_input_scale", &ODEPhase<DODE>::get_input_scale);
        phase.def("get_defect", &ODEPhase<DODE>::get_defect);
    }
};

} // namespace Tycho

#endif // TYCHO_PYTHON_BINDINGS
