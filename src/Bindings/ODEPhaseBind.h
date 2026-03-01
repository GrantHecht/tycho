#pragma once
#ifdef TYCHO_PYTHON_BINDINGS

// Out-of-class definitions of ODEPhase binding methods.
// Included from ODEPhase.h under TYCHO_PYTHON_BINDINGS.

namespace Tycho {

template <class DODE>
template <class PyClass>
void ODEPhase<DODE>::BuildImpl(PyClass &phase) {
    phase.def(nb::init<DODE, TranscriptionModes>());
    phase.def(nb::init<DODE, TranscriptionModes, const std::vector<Eigen::VectorXd> &, int>());
    phase.def(
        nb::init<DODE, TranscriptionModes, const std::vector<Eigen::VectorXd> &, int, bool>());

    phase.def(nb::init<DODE, std::string>());
    phase.def(nb::init<DODE, std::string, const std::vector<Eigen::VectorXd> &>());

    phase.def(nb::init<DODE, std::string, const std::vector<Eigen::VectorXd> &, int>());
    phase.def(nb::init<DODE, std::string, const std::vector<Eigen::VectorXd> &, int, bool>());
}

template <class DODE>
void ODEPhase<DODE>::Build(nb::module_ &m) {
    auto phase =
        nb::class_<ODEPhase<DODE>, ODEPhaseBase>(m, "phase");

    BuildImpl(phase);
    phase.def_rw("integrator", &ODEPhase<DODE>::integrator);
    phase.def_rw("EnableHessianSparsity", &ODEPhase<DODE>::EnableHessianSparsity);
    phase.def_rw("OldShootingDefect", &ODEPhase<DODE>::OldShootingDefect);

    phase.def("get_input_scale", &ODEPhase<DODE>::get_input_scale);
    phase.def("get_defect", &ODEPhase<DODE>::get_defect);
}

} // namespace Tycho

#endif // TYCHO_PYTHON_BINDINGS
