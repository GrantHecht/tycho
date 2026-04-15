// =============================================================================
// Tycho — ODE::phase() and IntegratorBuilder::build() implementations
//
// Defined out-of-line to avoid dragging ode_phase.h (and the ODEPhaseBase
// transcription machinery) into the runtime_ode.h public header.
//
// Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt
// =============================================================================

#include "tycho/detail/optimal_control/builder/runtime_ode.h"
#include "tycho/detail/optimal_control/builder/phase_wrapper.h"
#include "tycho/detail/optimal_control/phase/ode_phase.h"

#include <fmt/format.h>

namespace tycho {

ODE::DynODE ODE::make_dyn_ode() const {
    DynODE ode = generic_ode();
    if (registry_) {
        for (const auto &[name, idxs] : registry_->entries())
            ode.add_idx(name, idxs);
    }
    return ode;
}

Phase ODE::phase(TranscriptionModes mode, const std::vector<Eigen::VectorXd> &traj,
                 int num_segments) const {
    if (traj.empty())
        throw std::invalid_argument("ODE::phase: trajectory must not be empty");
    if (num_segments <= 0)
        throw std::invalid_argument(
            fmt::format("ODE::phase: num_segments must be positive (got {})", num_segments));
    int expected = xtup_size();
    for (size_t i = 0; i < traj.size(); ++i) {
        if (traj[i].size() != expected) {
            throw std::invalid_argument(
                fmt::format("ODE::phase: trajectory point {} has size {}, expected {} "
                            "(XtUP = {}+1+{}+{})",
                            i, traj[i].size(), expected, xvars_, uvars_, pvars_));
        }
    }

    DynODE ode = make_dyn_ode();

    auto phase_ptr = std::make_shared<oc::ODEPhase<DynODE>>(ode, mode, traj, num_segments);

    if (registry_) {
        return Phase(phase_ptr, *registry_);
    }
    return Phase(phase_ptr, VarRegistry(xvars_, uvars_, pvars_));
}

IntegratorBuilder ODE::integrator() const { return IntegratorBuilder(*this); }

ODE::DynIntegrator IntegratorBuilder::build() const {
    if (step_ <= 0.0) {
        throw std::logic_error(
            "IntegratorBuilder: .step(defstep) must be called with a positive value before "
            ".build() (current step_ = " +
            std::to_string(step_) + ")");
    }

    auto ode = ode_->make_dyn_ode();

    switch (kind_) {
    case ControlKind::None:
        return ODE::DynIntegrator(ode, method_, step_);

    case ControlKind::IndexedLaw:
        if (varlocs_.size() == 0) {
            throw std::logic_error("IntegratorBuilder::build: .control(ulaw, varlocs) was called "
                                   "with an empty varlocs vector");
        }
        return ODE::DynIntegrator(ode, method_, step_, ulaw_, varlocs_);

    case ControlKind::NamedLaw: {
        if (name_varlocs_.empty()) {
            throw std::logic_error("IntegratorBuilder::build: .control(ulaw, names) was called "
                                   "with an empty name list");
        }
        ode_->check_registry();
        Eigen::VectorXi resolved = ode_->registry_->resolve(name_varlocs_);
        return ODE::DynIntegrator(ode, method_, step_, ulaw_, resolved);
    }

    case ControlKind::Const: {
        if (u_const_.size() == 0) {
            throw std::logic_error("IntegratorBuilder::build: .control(u_const) was called with "
                                   "an empty u_const vector");
        }
        Eigen::VectorXi tloc(1);
        tloc[0] = ode.t_var();
        GenericFunction<-1, -1> ucon = vf::Constant<-1, -1>(1, u_const_);
        return ODE::DynIntegrator(ode, method_, step_, ucon, tloc);
    }

    case ControlKind::TableIndexed:
        if (!tab_) {
            throw std::logic_error("IntegratorBuilder::build: .control(tab, ulocs) was called "
                                   "with a null LGLInterpTable");
        }
        if (varlocs_.size() == 0) {
            throw std::logic_error("IntegratorBuilder::build: .control(tab, ulocs) was called "
                                   "with an empty ulocs vector");
        }
        return ODE::DynIntegrator(ode, method_, step_, tab_, varlocs_);

    case ControlKind::TableAuto:
        if (!tab_) {
            throw std::logic_error(
                "IntegratorBuilder::build: .control(tab) was called with a null LGLInterpTable");
        }
        return ODE::DynIntegrator(ode, method_, step_, tab_);
    }
    // Exhaustive switch above — no default:. Any ControlKind added later
    // without a case label is a clang -Wswitch error at compile time.
    __builtin_unreachable();
}

} // namespace tycho
