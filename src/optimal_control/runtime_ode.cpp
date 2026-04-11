// =============================================================================
// Tycho — ODE::phase() and ODE::integrator() implementations
//
// Defined out-of-line to avoid pulling heavy template headers into the
// ODE header.
//
// Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt
// =============================================================================

#include "tycho/detail/optimal_control/builder/runtime_ode.h"
#include "tycho/detail/optimal_control/builder/phase_wrapper.h"
#include "tycho/detail/optimal_control/phase/ode_phase.h"

#include <fmt/format.h>

namespace tycho {

// ── Private helper ─────────────────────────────────────────────────────

ODE::DynODE ODE::make_dyn_ode() const {
    DynODE ode = generic_ode();
    if (registry_) {
        for (const auto &[name, idxs] : registry_->entries())
            ode.add_idx(name, idxs);
    }
    return ode;
}

// ── Phase construction ─────────────────────────────────────────────────

Phase ODE::phase(TranscriptionModes mode, const std::vector<Eigen::VectorXd> &traj,
                 int num_segments, bool lerp_ig) const {
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

    if (lerp_ig) {
        phase_ptr->set_traj(traj, num_segments, true);
    }

    if (registry_) {
        return Phase(phase_ptr, *registry_);
    }
    return Phase(phase_ptr, VarRegistry(xvars_, uvars_, pvars_));
}

// ── Integrator construction ────────────────────────────────────────────

// Basic (no control law)

ODE::DynIntegrator ODE::integrator(double defstep) const {
    return DynIntegrator(make_dyn_ode(), defstep);
}

ODE::DynIntegrator ODE::integrator(IVPAlg method, double defstep) const {
    return DynIntegrator(make_dyn_ode(), method, defstep);
}

// Control law — index-based

ODE::DynIntegrator ODE::integrator(double defstep, GenericFunction<-1, -1> ulaw,
                                   const Eigen::VectorXi &varlocs) const {
    return DynIntegrator(make_dyn_ode(), defstep, ulaw, varlocs);
}

ODE::DynIntegrator ODE::integrator(IVPAlg method, double defstep,
                                   GenericFunction<-1, -1> ulaw,
                                   const Eigen::VectorXi &varlocs) const {
    return DynIntegrator(make_dyn_ode(), method, defstep, ulaw, varlocs);
}

// Control law — named variable locations

ODE::DynIntegrator ODE::integrator(double defstep, GenericFunction<-1, -1> ulaw,
                                   std::initializer_list<std::string> varlocs) const {
    check_registry();
    return DynIntegrator(make_dyn_ode(), defstep, ulaw, registry_->resolve(varlocs));
}

ODE::DynIntegrator ODE::integrator(IVPAlg method, double defstep,
                                   GenericFunction<-1, -1> ulaw,
                                   std::initializer_list<std::string> varlocs) const {
    check_registry();
    return DynIntegrator(make_dyn_ode(), method, defstep, ulaw, registry_->resolve(varlocs));
}

// Constant control vector

ODE::DynIntegrator ODE::integrator(double defstep, const Eigen::VectorXd &u_const) const {
    return DynIntegrator(make_dyn_ode(), defstep, u_const);
}

ODE::DynIntegrator ODE::integrator(IVPAlg method, double defstep,
                                   const Eigen::VectorXd &u_const) const {
    auto ode = make_dyn_ode();
    Eigen::VectorXi tloc(1);
    tloc[0] = ode.t_var();
    GenericFunction<-1, -1> ucon = vf::Constant<-1, -1>(1, u_const);
    return DynIntegrator(ode, method, defstep, ucon, tloc);
}

// LGL interpolation table + explicit control locations

ODE::DynIntegrator ODE::integrator(double defstep, std::shared_ptr<oc::LGLInterpTable> tab,
                                   const Eigen::VectorXi &ulocs) const {
    return DynIntegrator(make_dyn_ode(), defstep, std::move(tab), ulocs);
}

ODE::DynIntegrator ODE::integrator(IVPAlg method, double defstep,
                                   std::shared_ptr<oc::LGLInterpTable> tab,
                                   const Eigen::VectorXi &ulocs) const {
    return DynIntegrator(make_dyn_ode(), method, defstep, std::move(tab), ulocs);
}

// LGL interpolation table (auto control locations)

ODE::DynIntegrator ODE::integrator(double defstep,
                                   std::shared_ptr<oc::LGLInterpTable> tab) const {
    return DynIntegrator(make_dyn_ode(), defstep, std::move(tab));
}

ODE::DynIntegrator ODE::integrator(IVPAlg method, double defstep,
                                   std::shared_ptr<oc::LGLInterpTable> tab) const {
    return DynIntegrator(make_dyn_ode(), method, defstep, std::move(tab));
}

} // namespace tycho
