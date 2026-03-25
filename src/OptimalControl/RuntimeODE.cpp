// =============================================================================
// Tycho — RuntimeODE::phase() implementation
//
// Defined out-of-line to avoid pulling ODEPhase.h (heavy template) into the
// RuntimeODE header.
//
// Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt
// =============================================================================

#include "tycho/detail/ODEPhase.h"
#include "tycho/detail/phase_wrapper.h"
#include "tycho/detail/runtime_ode.h"

namespace Tycho {

Phase RuntimeODE::phase(TranscriptionModes mode, const std::vector<Eigen::VectorXd> &traj,
                        int num_segments) const {
    DynODE ode = generic_ode();

    // Populate the GenericODE's ODESize FlatMap from the registry so that
    // ODEPhaseBase's string-based VarIndexType overloads also work.
    // ODEPhase's constructor auto-copies via set_idxs(this->ode.get_idxs()).
    if (registry_) {
        for (const auto &[name, idxs] : registry_->entries()) {
            ode.add_idx(name, idxs);
        }
    }

    auto phase_ptr = std::make_shared<ODEPhase<DynODE>>(ode, mode, traj, num_segments);

    if (registry_) {
        return Phase(phase_ptr, *registry_);
    }
    return Phase(phase_ptr, VarRegistry(xvars_, uvars_, pvars_));
}

} // namespace Tycho
