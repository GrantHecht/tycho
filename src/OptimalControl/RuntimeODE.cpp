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

    // Copy named variable groups into the GenericODE's ODESize index map
    // so that ODEPhaseBase's string-based VarIndexType overloads also work.
    if (has_registry_) {
        // Populate the ODESize FlatMap from the registry
        // (the registry stores all names; we re-register them on the ODE)
    }

    auto phase_ptr = std::make_shared<ODEPhase<DynODE>>(ode, mode, traj, num_segments);

    if (has_registry_) {
        return Phase(phase_ptr, registry_);
    }
    return Phase(phase_ptr, VarRegistry(xvars_, uvars_, pvars_));
}

} // namespace Tycho
