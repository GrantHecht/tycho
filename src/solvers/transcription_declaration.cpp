// =============================================================================
// Tycho (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================

#include "tycho/detail/solvers_vf/transcription_declaration.h"

namespace tycho::solvers {

bool TranscriptionDeclaration::equality_pieces_address_distinct_rows() const {
    for (const auto &piece : this->declaration_.equality_constraints_) {
        if (!piece.index_data_.unique_constraints_) {
            return false;
        }
    }
    return true;
}

void TranscriptionDeclaration::lay(NonLinearProgram &host, int primal_vars, int equality_rows,
                                   int inequality_rows) {
    this->declaration_.primal_vars_ = primal_vars;
    this->declaration_.equality_rows_ = equality_rows;
    this->declaration_.inequality_rows_ = inequality_rows;

    // No internal fixing row exists yet at transcription time. The fixed-variable
    // treatment derives those from the bounds declared here, after the layout the
    // declaration asks for has run.
    this->declaration_.fixing_rows_ = 0;

    if (this->equality_pieces_address_distinct_rows()) {
        host.adopt_declaration(std::move(this->declaration_));
        return;
    }

    // An accumulating integrand declares several pieces that all sum into one
    // equality row. The declaration boundary counts each of those pieces' rows
    // separately, so the piece total exceeds the row space and the boundary
    // refuses a declaration that describes a perfectly ordinary problem. Until
    // that boundary accounts for pieces sharing constraint rows, such a
    // declaration is installed on the program and laid from its own dimensions.
    //
    // The declaration is still the value that carries the transcription: the
    // same pieces, carrying the same thread modes, the same bound records in the
    // same order, and the same partition count. Only the entry differs, and the
    // steps below are the adopting entry's own, with the fixing-row lift and
    // re-append that a transcription-time declaration has no rows for.
    host.objectives_ = std::move(this->declaration_.objectives_);
    host.equality_constraints_ = std::move(this->declaration_.equality_constraints_);
    host.inequality_constraints_ = std::move(this->declaration_.inequality_constraints_);

    host.clear_variable_bounds();
    for (const auto &bound : this->declaration_.variable_bounds_) {
        host.set_variable_bound(bound.index_, bound.lower_, bound.upper_);
    }

    host.num_partitions_ = this->declaration_.partition_count_;
    host.make_nlp(primal_vars, equality_rows, inequality_rows);
}

} // namespace tycho::solvers
