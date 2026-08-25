// =============================================================================
// Tycho (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================

#include "tycho/detail/solvers_vf/transcription_declaration.h"

namespace tycho::solvers {

int TranscriptionDeclaration::equality_piece_rows() const {
    long long rows = 0;
    for (const auto &piece : this->declaration_.equality_constraints_) {
        rows += piece.num_con_eles();
    }
    if (rows > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(
            fmt::format("TranscriptionDeclaration: the equality pieces claim {0} rows in total, "
                        "past the {1} a declaration can state",
                        rows, std::numeric_limits<int>::max()));
    }
    return int(rows);
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

    // An accumulating integrand declares several pieces that all sum into one
    // equality row, so the equality pieces' claimed row total can run past the
    // declared equality-row count. The declaration boundary states that excess
    // directly rather than refusing it: how far the piece sum runs past the
    // declared count is exactly what a shared-row overcount is for. Zero for a
    // transcription whose pieces claim disjoint rows, which is every
    // transcription but the accumulating-integrand one.
    this->declaration_.equality_shared_row_overcount_ = this->equality_piece_rows() - equality_rows;

    host.adopt_declaration(std::move(this->declaration_));
}

} // namespace tycho::solvers
