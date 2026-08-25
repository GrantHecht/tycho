// =============================================================================
// Tycho (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================

#include "tycho/detail/solvers_vf/transcription_declaration.h"

#include <cstdint>

namespace tycho::solvers {

namespace {

/// One piece's claimed row count, computed from its own dimensions in 64-bit
/// arithmetic. ConstraintFunction::num_con_eles() multiplies output rows by
/// application count in `int`, which is not safe to accumulate from here: a
/// piece with a large output-row count or application count can overflow
/// that product before this function ever sees the result.
/// @throws std::invalid_argument if either dimension is negative.
std::int64_t claimed_rows(const ConstraintFunction &piece) {
    const std::int64_t output_rows = piece.function_.output_rows();
    const std::int64_t applications = piece.index_data_.num_appl();
    if (output_rows < 0 || applications < 0) {
        throw std::invalid_argument(fmt::format(
            "TranscriptionDeclaration: an equality piece reports {0} output rows over {1} "
            "applications; neither may be negative",
            output_rows, applications));
    }
    return output_rows * applications;
}

} // namespace

int TranscriptionDeclaration::equality_piece_rows() const {
    std::int64_t rows = 0;
    for (const auto &piece : this->declaration_.equality_constraints_) {
        rows += claimed_rows(piece);
    }
    if (rows < 0 || rows > static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(
            fmt::format("TranscriptionDeclaration: the equality pieces claim {0} rows in total, "
                        "outside the [0, {1}] a declaration can state",
                        rows, std::numeric_limits<int>::max()));
    }
    return static_cast<int>(rows);
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

    // An accumulating integrand marks its pieces as sharing rows (see
    // PhaseIndexer::add_accumulation, which clears unique_constraints_ on both
    // the accumulator piece and the integrand piece it feeds), and only those
    // pieces' claimed row total can run past the declared equality-row count.
    // A transcription that declares no such piece has an equality piece sum
    // that must equal the declared row count exactly, so its overcount is
    // fixed at zero rather than computed as a difference -- that keeps hven's
    // piece-sum conjunct (claimed == declared + overcount) a live check
    // against a drifting piece list for every transcription but the
    // accumulating-integrand one, instead of a tautology for all of them.
    bool shares_rows = false;
    for (const auto &piece : this->declaration_.equality_constraints_) {
        shares_rows = shares_rows || !piece.index_data_.unique_constraints_;
    }

    if (shares_rows) {
        const std::int64_t claimed = this->equality_piece_rows();
        const std::int64_t overcount = claimed - static_cast<std::int64_t>(equality_rows);
        if (overcount < 0 ||
            overcount > static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument(fmt::format(
                "TranscriptionDeclaration: the equality pieces claim {0} rows in total, but the "
                "declaration states {1} equality rows; a shared-row overcount is how far the "
                "piece sum runs past the declared row count, and it cannot be negative or past "
                "what a declaration can state",
                claimed, equality_rows));
        }
        this->declaration_.equality_shared_row_overcount_ = static_cast<int>(overcount);
    } else {
        this->declaration_.equality_shared_row_overcount_ = 0;
    }

    host.adopt_declaration(std::move(this->declaration_));
}

} // namespace tycho::solvers
