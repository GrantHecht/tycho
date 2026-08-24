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

void TranscriptionDeclaration::validate_except_equality_row_sum() {
    // The declared counts in their own right. The substitution below hands the
    // boundary a different equality-row count, so the declared one is checked
    // here instead of there.
    if (this->declaration_.equality_rows_ < 0) {
        throw std::invalid_argument(
            fmt::format("TranscriptionDeclaration: the equality-row count is {0}, which is not a "
                        "count",
                        this->declaration_.equality_rows_));
    }

    // Everything else the declaration boundary refuses -- negative dimensions,
    // the partition floor, the INEQUALITY piece sum, bound indices, NaN bounds,
    // inverted records and empty intersections -- runs unchanged. Only the
    // equality piece sum is stood down, and only by handing the boundary the sum
    // the shared rows actually produce, so that one conjunct passes and no other
    // is skipped. The declaration is this object's own and is restored before the
    // call returns, including on the refusal path.
    AggregateDeclaration &declaration = this->declaration_;
    const int declared_equality_rows = declaration.equality_rows_;
    declaration.equality_rows_ = this->equality_piece_rows();
    try {
        declaration.validate();
    } catch (...) {
        declaration.equality_rows_ = declared_equality_rows;
        throw;
    }
    declaration.equality_rows_ = declared_equality_rows;
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
    //
    // Every refusal the boundary can make is still made, and made FIRST, before
    // anything of the program is written: only the one conjunct the shared rows
    // break is stood down.
    this->validate_except_equality_row_sum();

    host.objectives_ = std::move(this->declaration_.objectives_);
    host.equality_constraints_ = std::move(this->declaration_.equality_constraints_);
    host.inequality_constraints_ = std::move(this->declaration_.inequality_constraints_);

    // The adopting entry's own resets, for the same reason it makes them: the
    // three lists have just been replaced wholesale, so whatever internal fixing
    // rows the previous layout counted went with them. The layout call below
    // discards internal rows by that count before it lays, and a stale count
    // would truncate the equality list that was just installed. True of a freshly
    // constructed program by construction; written down rather than assumed.
    host.internal_fixed_cons_ = 0;
    host.user_equal_cons_ = 0;
    host.equal_cons_ = 0;

    host.clear_variable_bounds();
    for (const auto &bound : this->declaration_.variable_bounds_) {
        host.set_variable_bound(bound.index_, bound.lower_, bound.upper_);
    }

    host.num_partitions_ = this->declaration_.partition_count_;
    host.make_nlp(primal_vars, equality_rows, inequality_rows);
}

} // namespace tycho::solvers
