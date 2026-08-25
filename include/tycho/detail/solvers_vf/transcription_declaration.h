// =============================================================================
// Tycho (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================
//
// The value a tycho transcription hands the model contract.
//
// Transcription used to write the solver program's own members: it pushed
// pieces onto the program's three master lists, wrote each piece's thread mode
// onto the list element it had just appended, staged bounds on the program, and
// then asked the program to lay itself out. Every one of those was a write into
// a laid-out object's internals, spread over four call sites and ten separate
// thread-mode writes.
//
// What replaces it is a VALUE. A transcription declares its pieces, their
// thread modes, its variable bounds and the partition count it wants into a
// declaration, and hands the whole declaration over once. The layout is then a
// pure function of that declaration and the partition count actually adopted,
// which is the property a layout-determinism check asserts against and the
// property the structural key keys on.

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include <hven/model/aggregate_declaration.h>
#include <hven/model/non_linear_program.h>

#include "tycho/detail/hven_namespaces.h"

namespace tycho::solvers {

/// @brief The pieces, thread modes, bounds and partition count one
///        transcription declares, gathered into the value the model contract
///        adopts.
///
/// A builder, not a container: pieces go in with the thread mode they are to be
/// partitioned under, bounds go in as records in declaration order, and one
/// call hands the finished declaration to the program that lays it. The builder
/// is spent by that call -- its piece lists move out -- so one builder serves
/// one transcription.
///
/// The three index-returning add entries return the position of the piece just
/// declared, which is the identity the optimal-control layer records to find
/// its constraint rows again after the layout runs.
class TranscriptionDeclaration {
  public:
    /// @brief Starts a declaration that will request @p partition_count
    ///        partitions.
    /// @param partition_count the requested count; values below 1 request 1,
    ///        matching the solver program's own constructor.
    explicit TranscriptionDeclaration(int partition_count) {
        this->declaration_.partition_count_ = std::max(partition_count, 1);
    }

    /// @brief Declares an objective piece under @p mode.
    /// @param piece the piece, moved from.
    /// @param mode the thread assignment policy the partitioner reads.
    /// @return the piece's position in the objective list.
    int add_objective(ObjectiveFunction piece, ThreadingFlags mode) {
        piece.set_thread_mode(mode);
        this->declaration_.objectives_.push_back(std::move(piece));
        return int(this->declaration_.objectives_.size()) - 1;
    }

    /// @brief Declares an equality-constraint piece under @p mode.
    /// @param piece the piece, moved from.
    /// @param mode the thread assignment policy the partitioner reads.
    /// @return the piece's position in the equality list.
    int add_equality(ConstraintFunction piece, ThreadingFlags mode) {
        piece.set_thread_mode(mode);
        this->declaration_.equality_constraints_.push_back(std::move(piece));
        return int(this->declaration_.equality_constraints_.size()) - 1;
    }

    /// @brief Declares an inequality-constraint piece under @p mode.
    /// @param piece the piece, moved from.
    /// @param mode the thread assignment policy the partitioner reads.
    /// @return the piece's position in the inequality list.
    int add_inequality(ConstraintFunction piece, ThreadingFlags mode) {
        piece.set_thread_mode(mode);
        this->declaration_.inequality_constraints_.push_back(std::move(piece));
        return int(this->declaration_.inequality_constraints_.size()) - 1;
    }

    int objective_count() const { return int(this->declaration_.objectives_.size()); }
    int equality_count() const { return int(this->declaration_.equality_constraints_.size()); }
    int inequality_count() const { return int(this->declaration_.inequality_constraints_.size()); }

    /// @brief The declared objective piece at @p index, for a caller that has
    ///        more to say about a piece it has just declared.
    /// @throws std::invalid_argument if @p index is not a declared position.
    ObjectiveFunction &objective(int index) {
        this->require_index(index, this->objective_count(), "objective");
        return this->declaration_.objectives_[std::size_t(index)];
    }

    /// @brief The declared equality piece at @p index.
    /// @throws std::invalid_argument if @p index is not a declared position.
    ConstraintFunction &equality(int index) {
        this->require_index(index, this->equality_count(), "equality constraint");
        return this->declaration_.equality_constraints_[std::size_t(index)];
    }

    /// @brief The declared inequality piece at @p index.
    /// @throws std::invalid_argument if @p index is not a declared position.
    ConstraintFunction &inequality(int index) {
        this->require_index(index, this->inequality_count(), "inequality constraint");
        return this->declaration_.inequality_constraints_[std::size_t(index)];
    }

    /// @brief Declares a bound on one primal variable.
    ///
    /// Recorded verbatim and in declaration order; repeated records on one
    /// index are intersected tightest-wins when the bounds are materialized. A
    /// record that bounds neither side narrows nothing and is dropped here, so
    /// that a declaration carries the same record list the solver program's own
    /// staging entry would have kept.
    ///
    /// @param global_index the primal variable the bound applies to.
    /// @param lower the lower side.
    /// @param upper the upper side.
    /// @throws std::invalid_argument if either side is NaN, since a bound that
    ///         cannot be compared cannot take part in the tightest-wins merge.
    void set_variable_bound(int global_index, double lower, double upper) {
        if (std::isnan(lower) || std::isnan(upper)) {
            throw std::invalid_argument(
                fmt::format("set_variable_bound: bound for index {0} is NaN (lower={1}, "
                            "upper={2})",
                            global_index, lower, upper));
        }
        constexpr double kInf = std::numeric_limits<double>::infinity();
        if (lower == -kInf && upper == kInf) {
            return;
        }
        this->declaration_.variable_bounds_.push_back(
            hven::solvers::VariableBound{global_index, lower, upper});
    }

    /// @brief Hands the finished declaration to @p host and lays it out.
    ///
    /// The dimensions arrive here rather than through the add entries because a
    /// transcription only knows them once every piece is in: the optimal-control
    /// layer counts rows as it indexes, and the row space is the running count it
    /// ends on.
    ///
    /// @param host the solver program that lays the declaration out.
    /// @param primal_vars the declared primal-variable count.
    /// @param equality_rows the declared equality-row count.
    /// @param inequality_rows the declared inequality-row count.
    /// @throws std::invalid_argument through the declaration's own validation --
    ///         a non-positive partition count, negative dimensions, piece row
    ///         counts that do not sum to the declared row counts, an
    ///         out-of-range bound index, a NaN bound, an inverted record or an
    ///         empty bound intersection.
    void lay(NonLinearProgram &host, int primal_vars, int equality_rows, int inequality_rows);

  private:
    static void require_index(int index, int count, const char *what) {
        if (index < 0 || index >= count) {
            throw std::invalid_argument(
                fmt::format("TranscriptionDeclaration: {0} piece {1} was asked for, but {2} have "
                            "been declared",
                            what, index, count));
        }
    }

    /// @brief The equality-row total the declaration's pieces claim: their
    ///        output rows times their application count, summed. An
    ///        accumulating integrand's pieces sum this past the declared
    ///        equality-row count, since several of them claim the same row;
    ///        the difference is what lay() declares as the shared-row
    ///        overcount.
    /// @throws std::invalid_argument if that total is past what an int states.
    int equality_piece_rows() const;

    hven::solvers::AggregateDeclaration declaration_;
};

/// @brief Binds a transcription's declaration into a slot for the length of one
///        transcription, and clears the slot however that transcription ends.
///
/// The declaration lives on the stack of the call that lays it, so a slot
/// naming it must be emptied when that call returns -- including when it returns
/// by throwing, which is when a bare assignment at the end of the body would be
/// skipped and the slot left naming storage that is gone.
class DeclarationBinding {
  public:
    /// @param slot the pointer to bind and later clear.
    /// @param declaration the declaration to name.
    DeclarationBinding(TranscriptionDeclaration *&slot, TranscriptionDeclaration &declaration)
        : slot_(slot) {
        this->slot_ = &declaration;
    }

    ~DeclarationBinding() { this->slot_ = nullptr; }

    DeclarationBinding(const DeclarationBinding &) = delete;
    DeclarationBinding &operator=(const DeclarationBinding &) = delete;

  private:
    TranscriptionDeclaration *&slot_;
};

} // namespace tycho::solvers
