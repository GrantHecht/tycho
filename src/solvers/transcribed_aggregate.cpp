// =============================================================================
// Tycho (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================

#include "tycho/detail/solvers_vf/transcribed_aggregate.h"

#include <cstddef>
#include <cstring>
#include <string>
#include <utility>

#include <fmt/format.h>

namespace tycho::solvers {

namespace {

/// The request's flags, by name, so a refusal says which shape was asked for
/// rather than which bit pattern.
std::string request_shape_name(hven::solvers::EvalRequest request) {
    using hven::solvers::EvalRequest;
    struct Flag {
        EvalRequest bit_;
        const char *name_;
    };
    static constexpr Flag kFlags[] = {
        {EvalRequest::kObjectiveValue, "objective value"},
        {EvalRequest::kObjectiveGradient, "objective gradient"},
        {EvalRequest::kObjectiveHessian, "objective Hessian"},
        {EvalRequest::kConstraintValues, "constraint values"},
        {EvalRequest::kConstraintAdjointGradient, "constraint adjoint gradient"},
        {EvalRequest::kConstraintJacobian, "constraint Jacobian"},
        {EvalRequest::kConstraintAdjointHessian, "constraint adjoint Hessian"},
    };

    std::string named;
    for (const Flag &flag : kFlags) {
        if (hven::solvers::has_request(request, flag.bit_)) {
            if (!named.empty()) {
                named += " + ";
            }
            named += flag.name_;
        }
    }
    if (named.empty()) {
        named = "no output";
    }
    return fmt::format("{0} (flags 0x{1:x})", named, static_cast<std::uint32_t>(request));
}

} // namespace

TranscribedAggregate::TranscribedAggregate(std::shared_ptr<NonLinearProgram> host)
    : host_(std::move(host)) {
    if (this->host_ == nullptr) {
        throw std::invalid_argument("TranscribedAggregate: the program handle must not be null");
    }
    this->snapshot_layout();
}

int TranscribedAggregate::negotiate_partition_count(int requested) {
    const int adopted = this->host_->negotiate_partition_count(requested);
    this->snapshot_layout();
    return adopted;
}

int TranscribedAggregate::kkt_dimension() const {
    this->publish();
    return this->kkt_dimension_;
}

Eigen::Ref<const Eigen::VectorXi> TranscribedAggregate::kkt_claim_rows() const {
    this->publish();
    return this->claim_rows_;
}

Eigen::Ref<const Eigen::VectorXi> TranscribedAggregate::kkt_claim_cols() const {
    this->publish();
    return this->claim_cols_;
}

Eigen::Ref<const Eigen::VectorXi> TranscribedAggregate::kkt_claim_partitions() const {
    this->publish();
    return this->claim_partitions_;
}

hven::solvers::ClaimBlock TranscribedAggregate::hessian_claims() const {
    this->publish();
    return this->hessian_;
}

hven::solvers::ClaimBlock TranscribedAggregate::equality_jacobian_claims() const {
    this->publish();
    return this->equality_jacobian_;
}

hven::solvers::ClaimBlock TranscribedAggregate::inequality_jacobian_claims() const {
    this->publish();
    return this->inequality_jacobian_;
}

Eigen::Ref<const Eigen::VectorXi> TranscribedAggregate::objective_gradient_claim_rows() const {
    this->publish();
    return this->objective_gradient_rows_;
}

TranscribedAggregate::DeclaredShape TranscribedAggregate::shape_of(const NonLinearProgram &host) {
    return DeclaredShape{host.primal_vars_,    host.equal_cons_,          host.inequal_cons_,
                         host.num_partitions_, host.internal_fixed_cons_, host.num_user_kkt_elems_};
}

void TranscribedAggregate::publish() const {
    this->refresh_if_relaid();
    this->materialize_stream();
}

void TranscribedAggregate::refresh_if_relaid() const {
    if (this->host_->structure_epoch() == this->read_at_epoch_) {
        return;
    }

    // A fixed-variable treatment that ELIMINATES variables re-lays the program
    // in a narrower space: the surviving coordinates are renumbered and the
    // eliminated ones name no entry at all. None of that is declaration data --
    // the declaration is the same pieces over the same variables either way --
    // and this surface is stated in declaration identities, so the stream must
    // not move with it. The layout read at the last un-eliminated lay IS the
    // declaration-space answer, and it is kept.
    //
    // There is always such a layout to keep: the constructor copies one and
    // refuses the program outright if it cannot, so no view exists to be
    // refreshed unless a declaration-space layout was taken at construction.
    // Whether the stream has been stated from it yet does not matter here --
    // the snapshot is what the statement is made from, and it survives.
    if (this->host_->is_reduced()) {
        const DeclaredShape now = shape_of(*this->host_);
        if (!(now == this->read_at_shape_)) {
            throw std::invalid_argument(fmt::format(
                "TranscribedAggregate: the program was re-laid into a different declared shape "
                "({0} equality rows, {1} of them internal, {2} claim slots, against {3}/{4}/{5} "
                "when the stream was read) while a fixed-variable treatment has variables "
                "eliminated, so the declaration-space stream can be neither kept nor re-read. "
                "Re-transcribe, or read the stream before configuring a treatment",
                now.equality_rows_, now.internal_rows_, now.claim_slots_,
                this->read_at_shape_.equality_rows_, this->read_at_shape_.internal_rows_,
                this->read_at_shape_.claim_slots_));
        }
        // Same declaration, narrower engine space: the snapshot stands, and so
        // does anything already stated from it.
        this->read_at_epoch_ = this->host_->structure_epoch();
        return;
    }

    this->snapshot_layout();
}

void TranscribedAggregate::snapshot_layout() const {
    const NonLinearProgram &host = *this->host_;

    // The single enforcement point of the rule the whole coordinate mapping
    // rests on. This view states a layout's coordinates as declared identities,
    // which it may do only while the two spaces are the same one: a
    // fixed-variable treatment that has eliminated variables lays a narrower
    // space, in which the surviving coordinates are renumbered and an
    // eliminated one is recorded as a negative placeholder that names neither a
    // coordinate nor the domain it came from.
    //
    // Checked here rather than at the callers because all three of them reach
    // this routine -- construction, a partition renegotiation, and a refresh
    // across a re-lay -- and only the last has a layout to fall back on, which
    // it keeps without coming here. The other two have nothing to keep, so for
    // them the reduced layout is a refusal.
    if (host.is_reduced()) {
        throw std::invalid_argument(
            "TranscribedAggregate: the program's fixed-variable treatment has eliminated "
            "variables, so the layout on hand names coordinates in a narrower space than the "
            "declaration, and no declaration-space stream can be read from it. Build the view "
            "from the transcription, before a treatment is configured");
    }

    // The program's own dimensions rather than its declaration's. They are the
    // same three numbers -- the declaration reports what was laid -- and reading
    // them here keeps a lay from copying all three master piece lists into the
    // declaration cache for the sake of three integers.
    const int primal = host.primal_vars_;
    const int equality_rows = host.equal_cons_;
    const int inequality_rows = host.inequal_cons_;
    this->kkt_dimension_ = primal + equality_rows + inequality_rows;

    const int total = host.num_user_kkt_elems_;
    const int gradient_slots = host.num_pgx_elems_;

    LaidSnapshot snapshot;
    snapshot.slots_ = total;
    snapshot.gradient_slots_ = gradient_slots;
    snapshot.primal_ = primal;
    // The program keeps a slack block between the primal block and the
    // constraint rows; the assembled space the claims are stated in has none,
    // so the restatement needs the width to move a constraint row down by.
    snapshot.slack_ = host.slack_vars_;
    snapshot.equality_rows_ = equality_rows;

    // One allocation for all four runs, and no zero-fill: every word is
    // overwritten by the copies below, and the project builds with Eigen's
    // initialise-by-zero on, so an Eigen vector here would memset the whole
    // buffer first.
    const std::size_t slots = static_cast<std::size_t>(total);
    const std::size_t words = 3 * slots + static_cast<std::size_t>(gradient_slots);
    snapshot.data_ = std::make_unique_for_overwrite<int[]>(words);

    int *const buffer = snapshot.data_.get();
    if (total > 0) {
        const std::size_t bytes = slots * sizeof(int);
        std::memcpy(buffer, host.kkt_coeff_rows_.data(), bytes);
        std::memcpy(buffer + total, host.kkt_coeff_cols_.data(), bytes);
        std::memcpy(buffer + 2 * total, host.kkt_coeff_part_ids_.data(), bytes);
    }
    if (gradient_slots > 0) {
        std::memcpy(buffer + 3 * total, host.rhs_coeff_rows_.data() + host.pgx_data_start_,
                    static_cast<std::size_t>(gradient_slots) * sizeof(int));
    }

    this->laid_ = std::move(snapshot);
    // What was published, if anything was, describes the layout this one
    // replaces. Marked stale rather than restated: the next reader states it,
    // and in every workflow tycho runs today there is no next reader.
    this->stream_current_ = false;

    this->read_at_epoch_ = host.structure_epoch();
    this->read_at_shape_ = shape_of(host);
}

void TranscribedAggregate::materialize_stream() const {
    if (this->stream_current_) {
        return;
    }

    const LaidSnapshot &laid = this->laid_;

    // The laid row bands. A constraint row moves down by the slack width, which
    // is what the assembled space having no slack block amounts to.
    const int primal = laid.primal_;
    const int slack = laid.slack_;
    const int equality_base = primal + slack;
    const int inequality_base = equality_base + laid.equality_rows_;

    // The snapshot is read through raw pointers into locals rather than through
    // its members. The classification below writes into the arrays it is
    // building, and a compiler that cannot rule out aliasing between those
    // writes and the snapshot reloads every bound and every data pointer on
    // each slot; read once here, it does not have to.
    const int total = laid.slots_;
    const int *const laid_rows = laid.rows();
    const int *const laid_cols = laid.cols();
    const int *const laid_partitions = laid.partitions();

    // Every refusal a slot can make, folded into four accumulators so the pass
    // carries no branch and the compiler can widen it. The fold keeps the sign
    // bit of every coordinate it saw, and the three running counts are of the
    // rows below each band edge, which is where the size of each domain's run
    // comes from: the Hessian run is what sits below the primal edge, the
    // equality run what sits between the two constraint edges, the inequality
    // run the rest. A slack row is a row between the primal edge and the
    // equality edge, so the two lower counts disagreeing is exactly one having
    // been claimed.
    int coordinate_signs = 0;
    int below_primal = 0;
    int below_equality_base = 0;
    int below_inequality_base = 0;
    for (int slot = 0; slot < total; slot++) {
        const int row = laid_rows[slot];
        const int col = laid_cols[slot];
        coordinate_signs |= row | col;
        below_primal += int(row < primal);
        below_equality_base += int(row < equality_base);
        below_inequality_base += int(row < inequality_base);
    }

    if (coordinate_signs < 0 || below_equality_base != below_primal) {
        // Something the stream cannot state is in the layout. Which slot, and
        // which of the two, is worth a second walk to say exactly -- and this
        // walk makes the refusal on the first offending slot in slot order,
        // which is the one the fold above cannot name.
        for (int slot = 0; slot < total; slot++) {
            const int row = laid_rows[slot];
            const int col = laid_cols[slot];
            if (row < 0 || col < 0) {
                // Only an eliminated coordinate is recorded negative, and this
                // snapshot was never taken from an eliminated layout (see
                // snapshot_layout). Reaching one here means the layout and the
                // reduction flag disagreed, which would put dropped claims of
                // one domain into another's run.
                throw std::invalid_argument(fmt::format(
                    "TranscribedAggregate: claim slot {0} names coordinate ({1}, {2}) in a layout "
                    "that reports no eliminated variables; a negative coordinate is how the "
                    "layout records an elimination, and its domain cannot be told from the row "
                    "band",
                    slot, row, col));
            }
            if (row >= primal && row < equality_base) {
                throw std::invalid_argument(fmt::format(
                    "TranscribedAggregate: the layout claims KKT row {0}, which is a slack row; "
                    "the claim stream is stated in a space with no slack block",
                    row));
            }
        }
    }

    const hven::solvers::ClaimBlock hessian{0, below_primal};
    const hven::solvers::ClaimBlock equality{below_primal,
                                             below_inequality_base - below_equality_base};
    const hven::solvers::ClaimBlock inequality{below_inequality_base,
                                               total - below_inequality_base};

    if (hessian.count_ + equality.count_ + inequality.count_ != total) {
        throw std::invalid_argument(
            fmt::format("TranscribedAggregate: the three domain runs cover {0} of the layout's {1} "
                        "claim slots",
                        hessian.count_ + equality.count_ + inequality.count_, total));
    }

    Eigen::VectorXi rows(total);
    Eigen::VectorXi cols(total);
    Eigen::VectorXi partitions(total);
    int *const out_rows = rows.data();
    int *const out_cols = cols.data();
    int *const out_partitions = partitions.data();

    // One pass, three cursors: each domain's run is filled in the slot order
    // the layout claims in, which is what makes each run the program's own
    // serial partition-index order.
    //
    // Both constraint bands move down by the same amount. The assembled space
    // has no slack block, so an equality row at primal + slack + r lands at
    // primal + r and an inequality row at primal + slack + me + r lands at
    // primal + me + r: in each case the laid row less the slack width.
    int next_hessian = hessian.start_;
    int next_equality = equality.start_;
    int next_inequality = inequality.start_;
    // Not unrolled. Three cursors, three input arrays and three output arrays
    // is already more live state than the register file holds; unrolled it
    // spills all of them and reloads them on every slot, which costs more than
    // the unrolling saves -- measured at about a sixth of this routine.
#if defined(__clang__)
#pragma clang loop unroll(disable)
#endif
    for (int slot = 0; slot < total; slot++) {
        const int row = laid_rows[slot];
        const int col = laid_cols[slot];

        int out_slot = 0;
        int out_row = 0;
        int out_col = 0;
        if (row < primal) {
            out_slot = next_hessian++;
            // The layout records a Hessian element in whichever order the piece
            // walked it; the claim convention names it on the upper triangle.
            out_row = row < col ? row : col;
            out_col = row < col ? col : row;
        } else {
            out_slot = row < inequality_base ? next_equality++ : next_inequality++;
            out_row = row - slack;
            out_col = col;
        }

        out_rows[out_slot] = out_row;
        out_cols[out_slot] = out_col;
        out_partitions[out_slot] = laid_partitions[slot];
    }

    const int gradient_slots = laid.gradient_slots_;
    Eigen::VectorXi gradient_rows(gradient_slots);
    const int *const laid_gradient_rows = laid.gradient_rows();
    int *const out_gradient_rows = gradient_rows.data();
    for (int slot = 0; slot < gradient_slots; slot++) {
        const int row = laid_gradient_rows[slot];
        if (row < 0 || row >= primal) {
            throw std::invalid_argument(fmt::format(
                "TranscribedAggregate: objective-gradient claim slot {0} names row {1}, outside "
                "the {2} declared variables",
                slot, row, primal));
        }
        out_gradient_rows[slot] = row;
    }

    // Committed together, and the stream marked current only once they are all
    // in place, so a refusal above leaves the view stale rather than half
    // written -- and the next reader is refused in the same terms rather than
    // handed a partial stream.
    this->claim_rows_ = std::move(rows);
    this->claim_cols_ = std::move(cols);
    this->claim_partitions_ = std::move(partitions);
    this->objective_gradient_rows_ = std::move(gradient_rows);
    this->hessian_ = hessian;
    this->equality_jacobian_ = equality;
    this->inequality_jacobian_ = inequality;

    // The snapshot buffer is provably dead from here on: nothing reads
    // laid_.data_ again until a re-lay calls snapshot_layout(), which installs
    // a fresh buffer before anything could observe this one gone. Releasing it
    // here means a consumer that actually reads the stream holds one copy of
    // the layout, not two, for the rest of this snapshot's lifetime.
    this->laid_.data_.reset();
    this->stream_current_ = true;
}

void TranscribedAggregate::assemble_impl(const hven::solvers::CandidatePoint &point,
                                         hven::solvers::EvalRequest request,
                                         hven::solvers::KktScatterView kkt,
                                         hven::solvers::RhsScatterView rhs) {
    static_cast<void>(point);
    static_cast<void>(kkt);
    static_cast<void>(rhs);
    throw std::invalid_argument(fmt::format(
        "TranscribedAggregate: an assembly of shape {0} was asked for against a destination the "
        "caller laid, and this view cannot fill one: the program behind it binds the KKT value "
        "array its location tables address, so a fill can only land there. Assemble through that "
        "program, or consume its claim stream and wait on the engine-side path that fills a "
        "foreign destination",
        request_shape_name(request)));
}

void TranscribedAggregate::evaluate_candidate_values_impl(
    const hven::solvers::CandidatePoint &point, hven::solvers::CandidateValues out) {
    this->host_->evaluate_candidate_values(point, out);
}

void TranscribedAggregate::evaluate_candidate_first_order_impl(
    const hven::solvers::CandidatePoint &point, hven::solvers::CandidateFirstOrder out) {
    this->host_->evaluate_candidate_first_order(point, out);
}

} // namespace tycho::solvers
