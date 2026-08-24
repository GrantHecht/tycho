// =============================================================================
// Tycho (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================

#include "tycho/detail/solvers_vf/transcribed_aggregate.h"

#include <algorithm>
#include <string>
#include <utility>

#include <fmt/format.h>

namespace tycho::solvers {

namespace {

/// Which of the three claim domains a laid coordinate belongs to, read off the
/// row band it sits in.
enum class ClaimDomainOfSlot { kHessian, kEqualityJacobian, kInequalityJacobian };

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
    this->read_layout();
}

int TranscribedAggregate::negotiate_partition_count(int requested) {
    // A partition count decides the layout, so adopting one RE-LAYS the program
    // -- which resets the location table every scatter addresses and unbinds the
    // destination those offsets described. A consumer that has already analysed
    // this program holds a table that the re-lay would silently empty, and the
    // program's own re-analysis runs only when a fixed-variable treatment
    // reports a change, so the emptied table would survive into the next solve.
    //
    // The program says whether such a consumer exists: it publishes the value
    // array its tables are bound to, and a re-lay clears that. Refused while one
    // is bound, rather than re-laid underneath it.
    if (this->host_->bound_kkt_destination() != nullptr) {
        throw std::invalid_argument(fmt::format(
            "TranscribedAggregate::negotiate_partition_count: {0} partitions were requested, but "
            "the program this view publishes is already analysed against a consumer's KKT "
            "destination, and adopting a count re-lays it -- which would leave that consumer "
            "addressing an emptied location table. The partition count is declaration data: set "
            "it on the problem and transcribe again",
            requested));
    }

    const int adopted = this->host_->negotiate_partition_count(requested);
    this->read_layout();
    return adopted;
}

int TranscribedAggregate::kkt_dimension() const {
    this->refresh_if_relaid();
    return this->kkt_dimension_;
}

Eigen::Ref<const Eigen::VectorXi> TranscribedAggregate::kkt_claim_rows() const {
    this->refresh_if_relaid();
    return this->claim_rows_;
}

Eigen::Ref<const Eigen::VectorXi> TranscribedAggregate::kkt_claim_cols() const {
    this->refresh_if_relaid();
    return this->claim_cols_;
}

Eigen::Ref<const Eigen::VectorXi> TranscribedAggregate::kkt_claim_partitions() const {
    this->refresh_if_relaid();
    return this->claim_partitions_;
}

hven::solvers::ClaimBlock TranscribedAggregate::hessian_claims() const {
    this->refresh_if_relaid();
    return this->hessian_;
}

hven::solvers::ClaimBlock TranscribedAggregate::equality_jacobian_claims() const {
    this->refresh_if_relaid();
    return this->equality_jacobian_;
}

hven::solvers::ClaimBlock TranscribedAggregate::inequality_jacobian_claims() const {
    this->refresh_if_relaid();
    return this->inequality_jacobian_;
}

Eigen::Ref<const Eigen::VectorXi> TranscribedAggregate::objective_gradient_claim_rows() const {
    this->refresh_if_relaid();
    return this->objective_gradient_rows_;
}

TranscribedAggregate::DeclaredShape TranscribedAggregate::shape_of(const NonLinearProgram &host) {
    return DeclaredShape{host.primal_vars_,    host.equal_cons_,          host.inequal_cons_,
                         host.num_partitions_, host.internal_fixed_cons_, host.num_user_kkt_elems_};
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
    // not move with it. The stream read at the last un-eliminated layout IS the
    // declaration-space answer, and it is kept.
    //
    // There is always such a stream to keep: the constructor reads one and
    // refuses the program outright if it cannot, so no view exists to be
    // refreshed unless a declaration-space stream was read at construction.
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
        // Same declaration, narrower engine space: the published stream stands.
        this->read_at_epoch_ = this->host_->structure_epoch();
        return;
    }

    this->read_layout();
}

void TranscribedAggregate::read_layout() const {
    const NonLinearProgram &host = *this->host_;

    // The single enforcement point of the rule the whole coordinate mapping
    // below rests on. This routine reads the layout's coordinates and states
    // them as declared identities, which it may do only while the two spaces
    // are the same one: a fixed-variable treatment that has eliminated
    // variables lays a narrower space, in which the surviving coordinates are
    // renumbered and an eliminated one is recorded as a negative placeholder
    // that names neither a coordinate nor the domain it came from.
    //
    // Checked here rather than at the callers because all three of them reach
    // this routine -- construction, a partition renegotiation, and a refresh
    // across a re-lay -- and only the last has a published stream to fall back
    // on, which it keeps without coming here. The other two have nothing to
    // keep, so for them the reduced layout is a refusal.
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

    // The laid row bands. The program keeps a slack block between the primal
    // block and the constraint rows; the assembled space the claims are stated
    // in has none, so a constraint row moves down by the slack width.
    const int slack = host.slack_vars_;
    const int equality_base = primal + slack;
    const int inequality_base = equality_base + equality_rows;

    auto domain_of = [&](int slot, int row, int col) {
        if (row < 0 || col < 0) {
            // Only an eliminated coordinate is recorded negative, and this
            // routine never runs against an eliminated layout (see
            // refresh_if_relaid). Reaching one here means the layout and the
            // reduction flag disagree, which would put dropped claims of one
            // domain into another's run.
            throw std::invalid_argument(fmt::format(
                "TranscribedAggregate: claim slot {0} names coordinate ({1}, {2}) in a layout "
                "that reports no eliminated variables; a negative coordinate is how the layout "
                "records an elimination, and its domain cannot be told from the row band",
                slot, row, col));
        }
        if (row < primal) {
            return ClaimDomainOfSlot::kHessian;
        }
        if (row < equality_base) {
            throw std::invalid_argument(fmt::format(
                "TranscribedAggregate: the layout claims KKT row {0}, which is a slack row; the "
                "claim stream is stated in a space with no slack block",
                row));
        }
        if (row < inequality_base) {
            return ClaimDomainOfSlot::kEqualityJacobian;
        }
        return ClaimDomainOfSlot::kInequalityJacobian;
    };

    const int total = host.num_user_kkt_elems_;
    Eigen::VectorXi rows(total);
    Eigen::VectorXi cols(total);
    Eigen::VectorXi partitions(total);

    int next = 0;
    auto emit_domain = [&](ClaimDomainOfSlot wanted) {
        const int start = next;
        for (int slot = 0; slot < total; slot++) {
            const int row = host.kkt_coeff_rows_[slot];
            const int col = host.kkt_coeff_cols_[slot];
            if (domain_of(slot, row, col) != wanted) {
                continue;
            }

            int out_row = row;
            int out_col = col;
            if (wanted == ClaimDomainOfSlot::kHessian) {
                // The layout records a Hessian element in whichever order the
                // piece walked it; the claim convention names it on the upper
                // triangle.
                out_row = std::min(row, col);
                out_col = std::max(row, col);
            } else if (wanted == ClaimDomainOfSlot::kEqualityJacobian) {
                out_row = primal + (row - equality_base);
            } else {
                out_row = primal + equality_rows + (row - inequality_base);
            }

            rows[next] = out_row;
            cols[next] = out_col;
            partitions[next] = host.kkt_coeff_part_ids_[slot];
            next++;
        }
        return hven::solvers::ClaimBlock{start, next - start};
    };

    const hven::solvers::ClaimBlock hessian = emit_domain(ClaimDomainOfSlot::kHessian);
    const hven::solvers::ClaimBlock equality = emit_domain(ClaimDomainOfSlot::kEqualityJacobian);
    const hven::solvers::ClaimBlock inequality =
        emit_domain(ClaimDomainOfSlot::kInequalityJacobian);

    if (next != total) {
        throw std::invalid_argument(
            fmt::format("TranscribedAggregate: the three domain runs cover {0} of the layout's {1} "
                        "claim slots",
                        next, total));
    }

    const int gradient_slots = host.num_pgx_elems_;
    Eigen::VectorXi gradient_rows(gradient_slots);
    for (int slot = 0; slot < gradient_slots; slot++) {
        const int row = host.rhs_coeff_rows_[host.pgx_data_start_ + slot];
        if (row < 0 || row >= primal) {
            throw std::invalid_argument(fmt::format(
                "TranscribedAggregate: objective-gradient claim slot {0} names row {1}, outside "
                "the {2} declared variables",
                slot, row, primal));
        }
        gradient_rows[slot] = row;
    }

    // Committed together, so a refusal above leaves the previously published
    // stream standing rather than a half-written one.
    this->claim_rows_ = std::move(rows);
    this->claim_cols_ = std::move(cols);
    this->claim_partitions_ = std::move(partitions);
    this->objective_gradient_rows_ = std::move(gradient_rows);
    this->hessian_ = hessian;
    this->equality_jacobian_ = equality;
    this->inequality_jacobian_ = inequality;

    this->read_at_epoch_ = host.structure_epoch();
    this->read_at_shape_ = shape_of(host);
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
