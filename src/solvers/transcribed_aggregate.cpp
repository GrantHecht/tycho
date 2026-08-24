// =============================================================================
// Tycho (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================

#include "tycho/detail/solvers_vf/transcribed_aggregate.h"

#include <algorithm>
#include <utility>

#include <fmt/format.h>

namespace tycho::solvers {

namespace {

/// Which of the three claim domains a laid coordinate belongs to, read off the
/// row band it sits in.
enum class ClaimDomainOfSlot { kHessian, kEqualityJacobian, kInequalityJacobian, kDropped };

} // namespace

TranscribedAggregate::TranscribedAggregate(std::shared_ptr<NonLinearProgram> host)
    : host_(std::move(host)) {
    if (this->host_ == nullptr) {
        throw std::invalid_argument("TranscribedAggregate: the program handle must not be null");
    }
    this->read_layout();
}

int TranscribedAggregate::negotiate_partition_count(int requested) {
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

void TranscribedAggregate::refresh_if_relaid() const {
    if (!this->ever_read_ || this->host_->structure_epoch() != this->read_at_epoch_) {
        this->read_layout();
    }
}

void TranscribedAggregate::read_layout() const {
    const NonLinearProgram &host = *this->host_;
    const hven::solvers::AggregateDeclaration &declared = host.declaration();

    const int primal = declared.primal_vars_;
    const int equality_rows = declared.equality_rows_;
    const int inequality_rows = declared.inequality_rows_;
    this->kkt_dimension_ = primal + equality_rows + inequality_rows;

    // The laid row bands. The program keeps a slack block between the primal
    // block and the constraint rows; the assembled space the claims are stated
    // in has none, so a constraint row moves down by the slack width and a
    // primal coordinate moves back into the declared variable space.
    const int laid_primal = host.reduced_primal_vars();
    const int slack = host.slack_vars_;
    const int equality_base = laid_primal + slack;
    const int inequality_base = equality_base + host.equal_cons_;

    const bool reduced = host.is_reduced();

    auto declared_variable = [&](int laid_index) {
        return reduced ? host.reduced_to_full_[laid_index] : laid_index;
    };

    auto domain_of = [&](int row, int col) {
        if (row < 0 || col < 0) {
            return ClaimDomainOfSlot::kDropped;
        }
        if (row < laid_primal) {
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
    this->claim_rows_.resize(total);
    this->claim_cols_.resize(total);
    this->claim_partitions_.resize(total);

    // A dropped claim keeps its (-1, -1) and is grouped with the domain of the
    // piece list it came out of, which the partition tag alone cannot say. It is
    // grouped with the Hessian run: every piece list claims Hessian slots, so
    // that run is the one every dropped claim could have come from.
    int next = 0;
    auto emit_domain = [&](ClaimDomainOfSlot wanted, bool take_dropped) {
        const int start = next;
        for (int slot = 0; slot < total; slot++) {
            const int row = host.kkt_coeff_rows_[slot];
            const int col = host.kkt_coeff_cols_[slot];
            const ClaimDomainOfSlot domain = domain_of(row, col);
            if (domain == ClaimDomainOfSlot::kDropped) {
                if (!take_dropped) {
                    continue;
                }
            } else if (domain != wanted) {
                continue;
            }

            int out_row = row;
            int out_col = col;
            if (domain == ClaimDomainOfSlot::kHessian) {
                // The layout records a Hessian element in whichever order the
                // piece walked it; the claim convention names it on the upper
                // triangle, so both endpoints are mapped and then ordered.
                const int a = declared_variable(row);
                const int b = declared_variable(col);
                out_row = std::min(a, b);
                out_col = std::max(a, b);
            } else if (domain == ClaimDomainOfSlot::kEqualityJacobian) {
                out_row = primal + (row - equality_base);
                out_col = declared_variable(col);
            } else if (domain == ClaimDomainOfSlot::kInequalityJacobian) {
                out_row = primal + equality_rows + (row - inequality_base);
                out_col = declared_variable(col);
            }

            this->claim_rows_[next] = out_row;
            this->claim_cols_[next] = out_col;
            this->claim_partitions_[next] = host.kkt_coeff_part_ids_[slot];
            next++;
        }
        return hven::solvers::ClaimBlock{start, next - start};
    };

    this->hessian_ = emit_domain(ClaimDomainOfSlot::kHessian, /*take_dropped=*/true);
    this->equality_jacobian_ = emit_domain(ClaimDomainOfSlot::kEqualityJacobian, false);
    this->inequality_jacobian_ = emit_domain(ClaimDomainOfSlot::kInequalityJacobian, false);

    if (next != total) {
        throw std::invalid_argument(
            fmt::format("TranscribedAggregate: the three domain runs cover {0} of the layout's {1} "
                        "claim slots",
                        next, total));
    }

    const int gradient_slots = host.num_pgx_elems_;
    this->objective_gradient_rows_.resize(gradient_slots);
    for (int slot = 0; slot < gradient_slots; slot++) {
        const int row = host.rhs_coeff_rows_[host.pgx_data_start_ + slot];
        this->objective_gradient_rows_[slot] = row < 0 ? -1 : declared_variable(row);
    }

    this->read_at_epoch_ = host.structure_epoch();
    this->ever_read_ = true;
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
        static_cast<int>(request)));
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
