// =============================================================================
// Tycho (Copyright 2026-present Grant R. Hecht, Apache 2.0 — see LICENSE.txt)
// =============================================================================
//
// The transcribed problem, published as a claim-stream-bearing model provider.
//
// A transcription declares pieces and lays them out on the solver program that
// runs them. That program publishes a declaration and evaluation entries, which
// is everything a consumer needs in order to ASK for a fill against a
// destination the program itself laid. It does not publish WHERE each fill
// lands, and a consumer that wants to lay a destination OF ITS OWN -- a
// sequential-quadratic driver, an engine-independent scorer -- needs that
// second half: the per-slot coordinates, and the per-domain slot ranges.
//
// This class is that second half. It reads the layout the program produced and
// restates it in the coordinate convention the claim-stream contract fixes:
// the square assembled space, n + me + mi on a side, laid
// [primal | equality rows | inequality rows], with the Lagrangian Hessian on
// the upper triangle, an equality Jacobian claim at (n + r, c) and an
// inequality Jacobian claim at (n + me + r, c). No slack block and no solver
// coefficients: those are a consumer's own storage and a consumer's own step.
//
// WHY IT WRAPS THE PROGRAM RATHER THAN REPLACING IT. The interior-point engine
// consumes the transcribed problem through that program, and the program binds
// the KKT destination its location tables address. Widening the engine to
// consume a foreign aggregate is separate work; until it lands, the program is
// the host the engine evaluates through, and this class is the structural view
// beside it.

#pragma once

#include <memory>
#include <stdexcept>

#include <Eigen/Core>

#include <hven/model/claim_stream_source.h>
#include <hven/model/non_linear_program.h>

#include "tycho/detail/hven_namespaces.h"

namespace tycho::solvers {

/// @brief The claim stream of a transcribed problem, over the program that
///        lays it.
///
/// STREAM SHAPE. The program claims serially, one partition at a time in
/// partition-index order, and never from a worker thread. This view keeps that
/// order and groups it: the Lagrangian Hessian's slots first, then the equality
/// Jacobian's, then the inequality Jacobian's, each one contiguous run, and
/// within a run the program's own serial partition-index order. The stream is
/// therefore a pure function of the declaration and the adopted partition
/// count, and a consumer may address it as three runs.
///
/// WHAT A SLOT MEANS. One slot per stored matrix element one piece will sum
/// into -- not one slot per distinct coordinate. Two pieces contributing to one
/// coordinate hold two slots naming that coordinate, which is how a transcribed
/// problem's overlapping pieces compose. A consumer laying a destination from
/// this stream therefore maps slots onto coordinates rather than assuming the
/// map is injective.
///
/// FIXED-VARIABLE TREATMENTS DO NOT MOVE THE STREAM. A treatment that
/// ELIMINATES variables re-lays the program in a narrower space: the surviving
/// coordinates are renumbered and an eliminated one names no entry at all. That
/// is the engine's own reduction, not declaration data -- the declaration is
/// the same pieces over the same variables whichever treatment is configured --
/// and this surface is stated in declared identities. So the stream read at the
/// last un-eliminated layout is what stays published across such a re-lay, and
/// the claims a consumer reads are the same under every treatment that leaves
/// the declaration alone. A treatment that CHANGES the declaration (an internal
/// equality row per fixed variable) does move the stream, and correctly so.
///
/// VIEW VALIDITY. Everything published here describes the program's structures
/// as last laid. A re-lay replaces them and moves the structure epoch; this
/// view re-reads the program whenever the epoch it was built against has moved,
/// except across the elimination above, where it keeps what it has.
///
/// CONCURRENCY. The contract's posture applies unchanged -- one operation at a
/// time, structural mutation included -- and this view adds no thread safety of
/// its own.
class TranscribedAggregate final : public hven::solvers::ClaimStreamSource {
  public:
    /// @brief Builds the view over a laid program.
    /// @param host the program the transcription laid; must be non-null, already
    ///        laid out, and not yet reduced by a fixed-variable treatment --
    ///        a view is built from the transcription, which is where the
    ///        declaration-space stream exists to be read.
    /// @throws std::invalid_argument if @p host is null, if its treatment has
    ///         eliminated variables, or if its layout claims a coordinate this
    ///         convention cannot restate.
    explicit TranscribedAggregate(std::shared_ptr<NonLinearProgram> host);

    /// @brief The program this view publishes.
    const NonLinearProgram &host() const { return *this->host_; }

    /// @brief The declaration the program's structures were laid from.
    const hven::solvers::AggregateDeclaration &declaration() const override {
        return this->host_->declaration();
    }

    /// @brief Asks the program to adopt a partition count and returns what it
    ///        adopted.
    ///
    /// Adopting a count RE-LAYS the program, which resets the location table
    /// every scatter addresses. A consumer that has already analysed this
    /// program holds a table the re-lay would empty, and the program re-analyses
    /// itself only when a fixed-variable treatment reports a change -- so the
    /// emptied table would survive into the next solve. While such a consumer is
    /// bound, this call is refused rather than served.
    ///
    /// @param requested the partition count the caller wants.
    /// @return the adopted count.
    /// @throws std::invalid_argument if @p requested is below 1, or if the
    ///         program is already analysed against a consumer's destination.
    int negotiate_partition_count(int requested) override;

    /// @brief The program's evaluation thread budget.
    int evaluation_threads() const override { return this->host_->evaluation_threads(); }

    /// @brief Sets the program's evaluation thread budget.
    ///
    /// Threads and partitions carry no numeric relationship: a thread may serve
    /// several partitions, and the answer is the same at every reported count.
    /// The partition count decides the layout and the structural key; the
    /// thread budget decides only how many workers walk it.
    ///
    /// @param n the requested thread count.
    void set_evaluation_threads(int n) override { this->host_->set_evaluation_threads(n); }

    /// @brief The program's structural key.
    hven::solvers::ModelStructureKey model_structure_key() const override {
        return this->host_->model_structure_key();
    }

    /// @brief The program's structure epoch, which is the one a re-lay moves.
    hven::solvers::StructureEpoch structure_epoch() const override {
        return this->host_->structure_epoch();
    }

    /// @brief The program's capabilities, unchanged: this view adds no
    ///        evaluation path of its own.
    hven::solvers::AggregateCapability capabilities() const override {
        return this->host_->capabilities();
    }

    /// @brief The program's identity probe at @p x.
    /// @param x the point to probe, in declaration space.
    hven::solvers::IdentityProbe probe_identity(hven::ConstVecRef x) override {
        return this->host_->probe_identity(x);
    }

    /// @brief Edge dimension of the assembled space the claims are stated in:
    ///        n + me + mi.
    int kkt_dimension() const;

    /// @brief Claim slot to assembled KKT row, in claim order.
    Eigen::Ref<const Eigen::VectorXi> kkt_claim_rows() const override;

    /// @brief Claim slot to assembled KKT column, in claim order.
    Eigen::Ref<const Eigen::VectorXi> kkt_claim_cols() const override;

    /// @brief The KKT claim slots the Lagrangian Hessian scatters through.
    hven::solvers::ClaimBlock hessian_claims() const override;

    /// @brief The KKT claim slots the equality Jacobian scatters through.
    hven::solvers::ClaimBlock equality_jacobian_claims() const override;

    /// @brief The KKT claim slots the inequality Jacobian scatters through.
    hven::solvers::ClaimBlock inequality_jacobian_claims() const override;

    /// @brief Claim slot to row of the objective-gradient arena.
    Eigen::Ref<const Eigen::VectorXi> objective_gradient_claim_rows() const override;

    /// @brief The partition each KKT claim slot belongs to, in claim order.
    ///
    /// Not part of the claim-stream contract. Published because exclusivity is
    /// a property of this map and a check needs to be able to read it.
    Eigen::Ref<const Eigen::VectorXi> kkt_claim_partitions() const;

  protected:
    /// @brief Refuses an assembly against a caller's own destination.
    ///
    /// The program behind this view binds the KKT value array its location
    /// tables address, so a fill can only land in that array; a destination the
    /// caller laid is not one this view can scatter into. Publishing the claim
    /// stream is what lets a consumer lay such a destination, and filling it
    /// waits on the engine-side work that consumes a foreign aggregate
    /// directly.
    ///
    /// @param point   the point, ignored.
    /// @param request the evaluation shape, named in the refusal.
    /// @param kkt     the KKT destination, ignored.
    /// @param rhs     the right-hand-side destinations, ignored.
    /// @throws std::invalid_argument always.
    void assemble_impl(const hven::solvers::CandidatePoint &point,
                       hven::solvers::EvalRequest request, hven::solvers::KktScatterView kkt,
                       hven::solvers::RhsScatterView rhs) override;

    /// @brief The program's own candidate values at @p point.
    ///
    /// Caller-owned storage indexed by declared identities, which is a space
    /// this view shares with the program: the call forwards unchanged.
    void evaluate_candidate_values_impl(const hven::solvers::CandidatePoint &point,
                                        hven::solvers::CandidateValues out) override;

    /// @brief The program's own first-order candidate evaluation at @p point.
    void evaluate_candidate_first_order_impl(const hven::solvers::CandidatePoint &point,
                                             hven::solvers::CandidateFirstOrder out) override;

  private:
    /// @brief The declared shape a published stream was read at.
    ///
    /// What tells a re-lay that changed the DECLARATION from one that only
    /// changed the engine's own reduction. Compared only on the second kind, to
    /// decide whether the stream on hand still describes the declaration.
    struct DeclaredShape {
        int primal_vars_ = 0;
        int equality_rows_ = 0;
        int inequality_rows_ = 0;
        int partition_count_ = 0;
        int internal_rows_ = 0;
        int claim_slots_ = 0;

        friend bool operator==(const DeclaredShape &, const DeclaredShape &) = default;
    };

    static DeclaredShape shape_of(const NonLinearProgram &host);

    /// @brief Re-reads the program's layout if the epoch it was read at moved.
    void refresh_if_relaid() const;

    /// @brief Reads the program's layout into this view's published arrays.
    ///
    /// The one place the declared-space rule is enforced: every entry that reads
    /// a layout comes through here, and a program whose treatment has eliminated
    /// variables is refused rather than read in its narrower space.
    ///
    /// @throws std::invalid_argument if the program has variables eliminated, if
    ///         the layout claims a slack row, if it claims a negative coordinate
    ///         while reporting no elimination, or if the three domain runs do
    ///         not cover its claim slots.
    void read_layout() const;

    std::shared_ptr<NonLinearProgram> host_;

    mutable hven::solvers::StructureEpoch read_at_epoch_{};
    mutable DeclaredShape read_at_shape_{};

    mutable Eigen::VectorXi claim_rows_;
    mutable Eigen::VectorXi claim_cols_;
    mutable Eigen::VectorXi claim_partitions_;
    mutable Eigen::VectorXi objective_gradient_rows_;
    mutable hven::solvers::ClaimBlock hessian_{};
    mutable hven::solvers::ClaimBlock equality_jacobian_{};
    mutable hven::solvers::ClaimBlock inequality_jacobian_{};
    mutable int kkt_dimension_ = 0;
};

} // namespace tycho::solvers
