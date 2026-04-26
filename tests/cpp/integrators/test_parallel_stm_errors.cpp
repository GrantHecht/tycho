// integrate_stm_parallel composite-exception aggregation pin —
// verifies that when multiple async STM segments throw, the primary is
// preserved and secondary failures are joined into a composite runtime_error
// (up to kMaxExtras = 5, then "... and N more").
//
// The composite path is only reachable when the synchronous pre-integration
// in the main thread succeeds for every segment start AND the async stm_op
// tasks on worker threads throw. We achieve that by gating the throw on the
// calling thread id: the test captures the main thread id at construction
// and compute_impl throws only when invoked from a different thread.

#include "integrator_test_utils.h"
#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <thread>

using namespace tycho;
using namespace TychoTest;

namespace {

struct WorkerThreadThrowingSHO
    : oc::StaticODE<WorkerThreadThrowingSHO, 2, 0, 0, vf::DenseDerivativeMode::FDiffFwd,
                    vf::DenseDerivativeMode::FDiffFwd> {
    using Base = oc::StaticODE<WorkerThreadThrowingSHO, 2, 0, 0, vf::DenseDerivativeMode::FDiffFwd,
                               vf::DenseDerivativeMode::FDiffFwd>;

    std::thread::id main_thread_id_;

    WorkerThreadThrowingSHO() : main_thread_id_(std::this_thread::get_id()) {
        this->set_ode_size(2, 0, 0);
    }

    template <class InType, class OutType>
    inline void compute_impl(vf::CVecRef<InType> x_, vf::CVecRef<OutType> fx_) const {
        using Scalar = typename InType::Scalar;
        auto fx = fx_.const_cast_derived();
        if (std::this_thread::get_id() != main_thread_id_) {
            throw std::runtime_error("WorkerThreadThrowingSHO: worker thread at t=" +
                                     std::to_string(double(x_[2])));
        }
        fx[0] = Scalar(x_[1]);
        fx[1] = Scalar(-x_[0]);
    }
};

// Structural twin of WorkerThreadThrowingSHO without the throw check —
// gives a non-throwing baseline that uses the same StaticODE / FDiffFwd
// pathway, so accept-count calibration in
// STMParallelWorkerThrowsLeavesMainCountersCurrent compares apples to
// apples (different DenseDerivativeMode between SHO and the throwing
// variant produces noticeably different step cadences).
struct NonThrowingSHOTwin
    : oc::StaticODE<NonThrowingSHOTwin, 2, 0, 0, vf::DenseDerivativeMode::FDiffFwd,
                    vf::DenseDerivativeMode::FDiffFwd> {
    using Base = oc::StaticODE<NonThrowingSHOTwin, 2, 0, 0, vf::DenseDerivativeMode::FDiffFwd,
                               vf::DenseDerivativeMode::FDiffFwd>;

    NonThrowingSHOTwin() { this->set_ode_size(2, 0, 0); }

    template <class InType, class OutType>
    inline void compute_impl(vf::CVecRef<InType> x_, vf::CVecRef<OutType> fx_) const {
        using Scalar = typename InType::Scalar;
        auto fx = fx_.const_cast_derived();
        fx[0] = Scalar(x_[1]);
        fx[1] = Scalar(-x_[0]);
    }
};

// Main-AND-worker throwing variant — main throws once t exceeds a threshold,
// so the synchronous inter-segment integrate_core inside the dispatch loop
// raises while worker tasks are still submitted. Exercises the unwind-drain
// path (P0.5), which previously swallowed worker diagnostics with an empty
// catch(...) and rethrew only the main-thread exception.
struct MainLateThrowingSHO
    : oc::StaticODE<MainLateThrowingSHO, 2, 0, 0, vf::DenseDerivativeMode::FDiffFwd,
                    vf::DenseDerivativeMode::FDiffFwd> {
    using Base = oc::StaticODE<MainLateThrowingSHO, 2, 0, 0, vf::DenseDerivativeMode::FDiffFwd,
                               vf::DenseDerivativeMode::FDiffFwd>;

    std::thread::id main_thread_id_;
    double main_throw_after_t_ = 1.5;

    MainLateThrowingSHO() : main_thread_id_(std::this_thread::get_id()) {
        this->set_ode_size(2, 0, 0);
    }

    template <class InType, class OutType>
    inline void compute_impl(vf::CVecRef<InType> x_, vf::CVecRef<OutType> fx_) const {
        using Scalar = typename InType::Scalar;
        auto fx = fx_.const_cast_derived();
        const bool is_main = std::this_thread::get_id() == main_thread_id_;
        const double t = double(x_[2]);
        if (!is_main) {
            throw std::runtime_error("MainLateThrowingSHO: worker failure at t=" +
                                     std::to_string(t));
        }
        if (is_main && t > main_throw_after_t_) {
            throw std::runtime_error("MainLateThrowingSHO: main-thread failure at t=" +
                                     std::to_string(t));
        }
        fx[0] = Scalar(x_[1]);
        fx[1] = Scalar(-x_[0]);
    }
};

} // namespace

TEST_F(IntegratorTest, STMParallelMultipleSegmentFailuresYieldComposite) {
    WorkerThreadThrowingSHO ode;
    Integrator<WorkerThreadThrowingSHO> integ(ode, IVPAlg::DOPRI87, 0.05);
    integ.set_abs_tol(1e-8);
    integ.set_rel_tol(1e-8);

    Eigen::Vector3d x0;
    x0 << 1.0, 0.0, 0.0;
    const double tf = 4.0;

    // n_parts = 4 → 4 worker stm tasks all fail. Primary + 3 secondaries,
    // all below kMaxExtras = 5 ⇒ composite lists every secondary, no '... and N more'.
    try {
        (void)integ.integrate_stm_parallel(x0, tf, 4);
        FAIL() << "integrate_stm_parallel should have thrown";
    } catch (const std::runtime_error &e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("integrate_stm_parallel"), std::string::npos) << msg;
        EXPECT_NE(msg.find("primary segment failure"), std::string::npos) << msg;
        EXPECT_NE(msg.find("additional failures (3)"), std::string::npos) << msg;
        std::size_t count = 0;
        std::size_t pos = 0;
        const std::string needle = "WorkerThreadThrowingSHO";
        while ((pos = msg.find(needle, pos)) != std::string::npos) {
            ++count;
            pos += needle.size();
        }
        EXPECT_GE(count, 4u) << "Expected primary + 3 secondaries: " << msg;
        EXPECT_EQ(msg.find("... and"), std::string::npos)
            << "n_parts=4 has 3 extras, below kMaxExtras=5 ⇒ no trailing suffix";
    }
}

TEST_F(IntegratorTest, STMParallelExtraFailuresCapWithAndNMore) {
    WorkerThreadThrowingSHO ode;
    Integrator<WorkerThreadThrowingSHO> integ(ode, IVPAlg::DOPRI87, 0.05);
    integ.set_abs_tol(1e-8);
    integ.set_rel_tol(1e-8);

    Eigen::Vector3d x0;
    x0 << 1.0, 0.0, 0.0;
    const double tf = 8.0;

    // n_parts = 8 → primary + 7 secondaries. kMaxExtras = 5 ⇒ shows 5 and
    // appends "... and 2 more".
    try {
        (void)integ.integrate_stm_parallel(x0, tf, 8);
        FAIL() << "integrate_stm_parallel should have thrown";
    } catch (const std::runtime_error &e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("additional failures (7)"), std::string::npos) << msg;
        EXPECT_NE(msg.find("... and 2 more"), std::string::npos)
            << "Expected 'and 2 more' cap suffix: " << msg;
    }
}

// When workers throw, the main thread has already run n_parts-1 segments
// synchronously while dispatching tasks. Those completed segments' accept
// counts must publish to get_naccept() via the threadpool branch's RAII
// CounterWriteback (commit b5db505 hoisted the writeback to cover the
// total_main_na accumulator across all completed main-thread segments).
//
// Bracket pin: a bare EXPECT_GT(get_naccept(), 0) would still pass if the
// writeback published only the last per-segment local main_na (the pre-b5db505
// bug), or if some unrelated bookkeeping wrote 1. This test calibrates the
// expected count against a non-throwing baseline (identical SHO dynamics, same
// n_parts) and asserts the post-throw counter falls within
// [baseline * (n_parts - 1) / n_parts, baseline] — i.e. main thread completed
// at least (n_parts - 1) segments and at most all n_parts. A regression that
// reverts to per-segment-local writeback would produce ~baseline / n_parts,
// failing the lower bound.
TEST_F(IntegratorTest, STMParallelWorkerThrowsLeavesMainCountersCurrent) {
    constexpr int kNParts = 4;
    constexpr double kTf = 4.0;

    Eigen::Vector3d x0;
    x0 << 1.0, 0.0, 0.0;

    // Calibrate against a structural twin of WorkerThreadThrowingSHO that
    // omits the throw — same StaticODE / FDiffFwd pathway → matching step
    // cadence → meaningful comparison. Using SHO (expression-based ODE) here
    // would diverge by a ~3x step-count factor due to different
    // DenseDerivativeMode codepaths in calculate_jacobian.
    NonThrowingSHOTwin baseline_ode;
    Integrator<NonThrowingSHOTwin> baseline_integ(baseline_ode, IVPAlg::DOPRI87, 0.05);
    baseline_integ.set_abs_tol(1e-8);
    baseline_integ.set_rel_tol(1e-8);
    EXPECT_EQ(baseline_integ.get_naccept(), 0);
    (void)baseline_integ.integrate_stm_parallel(x0, kTf, kNParts);
    const int64_t baseline_naccept = baseline_integ.get_naccept();
    ASSERT_GT(baseline_naccept, 0)
        << "Calibration run produced no accepted steps — baseline ODE likely misconfigured.";

    // The throw originates when futures are drained AFTER the for-loop has
    // submitted all tasks AND the main thread has run all (n_parts - 1) main
    // segments. So total_main_na should equal baseline modulo small
    // FP-timing divergence. Per-segment-local writeback regression would
    // publish only the last segment's main_na ≈ baseline / (n_parts - 1).
    // Lower bound at baseline / 2 catches that (~33% < 50%) while allowing
    // a wide margin for any FP drift; upper bound at 2*baseline catches
    // accidental double-counting.
    const int64_t lower_bound = baseline_naccept / 2;
    const int64_t upper_bound = 2 * baseline_naccept;

    WorkerThreadThrowingSHO ode;
    Integrator<WorkerThreadThrowingSHO> integ(ode, IVPAlg::DOPRI87, 0.05);
    integ.set_abs_tol(1e-8);
    integ.set_rel_tol(1e-8);
    EXPECT_EQ(integ.get_naccept(), 0);

    EXPECT_THROW((void)integ.integrate_stm_parallel(x0, kTf, kNParts), std::runtime_error);

    const int64_t observed = integ.get_naccept();
    EXPECT_GE(observed, lower_bound)
        << "Writeback should reflect accumulated main-thread accept count, not "
           "per-segment-local (baseline=" << baseline_naccept << ", lower_bound="
        << lower_bound << ", observed=" << observed << "). A regression that publishes "
           "only the last segment's local main_na would land near "
        << (baseline_naccept / (kNParts - 1)) << ".";
    EXPECT_LE(observed, upper_bound)
        << "Writeback should not exceed 2*baseline (upper_bound=" << upper_bound
        << ", observed=" << observed << ") — guards against accidental double-counting.";
}

// P0.5: when the synchronous main-thread integrate_core inside the dispatch
// loop throws, the unwind handler must aggregate any worker failures rather
// than silently dropping them. This is distinct from the post-loop drain path
// covered by STMParallelMultipleSegmentFailuresYieldComposite.
TEST_F(IntegratorTest, STMParallelMainUnwindAggregatesWorkerFailures) {
    MainLateThrowingSHO ode;
    Integrator<MainLateThrowingSHO> integ(ode, IVPAlg::DOPRI87, 0.05);
    integ.set_abs_tol(1e-8);
    integ.set_rel_tol(1e-8);

    Eigen::Vector3d x0;
    x0 << 1.0, 0.0, 0.0;
    const double tf = 4.0;

    // Main-thread fails after t > 1.5, while workers fail unconditionally.
    // With n_parts = 4 and segments of length 1.0, main succeeds on seg 0
    // (t: 0 -> 1) but throws during seg 1 (t crosses 1.5). At that point
    // tasks 0 and 1 are submitted; both are pending worker failures.
    try {
        (void)integ.integrate_stm_parallel(x0, tf, 4);
        FAIL() << "integrate_stm_parallel should have thrown";
    } catch (const std::runtime_error &e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("main-thread segment failure"), std::string::npos)
            << "Unwind composite should name the main-thread source: " << msg;
        EXPECT_NE(msg.find("main-thread failure"), std::string::npos)
            << "Main-thread exception text must be preserved: " << msg;
        EXPECT_NE(msg.find("worker failures"), std::string::npos)
            << "Worker failure count header must appear: " << msg;
        EXPECT_NE(msg.find("worker failure at t="), std::string::npos)
            << "At least one worker what() must be included: " << msg;
    }
}
