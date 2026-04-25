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
// counts must be visible in get_naccept() after the call (success or
// exception), via the RAII CounterWriteback over the threadpool branch.
TEST_F(IntegratorTest, STMParallelWorkerThrowsLeavesMainCountersCurrent) {
    WorkerThreadThrowingSHO ode;
    Integrator<WorkerThreadThrowingSHO> integ(ode, IVPAlg::DOPRI87, 0.05);
    integ.set_abs_tol(1e-8);
    integ.set_rel_tol(1e-8);

    Eigen::Vector3d x0;
    x0 << 1.0, 0.0, 0.0;
    const double tf = 4.0;

    EXPECT_EQ(integ.get_naccept(), 0);

    EXPECT_THROW((void)integ.integrate_stm_parallel(x0, tf, 4), std::runtime_error);

    // Main thread completed n_parts-1 = 3 synchronous propagation segments
    // before the futures were drained and threw. Those segments' accept
    // counts must publish to the member counter via RAII writeback.
    EXPECT_GT(integ.get_naccept(), 0)
        << "Main-thread accept count from completed pre-throw segments must "
           "land in get_naccept() via the threadpool branch's CounterWriteback.";
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
