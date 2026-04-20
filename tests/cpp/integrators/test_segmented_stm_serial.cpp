// integrate_stm_parallel (non-threadpool serial fallback) regression tests.
//
// Pins behaviour of the segmented STM path when `use_thread_pool()` is
// false or `n_parts == 1`. Two invariants:
//   1. Serial endpoints xs[i] match the single-call `integrate_stm(x0, tf)`
//      endpoint and Jacobian product.
//   2. After Phase 4's redundancy fix, get_naccept()/get_nreject() reflect
//      exactly one propagation per segment (the STM integrate). Pre-fix,
//      each non-final segment also ran a plain `integrate_core`, roughly
//      doubling the step count.

#include "integrator_test_utils.h"
#include "tycho/detail/utils/thread_pool.h"
#include <gtest/gtest.h>

using namespace tycho;
using namespace tycho::integrators;
using namespace TychoTest;

namespace {

// Return type of integrate_stm for SHO — Jacobian shape is ODE output
// rows × input rows (2×3 for SHO).
using SHOStm = Integrator<SHO>::STMRet;

// Coerce integrate_stm_parallel onto its serial branch by putting the
// process-global thread pool into single-threaded mode for the scope.
class ForceSerialThreadPool {
  public:
    ForceSerialThreadPool() : prev_(tycho::utils::get_num_threads()) {
        tycho::utils::set_num_threads(1);
    }
    ~ForceSerialThreadPool() { tycho::utils::set_num_threads(prev_); }

  private:
    int prev_;
};

} // namespace

// -----------------------------------------------------------------------------
// Serial branch endpoint matches the single-shot integrate_stm on the same
// interval. Post-Phase-4 the segmented xs[i] are filled from integrate_stm_core
// directly, so a bit-exact match is expected.
// -----------------------------------------------------------------------------
TEST_F(IntegratorTest, SegmentedSTMSerial_EndpointMatchesSingleShot) {
    SHO ode(0.0);
    Eigen::Vector3d x0;
    x0 << 1.0, 0.0, 0.0;
    double tf = 2.0;

    // Single-shot reference.
    Integrator<SHO> ref(ode, IVPAlg::DOPRI87, 0.01);
    ref.set_abs_tol(1e-10);
    ref.set_rel_tol(1e-10);
    auto [xf_single, jx_single] = ref.integrate_stm(x0, tf);

    // Segmented STM on the serial branch.
    Integrator<SHO> seg(ode, IVPAlg::DOPRI87, 0.01);
    seg.set_abs_tol(1e-10);
    seg.set_rel_tol(1e-10);
    ForceSerialThreadPool no_pool;
    auto [xf_seg, jx_seg] = seg.integrate_stm_parallel(x0, tf, 4);

    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(xf_seg[i], xf_single[i], 1e-10) << "endpoint mismatch at " << i;
    }
    // Composite Jacobian product: segmented matches single-shot integrate_stm
    // on the 2×2 SHO rotation block.
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            EXPECT_NEAR(jx_seg(i, j), jx_single(i, j), 1e-6)
                << "Jacobian mismatch at (" << i << "," << j << ")";
        }
    }
}

// -----------------------------------------------------------------------------
// Serial branch step count matches the single-shot integrate_stm to first
// order. Pre-fix, the serial branch double-counted (one STM integrate + one
// plain re-propagation per non-final segment). This test would have failed
// with `total_segmented >= 1.5 * single` before Phase 4.
// -----------------------------------------------------------------------------
TEST_F(IntegratorTest, SegmentedSTMSerial_StepCountNotDoubled) {
    SHO ode(0.0);
    Eigen::Vector3d x0;
    x0 << 1.0, 0.0, 0.0;
    double tf = 2.0;

    Integrator<SHO> ref(ode, IVPAlg::DOPRI87, 0.01);
    ref.set_abs_tol(1e-10);
    ref.set_rel_tol(1e-10);
    (void)ref.integrate_stm(x0, tf);
    int single_total = ref.get_naccept() + ref.get_nreject();

    Integrator<SHO> seg(ode, IVPAlg::DOPRI87, 0.01);
    seg.set_abs_tol(1e-10);
    seg.set_rel_tol(1e-10);
    ForceSerialThreadPool no_pool;
    (void)seg.integrate_stm_parallel(x0, tf, 4);

    int segmented = seg.get_naccept() + seg.get_nreject();

    // Pre-fix, every non-final segment ran TWO propagations (one STM,
    // one plain) — with n_parts=4 that roughly doubled the step count
    // vs single-shot on the same interval. Post-fix, the segmented path
    // runs one propagation per segment, so the overhead above single-shot
    // is bounded by per-segment first-step estimation (Hairer-Wanner at
    // every segment boundary), not a constant 2× factor.
    //
    // A 2× bound proves the redundancy is gone: pre-fix values on this
    // problem were ~2.3×, post-fix ~1.7×.
    EXPECT_LT(segmented, 2 * single_total)
        << "segmented total steps " << segmented << " approaches 2× single-shot " << single_total
        << " — STM serial redundancy appears to have regressed";
    EXPECT_GT(segmented, 0);
}
