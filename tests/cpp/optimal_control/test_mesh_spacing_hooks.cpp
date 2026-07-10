///////////////////////////////////////////////////////////////////////////////
// LGLMeshSpacing CRTP hook regression tests (OC review §1.16)
//
// LGLMeshSpacing previously declared a member named `compute` that SHADOWED
// the CRTP dispatch entry point `ComputableBase::compute` instead of
// implementing the `compute_impl` hook (unlike its sibling
// SingleMeshSpacing, which correctly implements `compute_impl`). Shadowing
// the base `compute` means the base's SuperScalar (vectorized) dispatch path
// — which scalarizes a SuperScalar call into per-lane `compute_impl` calls
// for non-`Vectorizable` functions — is never exercised for this type.
//
// These tests characterize: (1) the scalar residual values are correct, and
// (2) invoking `.compute()` with a SuperScalar (`Eigen::Array<double, W, 1>`)
// scalar type routes through the base dispatcher's scalarize-per-lane path
// and produces results identical to the scalar path, lane by lane.
//
// IMPORTANT: every `.compute(...)` call below is made through a reference to
// the CRTP base (`LGLMeshSpacing<CSC>::Base`, i.e. `VectorFunction<...>`),
// NOT through the concrete derived type. Unqualified member lookup on a
// concretely-typed object starts at the DERIVED scope, so pre-fix it would
// resolve to the shadowing `compute` (same body) and the tests would pass
// with or without the fix. Calling through the base reference forces the
// dispatcher `ComputableBase::compute` to instantiate, which requires
// `Derived::compute_impl` — so this file FAILS TO COMPILE against the
// pre-fix shadowed header ("no member named 'compute_impl' in
// 'tycho::oc::LGLMeshSpacing<N>'").
///////////////////////////////////////////////////////////////////////////////

#include "oc_test_utils.h"
#include <gtest/gtest.h>
// Not part of any umbrella header: the only other consumer
// (src/optimal_control/ode_phase_base.cpp) includes it directly too.
#include <tycho/detail/optimal_control/transcription/mesh_spacing_constraints.h>

using namespace tycho;
using namespace TychoTest;

namespace {

// Fills each lane of a SuperScalar-valued input vector with the same scalar
// values, perturbed slightly per lane so that a bug which only reads lane 0
// (or broadcasts incorrectly) would be caught.
template <int CSC>
typename LGLMeshSpacing<CSC>::template Input<DefaultSuperScalar>
make_superscalar_input(const Eigen::Matrix<double, CSC, 1> &x_scalar) {
    typename LGLMeshSpacing<CSC>::template Input<DefaultSuperScalar> x_ss;
    constexpr int W = DefaultSuperScalar::SizeAtCompileTime;
    for (int j = 0; j < CSC; ++j) {
        DefaultSuperScalar lane;
        for (int lane_i = 0; lane_i < W; ++lane_i) {
            // Small per-lane perturbation on the interior nodes only, keeping
            // t0 < t1 < ... < tN strictly increasing for every lane.
            double bump = (j == 0 || j == CSC - 1) ? 0.0 : 1e-3 * static_cast<double>(lane_i);
            lane[lane_i] = x_scalar[j] + bump;
        }
        x_ss[j] = lane;
    }
    return x_ss;
}

} // namespace

TEST(MeshSpacingHooks, LGLMeshSpacingComputeImplResidual) {
    // LGL5 (CSC=3): CardinalSpacings = {0.0, 0.5, 1.0}.
    LGLMeshSpacing<3> f;
    LGLMeshSpacing<3>::Base &fb = f; // force base-scope lookup (CRTP dispatcher)
    Eigen::Matrix<double, 3, 1> x;
    x << 0.0, 0.4, 1.0;
    Eigen::Matrix<double, 1, 1> fx;
    fx.setZero();

    fb.compute(x, fx); // ComputableBase::compute -> Derived::compute_impl

    // Independent oracle: spacing 0.5 on [0, 1], node at 0.4 -> 0.5 - 0.4 = 0.1.
    EXPECT_TRUE(fx.allFinite());
    EXPECT_NEAR(fx[0], 0.1, 1e-14);
}

TEST(MeshSpacingHooks, LGLMeshSpacingComputeImplResidualLGL7) {
    // LGL7 (CSC=4): CardinalSpacings = {0, Ti2, Ti3, 1}, two interior residuals.
    LGLMeshSpacing<4> f;
    LGLMeshSpacing<4>::Base &fb = f; // force base-scope lookup (CRTP dispatcher)
    Eigen::Matrix<double, 4, 1> x;
    x << 0.0, 0.3, 0.8, 1.2;
    Eigen::Matrix<double, 2, 1> fx;
    fx.setZero();

    fb.compute(x, fx);

    // Independent oracle, hard-coded: the LGL7 interior cardinal nodes are the
    // published values Ti2 = 2.65575603264643e-1 and Ti3 = 7.34424396735357e-1
    // (roots-based Lobatto abscissae mapped to [0, 1]; note Ti2 + Ti3 = 1).
    // h = 1.2, so residuals are Ti2 - 0.3/1.2 and Ti3 - 0.8/1.2.
    constexpr double kTi2 = 2.65575603264643e-1;
    constexpr double kTi3 = 7.34424396735357e-1;
    EXPECT_TRUE(fx.allFinite());
    EXPECT_NEAR(fx[0], kTi2 - 0.25, 1e-14);
    EXPECT_NEAR(fx[1], kTi3 - 2.0 / 3.0, 1e-14);
}

TEST(MeshSpacingHooks, LGLMeshSpacingSuperScalarDispatchMatchesScalar) {
    // Not Vectorizable, so ComputableBase::compute must scalarize the
    // SuperScalar call into per-lane compute_impl(double, ...) calls. Before
    // the fix (LGLMeshSpacing::compute shadowing the base), this branch of
    // the base dispatcher was unreachable via ordinary calls through the
    // derived type.
    static_assert(!Vectorizable<LGLMeshSpacing<3>>);

    LGLMeshSpacing<3> f;
    LGLMeshSpacing<3>::Base &fb = f; // force base-scope lookup (CRTP dispatcher)
    Eigen::Matrix<double, 3, 1> x_scalar;
    x_scalar << 0.0, 0.4, 1.0;

    auto x_ss = make_superscalar_input<3>(x_scalar);
    LGLMeshSpacing<3>::Output<DefaultSuperScalar> fx_ss;
    fx_ss.setZero();
    fb.compute(x_ss, fx_ss);

    constexpr int W = DefaultSuperScalar::SizeAtCompileTime;
    for (int lane_i = 0; lane_i < W; ++lane_i) {
        Eigen::Matrix<double, 3, 1> x_lane;
        for (int j = 0; j < 3; ++j)
            x_lane[j] = x_ss[j][lane_i];

        Eigen::Matrix<double, 1, 1> fx_lane;
        fx_lane.setZero();
        fb.compute(x_lane, fx_lane);

        EXPECT_TRUE(std::isfinite(fx_ss[0][lane_i]));
        EXPECT_DOUBLE_EQ(fx_ss[0][lane_i], fx_lane[0]);
    }
}

TEST(MeshSpacingHooks, LGLMeshSpacingSuperScalarDispatchMatchesScalarLGL7) {
    static_assert(!Vectorizable<LGLMeshSpacing<4>>);

    LGLMeshSpacing<4> f;
    LGLMeshSpacing<4>::Base &fb = f; // force base-scope lookup (CRTP dispatcher)
    Eigen::Matrix<double, 4, 1> x_scalar;
    x_scalar << 0.0, 0.3, 0.8, 1.2;

    auto x_ss = make_superscalar_input<4>(x_scalar);
    LGLMeshSpacing<4>::Output<DefaultSuperScalar> fx_ss;
    fx_ss.setZero();
    fb.compute(x_ss, fx_ss);

    constexpr int W = DefaultSuperScalar::SizeAtCompileTime;
    for (int lane_i = 0; lane_i < W; ++lane_i) {
        Eigen::Matrix<double, 4, 1> x_lane;
        for (int j = 0; j < 4; ++j)
            x_lane[j] = x_ss[j][lane_i];

        Eigen::Matrix<double, 2, 1> fx_lane;
        fx_lane.setZero();
        fb.compute(x_lane, fx_lane);

        for (int k = 0; k < 2; ++k) {
            EXPECT_TRUE(std::isfinite(fx_ss[k][lane_i]));
            EXPECT_DOUBLE_EQ(fx_ss[k][lane_i], fx_lane[k]);
        }
    }
}
